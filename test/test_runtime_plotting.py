import json
import os
import unittest
from unittest.mock import patch

import numpy as np
from examples.host_runtime_live_viewer_demo import HostRuntimeLiveViewerDemo
from mock_environment import HasEnvironmentCase

from ndscan.define.fragment import ExpFragment
from ndscan.define.result_channels import FloatChannel
from ndscan.plots.runtime.fitting import BuiltinFitBackend, FitRequest
from ndscan.plots.runtime.live import snapshot_from_live_values
from ndscan.plots.runtime.viewer import (
    _NO_GROUP_KEY,
    _PLOT_MODE_1D,
    _PLOT_MODE_2D_IMAGE,
    _PLOT_MODE_2D_SCATTER,
    _SiteColumnWidget,
    RuntimePlotViewer,
    _default_x_choices,
    _default_y_choices,
    _group_by_choices,
)
from ndscan.runtime.api import make_fragment_prepared_scan_exp
from ndscan.scan.request import ScanRequest
from ndscan._qt import QtWidgets


class AppletLaunchFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_result("value", FloatChannel)

    def run_once(self):
        self.value.push(1.0)


class RuntimeLiveSnapshotTest(unittest.TestCase):
    @staticmethod
    def _make_basic_root_values() -> dict[str, object]:
        prefix = "ndscan.rid_0.site.root."
        return {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.x",
                            "description": "X Axis",
                            "spec": {"unit": "kHz"},
                        },
                    },
                    "param_1": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.y",
                            "description": "Y Axis",
                            "spec": {"unit": "kHz"},
                        },
                    },
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "channel_value": {
                        "path": "detector/counts",
                        "description": "Detected Counts",
                        "type": "float",
                        "unit": "cts",
                    }
                }
            ),
            prefix + "points.param_0": [0.0, 1.0],
            prefix + "points.param_1": [2.0, 3.0],
            prefix + "points.channel_value": [20.0, 21.0],
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
        }

    def test_snapshot_from_live_values_reconstructs_site_tree(self):
        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps({"x": {"is_scanned": True}}),
            prefix + "scan.channels": json.dumps({"channel_value": {"type": "float"}}),
            prefix + "points.x": [0.0, 1.0],
            prefix + "points.channel_value": [10.0, 11.0],
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
            prefix + "scan_x.site.path": json.dumps(["scan_x"]),
            prefix + "scan_x.site.parent_path": json.dumps([]),
            prefix + "scan_x.site.fragment_fqn": "ChildFragment",
            prefix + "scan_x.scan.parameters": json.dumps({"child_x": {"is_scanned": True}}),
            prefix + "scan_x.scan.channels": json.dumps({"child_value": {"type": "float"}}),
            prefix + "scan_x.points.child_x": [2.0, 3.0, 4.0, 5.0],
            prefix + "scan_x.points.child_value": [20.0, 30.0, 40.0, 50.0],
            prefix + "scan_x.segments.start_index": [0, 2],
            prefix + "scan_x.segments.parent_point_index": [0, 1],
            prefix + "scan_x.state.current_segment": 1,
            prefix + "scan_x.state.num_points": 4,
            prefix + "scan_x.state.completed": False,
        }

        snapshot = snapshot_from_live_values(prefix, values)

        root = snapshot.get_site(())
        child = snapshot.get_site(("scan_x",))

        self.assertEqual(root.fragment_fqn, "RootFragment")
        self.assertEqual(list(root.point_data["x"]), [0.0, 1.0])
        self.assertEqual([site.path for site in snapshot.child_sites(())], [("scan_x",)])
        self.assertEqual(len(child.segments_for_parent_point(1)), 1)
        self.assertEqual(
            child.slice_point_data(2, 4)["child_value"],
            [40.0, 50.0],
        )

    def test_runtime_choice_labels_use_schema_descriptions(self):
        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.pseudoparams": json.dumps(
                {
                    "pseudoparam_0": {
                        "variable": {
                            "name": "logical_x",
                            "description": "Logical X",
                            "spec": {"unit": "MHz"},
                        }
                    }
                }
            ),
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.drive_frequency",
                            "description": "Drive Frequency",
                            "spec": {"unit": "kHz"},
                        },
                    }
                    ,
                    "param_1": {
                        "path": "child",
                        "is_scanned": False,
                        "param": {
                            "fqn": "demo.child.detuning",
                            "description": "Mapped Detuning",
                            "spec": {"unit": "MHz"},
                        },
                    },
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "counts": {
                        "path": "detector/counts_raw",
                        "description": "Raw Counts",
                        "type": "opaque",
                    },
                    "channel_value": {
                        "path": "detector/counts",
                        "description": "Detected Counts",
                        "type": "float",
                        "unit": "cts",
                    }
                }
            ),
            prefix + "points.pseudoparam_0": [0.0, 1.0],
            prefix + "points.param_0": [10.0, 11.0],
            prefix + "points.param_1": [2.0, 3.0],
            prefix + "points.counts": [[1, 2, 3], [4, 5, 6]],
            prefix + "points.channel_value": [20.0, 21.0],
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
        }

        snapshot = snapshot_from_live_values(prefix, values)
        root = snapshot.get_site(())

        self.assertEqual(
            _default_x_choices(root),
            [
                ("pseudoparam_0", "logical_x (Logical X / MHz)"),
                ("param_0", "drive_frequency (Drive Frequency / kHz)"),
                ("param_1", "child/detuning (Mapped Detuning / MHz)"),
                ("channel_value", "detector/counts (Detected Counts / cts)"),
                ("__point_index__", "point_index"),
            ],
        )
        self.assertEqual(
            _default_y_choices(root),
            [
                ("pseudoparam_0", "logical_x (Logical X / MHz)"),
                ("param_0", "drive_frequency (Drive Frequency / kHz)"),
                ("param_1", "child/detuning (Mapped Detuning / MHz)"),
                ("channel_value", "detector/counts (Detected Counts / cts)"),
                ("__point_index__", "point_index"),
            ],
        )

    def test_site_column_widget_switches_controls_for_2d_scatter_mode(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()

        snapshot = snapshot_from_live_values("ndscan.rid_0.site.root.", values)
        root = snapshot.get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_2D_SCATTER,
            selected_x_key="param_0",
            selected_y_key="param_1",
            selected_z_key="channel_value",
            selected_group_key=None,
            show_lines=False,
        )
        self.assertFalse(widget._z_combo.isHidden())
        self.assertTrue(widget._show_lines_checkbox.isHidden())
        self.assertTrue(widget._group_combo.isHidden())

        widget.update_state(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key="channel_value",
            selected_group_key="param_1",
            show_lines=False,
        )
        self.assertTrue(widget._z_combo.isHidden())
        self.assertFalse(widget._show_lines_checkbox.isHidden())
        self.assertFalse(widget._group_combo.isHidden())

        widget.update_state(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=1,
            selected_plot_mode=_PLOT_MODE_2D_IMAGE,
            selected_x_key="param_0",
            selected_y_key="param_1",
            selected_z_key="channel_value",
            selected_group_key=None,
            show_lines=False,
        )
        self.assertFalse(widget._z_combo.isHidden())
        self.assertTrue(widget._show_lines_checkbox.isHidden())
        self.assertTrue(widget._group_combo.isHidden())
        self.assertTrue(widget._image_item.isVisible())
        self.assertFalse(widget._colorbar.isHidden())
        self.assertTrue(widget._crosshair_x.isVisible())
        self.assertTrue(widget._crosshair_y.isVisible())

    def test_site_column_widget_accepts_numpy_array_points_in_click_handler(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        snapshot = snapshot_from_live_values(
            "ndscan.rid_0.site.root.", self._make_basic_root_values()
        )
        root = snapshot.get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_2D_SCATTER,
            selected_x_key="param_0",
            selected_y_key="param_1",
            selected_z_key="channel_value",
            selected_group_key=None,
            show_lines=False,
        )

        class _FakeDelta:
            def __init__(self, length):
                self._length = length

            def length(self):
                return self._length

        class _FakePos:
            def __init__(self, value):
                self._value = value

            def __sub__(self, other):
                return _FakeDelta(abs(self._value - other._value))

        class _FakePoint:
            def __init__(self, value, source_index):
                self._pos = _FakePos(value)
                self._source_index = source_index

            def pos(self):
                return self._pos

            def data(self):
                return self._source_index

        class _FakeEvent:
            def __init__(self, value):
                self._pos = _FakePos(value)

            def pos(self):
                return self._pos

        seen = []
        widget.point_selected.connect(lambda path, source_index: seen.append((path, source_index)))
        widget._point_clicked(
            None,
            np.array([_FakePoint(0.0, 4), _FakePoint(1.0, 9)], dtype=object),
            _FakeEvent(0.8),
        )

        self.assertEqual(seen, [((), 9)])

    def test_runtime_viewer_does_not_overwrite_open_combo_selection(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        viewer = RuntimePlotViewer(prefix)

        initial_values = self._make_basic_root_values()
        viewer.data_changed(initial_values)
        column = viewer._columns[0]
        column._x_combo.setCurrentIndex(column._x_combo.findData("param_1"))
        column._x_combo._popup_visible = True

        updated_values = dict(initial_values)
        updated_values[prefix + "points.param_0"] = [0.0, 1.0, 2.0]
        updated_values[prefix + "points.param_1"] = [2.0, 3.0, 4.0]
        updated_values[prefix + "points.channel_value"] = [20.0, 21.0, 22.0]
        updated_values[prefix + "state.num_points"] = 3
        viewer.data_changed(updated_values)

        self.assertEqual(viewer._snapshot.get_site(()).num_points, 3)
        self.assertEqual(column._current_combo_data(column._x_combo), "param_1")

        column._x_combo._popup_visible = False
        viewer._rebuild_columns()
        self.assertEqual(column._current_combo_data(column._x_combo), "param_1")

    def test_runtime_viewer_pause_mode_freezes_until_resumed(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        viewer = RuntimePlotViewer(prefix)

        initial_values = self._make_basic_root_values()
        viewer.data_changed(initial_values)
        self.assertEqual(viewer._snapshot.get_site(()).num_points, 2)

        viewer._pause_checkbox.setChecked(True)
        self.assertTrue(viewer._paused)
        self.assertIn("Paused", viewer._status.text())

        updated_values = dict(initial_values)
        updated_values[prefix + "points.param_0"] = [0.0, 1.0, 2.0]
        updated_values[prefix + "points.param_1"] = [2.0, 3.0, 4.0]
        updated_values[prefix + "points.channel_value"] = [20.0, 21.0, 22.0]
        updated_values[prefix + "state.num_points"] = 3
        viewer.data_changed(updated_values)

        self.assertEqual(viewer._snapshot.get_site(()).num_points, 2)
        self.assertIsNotNone(viewer._pending_values)
        self.assertIn("updates buffered", viewer._status.text())

        viewer._pause_checkbox.setChecked(False)

        self.assertFalse(viewer._paused)
        self.assertIsNone(viewer._pending_values)
        self.assertEqual(viewer._snapshot.get_site(()).num_points, 3)
        self.assertEqual(viewer._status.text(), "Live prepared-runtime scan")

    def test_site_column_widget_supports_point_index_on_y_axis(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        snapshot = snapshot_from_live_values(
            "ndscan.rid_0.site.root.", self._make_basic_root_values()
        )
        root = snapshot.get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=1,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="__point_index__",
            selected_z_key="channel_value",
            selected_group_key="param_1",
            show_lines=False,
        )

        plotted_points = widget._scatter.points()
        self.assertEqual(len(plotted_points), 2)
        self.assertEqual(widget._highlight.points()[0].data(), 1)
        self.assertIn("selected point 1:", widget._selected_readout.text())
        self.assertIn("point_index = 1", widget._selected_readout.text())

    def test_site_column_widget_updates_cursor_readout(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        snapshot = snapshot_from_live_values(
            "ndscan.rid_0.site.root.", self._make_basic_root_values()
        )
        root = snapshot.get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key="channel_value",
            selected_group_key=None,
            show_lines=False,
        )

        widget._last_cursor_plot_pos = (1.25, 23.5)
        widget._update_cursor_readout()

        self.assertIn("x (X Axis / kHz) = 1.25", widget._cursor_readout.text())
        self.assertIn("detector/counts (Detected Counts / cts) = 23.5", widget._cursor_readout.text())

    def test_group_by_choices_only_offer_other_scanned_axes(self):
        snapshot = snapshot_from_live_values(
            "ndscan.rid_0.site.root.", self._make_basic_root_values()
        )
        root = snapshot.get_site(())

        self.assertEqual(
            _group_by_choices(root, "param_0"),
            [
                (_NO_GROUP_KEY, "none"),
                ("param_1", "y (Y Axis / kHz)"),
            ],
        )

    def test_site_column_widget_groups_1d_series_by_scanned_axis(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()
        values["ndscan.rid_0.site.root.points.param_0"] = [0.0, 1.0, 0.0, 1.0]
        values["ndscan.rid_0.site.root.points.param_1"] = [2.0, 2.0, 3.0, 3.0]
        values["ndscan.rid_0.site.root.points.channel_value"] = [20.0, 21.0, 30.0, 31.0]
        values["ndscan.rid_0.site.root.state.num_points"] = 4

        snapshot = snapshot_from_live_values("ndscan.rid_0.site.root.", values)
        root = snapshot.get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=2,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key="channel_value",
            selected_group_key="param_1",
            show_lines=True,
        )

        self.assertEqual(widget._group_combo.currentData(), "param_1")
        self.assertEqual(len(widget._scatter.points()), 4)
        self.assertTrue(widget._legend.isVisible())
        self.assertIn("grouped by y (Y Axis / kHz)", widget._status.text())
        self.assertIn("y (Y Axis / kHz) = 3", widget._selected_readout.text())

    def test_grouped_legend_uses_group_colors_even_without_lines(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()
        values["ndscan.rid_0.site.root.points.param_0"] = [0.0, 1.0, 0.0, 1.0]
        values["ndscan.rid_0.site.root.points.param_1"] = [2.0, 2.0, 3.0, 3.0]
        values["ndscan.rid_0.site.root.points.channel_value"] = [20.0, 21.0, 30.0, 31.0]
        values["ndscan.rid_0.site.root.state.num_points"] = 4

        snapshot = snapshot_from_live_values("ndscan.rid_0.site.root.", values)
        root = snapshot.get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key="channel_value",
            selected_group_key="param_1",
            show_lines=False,
        )

        line_pen = widget._line_item.opts["pen"]
        extra_pen = widget._extra_line_items[0].opts["pen"]
        self.assertNotEqual(line_pen.color().getRgb(), extra_pen.color().getRgb())

    def test_builtin_fit_backend_linear_model(self):
        backend = BuiltinFitBackend()
        result = backend.fit(
            "linear",
            FitRequest(
                x=np.asarray([0.0, 1.0, 2.0], dtype=float),
                y=np.asarray([1.0, 3.0, 5.0], dtype=float),
            ),
        )

        self.assertAlmostEqual(result.parameters["slope"], 2.0, places=6)
        self.assertAlmostEqual(result.parameters["offset"], 1.0, places=6)
        self.assertEqual(result.model_id, "linear")
        self.assertEqual(len(result.curve_x), 200)
        self.assertEqual(len(result.curve_y), 200)

    def test_site_column_widget_can_fit_visible_1d_data(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        snapshot = snapshot_from_live_values(
            "ndscan.rid_0.site.root.", self._make_basic_root_values()
        )
        root = snapshot.get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key="channel_value",
            selected_group_key=None,
            show_lines=False,
        )

        widget._fit_model_combo.setCurrentIndex(widget._fit_model_combo.findData("linear"))
        widget._fit_now()

        self.assertTrue(widget._fit_active)
        self.assertTrue(widget._fit_curve_item.isVisible())
        self.assertIn("fit: linear;", widget._fit_readout.text())
        self.assertIn("slope=", widget._fit_readout.text())

    def test_site_column_widget_can_fit_selected_group(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()
        values["ndscan.rid_0.site.root.points.param_0"] = [0.0, 1.0, 0.0, 1.0]
        values["ndscan.rid_0.site.root.points.param_1"] = [2.0, 2.0, 3.0, 3.0]
        values["ndscan.rid_0.site.root.points.channel_value"] = [20.0, 21.0, 30.0, 31.0]
        values["ndscan.rid_0.site.root.state.num_points"] = 4

        snapshot = snapshot_from_live_values("ndscan.rid_0.site.root.", values)
        root = snapshot.get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=2,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key="channel_value",
            selected_group_key="param_1",
            show_lines=False,
        )

        widget._fit_model_combo.setCurrentIndex(widget._fit_model_combo.findData("linear"))
        widget._fit_target_combo.setCurrentIndex(
            widget._fit_target_combo.findData("selected_group")
        )
        widget._fit_now()

        self.assertTrue(widget._fit_active)
        self.assertAlmostEqual(widget._fit_target_group_value, 3.0, places=6)
        self.assertIn("fit: linear on y (Y Axis / kHz) = 3;", widget._fit_readout.text())

    def test_site_column_widget_clears_fit_when_data_updates(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        snapshot = snapshot_from_live_values(
            "ndscan.rid_0.site.root.", self._make_basic_root_values()
        )
        root = snapshot.get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key="channel_value",
            selected_group_key=None,
            show_lines=False,
        )

        widget._fit_model_combo.setCurrentIndex(widget._fit_model_combo.findData("linear"))
        widget._fit_now()
        self.assertTrue(widget._fit_active)

        updated_values = self._make_basic_root_values()
        updated_values["ndscan.rid_0.site.root.points.param_0"] = [0.0, 1.0, 2.0]
        updated_values["ndscan.rid_0.site.root.points.param_1"] = [2.0, 3.0, 4.0]
        updated_values["ndscan.rid_0.site.root.points.channel_value"] = [20.0, 21.0, 22.0]
        updated_values["ndscan.rid_0.site.root.state.num_points"] = 3
        updated_snapshot = snapshot_from_live_values("ndscan.rid_0.site.root.", updated_values)
        updated_root = updated_snapshot.get_site(())

        widget.update_state(
            site=updated_root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key="channel_value",
            selected_group_key=None,
            show_lines=False,
        )

        self.assertFalse(widget._fit_active)
        self.assertFalse(widget._fit_curve_item.isVisible())
        self.assertEqual(widget._fit_readout.text(), "fit: none")


class RuntimeAppletLaunchTest(HasEnvironmentCase):
    def test_prepared_scan_experiment_launches_runtime_applet(self):
        RuntimeAppletExperiment = make_fragment_prepared_scan_exp(
            AppletLaunchFragment,
            lambda fragment: ScanRequest.single(metadata={"demo_name": "runtime_applet"}),
        )

        exp = self.create(RuntimeAppletExperiment)
        exp.prepare()
        exp.run()

        self.ccb.issue.assert_called_once()
        args, kwargs = self.ccb.issue.call_args
        self.assertEqual(args[0], "create_applet")
        self.assertIn("-m ndscan.runtime_applet", args[2])
        self.assertIn("--prefix=ndscan.rid_0.site.root.", args[2])
        self.assertEqual(kwargs["group"], "ndscan")

    @patch("examples.host_runtime_live_viewer_demo.time.sleep", return_value=None)
    def test_live_viewer_demo_example_runs(self, _sleep):
        exp = self.create(HostRuntimeLiveViewerDemo)
        exp.prepare()
        exp.run()

        dataset_keys = set(self.dataset_db.data.keys())
        self.assertIn("ndscan.rid_0.site.root.scan_outer.site.path", dataset_keys)
        self.assertIn(
            "ndscan.rid_0.site.root.scan_outer.coarse_scan.site.path",
            dataset_keys,
        )
        self.assertIn(
            "ndscan.rid_0.site.root.scan_outer.fine_scan.site.path",
            dataset_keys,
        )
        for actual, expected in zip(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.points.channel_0"),
            [-1.8, -0.8, 0.2, 1.1, 2.0],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.points.channel_1"),
            [-1.8, -0.8, 0.2, 1.1, 2.0],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.coarse_scan.points.param_1"),
            [-1.8] * 9 + [-0.8] * 9 + [0.2] * 9 + [1.1] * 9 + [2.0] * 9,
        )
        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.coarse_scan.points.param_2"),
            [0.10] * 9 + [0.16] * 9 + [0.24] * 9 + [0.33] * 9 + [0.45] * 9,
        )
        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.fine_scan.points.param_1"),
            [-1.8] * 7 + [-0.8] * 7 + [0.2] * 7 + [1.1] * 7 + [2.0] * 7,
        )
        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.fine_scan.points.param_2"),
            [0.10] * 7 + [0.16] * 7 + [0.24] * 7 + [0.33] * 7 + [0.45] * 7,
        )

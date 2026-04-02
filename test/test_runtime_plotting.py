import json
import os
import unittest
from unittest.mock import patch

import numpy as np
from examples.host_runtime_live_viewer_demo import HostRuntimeLiveViewerDemo
from mock_environment import HasEnvironmentCase

from ndscan.define.fragment import ExpFragment
from ndscan.define.result_channels import FloatChannel
from ndscan.plots.runtime.fitting import (
    BuiltinFitBackend,
    FitRequest,
    default_fit_backend,
)
from ndscan.plots.runtime.live import snapshot_from_live_values
from ndscan.plots.runtime.viewer import (
    _ARRAY_SERIES_GROUP_KEY,
    _DisplayedPointSelection,
    _NO_GROUP_KEY,
    _PLOT_MODE_BO,
    _PLOT_MODE_1D,
    _PLOT_MODE_2D_IMAGE,
    _PLOT_MODE_2D_SCATTER,
    _REPEAT_COMBINE_SEM,
    _REPEAT_COMBINE_STD,
    _SeriesUiState,
    _SiteColumnWidget,
    RuntimePlotViewer,
    _default_x_choices,
    _default_y_choices,
    _group_by_choices,
    _wrap_overlay_text,
)
from ndscan.runtime.api import make_fragment_prepared_scan_exp
from ndscan.scan.request import ScanRequest
from ndscan._qt import QtGui, QtWidgets


class AppletLaunchFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_result("value", FloatChannel)

    def run_once(self):
        self.value.push(1.0)


class RuntimeLiveSnapshotTest(unittest.TestCase):
    def test_overlay_text_wraps_to_at_most_two_lines(self):
        wrapped = _wrap_overlay_text(
            "straight_line: m=1.23456, b=10.2345, extra=123.4, more=456.7, tail=999.1",
            max_chars=32,
            max_lines=2,
        )

        lines = wrapped.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[-1].endswith("…"))

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

    def test_runtime_choices_use_declared_schema_before_points_exist(self):
        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.frequency",
                            "description": "Probe Frequency",
                            "type": "float",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "channel_value": {
                        "path": "detector/probability",
                        "description": "Survival Probability",
                        "type": "float",
                    }
                }
            ),
            prefix + "state.num_points": 0,
            prefix + "state.completed": False,
        }

        snapshot = snapshot_from_live_values(prefix, values)
        root = snapshot.get_site(())

        self.assertEqual(
            _default_x_choices(root),
            [
                ("param_0", "frequency (Probe Frequency / MHz)"),
                ("channel_value", "detector/probability (Survival Probability)"),
                ("__point_index__", "point_index"),
            ],
        )
        self.assertEqual(
            _default_y_choices(root),
            [
                ("param_0", "frequency (Probe Frequency / MHz)"),
                ("channel_value", "detector/probability (Survival Probability)"),
                ("__point_index__", "point_index"),
            ],
        )

    def test_runtime_choices_include_numeric_array_channels(self):
        prefix = "ndscan.rid_0.site.root."
        values = {
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
                            "type": "float",
                            "spec": {"unit": "kHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "counts": {
                        "path": "detector/counts",
                        "description": "ROI Counts",
                        "type": "array",
                        "element_type": "int",
                        "shape": [2, 1],
                        "dim_names": ["group", "roi"],
                        "unit": "cts",
                    }
                }
            ),
            prefix + "points.param_0": [0.0, 1.0],
            prefix + "points.counts": np.asarray([[[10], [20]], [[11], [21]]], dtype=int),
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
        }

        snapshot = snapshot_from_live_values(prefix, values)
        root = snapshot.get_site(())

        self.assertEqual(
            _default_y_choices(root),
            [
                ("param_0", "x (X Axis / kHz)"),
                ("counts", "detector/counts (ROI Counts / cts)"),
                ("__point_index__", "point_index"),
            ],
        )

    def test_site_column_widget_shows_run_id_overlay(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()
        prefix = "ndscan.rid_0.site.root."
        values[prefix + "site.source_id"] = "rid_123"
        snapshot = snapshot_from_live_values(prefix, values)
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
            selected_z_key=None,
            selected_group_key=None,
            show_lines=False,
        )

        self.assertEqual(widget._run_id_overlay.display_text(), "RID 123")
        self.assertTrue(widget._run_id_overlay.isVisible())

    def test_site_column_widget_plots_selected_array_slice(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.frequency",
                            "description": "Probe Frequency",
                            "type": "float",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "counts": {
                        "path": "detector/counts",
                        "description": "ROI Counts",
                        "type": "array",
                        "element_type": "int",
                        "shape": [2, 1],
                        "dim_names": ["group", "roi"],
                        "unit": "cts",
                    }
                }
            ),
            prefix + "points.param_0": [0.0, 1.0],
            prefix + "points.counts": np.asarray([[[10], [20]], [[11], [21]]], dtype=int),
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
        }

        root = snapshot_from_live_values(prefix, values).get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="counts",
            selected_group_key=None,
            selected_x_index_tokens=None,
            selected_y_index_tokens=("1", "0"),
            selected_z_index_tokens=None,
            selected_z_key=None,
            show_lines=False,
        )

        self.assertTrue(widget._y_index_widget.has_array_schema())
        np.testing.assert_allclose(widget._current_plot_arrays[0], np.asarray([0.0, 1.0]))
        np.testing.assert_allclose(widget._current_plot_arrays[1], np.asarray([20.0, 21.0]))
        self.assertIn("[group=1, roi=0]", widget._current_y_label)

    def test_site_column_widget_treats_array_range_as_repeated_points_by_default(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.frequency",
                            "description": "Probe Frequency",
                            "type": "float",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "counts": {
                        "path": "detector/counts",
                        "description": "ROI Counts",
                        "type": "array",
                        "element_type": "int",
                        "shape": [2, 1],
                        "dim_names": ["group", "roi"],
                        "unit": "cts",
                    }
                }
            ),
            prefix + "points.param_0": [0.0, 1.0],
            prefix + "points.counts": np.asarray([[[10], [20]], [[11], [21]]], dtype=int),
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
        }

        root = snapshot_from_live_values(prefix, values).get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="counts",
            selected_group_key=None,
            selected_x_index_tokens=None,
            selected_y_index_tokens=("all", "0"),
            selected_z_index_tokens=None,
            selected_z_key=None,
            show_lines=False,
        )

        self.assertEqual(widget._rendered_group_label_map, {})
        self.assertIsNone(widget._current_group_label)
        self.assertEqual(len(widget._current_plot_arrays[0]), 4)
        self.assertIn("repeated group values", widget._status.text())

    def test_site_column_widget_can_render_multiple_series_rows(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.frequency",
                            "description": "Probe Frequency",
                            "type": "float",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "channel_0": {
                        "path": "detector/probability",
                        "description": "Probability",
                        "type": "float",
                    },
                    "channel_1": {
                        "path": "detector/counts",
                        "description": "Counts",
                        "type": "float",
                    },
                }
            ),
            prefix + "points.param_0": [0.0, 1.0, 2.0],
            prefix + "points.channel_0": [0.1, 0.5, 0.9],
            prefix + "points.channel_1": [10.0, 11.0, 12.0],
            prefix + "state.num_points": 3,
            prefix + "state.completed": False,
        }

        root = snapshot_from_live_values(prefix, values).get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_0",
            selected_z_key=None,
            selected_group_key=None,
            show_lines=True,
            selected_series_states=(
                _SeriesUiState(y_key="channel_0"),
                _SeriesUiState(y_key="channel_1"),
            ),
        )

        self.assertEqual(len(widget._series_rows), 2)
        self.assertEqual(len(widget._current_visible_fit_series), 2)
        self.assertTrue(widget._legend.isVisible())
        self.assertIn("2 series", widget._status.text())

    def test_site_column_widget_can_shrink_visible_series_rows(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.frequency",
                            "description": "Probe Frequency",
                            "type": "float",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "channel_0": {
                        "path": "detector/probability",
                        "description": "Probability",
                        "type": "float",
                    },
                    "channel_1": {
                        "path": "detector/counts",
                        "description": "Counts",
                        "type": "float",
                    },
                }
            ),
            prefix + "points.param_0": [0.0, 1.0, 2.0],
            prefix + "points.channel_0": [0.1, 0.5, 0.9],
            prefix + "points.channel_1": [10.0, 11.0, 12.0],
            prefix + "state.num_points": 3,
            prefix + "state.completed": False,
        }

        root = snapshot_from_live_values(prefix, values).get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_0",
            selected_z_key=None,
            selected_group_key=None,
            show_lines=True,
            selected_series_states=(
                _SeriesUiState(y_key="channel_0"),
                _SeriesUiState(y_key="channel_1"),
            ),
        )

        widget.update_state(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_display_point_key=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_0",
            selected_z_key=None,
            selected_group_key=None,
            show_lines=True,
            selected_series_states=(
                _SeriesUiState(y_key="channel_0"),
            ),
        )

        self.assertEqual(len(widget._current_visible_fit_series), 1)
        self.assertFalse(widget._legend.isVisible())

    def test_site_column_widget_can_group_by_array_range_dimension(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.frequency",
                            "description": "Probe Frequency",
                            "type": "float",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "counts": {
                        "path": "detector/counts",
                        "description": "ROI Counts",
                        "type": "array",
                        "element_type": "int",
                        "shape": [2, 1],
                        "dim_names": ["group", "roi"],
                        "unit": "cts",
                    }
                }
            ),
            prefix + "points.param_0": [0.0, 1.0],
            prefix + "points.counts": np.asarray([[[10], [20]], [[11], [21]]], dtype=int),
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
        }

        root = snapshot_from_live_values(prefix, values).get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="counts",
            selected_group_key=_ARRAY_SERIES_GROUP_KEY,
            selected_x_index_tokens=None,
            selected_y_index_tokens=("all", "0"),
            selected_z_index_tokens=None,
            selected_z_key=None,
            show_lines=False,
        )

        self.assertEqual(
            widget._rendered_group_label_map,
            {0.0: "group=0", 1.0: "group=1"},
        )
        self.assertEqual(widget._current_group_label, "group")
        self.assertIn("split by group", widget._status.text())

    def test_repeat_combine_can_average_over_array_range_when_not_grouped(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.frequency",
                            "description": "Probe Frequency",
                            "type": "float",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "counts": {
                        "path": "detector/counts",
                        "description": "ROI Counts",
                        "type": "array",
                        "element_type": "float",
                        "shape": [2, 1],
                        "dim_names": ["group", "roi"],
                        "unit": "cts",
                    }
                }
            ),
            prefix + "points.param_0": [0.0, 1.0],
            prefix + "points.counts": np.asarray([[[10.0], [20.0]], [[14.0], [24.0]]]),
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
        }

        root = snapshot_from_live_values(prefix, values).get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="counts",
            selected_group_key=None,
            selected_x_index_tokens=None,
            selected_y_index_tokens=("all", "0"),
            selected_z_index_tokens=None,
            selected_z_key=None,
            show_lines=False,
            selected_repeat_combine_mode=_REPEAT_COMBINE_STD,
        )

        np.testing.assert_allclose(widget._current_plot_arrays[0], np.asarray([0.0, 1.0]))
        np.testing.assert_allclose(widget._current_plot_arrays[1], np.asarray([15.0, 19.0]))

    def test_runtime_viewer_offers_bo_dashboard_mode_for_root_nubo_sites(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "BoRootFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.x",
                            "description": "X",
                            "type": "float",
                            "spec": {},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "channel_0": {
                        "path": "objective",
                        "description": "Objective",
                        "type": "float",
                    }
                }
            ),
            prefix + "scan.point_policy": json.dumps(
                {
                    "kind": "ask_tell_optimiser",
                    "backend": {
                        "kind": "nubo_bayesian_optimisation",
                        "dims": 1,
                        "bounds": [[-1.0], [1.0]],
                        "minimise": True,
                        "fit_steps": 2,
                        "fit_lr": 0.05,
                        "surrogate_num_starts": 1,
                        "observation_noise_floor": 1e-6,
                    },
                    "observation_extractor": {
                        "kind": "scalar_channel",
                        "channel_key": "channel_0",
                        "noise_channel_key": None,
                    },
                }
            ),
            prefix + "points.param_0": [-1.0, 0.0, 1.0],
            prefix + "points.channel_0": [1.0, 0.2, 0.8],
            prefix + "points.metadata.decision_source": ["seed", "bo", "explore"],
            prefix + "state.num_points": 3,
            prefix + "state.completed": False,
        }

        snapshot = snapshot_from_live_values(prefix, values)
        root = snapshot.get_site(())

        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=None,
            selected_x_key=None,
            selected_y_key=None,
            selected_z_key=None,
            selected_group_key=None,
            show_lines=True,
        )

        mode_values = [
            widget._plot_mode_combo.itemData(index)
            for index in range(widget._plot_mode_combo.count())
        ]
        self.assertIn(_PLOT_MODE_BO, mode_values)
        self.assertEqual(widget._selected_plot_mode(), _PLOT_MODE_BO)
        self.assertIs(widget._plot_stack.currentWidget(), widget._bo_plot_widget)

        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key=None,
            selected_y_key=None,
            selected_z_key=None,
            selected_group_key=None,
            show_lines=False,
        )
        self.assertEqual(widget._x_combo.currentData(), "param_0")
        self.assertEqual(widget._y_combo.currentData(), "channel_0")

    def test_repeated_site_defaults_x_to_time_when_parameter_is_constant(self):
        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RepeatFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": False,
                        "param": {
                            "fqn": "demo.repeat.probe_frequency",
                            "description": "Probe Frequency",
                            "type": "float",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "channel_value": {
                        "path": "detector/survived",
                        "description": "Survived",
                        "type": "float",
                    }
                }
            ),
            prefix + "points.param_0": [0.2, 0.2, 0.2],
            prefix + "points.channel_value": [1.0, 0.0, 1.0],
            prefix + "points.acquired_at_unix": [10.0, 11.0, 12.0],
            prefix + "state.num_points": 3,
            prefix + "state.completed": False,
        }

        snapshot = snapshot_from_live_values(prefix, values)
        root = snapshot.get_site(())

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key=None,
            selected_y_key=None,
            selected_z_key=None,
            selected_group_key=None,
            show_lines=False,
        )

        self.assertEqual(widget._x_combo.currentData(), "acquired_at_unix")
        self.assertEqual(widget._y_combo.currentData(), "channel_value")

    def test_repeated_site_defaults_x_to_point_index_without_time_stream(self):
        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RepeatFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": False,
                        "param": {
                            "fqn": "demo.repeat.probe_frequency",
                            "description": "Probe Frequency",
                            "type": "float",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "channel_value": {
                        "path": "detector/survived",
                        "description": "Survived",
                        "type": "float",
                    }
                }
            ),
            prefix + "points.param_0": [0.2, 0.2, 0.2],
            prefix + "points.channel_value": [1.0, 0.0, 1.0],
            prefix + "state.num_points": 3,
            prefix + "state.completed": False,
        }

        snapshot = snapshot_from_live_values(prefix, values)
        root = snapshot.get_site(())

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key=None,
            selected_y_key=None,
            selected_z_key=None,
            selected_group_key=None,
            show_lines=False,
        )

        self.assertEqual(widget._x_combo.currentData(), "__point_index__")
        self.assertEqual(widget._y_combo.currentData(), "channel_value")

    def test_snapshot_from_live_values_decodes_analysis_artifacts(self):
        prefix = "ndscan.rid_0.site.root."
        values = self._make_basic_root_values()
        values[prefix + "analysis.artifact.line_fit"] = json.dumps(
            {
                "kind": "model_fit",
                "provider": "sensible_fitting",
                "model_id": "straight_line",
                "parameters": {
                    "slope": {"value": 2.0, "stderr": 0.1},
                },
            }
        )
        values[prefix + "analysis.online_artifact.running_fit"] = json.dumps(
            {
                "kind": "model_fit",
                "provider": "sensible_fitting",
                "model_id": "straight_line",
                "parameters": {
                    "slope": {"value": 2.1, "stderr": 0.2},
                },
            }
        )

        snapshot = snapshot_from_live_values(prefix, values)
        root = snapshot.get_site(())

        self.assertEqual(root.analysis_artifacts["line_fit"]["model_id"], "straight_line")
        self.assertAlmostEqual(
            root.online_analysis_artifacts["running_fit"]["parameters"]["slope"]["value"],
            2.1,
            places=6,
        )

    def test_site_column_widget_renders_final_curve_annotations(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()
        values["ndscan.rid_0.site.root.analysis.annotations"] = json.dumps(
            [
                {
                    "kind": "curve",
                    "coordinates": {
                        "param_0": {"kind": "fixed", "value": [0.0, 1.0]},
                        "channel_value": {
                            "kind": "fixed",
                            "value": [19.5, 21.5],
                        },
                    },
                    "parameters": {},
                    "data": {},
                }
            ]
        )

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
            selected_group_key=None,
            show_lines=False,
        )

        self.assertEqual(len(widget._annotation_items), 1)
        x_data, y_data = widget._annotation_items[0].getData()
        np.testing.assert_allclose(x_data, [0.0, 1.0])
        np.testing.assert_allclose(y_data, [19.5, 21.5])

    def test_site_column_widget_renders_error_bars_from_channel_display_hints(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()
        prefix = "ndscan.rid_0.site.root."
        values[prefix + "scan.channels"] = json.dumps(
            {
                "channel_value": {
                    "path": "detector/probability",
                    "description": "Survival Probability",
                    "type": "float",
                },
                "channel_error": {
                    "path": "detector/probability_error",
                    "description": "Survival Probability Error",
                    "type": "float",
                    "display_hints": {
                        "error_bar_for": "detector/probability",
                    },
                },
            }
        )
        values[prefix + "points.channel_value"] = [0.3, 0.6]
        values[prefix + "points.channel_error"] = [0.05, 0.08]

        snapshot = snapshot_from_live_values(prefix, values)
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

        np.testing.assert_allclose(widget._error_bar_item.opts["x"], [0.0, 1.0])
        np.testing.assert_allclose(widget._error_bar_item.opts["y"], [0.3, 0.6])
        np.testing.assert_allclose(widget._error_bar_item.opts["top"], [0.05, 0.08])
        np.testing.assert_allclose(widget._error_bar_item.opts["bottom"], [0.05, 0.08])
        expected_color = QtGui.QColor("#1f77b4").darker(145)
        expected_color.setAlpha(220)
        self.assertEqual(
            widget._error_bar_item.opts["pen"].color().getRgb(),
            expected_color.getRgb(),
        )

    def test_site_column_widget_renders_final_artifact_curve_annotations(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()
        prefix = "ndscan.rid_0.site.root."
        values[prefix + "state.completed"] = True
        values[prefix + "analysis.artifact.line_fit"] = json.dumps(
            {
                "kind": "model_fit",
                "provider": "sensible_fitting",
                "model_id": "straight_line",
                "model_name": "straight line",
                "parameters": {
                    "m": {"value": 2.0},
                    "b": {"value": 19.5},
                },
                "stats": {"success": True},
            }
        )
        values[prefix + "analysis.annotations"] = json.dumps(
            [
                {
                    "kind": "artifact_curve",
                    "coordinates": {},
                    "parameters": {
                        "artifact": "line_fit",
                        "x_axis": "param_0",
                        "y_axis": "channel_value",
                    },
                    "data": {},
                }
            ]
        )

        snapshot = snapshot_from_live_values(prefix, values)
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

        self.assertEqual(len(widget._annotation_items), 1)
        x_data, y_data = widget._annotation_items[0].getData()
        self.assertEqual(len(x_data), 200)
        self.assertAlmostEqual(float(x_data[0]), 0.0)
        self.assertAlmostEqual(float(x_data[-1]), 1.0)
        self.assertAlmostEqual(float(y_data[0]), 19.5)
        self.assertAlmostEqual(float(y_data[-1]), 21.5)
        self.assertIn("analysis (final):", widget._artifact_readout.text())
        self.assertIn("line_fit:", widget._artifact_readout.text())
        self.assertIn("status: ok", widget._artifact_readout.text())

    def test_site_column_widget_matches_artifact_curve_to_array_slice(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
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
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "channel_value": {
                        "path": "detector/probability",
                        "description": "Probability",
                        "type": "array",
                        "element_type": "float",
                        "shape": [2],
                        "dim_names": ["roi"],
                    }
                }
            ),
            prefix + "points.param_0": [0.0, 1.0],
            prefix + "points.channel_value": np.asarray(
                [[0.2, 0.6], [0.3, 0.7]],
                dtype=float,
            ),
            prefix + "state.num_points": 2,
            prefix + "state.completed": True,
            prefix + "analysis.artifact.line_fit": json.dumps(
                {
                    "kind": "model_fit",
                    "provider": "sensible_fitting",
                    "model_id": "straight_line",
                    "model_name": "straight line",
                    "parameters": {
                        "m": {"value": 0.1},
                        "b": {"value": 0.2},
                    },
                    "stats": {"success": True},
                }
            ),
            prefix + "analysis.annotations": json.dumps(
                [
                    {
                        "kind": "artifact_curve",
                        "coordinates": {},
                        "parameters": {
                            "artifact": "line_fit",
                            "x_axis": "param_0",
                            "y_axis": "channel_value",
                            "y_indices": [0],
                        },
                        "data": {},
                    }
                ]
            ),
        }

        root = snapshot_from_live_values(prefix, values).get_site(())
        matching_widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key=None,
            selected_group_key=None,
            show_lines=False,
            selected_y_index_tokens=("0",),
        )
        non_matching_widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key=None,
            selected_group_key=None,
            show_lines=False,
            selected_y_index_tokens=("1",),
        )

        self.assertEqual(len(matching_widget._annotation_items), 1)
        self.assertEqual(len(non_matching_widget._annotation_items), 0)

    def test_site_column_widget_renders_artifact_location_annotations(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()
        prefix = "ndscan.rid_0.site.root."
        values[prefix + "state.completed"] = True
        values[prefix + "analysis.artifact.line_fit"] = json.dumps(
            {
                "kind": "model_fit",
                "provider": "sensible_fitting",
                "model_id": "straight_line",
                "model_name": "straight line",
                "parameters": {
                    "m": {"value": 2.0},
                    "b": {"value": 19.5},
                    "x0": {"value": 0.5, "stderr": 0.1},
                },
                "stats": {"success": True},
            }
        )
        values[prefix + "analysis.annotations"] = json.dumps(
            [
                {
                    "kind": "artifact_location",
                    "coordinates": {},
                    "parameters": {
                        "artifact": "line_fit",
                        "axis": "param_0",
                        "parameter": "x0",
                        "associated_channels": ["channel_value"],
                    },
                    "data": {},
                }
            ]
        )

        snapshot = snapshot_from_live_values(prefix, values)
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

        self.assertEqual(len(widget._annotation_items), 3)
        positions = sorted(float(item.value()) for item in widget._annotation_items)
        np.testing.assert_allclose(positions, [0.4, 0.5, 0.6])

    def test_site_column_widget_prefers_online_annotations_for_active_child_slice(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "state.num_points": 2,
            prefix + "scan_x.site.path": json.dumps(["scan_x"]),
            prefix + "scan_x.site.parent_path": json.dumps([]),
            prefix + "scan_x.site.fragment_fqn": "ChildFragment",
            prefix + "scan_x.scan.parameters": json.dumps(
                {
                    "child_x": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.child.x",
                            "description": "Child X",
                            "spec": {"unit": "Hz"},
                        },
                    }
                }
            ),
            prefix + "scan_x.scan.channels": json.dumps(
                {
                    "child_value": {
                        "path": "child/value",
                        "description": "Child Value",
                        "type": "float",
                    }
                }
            ),
            prefix + "scan_x.points.child_x": [0.0, 1.0, 2.0, 3.0],
            prefix + "scan_x.points.child_value": [10.0, 11.0, 20.0, 21.0],
            prefix + "scan_x.segments.start_index": [0, 2],
            prefix + "scan_x.segments.parent_point_index": [0, 1],
            prefix + "scan_x.state.current_segment": 1,
            prefix + "scan_x.state.num_points": 4,
            prefix + "scan_x.state.completed": False,
            prefix + "scan_x.segments.analysis.final_feedback": [
                json.dumps(
                    {
                        "outputs": {},
                        "artifacts": {
                            "line_fit": {
                                "kind": "model_fit",
                                "provider": "sensible_fitting",
                                "model_id": "straight_line",
                                "parameters": {
                                    "m": {"value": 2.0},
                                    "b": {"value": 9.5},
                                },
                            }
                        },
                        "annotations": [
                            {
                                "kind": "artifact_curve",
                                "coordinates": {},
                                "parameters": {
                                    "artifact": "line_fit",
                                    "x_axis": "child_x",
                                    "y_axis": "child_value",
                                },
                                "data": {},
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "outputs": {},
                        "artifacts": {
                            "line_fit": {
                                "kind": "model_fit",
                                "provider": "sensible_fitting",
                                "model_id": "straight_line",
                                "parameters": {
                                    "m": {"value": 2.0},
                                    "b": {"value": 15.0},
                                },
                            }
                        },
                        "annotations": [
                            {
                                "kind": "artifact_curve",
                                "coordinates": {},
                                "parameters": {
                                    "artifact": "line_fit",
                                    "x_axis": "child_x",
                                    "y_axis": "child_value",
                                },
                                "data": {},
                            }
                        ],
                    }
                ),
            ],
            prefix + "scan_x.analysis.online_annotation.running_summary": json.dumps(
                [
                    {
                        "kind": "curve",
                        "coordinates": {
                            "child_x": {"kind": "fixed", "value": [2.0, 3.0]},
                            "child_value": {
                                "kind": "fixed",
                                "value": [19.5, 21.5],
                            },
                        },
                        "parameters": {},
                        "data": {},
                    }
                ]
            ),
        }

        snapshot = snapshot_from_live_values(prefix, values)
        child = snapshot.get_site(("scan_x",))
        self.assertEqual(len(child.segment_final_analysis), 2)
        self.assertEqual(
            child.final_analysis_for_segment(0).annotations[0]["kind"],
            "artifact_curve",
        )

        widget = _SiteColumnWidget(
            site=child,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="child_x",
            selected_y_key="child_value",
            selected_z_key="child_value",
            selected_group_key=None,
            show_lines=False,
        )

        self.assertEqual(len(widget._annotation_items), 1)
        x_data, y_data = widget._annotation_items[0].getData()
        np.testing.assert_allclose(x_data, [2.0, 3.0])
        np.testing.assert_allclose(y_data, [19.5, 21.5])

        widget.update_state(
            site=child,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=0,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="child_x",
            selected_y_key="child_value",
            selected_z_key="child_value",
            selected_group_key=None,
            show_lines=False,
        )

        self.assertEqual(len(widget._annotation_items), 1)
        x_data, y_data = widget._annotation_items[0].getData()
        self.assertEqual(len(x_data), 200)
        self.assertAlmostEqual(float(x_data[0]), 0.0)
        self.assertAlmostEqual(float(x_data[-1]), 1.0)
        self.assertAlmostEqual(float(y_data[0]), 9.5)
        self.assertAlmostEqual(float(y_data[-1]), 11.5)

    def test_active_child_segment_does_not_reuse_previous_final_annotation(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "state.num_points": 2,
            prefix + "scan_x.site.path": json.dumps(["scan_x"]),
            prefix + "scan_x.site.parent_path": json.dumps([]),
            prefix + "scan_x.site.fragment_fqn": "ChildFragment",
            prefix + "scan_x.scan.parameters": json.dumps(
                {
                    "child_x": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.child.x",
                            "description": "Child X",
                            "spec": {"unit": "Hz"},
                        },
                    }
                }
            ),
            prefix + "scan_x.scan.channels": json.dumps(
                {
                    "child_value": {
                        "path": "child/value",
                        "description": "Child Value",
                        "type": "float",
                    }
                }
            ),
            prefix + "scan_x.points.child_x": [0.0, 1.0, 2.0],
            prefix + "scan_x.points.child_value": [10.0, 11.0, 20.0],
            prefix + "scan_x.segments.start_index": [0, 2],
            prefix + "scan_x.segments.parent_point_index": [0, 1],
            prefix + "scan_x.state.current_segment": 1,
            prefix + "scan_x.state.num_points": 3,
            prefix + "scan_x.state.completed": False,
            prefix + "scan_x.segments.analysis.final_feedback": [
                json.dumps(
                    {
                        "outputs": {},
                        "artifacts": {
                            "line_fit": {
                                "kind": "model_fit",
                                "provider": "sensible_fitting",
                                "model_id": "straight_line",
                                "parameters": {
                                    "m": {"value": 1.0},
                                    "b": {"value": 10.0},
                                },
                            }
                        },
                        "annotations": [
                            {
                                "kind": "artifact_curve",
                                "coordinates": {},
                                "parameters": {
                                    "artifact": "line_fit",
                                    "x_axis": "child_x",
                                    "y_axis": "child_value",
                                },
                                "data": {},
                            }
                        ],
                    }
                ),
            ],
        }

        snapshot = snapshot_from_live_values(prefix, values)
        child = snapshot.get_site(("scan_x",))
        widget = _SiteColumnWidget(
            site=child,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="child_x",
            selected_y_key="child_value",
            selected_z_key="child_value",
            selected_group_key=None,
            show_lines=False,
        )

        self.assertEqual(widget._annotation_items, [])

        widget.update_state(
            site=child,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=0,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="child_x",
            selected_y_key="child_value",
            selected_z_key="child_value",
            selected_group_key=None,
            show_lines=False,
        )
        self.assertEqual(len(widget._annotation_items), 1)
        self.assertIn("straight_line:", widget._run_id_overlay.display_text())
        self.assertIn("m=1", widget._run_id_overlay.display_text())
        self.assertIn("b=10", widget._run_id_overlay.display_text())

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
        self.assertTrue(widget._repeat_combine_combo.isHidden())

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
        self.assertFalse(widget._repeat_combine_combo.isHidden())

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
        self.assertTrue(widget._repeat_combine_combo.isHidden())
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

    def test_array_range_scatter_points_carry_distinct_display_selections(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.frequency",
                            "description": "Probe Frequency",
                            "type": "float",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "counts": {
                        "path": "detector/counts",
                        "description": "ROI Counts",
                        "type": "array",
                        "element_type": "int",
                        "shape": [2, 1],
                        "dim_names": ["group", "roi"],
                        "unit": "cts",
                    }
                }
            ),
            prefix + "points.param_0": [0.0, 1.0],
            prefix + "points.counts": np.asarray([[[10], [20]], [[11], [21]]], dtype=int),
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
        }

        root = snapshot_from_live_values(prefix, values).get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="counts",
            selected_group_key=_ARRAY_SERIES_GROUP_KEY,
            selected_x_index_tokens=None,
            selected_y_index_tokens=("all", "0"),
            selected_z_index_tokens=None,
            selected_z_key=None,
            show_lines=False,
        )

        selections = [point.data() for point in widget._scatter.points()]
        self.assertEqual(len(selections), 4)
        self.assertTrue(all(isinstance(selection, _DisplayedPointSelection) for selection in selections))
        self.assertEqual(selections[-1].source_index, 1)
        self.assertNotEqual(selections[1].display_key, selections[3].display_key)

    def test_array_range_selection_details_follow_clicked_group_point(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.frequency",
                            "description": "Probe Frequency",
                            "type": "float",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "counts": {
                        "path": "detector/counts",
                        "description": "ROI Counts",
                        "type": "array",
                        "element_type": "int",
                        "shape": [2, 1],
                        "dim_names": ["group", "roi"],
                        "unit": "cts",
                    }
                }
            ),
            prefix + "points.param_0": [0.0, 1.0],
            prefix + "points.counts": np.asarray([[[10], [20]], [[11], [21]]], dtype=int),
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
        }

        root = snapshot_from_live_values(prefix, values).get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=None,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="counts",
            selected_group_key=_ARRAY_SERIES_GROUP_KEY,
            selected_x_index_tokens=None,
            selected_y_index_tokens=("all", "0"),
            selected_z_index_tokens=None,
            selected_z_key=None,
            show_lines=False,
        )

        selected = widget._scatter.points()[-1].data()
        assert isinstance(selected, _DisplayedPointSelection)
        widget.update_state(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=selected.source_index,
            selected_display_point_key=selected.display_key,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="counts",
            selected_group_key=_ARRAY_SERIES_GROUP_KEY,
            selected_x_index_tokens=None,
            selected_y_index_tokens=("all", "0"),
            selected_z_index_tokens=None,
            selected_z_key=None,
            show_lines=False,
        )

        self.assertIn("selected point 1:", widget._selected_readout.text())
        self.assertIn("Probe Frequency / MHz) = 1", widget._selected_readout.text())
        self.assertIn("ROI Counts / cts) [group=all, roi=0] = 21", widget._selected_readout.text())
        self.assertIn("group = group=1", widget._selected_readout.text())
        highlight = widget._highlight.points()
        self.assertEqual(len(highlight), 1)
        self.assertEqual(highlight[0].data(), 1)
        self.assertAlmostEqual(float(highlight[0].pos().x()), 1.0)
        self.assertAlmostEqual(float(highlight[0].pos().y()), 21.0)

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

    def test_runtime_viewer_preserves_child_axis_selection_when_parent_point_changes(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "points.param_0": [0.0, 1.0],
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
            prefix + "scan_x.site.path": json.dumps(["scan_x"]),
            prefix + "scan_x.site.parent_path": json.dumps([]),
            prefix + "scan_x.site.fragment_fqn": "ChildFragment",
            prefix + "scan_x.scan.parameters": json.dumps(
                {
                    "child_param": {
                        "path": "",
                        "is_scanned": False,
                        "param": {
                            "fqn": "demo.child.probe_frequency",
                            "description": "Probe Frequency",
                            "spec": {"unit": "MHz"},
                        },
                    }
                }
            ),
            prefix + "scan_x.scan.channels": json.dumps(
                {
                    "child_value": {
                        "path": "child/survived",
                        "description": "Survived",
                        "type": "float",
                    }
                }
            ),
            prefix + "scan_x.points.child_param": [0.2, 0.2, 0.3, 0.3],
            prefix + "scan_x.points.child_value": [1.0, 0.0, 0.0, 1.0],
            prefix + "scan_x.points.acquired_at_unix": [10.0, 11.0, 12.0, 13.0],
            prefix + "scan_x.segments.start_index": [0, 2],
            prefix + "scan_x.segments.parent_point_index": [0, 1],
            prefix + "scan_x.state.current_segment": 1,
            prefix + "scan_x.state.num_points": 4,
            prefix + "scan_x.state.completed": False,
        }

        viewer = RuntimePlotViewer(prefix)
        viewer.data_changed(values)

        child_path = ("scan_x",)
        viewer._on_x_key_changed(child_path, "__point_index__")
        viewer._on_y_key_changed(child_path, "child_value")
        viewer._on_point_selected((), 0)

        self.assertEqual(viewer._ui_state_for(child_path).x_key, "__point_index__")
        self.assertEqual(viewer._ui_state_for(child_path).y_key, "child_value")
        self.assertEqual(viewer._columns[1]._x_combo.currentData(), "__point_index__")
        self.assertEqual(viewer._columns[1]._y_combo.currentData(), "child_value")

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

    def test_site_column_widget_can_combine_repeated_x_points_with_std(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()
        values["ndscan.rid_0.site.root.points.param_0"] = [0.0, 0.0, 1.0, 1.0]
        values["ndscan.rid_0.site.root.points.channel_value"] = [10.0, 14.0, 20.0, 24.0]
        values["ndscan.rid_0.site.root.state.num_points"] = 4

        snapshot = snapshot_from_live_values("ndscan.rid_0.site.root.", values)
        root = snapshot.get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=1,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key="channel_value",
            selected_group_key=None,
            show_lines=True,
            selected_repeat_combine_mode=_REPEAT_COMBINE_STD,
        )

        self.assertEqual(len(widget._scatter.points()), 4)
        self.assertEqual(len(widget._summary_scatter.points()), 2)
        np.testing.assert_allclose(
            [point.pos().x() for point in widget._summary_scatter.points()],
            [0.0, 1.0],
        )
        np.testing.assert_allclose(
            [point.pos().y() for point in widget._summary_scatter.points()],
            [12.0, 22.0],
        )
        np.testing.assert_allclose(widget._error_bar_item.opts["x"], [0.0, 1.0])
        np.testing.assert_allclose(widget._error_bar_item.opts["y"], [12.0, 22.0])
        np.testing.assert_allclose(widget._error_bar_item.opts["top"], [2.0, 2.0])
        self.assertIn("repeated x: mean ± std", widget._status.text())
        self.assertLess(
            widget._scatter.points()[0].brush().color().alpha(),
            widget._summary_scatter.points()[0].brush().color().alpha(),
        )
        self.assertEqual(widget._highlight.points()[0].data(), 1)

    def test_site_column_widget_can_combine_repeated_x_points_with_sem(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()
        values["ndscan.rid_0.site.root.points.param_0"] = [0.0, 0.0, 1.0, 1.0]
        values["ndscan.rid_0.site.root.points.channel_value"] = [10.0, 14.0, 20.0, 24.0]
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
            selected_group_key=None,
            show_lines=True,
            selected_repeat_combine_mode=_REPEAT_COMBINE_SEM,
        )

        np.testing.assert_allclose(widget._error_bar_item.opts["x"], [0.0, 1.0])
        np.testing.assert_allclose(widget._error_bar_item.opts["y"], [12.0, 22.0])
        np.testing.assert_allclose(
            widget._error_bar_item.opts["top"],
            [np.sqrt(2.0), np.sqrt(2.0)],
        )
        self.assertIn("repeated x: mean ± sem", widget._status.text())

    def test_site_column_widget_can_combine_repeated_points_with_grouping(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        values = self._make_basic_root_values()
        values["ndscan.rid_0.site.root.points.param_0"] = [0.0, 0.0, 1.0, 1.0] * 2
        values["ndscan.rid_0.site.root.points.param_1"] = [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        values["ndscan.rid_0.site.root.points.channel_value"] = [10.0, 14.0, 20.0, 24.0, 30.0, 34.0, 40.0, 44.0]
        values["ndscan.rid_0.site.root.state.num_points"] = 8

        snapshot = snapshot_from_live_values("ndscan.rid_0.site.root.", values)
        root = snapshot.get_site(())
        widget = _SiteColumnWidget(
            site=root,
            child_site_options=[],
            selected_child_path=None,
            parent_point_index=None,
            selected_point_index=5,
            selected_plot_mode=_PLOT_MODE_1D,
            selected_x_key="param_0",
            selected_y_key="channel_value",
            selected_z_key="channel_value",
            selected_group_key="param_1",
            show_lines=True,
            selected_repeat_combine_mode=_REPEAT_COMBINE_STD,
        )

        self.assertEqual(len(widget._scatter.points()), 8)
        self.assertEqual(len(widget._summary_scatter.points()), 4)
        self.assertTrue(widget._legend.isVisible())
        self.assertIn("grouped by y (Y Axis / kHz)", widget._status.text())
        self.assertIn("repeated x: mean ± std", widget._status.text())

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

    def test_default_fit_backend_can_return_model_fit_artifact(self):
        backend = default_fit_backend()
        if not any(model.model_id == "straight_line" for model in backend.models()):
            self.skipTest("sensible-fitting backend not available")
        result = backend.fit(
            "straight_line",
            FitRequest(
                x=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float),
                y=np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float),
            ),
        )

        self.assertIsNotNone(result.artifact)
        self.assertEqual(result.artifact.model_id, "straight_line")
        self.assertIn("m=", result.summary())

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

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from artiq.language import units
from mock_environment import ExpFragmentCase

import examples._roi_condition_stats as roi_condition_stats
import examples.prepared_scan_results_api_demo as results_api_demo
import examples.prepared_scan_three_image_rearrangement as three_image_rearrangement
import examples.lab_offline_results_helpers as lab_results_helpers
import examples.plot_three_image_rearrangement_quick_analysis as three_image_quick
import examples.plot_three_image_rearrangement_snapshot as three_image_plot
from ndscan.define.fragment import ExpFragment
from ndscan.define.parameters import FloatParam
from ndscan.define.result_channels import FloatChannel
from ndscan.results.scan_site_reader import read_scan_site_snapshot
from ndscan.results.series import series_slices_along_axis
from ndscan.runtime.api import PreparedScan, prepare_child_scan
from ndscan.scan.mapping import ParameterMapping, ScanVariable
from ndscan.scan.point_policy import BasePoint, ExplicitPointPolicy
from ndscan.scan.request import ExecutionPolicy, ScanRequest
from ndscan.submission.scan_submission_schema import compile_scan_submission_schema


def _execute_and_inspect(scan):
    scan.execute()
    return scan.inspect()


class PlainAddOneFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("value", FloatParam, "value", 0.0)
        self.setattr_result("result", FloatChannel)

    def run_once(self):
        self.result.push(self.value.get() + 1.0)


class MetadataPolicyFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("value", FloatParam, "value", 0.0)
        self.setattr_result("result", FloatChannel)

    def run_once(self):
        self.result.push(self.value.get())


class MetadataRecordingPointPolicy(ExplicitPointPolicy):
    def __init__(self, labels):
        super().__init__(1, [(float(index),) for index in range(len(labels))])
        self._labels = tuple(labels)

    def next_batch(self, max_points: int):
        batch = super().next_batch(max_points)
        return [
            BasePoint(
                index=point.index,
                axis_values=point.axis_values,
                metadata={"decision_source": self._labels[point.index]},
            )
            for point in batch
        ]


class PhysicalDriveFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("drive", FloatParam, "drive", 0.0)
        self.setattr_result("result", FloatChannel)

    def run_once(self):
        self.result.push(2.0 * self.drive.get())


class NestedChildScanParent(ExpFragment):
    def build_fragment(self):
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_fragment("child", PlainAddOneFragment, detached=True)
        self.child_scan = prepare_child_scan(self, self.child, name="child_scan")
        self.setattr_result("child_total", FloatChannel)

    def run_once(self):
        self.child_scan.configure(
            ScanRequest.explicit(
                [self.child.value],
                [[self.outer.get()], [self.outer.get() + 1.0]],
            )
        )
        self.child_scan.execute()
        child_result = self.child_scan.inspect()
        self.child_total.push(sum(child_result.values[self.child.result]))


class DeepNestedGrandchild(ExpFragment):
    def build_fragment(self):
        self.setattr_param("value", FloatParam, "value", 0.0)
        self.setattr_result("result", FloatChannel)

    def run_once(self):
        self.result.push(self.value.get())


class DeepNestedChild(ExpFragment):
    def build_fragment(self):
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_fragment("grandchild", DeepNestedGrandchild, detached=True)
        self.grandchild_scan = prepare_child_scan(
            self, self.grandchild, name="grandchild_scan"
        )
        self.setattr_result("child_total", FloatChannel)

    def run_once(self):
        self.grandchild_scan.configure(
            ScanRequest.explicit(
                [self.grandchild.value],
                [[self.outer.get()], [self.outer.get() + 1.0]],
            )
        )
        self.grandchild_scan.execute()
        grandchild_result = self.grandchild_scan.inspect()
        self.child_total.push(sum(grandchild_result.values[self.grandchild.result]))


class DeepNestedParent(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("child", DeepNestedChild, detached=True)
        self.child_scan = prepare_child_scan(self, self.child, name="child_scan")
        self.setattr_result("root_total", FloatChannel)

    def run_once(self):
        self.child_scan.configure(
            ScanRequest.explicit(
                [self.child.outer],
                [[10.0], [20.0]],
            )
        )
        self.child_scan.execute()
        child_result = self.child_scan.inspect()
        self.root_total.push(sum(child_result.values[self.child.child_total]))


def _load_plot_prepared_scan_snapshot_helpers():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "plot_prepared_scan_snapshot.py"
    spec = importlib.util.spec_from_file_location("_plot_prepared_scan_snapshot", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScanSiteReaderCase(ExpFragmentCase):
    def _write_snapshot(self, owner, path, *, preview_complete=False):
        dataset_mgr = owner._HasEnvironment__dataset_mgr
        with h5py.File(path, "w") as h5_file:
            dataset_mgr.write_hdf5(h5_file)
            h5_file["preview_complete"] = preview_complete
            h5_file["preview_time"] = 1234.0

    def test_reads_root_site_from_hdf5_snapshot(self):
        fragment = self.create(PlainAddOneFragment)
        session = PreparedScan(
            fragment,
            fragment,
            ScanRequest.cartesian(
                [(fragment.value, [0.0, 1.0, 2.0])],
                execution_policy=ExecutionPolicy(max_points_per_batch=2),
            ),
        )
        session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name, preview_complete=False)
            snapshot = read_scan_site_snapshot(tmp.name)
            lab_run = lab_results_helpers.LabNdscanRun.open(tmp.name)

        site = snapshot.get_site(())
        lab_site = lab_run.site(())
        self.assertEqual(snapshot.top_level_metadata["preview_complete"], False)
        self.assertEqual(site.path, ())
        self.assertEqual(list(site.raw_points["param_0"]), [0.0, 1.0, 2.0])
        self.assertEqual(list(site.raw_points["channel_0"]), [1.0, 2.0, 3.0])
        self.assertEqual(site.choose_default_x_path(), "value")
        self.assertEqual(site.axis_paths(), ["value"])
        self.assertEqual(site.axis_storage_keys(), ["param_0"])
        self.assertEqual(
            [
                (batch.start_index, batch.stop_index, batch.length)
                for batch in site.batches()
            ],
            [(0, 2, 2), (2, 3, 1)],
        )
        self.assertEqual(
            [batch.start_unix_time is not None for batch in site.batches()],
            [True, True],
        )
        self.assertEqual(
            site.axes,
            [
                {
                    "index": 0,
                    "axis_key": "axis_0",
                    "storage_key": "param_0",
                    "kind": "parameter",
                    "path": "value",
                    "description": "value",
                    "unit": None,
                    "scale": 1.0,
                    "type": "float",
                }
            ],
        )
        self.assertEqual(site.available_parameter_paths(), ["value"])
        self.assertEqual(site.available_channel_paths(), ["result"])
        self.assertEqual(
            site.available_series_paths(),
            ["value", "result", "acquired_at_unix", "point_index"],
        )
        self.assertEqual(site.require_channel_storage_key("result"), "channel_0")
        self.assertEqual(list(site.series("result")), [1.0, 2.0, 3.0])
        self.assertEqual(site.require_series_storage_key("value"), "param_0")
        self.assertEqual(
            site.require_series_storage_key("acquired_at_unix"),
            "acquired_at_unix",
        )
        self.assertEqual(list(site.series("value")), [0.0, 1.0, 2.0])
        self.assertEqual(list(site.series("point_index")), [0, 1, 2])
        self.assertEqual(list(site.series("value")), [0.0, 1.0, 2.0])
        self.assertEqual(list(site.series("result")), [1.0, 2.0, 3.0])
        self.assertEqual(list(site.series("point_index")), [0, 1, 2])
        self.assertEqual(np.asarray(site.series("acquired_at_unix")).shape, (3,))
        self.assertEqual(
            [
                (item.path, item.kind, item.shape, str(item.dtype))
                for item in site.describe_series()
            ],
            [
                ("value", "parameter", (3,), "float64"),
                ("result", "channel", (3,), "float64"),
                ("acquired_at_unix", "runtime", (3,), "float64"),
                ("point_index", "runtime", (3,), "int64"),
            ],
        )
        descriptions = {
            item.path: item
            for item in site.describe_series()
        }
        self.assertEqual(descriptions["value"].label, "value")
        self.assertIsNone(descriptions["value"].unit)
        self.assertEqual(descriptions["value"].scale, 1)
        self.assertEqual(descriptions["value"].is_numeric, True)
        self.assertEqual(descriptions["result"].label, "result")
        self.assertEqual(list(lab_site.series("value")), [0.0, 1.0, 2.0])
        self.assertEqual(list(lab_site.series("result")), [1.0, 2.0, 3.0])
        plot_choices = site.describe_plot_choices()
        self.assertEqual(
            [item.path for item in plot_choices.x.choices],
            ["value", "result", "acquired_at_unix", "point_index"],
        )
        self.assertEqual(plot_choices.x.default_path, "value")
        self.assertEqual(plot_choices.y.default_path, "result")
        self.assertEqual(plot_choices.x.default.label, "value")
        self.assertEqual(plot_choices.y.default.label, "result")

    def test_raises_clear_error_for_missing_channel_path(self):
        fragment = self.create(PlainAddOneFragment)
        session = PreparedScan(
            fragment,
            fragment,
            ScanRequest.single(),
        )
        session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        site = snapshot.get_site(())
        with self.assertRaises(KeyError) as ctx:
            site.require_channel_storage_key("missing")
        self.assertIn("available paths", str(ctx.exception))

    def test_reads_pseudoparams_and_derived_params(self):
        fragment = self.create(PhysicalDriveFragment)
        logical_drive = ScanVariable("laser_frequency", description="Logical drive")
        request = ScanRequest.cartesian([(logical_drive, [0.0, 1.0, 2.0])]).with_parameter_mappings(
            [
                ParameterMapping.single_target(
                    fragment.drive,
                    [logical_drive],
                    lambda values: values[logical_drive] + 0.5,
                    description="Offset physical drive from logical axis",
                )
            ]
        )
        session = PreparedScan(fragment, fragment, request)
        session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        site = snapshot.get_site(())
        self.assertEqual(list(site.raw_points["pseudoparam_0"]), [0.0, 1.0, 2.0])
        self.assertEqual(list(site.raw_points["param_0"]), [0.5, 1.5, 2.5])
        self.assertEqual(site.choose_default_x_path(), "laser_frequency")
        self.assertEqual(site.axis_paths(), ["laser_frequency"])
        self.assertEqual(site.axis_storage_keys(), ["pseudoparam_0"])

    def test_reads_fixed_pseudoparams_from_schema_compiled_scan(self):
        fragment = self.create(PhysicalDriveFragment)
        request, overrides = compile_scan_submission_schema(
            fragment,
            {
                "version": 1,
                "mode": {"type": "grid"},
                "entries": [
                    {
                        "id": "logical_drive",
                        "kind": "pseudoparam",
                        "mode": {
                            "type": "scan",
                            "generator": {
                                "type": "list",
                                "range": {
                                    "values": [0.0, 1.0, 2.0],
                                    "randomise_order": False,
                                },
                            },
                        },
                    },
                    {
                        "id": "offset",
                        "kind": "pseudoparam",
                        "mode": {
                            "type": "fixed",
                            "value": 0.5,
                        },
                    },
                    {
                        "id": "drive",
                        "kind": "param",
                        "target": {"fqn": fragment.drive.parameter.fqn, "path": "*"},
                        "mode": {
                            "type": "rebind",
                            "expr": "logical_drive + offset",
                        },
                    },
                ],
            },
        )
        session = PreparedScan(fragment, fragment, request, overrides=overrides)
        session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        site = snapshot.get_site(())
        self.assertEqual(
            site.fixed_pseudoparams,
            {
                "offset": {
                    "variable": {
                        "name": "offset",
                        "description": "",
                        "type": "float",
                        "spec": {},
                    },
                    "value": 0.5,
                }
            },
        )

    def test_reads_nested_child_sites(self):
        parent = self.create(NestedChildScanParent)
        session = PreparedScan(
            parent,
            parent,
            ScanRequest.explicit([parent.outer], [[10.0]]),
        )
        session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(parent, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        self.assertIn((), snapshot.sites)
        self.assertIn(("child_scan",), snapshot.sites)

        root_site = snapshot.get_site(())
        self.assertFalse(
            any(key.startswith("subscans.") for key in root_site.metadata)
        )
        child_site = snapshot.get_site(("child_scan",))
        self.assertEqual(child_site.parent_path, ())
        self.assertTrue(child_site.segmented)
        self.assertEqual(list(child_site.raw_points["param_0"]), [10.0, 11.0])
        self.assertEqual(list(child_site.raw_points["channel_0"]), [11.0, 12.0])

    def test_reads_string_point_metadata_streams(self):
        fragment = self.create(MetadataPolicyFragment)
        request = ScanRequest(
            axes=(fragment.value,),
            point_policy=MetadataRecordingPointPolicy(["seed", "bo", "explore"]),
        )
        session = PreparedScan(fragment, fragment, request)
        session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        site = snapshot.get_site(())
        self.assertEqual(
            site.raw_points["metadata.decision_source"],
            ["seed", "bo", "explore"],
        )

    def test_exposes_child_site_segments_for_parent_points(self):
        parent = self.create(NestedChildScanParent)
        session = PreparedScan(
            parent,
            parent,
            ScanRequest.explicit([parent.outer], [[10.0], [20.0]]),
        )
        session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(parent, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        self.assertEqual([site.path for site in snapshot.child_sites(())], [("child_scan",)])

        child_site = snapshot.get_site(("child_scan",))
        segments = child_site.segments()
        self.assertEqual(
            [(segment.start_index, segment.stop_index, segment.parent_point_index) for segment in segments],
            [(0, 2, 0), (2, 4, 1)],
        )

        second_parent_segments = child_site.segments_for_parent_point(1)
        self.assertEqual(len(second_parent_segments), 1)
        second_parent_data = child_site.slice_raw_points(
            second_parent_segments[0].start_index,
            second_parent_segments[0].stop_index,
        )
        self.assertEqual(second_parent_data["param_0"], [20.0, 21.0])
        self.assertEqual(second_parent_data["channel_0"], [21.0, 22.0])

    def test_plot_helper_builds_recursive_detail_panels(self):
        root = self.create(DeepNestedParent)
        session = PreparedScan(root, root, ScanRequest.single())
        session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(root, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        plot_helpers = _load_plot_prepared_scan_snapshot_helpers()
        selection_chain = [((), 0)]
        child_panels = plot_helpers._build_visible_site_panels(snapshot, selection_chain)
        self.assertEqual([panel.site.path for panel in child_panels], [("child_scan",)])
        self.assertEqual(child_panels[0].raw_points["param_0"], [10.0, 20.0])

        selection_chain = plot_helpers._update_selection_chain(
            snapshot,
            (),
            selection_chain,
            ("child_scan",),
            1,
        )
        self.assertEqual(selection_chain, [((), 0), (("child_scan",), 1)])

        recursive_panels = plot_helpers._build_visible_site_panels(snapshot, selection_chain)
        self.assertEqual(
            [panel.site.path for panel in recursive_panels],
            [("child_scan",), ("child_scan", "grandchild_scan")],
        )
        self.assertEqual(recursive_panels[1].raw_points["param_0"], [20.0, 21.0])

    def test_reads_imaging_blob_from_three_image_snapshot(self):
        with (
            patch.object(three_image_rearrangement, "POINT_DELAY_S", 0.0),
            patch.object(three_image_rearrangement, "DEFAULT_NUM_SHOTS", 6),
        ):
            fragment = self.create(
                three_image_rearrangement.ThreeImageRearrangementStatisticsFragment
            )
            session = PreparedScan(
                fragment,
                fragment,
                three_image_rearrangement.build_three_image_rearrangement_statistics_request(
                    fragment
                ),
            )
            session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        site = snapshot.get_site(("repeat_scan",))
        blob = site.require_metadata_blob(three_image_rearrangement.IMAGING_READOUT_BLOB_NAME)

        self.assertEqual(
            site.available_metadata_blob_names(),
            [
                "analysis_note",
                three_image_rearrangement.IMAGING_READOUT_BLOB_NAME,
            ],
        )
        self.assertEqual(
            site.metadata_blobs()[three_image_rearrangement.IMAGING_READOUT_BLOB_NAME],
            blob,
        )
        self.assertEqual(
            site.metadata_blobs()["analysis_note"],
            "threshold three-image ROI counts into occupancy probabilities",
        )
        self.assertEqual(blob["namespace"], three_image_rearrangement.IMAGING_READOUT_BLOB_NAME)
        self.assertEqual(blob["version"], three_image_rearrangement.IMAGING_READOUT_BLOB_VERSION)
        self.assertEqual(
            blob["images"]["image0"]["image_channel"],
            "shot/image0",
        )
        self.assertEqual(
            blob["images"]["image2"]["counts_channel"],
            "shot/counts_image2",
        )
        self.assertEqual(
            blob["occupancy_rule"]["threshold_parameter_path"],
            "threshold_counts",
        )
        self.assertEqual(
            len(blob["images"]["image1"]["rois"]),
            three_image_rearrangement.NUM_GROUPS,
        )

    def test_three_image_plot_helper_builds_average_images_from_blob(self):
        with (
            patch.object(three_image_rearrangement, "POINT_DELAY_S", 0.0),
            patch.object(three_image_rearrangement, "DEFAULT_NUM_SHOTS", 5),
        ):
            fragment = self.create(
                three_image_rearrangement.ThreeImageRearrangementStatisticsFragment
            )
            session = PreparedScan(
                fragment,
                fragment,
                three_image_rearrangement.build_three_image_rearrangement_statistics_request(
                    fragment
                ),
            )
            session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        site = snapshot.get_site(("repeat_scan",))
        readout = three_image_plot.ThreeImageReadout.from_site(
            site,
            blob_name=three_image_rearrangement.IMAGING_READOUT_BLOB_NAME,
        )
        payloads = readout.average_image_payloads()

        self.assertEqual(len(payloads), three_image_rearrangement.NUM_IMAGES)
        for image_index, payload in enumerate(payloads):
            self.assertEqual(
                np.asarray(payload["average_image"]).shape,
                three_image_rearrangement.IMAGE_SHAPES[image_index],
            )
            self.assertEqual(
                len(payload["rois"]),
                three_image_rearrangement.NUM_GROUPS,
            )

    def test_three_image_series_descriptions_include_display_metadata(self):
        with (
            patch.object(three_image_rearrangement, "POINT_DELAY_S", 0.0),
            patch.object(three_image_rearrangement, "DEFAULT_NUM_SHOTS", 5),
        ):
            fragment = self.create(
                three_image_rearrangement.ThreeImageRearrangementStatisticsFragment
            )
            request = ScanRequest.cartesian(
                [
                    (
                        fragment.shot.probe_frequency,
                        [9.7, 10.0, 10.3],
                    )
                ],
                metadata={"demo_name": "three_image_series_display_metadata"},
            )
            session = PreparedScan(fragment, fragment, request)
            session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        site = snapshot.get_site(())
        descriptions = {
            item.path: item
            for item in site.describe_series()
        }
        probe_frequency = descriptions["shot/probe_frequency"]
        self.assertEqual(probe_frequency.description, "Probe frequency")
        self.assertEqual(probe_frequency.unit, "MHz")
        self.assertEqual(probe_frequency.scale, units.MHz)
        self.assertEqual(
            probe_frequency.label,
            "shot/probe_frequency (Probe frequency / MHz)",
        )
        self.assertEqual(probe_frequency.is_numeric, True)

        plot_choices = site.describe_plot_choices()
        self.assertEqual(plot_choices.x.default_path, "shot/probe_frequency")
        self.assertEqual(
            plot_choices.x.default.label,
            "shot/probe_frequency (Probe frequency / MHz)",
        )
        self.assertIsNotNone(plot_choices.y.default_path)
        self.assertNotEqual(
            plot_choices.y.default_path,
            plot_choices.x.default_path,
        )

    def test_three_image_plot_helper_direct_series_access_supports_simple_plotting(self):
        with (
            patch.object(three_image_rearrangement, "POINT_DELAY_S", 0.0),
            patch.object(three_image_rearrangement, "DEFAULT_NUM_SHOTS", 5),
        ):
            fragment = self.create(
                three_image_rearrangement.ThreeImageRearrangementStatisticsFragment
            )
            request = ScanRequest.cartesian(
                [
                    (
                        fragment.shot.probe_frequency,
                        [9.7, 10.0, 10.3],
                    )
                ],
                metadata={"demo_name": "three_image_plot_helper_payload"},
            )
            session = PreparedScan(fragment, fragment, request)
            session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        site = snapshot.get_site(())
        descriptions = {
            item.path: item
            for item in site.describe_series()
        }
        self.assertEqual(
            descriptions["bright_probability_image2_given_bright_image1_by_trap"].point_shape,
            (
                three_image_rearrangement.NUM_GROUPS,
                three_image_rearrangement.COMMON_IMAGE12_ROIS,
            ),
        )
        self.assertEqual(
            descriptions["bright_probability_image2_given_bright_image1_by_trap"].dim_names,
            ("group", "roi"),
        )
        self.assertEqual(
            np.asarray(
                site.series("bright_probability_image2_given_bright_image1_by_trap")
            ).shape,
            (
                3,
                three_image_rearrangement.NUM_GROUPS,
                three_image_rearrangement.COMMON_IMAGE12_ROIS,
            ),
        )
        x_values = np.asarray(site.series("shot/probe_frequency"), dtype=float)
        y_values = np.asarray(
            site.series("bright_pair_probability_image2_given_pair_image1"),
            dtype=float,
        )
        y_errors = np.asarray(
            site.series("bright_pair_probability_error_image2_given_pair_image1"),
            dtype=float,
        )
        order = np.argsort(x_values)
        np.testing.assert_allclose(x_values[order], np.array([9.7, 10.0, 10.3]))
        self.assertEqual(y_values.shape, (3,))
        self.assertEqual(y_errors.shape, (3,))
        group_slices = series_slices_along_axis(
            site,
            "bright_pair_probability_image2_given_pair_image1_by_group",
            axis=0,
            indices=[0, 1],
        )
        self.assertEqual(set(group_slices), {0, 1})
        self.assertEqual(np.asarray(group_slices[0]).shape, (3,))

    def test_three_image_plot_helper_builds_saved_vs_recomputed_payload(self):
        with (
            patch.object(three_image_rearrangement, "POINT_DELAY_S", 0.0),
            patch.object(three_image_rearrangement, "DEFAULT_NUM_SHOTS", 7),
        ):
            fragment = self.create(
                three_image_rearrangement.ThreeImageRearrangementStatisticsFragment
            )
            request = ScanRequest.cartesian(
                [
                    (
                        fragment.shot.probe_frequency,
                        [9.8, 10.0, 10.2],
                    )
                ],
                metadata={"demo_name": "three_image_plot_helper_saved_vs_recomputed"},
            )
            session = PreparedScan(fragment, fragment, request)
            session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        root_site = snapshot.get_site(())
        repeat_site = snapshot.get_site(("repeat_scan",))
        readout = three_image_plot.ThreeImageReadout.from_site(
            repeat_site,
            blob_name=three_image_rearrangement.IMAGING_READOUT_BLOB_NAME,
        )

        comparison_payload = three_image_plot.build_saved_vs_recomputed_probability_payload(
            root_site,
            readout,
            x="shot/probe_frequency",
            saved_y="bright_pair_probability_image2_given_pair_image1",
            given_syntax="1[0,1]",
            event_syntax="2[0,1]",
        )
        np.testing.assert_allclose(
            comparison_payload["saved_y_values"],
            comparison_payload["recomputed_y_values"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            comparison_payload["difference"],
            np.zeros(3, dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(
            comparison_payload["threshold"],
            three_image_rearrangement.DEFAULT_THRESHOLD_COUNTS,
        )

        histogram_payloads = readout.threshold_histogram_payloads()
        self.assertEqual(len(histogram_payloads), three_image_rearrangement.NUM_IMAGES)
        for image_index, payload in enumerate(histogram_payloads):
            self.assertEqual(payload["image_index"], image_index)
            self.assertEqual(payload["num_groups"], three_image_rearrangement.NUM_GROUPS)
            self.assertEqual(
                payload["num_rois"],
                three_image_rearrangement.NUM_ROIS_BY_IMAGE[image_index],
            )
            self.assertEqual(
                np.asarray(payload["bright_fraction_by_roi_group"]).shape,
                (
                    three_image_rearrangement.NUM_ROIS_BY_IMAGE[image_index],
                    three_image_rearrangement.NUM_GROUPS,
                ),
            )
            first_samples = payload["samples_by_roi_group"][0][0]
            self.assertEqual(len(first_samples), payload["num_points"])

        ad_hoc_payload = three_image_quick.build_adhoc_conditional_payload(
            lab_results_helpers.LabNdscanSite(root_site),
            readout,
            x="shot/probe_frequency",
            given_syntax=None,
            event_syntax="2[0] | 1[0]",
        )
        self.assertEqual(np.asarray(ad_hoc_payload["pooled_probability"]).shape, (3,))
        self.assertEqual(
            np.asarray(ad_hoc_payload["probability_by_group"]).shape,
            (3, three_image_rearrangement.NUM_GROUPS),
        )
        self.assertEqual(
            ad_hoc_payload["event_syntax"],
            "2[0] | 1[0]",
        )

    def test_results_api_demo_snapshot_exposes_generic_offline_surface(self):
        demo_repeat_count = 6
        with (
            patch.object(
                results_api_demo,
                "ROOT_LOGICAL_FREQUENCY_POINTS_MHZ",
                [-0.6, 0.0, 0.6],
            ),
            patch.object(results_api_demo, "DEFAULT_REPEAT_COUNT", demo_repeat_count),
            patch.object(results_api_demo, "ROOT_MAX_POINTS_PER_BATCH", 2),
            patch.object(results_api_demo, "REPEAT_MAX_POINTS_PER_BATCH", 3),
        ):
            fragment = self.create(results_api_demo.ResultsApiDemoFragment)
            session = PreparedScan(
                fragment,
                fragment,
                results_api_demo.build_results_api_demo_request(fragment),
            )
            session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        root_site = snapshot.get_site(())
        repeat_site = snapshot.get_site(("repeat_scan",))

        self.assertEqual([site.path for site in snapshot.child_sites(())], [("repeat_scan",)])
        self.assertEqual(root_site.available_pseudoparam_paths(), [])
        self.assertIn("logical_frequency", root_site.available_parameter_paths())
        self.assertIn("shot/drive_frequency", root_site.available_parameter_paths())
        self.assertIn("mean_signal", root_site.available_channel_paths())
        self.assertEqual(root_site.choose_default_x_path(), "logical_frequency")

        root_descriptions = {item.path: item for item in root_site.describe_series()}
        self.assertEqual(
            root_descriptions["mean_trace"].point_shape,
            (results_api_demo.TRACE_LENGTH,),
        )
        self.assertEqual(
            root_descriptions["mean_trace"].dim_names,
            ("sample",),
        )

        self.assertIn("demo_config", root_site.available_metadata_blob_names())
        self.assertEqual(
            root_site.require_metadata_blob("demo_config")["trace_length"],
            results_api_demo.TRACE_LENGTH,
        )
        self.assertIn("scan.parameter_mappings", root_site.metadata)
        self.assertIn("scan.point_policy", root_site.metadata)
        self.assertIn("carrier_frequency", root_site.fixed_pseudoparams)
        logical_frequency_values = np.asarray(
            root_site.series("logical_frequency"), dtype=float
        )
        drive_frequency_values = np.asarray(
            root_site.series("shot/drive_frequency"), dtype=float
        )
        np.testing.assert_allclose(
            drive_frequency_values,
            logical_frequency_values
            + results_api_demo.CARRIER_FREQUENCY_MHZ * units.MHz,
        )
        mean_signal_values = np.asarray(root_site.series("mean_signal"), dtype=float)
        self.assertGreater(mean_signal_values[-1], mean_signal_values[0])

        self.assertEqual(set(root_site.analysis_outputs), {"fit_slope", "fit_intercept"})
        self.assertIn("signal_line_fit", root_site.analysis_artifacts)
        self.assertIn("running_signal_fit", root_site.online_analysis_results)
        self.assertIn("running_signal_fit", root_site.online_analysis_artifacts)

        repeat_plot_choices = repeat_site.describe_plot_choices()
        self.assertIn(
            repeat_plot_choices.x.default_path,
            {"shot/drive_frequency", "acquired_at_unix", "point_index"},
        )
        self.assertIn("repeat_config", repeat_site.available_metadata_blob_names())
        self.assertIn("repeat_stats", repeat_site.online_analysis_results)
        self.assertIn("repeat_stats", repeat_site.online_analysis_artifacts)

        segments = repeat_site.segments_for_parent_point(0)
        self.assertEqual(len(segments), 1)
        segment_analysis = repeat_site.analysis_for_segment(segments[0].index)
        self.assertIsNotNone(segment_analysis)
        self.assertIn("mean_signal", segment_analysis.outputs)
        self.assertIn("repeat_statistics_summary", segment_analysis.artifacts)

        segment_raw_points = repeat_site.slice_raw_points(
            segments[0].start_index,
            segments[0].stop_index,
        )
        shot_trace_key = repeat_site.require_channel_storage_key("shot/shot_trace")
        raw_trace_values = np.asarray(segment_raw_points[shot_trace_key], dtype=float)
        self.assertEqual(
            raw_trace_values.shape,
            (demo_repeat_count, results_api_demo.TRACE_LENGTH),
        )

    def test_three_image_saved_statistic_can_be_recomputed_offline_from_raw_counts(self):
        with (
            patch.object(three_image_rearrangement, "POINT_DELAY_S", 0.0),
            patch.object(three_image_rearrangement, "DEFAULT_NUM_SHOTS", 7),
        ):
            fragment = self.create(
                three_image_rearrangement.ThreeImageRearrangementStatisticsFragment
            )
            request = ScanRequest.cartesian(
                [
                    (
                        fragment.shot.probe_frequency,
                        [9.8, 10.0, 10.2],
                    )
                ],
                metadata={"demo_name": "three_image_offline_recompute"},
            )
            session = PreparedScan(fragment, fragment, request)
            session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_scan_site_snapshot(tmp.name)

        root_site = snapshot.get_site(())
        repeat_site = snapshot.get_site(("repeat_scan",))
        blob = repeat_site.require_metadata_blob(
            three_image_rearrangement.IMAGING_READOUT_BLOB_NAME
        )
        threshold_fqn = blob["occupancy_rule"]["threshold_parameter_fqn"]
        threshold = next(
            entry["value"]
            for entry in repeat_site.fixed_parameters.values()
            if entry["param"]["fqn"] == threshold_fqn
        )

        pair_image1 = roi_condition_stats.parse_condition_syntax("1[0,1]")
        bright_pair_image2 = roi_condition_stats.parse_condition_syntax("2[0,1]")

        saved = np.asarray(
            root_site.series(
                "bright_pair_probability_image2_given_pair_image1"
            ),
            dtype=float,
        )
        recomputed = []
        for point_index in range(root_site.num_points):
            segments = repeat_site.segments_for_parent_point(point_index)
            self.assertEqual(len(segments), 1)
            segment = segments[0]
            segment_data = repeat_site.slice_raw_points(segment.start_index, segment.stop_index)
            counts_by_image = [
                np.asarray(segment_data[repeat_site.require_channel_storage_key(path)])
                for path in (
                    "shot/counts_image0",
                    "shot/counts_image1",
                    "shot/counts_image2",
                )
            ]
            occupancy_stack = roi_condition_stats.counts_to_occupancy_stack(
                counts_by_image,
                threshold=threshold,
            )
            bright_pair = roi_condition_stats.conditional_binomial(
                occupancy_stack,
                given=pair_image1,
                event=bright_pair_image2,
            )
            recomputed.append(bright_pair.pooled_probability)

        np.testing.assert_allclose(saved, np.asarray(recomputed), rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

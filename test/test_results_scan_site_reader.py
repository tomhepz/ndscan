import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import h5py

from mock_environment import ExpFragmentCase

from ndscan.experiment import (
    BasePoint,
    compile_host_scan_schema,
    ExpFragment,
    ExplicitPointPolicy,
    FloatChannel,
    FloatParam,
    ParameterMapping,
    prepare_child_scan,
    ScanRequest,
    ScanVariable,
    PreparedScan,
)
from ndscan.results.scan_site_reader import read_host_runtime_snapshot


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


def _load_plot_host_runtime_snapshot_helpers():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "plot_host_runtime_snapshot.py"
    spec = importlib.util.spec_from_file_location("_plot_host_runtime_snapshot", module_path)
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
            ScanRequest.cartesian([(fragment.value, [0.0, 1.0, 2.0])]),
        )
        session.execute()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name, preview_complete=False)
            snapshot = read_host_runtime_snapshot(tmp.name)

        site = snapshot.get_site(())
        self.assertEqual(snapshot.top_level_metadata["preview_complete"], False)
        self.assertEqual(site.path, ())
        self.assertEqual(list(site.point_data["param_0"]), [0.0, 1.0, 2.0])
        self.assertEqual(list(site.point_data["channel_0"]), [1.0, 2.0, 3.0])
        self.assertEqual(site.choose_default_x_key(), ("param", "param_0"))

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
            snapshot = read_host_runtime_snapshot(tmp.name)

        site = snapshot.get_site(())
        self.assertEqual(list(site.point_data["pseudoparam_0"]), [0.0, 1.0, 2.0])
        self.assertEqual(list(site.point_data["param_0"]), [0.5, 1.5, 2.5])
        self.assertEqual(site.choose_default_x_key(), ("pseudoparam", "pseudoparam_0"))

    def test_reads_fixed_pseudoparams_from_schema_compiled_scan(self):
        fragment = self.create(PhysicalDriveFragment)
        request, overrides = compile_host_scan_schema(
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
            snapshot = read_host_runtime_snapshot(tmp.name)

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
            snapshot = read_host_runtime_snapshot(tmp.name)

        self.assertIn((), snapshot.sites)
        self.assertIn(("child_scan",), snapshot.sites)

        child_site = snapshot.get_site(("child_scan",))
        self.assertEqual(child_site.parent_path, ())
        self.assertTrue(child_site.segmented)
        self.assertEqual(list(child_site.point_data["param_0"]), [10.0, 11.0])
        self.assertEqual(list(child_site.point_data["channel_0"]), [11.0, 12.0])

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
            snapshot = read_host_runtime_snapshot(tmp.name)

        site = snapshot.get_site(())
        self.assertEqual(
            site.point_data["metadata.decision_source"],
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
            snapshot = read_host_runtime_snapshot(tmp.name)

        self.assertEqual([site.path for site in snapshot.child_sites(())], [("child_scan",)])

        child_site = snapshot.get_site(("child_scan",))
        segments = child_site.segments()
        self.assertEqual(
            [(segment.start_index, segment.stop_index, segment.parent_point_index) for segment in segments],
            [(0, 2, 0), (2, 4, 1)],
        )

        second_parent_segments = child_site.segments_for_parent_point(1)
        self.assertEqual(len(second_parent_segments), 1)
        second_parent_data = child_site.slice_point_data(
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
            snapshot = read_host_runtime_snapshot(tmp.name)

        plot_helpers = _load_plot_host_runtime_snapshot_helpers()
        selection_chain = [((), 0)]
        child_panels = plot_helpers._build_visible_site_panels(snapshot, selection_chain)
        self.assertEqual([panel.site.path for panel in child_panels], [("child_scan",)])
        self.assertEqual(child_panels[0].point_data["param_0"], [10.0, 20.0])

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
        self.assertEqual(recursive_panels[1].point_data["param_0"], [20.0, 21.0])


if __name__ == "__main__":
    unittest.main()

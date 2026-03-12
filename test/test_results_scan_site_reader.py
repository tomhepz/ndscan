import tempfile
import unittest

import h5py

from mock_environment import ExpFragmentCase

from ndscan.experiment import (
    ExpFragment,
    FloatChannel,
    FloatParam,
    ParameterMapping,
    ScanRequest,
    ScanVariable,
    HostScanSession,
    run_subscan,
)
from ndscan.results.scan_site_reader import read_host_runtime_snapshot


class PlainAddOneFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("value", FloatParam, "value", 0.0)
        self.setattr_result("result", FloatChannel)

    def run_once(self):
        self.result.push(self.value.get() + 1.0)


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
        self.setattr_result("child_total", FloatChannel)

    def run_once(self):
        child_result = run_subscan(
            self,
            self.child,
            ScanRequest.explicit(
                [self.child.value],
                [[self.outer.get()], [self.outer.get() + 1.0]],
            ),
            name="child_scan",
        )
        self.child_total.push(sum(child_result.values[self.child.result]))


class ScanSiteReaderCase(ExpFragmentCase):
    def _write_snapshot(self, owner, path, *, preview_complete=False):
        dataset_mgr = owner._HasEnvironment__dataset_mgr
        with h5py.File(path, "w") as h5_file:
            dataset_mgr.write_hdf5(h5_file)
            h5_file["preview_complete"] = preview_complete
            h5_file["preview_time"] = 1234.0

    def test_reads_root_site_from_hdf5_snapshot(self):
        fragment = self.create(PlainAddOneFragment)
        session = HostScanSession(
            fragment,
            fragment,
            ScanRequest.cartesian([(fragment.value, [0.0, 1.0, 2.0])]),
        )
        session.run()

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
        session = HostScanSession(fragment, fragment, request)
        session.run()

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self._write_snapshot(fragment, tmp.name)
            snapshot = read_host_runtime_snapshot(tmp.name)

        site = snapshot.get_site(())
        self.assertEqual(list(site.point_data["pseudoparam_0"]), [0.0, 1.0, 2.0])
        self.assertEqual(list(site.point_data["param_0"]), [0.5, 1.5, 2.5])
        self.assertEqual(site.choose_default_x_key(), ("pseudoparam", "pseudoparam_0"))

    def test_reads_nested_child_sites(self):
        parent = self.create(NestedChildScanParent)
        session = HostScanSession(
            parent,
            parent,
            ScanRequest.explicit([parent.outer], [[10.0]]),
        )
        session.run()

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


if __name__ == "__main__":
    unittest.main()

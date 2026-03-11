"""Tests for the new host-only runtime."""

import json
import unittest

import numpy as np
from fixtures import AddOneFragment
from mock_environment import HasEnvironmentCase

from ndscan.experiment import (
    CartesianPointSource,
    ExpFragment,
    ExplicitPointSource,
    FloatChannel,
    FloatParam,
    HostScanSession,
    ScanRequest,
    ScanSite,
    ZipPointSource,
    kernel,
    make_child_scan_site,
    make_fragment_host_scan_exp,
    run_host_scan,
)


class PointSourceTest(unittest.TestCase):
    def test_cartesian_point_source(self):
        source = CartesianPointSource([[0, 1], [10, 20]])
        self.assertEqual(
            [point.axis_values for point in source],
            [(0, 10), (0, 20), (1, 10), (1, 20)],
        )

    def test_zip_point_source(self):
        source = ZipPointSource([[0, 1, 2], [10, 11, 12]])
        self.assertEqual(
            [point.axis_values for point in source],
            [(0, 10), (1, 11), (2, 12)],
        )

    def test_explicit_point_source(self):
        source = ExplicitPointSource(2, [(0, 10), (3, 13)])
        self.assertEqual(
            [point.axis_values for point in source],
            [(0, 10), (3, 13)],
        )


class TwoParamAddFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("a", FloatParam, "a", 0.0)
        self.setattr_param("b", FloatParam, "b", 0.0)
        self.setattr_result("sum", FloatChannel)

    def run_once(self):
        self.sum.push(self.a.get() + self.b.get())


class HostCallsKernelHelperFragment(ExpFragment):
    """Show that the host runtime can orchestrate scans around kernel helper calls."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("value", FloatParam, "Value", 0.0)
        self.setattr_result("result", FloatChannel)

    @kernel
    def _prepare_hardware(self):
        pass

    @kernel
    def _touch_hardware(self):
        pass

    def device_setup(self):
        self._prepare_hardware()

    def run_once(self):
        self._touch_hardware()
        self.result.push(self.value.get() + 1.0)


class NestedChildScanParent(ExpFragment):
    """Parent fragment that launches a child scan during each outer point.

    The child fragment is detached from the parent's normal execution/result traversal
    so that the nested scan is the only thing that drives it.
    """

    def build_fragment(self):
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_fragment("child", AddOneFragment)
        self.detach_fragment(self.child)
        self.setattr_result("child_total", FloatChannel)

    def run_once(self):
        child_request = ScanRequest.cartesian(
            [(self.child.value, [self.outer.get(), self.outer.get() + 1.0])],
            site=make_child_scan_site("child_scan"),
        )
        child_result = run_host_scan(self, self.child, child_request)
        self.child_total.push(sum(child_result.values[self.child.result]))


class RecursiveLeafScanFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("base", FloatParam, "base", 0.0)
        self.setattr_fragment("leaf", AddOneFragment)
        self.detach_fragment(self.leaf)
        self.setattr_result("leaf_total", FloatChannel)

    def run_once(self):
        leaf_request = ScanRequest.explicit(
            [self.leaf.value],
            [[self.base.get()], [self.base.get() + 0.5]],
            site=make_child_scan_site("leaf_scan"),
        )
        leaf_result = run_host_scan(self, self.leaf, leaf_request)
        self.leaf_total.push(sum(leaf_result.values[self.leaf.result]))


class RecursiveScanParent(ExpFragment):
    """Root fragment that scans a fragment which itself launches nested scans."""

    def build_fragment(self):
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_fragment("middle", RecursiveLeafScanFragment)
        self.detach_fragment(self.middle)
        self.setattr_result("middle_total", FloatChannel)

    def run_once(self):
        middle_request = ScanRequest.explicit(
            [self.middle.base],
            [[self.outer.get()], [self.outer.get() + 10.0]],
            site=make_child_scan_site("middle_scan"),
        )
        middle_result = run_host_scan(self, self.middle, middle_request)
        self.middle_total.push(sum(middle_result.values[self.middle.leaf_total]))


def _fit_line_through_origin(xs, ys) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return float(np.dot(xs, ys) / np.dot(xs, xs))


def _fit_power_law_exponent(ps, ms) -> float:
    ps = np.asarray(ps, dtype=float)
    ms = np.asarray(ms, dtype=float)
    log_p = np.log(ps)
    log_m = np.log(ms)
    return float(np.dot(log_p, log_m) / np.dot(log_p, log_p))


class ManualLineAnalysisFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("p", FloatParam, "p", 2.0)
        self.setattr_param("e", FloatParam, "e", 2.0)
        self.setattr_param("x", FloatParam, "x", 0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push((self.p.get() ** self.e.get()) * self.x.get())


class ManualScanXFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("line", ManualLineAnalysisFragment)
        self.detach_fragment(self.line)
        self.setattr_result("m", FloatChannel)

    def run_once(self):
        request = ScanRequest.explicit(
            [self.line.x],
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            site=make_child_scan_site("scan_x"),
        )
        result = run_host_scan(self, self.line, request)
        xs = next(iter(result.coordinates.values()))
        ys = result.values[self.line.y]
        self.m.push(_fit_line_through_origin(xs, ys))


class ManualHowDoesPVaryFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("scan_x", ManualScanXFragment)
        self.detach_fragment(self.scan_x)
        self.setattr_result("fit_e", FloatChannel)

    def run_once(self):
        request = ScanRequest.explicit(
            [self.scan_x.line.p],
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            site=make_child_scan_site("scan_p"),
        )
        result = run_host_scan(self, self.scan_x, request)
        ps = next(iter(result.coordinates.values()))
        ms = result.values[self.scan_x.m]
        self.fit_e.push(_fit_power_law_exponent(ps, ms))


class HostRuntimeCase(HasEnvironmentCase):
    def test_host_scan_session_writes_root_scan_site(self):
        self.dataset_db.data["system_id"] = (True, "system")
        fragment = self.create(AddOneFragment, [])
        request = ScanRequest.cartesian([(fragment.value, [0.0, 1.0, 2.0])])

        session = HostScanSession(fragment, fragment, request)
        result = session.run()

        prefix = result.site_prefix
        self.assertEqual(prefix, "ndscan.rid_0.site.root.")
        self.assertEqual(self.dataset_db.get(prefix + "fragment_fqn"), "fixtures.AddOneFragment")
        self.assertEqual(self.dataset_db.get(prefix + "source_id"), "system_0")
        self.assertEqual(self.dataset_db.get(prefix + "completed"), True)
        self.assertEqual(json.loads(self.dataset_db.get(prefix + "site_path")), [])
        self.assertEqual(
            self.dataset_db.get(prefix + "points.axis_0"),
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(
            self.dataset_db.get(prefix + "points.channel_0"),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            result.coordinates[("fixtures.AddOneFragment.value", "")],
            [0.0, 1.0, 2.0],
        )

    def test_host_scan_session_supports_zipped_points(self):
        fragment = self.create(TwoParamAddFragment, [])
        request = ScanRequest.zipped(
            [
                (fragment.a, [1.0, 2.0, 3.0]),
                (fragment.b, [10.0, 20.0, 30.0]),
            ],
            site=ScanSite(path=("zipped_demo",)),
        )

        session = HostScanSession(fragment, fragment, request)
        result = session.run()

        prefix = result.site_prefix
        self.assertEqual(prefix, "ndscan.rid_0.site.root.zipped_demo.")
        self.assertEqual(
            self.dataset_db.get(prefix + "points.axis_0"),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            self.dataset_db.get(prefix + "points.axis_1"),
            [10.0, 20.0, 30.0],
        )
        self.assertEqual(
            self.dataset_db.get(prefix + "points.channel_0"),
            [11.0, 22.0, 33.0],
        )

    def test_host_scan_experiment_adapter_runs(self):
        self.dataset_db.data["system_id"] = (True, "system")
        HostAddOneScan = make_fragment_host_scan_exp(
            AddOneFragment,
            lambda fragment: ScanRequest.explicit(
                [fragment.value],
                [[5.0], [7.0]],
            ),
        )

        exp = self.create(HostAddOneScan)
        exp.prepare()
        exp.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.dataset_db.get(prefix + "points.axis_0"), [5.0, 7.0])
        self.assertEqual(self.dataset_db.get(prefix + "points.channel_0"), [6.0, 8.0])

    def test_host_scan_session_allows_kernel_helpers_inside_host_methods(self):
        fragment = self.create(HostCallsKernelHelperFragment, [])
        request = ScanRequest.explicit([fragment.value], [[4.0], [5.0]])

        session = HostScanSession(fragment, fragment, request)
        session.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.dataset_db.get(prefix + "points.axis_0"), [4.0, 5.0])
        self.assertEqual(self.dataset_db.get(prefix + "points.channel_0"), [5.0, 6.0])

    def test_make_child_scan_site_requires_active_parent_point(self):
        with self.assertRaises(RuntimeError):
            make_child_scan_site("not_inside_a_scan")

    def test_nested_child_scan_uses_same_runtime_core(self):
        parent = self.create(NestedChildScanParent, [])
        request = ScanRequest.explicit([parent.outer], [[10.0], [20.0]])

        session = HostScanSession(parent, parent, request)
        session.run()

        root_prefix = "ndscan.rid_0.site.root."
        child_prefix = "ndscan.rid_0.site.root.child_scan."

        self.assertEqual(self.dataset_db.get(root_prefix + "points.axis_0"), [10.0, 20.0])
        self.assertEqual(
            self.dataset_db.get(root_prefix + "points.channel_0"),
            [23.0, 43.0],
        )

        self.assertEqual(json.loads(self.dataset_db.get(child_prefix + "site_path")), ["child_scan"])
        self.assertEqual(
            json.loads(self.dataset_db.get(child_prefix + "parent_site_path")),
            [],
        )
        self.assertEqual(self.dataset_db.get(child_prefix + "starts"), [0, 2])
        self.assertEqual(
            self.dataset_db.get(child_prefix + "parent_point_indices"),
            [0, 1],
        )
        self.assertEqual(
            self.dataset_db.get(child_prefix + "points.axis_0"),
            [10.0, 11.0, 20.0, 21.0],
        )
        self.assertEqual(
            self.dataset_db.get(child_prefix + "points.channel_0"),
            [11.0, 12.0, 21.0, 22.0],
        )

    def test_recursive_nested_scans_build_natural_site_tree(self):
        parent = self.create(RecursiveScanParent, [])
        request = ScanRequest.explicit([parent.outer], [[1.0], [2.0]])

        session = HostScanSession(parent, parent, request)
        session.run()

        root_prefix = "ndscan.rid_0.site.root."
        middle_prefix = "ndscan.rid_0.site.root.middle_scan."
        leaf_prefix = "ndscan.rid_0.site.root.middle_scan.leaf_scan."

        self.assertEqual(
            self.dataset_db.get(root_prefix + "points.channel_0"),
            [29.0, 33.0],
        )

        self.assertEqual(
            self.dataset_db.get(middle_prefix + "starts"),
            [0, 2],
        )
        self.assertEqual(
            self.dataset_db.get(middle_prefix + "parent_point_indices"),
            [0, 1],
        )
        self.assertEqual(
            self.dataset_db.get(middle_prefix + "points.axis_0"),
            [1.0, 11.0, 2.0, 12.0],
        )
        self.assertEqual(
            self.dataset_db.get(middle_prefix + "points.channel_0"),
            [4.5, 24.5, 6.5, 26.5],
        )

        self.assertEqual(
            json.loads(self.dataset_db.get(leaf_prefix + "parent_site_path")),
            ["middle_scan"],
        )
        self.assertEqual(
            self.dataset_db.get(leaf_prefix + "starts"),
            [0, 2, 4, 6],
        )
        self.assertEqual(
            self.dataset_db.get(leaf_prefix + "parent_point_indices"),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            self.dataset_db.get(leaf_prefix + "points.axis_0"),
            [1.0, 1.5, 11.0, 11.5, 2.0, 2.5, 12.0, 12.5],
        )
        self.assertEqual(
            self.dataset_db.get(leaf_prefix + "points.channel_0"),
            [2.0, 2.5, 12.0, 12.5, 3.0, 3.5, 13.0, 13.5],
        )

    def test_manual_nested_analysis_chain_matches_example_style(self):
        parent = self.create(ManualHowDoesPVaryFragment, [])
        session = HostScanSession(parent, parent, ScanRequest.single())
        session.run()

        root_prefix = "ndscan.rid_0.site.root."
        p_prefix = "ndscan.rid_0.site.root.scan_p."
        x_prefix = "ndscan.rid_0.site.root.scan_p.scan_x."

        self.assertAlmostEqual(
            self.dataset_db.get(root_prefix + "points.channel_0")[0],
            2.0,
            places=6,
        )
        self.assertEqual(
            self.dataset_db.get(p_prefix + "points.axis_0"),
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        self.assertEqual(
            self.dataset_db.get(p_prefix + "points.channel_0"),
            [1.0, 4.0, 9.0, 16.0, 25.0],
        )
        self.assertEqual(self.dataset_db.get(x_prefix + "starts"), [0, 6, 12, 18, 24])

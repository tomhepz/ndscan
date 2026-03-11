"""Tests for the new host-only runtime."""

import json
import unittest
from unittest.mock import patch

import numpy as np
from artiq.language.core import TerminationRequested
from mock_environment import HasEnvironmentCase

from ndscan.experiment import (
    CartesianPointSource,
    ConcatPointSource,
    CustomAnalysis,
    ExpFragment,
    ExplicitPointSource,
    FloatChannel,
    FloatParam,
    PointObservation,
    ProductPointSource,
    RecursiveMidpointPointSource1D,
    HostScanSession,
    ScanRequest,
    ScanSite,
    RestartKernelTransitoryError,
    UntilConditionPointSource,
    ZipPointSource,
    annotations,
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

    def test_explicit_point_source_next_batch(self):
        source = ExplicitPointSource(2, [(0, 10), (3, 13), (4, 14)])
        self.assertEqual(
            [point.axis_values for point in source.next_batch(2)],
            [(0, 10), (3, 13)],
        )
        self.assertFalse(source.is_finished())
        self.assertEqual(
            [point.axis_values for point in source.next_batch(2)],
            [(4, 14)],
        )
        self.assertTrue(source.is_finished())

    def test_concat_point_source_runs_children_in_sequence(self):
        source = ConcatPointSource(
            [
                ExplicitPointSource(1, [(0,), (1,)]),
                ExplicitPointSource(1, [(10,), (11,)]),
            ]
        )
        self.assertEqual(
            [point.axis_values for point in source],
            [(0,), (1,), (10,), (11,)],
        )
        self.assertEqual(
            source.describe()["children"][0]["kind"],
            "explicit",
        )

    def test_product_point_source_combines_child_axes(self):
        source = ProductPointSource(
            [
                ExplicitPointSource(1, [(0,), (1,)]),
                ExplicitPointSource(2, [(10, 100), (20, 200)]),
            ]
        )
        self.assertEqual(
            [point.axis_values for point in source],
            [
                (0, 10, 100),
                (0, 20, 200),
                (1, 10, 100),
                (1, 20, 200),
            ],
        )

    def test_recursive_midpoint_point_source_refines_breadth_first(self):
        source = RecursiveMidpointPointSource1D(0.0, 8.0, max_depth=3)
        self.assertEqual(
            [point.axis_values for point in source],
            [(0.0,), (8.0,), (4.0,), (2.0,), (6.0,), (1.0,), (3.0,), (5.0,), (7.0,)],
        )

    def test_until_condition_point_source_stops_after_predicate_matches(self):
        source = UntilConditionPointSource(
            ExplicitPointSource(1, [(0,), (1,), (2,), (3,)]),
            lambda observation: observation.channel_values["channel_0"] >= 30,
            predicate_description="channel_0 >= 30",
        )

        first = source.next_batch(1)
        self.assertEqual([point.axis_values for point in first], [(0,)])

        source.observe(
            PointObservation(
                point_index=0,
                axis_values={"axis_0": 0},
                channel_values={"channel_0": 10},
            )
        )
        self.assertFalse(source.is_finished())

        second = source.next_batch(1)
        self.assertEqual([point.axis_values for point in second], [(1,)])
        source.observe(
            PointObservation(
                point_index=1,
                axis_values={"axis_0": 1},
                channel_values={"channel_0": 30},
            )
        )
        self.assertTrue(source.is_finished())
        self.assertEqual(source.next_batch(1), [])


class TwoParamAddFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("a", FloatParam, "a", 0.0)
        self.setattr_param("b", FloatParam, "b", 0.0)
        self.setattr_result("sum", FloatChannel)

    def run_once(self):
        self.sum.push(self.a.get() + self.b.get())


class PlainAddOneFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("value", FloatParam, "value", 0.0)
        self.setattr_result("result", FloatChannel)

    def run_once(self):
        self.result.push(self.value.get() + 1.0)


class VisibleAndHiddenNumericFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("value", FloatParam, "value", 0.0)
        self.setattr_result("visible", FloatChannel)
        self.setattr_result("hidden", FloatChannel, save_by_default=False)

    def run_once(self):
        self.visible.push(self.value.get() + 1.0)
        self.hidden.push(self.value.get() + 10.0)


class CountingLifecycleFragment(ExpFragment):
    """Fragment whose host lifecycle is easy to assert against in batching tests."""

    def build_fragment(self):
        self.setattr_param("value", FloatParam, "value", 0.0)
        self.setattr_result("result", FloatChannel)
        self.host_setup_calls = 0
        self.host_cleanup_calls = 0

    def host_setup(self):
        self.host_setup_calls += 1

    def host_cleanup(self):
        self.host_cleanup_calls += 1

    def run_once(self):
        self.result.push(self.value.get() + 1.0)


class RestartOnceFragment(ExpFragment):
    """Fragment that forces one host-context restart for a chosen scan point."""

    def build_fragment(self):
        self.setattr_param("value", FloatParam, "value", 0.0)
        self.setattr_result("result", FloatChannel)
        self.host_setup_calls = 0
        self.host_cleanup_calls = 0
        self._restart_values = {2.0}

    def host_setup(self):
        self.host_setup_calls += 1

    def host_cleanup(self):
        self.host_cleanup_calls += 1

    def run_once(self):
        value = self.value.get()
        if value in self._restart_values:
            self._restart_values.remove(value)
            raise RestartKernelTransitoryError("restart host context")
        self.result.push(value + 1.0)


class RecordingBatchPointSource(ExplicitPointSource):
    """Explicit source that records how the runner consumes batched points."""

    def __init__(
        self,
        axis_count,
        points,
        *,
        preferred_batch_size=None,
    ):
        super().__init__(axis_count, points)
        self._preferred_batch_size = preferred_batch_size
        self.requested_batch_limits = []
        self.observed_batches = []

    def next_batch(self, max_points: int):
        self.requested_batch_limits.append(max_points)
        return super().next_batch(max_points)

    def preferred_batch_size(self, default: int) -> int:
        if self._preferred_batch_size is None:
            return default
        return self._preferred_batch_size

    def observe(self, observation):
        raise AssertionError("Host runtime should call observe_batch() at batch boundaries")

    def observe_batch(self, observations):
        self.observed_batches.append(
            [observation.axis_values["axis_0"] for observation in observations]
        )


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
        self.setattr_fragment("child", PlainAddOneFragment, detached=True)
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
        self.setattr_fragment("leaf", PlainAddOneFragment, detached=True)
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
        self.setattr_fragment("middle", RecursiveLeafScanFragment, detached=True)
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


class AnalysedLineFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("p", FloatParam, "p", 2.0)
        self.setattr_param("e", FloatParam, "e", 2.0)
        self.setattr_param("x", FloatParam, "x", 0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push((self.p.get() ** self.e.get()) * self.x.get())

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.x],
                self._analyse_gradient,
                [FloatChannel("m", "Extracted slope")],
            )
        ]

    def _analyse_gradient(self, axis_values, result_values, analysis_results):
        xs = axis_values[self.x]
        ys = result_values[self.y]

        m = _fit_line_through_origin(xs, ys)
        fit_xs = np.linspace(min(xs), max(xs), 20)
        fit_ys = m * fit_xs

        analysis_results["m"].push(m)
        return [
            annotations.curve_1d(
                x_axis=self.x,
                x_values=fit_xs,
                y_axis=self.y,
                y_values=fit_ys,
            )
        ]


class AnalysedScanXFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("line", AnalysedLineFragment, detached=True)
        self.setattr_result("m", FloatChannel)

    def run_once(self):
        request = ScanRequest.explicit(
            [self.line.x],
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            site=make_child_scan_site("scan_x"),
        )
        result = run_host_scan(self, self.line, request)
        self.m.push(result.analysis_results["m"])

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.line.p],
                self._analyse_exponent,
                [FloatChannel("fit_e", "Extracted exponent")],
            )
        ]

    def _analyse_exponent(self, axis_values, result_values, analysis_results):
        ps = axis_values[self.line.p]
        ms = result_values[self.m]

        fit_e = _fit_power_law_exponent(ps, ms)
        analysis_results["fit_e"].push(fit_e)
        return []


class AnalysedHowDoesPVaryFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("scan_x", AnalysedScanXFragment, detached=True)
        self.setattr_result("fit_e", FloatChannel)

    def run_once(self):
        request = ScanRequest.explicit(
            [self.scan_x.line.p],
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            site=make_child_scan_site("scan_p"),
        )
        result = run_host_scan(self, self.scan_x, request)
        self.fit_e.push(result.analysis_results["fit_e"])


class HostRuntimeCase(HasEnvironmentCase):
    def d(self, prefix, key):
        return self.dataset_db.get(prefix + key)

    def j(self, prefix, key):
        return json.loads(self.d(prefix, key))

    def test_host_scan_session_writes_root_scan_site(self):
        self.dataset_db.data["system_id"] = (True, "system")
        fragment = self.create(PlainAddOneFragment, [])
        request = ScanRequest.cartesian([(fragment.value, [0.0, 1.0, 2.0])])

        session = HostScanSession(fragment, fragment, request)
        result = session.run()

        prefix = result.site_prefix
        self.assertEqual(prefix, "ndscan.rid_0.site.root.")
        self.assertEqual(
            self.d(prefix, "site.fragment_fqn"),
            f"{__name__}.PlainAddOneFragment",
        )
        self.assertEqual(self.d(prefix, "site.source_id"), "system_0")
        self.assertEqual(self.d(prefix, "state.completed"), True)
        self.assertEqual(self.d(prefix, "state.num_points"), 3)
        self.assertEqual(self.j(prefix, "site.path"), [])
        self.assertEqual(
            self.j(prefix, "scan.axes"),
            {
                "axis_0": {
                    "path": "",
                    "param": {
                        "description": "value",
                        "default": "0.0",
                        "fqn": f"{__name__}.PlainAddOneFragment.value",
                        "spec": {
                            "is_scannable": True,
                            "scale": 1.0,
                            "step": 0.1,
                        },
                        "type": "float",
                    },
                }
            },
        )
        self.assertEqual(list(self.j(prefix, "scan.channels").keys()), ["channel_0"])
        self.assertEqual(
            self.d(prefix, "points.axis_0"),
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(
            self.d(prefix, "points.channel_0"),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            result.coordinates[(f"{__name__}.PlainAddOneFragment.value", "")],
            [0.0, 1.0, 2.0],
        )

    def test_host_scan_session_records_root_and_point_unix_timestamps(self):
        fragment = self.create(PlainAddOneFragment, [])
        request = ScanRequest.cartesian([(fragment.value, [0.0, 1.0, 2.0])])

        with patch(
            "ndscan.experiment.host_runtime.time.time",
            side_effect=[1000.0, 1001.0, 1002.0, 1003.0],
        ):
            session = HostScanSession(fragment, fragment, request)
            session.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.d(prefix, "site.start_unix_time"), 1000.0)
        self.assertEqual(
            self.d(prefix, "points.acquired_at_unix"),
            [1001.0, 1002.0, 1003.0],
        )

    def test_host_scan_session_executes_and_observes_points_in_batches(self):
        fragment = self.create(CountingLifecycleFragment, [])
        point_source = RecordingBatchPointSource(
            1,
            [(0.0,), (1.0,), (2.0,), (3.0,), (4.0,)],
            preferred_batch_size=2,
        )
        request = ScanRequest(
            axes=(fragment.value,),
            point_source=point_source,
            max_points_per_batch=4,
        )

        session = HostScanSession(fragment, fragment, request)
        session.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(point_source.requested_batch_limits, [2, 2, 2])
        self.assertEqual(point_source.observed_batches, [[0.0, 1.0], [2.0, 3.0], [4.0]])
        self.assertEqual(fragment.host_setup_calls, 3)
        self.assertEqual(fragment.host_cleanup_calls, 3)
        self.assertEqual(self.scheduler.num_check_pause_calls, 2)
        self.assertEqual(self.d(prefix, "points.axis_0"), [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_host_scan_session_flushes_completed_batch_before_pause(self):
        fragment = self.create(CountingLifecycleFragment, [])
        point_source = RecordingBatchPointSource(
            1,
            [(0.0,), (1.0,), (2.0,), (3.0,)],
            preferred_batch_size=2,
        )
        request = ScanRequest(
            axes=(fragment.value,),
            point_source=point_source,
            max_points_per_batch=2,
        )
        self.scheduler.num_check_pause_calls_until_termination = 1

        session = HostScanSession(fragment, fragment, request)
        with self.assertRaises(TerminationRequested):
            session.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(point_source.observed_batches, [[0.0, 1.0]])
        self.assertEqual(fragment.host_setup_calls, 1)
        self.assertEqual(fragment.host_cleanup_calls, 1)
        self.assertEqual(self.scheduler.num_check_pause_calls, 1)
        self.assertEqual(self.d(prefix, "points.axis_0"), [0.0, 1.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 2.0])
        self.assertEqual(self.d(prefix, "state.num_points"), 2)

    def test_host_scan_session_flushes_partial_batch_before_restarting(self):
        fragment = self.create(RestartOnceFragment, [])
        point_source = RecordingBatchPointSource(
            1,
            [(0.0,), (1.0,), (2.0,), (3.0,)],
        )
        request = ScanRequest(
            axes=(fragment.value,),
            point_source=point_source,
            max_points_per_batch=3,
        )

        session = HostScanSession(fragment, fragment, request)
        session.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(point_source.requested_batch_limits, [3, 3])
        self.assertEqual(point_source.observed_batches, [[0.0, 1.0], [2.0], [3.0]])
        self.assertEqual(fragment.host_setup_calls, 3)
        self.assertEqual(fragment.host_cleanup_calls, 3)
        self.assertEqual(self.d(prefix, "points.axis_0"), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 2.0, 3.0, 4.0])

    def test_numeric_channels_can_opt_out_of_save_by_default(self):
        fragment = self.create(VisibleAndHiddenNumericFragment, [])
        request = ScanRequest.explicit([fragment.value], [[1.0], [2.0]])

        session = HostScanSession(fragment, fragment, request)
        result = session.run()

        prefix = result.site_prefix
        self.assertEqual(
            self.d(prefix, "points.channel_0"),
            [2.0, 3.0],
        )
        self.assertNotIn(prefix + "points.channel_1", self.dataset_db.data)
        self.assertIn(fragment.visible, result.values)
        self.assertNotIn(fragment.hidden, result.values)

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
            self.d(prefix, "points.axis_0"),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            self.d(prefix, "points.axis_1"),
            [10.0, 20.0, 30.0],
        )
        self.assertEqual(
            self.d(prefix, "points.channel_0"),
            [11.0, 22.0, 33.0],
        )

    def test_host_scan_session_respects_until_condition_point_source(self):
        fragment = self.create(PlainAddOneFragment, [])
        request = ScanRequest(
            axes=(fragment.value,),
            point_source=UntilConditionPointSource(
                ExplicitPointSource(1, [(0.0,), (1.0,), (2.0,), (3.0,)]),
                lambda observation: observation.channel_values["channel_0"] >= 3.0,
                predicate_description="channel_0 >= 3.0",
            ),
        )

        session = HostScanSession(fragment, fragment, request)
        session.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.d(prefix, "points.axis_0"), [0.0, 1.0, 2.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 2.0, 3.0])

    def test_host_scan_session_executes_default_analyses_after_run(self):
        fragment = self.create(AnalysedLineFragment, [])
        request = ScanRequest.explicit(
            [fragment.x],
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        )

        session = HostScanSession(fragment, fragment, request)
        result = session.run()

        prefix = result.site_prefix
        self.assertAlmostEqual(result.analysis_results["m"], 4.0, places=6)
        self.assertAlmostEqual(self.d(prefix, "analysis.output.m"), 4.0)

        metadata = self.j(prefix, "analysis.outputs")
        self.assertIn("m", metadata)
        self.assertEqual(metadata["m"]["path"], "m")

        annotations_data = self.j(prefix, "analysis.annotations")
        self.assertEqual(len(annotations_data), 1)
        self.assertEqual(annotations_data[0]["kind"], "curve")
        self.assertEqual(result.annotations, annotations_data)

    def test_host_scan_experiment_adapter_runs(self):
        self.dataset_db.data["system_id"] = (True, "system")
        HostAddOneScan = make_fragment_host_scan_exp(
            PlainAddOneFragment,
            lambda fragment: ScanRequest.explicit(
                [fragment.value],
                [[5.0], [7.0]],
            ),
        )

        exp = self.create(HostAddOneScan)
        exp.prepare()
        exp.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.d(prefix, "points.axis_0"), [5.0, 7.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [6.0, 8.0])

    def test_host_scan_session_allows_kernel_helpers_inside_host_methods(self):
        fragment = self.create(HostCallsKernelHelperFragment, [])
        request = ScanRequest.explicit([fragment.value], [[4.0], [5.0]])

        session = HostScanSession(fragment, fragment, request)
        session.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.d(prefix, "points.axis_0"), [4.0, 5.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [5.0, 6.0])

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

        self.assertEqual(self.d(root_prefix, "points.axis_0"), [10.0, 20.0])
        self.assertEqual(
            self.d(root_prefix, "points.channel_0"),
            [23.0, 43.0],
        )

        self.assertEqual(self.j(child_prefix, "site.path"), ["child_scan"])
        self.assertEqual(
            self.j(child_prefix, "site.parent_path"),
            [],
        )
        self.assertEqual(self.d(child_prefix, "segments.start_index"), [0, 2])
        self.assertEqual(
            self.d(child_prefix, "segments.parent_point_index"),
            [0, 1],
        )
        self.assertEqual(
            self.d(child_prefix, "points.axis_0"),
            [10.0, 11.0, 20.0, 21.0],
        )
        self.assertEqual(
            self.d(child_prefix, "points.channel_0"),
            [11.0, 12.0, 21.0, 22.0],
        )

    def test_nested_child_scan_records_segment_start_unix_timestamps(self):
        parent = self.create(NestedChildScanParent, [])
        request = ScanRequest.explicit([parent.outer], [[10.0], [20.0]])

        with patch(
            "ndscan.experiment.host_runtime.time.time",
            side_effect=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        ):
            session = HostScanSession(parent, parent, request)
            session.run()

        root_prefix = "ndscan.rid_0.site.root."
        child_prefix = "ndscan.rid_0.site.root.child_scan."
        self.assertEqual(self.d(root_prefix, "site.start_unix_time"), 1.0)
        self.assertEqual(self.d(root_prefix, "points.acquired_at_unix"), [6.0, 11.0])
        self.assertEqual(self.d(child_prefix, "site.start_unix_time"), 7.0)
        self.assertEqual(
            self.d(child_prefix, "segments.start_unix_time"),
            [3.0, 8.0],
        )
        self.assertEqual(
            self.d(child_prefix, "points.acquired_at_unix"),
            [4.0, 5.0, 9.0, 10.0],
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
            self.d(root_prefix, "points.channel_0"),
            [29.0, 33.0],
        )

        self.assertEqual(
            self.d(middle_prefix, "segments.start_index"),
            [0, 2],
        )
        self.assertEqual(
            self.d(middle_prefix, "segments.parent_point_index"),
            [0, 1],
        )
        self.assertEqual(
            self.d(middle_prefix, "points.axis_0"),
            [1.0, 11.0, 2.0, 12.0],
        )
        self.assertEqual(
            self.d(middle_prefix, "points.channel_0"),
            [4.5, 24.5, 6.5, 26.5],
        )

        self.assertEqual(
            self.j(leaf_prefix, "site.parent_path"),
            ["middle_scan"],
        )
        self.assertEqual(
            self.d(leaf_prefix, "segments.start_index"),
            [0, 2, 4, 6],
        )
        self.assertEqual(
            self.d(leaf_prefix, "segments.parent_point_index"),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            self.d(leaf_prefix, "points.axis_0"),
            [1.0, 1.5, 11.0, 11.5, 2.0, 2.5, 12.0, 12.5],
        )
        self.assertEqual(
            self.d(leaf_prefix, "points.channel_0"),
            [2.0, 2.5, 12.0, 12.5, 3.0, 3.5, 13.0, 13.5],
        )

    def test_nested_default_analyses_chain_matches_example_style(self):
        parent = self.create(AnalysedHowDoesPVaryFragment, [])
        session = HostScanSession(parent, parent, ScanRequest.single())
        result = session.run()

        root_prefix = "ndscan.rid_0.site.root."
        p_prefix = "ndscan.rid_0.site.root.scan_p."
        x_prefix = "ndscan.rid_0.site.root.scan_p.scan_x."

        self.assertAlmostEqual(
            self.d(root_prefix, "points.channel_0")[0],
            2.0,
            places=6,
        )
        self.assertEqual(
            self.d(p_prefix, "points.axis_0"),
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        self.assertEqual(
            self.d(p_prefix, "points.channel_0"),
            [1.0, 4.0, 9.0, 16.0, 25.0],
        )
        self.assertEqual(self.d(x_prefix, "segments.start_index"), [0, 6, 12, 18, 24])
        self.assertAlmostEqual(result.values[parent.fit_e][0], 2.0, places=6)
        self.assertAlmostEqual(self.d(p_prefix, "analysis.output.fit_e"), 2.0)
        self.assertAlmostEqual(self.d(x_prefix, "analysis.output.m"), 25.0)

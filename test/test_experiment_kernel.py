"""
Tests that actually run against a core device (by default, the emulator).

Collecting them all in a single module might or might not turn out to be a good idea;
could also keep them inline with the other test_experiment_* unit test modules.
"""

import math
import json
import unittest
from collections import Counter
from dataclasses import dataclass
from enum import Enum, unique

import numpy as np
from artiq.language import kernel, portable, rpc
from emulator_environment import KernelEmulatorCase
from fixtures import TrivialKernelFragment
from examples.prepared_scan_kernel_nested import (
    PreparedKernelNestedVariationFragment,
)
from examples.prepared_scan_kernel_nested_ttl import (
    PreparedKernelNestedTtlFragment,
)
from examples.prepared_scan_kernel_online_fit import (
    PreparedKernelOnlineFitFragment,
)
try:
    from examples.prepared_scan_kernel_bayesian_optimisation import (
        KernelBayesianOptimisationFragment,
        make_request as make_kernel_bo_request,
    )

    _KERNEL_BO_DEPS_AVAILABLE = True
except ImportError:
    _KERNEL_BO_DEPS_AVAILABLE = False

from ndscan.define.fragment import (
    AggregateExpFragment,
    ExpFragment,
    RestartKernelTransitoryError,
    TransitoryError,
)
from ndscan.define.parameters import (
    BoolParam,
    EnumParam,
    FloatParam,
    IntParam,
    StringParam,
)
from ndscan.define.result_channels import FloatChannel, IntChannel, OpaqueChannel
from ndscan.legacy.entry_point import make_fragment_scan_exp, run_fragment_once
from ndscan.legacy.scan_generator import LinearGenerator, ListGenerator
from ndscan.legacy.subscan import SubscanExpFragment, setattr_subscan
from ndscan.runtime.api import (
    PreparedScan,
    prepare_child_scan,
    setattr_prepared_child_scan,
)
from ndscan.scan.mapping import ParameterMapping, ScanVariable
from ndscan.scan.request import ExecutionPolicy, ScanRequest
from ndscan.utils import SCHEMA_REVISION, SCHEMA_REVISION_KEY


def _execute_and_inspect(scan):
    scan.execute()
    return scan.inspect()


class RunOneKernelCase(KernelEmulatorCase):
    def test_run_once_kernel(self):
        fragment = self.create(TrivialKernelFragment, [])
        run_fragment_once(fragment)


class KernelStreamingLeafFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    @kernel
    def run_once(self):
        self.y.push(self.x.get() + 1.0)


class KernelMappedDriveFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("drive", FloatParam, "drive", default=0.0)
        self.setattr_result("result", FloatChannel)

    @kernel
    def run_once(self):
        self.result.push(2.0 * self.drive.get())


class KernelAdHocMappedParamFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("logical_drive", FloatParam, "logical drive", default=0.0)
        self.setattr_param("drive", FloatParam, "drive", default=0.0)
        self.setattr_result("result", FloatChannel)

    @kernel
    def run_once(self):
        self.result.push(2.0 * self.drive.get())


class KernelRebindDriveWrapperFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_fragment("child", KernelMappedDriveFragment)
        self.setattr_param("logical_drive", FloatParam, "logical drive", default=0.0)
        self.rebind_param(
            self.child.drive,
            [self.logical_drive],
            lambda values: values[self.logical_drive] + 0.5,
            description="Offset the child drive from the wrapper logical axis",
        )

    @kernel
    def run_once(self):
        self.child.run_once()


class HostOnlyPreparedChildLeaf(ExpFragment):
    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push(self.x.get() + 1.0)


class KernelPreparedChildLeaf(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    @kernel
    def run_once(self):
        self.y.push(self.x.get() + 1.0)


class KernelPreparedGrandchildLeaf(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("z", FloatParam, "z", default=0.0)
        self.setattr_result("value", FloatChannel)

    @kernel
    def run_once(self):
        self.value.push(self.z.get() + 2.0)


class KernelPreparedChildScanParent(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("outer", FloatParam, "outer", default=0.0)
        self.child_scan = setattr_prepared_child_scan(
            self,
            "child",
            HostOnlyPreparedChildLeaf,
            scan_name="child_scan",
        )
        self.setattr_result("result", FloatChannel)

    def host_setup(self):
        self.child_scan.configure(
            ScanRequest.explicit([self.child.x], [[2.0], [3.0]])
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.child_scan.acquire()
        self.result.push(self.outer.get() + 10.0)


class KernelPreparedKernelChildScanParent(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("outer", FloatParam, "outer", default=0.0)
        self.child_scan = setattr_prepared_child_scan(
            self,
            "child",
            KernelPreparedChildLeaf,
            scan_name="child_scan",
        )
        self.setattr_result("result", FloatChannel)

    def host_setup(self):
        self.child_scan.configure(
            ScanRequest.explicit([self.child.x], [[2.0], [3.0]])
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.child_scan.acquire()
        self.result.push(self.outer.get() + 20.0)


class KernelPreparedNestedChild(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.grandchild_scan = setattr_prepared_child_scan(
            self,
            "grandchild",
            KernelPreparedGrandchildLeaf,
            scan_name="grandchild_scan",
        )
        self.setattr_result("y", FloatChannel)

    def host_setup(self):
        self.grandchild_scan.configure(
            ScanRequest.explicit([self.grandchild.z], [[5.0], [6.0]])
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.grandchild_scan.acquire()
        self.y.push(self.x.get() + 10.0)


class KernelPreparedNestedParent(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("outer", FloatParam, "outer", default=0.0)
        self.child_scan = setattr_prepared_child_scan(
            self,
            "child",
            KernelPreparedNestedChild,
            scan_name="child_scan",
        )
        self.setattr_result("result", FloatChannel)

    def host_setup(self):
        # Prime the detached child before the outer kernel is first compiled so the
        # grandchild prepared scan has already fixed its compiler-visible shape.
        self.child_scan.prime()
        self.child_scan.configure(
            ScanRequest.explicit([self.child.x], [[1.0], [2.0]])
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.child_scan.acquire()
        self.result.push(self.outer.get() + 100.0)


class KernelPreparedMappedChildParent(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("outer", FloatParam, "outer", default=0.0)
        self.child_scan = setattr_prepared_child_scan(
            self,
            "child",
            KernelMappedDriveFragment,
            scan_name="child_scan",
        )
        self.setattr_result("result", FloatChannel)

    def host_setup(self):
        logical_drive = ScanVariable("logical_drive")
        self.child_scan.configure(
            ScanRequest.cartesian(
                [(logical_drive, [0.0, 1.0, 2.0])],
                execution_policy=ExecutionPolicy(max_points_per_batch=2),
            ).with_parameter_mappings(
                [
                    ParameterMapping.single_target(
                        self.child.drive,
                        [logical_drive],
                        lambda values: values[logical_drive] + 0.5,
                        description="Offset the physical drive from the logical axis",
                    )
                ]
            )
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.child_scan.acquire()
        self.result.push(self.outer.get() + 1.0)


class KernelStreamingPreparedScanCase(KernelEmulatorCase):
    def test_kernel_streaming_prepared_scan_reuses_one_kernel_entry(self):
        fragment = self.create(KernelStreamingLeafFragment, [])
        request = ScanRequest.linear(
            fragment.x,
            start=-10.0,
            stop=10.0,
            num_points=101,
            execution_policy=ExecutionPolicy(max_points_per_batch=10),
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(result.runtime_stats.batch_count, 11)
        self.assertEqual(result.runtime_stats.point_count, 101)
        values = result.values[fragment.y]
        self.assertEqual(len(values), 101)
        self.assertAlmostEqual(values[0], -9.0)
        self.assertAlmostEqual(values[50], 1.0)
        self.assertAlmostEqual(values[-1], 11.0)

    def test_kernel_streaming_prepared_scan_supports_pseudoparam_mappings(self):
        fragment = self.create(KernelMappedDriveFragment, [])
        logical_drive = ScanVariable(
            "logical_drive",
            description="Logical axis mapped onto the physical drive",
        )
        request = ScanRequest.cartesian(
            [(logical_drive, [0.0, 1.0, 2.0])],
            execution_policy=ExecutionPolicy(max_points_per_batch=2),
        ).with_parameter_mappings(
            [
                ParameterMapping.single_target(
                    fragment.drive,
                    [logical_drive],
                    lambda values: values[logical_drive] + 0.5,
                    description="Offset the physical drive from the logical axis",
                )
            ]
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(result.runtime_stats.batch_count, 2)
        self.assertEqual(result.values[fragment.result], [1.0, 3.0, 5.0])

        scheduler = fragment.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        prefix = f"ndscan.rid_{rid}.site.root."
        self.assertEqual(fragment.get_dataset(prefix + "points.pseudoparam_0"), [0.0, 1.0, 2.0])
        self.assertEqual(fragment.get_dataset(prefix + "points.param_0"), [0.5, 1.5, 2.5])
        self.assertEqual(fragment.get_dataset(prefix + "points.channel_0"), [1.0, 3.0, 5.0])

    def test_kernel_streaming_prepared_scan_supports_ad_hoc_param_to_param_mappings(
        self,
    ):
        fragment = self.create(KernelAdHocMappedParamFragment, [])
        request = ScanRequest.cartesian(
            [(fragment.logical_drive, [0.0, 1.0, 2.0])],
            execution_policy=ExecutionPolicy(max_points_per_batch=2),
        ).with_parameter_mappings(
            [
                ParameterMapping.single_target(
                    fragment.drive,
                    [fragment.logical_drive],
                    lambda values: values[fragment.logical_drive] + 0.5,
                    description="Offset the physical drive from the logical parameter",
                )
            ]
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(result.runtime_stats.batch_count, 2)
        self.assertEqual(result.values[fragment.result], [1.0, 3.0, 5.0])

        scheduler = fragment.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        prefix = f"ndscan.rid_{rid}.site.root."
        self.assertEqual(fragment.get_dataset(prefix + "points.param_0"), [0.0, 1.0, 2.0])
        self.assertEqual(fragment.get_dataset(prefix + "points.param_1"), [0.5, 1.5, 2.5])
        self.assertEqual(fragment.get_dataset(prefix + "points.channel_0"), [1.0, 3.0, 5.0])

    def test_kernel_streaming_prepared_scan_supports_wrapper_rebinds(self):
        fragment = self.create(KernelRebindDriveWrapperFragment, [])
        request = ScanRequest.cartesian(
            [(fragment.logical_drive, [0.0, 1.0, 2.0])],
            execution_policy=ExecutionPolicy(max_points_per_batch=2),
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(result.runtime_stats.batch_count, 2)
        self.assertEqual(
            result.values[fragment.child.result],
            [1.0, 3.0, 5.0],
        )

        scheduler = fragment.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        prefix = f"ndscan.rid_{rid}.site.root."
        self.assertEqual(fragment.get_dataset(prefix + "points.param_0"), [0.0, 1.0, 2.0])
        self.assertEqual(fragment.get_dataset(prefix + "points.param_1"), [0.5, 1.5, 2.5])
        self.assertEqual(fragment.get_dataset(prefix + "points.channel_0"), [1.0, 3.0, 5.0])

    def test_kernel_parent_can_acquire_prepared_host_child_scan(self):
        fragment = self.create(KernelPreparedChildScanParent, [])
        request = ScanRequest.explicit(
            [fragment.outer],
            [[1.0], [4.0]],
            execution_policy=ExecutionPolicy(max_points_per_batch=2),
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(result.runtime_stats.batch_count, 1)
        self.assertEqual(result.values[fragment.result], [11.0, 14.0])

        scheduler = fragment.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        prefix = f"ndscan.rid_{rid}.site.root.subscans.child_scan."

        self.assertEqual(fragment.get_dataset(prefix + "segments.start_index"), [0, 2])
        self.assertEqual(
            fragment.get_dataset(prefix + "segments.parent_point_index"), [0, 1]
        )
        self.assertEqual(
            fragment.get_dataset(prefix + "points.param_0"), [2.0, 3.0, 2.0, 3.0]
        )
        self.assertEqual(
            fragment.get_dataset(prefix + "points.channel_0"), [3.0, 4.0, 3.0, 4.0]
        )

    def test_kernel_parent_can_acquire_prepared_kernel_child_scan(self):
        fragment = self.create(KernelPreparedKernelChildScanParent, [])
        request = ScanRequest.explicit(
            [fragment.outer],
            [[1.0], [4.0]],
            execution_policy=ExecutionPolicy(max_points_per_batch=2),
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(result.runtime_stats.batch_count, 1)
        self.assertEqual(result.values[fragment.result], [21.0, 24.0])
        self.assertIsNotNone(fragment.child_scan.inspect())
        self.assertEqual(fragment.child_scan.inspect().runtime_stats.batch_count, 2)
        self.assertEqual(
            fragment.child_scan.inspect().runtime_stats.executor_entry_count, 1
        )

        scheduler = fragment.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        prefix = f"ndscan.rid_{rid}.site.root.subscans.child_scan."

        self.assertEqual(fragment.get_dataset(prefix + "segments.start_index"), [0, 2])
        self.assertEqual(
            fragment.get_dataset(prefix + "segments.parent_point_index"), [0, 1]
        )
        self.assertEqual(
            fragment.get_dataset(prefix + "points.param_0"), [2.0, 3.0, 2.0, 3.0]
        )
        self.assertEqual(
            fragment.get_dataset(prefix + "points.channel_0"), [3.0, 4.0, 3.0, 4.0]
        )

    def test_kernel_parent_can_acquire_prepared_kernel_subscan_of_subscan(self):
        fragment = self.create(KernelPreparedNestedParent, [])
        request = ScanRequest.explicit(
            [fragment.outer],
            [[1.0], [4.0]],
            execution_policy=ExecutionPolicy(max_points_per_batch=2),
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(result.runtime_stats.batch_count, 1)
        self.assertEqual(result.values[fragment.result], [101.0, 104.0])
        self.assertIsNotNone(fragment.child_scan.inspect())
        self.assertEqual(fragment.child_scan.inspect().runtime_stats.batch_count, 2)
        self.assertEqual(
            fragment.child_scan.inspect().runtime_stats.executor_entry_count, 1
        )
        self.assertIsNotNone(fragment.child.grandchild_scan.inspect())

        scheduler = fragment.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        child_prefix = f"ndscan.rid_{rid}.site.root.subscans.child_scan."
        grandchild_prefix = f"ndscan.rid_{rid}.site.root.subscans.child_scan.subscans.grandchild_scan."

        self.assertEqual(
            fragment.get_dataset(child_prefix + "segments.parent_point_index"), [0, 1]
        )
        self.assertEqual(
            fragment.get_dataset(child_prefix + "points.param_0"), [1.0, 2.0, 1.0, 2.0]
        )
        self.assertEqual(
            fragment.get_dataset(child_prefix + "points.channel_0"),
            [11.0, 12.0, 11.0, 12.0],
        )

        self.assertEqual(
            fragment.get_dataset(grandchild_prefix + "segments.parent_point_index"),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            fragment.get_dataset(grandchild_prefix + "points.param_0"),
            [5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0],
        )
        self.assertEqual(
            fragment.get_dataset(grandchild_prefix + "points.channel_0"),
            [7.0, 8.0, 7.0, 8.0, 7.0, 8.0, 7.0, 8.0],
        )

    def test_kernel_parent_can_acquire_prepared_kernel_child_with_pseudoparam_mapping(
        self,
    ):
        fragment = self.create(KernelPreparedMappedChildParent, [])
        request = ScanRequest.explicit([fragment.outer], [[1.0]])

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(result.values[fragment.result], [2.0])
        child_result = fragment.child_scan.inspect()
        self.assertIsNotNone(child_result)
        assert child_result is not None
        self.assertEqual(child_result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(child_result.runtime_stats.batch_count, 2)
        self.assertEqual(child_result.values[fragment.child.result], [1.0, 3.0, 5.0])

        prefix = child_result.site_prefix
        self.assertEqual(fragment.get_dataset(prefix + "points.pseudoparam_0"), [0.0, 1.0, 2.0])
        self.assertEqual(fragment.get_dataset(prefix + "points.param_0"), [0.5, 1.5, 2.5])
        self.assertEqual(fragment.get_dataset(prefix + "points.channel_0"), [1.0, 3.0, 5.0])

    def test_prepared_kernel_nested_example_runs(self):
        fragment = self.create(PreparedKernelNestedVariationFragment, [])
        request = ScanRequest.explicit([fragment.outer], [[1.0]])

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.values[fragment.completed], [1.0])

        scheduler = fragment.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        child_prefix = f"ndscan.rid_{rid}.site.root.subscans.scan_p."
        grandchild_prefix = f"ndscan.rid_{rid}.site.root.subscans.scan_p.subscans.scan_x."

        self.assertEqual(
            fragment.get_dataset(child_prefix + "points.param_0"),
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        self.assertEqual(
            fragment.get_dataset(child_prefix + "points.channel_0"),
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        self.assertEqual(
            fragment.get_dataset(grandchild_prefix + "segments.parent_point_index"),
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            fragment.get_dataset(grandchild_prefix + "points.param_0"),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 5,
        )
        self.assertEqual(
            fragment.get_dataset(grandchild_prefix + "points.channel_0"),
            [
                0.5,
                1.5,
                2.5,
                3.5,
                4.5,
                5.5,
                0.5,
                2.5,
                4.5,
                6.5,
                8.5,
                10.5,
                0.5,
                3.5,
                6.5,
                9.5,
                12.5,
                15.5,
                0.5,
                4.5,
                8.5,
                12.5,
                16.5,
                20.5,
                0.5,
                5.5,
                10.5,
                15.5,
                20.5,
                25.5,
            ],
        )

    def test_prepared_kernel_online_fit_example_runs(self):
        fragment = self.create(PreparedKernelOnlineFitFragment, [])
        request = ScanRequest.explicit([fragment.outer], [[1.0]])

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.values[fragment.completed], [1.0])
        self.assertAlmostEqual(result.values[fragment.fit_slope][0], 2.0, places=6)
        self.assertAlmostEqual(result.values[fragment.fit_intercept][0], 0.5, places=6)

        child_result = fragment.line_scan.inspect()
        self.assertEqual(child_result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(child_result.runtime_stats.batch_count, 2)
        fit_slope, fit_intercept = fragment.line_scan.get_outputs()
        self.assertAlmostEqual(fit_slope, 2.0, places=6)
        self.assertAlmostEqual(fit_intercept, 0.5, places=6)
        self.assertAlmostEqual(child_result.analysis_results["fit_slope"], 2.0, places=6)
        self.assertAlmostEqual(
            child_result.analysis_results["fit_intercept"], 0.5, places=6
        )
        self.assertIn("running_line_fit", child_result.online_analysis_results)

        prefix = child_result.site_prefix
        online_result = json.loads(
            fragment.get_dataset(prefix + "analysis.online_result.running_line_fit")
        )
        online_annotations = json.loads(
            fragment.get_dataset(prefix + "analysis.online_annotation.running_line_fit")
        )

        self.assertAlmostEqual(online_result["fit_slope"], 2.0, places=6)
        self.assertAlmostEqual(online_result["fit_intercept"], 0.5, places=6)
        self.assertEqual(
            child_result.online_analysis_results["running_line_fit"],
            online_result,
        )

    def test_prepared_kernel_nested_ttl_example_runs(self):
        fragment = self.create(PreparedKernelNestedTtlFragment, [])
        request = ScanRequest.explicit([fragment.outer], [[1.0]])

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.values[fragment.completed], [1.0])
        self.assertEqual(result.runtime_stats.executor_entry_count, 1)

        child_result = fragment.scan_p.inspect()
        self.assertEqual(child_result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(child_result.values[fragment.scan_x.current_p], [1.0, 2.0, 3.0])

        inner_result = fragment.scan_x.scan_x.inspect()
        self.assertEqual(inner_result.runtime_stats.executor_entry_count, 1)

        scheduler = fragment.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        child_prefix = f"ndscan.rid_{rid}.site.root.subscans.scan_p."
        inner_prefix = f"ndscan.rid_{rid}.site.root.subscans.scan_p.subscans.scan_x."

        self.assertEqual(
            fragment.get_dataset(child_prefix + "points.channel_0"),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            fragment.get_dataset(inner_prefix + "segments.parent_point_index"),
            [0, 1, 2],
        )
        self.assertEqual(
            fragment.get_dataset(inner_prefix + "points.channel_0"),
            [
                1.5,
                2.5,
                3.5,
                4.5,
                5.5,
                2.5,
                3.5,
                4.5,
                5.5,
                6.5,
                3.5,
                4.5,
                5.5,
                6.5,
                7.5,
            ],
        )

    @unittest.skipUnless(
        _KERNEL_BO_DEPS_AVAILABLE,
        "Optional Bayesian optimisation dependencies are not installed",
    )
    def test_kernel_bayesian_optimisation_example_reuses_one_kernel_entry(self):
        fragment = self.create(KernelBayesianOptimisationFragment, [])
        request = make_kernel_bo_request(
            fragment,
            fit_steps=4,
            fit_lr=0.08,
            acquisition_num_starts=1,
            surrogate_num_starts=1,
            batch_mc_samples=8,
            batch_acq_lr=0.08,
            batch_acq_steps=4,
            max_batches=3,
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        self.assertEqual(result.runtime_stats.executor_entry_count, 1)
        self.assertEqual(result.runtime_stats.batch_count, 5)
        self.assertEqual(len(result.values[fragment.cost]), 5)

        scheduler = fragment.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        prefix = f"ndscan.rid_{rid}.site.root."

        self.assertEqual(fragment.get_dataset(prefix + "state.num_points"), 5)
        self.assertEqual(len(fragment.get_dataset(prefix + "points.param_0")), 5)
        self.assertEqual(len(fragment.get_dataset(prefix + "points.channel_0")), 5)


@unique
class Colors(Enum):
    red = "a bright red"
    blue = "a deep blue"


@unique
class Numbers(Enum):
    one = 1
    two = 2


class SmorgasbordKernelFragment(ExpFragment):
    """Contains a mix of different parameter types to exercise kernel-side parameter
    handling with multiple different scanned types.

    For good measure, we call changed_after_use() on each type as well, and test that
    we can assign the values to member variables initialised with the host-side
    parameter type before.
    """

    def build_fragment(self) -> None:
        self.setattr_param("float", FloatParam, "Float", default=0.1)
        self.setattr_param("int", IntParam, "Int", default=42)
        self.setattr_param("string", StringParam, "String", default="'foo'")
        self.setattr_param("bool", BoolParam, "Bool", default=True)
        self.setattr_param("color", EnumParam, "Color", default=Colors.red)
        self.setattr_param("number", EnumParam, "Number", default=Numbers.one)

        self.setattr_result("float_result", FloatChannel)
        self.setattr_result("int_result", IntChannel)
        self.setattr_result("string_result", OpaqueChannel)
        self.setattr_result("bool_result", OpaqueChannel)
        self.setattr_result("color_result", OpaqueChannel)
        self.setattr_result("number_result", OpaqueChannel)

    def host_setup(self):
        self.float_val = self.float.get()
        self.int_val = self.int.get()
        self.string_val = self.string.get()
        self.bool_val = self.bool.get()
        self.color_val = self.color.get()
        self.number_val = self.number.get()

    @kernel
    def device_setup(self) -> None:
        if self.float.changed_after_use():
            self.float_val = self.float.use()
        if self.int.changed_after_use():
            self.int_val = self.int.use()
        if self.string.changed_after_use():
            self.string_val = self.string.use()
        if self.bool.changed_after_use():
            self.bool_val = self.bool.use()
        if self.color.changed_after_use():
            self.color_val = self.color.use()
        if self.number.changed_after_use():
            self.number_val = self.number.use()

    @kernel
    def run_once(self) -> None:
        self.float_result.push(self.float_val)
        self.int_result.push(self.int_val)
        self.string_result.push(self.string_val)
        self.bool_result.push(self.bool_val)
        self.color_result.push(self.color_val.value)
        self.number_result.push(self.number_val.value)


ScanSmorgasbordKernelFragment = make_fragment_scan_exp(SmorgasbordKernelFragment)


@dataclass
class ListScanDef:
    param_name: str
    schema_values: list
    result_values: list


class TestSmorgasbordKernelCase(KernelEmulatorCase):
    def test_smorgasbord_scan(self):
        exp = self.create(ScanSmorgasbordKernelFragment)
        scan_defs = [
            ListScanDef("float", [0.1, 0.2], [0.1, 0.2]),
            ListScanDef("int", [0, 1], [0, 1]),
            # FIXME: String scans currently cause memory corruption during attribute
            # writeback (unavoidable as escaping an array, not caught due to
            # https://github.com/m-labs/artiq/issues/1394).
            # ListScanDef("string", ["'foo'", "'bar'"], ["foo", "bar"]),
            ListScanDef("bool", [True, False], [True, False]),
            ListScanDef("color", ["red", "blue"], ["a bright red", "a deep blue"]),
            ListScanDef("number", ["one", "two"], [1, 2]),
        ]
        fragment_fqn = "test_experiment_kernel.SmorgasbordKernelFragment"

        def fqn(name):
            return f"{fragment_fqn}.{name}"

        for scan_def in scan_defs:
            exp.args._params["scan"]["axes"].append(
                {
                    "fqn": fqn(scan_def.param_name),
                    "path": "*",
                    "type": "list",
                    "range": {
                        "values": scan_def.schema_values,
                        "randomise_order": True,
                    },
                }
            )
        exp.prepare()
        exp.run()

        def d(key):
            return self.dataset_db.get("ndscan.rid_0." + key)

        self.assertEqual(d(SCHEMA_REVISION_KEY), SCHEMA_REVISION)
        self.assertEqual(d("completed"), True)
        self.assertEqual(d("fragment_fqn"), fragment_fqn)
        self.assertEqual(d("source_id"), "rid_0")
        num_points = math.prod(len(scan_def.schema_values) for scan_def in scan_defs)
        for i, scan_def in enumerate(scan_defs):
            num_repeats = num_points // len(scan_def.schema_values)
            self.assertEqual(
                Counter(d(f"points.axis_{i}")),
                Counter({k: num_repeats for k in scan_def.schema_values}),
            )
            self.assertEqual(
                Counter(d(f"points.channel_{scan_def.param_name}_result")),
                Counter({k: num_repeats for k in scan_def.result_values}),
            )


# # #


class OneTrivial(AggregateExpFragment):
    def build_fragment(self):
        return super().build_fragment(
            [self.setattr_fragment("a", TrivialKernelFragment)]
        )


class TwoTrivial(AggregateExpFragment):
    def build_fragment(self):
        return super().build_fragment(
            [
                self.setattr_fragment("a", TrivialKernelFragment),
                self.setattr_fragment("b", TrivialKernelFragment),
            ]
        )


class TwoPlusOneTrivial(AggregateExpFragment):
    def build_fragment(self):
        return super().build_fragment(
            [
                self.setattr_fragment("a", TwoTrivial),
                self.setattr_fragment("b", OneTrivial),
            ]
        )


TwoPlusOneTrivialScan = make_fragment_scan_exp(TwoPlusOneTrivial)


class TestAggregateCase(KernelEmulatorCase):
    def test_two_plus_one(self):
        exp = self.create(TwoPlusOneTrivialScan)
        exp.prepare()
        exp.run()


# # #


class Inner(ExpFragment):
    def build_fragment(self) -> None:
        self.setattr_device("core")
        self.setattr_param("param_float", FloatParam, "Float param", default=0.0)
        self.setattr_param("param_int", IntParam, "Int param", default=0.0)
        self.setattr_result("result", FloatChannel)

    @kernel
    def run_once(self) -> None:
        self.result.push(self.param_float.get() + self.param_int.get())


INT_GEN = ListGenerator([-1, 0, 1], randomise_order=False)
FLOAT_GEN = LinearGenerator(-1.0, 1.0, 11, randomise_order=False)


class FloatSetattrSubscan(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("inner", Inner)
        setattr_subscan(self, "float_scan", self.inner, [(self.inner, "param_float")])

    def host_setup(self):
        self.float_scan.set_scan_spec([(self.inner.param_float, FLOAT_GEN)])
        super().host_setup()

    @kernel
    def run_once(self) -> None:
        self.float_scan.acquire()


class IntSetattrSubscan(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("inner", Inner)
        setattr_subscan(self, "int_scan", self.inner, [(self.inner, "param_int")])

    def host_setup(self):
        self.int_scan.set_scan_spec([(self.inner.param_int, INT_GEN)])

    @kernel
    def run_once(self) -> None:
        self.int_scan.acquire()


class SetattrParent(AggregateExpFragment):
    def build_fragment(self) -> None:
        self.setattr_fragment("int_frag", IntSetattrSubscan)
        self.setattr_fragment("float_frag", FloatSetattrSubscan)
        super().build_fragment([self.int_frag, self.float_frag])


SetattrParentScan = make_fragment_scan_exp(SetattrParent)


class FloatSubscanExpFragment(SubscanExpFragment):
    pass


class FloatFragmentSubscan(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("inner", Inner)
        self.setattr_fragment(
            "scan",
            FloatSubscanExpFragment,
            self,
            self.inner,
            [(self.inner, "param_float")],
        )

    def host_setup(self):
        self.scan.configure([(self.inner.param_float, FLOAT_GEN)])
        super().host_setup()

    @kernel
    def run_once(self) -> None:
        self.scan.run_once()


class IntSubscanExpFragment(SubscanExpFragment):
    pass


class IntFragmentSubscan(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("inner", Inner)
        self.setattr_fragment(
            "scan",
            IntSubscanExpFragment,
            self,
            self.inner,
            [(self.inner, "param_int")],
        )

    def host_setup(self):
        self.scan.configure([(self.inner.param_int, INT_GEN)])

    @kernel
    def run_once(self) -> None:
        self.scan.run_once()


class FragmentSubscanParent(AggregateExpFragment):
    def build_fragment(self) -> None:
        self.setattr_fragment("int_frag", IntFragmentSubscan)
        self.setattr_fragment("float_frag", FloatFragmentSubscan)
        super().build_fragment([self.int_frag, self.float_frag])


FragmentSubscanParentScan = make_fragment_scan_exp(FragmentSubscanParent)


class FloatSubclassSubscan(SubscanExpFragment):
    def build_fragment(self):
        self.setattr_fragment("inner", Inner)
        super().build_fragment(
            self,
            self.inner,
            [(self.inner, "param_float")],
        )

    def host_setup(self):
        self.configure([(self.inner.param_float, FLOAT_GEN)])
        super().host_setup()


class IntSubclassSubscan(SubscanExpFragment):
    def build_fragment(self):
        self.setattr_fragment("inner", Inner)
        super().build_fragment(
            self,
            self.inner,
            [(self.inner, "param_int")],
        )

    def host_setup(self):
        self.configure([(self.inner.param_int, INT_GEN)])
        super().host_setup()


class SubclassSubscanParent(AggregateExpFragment):
    def build_fragment(self) -> None:
        self.setattr_fragment("int_frag", IntSubclassSubscan)
        self.setattr_fragment("float_frag", FloatSubclassSubscan)
        super().build_fragment([self.int_frag, self.float_frag])


SubclassSubscanParentScan = make_fragment_scan_exp(SubclassSubscanParent)


class TestSubscanKernelCase(KernelEmulatorCase):
    def _test_subscan(self, cls, fragment_fqn, float_channel_name, int_channel_name):
        exp = self.create(cls)
        exp.prepare()
        exp.run()

        def d(key):
            return self.dataset_db.get("ndscan.rid_0." + key)

        self.assertEqual(d(SCHEMA_REVISION_KEY), SCHEMA_REVISION)
        self.assertEqual(d("completed"), True)
        self.assertEqual(d("fragment_fqn"), fragment_fqn)
        self.assertEqual(d("source_id"), "rid_0")

        np.testing.assert_array_max_ulp(
            d(f"point.{float_channel_name}"), FLOAT_GEN.points_for_level(0)
        )
        np.testing.assert_array_max_ulp(
            d(f"point.{int_channel_name}"), INT_GEN.points_for_level(0)
        )

    def test_setattr_subscan(self):
        self._test_subscan(
            SetattrParentScan,
            "test_experiment_kernel.SetattrParent",
            "float_scan_channel_result",
            "int_scan_channel_result",
        )

    def test_fragment_subscan(self):
        self._test_subscan(
            FragmentSubscanParentScan,
            "test_experiment_kernel.FragmentSubscanParent",
            "float_frag_scan__channel_result",
            "int_frag_scan__channel_result",
        )

    def test_subclass_subscan(self):
        self._test_subscan(
            SubclassSubscanParentScan,
            "test_experiment_kernel.SubclassSubscanParent",
            "float_frag__channel_result",
            "int_frag__channel_result",
        )


# # #


class KernelAddOneFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("value", FloatParam, "Value to return", 0.0)
        self.setattr_result("result", FloatChannel)

        self.num_prepare_calls = 0
        self.num_host_setup_calls = 0
        self.num_device_setup_calls = 0
        self.num_host_cleanup_calls = 0
        self.num_device_cleanup_calls = 0

    def prepare(self):
        self.num_prepare_calls += 1

    def host_setup(self):
        self.num_host_setup_calls += 1

    @kernel
    def device_setup(self):
        self.num_device_setup_calls += 1

    def host_cleanup(self):
        self.num_host_cleanup_calls += 1

    @kernel
    def device_cleanup(self):
        self.num_device_cleanup_calls += 1

    @kernel
    def run_once(self):
        self.result.push(self.value.get() + 1)


KernelAddOneFragmentScan = make_fragment_scan_exp(KernelAddOneFragment)


class KernelAddOneFragmentSubscan(SubscanExpFragment):
    def build_fragment(self):
        self.setattr_fragment("add_one", KernelAddOneFragment)
        self.setattr_param(
            "num_scan_points", IntParam, "Number of subscan points", default=3
        )
        super().build_fragment(self, self.add_one, [(self.add_one, "value")])

    @rpc(flags={"async"})
    def configure_scan(self):
        if self.num_scan_points.changed_after_use():
            self.configure(
                [
                    (
                        self.add_one.value,
                        LinearGenerator(
                            0.0,
                            1.0,
                            num_points=self.num_scan_points.use(),
                            randomise_order=True,
                        ),
                    )
                ]
            )

    def host_setup(self):
        # Run at least once before kernel starts such that all the fields
        # are initialised (required for the ARTIQ compiler).
        self.configure_scan()
        super().host_setup()

    @kernel
    def device_setup(self):
        # Update scan if num_scan_points was changed (can be left out if
        # there are no scannable parameters influencing the scan settings).
        self.configure_scan()
        self.device_setup_subfragments()


KernelAddOneFragmentSubscanScan = make_fragment_scan_exp(KernelAddOneFragmentSubscan)


class TestLifetimeCountsCase(KernelEmulatorCase):
    def test_direct_counts(self):
        exp = self.create(KernelAddOneFragmentScan)
        values = [0.0, 1.0, 2.0]
        fragment_fqn = "test_experiment_kernel.KernelAddOneFragment"

        exp.args._params["scan"]["axes"].append(
            {
                "fqn": f"{fragment_fqn}.value",
                "path": "*",
                "type": "list",
                "range": {
                    "values": values,
                    "randomise_order": True,
                },
            }
        )

        exp.prepare()
        exp.run()

        f: KernelAddOneFragment = exp.fragment

        self.assertEqual(f.num_prepare_calls, 1)
        self.assertEqual(f.num_host_setup_calls, 1)
        self.assertEqual(f.num_device_setup_calls, len(values))
        self.assertEqual(f.num_device_cleanup_calls, 1)
        self.assertEqual(f.num_host_cleanup_calls, 1)

    def test_subscan_counts(self):
        exp = self.create(KernelAddOneFragmentSubscanScan)
        exp.prepare()
        exp.run()

        f: KernelAddOneFragment = exp.fragment.add_one

        # FIXME: prepare() is currently not forwarded (but probably should be?)
        # self.assertEqual(f.num_prepare_calls, 1)
        self.assertEqual(f.num_host_setup_calls, 1)
        self.assertEqual(f.num_device_setup_calls, 3)
        self.assertEqual(f.num_device_cleanup_calls, 1)
        self.assertEqual(f.num_host_cleanup_calls, 1)

    def test_subscan_scan_counts(self):
        exp = self.create(KernelAddOneFragmentSubscanScan)
        fragment_fqn = "test_experiment_kernel.KernelAddOneFragmentSubscan"

        num_pointss = [2, 3, 4]
        exp.args._params["scan"]["axes"].append(
            {
                "fqn": f"{fragment_fqn}.num_scan_points",
                "path": "*",
                "type": "list",
                "range": {
                    "values": num_pointss,
                    "randomise_order": True,
                },
            }
        )

        exp.prepare()
        exp.run()

        f: KernelAddOneFragment = exp.fragment.add_one
        # FIXME: prepare() is currently not forwarded (but probably should be?)
        # self.assertEqual(f.num_prepare_calls, 1)
        self.assertEqual(f.num_host_setup_calls, 1)
        self.assertEqual(f.num_device_setup_calls, sum(num_pointss))
        self.assertEqual(f.num_device_cleanup_calls, 1)
        self.assertEqual(f.num_host_cleanup_calls, 1)


# # #


class KernelMultiPointTransitoryErrorFragment(ExpFragment):
    """TransitoryErrorFragment that resets counters after run_once() has completed
    successfully (for testing scan behaviour).

    fail_at_point is invoked with a continuously increasing point index to determine
    whether the current point should require the configured number of retries or succeed
    immediately.
    """

    def build_fragment(
        self,
        num_device_setup_to_fail=0,
        num_device_setup_to_restart_fail=0,
        num_run_once_to_fail=0,
        num_run_once_to_restart_fail=0,
        fail_at_point=lambda point_idx: True,
    ):
        self.setattr_device("core")

        self.fail_at_point = fail_at_point
        self.orig_num_device_setup_to_fail = num_device_setup_to_fail
        self.orig_num_device_setup_to_restart_fail = num_device_setup_to_restart_fail
        self.orig_num_run_once_to_fail = num_run_once_to_fail
        self.orig_num_run_once_to_restart_fail = num_run_once_to_restart_fail
        self.point_idx = 0
        self.reset_counters()

        self.setattr_param("value", IntParam, "Value to produce", default=42)
        self.setattr_result("result", IntChannel)

    @portable
    def reset_counters(self):
        if self.fail_at_point(self.point_idx):
            self.num_device_setup_to_fail = self.orig_num_device_setup_to_fail
            self.num_device_setup_to_restart_fail = (
                self.orig_num_device_setup_to_restart_fail
            )
            self.num_run_once_to_fail = self.orig_num_run_once_to_fail
            self.num_run_once_to_restart_fail = self.orig_num_run_once_to_restart_fail
        else:
            self.num_device_setup_to_fail = 0
            self.num_device_setup_to_restart_fail = 0
            self.num_run_once_to_fail = 0
            self.num_run_once_to_restart_fail = 0

    @kernel
    def device_setup(self):
        if self.num_device_setup_to_restart_fail > 0:
            self.num_device_setup_to_restart_fail -= 1
            raise RestartKernelTransitoryError
        if self.num_device_setup_to_fail > 0:
            self.num_device_setup_to_fail -= 1
            raise TransitoryError

    @kernel
    def run_once(self):
        if self.num_run_once_to_restart_fail > 0:
            self.num_run_once_to_restart_fail -= 1
            raise RestartKernelTransitoryError
        if self.num_run_once_to_fail > 0:
            self.num_run_once_to_fail -= 1
            raise TransitoryError
        self.result.push(self.value.get())
        self.point_idx += 1
        self.reset_counters()


@unique
class ConfigureMode(Enum):
    once_in_build = 0
    every_host_setup = 1


class KernelTransitoryErrorSubscan(SubscanExpFragment):
    def build_fragment(self, configure_mode: ConfigureMode, **kwargs):
        self.configure_mode = configure_mode
        self.setattr_fragment("frag", KernelMultiPointTransitoryErrorFragment, **kwargs)
        super().build_fragment(self, self.frag, [(self.frag, "value")])
        self.axis_generators = [
            (self.frag.value, LinearGenerator(0, 10, 11, randomise_order=True))
        ]
        if self.configure_mode == ConfigureMode.once_in_build:
            self.configure(self.axis_generators)

    def host_setup(self):
        if self.configure_mode == ConfigureMode.every_host_setup:
            self.configure(self.axis_generators)
        super().host_setup()


def _fail_every(count):
    def result(i) -> bool:
        return bool(i % count == 1)

    return result


class KernelTransitoryErrorSubscanCase(KernelEmulatorCase):
    def _test_with_kwargs(self, configure_mode: ConfigureMode, **kwargs):
        subscan = self.create(
            KernelTransitoryErrorSubscan,
            [],
            configure_mode,
            **kwargs,
        )
        results = run_fragment_once(subscan)
        inputs = results[subscan._axis_0]
        outputs = results[subscan._channel_result]
        np.testing.assert_array_equal(np.sort(inputs), np.arange(11))
        np.testing.assert_array_equal(inputs, outputs)

    def test_nominal(self):
        def never(i) -> bool:
            return False

        self._test_with_kwargs(ConfigureMode.once_in_build, fail_at_point=never)

    def test_transitory_setup_once(self):
        self._test_with_kwargs(
            ConfigureMode.once_in_build,
            num_device_setup_to_fail=2,
            fail_at_point=_fail_every(3),
        )

    def test_transitory_run_once(self):
        self._test_with_kwargs(
            ConfigureMode.once_in_build,
            num_run_once_to_fail=2,
            fail_at_point=_fail_every(3),
        )

    def test_restart_transitory_setup_once(self):
        self._test_with_kwargs(
            ConfigureMode.once_in_build,
            num_device_setup_to_restart_fail=2,
            fail_at_point=_fail_every(3),
        )

    def test_restart_transitory_run_once(self):
        self._test_with_kwargs(
            ConfigureMode.once_in_build,
            num_run_once_to_restart_fail=2,
            fail_at_point=_fail_every(3),
        )

    def test_transitory_setup_every(self):
        self._test_with_kwargs(
            ConfigureMode.every_host_setup,
            num_device_setup_to_fail=2,
            fail_at_point=_fail_every(3),
        )

    def test_transitory_run_every(self):
        self._test_with_kwargs(
            ConfigureMode.every_host_setup,
            num_run_once_to_fail=2,
            fail_at_point=_fail_every(3),
        )

    def test_restart_transitory_setup_every(self):
        self._test_with_kwargs(
            ConfigureMode.every_host_setup,
            num_device_setup_to_restart_fail=2,
            fail_at_point=_fail_every(12),
        )

    def test_restart_transitory_run_every(self):
        self._test_with_kwargs(
            ConfigureMode.every_host_setup,
            num_run_once_to_restart_fail=2,
            fail_at_point=_fail_every(12),
        )

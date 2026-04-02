"""Tests for the new host-only runtime."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import ndscan.experiment as experiment_facade
from artiq.experiment import kernel
from artiq.language.core import TerminationRequested
from examples.host_runtime_grouped_line_family import HostRuntimeGroupedLineFamily
import examples.host_runtime_counts_field_calibration as counts_field_calibration
import examples.host_runtime_interleaved_spectroscopy_patterns as interleaved_patterns
from examples.host_runtime_field_shift_spectroscopy import (
    HostRuntimeFieldShiftSpectroscopy,
    TRUE_CENTER_AT_ZERO,
    TRUE_SHIFT_PER_FIELD,
)
from examples.host_runtime_prepared_root_linear_scan import (
    HostRuntimePreparedRootLinearScan,
)
from examples.host_runtime_rabi_flop_2d import HostRuntimeRabiFlop2D
from mock_environment import HasEnvironmentCase
from sipyco import pyon

from ndscan.dashboard.submission import HostSubmissionBackend, select_submission_backend
from ndscan.define import annotations
from ndscan.define.default_analysis import AnalysisFeedback, CustomAnalysis, OnlineFit
from ndscan.define.fragment import ExpFragment, RestartKernelTransitoryError
from ndscan.define.parameters import FloatParam
from ndscan.define.result_channels import FloatChannel, IntChannel
from ndscan.fits import (
    artifact_parameter_value,
    build_builtin_model,
    fit_data_with_model,
    import_sensible_fitting,
)
from ndscan.runtime.api import (
    PointObservation,
    PreparedScan,
    make_child_scan_site,
    make_fragment_prepared_dashboard_scan_exp,
    make_fragment_prepared_scan_exp,
    prepare_child_scan,
    setattr_prepared_child_scan,
)
from ndscan.scan.mapping import ParameterMapping, ScanVariable
from ndscan.scan.point_policy import (
    BasePoint,
    BatchFeedback,
    CartesianPointPolicy,
    ConcatPointPolicy,
    ExplicitPointPolicy,
    GradientDescentPointPolicy,
    PointPolicy,
    ProductPointPolicy,
    RecursiveMidpointPointPolicy1D,
    RepeatPointPolicy,
    ShuffledPointPolicy,
    SinglePointPolicy,
    UntilConditionPointPolicy,
    ZipPointPolicy,
)
from ndscan.scan.request import ExecutionPolicy, PreviewPolicy, ScanRequest
from ndscan.schema.scan_site import ScanSite
from ndscan.submission.host_scan_schema import (
    HostScanSpec,
    compile_host_scan_schema,
    compile_host_scan_spec,
)
from ndscan.utils import PARAMS_ARG_KEY
from ndscan.utils import FIT_OBJECTS

_REAL_NP_LINSPACE = np.linspace


def _execute_and_inspect(scan):
    scan.execute()
    return scan.inspect()


class DictShimFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_result("value", FloatChannel)

    def run_once(self):
        self.value.push(1.23)


class GridSchemaFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_param("y", FloatParam, "y", default=0.0)
        self.setattr_param("z", FloatParam, "z", default=0.0)
        self.setattr_result("value", FloatChannel)

    def run_once(self):
        self.value.push(self.x.get() + self.y.get() + self.z.get())


class PointPolicyTest(unittest.TestCase):
    def test_cartesian_point_policy(self):
        source = CartesianPointPolicy([[0, 1], [10, 20]])
        self.assertEqual(
            [point.axis_values for point in source],
            [(0, 10), (0, 20), (1, 10), (1, 20)],
        )

    def test_zip_point_policy(self):
        source = ZipPointPolicy([[0, 1, 2], [10, 11, 12]])
        self.assertEqual(
            [point.axis_values for point in source],
            [(0, 10), (1, 11), (2, 12)],
        )

    def test_explicit_point_policy(self):
        source = ExplicitPointPolicy(2, [(0, 10), (3, 13)])
        self.assertEqual(
            [point.axis_values for point in source],
            [(0, 10), (3, 13)],
        )

    def test_explicit_point_policy_next_batch(self):
        source = ExplicitPointPolicy(2, [(0, 10), (3, 13), (4, 14)])
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

    def test_shuffled_point_policy_randomises_materialised_order_deterministically(self):
        source = ShuffledPointPolicy(
            CartesianPointPolicy([[0, 1], [10, 20]]),
            random_seed=123,
        )
        self.assertEqual(
            [point.axis_values for point in source],
            [(1, 10), (1, 20), (0, 20), (0, 10)],
        )

    def test_concat_point_policy_runs_children_in_sequence(self):
        source = ConcatPointPolicy(
            [
                ExplicitPointPolicy(1, [(0,), (1,)]),
                ExplicitPointPolicy(1, [(10,), (11,)]),
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

    def test_product_point_policy_combines_child_axes(self):
        source = ProductPointPolicy(
            [
                ExplicitPointPolicy(1, [(0,), (1,)]),
                ExplicitPointPolicy(2, [(10, 100), (20, 200)]),
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

    def test_recursive_midpoint_point_policy_refines_breadth_first(self):
        source = RecursiveMidpointPointPolicy1D(0.0, 8.0, max_depth=3)
        self.assertEqual(
            [point.axis_values for point in source],
            [(0.0,), (8.0,), (4.0,), (2.0,), (6.0,), (1.0,), (3.0,), (5.0,), (7.0,)],
        )

    def test_until_condition_point_policy_stops_after_predicate_matches(self):
        source = UntilConditionPointPolicy(
            ExplicitPointPolicy(1, [(0,), (1,), (2,), (3,)]),
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

    def test_until_condition_point_policy_can_stop_from_batch_feedback(self):
        source = UntilConditionPointPolicy(
            ExplicitPointPolicy(1, [(0,), (1,), (2,), (3,)]),
            lambda feedback: feedback.online_analysis_results["fit"]["error"] < 0.1,
            predicate_description="fit error < 0.1",
            min_observations=2,
            per_batch=True,
        )

        batch = source.next_batch(2)
        self.assertEqual([point.axis_values for point in batch], [(0,), (1,)])

        source.observe_batch(
            BatchFeedback(
                observations=(
                    PointObservation(
                        point_index=0,
                        axis_values={"axis_0": 0},
                        channel_values={"channel_0": 10},
                    ),
                    PointObservation(
                        point_index=1,
                        axis_values={"axis_0": 1},
                        channel_values={"channel_0": 20},
                    ),
                ),
                online_analyses={"fit": AnalysisFeedback(outputs={"error": 0.05})},
            )
        )
        self.assertTrue(source.is_finished())

    def test_repeat_point_policy_repeats_each_logical_point_fixed_times(self):
        source = RepeatPointPolicy(
            ExplicitPointPolicy(1, [(0,), (1,)]),
            repeats=3,
        )

        emitted = []
        while not source.is_finished():
            batch = source.next_batch(2)
            emitted.extend(point.axis_values for point in batch)
            source.observe_batch(
                BatchFeedback(
                    observations=tuple(
                        PointObservation(
                            point_index=point.index,
                            axis_values={"axis_0": point.axis_values[0]},
                            channel_values={"channel_0": point.axis_values[0]},
                        )
                        for point in batch
                    ),
                    axis_data={"axis_0": tuple(point.axis_values[0] for point in batch)},
                    result_data={
                        "channel_0": tuple(point.axis_values[0] for point in batch)
                    },
                )
            )

        self.assertEqual(
            emitted,
            [(0,), (0,), (0,), (1,), (1,), (1,)],
        )

    def test_repeat_point_policy_can_interleave_fixed_repeats_over_finite_points(self):
        source = RepeatPointPolicy(
            ExplicitPointPolicy(1, [(0,), (1,)]),
            repeats=3,
            schedule="interleaved",
        )

        emitted = []
        while not source.is_finished():
            batch = source.next_batch(2)
            emitted.extend(point.axis_values for point in batch)
            source.observe_batch(
                BatchFeedback(
                    observations=tuple(
                        PointObservation(
                            point_index=point.index,
                            axis_values={"axis_0": point.axis_values[0]},
                            channel_values={"channel_0": point.axis_values[0]},
                        )
                        for point in batch
                    ),
                    axis_data={"axis_0": tuple(point.axis_values[0] for point in batch)},
                    result_data={
                        "channel_0": tuple(point.axis_values[0] for point in batch)
                    },
                )
            )

        self.assertEqual(
            emitted,
            [(0,), (1,), (0,), (1,), (0,), (1,)],
        )


    def test_repeat_point_policy_can_stop_current_point_from_batch_feedback(self):
        source = RepeatPointPolicy(
            ExplicitPointPolicy(1, [(5.0,)]),
            stop_predicate=lambda feedback: len(feedback.result_data["channel_0"]) >= 3,
            min_repeats=1,
            max_repeats=5,
            predicate_description="three repeats seen",
        )

        num_observations = 0
        while not source.is_finished():
            batch = source.next_batch(1)
            num_observations += len(batch)
            source.observe_batch(
                BatchFeedback(
                    observations=tuple(
                        PointObservation(
                            point_index=point.index,
                            axis_values={"axis_0": point.axis_values[0]},
                            channel_values={"channel_0": 1.0},
                        )
                        for point in batch
                    ),
                    axis_data={"axis_0": (5.0,) * num_observations},
                    result_data={"channel_0": (1.0,) * num_observations},
                )
            )

        self.assertEqual(num_observations, 3)

    def test_repeat_point_policy_interleaved_stop_predicate_stays_local_to_one_point(self):
        source = RepeatPointPolicy(
            ExplicitPointPolicy(1, [(5.0,), (9.0,)]),
            stop_predicate=lambda feedback: len(feedback.result_data["channel_0"]) >= 2,
            min_repeats=1,
            max_repeats=4,
            predicate_description="two repeats seen",
            schedule="interleaved",
        )

        emitted = []
        while not source.is_finished():
            batch = source.next_batch(2)
            emitted.extend(point.axis_values for point in batch)
            source.observe_batch(
                BatchFeedback(
                    observations=tuple(
                        PointObservation(
                            point_index=point.index,
                            axis_values={"axis_0": point.axis_values[0]},
                            channel_values={"channel_0": 1.0},
                        )
                        for point in batch
                    ),
                    axis_data={"axis_0": tuple(point.axis_values[0] for point in batch)},
                    result_data={"channel_0": (1.0,) * len(batch)},
                )
            )

        self.assertEqual(emitted, [(5.0,), (9.0,), (5.0,), (9.0,)])


class HostScanSchemaCompilationTest(HasEnvironmentCase):
    def test_host_scan_base_class_is_not_exported_via_star_imports(self):
        self.assertNotIn("PreparedScanExperiment", experiment_facade.__all__)

    def test_host_scan_spec_round_trips_through_dict_transport(self):
        spec = HostScanSpec.from_dict(
            {
                "version": 1,
                "mode": {"type": "grid"},
                "entries": [
                    {
                        "id": "x",
                        "kind": "pseudoparam",
                        "mode": {
                            "type": "scan",
                            "generator": {
                                "type": "list",
                                "range": {
                                    "values": [1.0, 2.0],
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
                ],
                "execution": {"max_points_per_batch": 8},
                "metadata": {"demo_name": "round_trip"},
            }
        )

        self.assertEqual(HostScanSpec.from_dict(spec.to_dict()), spec)

    def test_compile_grid_schema_without_scan_entries_returns_single_request(self):
        fragment = self.create(DictShimFragment, [])
        request, overrides = compile_host_scan_schema(
            fragment,
            {
                "version": 1,
                "mode": {"type": "grid"},
                "entries": [],
                "execution": {},
                "metadata": {"demo_name": "dict_single"},
            },
        )

        self.assertEqual(request.axes, ())
        self.assertEqual(request.point_policy.describe()["kind"], "single")
        self.assertEqual(request.metadata["demo_name"], "dict_single")
        self.assertEqual(overrides, {})

    def test_compile_host_scan_spec_builds_same_grid_request_as_dict_path(self):
        fragment = self.create(GridSchemaFragment, [])
        spec = HostScanSpec.from_dict(
            {
                "version": 1,
                "mode": {"type": "grid"},
                "entries": [
                    {
                        "id": "x",
                        "kind": "param",
                        "target": {"fqn": fragment.x.parameter.fqn, "path": "*"},
                        "mode": {
                            "type": "scan",
                            "generator": {
                                "type": "list",
                                "range": {
                                    "values": [1.0, 2.0],
                                    "randomise_order": False,
                                },
                            },
                            "group": "pair",
                        },
                    },
                    {
                        "id": "y",
                        "kind": "param",
                        "target": {"fqn": fragment.y.parameter.fqn, "path": "*"},
                        "mode": {
                            "type": "scan",
                            "generator": {
                                "type": "list",
                                "range": {
                                    "values": [10.0, 11.0, 12.0],
                                    "randomise_order": False,
                                },
                            },
                            "group": "pair",
                        },
                    },
                    {
                        "id": "z",
                        "kind": "param",
                        "target": {"fqn": fragment.z.parameter.fqn, "path": "*"},
                        "mode": {
                            "type": "scan",
                            "generator": {
                                "type": "list",
                                "range": {
                                    "values": [100.0, 200.0],
                                    "randomise_order": False,
                                },
                            },
                        },
                    },
                ],
                "execution": {},
            }
        )

        request, overrides = compile_host_scan_spec(fragment, spec)

        self.assertEqual(overrides, {})
        self.assertEqual(
            [point.axis_values for point in request.point_policy],
            [
                (1.0, 10.0, 100.0),
                (1.0, 10.0, 200.0),
                (2.0, 11.0, 100.0),
                (2.0, 11.0, 200.0),
            ],
        )

    def test_compile_grid_schema_uses_zip_groups_with_shortest_length(self):
        fragment = self.create(GridSchemaFragment, [])
        request, overrides = compile_host_scan_schema(
            fragment,
            {
                "version": 1,
                "mode": {"type": "grid"},
                "entries": [
                    {
                        "id": "x",
                        "kind": "param",
                        "target": {"fqn": fragment.x.parameter.fqn, "path": "*"},
                        "mode": {
                            "type": "scan",
                            "generator": {
                                "type": "list",
                                "range": {
                                    "values": [1.0, 2.0],
                                    "randomise_order": False,
                                },
                            },
                            "group": "pair",
                        },
                    },
                    {
                        "id": "y",
                        "kind": "param",
                        "target": {"fqn": fragment.y.parameter.fqn, "path": "*"},
                        "mode": {
                            "type": "scan",
                            "generator": {
                                "type": "list",
                                "range": {
                                    "values": [10.0, 11.0, 12.0],
                                    "randomise_order": False,
                                },
                            },
                            "group": "pair",
                        },
                    },
                    {
                        "id": "z",
                        "kind": "param",
                        "target": {"fqn": fragment.z.parameter.fqn, "path": "*"},
                        "mode": {
                            "type": "scan",
                            "generator": {
                                "type": "list",
                                "range": {
                                    "values": [100.0, 200.0],
                                    "randomise_order": False,
                                },
                            },
                        },
                    },
                ],
                "execution": {},
            },
        )

        self.assertEqual(overrides, {})
        self.assertEqual(
            [point.axis_values for point in request.point_policy],
            [
                (1.0, 10.0, 100.0),
                (1.0, 10.0, 200.0),
                (2.0, 11.0, 100.0),
                (2.0, 11.0, 200.0),
            ],
        )

    def test_compile_grid_schema_supports_rebind_expressions(self):
        fragment = self.create(PhysicalDriveFragment, [])
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
                "execution": {},
            },
        )

        self.assertEqual(overrides, {})
        self.assertEqual(request.axes[0].name, "logical_drive")
        self.assertEqual(len(request.parameter_mappings), 1)
        self.assertEqual(len(request.fixed_pseudoparams), 1)
        self.assertEqual(request.fixed_pseudoparams[0].name, "offset")
        self.assertEqual(request.fixed_pseudoparams[0].value, 0.5)
        self.assertEqual(
            request.parameter_mappings[0].expression,
            "logical_drive + offset",
        )

    @patch("ndscan.submission.host_scan_schema.random.getrandbits", return_value=123)
    def test_compile_grid_schema_can_randomise_order_globally(self, _seed):
        fragment = self.create(GridSchemaFragment, [])
        request, overrides = compile_host_scan_schema(
            fragment,
            {
                "version": 1,
                "mode": {"type": "grid", "randomise_order_globally": True},
                "entries": [
                    {
                        "id": "x",
                        "kind": "param",
                        "target": {"fqn": fragment.x.parameter.fqn, "path": "*"},
                        "mode": {
                            "type": "scan",
                            "generator": {
                                "type": "list",
                                "range": {
                                    "values": [0.0, 1.0],
                                    "randomise_order": False,
                                },
                            },
                        },
                    },
                    {
                        "id": "y",
                        "kind": "param",
                        "target": {"fqn": fragment.y.parameter.fqn, "path": "*"},
                        "mode": {
                            "type": "scan",
                            "generator": {
                                "type": "list",
                                "range": {
                                    "values": [10.0, 20.0],
                                    "randomise_order": False,
                                },
                            },
                        },
                    },
                ],
                "execution": {},
            },
        )

        self.assertEqual(overrides, {})
        self.assertEqual(
            [point.axis_values for point in request.point_policy],
            [(1.0, 10.0), (1.0, 20.0), (0.0, 20.0), (0.0, 10.0)],
        )

    def test_make_fragment_prepared_scan_exp_accepts_compiled_schema_tuple(self):
        DictShimExperiment = make_fragment_prepared_scan_exp(
            DictShimFragment,
            lambda fragment: compile_host_scan_schema(
                fragment,
                {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {},
                    "metadata": {"demo_name": "dict_shim"},
                },
            ),
        )

        exp = self.create(DictShimExperiment)
        exp.prepare()
        exp.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.dataset_db.get(prefix + "state.num_points"), 1)
        self.assertEqual(self.dataset_db.get(prefix + "points.channel_0"), [1.23])

    def test_make_fragment_prepared_scan_exp_rejects_dict_schema(self):
        DictShimExperiment = make_fragment_prepared_scan_exp(
            DictShimFragment,
            {
                "version": 1,
                "mode": {"type": "grid"},
                "entries": [],
                "execution": {},
                "metadata": {"demo_name": "dict_shim"},
            },
        )

        exp = self.create(DictShimExperiment)
        with self.assertRaises(TypeError):
            exp.prepare()

    def test_make_fragment_prepared_scan_exp_rejects_host_scan_spec(self):
        DictShimExperiment = make_fragment_prepared_scan_exp(
            DictShimFragment,
            HostScanSpec.from_dict(
                {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {},
                    "metadata": {"demo_name": "typed_shim"},
                }
            ),
        )

        exp = self.create(DictShimExperiment)
        with self.assertRaises(TypeError):
            exp.prepare()


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


class LinearResponseFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("slope", FloatParam, "Slope", default=2.0)
        self.setattr_param("offset", FloatParam, "Offset", default=1.0)
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push(self.slope.get() * self.x.get() + self.offset.get())


class SequentialHitFragment(ExpFragment):
    """Leaf fragment emitting a deterministic sequence of Bernoulli-like shots."""

    def build_fragment(self):
        self.setattr_result("hit", FloatChannel)
        self._shots = [1.0, 0.0, 1.0, 1.0, 0.0]
        self._next_shot = 0

    def run_once(self):
        self.hit.push(self._shots[self._next_shot])
        self._next_shot += 1


class PhysicalDriveFragment(ExpFragment):
    """Hardware-facing fragment used to test runtime parameter mappings."""

    def build_fragment(self):
        self.setattr_param("drive", FloatParam, "drive", 0.0)
        self.setattr_result("result", FloatChannel)

    def run_once(self):
        self.result.push(2.0 * self.drive.get())


class WrapperRebindFragment(ExpFragment):
    """Wrapper fragment using ``rebind_param()`` to drive child hardware params.

    The child stays attached to the ordinary fragment lifecycle, so the wrapper only
    needs to invoke ``child.run_once()``. The child result channel itself later appears
    in the scan-site metadata and point data with its deeper path.
    """

    def build_fragment(self):
        self.setattr_fragment("child", PhysicalDriveFragment)
        self.setattr_param("logical_drive", FloatParam, "logical drive", 0.0)
        self.rebind_param(
            self.child.drive,
            [self.logical_drive],
            lambda values: values[self.logical_drive] + 0.5,
            description="Offset the child hardware drive from the logical wrapper axis",
        )

    def run_once(self):
        self.child.run_once()


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


class RecordingBatchPointPolicy(ExplicitPointPolicy):
    """Explicit source that records how the runner consumes batched points."""

    def __init__(
        self,
        axis_count,
        points,
        *,
        preferred_batch_size=None,
        point_metadata=None,
    ):
        super().__init__(axis_count, points)
        self._preferred_batch_size = preferred_batch_size
        self._point_metadata = (
            None if point_metadata is None else [dict(metadata) for metadata in point_metadata]
        )
        self.requested_batch_limits = []
        self.observed_batches = []
        self.observed_online_analysis_results = []
        self.observed_online_analysis_annotations = []
        self.observed_result_lengths = []

    def next_batch(self, max_points: int):
        self.requested_batch_limits.append(max_points)
        batch = super().next_batch(max_points)
        if self._point_metadata is None:
            return batch
        return [
            BasePoint(
                index=point.index,
                axis_values=point.axis_values,
                metadata=self._point_metadata[point.index],
            )
            for point in batch
        ]

    def preferred_batch_size(self, default: int) -> int:
        if self._preferred_batch_size is None:
            return default
        return self._preferred_batch_size

    def observe(self, observation):
        raise AssertionError("Host runtime should call observe_batch() at batch boundaries")

    def observe_batch(self, feedback: BatchFeedback):
        self.observed_batches.append(
            [observation.axis_values["axis_0"] for observation in feedback.observations]
        )
        self.observed_online_analysis_results.append(
            dict(feedback.online_analysis_results)
        )
        self.observed_online_analysis_annotations.append(
            {
                name: list(analysis.annotations)
                for name, analysis in feedback.online_analyses.items()
            }
        )
        self.observed_result_lengths.append(
            {
                channel.path: len(values)
                for channel, values in feedback.result_data.items()
            }
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


class MissingCoreKernelFragment(ExpFragment):
    """Kernel fragment used to pin the required ``self.core`` declaration."""

    def build_fragment(self):
        self.setattr_param("value", FloatParam, "Value", 0.0)
        self.setattr_result("result", FloatChannel)

    @kernel
    def run_once(self):
        self.result.push(self.value.get() + 1.0)


class OnlineGaussianFragment(ExpFragment):
    """Leaf fragment with a built-in online fit and a known exact model."""

    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", 0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        params = {"x0": 1.0, "y0": 0.5, "a": 2.0, "sigma": 1.2}
        self.y.push(FIT_OBJECTS["gaussian"].fitting_function(self.x.get(), params))

    def get_default_analyses(self):
        return [OnlineFit("gaussian", data={"x": self.x, "y": self.y})]


class OnlineAnnotatedFragment(ExpFragment):
    """Fragment with a simple online summary analysis and live annotations."""

    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", 0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push(self.x.get() + 1.0)

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.x],
                self._summarise_running_curve,
                analysis_results=[
                    IntChannel("num_points", "Running point count"),
                    FloatChannel("latest_x", "Latest x value"),
                    FloatChannel("latest_y", "Latest y value"),
                ],
                online_fn=self._summarise_running_curve,
                online_analysis_identifier="running_summary",
            )
        ]

    def _summarise_running_curve(self, axis_values, result_values, analysis_results):
        xs = np.asarray(axis_values[self.x], dtype=float)
        ys = np.asarray(result_values[self.y], dtype=float)
        analysis_results["num_points"].push(int(len(xs)))
        analysis_results["latest_x"].push(float(xs[-1]))
        analysis_results["latest_y"].push(float(ys[-1]))
        return [
            annotations.curve_1d(
                x_axis=self.x,
                x_values=xs,
                y_axis=self.y,
                y_values=ys,
            )
        ]


class FourDimQuadraticFragment(ExpFragment):
    """Convex objective surface used to test the gradient-descent point policy."""

    def build_fragment(self):
        self.setattr_param("x0", FloatParam, "x0", 0.0)
        self.setattr_param("x1", FloatParam, "x1", 0.0)
        self.setattr_param("x2", FloatParam, "x2", 0.0)
        self.setattr_param("x3", FloatParam, "x3", 0.0)
        self.setattr_result("loss", FloatChannel)

    def run_once(self):
        optimum = (1.0, -2.0, 0.5, 3.0)
        values = (
            self.x0.get(),
            self.x1.get(),
            self.x2.get(),
            self.x3.get(),
        )
        self.loss.push(sum((value - target) ** 2 for value, target in zip(values, optimum)))


class NestedChildScanParent(ExpFragment):
    """Parent fragment that launches a prepared child scan during each outer point."""

    def build_fragment(self):
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_fragment("child", PlainAddOneFragment, detached=True)
        self.child_scan = prepare_child_scan(self, self.child, name="child_scan")
        self.setattr_result("child_total", FloatChannel)

    def run_once(self):
        self.child_scan.configure(
            ScanRequest.cartesian(
                [(self.child.value, [self.outer.get(), self.outer.get() + 1.0])]
            )
        )
        self.child_scan.execute()
        child_result = self.child_scan.inspect()
        self.child_total.push(sum(child_result.values[self.child.result]))


class PreparedChildScanParent(ExpFragment):
    """Parent fragment that uses the prepared child-scan API."""

    def build_fragment(self):
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_fragment("child", PlainAddOneFragment, detached=True)
        self.child_scan = prepare_child_scan(self, self.child, name="child_scan")
        self.setattr_result("child_total", FloatChannel)

    def run_once(self):
        self.child_scan.configure(
            ScanRequest.cartesian(
                [(self.child.value, [self.outer.get(), self.outer.get() + 1.0])],
                execution_policy=ExecutionPolicy(max_points_per_batch=2),
            )
        )
        self.child_scan.execute()
        child_result = self.child_scan.inspect()
        self.child_total.push(sum(child_result.values[self.child.result]))


class UnconfiguredPreparedChildScanParent(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("child", PlainAddOneFragment, detached=True)
        self.child_scan = prepare_child_scan(self, self.child, name="child_scan")

    def run_once(self):
        self.child_scan.execute()


class ReusedPreparedChildScanParent(ExpFragment):
    """Parent fragment that reuses one prepared child-scan request twice."""

    def build_fragment(self):
        self.setattr_fragment("child", PlainAddOneFragment, detached=True)
        self.child_scan = prepare_child_scan(self, self.child, name="child_scan")
        self.setattr_result("child_total", FloatChannel)

    def run_once(self):
        self.child_scan.configure(
            ScanRequest.explicit([self.child.value], [[2.0], [3.0]])
        )
        self.child_scan.execute()
        first = self.child_scan.inspect()
        self.child_scan.execute()
        second = self.child_scan.inspect()
        self.child_total.push(
            sum(first.values[self.child.result]) + sum(second.values[self.child.result])
        )


class AutoDetachedPreparedChildScanParent(ExpFragment):
    """Parent that relies on prepare_child_scan() to detach a direct child."""

    def build_fragment(self):
        self.setattr_fragment("child", CountingLifecycleFragment)
        self.child_scan = prepare_child_scan(self, self.child, name="child_scan")
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_result("child_total", FloatChannel)

    def run_once(self):
        self.child_scan.configure(
            ScanRequest.cartesian(
                [(self.child.value, [self.outer.get(), self.outer.get() + 1.0])],
                execution_policy=ExecutionPolicy(max_points_per_batch=2),
            )
        )
        self.child_scan.execute()
        child_result = self.child_scan.inspect()
        self.child_total.push(sum(child_result.values[self.child.result]))


class SetattrPreparedChildScanParent(ExpFragment):
    """Parent using the ergonomic helper that creates and detaches the child."""

    def build_fragment(self):
        self.child_scan = setattr_prepared_child_scan(
            self,
            "child",
            PlainAddOneFragment,
            scan_name="child_scan",
        )
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_result("child_total", FloatChannel)

    def run_once(self):
        self.child_scan.configure(
            ScanRequest.cartesian(
                [(self.child.value, [self.outer.get(), self.outer.get() + 1.0])]
            )
        )
        self.child_scan.execute()
        child_result = self.child_scan.inspect()
        self.child_total.push(sum(child_result.values[self.child.result]))


class PreparedChildFixedOutputParent(ExpFragment):
    """Parent that consumes one fixed child output from a prepared child scan."""

    def build_fragment(self):
        self.child_scan = setattr_prepared_child_scan(
            self,
            "child",
            AnalysedLineFragment,
            scan_name="child_scan",
            expose_outputs=["m"],
        )
        self.setattr_result("child_slope", FloatChannel)

    def run_once(self):
        self.child_scan.configure(
            ScanRequest.explicit(
                [self.child.x],
                [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            )
        )
        self.child_scan.execute()
        (child_slope,) = self.child_scan.get_outputs()
        self.child_slope.push(child_slope)


class PreparedChildTupleOutputParent(ExpFragment):
    """Parent that consumes fixed child outputs via one tuple fetch."""

    def build_fragment(self):
        self.child_scan = setattr_prepared_child_scan(
            self,
            "child",
            AnalysedLineFragment,
            scan_name="child_scan",
            expose_outputs=["m"],
        )
        self.setattr_result("child_slope", FloatChannel)

    def run_once(self):
        self.child_scan.configure(
            ScanRequest.explicit(
                [self.child.x],
                [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            )
        )
        self.child_scan.execute()
        (child_slope,) = self.child_scan.get_outputs()
        self.child_slope.push(child_slope)


class PreparedChildExecuteOutputParent(ExpFragment):
    """Parent that consumes stable child outputs through execute()/outputs()."""

    def build_fragment(self):
        self.child_scan = setattr_prepared_child_scan(
            self,
            "child",
            AnalysedLineFragment,
            scan_name="child_scan",
            expose_outputs=["m"],
        )
        self.setattr_result("child_slope", FloatChannel)

    def run_once(self):
        self.child_scan.configure(
            ScanRequest.explicit(
                [self.child.x],
                [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            )
        )
        outputs = self.child_scan.execute()
        self.child_slope.push(outputs["m"])


class RecursiveLeafScanFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("base", FloatParam, "base", 0.0)
        self.setattr_fragment("leaf", PlainAddOneFragment, detached=True)
        self.leaf_scan = prepare_child_scan(self, self.leaf, name="leaf_scan")
        self.setattr_result("leaf_total", FloatChannel)

    def run_once(self):
        self.leaf_scan.configure(
            ScanRequest.explicit(
                [self.leaf.value],
                [[self.base.get()], [self.base.get() + 0.5]],
            )
        )
        self.leaf_scan.execute()
        leaf_result = self.leaf_scan.inspect()
        self.leaf_total.push(sum(leaf_result.values[self.leaf.result]))


class RecursiveScanParent(ExpFragment):
    """Root fragment that scans a fragment which itself launches nested scans."""

    def build_fragment(self):
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_fragment("middle", RecursiveLeafScanFragment, detached=True)
        self.middle_scan = prepare_child_scan(self, self.middle, name="middle_scan")
        self.setattr_result("middle_total", FloatChannel)

    def run_once(self):
        self.middle_scan.configure(
            ScanRequest.explicit(
                [self.middle.base],
                [[self.outer.get()], [self.outer.get() + 10.0]],
            )
        )
        self.middle_scan.execute()
        middle_result = self.middle_scan.inspect()
        self.middle_total.push(sum(middle_result.values[self.middle.leaf_total]))


class NestedHelperSiteOptionsParent(ExpFragment):
    """Parent fragment used to pin down prepared child site-merging behavior."""

    def build_fragment(self):
        self.setattr_fragment("child", PlainAddOneFragment, detached=True)
        self.child_scan = prepare_child_scan(
            self,
            self.child,
            name="custom_child",
            segmented=False,
            extra_metadata={"from_call": "call"},
        )
        self.setattr_result("child_result", FloatChannel)

    def run_once(self):
        request = ScanRequest.explicit(
            [self.child.value],
            [[2.0]],
            site=ScanSite(
                segmented=False,
                extra_metadata={"from_request": "request"},
            ),
        )
        self.child_scan.configure(request)
        self.child_scan.execute()
        result = self.child_scan.inspect()
        self.child_result.push(result.values[self.child.result][0])


class FixedParameterSubscanLeaf(ExpFragment):
    """Leaf fragment with fixed parameters that still participate in point results."""

    def build_fragment(self):
        self.setattr_param("gain", FloatParam, "gain", 2.0)
        self.setattr_param("offset", FloatParam, "offset", 1.0)
        self.setattr_param("x", FloatParam, "x", 0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push(self.gain.get() * self.x.get() + self.offset.get())


class FixedParameterSubscanParent(ExpFragment):
    """Parent fragment whose root and child sites each have their own fixed params."""

    def build_fragment(self):
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_param("multiplier", FloatParam, "multiplier", 3.0)
        self.setattr_fragment("child", FixedParameterSubscanLeaf, detached=True)
        self.child_scan = prepare_child_scan(self, self.child, name="inner")
        self.setattr_result("total", FloatChannel)

    def run_once(self):
        self.child_scan.configure(
            ScanRequest.explicit(
                [self.child.x],
                [[self.outer.get()], [self.outer.get() + 1.0]],
            )
        )
        self.child_scan.execute()
        child_result = self.child_scan.inspect()
        self.total.push(self.multiplier.get() * sum(child_result.values[self.child.y]))


def _fit_power_law_exponent(ps, ms) -> float:
    ps = np.asarray(ps, dtype=float)
    ms = np.asarray(ms, dtype=float)
    log_p = np.log(ps)
    log_m = np.log(ms)
    return float(np.dot(log_p, log_m) / np.dot(log_p, log_p))


def _line_fit_artifact(xs, ys) -> dict[str, object]:
    sensible = import_sensible_fitting()
    model = build_builtin_model("straight_line")
    data = sensible.FitData.normal(
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        x_label="x",
        y_label="y",
        label="line",
    )
    _, artifact = fit_data_with_model(
        model,
        data,
        model_id="straight_line",
        source={"x_key": "axis_0", "y_key": "channel_0"},
    )
    return artifact.to_dict()


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

    def _analyse_gradient(self, axis_values, result_values):
        xs = axis_values[self.x]
        ys = result_values[self.y]
        artifact = _line_fit_artifact(xs, ys)
        m = float(artifact_parameter_value(artifact, "m"))
        return AnalysisFeedback(
            outputs={"m": m},
            artifacts={"line_fit": artifact},
            annotations=[
                annotations.artifact_curve(
                    artifact="line_fit",
                    x_axis=self.x,
                    y_axis=self.y,
                )
            ],
        )


class AnalysedScanXFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("line", AnalysedLineFragment, detached=True)
        self.line_scan = prepare_child_scan(self, self.line, name="scan_x")
        self.setattr_result("m", FloatChannel)

    def run_once(self):
        self.line_scan.configure(
            ScanRequest.explicit(
                [self.line.x],
                [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            )
        )
        self.m.push(self.line_scan.execute()["m"])

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
        self.scan_p = prepare_child_scan(self, self.scan_x, name="scan_p")
        self.setattr_result("fit_e", FloatChannel)

    def run_once(self):
        self.scan_p.configure(
            ScanRequest.explicit(
                [self.scan_x.line.p],
                [[1.0], [2.0], [3.0], [4.0], [5.0]],
            )
        )
        self.fit_e.push(self.scan_p.execute()["fit_e"])


class HostRuntimeCase(HasEnvironmentCase):
    def d(self, prefix, key):
        return self.dataset_db.get(prefix + key)

    def j(self, prefix, key):
        return json.loads(self.d(prefix, key))

    def test_host_scan_session_writes_root_scan_site(self):
        self.dataset_db.data["system_id"] = (True, "system")
        fragment = self.create(PlainAddOneFragment, [])
        request = ScanRequest.cartesian([(fragment.value, [0.0, 1.0, 2.0])])

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

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
            self.j(prefix, "scan.parameters"),
            {
                "param_0": {
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
                    "is_scanned": True,
                    "scan_role": "direct",
                }
            },
        )
        self.assertEqual(self.j(prefix, "scan.pseudoparams"), {})
        self.assertEqual(list(self.j(prefix, "scan.channels").keys()), ["channel_0"])
        self.assertEqual(
            self.d(prefix, "points.param_0"),
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

    def test_host_scan_session_records_non_scanned_fixed_parameters(self):
        fragment = self.create(LinearResponseFragment, [])
        request = ScanRequest.cartesian([(fragment.x, [0.0, 1.0])])

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        prefix = result.site_prefix
        self.assertEqual(
            self.j(prefix, "scan.fixed_parameters"),
            {
                "fixed_param_0": {
                    "path": "",
                    "param": {
                        "description": "Slope",
                        "default": "2.0",
                        "fqn": f"{__name__}.LinearResponseFragment.slope",
                        "spec": {
                            "is_scannable": True,
                            "scale": 1.0,
                            "step": 0.1,
                        },
                        "type": "float",
                    },
                    "value": 2.0,
                },
                "fixed_param_1": {
                    "path": "",
                    "param": {
                        "description": "Offset",
                        "default": "1.0",
                        "fqn": f"{__name__}.LinearResponseFragment.offset",
                        "spec": {
                            "is_scannable": True,
                            "scale": 1.0,
                            "step": 0.1,
                        },
                        "type": "float",
                    },
                    "value": 1.0,
                },
            },
        )

    def test_scan_request_linear_helper_materialises_even_spacing(self):
        fragment = self.create(LinearResponseFragment, [])
        request = ScanRequest.linear(fragment.x, start=-2.0, stop=2.0, num_points=5)

        self.assertEqual(request.axes, (fragment.x,))
        self.assertEqual(
            [point.axis_values for point in request.point_policy],
            [(-2.0,), (-1.0,), (0.0,), (1.0,), (2.0,)],
        )

    def test_scan_request_with_repeats_wraps_fixed_repeat_policy(self):
        fragment = self.create(LinearResponseFragment, [])
        request = ScanRequest.explicit(
            [fragment.x],
            [[0.0], [1.0]],
            metadata={"demo_name": "repeat_sugar"},
            execution_policy=ExecutionPolicy(max_points_per_batch=3),
        ).with_repeats(repeats=2)

        self.assertEqual(request.axes, (fragment.x,))
        self.assertEqual(request.metadata["demo_name"], "repeat_sugar")
        self.assertEqual(request.execution_policy.max_points_per_batch, 3)
        self.assertEqual(request.point_policy.describe()["kind"], "repeat")
        self.assertEqual(
            request.point_policy.materialise_points(),
            [(0.0,), (0.0,), (1.0,), (1.0,)],
        )

    def test_scan_request_with_repeats_supports_interleaved_schedule(self):
        fragment = self.create(LinearResponseFragment, [])
        request = ScanRequest.explicit(
            [fragment.x],
            [[0.0], [1.0]],
        ).with_repeats(repeats=3, schedule="interleaved")

        self.assertEqual(request.point_policy.describe()["schedule"], "interleaved")
        self.assertEqual(
            request.point_policy.materialise_points(),
            [(0.0,), (1.0,), (0.0,), (1.0,), (0.0,), (1.0,)],
        )

    def test_host_scan_session_supports_ad_hoc_scan_variables_and_parameter_mappings(
        self,
    ):
        fragment = self.create(PhysicalDriveFragment, [])
        logical_drive = ScanVariable(
            "laser_frequency",
            description="Logical scan axis for the compensated drive",
        )
        request = ScanRequest.cartesian(
            [(logical_drive, [0.0, 1.0, 2.0])]
        ).with_parameter_mappings(
            [
                ParameterMapping.single_target(
                    fragment.drive,
                    [logical_drive],
                    lambda values: values[logical_drive] + 0.5,
                    description="Offset the physical drive from the logical scan axis",
                )
            ]
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        prefix = result.site_prefix
        self.assertEqual(self.d(prefix, "points.pseudoparam_0"), [0.0, 1.0, 2.0])
        self.assertEqual(self.d(prefix, "points.param_0"), [0.5, 1.5, 2.5])
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 3.0, 5.0])
        self.assertEqual(
            self.j(prefix, "scan.pseudoparams"),
            {
                "pseudoparam_0": {
                    "path": "",
                    "variable": {
                        "name": "laser_frequency",
                        "description": "Logical scan axis for the compensated drive",
                        "type": "float",
                        "spec": {},
                    },
                }
            },
        )
        self.assertEqual(
            self.j(prefix, "scan.parameters"),
            {
                "param_0": {
                    "path": "",
                    "param": {
                        "description": "drive",
                        "default": "0.0",
                        "fqn": f"{__name__}.PhysicalDriveFragment.drive",
                        "spec": {
                            "is_scannable": True,
                            "scale": 1.0,
                            "step": 0.1,
                        },
                        "type": "float",
                    },
                    "is_scanned": False,
                    "scan_role": "derived",
                }
            },
        )
        self.assertEqual(
            self.j(prefix, "scan.parameter_mappings"),
            {
                "mapping_0": {
                    "description": "Offset the physical drive from the logical scan axis",
                    "targets": [
                        {
                            "path": "",
                            "param": {
                                "description": "drive",
                                "default": "0.0",
                                "fqn": f"{__name__}.PhysicalDriveFragment.drive",
                                "spec": {
                                    "is_scannable": True,
                                    "scale": 1.0,
                                    "step": 0.1,
                                },
                                "type": "float",
                            },
                        }
                    ],
                    "dependencies": [
                        {
                            "kind": "pseudoparam",
                            "key": "pseudoparam_0",
                        }
                    ],
                }
            },
        )
        self.assertEqual(
            result.coordinates[("scan_variable.laser_frequency", "")],
            [0.0, 1.0, 2.0],
        )

    def test_wrapper_fragment_rebind_param_uses_same_runtime_mapping_path(self):
        fragment = self.create(WrapperRebindFragment, [])
        request = ScanRequest.cartesian(
            [(fragment.logical_drive, [0.0, 1.0, 2.0])]
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        prefix = result.site_prefix
        self.assertEqual(self.d(prefix, "points.param_0"), [0.0, 1.0, 2.0])
        self.assertEqual(self.d(prefix, "points.param_1"), [0.5, 1.5, 2.5])
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 3.0, 5.0])
        self.assertEqual(
            self.j(prefix, "scan.channels"),
            {
                "channel_0": {
                    "path": "child/result",
                    "description": "",
                    "type": "float",
                    "scale": 1.0,
                    "unit": "",
                }
            },
        )
        self.assertEqual(
            self.j(prefix, "scan.parameters"),
            {
                "param_0": {
                    "path": "",
                    "param": {
                        "description": "logical drive",
                        "default": "0.0",
                        "fqn": f"{__name__}.WrapperRebindFragment.logical_drive",
                        "spec": {
                            "is_scannable": True,
                            "scale": 1.0,
                            "step": 0.1,
                        },
                        "type": "float",
                    },
                    "is_scanned": True,
                    "scan_role": "direct",
                },
                "param_1": {
                    "path": "child",
                    "param": {
                        "description": "drive",
                        "default": "0.0",
                        "fqn": f"{__name__}.PhysicalDriveFragment.drive",
                        "spec": {
                            "is_scannable": True,
                            "scale": 1.0,
                            "step": 0.1,
                        },
                        "type": "float",
                    },
                    "is_scanned": False,
                    "scan_role": "derived",
                },
            },
        )
        self.assertEqual(self.j(prefix, "scan.pseudoparams"), {})
        self.assertEqual(
            self.j(prefix, "scan.parameter_mappings"),
            {
                "mapping_0": {
                    "description": "Offset the child hardware drive from the logical wrapper axis",
                    "targets": [
                        {
                            "path": "child",
                            "param": {
                                "description": "drive",
                                "default": "0.0",
                                "fqn": f"{__name__}.PhysicalDriveFragment.drive",
                                "spec": {
                                    "is_scannable": True,
                                    "scale": 1.0,
                                    "step": 0.1,
                                },
                                "type": "float",
                            },
                        }
                    ],
                    "dependencies": [
                        {
                            "kind": "parameter",
                            "key": "param_0",
                        }
                    ],
                }
            },
        )

    def test_schema_compiled_rebind_expression_uses_same_runtime_mapping_path(self):
        fragment = self.create(PhysicalDriveFragment, [])
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
                "execution": {},
            },
        )

        session = PreparedScan(fragment, fragment, request, overrides=overrides)
        result = _execute_and_inspect(session)

        prefix = result.site_prefix
        self.assertEqual(self.d(prefix, "points.pseudoparam_0"), [0.0, 1.0, 2.0])
        self.assertEqual(self.d(prefix, "points.param_0"), [0.5, 1.5, 2.5])
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 3.0, 5.0])
        self.assertEqual(
            self.j(prefix, "scan.fixed_pseudoparams"),
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
        self.assertEqual(
            self.j(prefix, "scan.parameter_mappings"),
            {
                "mapping_0": {
                    "description": "",
                    "expression": "logical_drive + offset",
                    "targets": [
                        {
                            "path": "",
                            "param": {
                                "description": "drive",
                                "default": "0.0",
                                "fqn": f"{__name__}.PhysicalDriveFragment.drive",
                                "spec": {
                                    "is_scannable": True,
                                    "scale": 1.0,
                                    "step": 0.1,
                                },
                                "type": "float",
                            },
                        }
                    ],
                    "dependencies": [
                        {
                            "kind": "pseudoparam",
                            "key": "pseudoparam_0",
                        }
                    ],
                }
            },
        )

    def test_parameter_mapping_from_text_compiles_to_normal_runtime_mapping(self):
        fragment = self.create(PhysicalDriveFragment, [])
        logical_drive = ScanVariable("logical_drive")
        spare = ScanVariable("spare")

        mapping = ParameterMapping.from_text(
            targets=[fragment.drive],
            symbols={
                "logical_drive": logical_drive,
                "offset": 0.5,
                "spare": spare,
            },
            expression="logical_drive + offset",
            description="GUI formula",
        )

        self.assertEqual(mapping.expression, "logical_drive + offset")
        self.assertEqual(mapping.dependencies, (logical_drive,))
        self.assertEqual(mapping.compute({logical_drive: 2.0}), {fragment.drive: 2.5})

    def test_host_scan_session_rejects_scan_axes_that_are_also_mapping_targets(self):
        fragment = self.create(PhysicalDriveFragment, [])
        request = ScanRequest.cartesian([(fragment.drive, [0.0, 1.0, 2.0])]).with_parameter_mappings(
            [
                ParameterMapping.single_target(
                    fragment.drive,
                    [fragment.drive],
                    lambda values: values[fragment.drive] + 1.0,
                    description="Invalid self-targeting mapping",
                )
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "cannot be both a direct scan axis and a mapping target",
        ):
            PreparedScan(fragment, fragment, request).execute()

    def test_host_scan_session_records_root_and_point_unix_timestamps(self):
        fragment = self.create(PlainAddOneFragment, [])
        request = ScanRequest.cartesian([(fragment.value, [0.0, 1.0, 2.0])])

        with patch(
            "ndscan.runtime.executors.time.time",
            side_effect=[999.0, 1000.0, 1001.0, 1002.0, 1003.0],
        ):
            session = PreparedScan(fragment, fragment, request)
            session.execute()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.d(prefix, "site.start_unix_time"), 1000.0)
        self.assertEqual(
            self.d(prefix, "points.acquired_at_unix"),
            [1001.0, 1002.0, 1003.0],
        )

    def test_host_scan_session_executes_and_observes_points_in_batches(self):
        fragment = self.create(CountingLifecycleFragment, [])
        point_policy = RecordingBatchPointPolicy(
            1,
            [(0.0,), (1.0,), (2.0,), (3.0,), (4.0,)],
            preferred_batch_size=2,
        )
        request = ScanRequest(
            axes=(fragment.value,),
            point_policy=point_policy,
            execution_policy=ExecutionPolicy(max_points_per_batch=4),
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)
        point_policy = session._last_request.point_policy

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(point_policy.requested_batch_limits, [2, 2, 2])
        self.assertEqual(point_policy.observed_batches, [[0.0, 1.0], [2.0, 3.0], [4.0]])
        self.assertEqual(
            point_policy.observed_online_analysis_results,
            [{}, {}, {}],
        )
        self.assertEqual(
            point_policy.observed_result_lengths,
            [
                {"result": 2},
                {"result": 4},
                {"result": 5},
            ],
        )
        self.assertEqual(fragment.host_setup_calls, 3)
        self.assertEqual(fragment.host_cleanup_calls, 3)
        self.assertEqual(self.scheduler.num_check_pause_calls, 2)
        self.assertEqual(result.runtime_stats.batch_count, 3)
        self.assertEqual(result.runtime_stats.point_count, 5)
        self.assertEqual(result.runtime_stats.executor_entry_count, 3)
        self.assertIsNotNone(result.runtime_stats.first_executor_entry_elapsed_s)
        self.assertGreaterEqual(result.runtime_stats.total_executor_elapsed_s, 0.0)
        self.assertGreaterEqual(result.runtime_stats.total_batch_finalize_elapsed_s, 0.0)
        self.assertEqual(self.d(prefix, "points.param_0"), [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_host_scan_session_persists_point_metadata_streams(self):
        fragment = self.create(CountingLifecycleFragment, [])
        point_policy = RecordingBatchPointPolicy(
            1,
            [(0.0,), (1.0,), (2.0,)],
            point_metadata=[
                {"decision_source": "seed"},
                {"decision_source": "bo"},
                {"decision_source": "explore"},
            ],
        )
        request = ScanRequest(
            axes=(fragment.value,),
            point_policy=point_policy,
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(
            self.d(prefix, "points.metadata.decision_source"),
            ["seed", "bo", "explore"],
        )

    def test_host_scan_session_flushes_completed_batch_before_pause(self):
        fragment = self.create(CountingLifecycleFragment, [])
        point_policy = RecordingBatchPointPolicy(
            1,
            [(0.0,), (1.0,), (2.0,), (3.0,)],
            preferred_batch_size=2,
        )
        request = ScanRequest(
            axes=(fragment.value,),
            point_policy=point_policy,
            execution_policy=ExecutionPolicy(max_points_per_batch=2),
        )
        self.scheduler.num_check_pause_calls_until_termination = 1

        session = PreparedScan(fragment, fragment, request)
        with self.assertRaises(TerminationRequested):
            session.execute()
        point_policy = session._last_request.point_policy

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(point_policy.observed_batches, [[0.0, 1.0]])
        self.assertEqual(fragment.host_setup_calls, 1)
        self.assertEqual(fragment.host_cleanup_calls, 1)
        self.assertEqual(self.scheduler.num_check_pause_calls, 1)
        self.assertEqual(self.d(prefix, "points.param_0"), [0.0, 1.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 2.0])
        self.assertEqual(self.d(prefix, "state.num_points"), 2)

    def test_host_scan_session_flushes_partial_batch_before_restarting(self):
        fragment = self.create(RestartOnceFragment, [])
        point_policy = RecordingBatchPointPolicy(
            1,
            [(0.0,), (1.0,), (2.0,), (3.0,)],
        )
        request = ScanRequest(
            axes=(fragment.value,),
            point_policy=point_policy,
            execution_policy=ExecutionPolicy(max_points_per_batch=3),
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)
        point_policy = session._last_request.point_policy

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(point_policy.requested_batch_limits, [3, 3])
        self.assertEqual(point_policy.observed_batches, [[0.0, 1.0], [2.0], [3.0]])
        self.assertEqual(fragment.host_setup_calls, 3)
        self.assertEqual(fragment.host_cleanup_calls, 3)
        self.assertEqual(result.runtime_stats.batch_count, 3)
        self.assertEqual(result.runtime_stats.point_count, 4)
        self.assertEqual(result.runtime_stats.executor_entry_count, 3)
        self.assertIsNotNone(result.runtime_stats.first_executor_entry_elapsed_s)
        self.assertEqual(self.d(prefix, "points.param_0"), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 2.0, 3.0, 4.0])

    def test_host_scan_session_writes_preview_hdf5_on_elapsed_batch_boundary(self):
        fragment = self.create(CountingLifecycleFragment, [])
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = os.path.join(tmpdir, "preview.h5")
            request = ScanRequest(
                axes=(fragment.value,),
                point_policy=ExplicitPointPolicy(1, [(0.0,), (1.0,), (2.0,)]),
                execution_policy=ExecutionPolicy(
                    max_points_per_batch=1,
                    preview_policy=PreviewPolicy(
                        path=preview_path,
                        min_interval_s=120.0,
                        write_on_completion=False,
                        remove_on_completion=False,
                    ),
                ),
            )
            with patch(
                "ndscan.runtime.context.time.monotonic",
                side_effect=[0.0, 60.0, 119.0, 121.0],
            ):
                session = PreparedScan(fragment, fragment, request)
                session.execute()

            self.assertTrue(os.path.exists(preview_path))
            with h5py.File(preview_path, "r") as preview_file:
                self.assertEqual(preview_file["rid"][()], 0)
                self.assertFalse(bool(preview_file["preview_complete"][()]))
                self.assertIn("datasets", preview_file)
                datasets = preview_file["datasets"]
                self.assertEqual(
                    datasets["ndscan.rid_0.site.root.points.param_0"][()].tolist(),
                    [0.0, 1.0, 2.0],
                )
                self.assertEqual(
                    datasets["ndscan.rid_0.site.root.points.channel_0"][()].tolist(),
                    [1.0, 2.0, 3.0],
                )
                self.assertEqual(
                    datasets["ndscan.rid_0.site.root.state.num_points"][()],
                    3,
                )
                self.assertFalse(
                    bool(datasets["ndscan.rid_0.site.root.state.completed"][()])
                )

    def test_host_scan_session_writes_completion_preview_when_enabled(self):
        fragment = self.create(PlainAddOneFragment, [])
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = os.path.join(tmpdir, "preview.h5")
            request = ScanRequest.explicit(
                [fragment.value],
                [[5.0]],
                execution_policy=ExecutionPolicy(
                    max_points_per_batch=1,
                    preview_policy=PreviewPolicy(
                        path=preview_path,
                        min_interval_s=9999.0,
                        write_on_completion=True,
                        remove_on_completion=False,
                    ),
                ),
            )

            with patch(
                "ndscan.runtime.context.time.monotonic",
                side_effect=[0.0, 1.0, 2.0],
            ):
                session = PreparedScan(fragment, fragment, request)
                session.execute()

            self.assertTrue(os.path.exists(preview_path))
            with h5py.File(preview_path, "r") as preview_file:
                self.assertTrue(bool(preview_file["preview_complete"][()]))
                datasets = preview_file["datasets"]
                self.assertTrue(
                    bool(datasets["ndscan.rid_0.site.root.state.completed"][()])
                )
                self.assertEqual(
                    datasets["ndscan.rid_0.site.root.points.channel_0"][()].tolist(),
                    [6.0],
                )

    def test_nested_scans_share_one_preview_coordinator(self):
        parent = self.create(NestedChildScanParent, [])
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = os.path.join(tmpdir, "preview.h5")
            request = ScanRequest.explicit(
                [parent.outer],
                [[10.0]],
                execution_policy=ExecutionPolicy(
                    max_points_per_batch=1,
                    preview_policy=PreviewPolicy(
                        path=preview_path,
                        min_interval_s=120.0,
                        write_on_completion=False,
                        remove_on_completion=False,
                    ),
                ),
            )

            with patch(
                "ndscan.runtime.context.time.monotonic",
                side_effect=[0.0, 60.0, 121.0, 122.0],
            ):
                session = PreparedScan(parent, parent, request)
                session.execute()

            self.assertTrue(os.path.exists(preview_path))
            with h5py.File(preview_path, "r") as preview_file:
                datasets = preview_file["datasets"]
                self.assertEqual(
                    datasets["ndscan.rid_0.site.root.child_scan.points.param_0"][
                        ()
                    ].tolist(),
                    [10.0, 11.0],
                )
                self.assertEqual(
                    datasets["ndscan.rid_0.site.root.child_scan.points.channel_0"][
                        ()
                    ].tolist(),
                    [11.0, 12.0],
                )

    def test_preview_policy_defaults_to_rid_and_owner_class_name(self):
        fragment = self.create(PlainAddOneFragment, [])
        self.scheduler.rid = 2484
        self.assertEqual(
            PreviewPolicy().resolve_path(fragment),
            "000002484-PlainAddOneFragment.preview.h5",
        )

    def test_preview_file_is_removed_after_successful_completion_by_default(self):
        fragment = self.create(PlainAddOneFragment, [])
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                request = ScanRequest.explicit(
                    [fragment.value],
                    [[5.0]],
                    execution_policy=ExecutionPolicy(
                        max_points_per_batch=1,
                        preview_policy=PreviewPolicy(
                            min_interval_s=0.0,
                            write_on_completion=False,
                        ),
                    ),
                )

                with patch(
                    "ndscan.runtime.context.time.monotonic",
                    side_effect=[0.0, 1.0],
                ):
                    session = PreparedScan(fragment, fragment, request)
                    session.execute()

                self.assertFalse(
                    os.path.exists("000000000-PlainAddOneFragment.preview.h5")
                )
            finally:
                os.chdir(previous_cwd)

    def test_host_scan_session_executes_online_analyses_at_batch_boundaries(self):
        fragment = self.create(OnlineGaussianFragment, [])
        point_policy = RecordingBatchPointPolicy(
            1,
            [(-2.0,), (-1.0,), (0.0,), (1.0,), (2.0,), (3.0,), (4.0,)],
            preferred_batch_size=4,
        )
        request = ScanRequest(
            axes=(fragment.x,),
            point_policy=point_policy,
            execution_policy=ExecutionPolicy(max_points_per_batch=4),
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)
        point_policy = session._last_request.point_policy

        prefix = result.site_prefix
        analysis_name = "fit_gaussian_channel_0"
        self.assertIn(analysis_name, self.j(prefix, "analysis.online"))
        self.assertIn(analysis_name, result.online_analysis_results)
        self.assertIn(analysis_name, point_policy.observed_online_analysis_results[-1])

        online_result = self.j(prefix, "analysis.online_result." + analysis_name)
        self.assertAlmostEqual(online_result["x0"], 1.0, places=3)
        self.assertAlmostEqual(online_result["sigma"], 1.2, places=3)
        self.assertEqual(
            self.j(prefix, "analysis.online_result." + analysis_name),
            point_policy.observed_online_analysis_results[-1][analysis_name],
        )
        self.assertEqual(
            point_policy.observed_online_analysis_annotations[-1][analysis_name],
            [],
        )

    def test_host_scan_session_publishes_online_analysis_annotations_and_feedback(self):
        fragment = self.create(OnlineAnnotatedFragment, [])
        point_policy = RecordingBatchPointPolicy(
            1,
            [(0.0,), (1.0,), (2.0,)],
            preferred_batch_size=2,
        )
        request = ScanRequest(
            axes=(fragment.x,),
            point_policy=point_policy,
            execution_policy=ExecutionPolicy(max_points_per_batch=2),
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)
        point_policy = session._last_request.point_policy

        prefix = result.site_prefix
        online_result = self.j(prefix, "analysis.online_result.running_summary")
        online_annotations = self.j(prefix, "analysis.online_annotation.running_summary")

        self.assertEqual(online_result["num_points"], 3)
        self.assertEqual(online_result["latest_x"], 2.0)
        self.assertEqual(online_result["latest_y"], 3.0)
        self.assertEqual(result.online_analysis_results["running_summary"], online_result)
        self.assertEqual(result.analysis_results["num_points"], 3)
        self.assertEqual(result.analysis_results["latest_x"], 2.0)
        self.assertEqual(result.analysis_results["latest_y"], 3.0)
        self.assertEqual(self.d(prefix, "analysis.output.num_points"), 3)
        self.assertEqual(self.d(prefix, "analysis.output.latest_x"), 2.0)
        self.assertEqual(self.d(prefix, "analysis.output.latest_y"), 3.0)
        self.assertEqual(
            result.online_analysis_annotations["running_summary"],
            online_annotations,
        )
        self.assertEqual(
            point_policy.observed_online_analysis_results[-1]["running_summary"],
            online_result,
        )
        self.assertEqual(
            point_policy.observed_online_analysis_annotations[-1]["running_summary"],
            online_annotations,
        )
        self.assertEqual(online_annotations[0]["kind"], "curve")

    def test_host_scan_session_supports_gradient_descent_point_policy(self):
        fragment = self.create(FourDimQuadraticFragment, [])
        point_policy = GradientDescentPointPolicy(
            initial_point=(0.0, 0.0, 0.0, 0.0),
            objective=lambda observation: observation.channel_values["channel_0"],
            probe_steps=(0.1, 0.1, 0.1, 0.1),
            learning_rate=0.5,
            max_iterations=3,
            gradient_tolerance=1e-9,
            objective_description="quadratic loss",
        )
        request = ScanRequest(
            axes=(fragment.x0, fragment.x1, fragment.x2, fragment.x3),
            point_policy=point_policy,
            execution_policy=ExecutionPolicy(max_points_per_batch=9),
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)
        point_policy = session._last_request.point_policy

        prefix = result.site_prefix
        self.assertEqual(self.d(prefix, "state.num_points"), 18)
        self.assertAlmostEqual(point_policy.best_value, 0.0, places=9)
        self.assertEqual(
            tuple(round(value, 6) for value in point_policy.best_point),
            (1.0, -2.0, 0.5, 3.0),
        )
        self.assertAlmostEqual(
            min(self.d(prefix, "points.channel_0")),
            0.0,
            places=9,
        )

    def test_numeric_channels_can_opt_out_of_save_by_default(self):
        fragment = self.create(VisibleAndHiddenNumericFragment, [])
        request = ScanRequest.explicit([fragment.value], [[1.0], [2.0]])

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

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

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        prefix = result.site_prefix
        self.assertEqual(prefix, "ndscan.rid_0.site.root.zipped_demo.")
        self.assertEqual(
            self.d(prefix, "points.param_0"),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            self.d(prefix, "points.param_1"),
            [10.0, 20.0, 30.0],
        )
        self.assertEqual(
            self.d(prefix, "points.channel_0"),
            [11.0, 22.0, 33.0],
        )

    def test_host_scan_session_respects_until_condition_point_policy(self):
        fragment = self.create(PlainAddOneFragment, [])
        request = ScanRequest(
            axes=(fragment.value,),
            point_policy=UntilConditionPointPolicy(
                ExplicitPointPolicy(1, [(0.0,), (1.0,), (2.0,), (3.0,)]),
                lambda observation: observation.channel_values["channel_0"] >= 3.0,
                predicate_description="channel_0 >= 3.0",
            ),
        )

        session = PreparedScan(fragment, fragment, request)
        session.execute()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.d(prefix, "points.param_0"), [0.0, 1.0, 2.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 2.0, 3.0])

    def test_host_scan_session_supports_repeated_points_without_dummy_axis(self):
        fragment = self.create(SequentialHitFragment, [])
        request = ScanRequest(
            axes=(),
            point_policy=RepeatPointPolicy(
                SinglePointPolicy(),
                stop_predicate=lambda feedback: len(feedback.result_data[fragment.hit]) >= 3,
                min_repeats=1,
                max_repeats=5,
                predicate_description="three shots gathered",
            ),
            execution_policy=ExecutionPolicy(max_points_per_batch=1),
        )

        session = PreparedScan(fragment, fragment, request)
        session.execute()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.d(prefix, "points.channel_0"), [1.0, 0.0, 1.0])
        self.assertEqual(self.d(prefix, "state.num_points"), 3)
        self.assertNotIn(prefix + "points.param_0", self.dataset_db.data)
        self.assertEqual(self.j(prefix, "scan.parameters"), {})

    def test_host_scan_session_executes_default_analyses_after_run(self):
        fragment = self.create(AnalysedLineFragment, [])
        request = ScanRequest.explicit(
            [fragment.x],
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        )

        session = PreparedScan(fragment, fragment, request)
        result = _execute_and_inspect(session)

        prefix = result.site_prefix
        self.assertAlmostEqual(result.analysis_results["m"], 4.0, places=6)
        self.assertAlmostEqual(self.d(prefix, "analysis.output.m"), 4.0)

        metadata = self.j(prefix, "analysis.outputs")
        self.assertIn("m", metadata)
        self.assertEqual(metadata["m"]["path"], "m")

        artifact = self.j(prefix, "analysis.artifact.line_fit")
        self.assertEqual(artifact["kind"], "model_fit")
        self.assertEqual(artifact["provider"], "sensible_fitting")
        self.assertEqual(artifact["model_id"], "straight_line")
        self.assertAlmostEqual(artifact["parameters"]["m"]["value"], 4.0, places=6)
        self.assertEqual(result.analysis_artifacts["line_fit"], artifact)

        annotations_data = self.j(prefix, "analysis.annotations")
        self.assertEqual(len(annotations_data), 1)
        self.assertEqual(annotations_data[0]["kind"], "artifact_curve")
        self.assertEqual(annotations_data[0]["parameters"]["artifact"], "line_fit")
        self.assertEqual(result.annotations, annotations_data)

    def test_host_scan_session_execute_returns_scan_outputs(self):
        fragment = self.create(AnalysedLineFragment, [])
        request = ScanRequest.explicit(
            [fragment.x],
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        )

        session = PreparedScan(
            fragment,
            fragment,
            request,
            expose_outputs=["m"],
        )
        outputs = session.execute()

        self.assertEqual(outputs.names, ("m",))
        self.assertEqual(outputs.as_tuple(), (4.0,))
        self.assertAlmostEqual(outputs["m"], 4.0, places=6)
        self.assertEqual(session.get_outputs(), (4.0,))
        self.assertEqual(session.outputs().as_tuple(), (4.0,))
        self.assertAlmostEqual(session.inspect().analysis_results["m"], 4.0, places=6)

    def test_host_scan_session_execute_can_select_explicit_output_subset(self):
        fragment = self.create(OnlineAnnotatedFragment, [])
        request = ScanRequest.explicit(
            [fragment.x],
            [[0.0], [1.0], [2.0]],
        )

        session = PreparedScan(
            fragment,
            fragment,
            request,
            expose_outputs=["latest_y", "num_points"],
        )
        outputs = session.execute()

        self.assertEqual(outputs.names, ("latest_y", "num_points"))
        self.assertEqual(outputs.as_tuple(), (3.0, 3))
        self.assertEqual(outputs["latest_y"], 3.0)
        self.assertEqual(outputs["num_points"], 3)
        self.assertEqual(session.get_outputs(), (3.0, 3))
        self.assertEqual(session.inspect().analysis_results["latest_x"], 2.0)
        self.assertEqual(set(session.inspect().analysis_results.keys()), {"num_points", "latest_x", "latest_y"})

    def test_prepared_scan_experiment_adapter_runs(self):
        self.dataset_db.data["system_id"] = (True, "system")
        HostAddOneScan = make_fragment_prepared_scan_exp(
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
        self.assertEqual(self.d(prefix, "points.param_0"), [5.0, 7.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [6.0, 8.0])

    def test_direct_prepared_root_example_runs(self):
        exp = self.create(HostRuntimePreparedRootLinearScan)
        exp.prepare()
        exp.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.d(prefix, "points.param_0"), [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [2.0, 4.0, 6.0, 8.0, 10.0])
        self.assertAlmostEqual(self.d(prefix, "analysis.output.m"), 2.0, places=6)
        self.assertEqual(exp.scan.get_outputs(), (2.0,))
        self.assertAlmostEqual(exp.fit_slope, 2.0, places=6)

    @patch("examples.host_runtime_grouped_line_family.time.sleep", return_value=None)
    def test_grouped_line_family_example_runs(self, _sleep):
        exp = self.create(HostRuntimeGroupedLineFamily)
        exp.prepare()
        exp.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(len(self.d(prefix, "points.param_0")), 36 * 5)
        self.assertEqual(len(self.d(prefix, "points.param_1")), 36 * 5)
        self.assertEqual(len(self.d(prefix, "points.channel_0")), 36 * 5)
        self.assertEqual(
            sorted(set(self.d(prefix, "points.param_1"))),
            [-2.0, -1.0, 0.0, 1.0, 2.0],
        )

    def test_field_shift_spectroscopy_example_runs(self):
        exp = self.create(HostRuntimeFieldShiftSpectroscopy)
        exp.prepare()
        exp.run()

        field_prefix = "ndscan.rid_0.site.root.scan_field."
        freq_prefix = field_prefix + "scan_frequency."

        self.assertEqual(len(self.d(field_prefix, "points.param_0")), 5)
        self.assertAlmostEqual(
            self.d(field_prefix, "analysis.output.fit_shift_per_field"),
            TRUE_SHIFT_PER_FIELD,
            delta=0.08,
        )
        self.assertAlmostEqual(
            self.d(field_prefix, "analysis.output.fit_centre_at_zero_field"),
            TRUE_CENTER_AT_ZERO,
            delta=0.1,
        )
        segment_feedback = [
            json.loads(item)
            for item in self.d(freq_prefix, "segments.analysis.final_feedback")
        ]
        self.assertEqual(segment_feedback[0]["annotations"][0]["kind"], "artifact_curve")
        self.assertEqual(
            segment_feedback[0]["annotations"][1]["kind"], "artifact_location"
        )

    def test_counts_shot_chunk_time_scan_example_runs(self):
        with (
            patch.object(counts_field_calibration, "POINT_DELAY_S", 0.0),
            patch.object(counts_field_calibration, "DEFAULT_SHOTS_PER_CHUNK", 12),
            patch.object(
                counts_field_calibration,
                "DURATION_POINTS",
                _REAL_NP_LINSPACE(0.0, 2.2, 9).tolist(),
            ),
        ):
            exp = self.create(counts_field_calibration.HostRuntimeCountsShotChunkTimeScan)
            exp.prepare()
            exp.run()

        prefix = "ndscan.rid_0.site.root."
        repeat_prefix = prefix + "repeat_scan."
        self.assertEqual(len(self.d(prefix, "points.param_0")), 9)
        self.assertAlmostEqual(
            self.d(prefix, "analysis.output.fit_pi_time"),
            0.5 / counts_field_calibration.DEFAULT_RABI_FREQUENCY,
            delta=0.22,
        )
        self.assertEqual(
            np.asarray(exp.get_dataset("last_image")).shape,
            counts_field_calibration.IMAGE_SHAPE,
        )
        self.assertEqual(
            np.asarray(self.d(repeat_prefix, "points.channel_0")[0]).shape,
            (
                counts_field_calibration.NUM_GROUPS,
                counts_field_calibration.NUM_ROIS,
            ),
        )
        annotations_data = self.j(prefix, "analysis.annotations")
        self.assertEqual([item["kind"] for item in annotations_data], ["artifact_curve"])

    def test_counts_frequency_calibration_example_runs(self):
        with (
            patch.object(counts_field_calibration, "POINT_DELAY_S", 0.0),
            patch.object(counts_field_calibration, "DEFAULT_SHOTS_PER_CHUNK", 14),
            patch.object(
                counts_field_calibration,
                "FREQUENCY_POINTS",
                _REAL_NP_LINSPACE(9.4, 12.6, 9).tolist(),
            ),
        ):
            exp = self.create(counts_field_calibration.HostRuntimeCountsFrequencyCalibration)
            exp.prepare()
            exp.run()

        prefix = "ndscan.rid_0.site.root."
        freq_prefix = prefix + "scan_frequency."
        repeat_prefix = freq_prefix + "repeat_scan."

        self.assertEqual(len(self.d(freq_prefix, "points.param_0")), 9)
        self.assertAlmostEqual(
            self.d(freq_prefix, "analysis.output.fit_center"),
            counts_field_calibration.TRUE_ZERO_FIELD_FREQUENCY,
            delta=0.18,
        )
        self.assertEqual(
            np.asarray(exp.get_dataset("last_image")).shape,
            counts_field_calibration.IMAGE_SHAPE,
        )
        self.assertEqual(
            np.asarray(self.d(repeat_prefix, "points.channel_0")[0]).shape,
            (
                counts_field_calibration.NUM_GROUPS,
                counts_field_calibration.NUM_ROIS,
            ),
        )
        self.assertEqual(len(self.d(prefix, "points.channel_0")), 1)
        self.assertEqual(len(self.d(prefix, "points.channel_1")), 1)

    def test_counts_field_calibration_example_runs(self):
        with (
            patch.object(counts_field_calibration, "POINT_DELAY_S", 0.0),
            patch.object(counts_field_calibration, "DEFAULT_SHOTS_PER_CHUNK", 14),
            patch.object(
                counts_field_calibration,
                "FREQUENCY_POINTS",
                _REAL_NP_LINSPACE(9.4, 12.6, 9).tolist(),
            ),
            patch.object(
                counts_field_calibration,
                "FIELD_POINTS",
                _REAL_NP_LINSPACE(-1.2, 1.2, 5).tolist(),
            ),
        ):
            exp = self.create(counts_field_calibration.HostRuntimeCountsFieldCalibration)
            exp.prepare()
            exp.run()

        field_prefix = "ndscan.rid_0.site.root.scan_field."
        freq_prefix = field_prefix + "scan_frequency."
        repeat_prefix = freq_prefix + "repeat_scan."

        self.assertEqual(len(self.d(field_prefix, "points.param_0")), 5)
        self.assertAlmostEqual(
            self.d(field_prefix, "analysis.output.fit_shift_per_current"),
            counts_field_calibration.TRUE_FREQUENCY_SHIFT_PER_CURRENT,
            delta=0.12,
        )
        self.assertAlmostEqual(
            self.d(field_prefix, "analysis.output.fit_center_at_zero_current"),
            counts_field_calibration.TRUE_ZERO_FIELD_FREQUENCY,
            delta=0.18,
        )
        self.assertEqual(
            np.asarray(exp.get_dataset("last_image")).shape,
            counts_field_calibration.IMAGE_SHAPE,
        )
        self.assertEqual(
            np.asarray(self.d(repeat_prefix, "points.channel_0")[0]).shape,
            (
                counts_field_calibration.NUM_GROUPS,
                counts_field_calibration.NUM_ROIS,
            ),
        )
        segment_feedback = [
            json.loads(item)
            for item in self.d(freq_prefix, "segments.analysis.final_feedback")
        ]
        self.assertEqual(segment_feedback[0]["annotations"][0]["kind"], "artifact_curve")

    def test_interleaved_spectroscopy_example_runs(self):
        with (
            patch.object(interleaved_patterns, "POINT_DELAY_S", 0.0),
            patch.object(
                interleaved_patterns,
                "DEMO_FREQUENCY_POINTS",
                _REAL_NP_LINSPACE(-0.35, 0.75, 7).tolist(),
            ),
            patch.object(interleaved_patterns, "TARGET_PROBABILITY_ERROR", 0.22),
            patch.object(interleaved_patterns, "MIN_SHOTS_PER_FREQUENCY", 3),
            patch.object(interleaved_patterns, "MAX_SHOTS_PER_FREQUENCY", 8),
        ):
            exp = self.create(interleaved_patterns.HostRuntimeInterleavedSpectroscopy)
            exp.prepare()
            exp.run()

        prefix = "ndscan.rid_0.site.root."
        probe_frequencies = self.d(prefix, "points.param_0")
        self.assertGreater(len(probe_frequencies), len(set(probe_frequencies)))
        self.assertAlmostEqual(
            self.d(prefix, "analysis.output.fit_center"),
            interleaved_patterns.TRUE_LINE_CENTER,
            delta=0.18,
        )
        self.assertEqual(self.d(prefix, "analysis.output.num_unique_frequencies"), 7)
        annotations_data = self.j(prefix, "analysis.annotations")
        self.assertEqual([item["kind"] for item in annotations_data], ["artifact_curve"])

    def test_interleaved_spectroscopy_subscan_example_runs(self):
        with (
            patch.object(interleaved_patterns, "POINT_DELAY_S", 0.0),
            patch.object(
                interleaved_patterns,
                "DEMO_FREQUENCY_POINTS",
                _REAL_NP_LINSPACE(-0.35, 0.75, 7).tolist(),
            ),
            patch.object(interleaved_patterns, "TARGET_PROBABILITY_ERROR", 0.22),
            patch.object(interleaved_patterns, "MIN_SHOTS_PER_FREQUENCY", 3),
            patch.object(interleaved_patterns, "MAX_SHOTS_PER_FREQUENCY", 8),
        ):
            exp = self.create(interleaved_patterns.HostRuntimeInterleavedSpectroscopySubscan)
            exp.prepare()
            exp.run()

        root_prefix = "ndscan.rid_0.site.root."
        child_prefix = root_prefix + "scan_frequency."
        self.assertEqual(len(self.d(root_prefix, "points.channel_0")), 1)
        self.assertAlmostEqual(
            self.d(root_prefix, "points.channel_0")[0],
            interleaved_patterns.TRUE_LINE_CENTER,
            delta=0.18,
        )
        probe_frequencies = self.d(child_prefix, "points.param_0")
        self.assertGreater(len(probe_frequencies), len(set(probe_frequencies)))
        annotations_data = self.j(child_prefix, "analysis.annotations")
        self.assertEqual([item["kind"] for item in annotations_data], ["artifact_curve"])

    def test_chunked_interleaved_spectroscopy_example_runs(self):
        with (
            patch.object(interleaved_patterns, "POINT_DELAY_S", 0.0),
            patch.object(
                interleaved_patterns,
                "DEMO_FREQUENCY_POINTS",
                _REAL_NP_LINSPACE(-0.35, 0.75, 7).tolist(),
            ),
            patch.object(interleaved_patterns, "CHUNK_SHOTS_PER_POINT", 3),
            patch.object(interleaved_patterns, "CHUNKED_TARGET_PROBABILITY_ERROR", 0.22),
            patch.object(
                interleaved_patterns,
                "MIN_TOTAL_SHOTS_PER_FREQUENCY",
                6,
            ),
            patch.object(
                interleaved_patterns,
                "MAX_TOTAL_SHOTS_PER_FREQUENCY",
                12,
            ),
        ):
            exp = self.create(interleaved_patterns.HostRuntimeChunkedInterleavedSpectroscopy)
            exp.prepare()
            exp.run()

        prefix = "ndscan.rid_0.site.root."
        probe_frequencies = self.d(prefix, "points.param_0")
        self.assertGreater(len(probe_frequencies), len(set(probe_frequencies)))
        self.assertIn(prefix + "points.channel_0", self.dataset_db.data)
        self.assertIn(prefix + "points.channel_1", self.dataset_db.data)
        self.assertIn(prefix + "points.channel_2", self.dataset_db.data)
        self.assertIn(prefix + "points.channel_3", self.dataset_db.data)
        self.assertAlmostEqual(
            self.d(prefix, "analysis.output.fit_center"),
            interleaved_patterns.TRUE_LINE_CENTER,
            delta=0.18,
        )
        annotations_data = self.j(prefix, "analysis.annotations")
        self.assertEqual([item["kind"] for item in annotations_data], ["artifact_curve"])

    @patch("examples.host_runtime_rabi_flop_2d.np.linspace")
    @patch("examples.host_runtime_rabi_flop_2d.time.sleep", return_value=None)
    def test_prepared_rabi_flop_2d_example_runs(self, _sleep, mock_linspace):
        def _small_linspace(start, stop, num, *args, **kwargs):
            if (float(start), float(stop), int(num)) == (0.2, 1.8, 90):
                return _REAL_NP_LINSPACE(start, stop, 4, *args, **kwargs)
            if (float(start), float(stop), int(num)) == (0.0, 2.5, 130):
                return _REAL_NP_LINSPACE(start, stop, 5, *args, **kwargs)
            return _REAL_NP_LINSPACE(start, stop, num, *args, **kwargs)

        mock_linspace.side_effect = _small_linspace

        exp = self.create(HostRuntimeRabiFlop2D)
        exp.prepare()
        exp.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(len(self.d(prefix, "points.param_0")), 4 * 5)
        self.assertEqual(len(self.d(prefix, "points.param_1")), 4 * 5)
        self.assertEqual(len(self.d(prefix, "points.channel_1")), 4 * 5)
        self.assertTrue(
            all(0.0 <= value <= 1.0 for value in self.d(prefix, "points.channel_1"))
        )

    def test_code_defined_prepared_scan_experiment_does_not_publish_ndscan_arguments(self):
        HostAddOneScan = make_fragment_prepared_scan_exp(
            PlainAddOneFragment,
            lambda fragment: ScanRequest.explicit([fragment.value], [[5.0], [7.0]]),
        )

        exp = self.create(HostAddOneScan)

        self.assertFalse(hasattr(exp, "args"))
        self.assertNotEqual(getattr(exp, "argument_ui", None), "ndscan")

    def test_prepared_dashboard_scan_experiment_publishes_ndscan_arguments(self):
        HostAddOneScan = make_fragment_prepared_dashboard_scan_exp(PlainAddOneFragment)

        exp = self.create(HostAddOneScan)

        self.assertEqual(exp.argument_ui, "ndscan")
        self.assertIn("schemata", exp.args._params)
        self.assertIn("instances", exp.args._params)
        self.assertIn("channels", exp.args._params)
        self.assertIn("overrides", exp.args._params)
        self.assertIn("host_scan", exp.args._params)
        self.assertIn("result", exp.args._params["channels"])

        backend = select_submission_backend(exp.args._params)
        self.assertIsInstance(backend, HostSubmissionBackend)
        self.assertTrue(backend.supports_editing)

    def test_prepared_dashboard_scan_experiment_applies_dashboard_overrides(self):
        sample_fragment = self.create(PlainAddOneFragment, [])
        HostAddOneScan = make_fragment_prepared_dashboard_scan_exp(PlainAddOneFragment)

        exp = self.create(
            HostAddOneScan,
            env_args={
                PARAMS_ARG_KEY: pyon.encode(
                    {
                        "overrides": {
                            sample_fragment.value.parameter.fqn: [
                                {"path": "*", "value": 10.0}
                            ]
                        }
                    }
                )
            },
        )
        exp.prepare()
        exp.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.d(prefix, "points.channel_0"), [11.0])

    def test_prepared_dashboard_scan_experiment_uses_transport_host_scan_when_present(self):
        sample_fragment = self.create(PlainAddOneFragment, [])
        default_schema = {
            "version": 1,
            "mode": {"type": "grid"},
            "entries": [
                {
                    "id": "value",
                    "kind": "param",
                    "target": {
                        "fqn": sample_fragment.value.parameter.fqn,
                        "path": "*",
                    },
                    "mode": {
                        "type": "scan",
                        "generator": {
                            "type": "list",
                            "range": {"values": [1.0], "randomise_order": False},
                        },
                    },
                }
            ],
            "execution": {},
        }
        HostSchemaScan = make_fragment_prepared_dashboard_scan_exp(
            PlainAddOneFragment,
            default_schema,
        )

        exp = self.create(
            HostSchemaScan,
            env_args={
                PARAMS_ARG_KEY: pyon.encode(
                    {
                        "host_scan": {
                            "version": 1,
                            "mode": {"type": "grid"},
                            "entries": [
                                {
                                    "id": "value",
                                    "kind": "param",
                                    "target": {
                                        "fqn": sample_fragment.value.parameter.fqn,
                                        "path": "*",
                                    },
                                    "mode": {
                                        "type": "scan",
                                        "generator": {
                                            "type": "list",
                                            "range": {
                                                "values": [2.0, 4.0],
                                                "randomise_order": False,
                                            },
                                        },
                                    },
                                }
                            ],
                            "execution": {},
                        }
                    }
                )
            },
        )
        exp.prepare()
        exp.run()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.d(prefix, "points.param_0"), [2.0, 4.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [3.0, 5.0])

    def test_host_scan_session_allows_kernel_helpers_inside_host_methods(self):
        fragment = self.create(HostCallsKernelHelperFragment, [])
        request = ScanRequest.explicit([fragment.value], [[4.0], [5.0]])

        session = PreparedScan(fragment, fragment, request)
        session.execute()

        prefix = "ndscan.rid_0.site.root."
        self.assertEqual(self.d(prefix, "points.param_0"), [4.0, 5.0])
        self.assertEqual(self.d(prefix, "points.channel_0"), [5.0, 6.0])

    def test_host_scan_session_requires_kernel_fragments_to_declare_core(self):
        fragment = self.create(MissingCoreKernelFragment, [])
        request = ScanRequest.single()

        with self.assertRaisesRegex(
            ValueError, r"must declare.*self\.setattr_device\('core'\)"
        ):
            PreparedScan(fragment, fragment, request).execute()

    def test_make_child_scan_site_requires_active_parent_point(self):
        with self.assertRaises(RuntimeError):
            make_child_scan_site("not_inside_a_scan")

    def test_nested_child_scan_uses_same_runtime_core(self):
        parent = self.create(NestedChildScanParent, [])
        request = ScanRequest.explicit([parent.outer], [[10.0], [20.0]])

        session = PreparedScan(parent, parent, request)
        session.execute()

        root_prefix = "ndscan.rid_0.site.root."
        child_prefix = "ndscan.rid_0.site.root.child_scan."

        self.assertEqual(self.d(root_prefix, "points.param_0"), [10.0, 20.0])
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
            self.d(child_prefix, "points.param_0"),
            [10.0, 11.0, 20.0, 21.0],
        )
        self.assertEqual(
            self.d(child_prefix, "points.channel_0"),
            [11.0, 12.0, 21.0, 22.0],
        )

    def test_prepared_child_scan_uses_same_runtime_core(self):
        parent = self.create(PreparedChildScanParent, [])
        request = ScanRequest.explicit([parent.outer], [[10.0], [20.0]])

        session = PreparedScan(parent, parent, request)
        session.execute()

        root_prefix = "ndscan.rid_0.site.root."
        child_prefix = "ndscan.rid_0.site.root.child_scan."

        self.assertEqual(self.d(root_prefix, "points.param_0"), [10.0, 20.0])
        self.assertEqual(
            self.d(root_prefix, "points.channel_0"),
            [23.0, 43.0],
        )
        self.assertEqual(self.j(child_prefix, "site.path"), ["child_scan"])
        self.assertEqual(self.d(child_prefix, "segments.start_index"), [0, 2])
        self.assertEqual(
            self.d(child_prefix, "segments.parent_point_index"),
            [0, 1],
        )
        self.assertEqual(
            self.d(child_prefix, "points.param_0"),
            [10.0, 11.0, 20.0, 21.0],
        )
        self.assertEqual(
            self.d(child_prefix, "points.channel_0"),
            [11.0, 12.0, 21.0, 22.0],
        )

    def test_prepared_child_scan_requires_configuration(self):
        parent = self.create(UnconfiguredPreparedChildScanParent, [])
        request = ScanRequest.single()

        session = PreparedScan(parent, parent, request)
        with self.assertRaisesRegex(RuntimeError, "has not been configured yet"):
            session.execute()

    def test_prepared_child_scan_reuses_request_with_fresh_point_policy(self):
        parent = self.create(ReusedPreparedChildScanParent, [])
        request = ScanRequest.single()

        session = PreparedScan(parent, parent, request)
        session.execute()

        root_prefix = "ndscan.rid_0.site.root."
        child_prefix = "ndscan.rid_0.site.root.child_scan."

        self.assertEqual(self.d(root_prefix, "points.channel_0"), [14.0])
        self.assertEqual(self.d(child_prefix, "segments.start_index"), [0, 2])
        self.assertEqual(self.d(child_prefix, "segments.parent_point_index"), [0, 0])
        self.assertEqual(self.d(child_prefix, "points.param_0"), [2.0, 3.0, 2.0, 3.0])
        self.assertEqual(
            self.d(child_prefix, "points.channel_0"), [3.0, 4.0, 3.0, 4.0]
        )

    def test_prepare_child_scan_auto_detaches_direct_child_during_build(self):
        parent = self.create(AutoDetachedPreparedChildScanParent, [])
        request = ScanRequest.explicit([parent.outer], [[10.0], [20.0]])

        session = PreparedScan(parent, parent, request)
        session.execute()

        self.assertIn(parent.child, parent._detached_subfragments)
        self.assertEqual(parent.child.host_setup_calls, 2)
        self.assertEqual(parent.child.host_cleanup_calls, 2)

    def test_setattr_prepared_child_scan_creates_detached_child_and_scan_handle(self):
        parent = self.create(SetattrPreparedChildScanParent, [])
        request = ScanRequest.explicit([parent.outer], [[10.0], [20.0]])

        session = PreparedScan(parent, parent, request)
        session.execute()

        self.assertTrue(hasattr(parent, "child"))
        self.assertIn(parent.child, parent._detached_subfragments)

        child_prefix = "ndscan.rid_0.site.root.child_scan."
        self.assertEqual(self.j(child_prefix, "site.path"), ["child_scan"])
        self.assertEqual(
            self.d(child_prefix, "points.param_0"),
            [10.0, 11.0, 20.0, 21.0],
        )

    def test_prepared_child_scan_can_expose_fixed_outputs(self):
        parent = self.create(PreparedChildFixedOutputParent, [])
        request = ScanRequest.single()

        session = PreparedScan(parent, parent, request)
        result = _execute_and_inspect(session)

        self.assertAlmostEqual(result.values[parent.child_slope][0], 4.0, places=6)
        self.assertEqual(parent.child_scan.get_outputs(), (4.0,))
        self.assertEqual(parent.child_scan.outputs().as_tuple(), (4.0,))

    def test_prepared_child_scan_can_fetch_fixed_outputs_as_tuple(self):
        parent = self.create(PreparedChildTupleOutputParent, [])
        request = ScanRequest.single()

        session = PreparedScan(parent, parent, request)
        result = _execute_and_inspect(session)

        self.assertAlmostEqual(result.values[parent.child_slope][0], 4.0, places=6)
        self.assertEqual(parent.child_scan.get_outputs(), (4.0,))
        self.assertEqual(parent.child_scan.outputs().as_tuple(), (4.0,))
        self.assertAlmostEqual(parent.child_scan.inspect().analysis_results["m"], 4.0, places=6)

    def test_prepared_child_scan_execute_returns_scan_outputs(self):
        parent = self.create(PreparedChildExecuteOutputParent, [])
        request = ScanRequest.single()

        session = PreparedScan(parent, parent, request)
        result = _execute_and_inspect(session)

        self.assertAlmostEqual(result.values[parent.child_slope][0], 4.0, places=6)
        self.assertEqual(parent.child_scan.outputs().names, ("m",))
        self.assertEqual(parent.child_scan.outputs().as_tuple(), (4.0,))
        self.assertAlmostEqual(parent.child_scan.outputs()["m"], 4.0, places=6)
        self.assertAlmostEqual(parent.child_scan.inspect().analysis_results["m"], 4.0, places=6)

    def test_nested_child_scan_uses_true_parent_point_indices_with_batched_parent(self):
        parent = self.create(NestedChildScanParent, [])
        request = ScanRequest.explicit(
            [parent.outer],
            [[10.0], [20.0], [30.0]],
            execution_policy=ExecutionPolicy(max_points_per_batch=3),
        )

        session = PreparedScan(parent, parent, request)
        session.execute()

        child_prefix = "ndscan.rid_0.site.root.child_scan."
        self.assertEqual(
            self.d(child_prefix, "segments.parent_point_index"),
            [0, 1, 2],
        )

    def test_nested_scan_records_fixed_parameters_per_site(self):
        parent = self.create(FixedParameterSubscanParent, [])
        request = ScanRequest.explicit([parent.outer], [[10.0]])

        session = PreparedScan(parent, parent, request)
        session.execute()

        root_prefix = "ndscan.rid_0.site.root."
        child_prefix = "ndscan.rid_0.site.root.inner."

        self.assertEqual(
            self.j(root_prefix, "scan.fixed_parameters"),
            {
                "fixed_param_0": {
                    "path": "",
                    "param": {
                        "description": "multiplier",
                        "default": "3.0",
                        "fqn": f"{__name__}.FixedParameterSubscanParent.multiplier",
                        "spec": {
                            "is_scannable": True,
                            "scale": 1.0,
                            "step": 0.1,
                        },
                        "type": "float",
                    },
                    "value": 3.0,
                }
            },
        )
        self.assertEqual(
            self.j(child_prefix, "scan.fixed_parameters"),
            {
                "fixed_param_0": {
                    "path": "child",
                    "param": {
                        "description": "gain",
                        "default": "2.0",
                        "fqn": f"{__name__}.FixedParameterSubscanLeaf.gain",
                        "spec": {
                            "is_scannable": True,
                            "scale": 1.0,
                            "step": 0.1,
                        },
                        "type": "float",
                    },
                    "value": 2.0,
                },
                "fixed_param_1": {
                    "path": "child",
                    "param": {
                        "description": "offset",
                        "default": "1.0",
                        "fqn": f"{__name__}.FixedParameterSubscanLeaf.offset",
                        "spec": {
                            "is_scannable": True,
                            "scale": 1.0,
                            "step": 0.1,
                        },
                        "type": "float",
                    },
                    "value": 1.0,
                },
            },
        )

    def test_nested_child_scan_records_segment_start_unix_timestamps(self):
        parent = self.create(NestedChildScanParent, [])
        request = ScanRequest.explicit([parent.outer], [[10.0], [20.0]])

        with patch(
            "ndscan.runtime.executors.time.time",
            side_effect=[
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
            ],
        ):
            session = PreparedScan(parent, parent, request)
            session.execute()

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

        session = PreparedScan(parent, parent, request)
        session.execute()

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
            self.d(middle_prefix, "points.param_0"),
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
            self.d(leaf_prefix, "points.param_0"),
            [1.0, 1.5, 11.0, 11.5, 2.0, 2.5, 12.0, 12.5],
        )
        self.assertEqual(
            self.d(leaf_prefix, "points.channel_0"),
            [2.0, 2.5, 12.0, 12.5, 3.0, 3.5, 13.0, 13.5],
        )

    def test_prepared_child_scan_merges_site_metadata_and_can_inherit_unsegmented_sites(self):
        parent = self.create(NestedHelperSiteOptionsParent, [])
        session = PreparedScan(parent, parent, ScanRequest.single())
        session.execute()

        child_prefix = "ndscan.rid_0.site.root.custom_child."
        self.assertEqual(self.d(child_prefix, "points.channel_0"), [3.0])
        self.assertEqual(self.d(child_prefix, "extra.from_request"), "request")
        self.assertEqual(self.d(child_prefix, "extra.from_call"), "call")
        self.assertNotIn(child_prefix + "segments.start_index", self.dataset_db.data)
        self.assertNotIn(child_prefix + "state.current_segment", self.dataset_db.data)

    def test_nested_default_analyses_chain_matches_example_style(self):
        parent = self.create(AnalysedHowDoesPVaryFragment, [])
        session = PreparedScan(parent, parent, ScanRequest.single())
        result = _execute_and_inspect(session)

        root_prefix = "ndscan.rid_0.site.root."
        p_prefix = "ndscan.rid_0.site.root.scan_p."
        x_prefix = "ndscan.rid_0.site.root.scan_p.scan_x."

        self.assertAlmostEqual(
            self.d(root_prefix, "points.channel_0")[0],
            2.0,
            places=6,
        )
        self.assertEqual(
            self.d(p_prefix, "points.param_0"),
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        np.testing.assert_allclose(
            self.d(p_prefix, "points.channel_0"),
            [1.0, 4.0, 9.0, 16.0, 25.0],
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(self.d(x_prefix, "segments.start_index"), [0, 6, 12, 18, 24])
        segment_feedback = [
            json.loads(item)
            for item in self.d(x_prefix, "segments.analysis.final_feedback")
        ]
        self.assertEqual(len(segment_feedback), 5)
        self.assertAlmostEqual(segment_feedback[0]["outputs"]["m"], 1.0, places=6)
        self.assertAlmostEqual(segment_feedback[-1]["outputs"]["m"], 25.0, places=6)
        self.assertEqual(
            segment_feedback[0]["annotations"][0]["kind"], "artifact_curve"
        )
        self.assertAlmostEqual(result.values[parent.fit_e][0], 2.0, places=6)
        self.assertAlmostEqual(self.d(p_prefix, "analysis.output.fit_e"), 2.0)
        self.assertAlmostEqual(self.d(x_prefix, "analysis.output.m"), 25.0)

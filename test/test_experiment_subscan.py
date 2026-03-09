"""
Tests for subscan functionality.
"""

import json

from fixtures import (
    AddOneCustomAnalysisFragment,
    AddOneFragment,
    ReboundAddOneFragment,
    TwoAnalysisAggregate,
    TwoAnalysisFragment,
)
from mock_environment import ExpFragmentCase, HasEnvironmentCase

from ndscan.experiment import *
from ndscan.utils import SCHEMA_REVISION, SCHEMA_REVISION_KEY


class Scan1DFragment(ExpFragment):
    def build_fragment(self, klass):
        self.setattr_fragment("child", klass)
        scan = setattr_subscan(self, "scan", self.child, [(self.child, "value")])
        assert self.scan == scan

    def run_once(self):
        return self.scan.run(
            [(self.child.value, LinearGenerator(0, 3, 4, False))],
            ScanOptions(seed=1234),
        )[:2]


class TwoAxisValueFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("a", FloatParam, "a", default=0.0)
        self.setattr_param("b", FloatParam, "b", default=0.0)
        self.setattr_result("result", FloatChannel)

    def run_once(self):
        self.result.push(self.a.get() * 100.0 + self.b.get())


class Scan2DZipFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("child", TwoAxisValueFragment)
        setattr_subscan(
            self,
            "scan",
            self.child,
            [(self.child, "a"), (self.child, "b")],
            expose_analysis_results=False,
        )

    def run_once(self):
        return self.scan.run(
            [
                (self.child.a, ListGenerator([1.0, 2.0, 4.0], False)),
                (self.child.b, ListGenerator([10.0, 20.0, 40.0], False)),
            ],
            ScanOptions(seed=1234),
            strategy="zip",
            execute_default_analyses=False,
        )[:2]


class Scan2DPointListFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("child", TwoAxisValueFragment)
        setattr_subscan(
            self,
            "scan",
            self.child,
            [(self.child, "a"), (self.child, "b")],
            expose_analysis_results=False,
        )

    def run_once(self):
        return self.scan.run(
            [
                (self.child.a, ListGenerator([0.0], False)),
                (self.child.b, ListGenerator([0.0], False)),
            ],
            ScanOptions(seed=1234),
            strategy={
                "kind": "point_list",
                "points": [[1.0, 10.0], [4.0, 40.0], [2.0, 20.0]],
            },
            execute_default_analyses=False,
        )[:2]


class SubscanCase(ExpFragmentCase):
    def test_1d_subscan_return(self):
        parent = self.create(Scan1DFragment, AddOneFragment)
        self._test_1d(parent, parent.child.result)

    def test_1d_rebound_subscan_return(self):
        parent = self.create(Scan1DFragment, ReboundAddOneFragment)
        self._test_1d(parent, parent.child.add_one.result)

    def test_2d_zip_subscan_return(self):
        parent = self.create(Scan2DZipFragment)
        coords, values = parent.run_once()
        self.assertEqual(coords[parent.child.a], [1.0, 2.0, 4.0])
        self.assertEqual(coords[parent.child.b], [10.0, 20.0, 40.0])
        self.assertEqual(values[parent.child.result], [110.0, 220.0, 440.0])

    def test_2d_point_list_subscan_return(self):
        parent = self.create(Scan2DPointListFragment)
        coords, values = parent.run_once()
        self.assertEqual(coords[parent.child.a], [1.0, 4.0, 2.0])
        self.assertEqual(coords[parent.child.b], [10.0, 40.0, 20.0])
        self.assertEqual(values[parent.child.result], [110.0, 440.0, 220.0])

    def test_2d_strategy_is_exposed_in_subscan_spec(self):
        zip_parent = self.create(Scan2DZipFragment)
        zip_results = run_fragment_once(zip_parent)
        zip_spec = json.loads(zip_results[zip_parent.scan_spec])
        self.assertEqual(zip_spec["strategy"], "zip")

        point_parent = self.create(Scan2DPointListFragment)
        point_results = run_fragment_once(point_parent)
        point_spec = json.loads(point_results[point_parent.scan_spec])
        self.assertEqual(point_spec["strategy"], "point_list")

    def _test_1d(self, parent, result_channel):
        coords, values = parent.run_once()

        expected_values = [float(n) for n in range(0, 4)]
        expected_results = [v + 1 for v in expected_values]
        self.assertEqual(coords, {parent.child.value: expected_values})
        self.assertEqual(values, {result_channel: expected_results})

    def test_1d_result_channels(self):
        parent = self.create(Scan1DFragment, AddOneFragment)
        results = run_fragment_once(parent)

        expected_values = [float(n) for n in range(0, 4)]
        expected_results = [v + 1 for v in expected_values]
        self.assertEqual(results[parent.scan_axis_0], expected_values)
        self.assertEqual(results[parent.scan_channel_result], expected_results)

        spec = json.loads(results[parent.scan_spec])
        self.assertEqual(spec["fragment_fqn"], "fixtures.AddOneFragment")
        self.assertEqual(spec["seed"], 1234)

        curve_annotation = {
            "kind": "computed_curve",
            "parameters": {
                "function_name": "lorentzian",
                "associated_channels": ["channel_result"],
            },
            "coordinates": {},
            "data": {
                "a": {
                    "analysis_name": "fit_lorentzian_channel_result",
                    "kind": "online_result",
                    "result_key": "a",
                },
                "fwhm": {
                    "analysis_name": "fit_lorentzian_channel_result",
                    "kind": "online_result",
                    "result_key": "fwhm",
                },
                "x0": {
                    "analysis_name": "fit_lorentzian_channel_result",
                    "kind": "online_result",
                    "result_key": "x0",
                },
                "y0": {
                    "analysis_name": "fit_lorentzian_channel_result",
                    "kind": "online_result",
                    "result_key": "y0",
                },
            },
        }
        location_annotation = {
            "kind": "location",
            "parameters": {"associated_channels": ["channel_result"]},
            "coordinates": {
                "axis_0": {
                    "analysis_name": "fit_lorentzian_channel_result",
                    "kind": "online_result",
                    "result_key": "x0",
                }
            },
            "data": {
                "axis_0_error": {
                    "analysis_name": "fit_lorentzian_channel_result",
                    "kind": "online_result",
                    "result_key": "x0_error",
                }
            },
        }
        self.assertEqual(spec["annotations"], [curve_annotation, location_annotation])
        self.assertEqual(
            spec["online_analyses"],
            {
                "fit_lorentzian_channel_result": {
                    "constants": {"y0": 1.0},
                    "data": {"y": "channel_result", "x": "axis_0"},
                    "fit_type": "lorentzian",
                    "initial_values": {"fwhm": 2.0},
                    "kind": "named_fit",
                }
            },
        )
        self.assertEqual(
            spec["channels"],
            {
                "result": {
                    "description": "",
                    "scale": 1.0,
                    "path": "child/result",
                    "type": "float",
                    "unit": "",
                }
            },
        )
        self.assertEqual(
            spec["axes"],
            [
                {
                    "min": 0,
                    "max": 3,
                    "path": "child",
                    "param": {
                        "description": "Value to return",
                        "default": "0.0",
                        "fqn": "fixtures.AddOneFragment.value",
                        "spec": {"is_scannable": True, "scale": 1.0, "step": 0.1},
                        "type": "float",
                    },
                    "increment": 1.0,
                }
            ],
        )

    def test_1d_custom_analysis(self):
        parent = self.create(Scan1DFragment, AddOneCustomAnalysisFragment)
        results = run_fragment_once(parent)
        annotations = json.loads(results[parent.scan_spec])["annotations"]
        x_location = {
            "coordinates": {"axis_0": {"kind": "fixed", "value": 1.5}},
            "data": {},
            "kind": "location",
            "parameters": {},
        }
        y_location = {
            "coordinates": {"channel_result": {"kind": "fixed", "value": 2.5}},
            "data": {},
            "kind": "location",
            "parameters": {},
        }
        # FIXME: This should probably use fuzzy comparison for the floating point
        # values.
        self.assertEqual(annotations, [x_location, y_location])

    def test_fragment_detach(self):
        parent = self.create(Scan1DFragment, AddOneFragment)
        run_fragment_once(parent)

        # Make sure the setup and cleanup methods aren't also called during the parent
        # fragment setup/cleanup (in addition to the subscan).
        self.assertEqual(parent.child.num_host_setup_calls, 1)
        self.assertEqual(parent.child.num_device_setup_calls, 4)
        self.assertEqual(parent.child.num_device_cleanup_calls, 1)
        self.assertEqual(parent.child.num_host_cleanup_calls, 1)


class RunSubscanTwiceFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("child", AddOneFragment)
        setattr_subscan(
            self,
            "scan",
            self.child,
            [(self.child, "value")],
            expose_analysis_results=False,
        )

    def run_once(self):
        r0 = self.scan.run([(self.child.value, LinearGenerator(0, 3, 4, False))])
        r1 = self.scan.run([(self.child.value, LinearGenerator(4, 7, 4, False))])
        return r0, r1


class RelationSubscanChildFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("p", FloatParam, "p", default=0.0)
        self.setattr_param("q", FloatParam, "q", default=0.0)
        self.bind_param_relation("q", [self.p], lambda p: p**2 + 4.0)
        self.setattr_result("result", FloatChannel)

    def run_once(self):
        self.result.push(self.q.get())


class RelationSubscanParentFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("child", RelationSubscanChildFragment)
        setattr_subscan(
            self,
            "scan",
            self.child,
            [(self.child, "p")],
            expose_analysis_results=False,
        )

    def run_once(self):
        self.scan.run(
            [(self.child.p, ListGenerator([3.0, 6.0, 7.0], False))],
            ScanOptions(seed=1234),
            execute_default_analyses=False,
        )


class RunSubscanTwiceCase(ExpFragmentCase):
    def test_1d_subscan_twice(self):
        parent = self.create(RunSubscanTwiceFragment)
        results = parent.run_once()

        for base, (coords, values, _) in zip([0, 4], results):
            expected_values = [float(n) for n in range(base, base + 4)]
            expected_results = [v + 1 for v in expected_values]
            self.assertEqual(coords, {parent.child.value: expected_values})
            self.assertEqual(values, {parent.child.result: expected_results})


class SubscanPreviewDatasetCase(ExpFragmentCase):
    def test_preview_stream_is_reset_between_subscan_runs(self):
        parent = self.create(RunSubscanTwiceFragment)
        parent.run_once()

        def d(key):
            return self.dataset_db.get(parent.scan._preview_dataset_prefix + key)

        self.assertEqual(d(SCHEMA_REVISION_KEY), SCHEMA_REVISION)
        self.assertEqual(d("completed"), True)
        self.assertEqual(d("source_id"), "rid_0")
        self.assertEqual(d("fragment_fqn"), "fixtures.AddOneFragment")
        self.assertEqual(d("points.axis_0"), [4.0, 5.0, 6.0, 7.0])
        self.assertEqual(d("points.channel_result"), [5.0, 6.0, 7.0, 8.0])


class SubscanFlatDatasetCase(ExpFragmentCase):
    def test_flat_stream_appends_across_subscan_runs_with_segments(self):
        parent = self.create(RunSubscanTwiceFragment)
        parent.run_once()

        def d(key):
            return self.dataset_db.get(parent.scan._flat_dataset_prefix + key)

        def j(key):
            value = d(key)
            if isinstance(value, str):
                return json.loads(value)
            return value

        self.assertEqual(d("points.axis_0"), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(
            d("points.channel_result"),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        self.assertEqual(d("starts"), [0, 4])
        self.assertEqual(d(SCHEMA_REVISION_KEY), SCHEMA_REVISION)
        self.assertEqual(d("source_id"), "rid_0")
        self.assertEqual(d("completed"), True)
        self.assertEqual(d("fragment_fqn"), "fixtures.AddOneFragment")
        self.assertEqual(d("strategy"), "grid")
        self.assertEqual(j("segment_fields"), {"starts": "starts"})

        axes = j("axes")
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0]["param"]["fqn"], "fixtures.AddOneFragment.value")
        self.assertNotIn("default", axes[0]["param"])

        channels = j("channels")
        self.assertIn("result", channels)
        self.assertEqual(channels["result"]["path"], "child/result")

        flat_prefix = parent.scan._flat_dataset_prefix
        self.assertIn("__", flat_prefix)
        flat_keys = sorted(
            key for key in self.dataset_db.data.keys() if key.startswith(flat_prefix)
        )
        self.assertEqual(
            set(flat_keys),
            {
                flat_prefix + SCHEMA_REVISION_KEY,
                flat_prefix + "axes",
                flat_prefix + "channels",
                flat_prefix + "completed",
                flat_prefix + "fragment_fqn",
                flat_prefix + "points.axis_0",
                flat_prefix + "points.channel_result",
                flat_prefix + "seed",
                flat_prefix + "segment_fields",
                flat_prefix + "strategy",
                flat_prefix + "source_id",
                flat_prefix + "starts",
            },
        )

    def test_flat_stream_records_relation_target_param(self):
        parent = self.create(RelationSubscanParentFragment)
        parent.run_once()

        flat_prefix = parent.scan._flat_dataset_prefix

        def d(key):
            return self.dataset_db.get(flat_prefix + key)

        self.assertEqual(d("points.axis_0"), [3.0, 6.0, 7.0])
        self.assertEqual(d("points.channel_result"), [13.0, 40.0, 53.0])
        self.assertEqual(d("points.param_q"), [13.0, 40.0, 53.0])
        self.assertNotIn(flat_prefix + "points.param_p", self.dataset_db.data)


class CharacteriseBulkPushFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("child", AddOneFragment)
        setattr_subscan(
            self,
            "scan",
            self.child,
            [(self.child, "value")],
            expose_analysis_results=False,
        )

    def run_once(self):
        self.scan.run(
            [(self.child.value, LinearGenerator(0.0, 3.0, 4, False))],
            execute_default_analyses=False,
        )


class BulkPushCharacterisationCase(ExpFragmentCase):
    def test_aggregate_channels_receive_one_push_per_subscan(self):
        parent = self.create(CharacteriseBulkPushFragment)

        axis_sink = ArraySink()
        result_sink = ArraySink()
        parent.scan_axis_0.set_sink(axis_sink)
        parent.scan_channel_result.set_sink(result_sink)

        parent.run_once()
        parent.run_once()

        expected_axis_row = [0.0, 1.0, 2.0, 3.0]
        expected_result_row = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(axis_sink.get_all(), [expected_axis_row, expected_axis_row])
        self.assertEqual(result_sink.get_all(), [expected_result_row, expected_result_row])


class RaggedSubscanFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param(
            "num_scan_points", IntParam, "Number of subscan points", default=2, min=2
        )
        self.setattr_fragment("child", AddOneFragment)
        setattr_subscan(
            self,
            "scan",
            self.child,
            [(self.child, "value")],
            expose_analysis_results=False,
        )

    def run_once(self):
        n = self.num_scan_points.get()
        self.scan.run(
            [
                (
                    self.child.value,
                    LinearGenerator(0.0, float(n - 1), n, False),
                )
            ],
            execute_default_analyses=False,
        )


RaggedSubscanFragmentScan = make_fragment_scan_exp(RaggedSubscanFragment)


class NestedRaggedLeafFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("p", FloatParam, "p", default=0.0)
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("result")

    def run_once(self):
        self.result.push(self.p.use() + self.x.use())


class NestedRaggedInnerSubscan(SubscanExpFragment):
    def build_fragment(self):
        self.setattr_fragment("leaf", NestedRaggedLeafFragment)
        super().build_fragment(
            self,
            "leaf",
            [(self.leaf, "x")],
            expose_analysis_results=False,
        )

    def _configure(self):
        p_now = self.leaf.p.use()
        n = 2 + (int(p_now) % 3) * 2
        self.configure(
            [(self.leaf.x, LinearGenerator(0.0, 5.0, n, False))],
            options=ScanOptions(),
        )

    def host_setup(self):
        self._configure()
        super().host_setup()

    def device_setup(self):
        self._configure()
        self.device_setup_subfragments()


class NestedRaggedOuterSubscan(SubscanExpFragment):
    def build_fragment(self):
        self.setattr_fragment("inner", NestedRaggedInnerSubscan)
        super().build_fragment(
            self,
            "inner",
            [(self.inner.leaf, "p")],
            expose_analysis_results=False,
        )

    def host_setup(self):
        self.configure(
            [(self.inner.leaf.p, LinearGenerator(0.0, 5.0, 10, False))],
            options=ScanOptions(),
        )
        super().host_setup()


NestedRaggedOuterSubscanScan = make_fragment_scan_exp(NestedRaggedOuterSubscan)


class NestedStrategyLeafFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("p", FloatParam, "p", default=0.0)
        self.setattr_param("z", FloatParam, "z", default=0.0)
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_param("y", FloatParam, "y", default=0.0)
        self.setattr_result("result", FloatChannel)

    def run_once(self):
        self.result.push(
            self.p.get() * 1000.0
            + self.z.get() * 100.0
            + self.x.get() * 10.0
            + self.y.get()
        )


class NestedStrategyInnerSubscan(SubscanExpFragment):
    def build_fragment(self):
        self.setattr_fragment("leaf", NestedStrategyLeafFragment)
        super().build_fragment(
            self,
            "leaf",
            [(self.leaf, "x"), (self.leaf, "y")],
            expose_analysis_results=False,
        )

    def _configure(self):
        self.configure(
            [
                (self.leaf.x, ListGenerator([1.0, 2.0], False)),
                (self.leaf.y, ListGenerator([10.0, 20.0], False)),
            ],
            options=ScanOptions(seed=11),
            strategy="zip",
        )

    def host_setup(self):
        self._configure()
        super().host_setup()

    def device_setup(self):
        self._configure()
        self.device_setup_subfragments()


class NestedStrategyOuterSubscan(SubscanExpFragment):
    def build_fragment(self):
        self.setattr_fragment("inner", NestedStrategyInnerSubscan)
        super().build_fragment(
            self,
            "inner",
            [(self.inner.leaf, "p"), (self.inner.leaf, "z")],
            expose_analysis_results=False,
        )

    def _configure(self):
        self.configure(
            [
                (self.inner.leaf.p, ListGenerator([0.0], False)),
                (self.inner.leaf.z, ListGenerator([0.0], False)),
            ],
            options=ScanOptions(seed=22),
            strategy={
                "kind": "point_list",
                "points": [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            },
        )

    def host_setup(self):
        self._configure()
        super().host_setup()

    def device_setup(self):
        self._configure()
        self.device_setup_subfragments()


class RaggedSubscanDatasetCase(HasEnvironmentCase):
    def test_nested_ragged_preview_is_broadcast_only(self):
        exp = self.create(NestedRaggedOuterSubscanScan)
        exp.prepare()
        exp.run()

        dataset_mgr = exp._HasEnvironment__dataset_mgr
        preview_keys = [k for k in self.dataset_db.data.keys() if ".subscan_preview." in k]
        self.assertTrue(preview_keys)

        ragged_preview_keys = []
        for key in preview_keys:
            value = self.dataset_db.data[key][1]
            if isinstance(value, list) and value and all(
                isinstance(v, list) for v in value
            ):
                lengths = {len(v) for v in value}
                if len(lengths) > 1:
                    ragged_preview_keys.append(key)

        self.assertTrue(ragged_preview_keys)
        for key in ragged_preview_keys:
            self.assertNotIn(key, dataset_mgr.local)

    def test_nested_ragged_outputs_are_not_copied_into_parent_flat_root(self):
        exp = self.create(NestedRaggedOuterSubscanScan)
        exp.prepare()
        exp.run()

        dataset_mgr = exp._HasEnvironment__dataset_mgr
        parent_flat_channel_keys = {
            key
            for key in dataset_mgr.local.keys()
            if ".subscan_flat.root_subscan_nestedraggedinnersubscan" in key
            and ".points.channel__" in key
        }
        self.assertFalse(
            any(key.endswith(".points.channel__axis_0") for key in parent_flat_channel_keys)
        )
        self.assertFalse(
            any(
                key.endswith(".points.channel__channel_result")
                for key in parent_flat_channel_keys
            )
        )

    def test_legacy_ragged_channels_are_not_archived(self):
        exp = self.create(RaggedSubscanFragmentScan)
        fragment_fqn = "test_experiment_subscan.RaggedSubscanFragment"
        exp.args._params["scan"]["axes"].append(
            {
                "fqn": fragment_fqn + ".num_scan_points",
                "path": "*",
                "type": "list",
                "range": {
                    "values": [2, 4, 3],
                    "randomise_order": False,
                },
            }
        )

        exp.prepare()
        exp.run()

        channel_by_path = {channel.path: channel for channel in exp.tlr._scan_result_sinks}
        axis_channel = channel_by_path["scan_axis_0"]
        value_channel = channel_by_path["scan_channel_result"]
        self.assertFalse(axis_channel.archive_by_default)
        self.assertFalse(value_channel.archive_by_default)

        axis_sink = exp.tlr._scan_result_sinks[axis_channel]
        value_sink = exp.tlr._scan_result_sinks[value_channel]
        self.assertFalse(axis_sink.archive)
        self.assertFalse(value_sink.archive)

        dataset_mgr = exp._HasEnvironment__dataset_mgr
        self.assertNotIn("ndscan.rid_0.points.channel_scan_axis_0", dataset_mgr.local)
        self.assertNotIn("ndscan.rid_0.points.channel_scan_channel_result", dataset_mgr.local)

    def test_ragged_subscan_is_written_as_list_of_lists(self):
        exp = self.create(RaggedSubscanFragmentScan)
        fragment_fqn = "test_experiment_subscan.RaggedSubscanFragment"
        exp.args._params["scan"]["axes"].append(
            {
                "fqn": fragment_fqn + ".num_scan_points",
                "path": "*",
                "type": "list",
                "range": {
                    "values": [2, 4, 3],
                    "randomise_order": False,
                },
            }
        )

        exp.prepare()
        exp.run()

        def d(key):
            return self.dataset_db.get("ndscan.rid_0." + key)

        self.assertEqual(d("points.axis_0"), [2, 4, 3])
        self.assertEqual(
            d("points.channel_scan_axis_0"),
            [[0.0, 1.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0]],
        )
        self.assertEqual(
            d("points.channel_scan_channel_result"),
            [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0]],
        )


class NestedStrategySubscanDatasetCase(ExpFragmentCase):
    def test_nested_strategy_subscan_writes_recursive_starts(self):
        parent = self.create(NestedStrategyOuterSubscan)
        run_fragment_once(parent)

        outer_prefix = parent._subscan._flat_dataset_prefix
        inner_prefix = parent.inner._subscan._flat_dataset_prefix

        def d(prefix, key):
            return self.dataset_db.get(prefix + key)

        self.assertEqual(d(outer_prefix, "strategy"), "point_list")
        self.assertEqual(d(inner_prefix, "strategy"), "zip")

        # One outer subscan execution with three points.
        self.assertEqual(d(outer_prefix, "starts"), [0])
        self.assertEqual(d(outer_prefix, "points.axis_0"), [1.0, 2.0, 3.0])
        self.assertEqual(d(outer_prefix, "points.axis_1"), [4.0, 5.0, 6.0])

        # Inner subscan runs once per outer point and has two zip points each time.
        self.assertEqual(d(inner_prefix, "starts"), [0, 2, 4])
        self.assertEqual(d(inner_prefix, "points.axis_0"), [1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        self.assertEqual(
            d(inner_prefix, "points.axis_1"),
            [10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
        )
        self.assertEqual(
            d(inner_prefix, "points.channel_result"),
            [1420.0, 1440.0, 2520.0, 2540.0, 3620.0, 3640.0],
        )


class SubscanAnalysisFragment(ExpFragment):
    def build_fragment(
        self, declare_both_scannable=False, always_execute_analyses=True
    ):
        self.always_execute_analyses = always_execute_analyses
        self.setattr_fragment("child", TwoAnalysisFragment)
        axes = [(self.child, "a")]
        if declare_both_scannable:
            axes.append((self.child, "b"))
        setattr_subscan(self, "scan", self.child, axes)
        self.had_result = False

    def run_once(self):
        _, _, analysis_results = self.scan.run(
            [(self.child.a, LinearGenerator(0.0, 1.0, 3, True))],
            execute_default_analyses=self.always_execute_analyses,
        )
        self.had_result = "result_a" in analysis_results


class AggregateSubscanAnalysisFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("child", TwoAnalysisAggregate)
        setattr_subscan(self, "scan", self.child, [(self.child, "a")])
        self.had_all_results = False

    def run_once(self):
        _, _, analysis_results = self.scan.run(
            [(self.child.a, LinearGenerator(0.0, 1.0, 3, True))]
        )
        self.had_all_results = all(
            f"{n}_result_a" in analysis_results for n in ("first", "second")
        )


class SubscanAnalysisCase(ExpFragmentCase):
    def test_simple_filtering(self):
        parent = self.create(SubscanAnalysisFragment)
        results = run_fragment_once(parent)
        spec = json.loads(results[parent.scan_spec])
        self.assertEqual(spec["analysis_results"], {"result_a": "scan_result_a"})
        self.assertEqual(results[parent.scan_result_a], 42.0)
        self.assertTrue(parent.had_result)

    def _test_subset_filtering(self, always_execute_analyses):
        parent = self.create(
            SubscanAnalysisFragment,
            declare_both_scannable=True,
            always_execute_analyses=always_execute_analyses,
        )
        results = run_fragment_once(parent)
        spec = json.loads(results[parent.scan_spec])

        # Shouldn't have a result channel, since it wasn't statically known which
        # axes would be scanned.
        self.assertEqual(spec.get("analysis_results", {}), {})

        # If requested, the analysis should have still been executed at run()-time,
        # though.
        self.assertEqual(parent.had_result, always_execute_analyses)

    def test_subset_filtering(self):
        self._test_subset_filtering(False)

    def test_subset_filtering_2(self):
        self._test_subset_filtering(True)

    def test_aggregate(self):
        # For simplicity, test AggregateExpFragment through an actual subscan instead of
        # manually verifying the analysis result handling/…
        parent = self.create(AggregateSubscanAnalysisFragment)
        results = run_fragment_once(parent)
        self.assertTrue(parent.had_all_results)
        self.assertEqual(results[parent.scan_first_result_a], 42.0)
        self.assertEqual(results[parent.scan_second_result_a], 42.0)

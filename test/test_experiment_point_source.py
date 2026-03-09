import unittest

import numpy as np
from mock_environment import ExpFragmentCase

from ndscan.experiment import (
    ExpFragment,
    FloatChannel,
    FloatParam,
    LinearGenerator,
    ListGenerator,
)
from ndscan.experiment.point_source import (
    IteratorPointSource,
    PointObservation,
    StrategyPointSource,
)
from ndscan.experiment.scan_generator import ScanOptions, generate_points
from ndscan.experiment.scan_runner import HostScanRunner, ScanAxis, ScanSpec


class PointSourceCase(unittest.TestCase):
    def test_iterator_point_source_next_and_take(self):
        source = IteratorPointSource(iter([(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]))
        self.assertEqual(source.next_point(), (1.0, 10.0))
        self.assertEqual(source.take_points(2), [(2.0, 20.0), (3.0, 30.0)])
        self.assertIsNone(source.next_point())

    def test_iterator_point_source_rejects_negative_chunk(self):
        source = IteratorPointSource(iter([]))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            source.take_points(-1)

    def test_strategy_point_source_matches_generate_points_zip(self):
        axis_generators = [
            ListGenerator([1.0, 2.0, 4.0], False),
            ListGenerator([10.0, 20.0, 40.0], False),
        ]
        options = ScanOptions(seed=5)
        expected = list(generate_points(axis_generators, options, strategy="zip"))

        source = StrategyPointSource(axis_generators, options, "zip")
        actual = []
        while True:
            point = source.next_point()
            if point is None:
                break
            actual.append(point)

        self.assertEqual(actual, expected)

    def test_strategy_point_source_take_points_chunks(self):
        axis_generators = [
            ListGenerator([0.0], False),
            ListGenerator([0.0], False),
        ]
        options = ScanOptions(seed=7)
        strategy = {"kind": "point_list", "points": [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]}
        source = StrategyPointSource(axis_generators, options, strategy)

        self.assertEqual(source.take_points(2), [(1.0, 10.0), (2.0, 20.0)])
        self.assertEqual(source.take_points(2), [(3.0, 30.0)])
        self.assertEqual(source.take_points(1), [])

    def test_adaptive_gaussian_1d_strategy_uses_observations(self):
        axis_generators = [LinearGenerator(0.0, 10.0, 2, False)]
        options = ScanOptions(seed=11)
        strategy = {
            "kind": "adaptive",
            "driver": "gaussian_1d",
            "config": {
                "lower": 0.0,
                "upper": 10.0,
                "num_points": 9,
                "warmup_points": 5,
                "candidate_count": 64,
            },
        }
        source = StrategyPointSource(axis_generators, options, strategy)

        taken = []
        for i in range(9):
            point = source.next_point()
            self.assertIsNotNone(point)
            x = point[0]
            taken.append(x)
            y = float(np.exp(-0.5 * ((x - 6.0) / 1.1) ** 2))
            source.observe(PointObservation(i, (x,), {"y": y}))

        self.assertIsNone(source.next_point())

        expected_warmup = np.linspace(0.0, 10.0, 5, endpoint=True).tolist()
        self.assertEqual(taken[:5], expected_warmup)
        self.assertTrue(any(4.0 <= x <= 8.0 for x in taken[5:]))
        diagnostics = source.diagnostics()
        self.assertEqual(diagnostics["driver"], "gaussian_1d")
        self.assertEqual(diagnostics["num_emitted"], 9)
        self.assertEqual(diagnostics["num_observed"], 9)

    def test_legacy_gaussian_adaptive_1d_strategy_still_supported(self):
        axis_generators = [LinearGenerator(0.0, 10.0, 2, False)]
        source = StrategyPointSource(
            axis_generators,
            ScanOptions(seed=3),
            {"kind": "gaussian_adaptive_1d", "num_points": 3, "warmup_points": 3},
        )
        self.assertEqual(source.preferred_batch_size(10), 1)
        self.assertEqual(source.next_point(), (0.0,))
        self.assertEqual(source.next_point(), (5.0,))
        self.assertEqual(source.next_point(), (10.0,))
        self.assertIsNone(source.next_point())

    def test_non_adaptive_strategy_uses_default_batch_size(self):
        source = StrategyPointSource(
            [ListGenerator([1.0, 2.0], False)],
            ScanOptions(seed=2),
            "grid",
        )
        self.assertEqual(source.preferred_batch_size(10), 10)


class RecordingPointSource(StrategyPointSource):
    def __init__(self, axis_generators, options, strategy="grid"):
        super().__init__(axis_generators, options, strategy)
        self.observations: list[PointObservation] = []

    def observe(self, observation: PointObservation) -> None:
        self.observations.append(observation)


class ObserveSourceFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push(self.x.get() + 1.0)


class PointSourceObserveCase(ExpFragmentCase):
    def test_runner_emits_point_observations(self):
        fragment = self.create(ObserveSourceFragment)
        fragment.y.set_sink(ListSink())
        axis = ScanAxis(fragment.x.parameter.describe(), "*", fragment.x._store)
        axis_values = [0.0, 2.0, 4.0]

        source = RecordingPointSource(
            [ListGenerator(axis_values, False)],
            ScanOptions(seed=1),
            "grid",
        )

        axis_sink = ListSink()
        spec = ScanSpec([axis], [ListGenerator(axis_values, False)], ScanOptions(seed=1))
        runner = HostScanRunner(fragment)
        runner.setup(fragment, spec.axes, [axis_sink], [])
        runner.set_point_source(source)
        fragment.host_setup()
        try:
            complete = runner.acquire(device_cleanup=True)
        finally:
            fragment.host_cleanup()
        self.assertTrue(complete)
        self.assertEqual(axis_sink.values, axis_values)

        self.assertEqual(len(source.observations), 3)
        for i, observation in enumerate(source.observations):
            self.assertEqual(observation.point_index, i)
            self.assertEqual(observation.axis_values, (axis_values[i],))
            self.assertIn("y", observation.result_values)
            self.assertEqual(observation.result_values["y"], axis_values[i] + 1.0)
            self.assertEqual(
                observation.axis_by_param,
                {("test_experiment_point_source.ObserveSourceFragment.x", "*"): axis_values[i]},
            )


class ListSink:
    def __init__(self):
        self.values = []

    def push(self, value):
        self.values.append(value)

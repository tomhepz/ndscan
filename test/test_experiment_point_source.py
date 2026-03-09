import unittest

from ndscan.experiment.point_source import IteratorPointSource, StrategyPointSource
from ndscan.experiment.scan_generator import ListGenerator, ScanOptions, generate_points


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

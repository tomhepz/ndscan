import unittest

from ndscan.experiment.scan_generator import (
    ListGenerator,
    RefiningGenerator,
    ScanOptions,
    generate_points,
)
from ndscan.experiment.scan_point_strategies import generate_points_for_strategy


class ScanPointStrategiesCase(unittest.TestCase):
    def test_grid_cartesian_order(self):
        points = list(
            generate_points_for_strategy(
                [
                    ListGenerator([1.0, 2.0], False),
                    ListGenerator([10.0, 20.0], False),
                ],
                ScanOptions(seed=1),
                "grid",
            )
        )
        self.assertEqual(
            points,
            [(1.0, 10.0), (2.0, 10.0), (1.0, 20.0), (2.0, 20.0)],
        )

    def test_zip_lockstep(self):
        points = list(
            generate_points_for_strategy(
                [
                    ListGenerator([1.0, 2.0, 4.0], False),
                    ListGenerator([10.0, 20.0, 40.0], False),
                ],
                ScanOptions(seed=1),
                "zip",
            )
        )
        self.assertEqual(points, [(1.0, 10.0), (2.0, 20.0), (4.0, 40.0)])

    def test_point_list_uses_explicit_rows(self):
        points = list(
            generate_points_for_strategy(
                [
                    ListGenerator([0.0], False),
                    ListGenerator([0.0], False),
                ],
                ScanOptions(seed=1),
                {"kind": "point_list", "points": [[4.0, 40.0], [2.0, 20.0]]},
            )
        )
        self.assertEqual(points, [(4.0, 40.0), (2.0, 20.0)])

    def test_zip_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal number of points"):
            list(
                generate_points_for_strategy(
                    [
                        ListGenerator([1.0, 2.0], False),
                        ListGenerator([10.0], False),
                    ],
                    ScanOptions(seed=1),
                    "zip",
                )
            )

    def test_zip_rejects_refining_generator(self):
        with self.assertRaisesRegex(ValueError, "single-level generators"):
            list(
                generate_points_for_strategy(
                    [
                        RefiningGenerator(0.0, 1.0, False),
                        ListGenerator([10.0, 20.0], False),
                    ],
                    ScanOptions(seed=1),
                    "zip",
                )
            )

    def test_generate_points_wrapper_delegates(self):
        options = ScanOptions(num_repeats=2, num_repeats_per_point=2, seed=1)
        points = list(
            generate_points(
                [
                    ListGenerator([0.0], False),
                    ListGenerator([0.0], False),
                ],
                options,
                {"kind": "point_list", "points": [[1.0, 10.0], [2.0, 20.0]]},
            )
        )
        self.assertEqual(
            points,
            [
                (1.0, 10.0),
                (1.0, 10.0),
                (2.0, 20.0),
                (2.0, 20.0),
                (1.0, 10.0),
                (1.0, 10.0),
                (2.0, 20.0),
                (2.0, 20.0),
            ],
        )

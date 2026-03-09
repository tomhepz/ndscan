import unittest

from ndscan.experiment.scan_strategy_specs import (
    extract_point_list_rows,
    get_scan_strategy_kind,
    parse_scan_strategy,
)


class ScanStrategySpecsCase(unittest.TestCase):
    def test_get_scan_strategy_kind_from_string(self):
        self.assertEqual(get_scan_strategy_kind("grid", ValueError), "grid")

    def test_get_scan_strategy_kind_from_dict(self):
        self.assertEqual(
            get_scan_strategy_kind({"kind": "point_list", "points": []}, ValueError),
            "point_list",
        )

    def test_get_scan_strategy_kind_rejects_bad_kind(self):
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            get_scan_strategy_kind({"kind": 12}, ValueError)

    def test_parse_scan_strategy_default(self):
        strategy, kind = parse_scan_strategy({}, ValueError)
        self.assertEqual(strategy, "grid")
        self.assertEqual(kind, "grid")

    def test_parse_scan_strategy_allowed_kinds(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            parse_scan_strategy(
                {"strategy": "unknown"},
                ValueError,
                allowed_kinds={"grid", "zip", "point_list"},
            )

    def test_parse_scan_strategy_allows_adaptive_dict(self):
        strategy, kind = parse_scan_strategy(
            {"strategy": {"kind": "adaptive", "driver": "gaussian_1d", "config": {}}},
            ValueError,
            allowed_kinds={"grid", "zip", "point_list", "adaptive"},
        )
        self.assertEqual(kind, "adaptive")
        self.assertEqual(strategy["driver"], "gaussian_1d")

    def test_extract_point_list_rows_ignores_non_point_list(self):
        rows = extract_point_list_rows("zip", 2, error_type=ValueError)
        self.assertEqual(rows, [])

    def test_extract_point_list_rows_happy_path(self):
        rows = extract_point_list_rows(
            {"kind": "point_list", "points": [[1.0, 10.0], [2.0, 20.0]]},
            2,
            error_type=ValueError,
            require_non_empty=True,
        )
        self.assertEqual(rows, [[1.0, 10.0], [2.0, 20.0]])

    def test_extract_point_list_rows_rejects_empty_when_required(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            extract_point_list_rows(
                {"kind": "point_list", "points": []},
                2,
                error_type=ValueError,
                require_non_empty=True,
            )

    def test_extract_point_list_rows_rejects_row_shape(self):
        with self.assertRaisesRegex(ValueError, "row length must match"):
            extract_point_list_rows(
                {"kind": "point_list", "points": [[1.0], [2.0, 20.0]]},
                2,
                error_type=ValueError,
            )

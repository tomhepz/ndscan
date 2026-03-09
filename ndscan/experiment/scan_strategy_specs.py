"""Scan strategy spec parsing/validation helpers.

This keeps strategy-shape logic out of entry point and point generation code so the
same semantics can be reused by top-level scans and subscans.
"""

from typing import Any

__all__ = [
    "get_scan_strategy_kind",
    "parse_scan_strategy",
    "extract_point_list_rows",
]


def get_scan_strategy_kind(
    strategy: str | dict[str, Any], error_type: type[Exception]
) -> str:
    if isinstance(strategy, dict):
        kind = strategy.get("kind", "grid")
    else:
        kind = strategy
    if not isinstance(kind, str) or not kind:
        raise error_type("scan strategy kind must be a non-empty string")
    return kind


def parse_scan_strategy(
    scan: dict[str, Any],
    error_type: type[Exception],
    allowed_kinds: set[str] | None = None,
) -> tuple[str | dict[str, Any], str]:
    strategy = scan.get("strategy", "grid")
    kind = get_scan_strategy_kind(strategy, error_type)
    if allowed_kinds is not None and kind not in allowed_kinds:
        raise error_type(
            "scan strategy kind must be one of "
            + ", ".join(sorted(allowed_kinds))
            + f"; got '{kind}'"
        )
    return strategy, kind


def extract_point_list_rows(
    strategy: str | dict[str, Any],
    num_axes: int,
    *,
    error_type: type[Exception],
    require_non_empty: bool = False,
) -> list[list[Any] | tuple[Any, ...]]:
    kind = get_scan_strategy_kind(strategy, error_type)
    if kind != "point_list":
        return []
    if not isinstance(strategy, dict):
        raise error_type("point_list strategy must be a dict with 'points'")

    rows = strategy.get("points", [])
    if not isinstance(rows, list):
        raise error_type("scan.strategy.points must be a list")
    if require_non_empty and not rows:
        raise error_type("scan.strategy.points must not be empty")

    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)):
            raise error_type(f"scan.strategy.points[{i}] must be a list/tuple")
        if len(row) != num_axes:
            raise error_type(
                "scan.strategy.points row length must match number of axes, "
                f"got {len(row)} values for {num_axes} axes at row {i}"
            )
    return rows

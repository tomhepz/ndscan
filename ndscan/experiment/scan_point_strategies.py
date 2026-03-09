"""Point-stream composition strategies for scan execution.

This module combines per-axis generators into a flat linear stream of points.
It is intentionally separate from :mod:`scan_generator` so strategy logic can be
tested and evolved independently.
"""

from collections.abc import Iterator
from itertools import product
from typing import Any

import numpy as np

from .scan_strategy_specs import extract_point_list_rows, get_scan_strategy_kind

__all__ = ["generate_points_for_strategy"]


def generate_points_for_strategy(
    axis_generators: list[Any],
    options: Any,
    strategy: str | dict[str, Any] = "grid",
) -> Iterator[Any]:
    """Generate a flat linear stream of points for the requested strategy.

    ``grid`` keeps Cartesian/refining behaviour.
    ``zip`` walks axes in lockstep (point i of each axis together).
    ``point_list`` consumes explicit shot vectors from ``strategy["points"]``.
    """
    strategy_kind = get_scan_strategy_kind(strategy, ValueError)

    if strategy_kind == "grid":
        return _generate_grid_points(axis_generators, options)
    if strategy_kind == "zip":
        return _generate_zip_points(axis_generators, options)
    if strategy_kind == "point_list":
        rows = extract_point_list_rows(
            strategy,
            len(axis_generators),
            error_type=ValueError,
            require_non_empty=False,
        )
        return _generate_rows(rows, options)
    raise ValueError(f"Unknown scan strategy '{strategy_kind}'")


def _generate_grid_points(axis_generators: list[Any], options: Any) -> Iterator[Any]:
    rng = np.random.RandomState(options.seed)

    # Stores computed coordinates for each axis, indexed first by axis order, then by
    # level.
    axis_level_points = [[] for _ in axis_generators]

    max_level = 0
    while True:
        found_new_levels = False
        for i, gen in enumerate(axis_generators[::-1]):
            if gen.has_level(max_level):
                axis_level_points[i].append(gen.points_for_level(max_level, rng))
                found_new_levels = True

        if not found_new_levels:
            return

        points = []
        for axis_levels in product(*(range(len(p)) for p in axis_level_points)):
            if all(level < max_level for level in axis_levels):
                continue
            points.extend(
                product(*(axis_points[level] for level, axis_points in zip(axis_levels, axis_level_points)))
            )

        for _ in range(options.num_repeats):
            if options.randomise_order_globally:
                rng.shuffle(points)
            for point in points:
                for _ in range(options.num_repeats_per_point):
                    yield point[::-1]

        max_level += 1


def _generate_zip_points(axis_generators: list[Any], options: Any) -> Iterator[Any]:
    rng = np.random.RandomState(options.seed)

    # Zip mode is intentionally simple: take level-0 points from each axis and walk
    # them in lockstep. This gives a flat linear stream without Cartesian expansion.
    axis_points = []
    for i, gen in enumerate(axis_generators):
        if not gen.has_level(0):
            raise ValueError(f"Zip strategy axis {i} has no level-0 points")
        if gen.has_level(1):
            raise ValueError(
                "Zip strategy currently supports only single-level generators; "
                f"axis {i} has refinement levels"
            )
        axis_points.append(list(gen.points_for_level(0, rng)))

    if not axis_points:
        return

    lengths = [len(points) for points in axis_points]
    if len(set(lengths)) != 1:
        raise ValueError(
            "Zip strategy requires equal number of points across all axes, got "
            + ", ".join(str(n) for n in lengths)
        )

    rows = [tuple(row) for row in zip(*axis_points)]
    return _generate_rows(rows, options)


def _generate_rows(
    rows: list[list[Any] | tuple[Any, ...]], options: Any
) -> Iterator[Any]:
    rng = np.random.RandomState(options.seed)
    rows = [tuple(row) for row in rows]
    for _ in range(options.num_repeats):
        if options.randomise_order_globally:
            rng.shuffle(rows)
        for row in rows:
            for _ in range(options.num_repeats_per_point):
                yield row

"""Point source abstractions for scan runners.

Point sources provide scan points one-by-one (or in small chunks) independently of how
those points were produced.
"""

from collections.abc import Iterator
from itertools import islice
from typing import Any

from .scan_point_strategies import generate_points_for_strategy

__all__ = ["PointSource", "IteratorPointSource", "StrategyPointSource"]


class PointSource:
    """A source of scan points consumed by the scan runners."""

    def next_point(self):
        """Return the next point tuple, or ``None`` when exhausted."""
        raise NotImplementedError

    def take_points(self, max_points: int) -> list[tuple[Any, ...]]:
        """Return up to ``max_points`` points from the source."""
        if max_points < 0:
            raise ValueError("max_points must be non-negative")

        points = []
        for _ in range(max_points):
            point = self.next_point()
            if point is None:
                break
            points.append(point)
        return points


class IteratorPointSource(PointSource):
    """Point source backed by a plain iterator."""

    def __init__(self, points: Iterator[tuple[Any, ...]]) -> None:
        self._points = points

    def next_point(self):
        return next(self._points, None)

    def take_points(self, max_points: int) -> list[tuple[Any, ...]]:
        if max_points < 0:
            raise ValueError("max_points must be non-negative")
        return list(islice(self._points, max_points))


class StrategyPointSource(IteratorPointSource):
    """Point source produced from axis generators and a composition strategy."""

    def __init__(
        self,
        axis_generators: list[Any],
        options: Any,
        strategy: str | dict[str, Any] = "grid",
    ) -> None:
        super().__init__(generate_points_for_strategy(axis_generators, options, strategy))

"""Point source abstractions for scan runners.

Point sources provide scan points one-by-one (or in small chunks) independently of how
those points were produced.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

from .scan_point_strategies import create_point_strategy_driver

__all__ = [
    "PointObservation",
    "PointSource",
    "IteratorPointSource",
    "StrategyPointSource",
]


@dataclass(frozen=True)
class PointObservation:
    """Transient per-point execution feedback passed to point sources."""

    point_index: int
    axis_values: tuple[Any, ...]
    result_values: dict[str, Any]
    axis_by_param: dict[tuple[str, str], Any] | None = None


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

    def observe(self, observation: PointObservation) -> None:
        """Receive feedback for a completed point."""
        del observation

    def preferred_batch_size(self, default: int) -> int:
        """Return preferred host-fetch batch size for chunked runners."""
        return default

    def diagnostics(self) -> dict[str, Any] | None:
        """Return strategy diagnostics for optional telemetry/logging."""
        return None


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


class StrategyPointSource(PointSource):
    """Point source produced from axis generators and a composition strategy."""

    def __init__(
        self,
        axis_generators: list[Any],
        options: Any,
        strategy: str | dict[str, Any] = "grid",
    ) -> None:
        self._driver = create_point_strategy_driver(axis_generators, options, strategy)
        self._iterator_source = (
            IteratorPointSource(self._driver)
            if isinstance(self._driver, Iterator)
            else None
        )

    def next_point(self):
        if self._iterator_source is not None:
            return self._iterator_source.next_point()
        return self._driver.next_point()

    def take_points(self, max_points: int) -> list[tuple[Any, ...]]:
        if self._iterator_source is not None:
            return self._iterator_source.take_points(max_points)
        return self._driver.take_points(max_points)

    def observe(self, observation: PointObservation) -> None:
        if hasattr(self._driver, "observe"):
            self._driver.observe(observation)

    def preferred_batch_size(self, default: int) -> int:
        if hasattr(self._driver, "preferred_batch_size"):
            return self._driver.preferred_batch_size(default)
        return default

    def diagnostics(self) -> dict[str, Any] | None:
        if hasattr(self._driver, "diagnostics"):
            return self._driver.diagnostics()
        return None

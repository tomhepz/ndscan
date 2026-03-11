"""Point selection primitives for the new host-only runtime.

The existing ``scan_generator.py`` module is tightly coupled to the legacy runtime's
idea of a scan being described as one generator per axis and then expanded into a
Cartesian product. The new host-only runtime starts from a smaller abstraction:

``PointSource`` objects yield already-composed points.

This keeps point choice independent from fragment execution and dataset writing:

- point sources choose *what* to run next,
- the host runtime decides *how* to execute it,
- the scan-site writer decides *how* to persist the resulting observation.

The first implementation deliberately keeps the interface minimal and synchronous. It
is sufficient for grid scans, zipped scans, explicit point lists, and later adaptive
strategies that want to observe completed points before choosing the next one.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

__all__ = [
    "BasePoint",
    "PointSource",
    "SinglePointSource",
    "CartesianPointSource",
    "ZipPointSource",
    "ExplicitPointSource",
]


@dataclass(frozen=True)
class BasePoint:
    """A fully chosen point, expressed only in terms of axis values.

    The point source does not know which fragment parameters these values will later be
    applied to; it only knows the ordered tuple of values and the point index assigned
    by the source.
    """

    index: int
    axis_values: tuple[Any, ...]


class PointSource:
    """Base class for host-runtime point sources.

    A point source is intentionally fragment-agnostic. It yields tuples of values whose
    meaning is defined by the scan program that consumes them.

    ``observe()`` is a no-op in the first implementation, but is kept here from the
    start so adaptive strategies can later update themselves from completed
    observations without changing the runner interface again.
    """

    @property
    def axis_count(self) -> int:
        """Return the number of axis values yielded for each point."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[BasePoint]:
        raise NotImplementedError

    def observe(self, _observation: Any) -> None:
        """Receive a completed point observation.

        Static point sources ignore observations, but adaptive sources can use this to
        choose later points.
        """

    def describe(self) -> dict[str, Any]:
        """Return a small serialisable description of the point strategy."""
        raise NotImplementedError


class SinglePointSource(PointSource):
    """Point source for a single empty point.

    This is the new runtime's equivalent of "run once with no scanned axes". Expressing
    the no-axes case as a point source keeps the runner model uniform.
    """

    @property
    def axis_count(self) -> int:
        return 0

    def __iter__(self) -> Iterator[BasePoint]:
        yield BasePoint(index=0, axis_values=())

    def describe(self) -> dict[str, Any]:
        return {"kind": "single"}


class CartesianPointSource(PointSource):
    """Yield the Cartesian product of per-axis value lists."""

    def __init__(self, axis_values: Sequence[Sequence[Any]]):
        self._axis_values = [tuple(values) for values in axis_values]

    @property
    def axis_count(self) -> int:
        return len(self._axis_values)

    def __iter__(self) -> Iterator[BasePoint]:
        for index, values in enumerate(product(*self._axis_values)):
            yield BasePoint(index=index, axis_values=tuple(values))

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "cartesian",
            "axis_count": self.axis_count,
            "points_per_axis": [len(values) for values in self._axis_values],
        }


class ZipPointSource(PointSource):
    """Yield points by zipping per-axis value lists together.

    This is the simplest non-Cartesian strategy and directly covers the common case of
    "scan these parameters together".
    """

    def __init__(self, axis_values: Sequence[Sequence[Any]]):
        self._axis_values = [tuple(values) for values in axis_values]
        lengths = {len(values) for values in self._axis_values}
        if len(lengths) > 1:
            raise ValueError("All zipped axis value lists must have the same length")

    @property
    def axis_count(self) -> int:
        return len(self._axis_values)

    def __iter__(self) -> Iterator[BasePoint]:
        for index, values in enumerate(zip(*self._axis_values, strict=True)):
            yield BasePoint(index=index, axis_values=tuple(values))

    def describe(self) -> dict[str, Any]:
        num_points = 0 if not self._axis_values else len(self._axis_values[0])
        return {
            "kind": "zip",
            "axis_count": self.axis_count,
            "num_points": num_points,
        }


class ExplicitPointSource(PointSource):
    """Yield a user-specified list of full points."""

    def __init__(self, axis_count: int, points: Iterable[Sequence[Any]]):
        self._axis_count = axis_count
        self._points = [tuple(point) for point in points]
        for point in self._points:
            if len(point) != axis_count:
                raise ValueError(
                    "Explicit point has wrong dimensionality: "
                    f"expected {axis_count}, got {len(point)}"
                )

    @property
    def axis_count(self) -> int:
        return self._axis_count

    def __iter__(self) -> Iterator[BasePoint]:
        for index, values in enumerate(self._points):
            yield BasePoint(index=index, axis_values=values)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "explicit",
            "axis_count": self.axis_count,
            "num_points": len(self._points),
        }

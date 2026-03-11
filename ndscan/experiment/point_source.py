"""Point selection primitives for the new host-only runtime.

The host runtime deliberately separates three concerns:

- point policies choose *what* points should run next,
- the runtime decides *how* to execute them,
- the scan-site writer decides *how* to persist completed observations.

The first version of the host runtime only needed static point lists, so a simple
iterator-style interface was enough. Recursive refinement, early-exit wrappers, and
future optimiser backends need a slightly richer contract:

- ask for the next batch of points,
- observe completed points afterwards,
- decide when the policy is finished.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
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
    "ConcatPointSource",
    "ProductPointSource",
    "RecursiveMidpointPointSource1D",
    "UntilConditionPointSource",
]


@dataclass(frozen=True)
class BasePoint:
    """A fully chosen point expressed as an ordered tuple of axis values.

    Point policies are intentionally fragment-agnostic: they do not know which concrete
    fragment parameters these values will later be applied to. They only choose tuples
    in a stable axis order and assign policy-local point indices.
    """

    index: int
    axis_values: tuple[Any, ...]


class PointSource:
    """Base class for host-runtime point policies.

    The interface is intentionally small:

    - ``next_batch(max_points)`` asks the policy for more work,
    - ``observe(...)`` / ``observe_batch(...)`` feed back completed points,
    - ``is_finished()`` reports whether any more points can be produced.

    ``__iter__`` remains as a convenience bridge for tests and for any code that still
    wants a simple point-by-point view, but the host runtime now consumes point
    policies through ``next_batch()`` directly.
    """

    @property
    def axis_count(self) -> int:
        """Return the number of axis values yielded for each point."""
        raise NotImplementedError

    def next_batch(self, max_points: int) -> list[BasePoint]:
        """Return up to ``max_points`` more points from this policy.

        Implementations should return an empty list once the policy is finished. A
        policy that can still make progress should not return an empty batch.
        """
        raise NotImplementedError

    def is_finished(self) -> bool:
        """Return whether the policy has no more points to produce."""
        raise NotImplementedError

    def preferred_batch_size(self, default: int) -> int:
        """Return the batch size this policy would like the runtime to use.

        Static policies usually accept the runtime default unchanged. Future
        observation-driven policies can override this to express a natural ask/tell
        granularity without requiring runner changes elsewhere.
        """
        return default

    def __iter__(self) -> Iterator[BasePoint]:
        while not self.is_finished():
            batch = self.next_batch(1)
            if not batch:
                raise RuntimeError(
                    f"{type(self).__name__} returned no points before finishing"
                )
            yield from batch

    def observe(self, _observation: Any) -> None:
        """Receive one completed point observation.

        Static policies ignore observations. Observation-driven policies can use this to
        update their internal state and choose later points.
        """

    def observe_batch(self, observations: Sequence[Any]) -> None:
        """Receive a sequence of completed observations.

        The default implementation simply forwards each observation to ``observe()``.
        Policies that care about batch boundaries can override this directly.
        """
        for observation in observations:
            self.observe(observation)

    def describe(self) -> dict[str, Any]:
        """Return a small serialisable description of the point strategy."""
        raise NotImplementedError


class _FinitePointSource(PointSource):
    """Base class for finite point sources backed by a fixed list of points."""

    def __init__(self, axis_count: int, points: Sequence[Sequence[Any]]):
        self._axis_count = axis_count
        self._points = [tuple(point) for point in points]
        for point in self._points:
            if len(point) != axis_count:
                raise ValueError(
                    "Point has wrong dimensionality: "
                    f"expected {axis_count}, got {len(point)}"
                )
        self._next_index = 0

    @property
    def axis_count(self) -> int:
        return self._axis_count

    def next_batch(self, max_points: int) -> list[BasePoint]:
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        start = self._next_index
        stop = min(start + max_points, len(self._points))
        self._next_index = stop
        return [
            BasePoint(index=i, axis_values=self._points[i]) for i in range(start, stop)
        ]

    def is_finished(self) -> bool:
        return self._next_index >= len(self._points)


class SinglePointSource(_FinitePointSource):
    """Point source for a single empty point.

    This is the host runtime's equivalent of "run once with no scanned axes". Keeping
    the no-axes case as a point policy keeps the runtime model uniform.
    """

    def __init__(self):
        super().__init__(axis_count=0, points=[()])

    def describe(self) -> dict[str, Any]:
        return {"kind": "single"}


class CartesianPointSource(_FinitePointSource):
    """Yield the Cartesian product of per-axis value lists."""

    def __init__(self, axis_values: Sequence[Sequence[Any]]):
        self._axis_values = [tuple(values) for values in axis_values]
        super().__init__(len(self._axis_values), list(product(*self._axis_values)))

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "cartesian",
            "axis_count": self.axis_count,
            "points_per_axis": [len(values) for values in self._axis_values],
        }


class ZipPointSource(_FinitePointSource):
    """Yield points by zipping per-axis value lists together.

    This is the simplest non-Cartesian strategy and directly covers the common case of
    "scan these parameters together".
    """

    def __init__(self, axis_values: Sequence[Sequence[Any]]):
        self._axis_values = [tuple(values) for values in axis_values]
        lengths = {len(values) for values in self._axis_values}
        if len(lengths) > 1:
            raise ValueError("All zipped axis value lists must have the same length")
        points = list(zip(*self._axis_values, strict=True))
        super().__init__(len(self._axis_values), points)

    def describe(self) -> dict[str, Any]:
        num_points = 0 if not self._axis_values else len(self._axis_values[0])
        return {
            "kind": "zip",
            "axis_count": self.axis_count,
            "num_points": num_points,
        }


class ExplicitPointSource(_FinitePointSource):
    """Yield a user-specified list of full points."""

    def __init__(self, axis_count: int, points: Iterable[Sequence[Any]]):
        self._explicit_points = [tuple(point) for point in points]
        super().__init__(axis_count, self._explicit_points)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "explicit",
            "axis_count": self.axis_count,
            "num_points": len(self._explicit_points),
        }


class ConcatPointSource(PointSource):
    """Run a sequence of point sources one after another.

    This is the simplest composition primitive for staged scans such as:

    - coarse sweep, then fine sweep,
    - scan one region, then another,
    - fixed prelude points before an adaptive phase.

    Observations are forwarded to the currently active child source in emitted order.
    The child receives the original observation object unchanged; child policies should
    therefore treat point indices as informational rather than as strict local ids.
    """

    def __init__(self, sources: Sequence[PointSource]):
        if not sources:
            raise ValueError("ConcatPointSource requires at least one child source")
        self._sources = list(sources)
        axis_counts = {source.axis_count for source in self._sources}
        if len(axis_counts) != 1:
            raise ValueError("All concatenated point sources must have the same axis count")
        self._source_index = 0
        self._next_index = 0
        self._pending_observation_sources = deque[PointSource]()

    @property
    def axis_count(self) -> int:
        return self._sources[0].axis_count

    def _advance_to_active_source(self) -> None:
        while (
            self._source_index < len(self._sources)
            and self._sources[self._source_index].is_finished()
        ):
            self._source_index += 1

    def next_batch(self, max_points: int) -> list[BasePoint]:
        if max_points <= 0:
            raise ValueError("max_points must be positive")

        batch = []
        while len(batch) < max_points:
            self._advance_to_active_source()
            if self._source_index >= len(self._sources):
                break

            source = self._sources[self._source_index]
            child_batch = source.next_batch(max_points - len(batch))
            if not child_batch:
                if source.is_finished():
                    self._source_index += 1
                    continue
                raise RuntimeError(
                    f"{type(source).__name__} returned no points before finishing"
                )

            for point in child_batch:
                batch.append(BasePoint(index=self._next_index, axis_values=point.axis_values))
                self._pending_observation_sources.append(source)
                self._next_index += 1

        return batch

    def is_finished(self) -> bool:
        self._advance_to_active_source()
        return self._source_index >= len(self._sources)

    def preferred_batch_size(self, default: int) -> int:
        self._advance_to_active_source()
        if self.is_finished():
            return default
        return self._sources[self._source_index].preferred_batch_size(default)

    def observe(self, observation: Any) -> None:
        if not self._pending_observation_sources:
            raise RuntimeError("Received observation for ConcatPointSource without a point")
        self._pending_observation_sources.popleft().observe(observation)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "concat",
            "axis_count": self.axis_count,
            "children": [source.describe() for source in self._sources],
        }


class ProductPointSource(_FinitePointSource):
    """Cartesian product of already-composed child point sources.

    This combinator is intended for finite, non-adaptive child policies. It eagerly
    materialises each child source once during construction and then forms the Cartesian
    product of the child points' axis tuples.

    Future adaptive policies that optimise over multiple parameters should usually be
    expressed as one policy over the full parameter set rather than as a product of
    smaller adaptive children.
    """

    def __init__(self, sources: Sequence[PointSource]):
        if not sources:
            raise ValueError("ProductPointSource requires at least one child source")
        self._sources = list(sources)
        materialised = [list(source) for source in self._sources]
        axis_count = sum(source.axis_count for source in self._sources)
        points = [
            tuple(value for point in combo for value in point.axis_values)
            for combo in product(*materialised)
        ]
        super().__init__(axis_count, points)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "product",
            "axis_count": self.axis_count,
            "children": [source.describe() for source in self._sources],
        }


class RecursiveMidpointPointSource1D(_FinitePointSource):
    """Recursively refine a closed 1D interval by inserting midpoints.

    The policy is deliberately simple and deterministic:

    - emit the endpoints first,
    - then recursively bisect intervals breadth-first,
    - stop after ``max_depth`` rounds of bisection.

    This is a useful first refinement policy because it already covers common "scan
    this range more densely each round" workflows without introducing any result-driven
    feedback yet.

    ``max_depth`` counts bisection rounds after the endpoints:

    - ``0`` -> just endpoints
    - ``1`` -> endpoints plus the global midpoint
    - ``2`` -> endpoints, global midpoint, then sub-interval midpoints
    """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        max_depth: int,
        *,
        splitter: Callable[[float, float], float] | None = None,
    ):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        self._min_value = float(min_value)
        self._max_value = float(max_value)
        self._max_depth = max_depth
        self._splitter = splitter or (lambda lo, hi: 0.5 * (lo + hi))
        super().__init__(1, self._generate_points())

    def _generate_points(self) -> list[tuple[float]]:
        points: list[tuple[float]] = []
        seen = set[float]()

        def add(value: float) -> None:
            if value in seen:
                return
            seen.add(value)
            points.append((value,))

        add(self._min_value)
        add(self._max_value)

        if self._max_depth == 0 or self._min_value == self._max_value:
            return points

        intervals = deque([(self._min_value, self._max_value, 0)])
        while intervals:
            lo, hi, depth = intervals.popleft()
            if depth >= self._max_depth:
                continue
            mid = float(self._splitter(lo, hi))
            add(mid)
            if depth + 1 < self._max_depth:
                intervals.append((lo, mid, depth + 1))
                intervals.append((mid, hi, depth + 1))
        return points

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "recursive_midpoint_1d",
            "axis_count": 1,
            "min": self._min_value,
            "max": self._max_value,
            "max_depth": self._max_depth,
        }


class UntilConditionPointSource(PointSource):
    """Stop an inner point source once an observation predicate becomes true.

    This is the minimal early-exit wrapper. It is intentionally generic: callers supply
    a predicate over completed observations, which makes it possible to use the same
    wrapper for conditions such as:

    - "stop once this monitored error is small enough",
    - "stop after the result exceeds a threshold",
    - "stop when an external convergence signal appears".

    The wrapper does not know where the predicate's state comes from. In the future,
    that state may be driven by online analyses, optimiser metrics, or plain point
    results.
    """

    def __init__(
        self,
        inner: PointSource,
        predicate: Callable[[Any], bool],
        *,
        predicate_description: str = "custom",
        min_observations: int = 1,
    ):
        if min_observations < 1:
            raise ValueError("min_observations must be at least 1")
        self._inner = inner
        self._predicate = predicate
        self._predicate_description = predicate_description
        self._min_observations = min_observations
        self._num_observations = 0
        self._stop_requested = False

    @property
    def axis_count(self) -> int:
        return self._inner.axis_count

    def next_batch(self, max_points: int) -> list[BasePoint]:
        if self._stop_requested:
            return []
        return self._inner.next_batch(max_points)

    def is_finished(self) -> bool:
        return self._stop_requested or self._inner.is_finished()

    def preferred_batch_size(self, default: int) -> int:
        return self._inner.preferred_batch_size(default)

    def observe(self, observation: Any) -> None:
        self._inner.observe(observation)
        self._num_observations += 1
        if self._num_observations >= self._min_observations:
            self._stop_requested = self._stop_requested or self._predicate(observation)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "until_condition",
            "axis_count": self.axis_count,
            "predicate": self._predicate_description,
            "min_observations": self._min_observations,
            "inner": self._inner.describe(),
        }

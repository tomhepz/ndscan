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

__all__ = ["generate_points_for_strategy", "create_point_strategy_driver"]


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


def create_point_strategy_driver(
    axis_generators: list[Any],
    options: Any,
    strategy: str | dict[str, Any] = "grid",
) -> Any:
    """Create a strategy driver used by :class:`.StrategyPointSource`.

    Static strategies return a plain iterator.
    Adaptive strategies return an object with ``next_point()/take_points()/observe()``.
    """
    strategy_kind = get_scan_strategy_kind(strategy, ValueError)

    # New canonical adaptive shape:
    #   {"kind": "adaptive", "driver": "gaussian_1d", "config": {...}}
    # Backward compatibility:
    #   {"kind": "gaussian_adaptive_1d", ...}
    if strategy_kind in {"adaptive", "gaussian_adaptive_1d"}:
        if not isinstance(strategy, dict):
            raise ValueError(f"{strategy_kind} strategy must be a dict")

        if strategy_kind == "adaptive":
            driver = strategy.get("driver", None)
            if driver != "gaussian_1d":
                raise ValueError(
                    "adaptive strategy currently supports only driver='gaussian_1d'"
                )
            config = strategy.get("config", {})
            if not isinstance(config, dict):
                raise ValueError("adaptive strategy.config must be a dict")
            return _GaussianAdaptive1DDriver(axis_generators, options, config)

        return _GaussianAdaptive1DDriver(axis_generators, options, strategy)

    return generate_points_for_strategy(axis_generators, options, strategy)


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


class _GaussianAdaptive1DDriver:
    """Host-side adaptive point chooser for 1D Gaussian-like responses.

    After warmup points, it repeatedly:
    1) fits a Gaussian model to observed data;
    2) picks the next x that maximises local Fisher-information gain.
    """

    def __init__(
        self,
        axis_generators: list[Any],
        options: Any,
        strategy: dict[str, Any],
    ) -> None:
        if len(axis_generators) != 1:
            raise ValueError("gaussian_adaptive_1d strategy requires exactly one axis")
        if options.num_repeats != 1 or options.num_repeats_per_point != 1:
            raise ValueError(
                "gaussian_adaptive_1d strategy requires num_repeats=1 and "
                "num_repeats_per_point=1"
            )
        if options.randomise_order_globally:
            raise ValueError(
                "gaussian_adaptive_1d strategy does not support randomise_order_globally"
            )

        self._lower, self._upper = self._resolve_bounds(axis_generators[0], strategy)
        self._num_points = int(strategy.get("num_points", 25))
        self._warmup_points = int(strategy.get("warmup_points", 5))
        self._candidate_count = int(strategy.get("candidate_count", 128))
        self._result_channel = strategy.get("result_channel", None)
        self._fit_iterations = int(strategy.get("fit_iterations", 8))
        self._min_sigma = float(strategy.get("min_sigma", 1e-6))
        if self._num_points < 1:
            raise ValueError("gaussian_adaptive_1d num_points must be >= 1")
        if self._warmup_points < 3:
            raise ValueError("gaussian_adaptive_1d warmup_points must be >= 3")
        if self._candidate_count < 8:
            raise ValueError("gaussian_adaptive_1d candidate_count must be >= 8")
        self._warmup_points = min(self._warmup_points, self._num_points)

        self._warmup_queue = [
            (float(x),)
            for x in np.linspace(
                self._lower, self._upper, num=self._warmup_points, endpoint=True
            )
        ]
        self._observed_x: list[float] = []
        self._observed_y: list[float] = []
        self._num_emitted = 0
        self._last_fit: tuple[float, float, float, float] | None = None

    def next_point(self):
        if self._num_emitted >= self._num_points:
            return None

        if self._warmup_queue:
            point = self._warmup_queue.pop(0)
        else:
            point = (self._choose_adaptive_point(),)

        self._num_emitted += 1
        return point

    def take_points(self, max_points: int) -> list[tuple[Any, ...]]:
        if max_points < 0:
            raise ValueError("max_points must be non-negative")
        if max_points == 0:
            return []

        # Keep adaptive scans strictly point-by-point so observe() can inform
        # subsequent selections even in chunking runners.
        point = self.next_point()
        if point is None:
            return []
        return [point]

    def preferred_batch_size(self, default: int = 1) -> int:
        del default
        return 1

    def diagnostics(self) -> dict[str, Any]:
        return {
            "driver": "gaussian_1d",
            "num_observed": len(self._observed_x),
            "num_emitted": self._num_emitted,
            "last_fit": self._last_fit,
        }

    def observe(self, observation: Any) -> None:
        if observation is None:
            return
        y_value = self._extract_result_value(observation.result_values)
        if y_value is None:
            return

        x_value = None
        axis_map = getattr(observation, "axis_by_param", None)
        if axis_map:
            if len(axis_map) != 1:
                return
            x_value = next(iter(axis_map.values()))
        elif len(observation.axis_values) == 1:
            x_value = observation.axis_values[0]
        else:
            return

        try:
            x = float(x_value)
            y = float(y_value)
        except (TypeError, ValueError):
            return
        self._observed_x.append(x)
        self._observed_y.append(y)

    def _choose_adaptive_point(self) -> float:
        if len(self._observed_x) < max(5, self._warmup_points):
            return self._fallback_point()

        params = self._fit_gaussian_params()
        if params is None:
            return self._fallback_point()
        self._last_fit = params

        candidates = np.linspace(
            self._lower, self._upper, num=self._candidate_count, endpoint=True
        )
        fisher = self._fisher_matrix(np.asarray(self._observed_x), params)
        ridge = np.eye(4) * 1e-9
        base = fisher + ridge
        _, base_logdet = np.linalg.slogdet(base)

        tol = (self._upper - self._lower) / (self._candidate_count * 2)
        observed = np.asarray(self._observed_x)
        best_x = None
        best_score = float("-inf")
        for x in candidates:
            if observed.size and np.min(np.abs(observed - x)) <= tol:
                continue
            j = self._gaussian_jacobian(np.asarray([x]), params)[0]
            info_gain = np.outer(j, j)
            sign, trial_logdet = np.linalg.slogdet(base + info_gain)
            if sign <= 0:
                continue
            score = trial_logdet - base_logdet
            if score > best_score:
                best_score = score
                best_x = float(x)
        if best_x is None:
            return self._fallback_point()
        return best_x

    def _fallback_point(self) -> float:
        # Fallback: pick the candidate maximally far from observed points.
        candidates = np.linspace(
            self._lower, self._upper, num=self._candidate_count, endpoint=True
        )
        if not self._observed_x:
            return float(candidates[len(candidates) // 2])
        observed = np.asarray(self._observed_x)
        distances = [float(np.min(np.abs(observed - x))) for x in candidates]
        return float(candidates[int(np.argmax(distances))])

    def _extract_result_value(self, result_values: dict[str, Any]) -> Any | None:
        if not result_values:
            return None

        if self._result_channel is not None:
            return result_values.get(self._result_channel, None)

        if len(result_values) == 1:
            return next(iter(result_values.values()))

        raise ValueError(
            "gaussian_adaptive_1d strategy needs strategy.result_channel when more "
            "than one result channel is present"
        )

    def _fit_gaussian_params(self) -> tuple[float, float, float, float] | None:
        xs = np.asarray(self._observed_x, dtype=float)
        ys = np.asarray(self._observed_y, dtype=float)
        if xs.size < 5:
            return None

        baseline = float(np.min(ys))
        amplitude = max(float(np.max(ys) - baseline), 1e-9)
        weights = np.clip(ys - baseline, 0.0, None)
        if np.sum(weights) > 0:
            mu = float(np.sum(xs * weights) / np.sum(weights))
            sigma = float(np.sqrt(np.sum(weights * (xs - mu) ** 2) / np.sum(weights)))
        else:
            mu = float(np.mean(xs))
            sigma = float((self._upper - self._lower) / 6.0)

        theta = np.array(
            [amplitude, np.clip(mu, self._lower, self._upper), max(sigma, self._min_sigma), baseline],
            dtype=float,
        )

        for _ in range(self._fit_iterations):
            model = self._gaussian_model(xs, theta)
            jac = self._gaussian_jacobian(xs, theta)
            residual = ys - model
            lhs = jac.T @ jac + np.eye(4) * 1e-6
            rhs = jac.T @ residual
            try:
                delta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                break
            theta += delta
            theta[0] = max(theta[0], 1e-9)
            theta[1] = float(np.clip(theta[1], self._lower, self._upper))
            theta[2] = max(abs(theta[2]), self._min_sigma)
        return tuple(float(v) for v in theta)

    def _fisher_matrix(
        self, xs: np.ndarray, params: tuple[float, float, float, float]
    ) -> np.ndarray:
        jac = self._gaussian_jacobian(xs, params)
        return jac.T @ jac

    @staticmethod
    def _gaussian_model(
        xs: np.ndarray, params: tuple[float, float, float, float] | np.ndarray
    ) -> np.ndarray:
        amp, mu, sigma, offset = params
        z = (xs - mu) / sigma
        return amp * np.exp(-0.5 * z * z) + offset

    @staticmethod
    def _gaussian_jacobian(
        xs: np.ndarray, params: tuple[float, float, float, float] | np.ndarray
    ) -> np.ndarray:
        amp, mu, sigma, _offset = params
        z = (xs - mu) / sigma
        exp_term = np.exp(-0.5 * z * z)
        d_amp = exp_term
        d_mu = amp * exp_term * (z / sigma)
        d_sigma = amp * exp_term * (z * z / sigma)
        d_offset = np.ones_like(xs)
        return np.column_stack([d_amp, d_mu, d_sigma, d_offset])

    @staticmethod
    def _resolve_bounds(
        axis_generator: Any, strategy: dict[str, Any]
    ) -> tuple[float, float]:
        lower = strategy.get("lower", None)
        upper = strategy.get("upper", None)
        if lower is not None and upper is not None:
            return float(min(lower, upper)), float(max(lower, upper))

        limits: dict[str, Any] = {}
        if hasattr(axis_generator, "describe_limits"):
            axis_generator.describe_limits(limits)
        if lower is None:
            lower = limits.get("min", None)
        if upper is None:
            upper = limits.get("max", None)

        if lower is None or upper is None:
            values = getattr(axis_generator, "values", None)
            if values is not None and len(values) > 0:
                lower = min(values) if lower is None else lower
                upper = max(values) if upper is None else upper

        if lower is None or upper is None:
            raise ValueError(
                "gaussian_adaptive_1d strategy needs lower/upper bounds "
                "or axis generator limits"
            )
        return float(min(lower, upper)), float(max(lower, upper))

"""Optional optimiser backends for prepared-runtime ask/tell point policies.

The new prepared runtime already has the right control flow for model-based optimisation:

- a point policy asks for the next batch of points,
- the runtime executes and persists that batch,
- online analysis runs if needed,
- batch feedback is returned to the policy.

This module keeps the optimiser-specific logic out of the core runtime. The first
backend implemented here is a NUBO/GPyTorch-based Bayesian optimisation backend that
can propose one batch at a time and optionally mix in user-supplied exploration
points.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from nubo.acquisition import MCExpectedImprovement, MCUpperConfidenceBound
from nubo.models import GaussianProcess, fit_gp
from nubo.optimisation import multi_sequential, single

from .point_policy import BatchFeedback, OptimiserObservation, OptimiserSuggestion

__all__ = [
    "CompositeExplorationStrategy",
    "ExplicitBatchExplorationStrategy",
    "LocalLengthscaleExplorationStrategy",
    "MhcsExplorationStrategy",
    "NuboBayesianOptimisationState",
    "NuboBatchBayesianOptimisationBackend",
    "ScalarChannelObjectiveExtractor",
    "ScheduledExplorationStrategy",
    "extract_scalar_channel_objective",
]


@dataclass(frozen=True)
class ScalarChannelObjectiveExtractor:
    """Extract optimiser observations from one scalar result channel.

    This is the common BO case:

    - one saved result channel is the objective,
    - an optional second saved result channel carries an uncertainty estimate,
    - the optimiser sees one scalar objective per completed point.

    The extractor is a real object rather than an anonymous closure so the runtime can
    persist enough metadata in ``scan.point_policy`` for offline GP refits.
    """

    channel_key: str
    noise_channel_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __call__(
        self, observation: Any, _feedback: BatchFeedback
    ) -> OptimiserObservation:
        axis_values = tuple(float(value) for value in observation.axis_values.values())
        objective = float(observation.channel_values[self.channel_key])
        noise_std = (
            None
            if self.noise_channel_key is None
            else float(observation.channel_values[self.noise_channel_key])
        )
        return OptimiserObservation(
            point=axis_values,
            objective=objective,
            noise_std=noise_std,
            metadata=dict(self.metadata),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "scalar_channel",
            "channel_key": self.channel_key,
            "noise_channel_key": self.noise_channel_key,
            "metadata": dict(self.metadata),
        }


def extract_scalar_channel_objective(
    channel_key: str,
    *,
    noise_channel_key: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ScalarChannelObjectiveExtractor:
    """Return a scalar-channel objective extractor for ask/tell optimisation."""

    return ScalarChannelObjectiveExtractor(
        channel_key=channel_key,
        noise_channel_key=noise_channel_key,
        metadata={} if metadata is None else dict(metadata),
    )


def _to_point_tensor(
    points: Sequence[Sequence[float]] | torch.Tensor,
    *,
    dims: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(points, torch.Tensor):
        tensor = points.to(dtype=dtype, device=device)
    else:
        tensor = torch.as_tensor(points, dtype=dtype, device=device)

    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.numel() == 0:
        return tensor.reshape(0, dims)
    if tensor.shape[1] != dims:
        raise ValueError(
            f"Expected points with dimensionality {dims}, got shape {tuple(tensor.shape)}"
        )
    return tensor


def _fit_exact_gp_model(
    x_train: torch.Tensor,
    y_train_score: torch.Tensor,
    y_train_err: torch.Tensor,
    *,
    lr: float,
    steps: int,
) -> tuple[GaussianProcess, FixedNoiseGaussianLikelihood]:
    likelihood = FixedNoiseGaussianLikelihood(
        noise=y_train_err.square(),
        learn_additional_noise=True,
    )
    gp = GaussianProcess(x_train, y_train_score, likelihood=likelihood)
    fit_gp(
        x_train,
        y_train_score,
        gp=gp,
        likelihood=likelihood,
        lr=lr,
        steps=steps,
    )
    gp.eval()
    likelihood.eval()
    return gp, likelihood


def _posterior_mean_std(
    gp: GaussianProcess,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    gp.eval()
    with torch.no_grad(), torch.amp.autocast("cpu", enabled=False):
        posterior = gp(x)
    mean = posterior.mean
    std = posterior.variance.clamp_min(1e-12).sqrt()
    return mean, std


def _surrogate_argmax(
    gp: GaussianProcess,
    bounds: torch.Tensor,
    *,
    num_starts: int,
) -> tuple[torch.Tensor, float]:
    train_inputs = gp.train_inputs[0]

    def negative_gp_mean(x: torch.Tensor | np.ndarray) -> float:
        if isinstance(x, np.ndarray):
            xt = torch.as_tensor(x, dtype=train_inputs.dtype, device=train_inputs.device)
        else:
            xt = x.to(dtype=train_inputs.dtype, device=train_inputs.device)
        xt = xt.reshape(1, -1)
        mean, _ = _posterior_mean_std(gp, xt)
        return -float(mean.item())

    x_max, _ = single(
        func=negative_gp_mean,
        bounds=bounds,
        method="L-BFGS-B",
        num_starts=num_starts,
    )
    x_max = x_max.reshape(-1)
    mu_score, _ = _posterior_mean_std(gp, x_max)
    return x_max, float(mu_score.item())


def _build_mc_acquisition(
    gp: GaussianProcess,
    bounds: torch.Tensor,
    *,
    acquisition_name: str,
    x_pending: torch.Tensor | None,
    samples: int,
    beta: float,
    surrogate_num_starts: int,
):
    acquisition_name = acquisition_name.lower()
    train_inputs = gp.train_inputs[0]
    if x_pending is not None:
        x_pending = x_pending.to(dtype=train_inputs.dtype, device=train_inputs.device)

    if acquisition_name == "ucb":
        return MCUpperConfidenceBound(
            gp=gp,
            beta=beta,
            x_pending=x_pending,
            samples=samples,
        )

    if acquisition_name == "ei":
        _, incumbent_score = _surrogate_argmax(
            gp,
            bounds,
            num_starts=surrogate_num_starts,
        )
        incumbent = torch.tensor(
            incumbent_score,
            dtype=gp.train_inputs[0].dtype,
            device=gp.train_inputs[0].device,
        )
        return MCExpectedImprovement(
            gp=gp,
            y_best=incumbent,
            x_pending=x_pending,
            samples=samples,
        )

    raise ValueError(
        f"Unknown batch acquisition {acquisition_name!r}. Expected 'ucb' or 'ei'."
    )


def _filter_near_duplicates(
    candidate_points: torch.Tensor,
    existing_points: torch.Tensor,
    bounds: torch.Tensor,
    *,
    min_normalised_distance: float,
) -> torch.Tensor:
    if candidate_points.numel() == 0:
        return candidate_points

    span = (bounds[1] - bounds[0]).to(candidate_points)
    kept: list[torch.Tensor] = []

    for point in candidate_points:
        references = [existing_points]
        if kept:
            references.append(torch.stack(kept))
        ref = torch.vstack(references)
        normalised_dist = torch.linalg.norm((ref - point) / span, dim=1)
        if torch.all(normalised_dist > min_normalised_distance):
            kept.append(point)

    if not kept:
        return candidate_points[:0]
    return torch.stack(kept)


def _sample_random_points(
    bounds: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if count <= 0:
        dims = int(bounds.shape[1])
        return torch.zeros((0, dims), dtype=bounds.dtype, device=bounds.device)

    random_points = torch.rand(
        (count, int(bounds.shape[1])),
        dtype=bounds.dtype,
        device=bounds.device,
        generator=generator,
    )
    span = bounds[1] - bounds[0]
    return bounds[0] + random_points * span


def _normalise_points(points: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    span = bounds[1] - bounds[0]
    return (points - bounds[0]) / span


def _unnormalise_points(points: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    span = bounds[1] - bounds[0]
    return bounds[0] + points * span


def _normalised_bounds(bounds: torch.Tensor) -> torch.Tensor:
    """Return dimensionless unit bounds with the same shape, dtype and device."""
    return torch.stack((torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])))


def _find_mhcs_candidate(
    points_norm: torch.Tensor,
    *,
    pair_batch_size: int,
    interior_tol: float,
) -> tuple[torch.Tensor | None, float]:
    num_points = points_norm.shape[0]
    if num_points < 2:
        return None, 0.0

    pair_i, pair_j = torch.triu_indices(
        num_points,
        num_points,
        offset=1,
        device=points_norm.device,
    )
    best_center = None
    best_radius = 0.0

    for start in range(0, pair_i.numel(), pair_batch_size):
        end = min(start + pair_batch_size, pair_i.numel())
        left = points_norm[pair_i[start:end]]
        right = points_norm[pair_j[start:end]]

        centers = 0.5 * (left + right)
        radii = 0.5 * torch.linalg.norm(left - right, dim=1)
        valid = radii > interior_tol
        if not torch.any(valid):
            continue

        min_dist = torch.cdist(centers, points_norm).amin(dim=1)
        valid = valid & (min_dist >= (radii - interior_tol))
        if not torch.any(valid):
            continue

        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        batch_radii = radii[valid_indices]
        batch_best_idx = valid_indices[int(torch.argmax(batch_radii))]
        batch_best_radius = float(radii[batch_best_idx])

        if batch_best_radius > best_radius:
            best_radius = batch_best_radius
            best_center = centers[batch_best_idx]

    return best_center, best_radius


def _generate_mhcs_points(
    existing_points: torch.Tensor,
    bounds: torch.Tensor,
    *,
    num_points: int,
    min_normalised_distance: float,
    pair_batch_size: int,
    interior_tol: float,
) -> torch.Tensor:
    if num_points <= 0 or existing_points.shape[0] < 2:
        return existing_points[:0]

    points_norm = torch.unique(_normalise_points(existing_points, bounds), dim=0)
    chosen_norm: list[torch.Tensor] = []

    for _ in range(num_points):
        working_points = points_norm
        if chosen_norm:
            working_points = torch.vstack((working_points, torch.stack(chosen_norm)))

        center_norm, radius = _find_mhcs_candidate(
            working_points,
            pair_batch_size=pair_batch_size,
            interior_tol=interior_tol,
        )
        if center_norm is None or radius <= interior_tol:
            break

        chosen_norm.append(center_norm)

    if not chosen_norm:
        return existing_points[:0]

    candidates = _unnormalise_points(torch.stack(chosen_norm), bounds)
    return _filter_near_duplicates(
        candidates,
        existing_points=existing_points,
        bounds=bounds,
        min_normalised_distance=min_normalised_distance,
    )


def _extract_lengthscales(gp: GaussianProcess, bounds: torch.Tensor) -> torch.Tensor:
    raw = gp.covar_module.base_kernel.lengthscale.detach().reshape(-1)
    span = (bounds[1] - bounds[0]).to(dtype=raw.dtype, device=raw.device)
    return raw.clamp(min=0.05 * span, max=0.5 * span)


def _generate_local_exploration_points(
    center: torch.Tensor,
    gp: GaussianProcess,
    bounds: torch.Tensor,
    existing_points: torch.Tensor,
    *,
    num_points: int,
    axis_points: int,
    pair_points: int,
    span_in_lengthscales: float,
    min_normalised_distance: float,
) -> torch.Tensor:
    center = center.reshape(-1)
    dims = center.numel()
    lengthscales = _extract_lengthscales(gp, bounds).to(
        dtype=center.dtype,
        device=center.device,
    )

    grids: list[torch.Tensor] = []
    axis_grid = torch.linspace(
        -span_in_lengthscales,
        span_in_lengthscales,
        axis_points,
        dtype=center.dtype,
        device=center.device,
    )

    for axis in range(dims):
        offsets = axis_grid * lengthscales[axis]
        points = center.repeat(axis_points, 1)
        points[:, axis] = torch.clamp(
            center[axis] + offsets,
            bounds[0, axis],
            bounds[1, axis],
        )
        grids.append(points)

    if dims > 1:
        pair_grid = torch.linspace(
            -span_in_lengthscales,
            span_in_lengthscales,
            pair_points,
            dtype=center.dtype,
            device=center.device,
        )

        for axis_i in range(dims):
            for axis_j in range(axis_i + 1, dims):
                grid_i, grid_j = torch.meshgrid(
                    pair_grid,
                    pair_grid,
                    indexing="ij",
                )
                points = center.repeat(grid_i.numel(), 1)
                points[:, axis_i] = torch.clamp(
                    center[axis_i] + grid_i.reshape(-1) * lengthscales[axis_i],
                    bounds[0, axis_i],
                    bounds[1, axis_i],
                )
                points[:, axis_j] = torch.clamp(
                    center[axis_j] + grid_j.reshape(-1) * lengthscales[axis_j],
                    bounds[0, axis_j],
                    bounds[1, axis_j],
                )
                grids.append(points)

    scan_points = torch.unique(torch.vstack(grids), dim=0)
    scan_points = _filter_near_duplicates(
        scan_points,
        existing_points=existing_points,
        bounds=bounds,
        min_normalised_distance=min_normalised_distance,
    )
    return scan_points[:num_points]


@dataclass(frozen=True)
class ExplicitBatchExplorationStrategy:
    """Inject explicit hand-picked exploration points on selected BO batches.

    This is the clean home for "manual search" phases: a caller can schedule exact
    points for specific optimiser batch indices without teaching the core backend
    anything about that search phase.
    """

    schedule: Mapping[int, Sequence[Sequence[float]] | torch.Tensor]

    def __call__(
        self,
        state: NuboBayesianOptimisationState,
        max_points: int,
    ) -> torch.Tensor:
        points = self.schedule.get(state.batch_index, ())
        return _to_point_tensor(
            points,
            dims=state.bounds.shape[1],
            dtype=state.bounds.dtype,
            device=state.bounds.device,
        )[:max_points]

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "explicit_batches",
            "batch_indices": sorted(int(index) for index in self.schedule),
        }


@dataclass(frozen=True)
class MhcsExplorationStrategy:
    """Propose points in large uncovered regions via MHCS-style geometry."""

    num_points: int = 1
    pair_batch_size: int = 4096
    interior_tol: float = 1e-6
    min_normalised_distance: float = 0.05

    def __call__(
        self,
        state: NuboBayesianOptimisationState,
        max_points: int,
    ) -> torch.Tensor:
        return _generate_mhcs_points(
            state.observed_points,
            state.bounds,
            num_points=min(max_points, self.num_points),
            min_normalised_distance=self.min_normalised_distance,
            pair_batch_size=self.pair_batch_size,
            interior_tol=self.interior_tol,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "mhcs",
            "num_points": self.num_points,
            "pair_batch_size": self.pair_batch_size,
        }


@dataclass(frozen=True)
class LocalLengthscaleExplorationStrategy:
    """Sample a small local grid around the current GP surrogate optimum."""

    num_points: int = 1
    axis_points: int = 7
    pair_points: int = 5
    span_in_lengthscales: float = 2.0
    min_normalised_distance: float = 0.05
    surrogate_num_starts: int = 20

    def __call__(
        self,
        state: NuboBayesianOptimisationState,
        max_points: int,
    ) -> torch.Tensor:
        # The backend fits the GP in dimensionless coordinates. Generate the local
        # length-scale grid in that same model space, then return ordinary physical
        # coordinates like every other exploration strategy.
        model_bounds = (
            state.normalised_bounds
            if state.normalised_bounds is not None
            else state.bounds
        )
        model_observed_points = (
            state.normalised_observed_points
            if state.normalised_observed_points is not None
            else state.observed_points
        )
        center, _ = _surrogate_argmax(
            state.gp,
            model_bounds,
            num_starts=self.surrogate_num_starts,
        )
        model_points = _generate_local_exploration_points(
            center,
            state.gp,
            model_bounds,
            model_observed_points,
            num_points=min(max_points, self.num_points),
            axis_points=self.axis_points,
            pair_points=self.pair_points,
            span_in_lengthscales=self.span_in_lengthscales,
            min_normalised_distance=self.min_normalised_distance,
        )
        if state.normalised_bounds is None:
            return model_points
        return _unnormalise_points(model_points, state.bounds)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "local_lengthscale",
            "num_points": self.num_points,
            "axis_points": self.axis_points,
            "pair_points": self.pair_points,
        }


@dataclass(frozen=True)
class CompositeExplorationStrategy:
    """Run several exploration strategies in order within one BO batch.

    This is the clean way to express policies like ``mhcs + local``: each component
    gets a chance to consume part of the available exploration budget, while the
    composition layer handles de-duplication against both observed and newly chosen
    points.
    """

    strategies: tuple[
        Callable[
            [NuboBayesianOptimisationState, int],
            Sequence[Sequence[float]] | torch.Tensor,
        ],
        ...,
    ]
    min_normalised_distance: float = 1e-6

    def __init__(
        self,
        strategies: Sequence[
            Callable[[NuboBayesianOptimisationState, int], Sequence[Sequence[float]] | torch.Tensor]
        ],
        *,
        min_normalised_distance: float = 1e-6,
    ):
        object.__setattr__(self, "strategies", tuple(strategies))
        object.__setattr__(self, "min_normalised_distance", min_normalised_distance)

    def __call__(
        self,
        state: NuboBayesianOptimisationState,
        max_points: int,
    ) -> torch.Tensor:
        if max_points <= 0:
            return torch.zeros(
                (0, state.bounds.shape[1]),
                dtype=state.bounds.dtype,
                device=state.bounds.device,
            )

        chosen = torch.zeros(
            (0, state.bounds.shape[1]),
            dtype=state.bounds.dtype,
            device=state.bounds.device,
        )
        working_observed = state.observed_points

        for strategy in self.strategies:
            remaining = max_points - int(chosen.shape[0])
            if remaining <= 0:
                break
            candidates = _to_point_tensor(
                strategy(state, remaining),
                dims=state.bounds.shape[1],
                dtype=state.bounds.dtype,
                device=state.bounds.device,
            )
            if candidates.numel() == 0:
                continue

            candidates = _filter_near_duplicates(
                candidates,
                torch.vstack((working_observed, chosen)),
                state.bounds,
                min_normalised_distance=self.min_normalised_distance,
            )
            if candidates.numel() == 0:
                continue
            candidates = candidates[:remaining]
            chosen = torch.vstack((chosen, candidates))

        return chosen

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "composite",
            "min_normalised_distance": self.min_normalised_distance,
            "children": [
                strategy.describe() if hasattr(strategy, "describe") else {"kind": type(strategy).__name__}
                for strategy in self.strategies
            ],
        }


@dataclass(frozen=True)
class ScheduledExplorationStrategy:
    """Apply an exploration strategy only on selected BO batch indices."""

    every_batches: int
    strategy: Callable[
        [NuboBayesianOptimisationState, int], Sequence[Sequence[float]] | torch.Tensor
    ]
    offset: int = 0

    def __post_init__(self) -> None:
        if self.every_batches <= 0:
            raise ValueError("every_batches must be positive")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")

    def __call__(
        self,
        state: NuboBayesianOptimisationState,
        max_points: int,
    ) -> torch.Tensor:
        if state.batch_index < self.offset:
            return torch.zeros(
                (0, state.bounds.shape[1]),
                dtype=state.bounds.dtype,
                device=state.bounds.device,
            )
        if (state.batch_index - self.offset) % self.every_batches != 0:
            return torch.zeros(
                (0, state.bounds.shape[1]),
                dtype=state.bounds.dtype,
                device=state.bounds.device,
            )
        return _to_point_tensor(
            self.strategy(state, max_points),
            dims=state.bounds.shape[1],
            dtype=state.bounds.dtype,
            device=state.bounds.device,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "scheduled",
            "every_batches": self.every_batches,
            "offset": self.offset,
            "strategy": (
                self.strategy.describe()
                if hasattr(self.strategy, "describe")
                else {"kind": type(self.strategy).__name__}
            ),
        }


def _propose_bo_batch(
    gp: GaussianProcess,
    bounds: torch.Tensor,
    *,
    batch_size: int,
    acquisition_name: str,
    x_pending: torch.Tensor | None,
    num_starts: int,
    surrogate_num_starts: int,
    lr: float,
    steps: int,
    samples: int,
    beta: float,
) -> torch.Tensor:
    dims = bounds.size(1)
    if batch_size <= 0:
        return torch.zeros((0, dims), dtype=bounds.dtype, device=bounds.device)

    acq = _build_mc_acquisition(
        gp,
        bounds,
        acquisition_name=acquisition_name,
        x_pending=x_pending,
        samples=samples,
        beta=beta,
        surrogate_num_starts=surrogate_num_starts,
    )
    train_inputs = gp.train_inputs[0]

    class _TypedAcquisitionWrapper:
        """Preserve the acquisition-function interface while enforcing dtype.

        NUBO's ``multi_sequential`` does not treat the acquisition as a plain callable;
        it also reads and mutates attributes such as ``x_pending`` and
        ``base_samples`` between candidate proposals. A bare Python wrapper function
        breaks that contract, so this tiny proxy delegates attribute access and
        assignment to the wrapped acquisition object.
        """

        def __init__(self, acquisition, dtype: torch.dtype, device: torch.device):
            object.__setattr__(self, "_acquisition", acquisition)
            object.__setattr__(self, "_dtype", dtype)
            object.__setattr__(self, "_device", device)

        def __call__(self, x: torch.Tensor | np.ndarray):
            if isinstance(x, np.ndarray):
                xt = torch.as_tensor(
                    x,
                    dtype=self._dtype,
                    device=self._device,
                )
            else:
                xt = x.to(dtype=self._dtype, device=self._device)
            return self._acquisition(xt)

        def __getattr__(self, name: str):
            return getattr(self._acquisition, name)

        def __setattr__(self, name: str, value):
            if name in {"_acquisition", "_dtype", "_device"}:
                object.__setattr__(self, name, value)
                return
            setattr(self._acquisition, name, value)

    typed_acquisition = _TypedAcquisitionWrapper(
        acq,
        dtype=train_inputs.dtype,
        device=train_inputs.device,
    )

    x_new, _ = multi_sequential(
        func=typed_acquisition,
        method="Adam",
        batch_size=batch_size,
        bounds=bounds,
        lr=lr,
        steps=steps,
        num_starts=num_starts,
    )
    return x_new.to(dtype=bounds.dtype, device=bounds.device).reshape(batch_size, -1)


@dataclass(frozen=True)
class NuboBayesianOptimisationState:
    """State passed to optional exploration strategies.

    ``observed_points`` and ``bounds`` always use the experiment's physical units.
    The GP itself is fitted in dimensionless coordinates; strategies which inspect
    the GP can use the optional normalised fields to work in the matching space.
    """

    observed_points: torch.Tensor
    observed_objective_scores: torch.Tensor
    observed_noise_std: torch.Tensor
    gp: GaussianProcess
    bounds: torch.Tensor
    batch_index: int
    normalised_observed_points: torch.Tensor | None = None
    normalised_bounds: torch.Tensor | None = None


class NuboBatchBayesianOptimisationBackend:
    """Ask/tell Bayesian optimisation backend using NUBO + GPyTorch.

    The backend is deliberately batch-oriented. One call to ``suggest(...)`` returns
    one logical optimisation batch. The prepared runtime can then execute that batch,
    publish it, run online analysis, and finally feed the completed observations back
    through ``observe(...)``.

    More complicated search phases such as manual search or MHCS exploration are best
    expressed as an *exploration strategy callback* rather than baked into the backend
    itself. That keeps this class focused on:

    - maintaining the GP state,
    - proposing BO points,
    - and optionally reserving part of each batch for externally defined exploration.
    """

    def __init__(
        self,
        bounds: Sequence[Sequence[float]] | torch.Tensor,
        *,
        batch_size: int,
        initial_points: Sequence[Sequence[float]] | torch.Tensor | None = None,
        initial_design_size: int = 0,
        random_seed: int | None = None,
        acquisition_name: str = "ucb",
        fit_steps: int = 400,
        fit_lr: float = 0.05,
        acquisition_num_starts: int = 5,
        surrogate_num_starts: int = 20,
        batch_mc_samples: int = 128,
        batch_acq_lr: float = 0.1,
        batch_acq_steps: int = 200,
        batch_ucb_beta: float = 1.96**2,
        max_batches: int | None = None,
        minimise: bool = True,
        min_normalised_distance: float = 1e-6,
        observation_noise_floor: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        exploration_strategy: Callable[
            [NuboBayesianOptimisationState, int], Sequence[Sequence[float]] | torch.Tensor
        ]
        | None = None,
    ):
        bounds_tensor = torch.as_tensor(bounds, dtype=dtype)
        if (
            bounds_tensor.ndim != 2
            or bounds_tensor.shape[0] != 2
            or bounds_tensor.shape[1] == 0
        ):
            raise ValueError(
                "bounds must have shape (2, dims) with at least one dimension"
            )
        if not torch.all(torch.isfinite(bounds_tensor)):
            raise ValueError("bounds must be finite")
        if torch.any(bounds_tensor[1] <= bounds_tensor[0]):
            raise ValueError("Every upper bound must be greater than its lower bound")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_batches is not None and max_batches < 1:
            raise ValueError("max_batches must be positive when specified")
        if fit_steps <= 0:
            raise ValueError("fit_steps must be positive")
        if fit_lr <= 0.0:
            raise ValueError("fit_lr must be positive")
        if observation_noise_floor <= 0.0:
            raise ValueError("observation_noise_floor must be positive")

        self._bounds = bounds_tensor
        self._dims = int(bounds_tensor.shape[1])
        self._batch_size = int(batch_size)
        self._acquisition_name = acquisition_name.lower()
        self._fit_steps = int(fit_steps)
        self._fit_lr = float(fit_lr)
        self._acquisition_num_starts = int(acquisition_num_starts)
        self._surrogate_num_starts = int(surrogate_num_starts)
        self._batch_mc_samples = int(batch_mc_samples)
        self._batch_acq_lr = float(batch_acq_lr)
        self._batch_acq_steps = int(batch_acq_steps)
        self._batch_ucb_beta = float(batch_ucb_beta)
        self._max_batches = max_batches
        self._minimise = minimise
        self._min_normalised_distance = float(min_normalised_distance)
        self._observation_noise_floor = float(observation_noise_floor)
        self._dtype = dtype
        self._exploration_strategy = exploration_strategy
        self._rng = torch.Generator(device=self._bounds.device)
        if random_seed is not None:
            self._rng.manual_seed(random_seed)

        seed_chunks = []
        if initial_points is not None:
            seed_chunks.append(
                _to_point_tensor(
                    initial_points,
                    dims=self._dims,
                    dtype=self._bounds.dtype,
                    device=self._bounds.device,
                )
            )
        if initial_design_size > 0:
            random_points = torch.rand(
                (initial_design_size, self._dims),
                dtype=self._bounds.dtype,
                device=self._bounds.device,
                generator=self._rng,
            )
            span = self._bounds[1] - self._bounds[0]
            seed_chunks.append(self._bounds[0] + random_points * span)

        if not seed_chunks:
            raise ValueError(
                "Provide either initial_points or initial_design_size for Bayesian optimisation"
            )
        self._seed_points = torch.vstack(seed_chunks)
        self._seed_index = 0

        self._x_obs = torch.zeros((0, self._dims), dtype=self._bounds.dtype, device=self._bounds.device)
        self._y_obs_score = torch.zeros((0,), dtype=self._bounds.dtype, device=self._bounds.device)
        self._y_obs_err = torch.zeros((0,), dtype=self._bounds.dtype, device=self._bounds.device)
        self._num_bo_batches = 0

    def _make_fallback_points(
        self,
        count: int,
        existing_points: torch.Tensor,
    ) -> torch.Tensor:
        if count <= 0:
            return existing_points[:0]

        kept: list[torch.Tensor] = []
        max_attempts = max(8, 32 * count)
        working_existing = existing_points
        for _ in range(max_attempts):
            candidate = _sample_random_points(
                self._bounds,
                count=1,
                generator=self._rng,
            )
            filtered = _filter_near_duplicates(
                candidate,
                working_existing,
                self._bounds,
                min_normalised_distance=self._min_normalised_distance,
            )
            if filtered.shape[0] == 0:
                continue
            kept.append(filtered[0])
            working_existing = torch.vstack((working_existing, filtered))
            if len(kept) >= count:
                return torch.stack(kept)

        # If the configured spacing threshold makes it impossible to find any new
        # points, still return random in-bounds suggestions so the optimiser can
        # continue instead of failing the runtime with an empty batch.
        remaining = count - len(kept)
        raw = _sample_random_points(
            self._bounds,
            count=remaining,
            generator=self._rng,
        )
        if not kept:
            return raw
        return torch.vstack((torch.stack(kept), raw))

    @property
    def axis_count(self) -> int:
        return self._dims

    def preferred_batch_size(self, _default: int) -> int:
        return self._batch_size

    def is_finished(self) -> bool:
        if self._seed_index < len(self._seed_points):
            return False
        if self._max_batches is None:
            return False
        return self._num_bo_batches >= self._max_batches

    def suggest(self, max_points: int) -> list[OptimiserSuggestion]:
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        if self.is_finished():
            return []

        limit = min(max_points, self._batch_size)
        if self._seed_index < len(self._seed_points):
            start = self._seed_index
            stop = min(start + limit, len(self._seed_points))
            self._seed_index = stop
            return [
                OptimiserSuggestion(
                    point=tuple(map(float, row.tolist())),
                    metadata={"decision_source": "seed"},
                )
                for row in self._seed_points[start:stop]
            ]

        # All model fitting and acquisition optimisation happens in a unit hypercube.
        # Keeping the optimiser's numerical coordinates dimensionless is essential:
        # otherwise an axis expressed near 120 MHz makes order-one voltage/amplitude
        # axes effectively invisible to the kernel, and an Adam step of 0.1 moves the
        # frequency by only 0.1 Hz.
        model_bounds = _normalised_bounds(self._bounds)
        model_x_obs = _normalise_points(self._x_obs, self._bounds)
        gp, _ = _fit_exact_gp_model(
            model_x_obs,
            self._y_obs_score,
            self._y_obs_err.clamp_min(self._observation_noise_floor),
            lr=self._fit_lr,
            steps=self._fit_steps,
        )

        explore_points = torch.zeros(
            (0, self._dims),
            dtype=self._bounds.dtype,
            device=self._bounds.device,
        )
        if self._exploration_strategy is not None:
            state = NuboBayesianOptimisationState(
                observed_points=self._x_obs,
                observed_objective_scores=self._y_obs_score,
                observed_noise_std=self._y_obs_err,
                gp=gp,
                bounds=self._bounds,
                batch_index=self._num_bo_batches,
                normalised_observed_points=model_x_obs,
                normalised_bounds=model_bounds,
            )
            explore_points = _to_point_tensor(
                self._exploration_strategy(state, limit),
                dims=self._dims,
                dtype=self._bounds.dtype,
                device=self._bounds.device,
            )
            explore_points = _filter_near_duplicates(
                explore_points,
                self._x_obs,
                self._bounds,
                min_normalised_distance=self._min_normalised_distance,
            )[:limit]

        model_explore_points = _normalise_points(explore_points, self._bounds)
        bo_batch_size = limit - int(explore_points.shape[0])
        model_bo_points = _propose_bo_batch(
            gp,
            model_bounds,
            batch_size=bo_batch_size,
            acquisition_name=self._acquisition_name,
            x_pending=(
                model_explore_points
                if model_explore_points.numel() > 0
                else None
            ),
            num_starts=self._acquisition_num_starts,
            surrogate_num_starts=self._surrogate_num_starts,
            lr=self._batch_acq_lr,
            steps=self._batch_acq_steps,
            samples=self._batch_mc_samples,
            beta=self._batch_ucb_beta,
        )
        bo_points = _unnormalise_points(model_bo_points, self._bounds)
        bo_points = _filter_near_duplicates(
            bo_points,
            torch.vstack((self._x_obs, explore_points)),
            self._bounds,
            min_normalised_distance=self._min_normalised_distance,
        )

        self._num_bo_batches += 1
        suggestions = [
            OptimiserSuggestion(
                point=tuple(map(float, row.tolist())),
                metadata={"decision_source": "explore"},
            )
            for row in explore_points
        ]
        suggestions.extend(
            OptimiserSuggestion(
                point=tuple(map(float, row.tolist())),
                metadata={"decision_source": "bo"},
            )
            for row in bo_points
        )
        if not suggestions:
            fallback_points = self._make_fallback_points(limit, self._x_obs)
            suggestions.extend(
                OptimiserSuggestion(
                    point=tuple(map(float, row.tolist())),
                    metadata={"decision_source": "fallback"},
                )
                for row in fallback_points
            )
        return suggestions

    def observe(self, observations: Sequence[OptimiserObservation]) -> None:
        if not observations:
            return

        new_x = _to_point_tensor(
            [observation.point for observation in observations],
            dims=self._dims,
            dtype=self._bounds.dtype,
            device=self._bounds.device,
        )
        new_y = torch.as_tensor(
            [
                -observation.objective if self._minimise else observation.objective
                for observation in observations
            ],
            dtype=self._bounds.dtype,
            device=self._bounds.device,
        )
        new_err = torch.as_tensor(
            [
                self._observation_noise_floor
                if observation.noise_std is None
                else max(float(observation.noise_std), self._observation_noise_floor)
                for observation in observations
            ],
            dtype=self._bounds.dtype,
            device=self._bounds.device,
        )

        self._x_obs = torch.vstack((self._x_obs, new_x))
        self._y_obs_score = torch.hstack((self._y_obs_score, new_y))
        self._y_obs_err = torch.hstack((self._y_obs_err, new_err))

    def describe(self) -> dict[str, Any]:
        exploration_description = None
        if self._exploration_strategy is not None:
            if hasattr(self._exploration_strategy, "describe"):
                exploration_description = self._exploration_strategy.describe()
            else:
                exploration_description = {
                    "kind": type(self._exploration_strategy).__name__
                }
        return {
            "kind": "nubo_bayesian_optimisation",
            "dims": self._dims,
            "bounds": self._bounds.tolist(),
            "batch_size": self._batch_size,
            "acquisition": self._acquisition_name,
            "fit_steps": self._fit_steps,
            "fit_lr": self._fit_lr,
            "acquisition_num_starts": self._acquisition_num_starts,
            "surrogate_num_starts": self._surrogate_num_starts,
            "batch_mc_samples": self._batch_mc_samples,
            "batch_acq_lr": self._batch_acq_lr,
            "batch_acq_steps": self._batch_acq_steps,
            "batch_ucb_beta": self._batch_ucb_beta,
            "max_batches": self._max_batches,
            "minimise": self._minimise,
            "dtype": str(self._dtype).replace("torch.", ""),
            "seed_points": int(self._seed_points.shape[0]),
            "min_normalised_distance": self._min_normalised_distance,
            "observation_noise_floor": self._observation_noise_floor,
            "input_normalisation": "bounds",
            "exploration_strategy": exploration_description,
        }

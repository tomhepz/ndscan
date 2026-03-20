"""Host-runtime Bayesian optimisation example with manual and geometric exploration.

This example shows the intended layering for more complicated adaptive search:

- the host runtime still owns batching, persistence, and nested scan structure,
- an ask/tell point policy wraps an optional optimiser backend,
- manual search and exploration phases are encoded as exploration strategies rather
  than hard-coded into the runtime itself.

The concrete search recipe here is:

1. start from a small random initial design,
2. run batch Bayesian optimisation proposals,
3. every third BO batch, also include:
   - an explicit hand-picked manual point on the first exploration batch,
   - one MHCS point in a large uncovered region,
   - one local grid point around the GP surrogate optimum.

That mirrors the kind of "manual search + geometric search + model-based BO" workflow
you described, while keeping the runtime API itself small.
"""

from __future__ import annotations

import math

from ndscan.experiment import (
    AskTellOptimiserPointPolicy,
    ExecutionPolicy,
    ExpFragment,
    FloatChannel,
    FloatParam,
    ScanRequest,
    make_fragment_host_scan_exp,
    make_fragment_host_dashboard_scan_exp
)

try:
    from ndscan.experiment.optimisation import (
        CompositeExplorationStrategy,
        ExplicitBatchExplorationStrategy,
        LocalLengthscaleExplorationStrategy,
        MhcsExplorationStrategy,
        NuboBatchBayesianOptimisationBackend,
        ScheduledExplorationStrategy,
        extract_scalar_channel_objective,
    )
except ModuleNotFoundError as exc:
    raise ImportError(
        "host_runtime_bayesian_optimisation requires the optional Bayesian "
        "optimisation stack (torch, gpytorch, nubo)"
    ) from exc


class MultiWellSurfaceFragment(ExpFragment):
    """Two-dimensional multimodal cost surface."""

    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_param("y", FloatParam, "y", default=0.0)
        self.setattr_result("cost", FloatChannel)

    def run_once(self):
        x = self.x.get()
        y = self.y.get()

        # A broad quadratic bowl with two attractive local wells makes exploration
        # meaningful without requiring any external benchmark package in the fragment.
        background = 0.12 * (x**2 + y**2)
        well_a = -1.6 * math.exp(-((x + 1.1) ** 2 / 0.5 + (y - 0.8) ** 2 / 0.7))
        well_b = -1.2 * math.exp(-((x - 1.4) ** 2 / 0.6 + (y + 1.0) ** 2 / 0.5))
        ripple = 0.08 * math.sin(2.5 * x) * math.cos(2.0 * y)
        self.cost.push(background + well_a + well_b + ripple)


def _make_request(fragment: MultiWellSurfaceFragment) -> ScanRequest:
    exploration = ScheduledExplorationStrategy(
        every_batches=3,
        strategy=CompositeExplorationStrategy(
            [
                ExplicitBatchExplorationStrategy(
                    {
                        # One explicit "manual search" batch on the first BO/exploration
                        # cycle. Later cycles naturally fall back to MHCS + local.
                        0: [(-1.75, 1.75)],
                    }
                ),
                MhcsExplorationStrategy(
                    num_points=1,
                    min_normalised_distance=0.08,
                ),
                LocalLengthscaleExplorationStrategy(
                    num_points=1,
                    axis_points=5,
                    pair_points=5,
                    span_in_lengthscales=1.5,
                    min_normalised_distance=0.08,
                    surrogate_num_starts=4,
                ),
            ],
            min_normalised_distance=0.08,
        ),
    )

    backend = NuboBatchBayesianOptimisationBackend(
        bounds=[[-2.0, -2.0], [2.0, 2.0]],
        batch_size=4,
        initial_design_size=8,
        random_seed=0,
        acquisition_name="ucb",
        fit_steps=40,
        acquisition_num_starts=3,
        surrogate_num_starts=5,
        batch_mc_samples=32,
        batch_acq_lr=0.08,
        batch_acq_steps=60,
        max_batches=20,
        minimise=True,
        min_normalised_distance=0.05,
        exploration_strategy=exploration,
    )

    return ScanRequest(
        axes=(fragment.x, fragment.y),
        point_policy=AskTellOptimiserPointPolicy(
            backend,
            extract_scalar_channel_objective("channel_0"),
        ),
        execution_policy=ExecutionPolicy(max_points_per_batch=4),
        metadata={"demo_name": "host_runtime_bayesian_optimisation"},
    )


HostRuntimeBayesianOptimisation = make_fragment_host_scan_exp(
    MultiWellSurfaceFragment,
    _make_request,
)

HostRuntimeBayesianOptimisationDashboard = make_fragment_host_dashboard_scan_exp(MultiWellSurfaceFragment)
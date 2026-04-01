"""Host-runtime Bayesian optimisation example with structured exploration phases.

This example shows the intended layering for more complicated adaptive search:

- the host runtime still owns batching, persistence, and nested scan structure,
- an ask/tell point policy wraps an optional optimiser backend,
- manual search and exploration phases are encoded as exploration strategies rather
  than hard-coded into the runtime itself.

    The concrete search recipe here is intentionally closer to the standalone NUBO-style
    script used to prototype the BO dashboard:

1. start from a denser random initial design,
2. run batch Bayesian optimisation proposals,
3. every 25th BO batch, also include:
   - one MHCS point in a large uncovered region,
   - one local grid point around the GP surrogate optimum.

This makes the live BO trace spend much less time repeatedly injecting local
exploration structure and behave more like a long-running BO search with occasional
"human-palatable" exploratory scan parts.

The structured exploration passes keep a relatively large duplicate-rejection radius.
The BO proposals themselves use a much lighter duplicate filter so they can keep
refining around the current optimum without stalling.
"""

from __future__ import annotations

import math

from artiq.experiment import *
from ndscan.define import *
from ndscan.define import annotations
from ndscan.scan import *
from ndscan.runtime.api import *

try:
    from ndscan.scan.optimisation import (
        CompositeExplorationStrategy,
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


DEFAULT_INITIAL_DESIGN_SIZE = 14
DEFAULT_BATCH_SIZE = 4
DEFAULT_FIT_STEPS = 200
DEFAULT_FIT_LR = 0.1
DEFAULT_ACQUISITION_NUM_STARTS = 5
DEFAULT_SURROGATE_NUM_STARTS = 20
DEFAULT_BATCH_MC_SAMPLES = 128
DEFAULT_BATCH_ACQ_LR = 0.1
DEFAULT_BATCH_ACQ_STEPS = 200
DEFAULT_EXPLORATION_EVERY_BATCHES = 3
DEFAULT_MAX_BATCHES = 201


def _make_request(fragment: MultiWellSurfaceFragment) -> ScanRequest:
    exploration = ScheduledExplorationStrategy(
        every_batches=DEFAULT_EXPLORATION_EVERY_BATCHES,
        offset=DEFAULT_EXPLORATION_EVERY_BATCHES - 1,
        strategy=CompositeExplorationStrategy(
            [
                MhcsExplorationStrategy(
                    num_points=4,
                    min_normalised_distance=0.02,
                ),
                LocalLengthscaleExplorationStrategy(
                    num_points=20,
                    axis_points=7,
                    pair_points=5,
                    span_in_lengthscales=2.0,
                    min_normalised_distance=0.02,
                    surrogate_num_starts=DEFAULT_SURROGATE_NUM_STARTS,
                ),
            ],
            min_normalised_distance=0.02,
        ),
    )

    backend = NuboBatchBayesianOptimisationBackend(
        bounds=[[-2.0, -2.0], [2.0, 2.0]],
        batch_size=DEFAULT_BATCH_SIZE,
        initial_design_size=DEFAULT_INITIAL_DESIGN_SIZE,
        random_seed=0,
        acquisition_name="ucb",
        fit_steps=DEFAULT_FIT_STEPS,
        fit_lr=DEFAULT_FIT_LR,
        acquisition_num_starts=DEFAULT_ACQUISITION_NUM_STARTS,
        surrogate_num_starts=DEFAULT_SURROGATE_NUM_STARTS,
        batch_mc_samples=DEFAULT_BATCH_MC_SAMPLES,
        batch_acq_lr=DEFAULT_BATCH_ACQ_LR,
        batch_acq_steps=DEFAULT_BATCH_ACQ_STEPS,
        batch_ucb_beta=1.96**2,
        max_batches=DEFAULT_MAX_BATCHES,
        minimise=True,
        min_normalised_distance=1e-6,
        exploration_strategy=exploration,
    )

    return ScanRequest(
        axes=(fragment.x, fragment.y),
        point_policy=AskTellOptimiserPointPolicy(
            backend,
            extract_scalar_channel_objective("channel_0"),
        ),
        execution_policy=ExecutionPolicy(max_points_per_batch=DEFAULT_BATCH_SIZE),
        metadata={"demo_name": "host_runtime_bayesian_optimisation"},
    )


HostRuntimeBayesianOptimisation = make_fragment_prepared_scan_exp(
    MultiWellSurfaceFragment,
    _make_request,
)

HostRuntimeBayesianOptimisationDashboard = make_fragment_prepared_dashboard_scan_exp(MultiWellSurfaceFragment)

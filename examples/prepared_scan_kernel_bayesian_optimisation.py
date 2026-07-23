"""Prepared-runtime Bayesian optimisation example with a resident kernel objective.

This example demonstrates the intended BO layering:

- the host still owns the ask/tell optimiser state and point selection,
- the objective fragment itself runs on the core device,
- BO batches can therefore remain fully host-directed while the core execution region
  is entered only once for the whole run.

The objective is intentionally simple and one-dimensional so the example stays focused
on execution shape rather than optimiser tuning. With ``max_points_per_batch = 1``,
each BO step becomes its own runtime batch even though the backend itself is capable
of larger logical batches. That makes
it easy to verify the desired property in tests:

- many BO batches,
- one resident kernel executor entry.
"""

from __future__ import annotations

from artiq.experiment import *

from ndscan.define import *
from ndscan.runtime.api import *
from ndscan.scan import *

try:
    from ndscan.scan.optimisation import (
        AskTellOptimiserPointPolicy,
        NuboBatchBayesianOptimisationBackend,
        extract_scalar_channel_objective,
    )
except ImportError as exc:
    raise ImportError(
        "prepared_scan_kernel_bayesian_optimisation requires the optional Bayesian "
        "optimisation stack (torch, gpytorch, nubo)"
    ) from exc


class KernelBayesianOptimisationFragment(ExpFragment):
    """One-dimensional objective evaluated on the core device."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("cost", FloatChannel)

    @kernel
    def run_once(self):
        x = self.x.get()
        self.cost.push((x - 0.3) ** 2)


DEFAULT_FIT_STEPS = 200
DEFAULT_FIT_LR = 0.05
DEFAULT_ACQUISITION_NUM_STARTS = 6
DEFAULT_SURROGATE_NUM_STARTS = 12
DEFAULT_BATCH_MC_SAMPLES = 64
DEFAULT_BATCH_ACQ_LR = 0.05
DEFAULT_BATCH_ACQ_STEPS = 80
DEFAULT_MAX_BATCHES = 50


def make_request(
    fragment: KernelBayesianOptimisationFragment,
    *,
    fit_steps: int = DEFAULT_FIT_STEPS,
    fit_lr: float = DEFAULT_FIT_LR,
    acquisition_num_starts: int = DEFAULT_ACQUISITION_NUM_STARTS,
    surrogate_num_starts: int = DEFAULT_SURROGATE_NUM_STARTS,
    batch_mc_samples: int = DEFAULT_BATCH_MC_SAMPLES,
    batch_acq_lr: float = DEFAULT_BATCH_ACQ_LR,
    batch_acq_steps: int = DEFAULT_BATCH_ACQ_STEPS,
    max_batches: int = DEFAULT_MAX_BATCHES,
) -> ScanRequest:
    """Build a BO request with heavier-than-test defaults.

    The defaults are intentionally more expensive than the tiny settings used in the
    tests so the live example produces a saner in-loop GP fit and acquisition trace.
    """

    backend = NuboBatchBayesianOptimisationBackend(
        bounds=[[-1.0], [1.0]],
        batch_size=2,
        initial_points=[[-1.0], [1.0]],
        random_seed=0,
        fit_steps=fit_steps,
        fit_lr=fit_lr,
        acquisition_name="ucb",
        acquisition_num_starts=acquisition_num_starts,
        surrogate_num_starts=surrogate_num_starts,
        batch_mc_samples=batch_mc_samples,
        batch_acq_lr=batch_acq_lr,
        batch_acq_steps=batch_acq_steps,
        max_batches=max_batches,
        minimise=True,
        min_normalised_distance=0.00,
    )

    return ScanRequest(
        axes=(fragment.x,),
        point_policy=AskTellOptimiserPointPolicy(
            backend,
            extract_scalar_channel_objective("channel_0"),
        ),
        execution_policy=ExecutionPolicy(max_points_per_batch=1),
        metadata={"demo_name": "prepared_scan_kernel_bayesian_optimisation"},
    )


PreparedScanKernelBayesianOptimisation = make_fragment_prepared_scan_exp(
    KernelBayesianOptimisationFragment,
    make_request,
)

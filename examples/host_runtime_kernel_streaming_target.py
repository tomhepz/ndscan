"""Smallest runnable host-runtime example for the kernel-streaming backend.

This remains the canonical acceptance/benchmark shape for the first resident-kernel
executor:

- one scanned numeric parameter,
- one scalar saved result channel,
- host-selected point batches,
- one resident kernel execution region for the whole scan.
"""

from __future__ import annotations

from ndscan.experiment import (
    ExecutionPolicy,
    ExpFragment,
    FloatChannel,
    FloatParam,
    ScanRequest,
    make_fragment_prepared_scan_exp,
    kernel,
)


class KernelStreamingTargetFragment(ExpFragment):
    """Tiny kernel-capable leaf fragment for executor bring-up.

    The fragment is deliberately simple:

    - one scanned numeric parameter,
    - one scalar saved result channel,
    - no nested scans,
    - no mappings,
    - no default analyses.

    That makes it the smallest useful target for proving:

    - one resident kernel execution region,
    - host-fed point batches,
    - no repeated compilation per point.
    """

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    @kernel
    def run_once(self):
        self.y.push(self.x.get() + 1.0)


def make_kernel_streaming_target_request(fragment: KernelStreamingTargetFragment) -> ScanRequest:
    """Return the canonical request for the first kernel-streaming acceptance test."""

    return ScanRequest.linear(
        fragment.x,
        start=-10.0,
        stop=10.0,
        num_points=101,
        execution_policy=ExecutionPolicy(max_points_per_batch=10),
        metadata={"demo_name": "host_runtime_kernel_streaming_target"},
    )


HostRuntimeKernelStreamingTarget = make_fragment_prepared_scan_exp(
    KernelStreamingTargetFragment,
    make_kernel_streaming_target_request,
)

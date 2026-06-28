"""Smallest runnable prepared-runtime example for the kernel-streaming backend.

This remains the canonical acceptance/benchmark shape for the first resident-kernel
executor:

- one scanned numeric parameter,
- one scalar saved result channel,
- host-selected point batches,
- one resident kernel execution region for the whole scan.
"""

from __future__ import annotations

from artiq.experiment import *
from ndscan.define import *
from ndscan.define import annotations
from ndscan.scan import *
from ndscan.runtime.api import *


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
        metadata={"demo_name": "prepared_scan_kernel_streaming_target"},
    )


PreparedScanKernelStreamingTarget = make_fragment_prepared_scan_exp(
    KernelStreamingTargetFragment,
    make_kernel_streaming_target_request,
)

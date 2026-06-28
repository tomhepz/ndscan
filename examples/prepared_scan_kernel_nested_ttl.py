"""Prepared nested-kernel prepared-runtime example with a real TTL leaf.

This is the hardware-oriented counterpart to
``prepared_scan_kernel_nested.py``:

1. the host still chooses all ``p`` and ``x`` points in batches,
2. the root, child, and leaf point bodies all run on the core device,
3. the leaf performs a real TTL pulse on ``ttl0``,
4. the nested call graph remains compiler-visible through prepared child scans.

So this example demonstrates the shape discussed for "one kernel region, host-chosen
points":

- the leaf is ``@kernel`` because it touches RTIO,
- the parent fragments are also ``@kernel`` so the nested prepared scans stay inside
  one compiled call graph,
- but the actual scan points still come from host-side ``ScanRequest`` objects.

The leaf also pushes the pulse width back into a result channel so the executed point
stream is visible in saved data. Inspect datasets such as:

- ``ndscan.rid_<rid>.site.root.subscans.scan_p.points.channel_0``
- ``ndscan.rid_<rid>.site.root.subscans.scan_p.subscans.scan_x.points.channel_0``
- ``ndscan.rid_<rid>.site.root.subscans.scan_p.subscans.scan_x.segments.parent_point_index``

This example uses ``ttl0`` because that device exists in the local ARTIQ setup used
for testing here. If your setup prefers a different scope trigger/output line, change
the device name in ``KernelTtlPulseLeafFragment.build_fragment()``.
"""

from __future__ import annotations

import numpy as np
from artiq.language.core import delay
from artiq.language.units import us

from artiq.experiment import *
from ndscan.define import *
from ndscan.define import annotations
from ndscan.scan import *
from ndscan.runtime.api import *


class KernelTtlPulseLeafFragment(ExpFragment):
    """Leaf fragment that emits a short TTL pulse for each scanned point."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_device("ttl0")
        self.setattr_param("p", FloatParam, "p", default=1.0)
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_param(
            "base_pulse_us",
            FloatParam,
            "Base pulse width",
            default=0.5,
            unit="us",
            min=0.0,
        )
        self.setattr_result("pulse_width_us", FloatChannel, unit="us")

    @kernel
    def device_setup(self):
        self.core.break_realtime()
        self.ttl0.off()

    @kernel
    def run_once(self):
        pulse_width_us = self.base_pulse_us.get() + self.p.get() + self.x.get()
        self.core.break_realtime()
        self.ttl0.on()
        delay(pulse_width_us * us)
        self.ttl0.off()
        self.pulse_width_us.push(pulse_width_us)


class PreparedKernelTtlScanXFragment(ExpFragment):
    """One ``p`` point that acquires an inner prepared scan over ``x``."""

    def build_fragment(self):
        self.setattr_device("core")
        self.scan_x = setattr_prepared_child_scan(
            self,
            "pulse",
            KernelTtlPulseLeafFragment,
            scan_name="scan_x",
        )
        self.setattr_result("current_p", FloatChannel)

    def host_setup(self):
        x_points = np.linspace(0.0, 4.0, 5).tolist()
        self.scan_x.configure(
            ScanRequest.cartesian(
                [(self.pulse.x, x_points)],
                metadata={"analysis_note": "prepared kernel TTL scan over x"},
                execution_policy=ExecutionPolicy(max_points_per_batch=2),
            )
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.scan_x.acquire()
        self.current_p.push(self.pulse.p.get())


class PreparedKernelNestedTtlFragment(ExpFragment):
    """Top-level fragment launching one prepared kernel child scan over ``p``."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("outer", FloatParam, "outer", default=1.0)
        self.scan_p = setattr_prepared_child_scan(
            self,
            "scan_x",
            PreparedKernelTtlScanXFragment,
            scan_name="scan_p",
        )
        self.setattr_result("completed", FloatChannel)

    def host_setup(self):
        # Prime the detached child before outer kernel compilation so its own inner
        # prepared scan has already fixed its compiler-visible shape.
        self.scan_p.prime()

        p_points = [1.0, 2.0, 3.0]
        self.scan_p.configure(
            ScanRequest.cartesian(
                [(self.scan_x.pulse.p, p_points)],
                metadata={"analysis_note": "prepared kernel TTL scan over p"},
                execution_policy=ExecutionPolicy(max_points_per_batch=2),
            )
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.scan_p.acquire()
        self.completed.push(self.outer.get())


PreparedScanKernelNestedTtl = make_fragment_prepared_scan_exp(
    PreparedKernelNestedTtlFragment,
    lambda fragment: ScanRequest.explicit(
        [fragment.outer],
        [[1.0]],
        metadata={"demo_name": "prepared_scan_kernel_nested_ttl"},
    ),
)

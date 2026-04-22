"""Prepared nested-kernel host-runtime example with a fixed core-side wait.

This is a timing-oriented counterpart to
``host_runtime_prepared_kernel_nested_ttl.py``. It keeps the same nested prepared
scan shape, but the leaf does not touch a TTL. Instead it waits on the core RTIO
counter for a fixed wall-clock duration before pushing a result.

The example is useful for benchmarking the host-observed ``core.run(...)`` boundary:

- the total in-kernel wait is known from the scan shape,
- any additional time inside ``core.run(...)`` is communication, RPC/result handling,
  compilation/upload, or other runtime overhead,
- no experiment hardware other than ``core`` is touched.
"""

from __future__ import annotations

import numpy as np
from artiq.experiment import *
from artiq.language.units import us

from ndscan.define import *
from ndscan.runtime.api import *
from ndscan.scan import *

DEFAULT_WALL_WAIT_US = 5000.0


class KernelFixedWaitLeafFragment(ExpFragment):
    """Leaf fragment that waits on the core wall clock for each scanned point."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("p", FloatParam, "p", default=1.0)
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_param(
            "wall_wait_us",
            FloatParam,
            "Wall wait per leaf point",
            default=DEFAULT_WALL_WAIT_US,
            unit="us",
            min=0.0,
        )
        self.setattr_result("reported_wait_us", FloatChannel, unit="us")

    @kernel
    def run_once(self):
        wall_wait_us = self.wall_wait_us.get()
        wait_mu = self.core.seconds_to_mu(wall_wait_us * us)
        self.core.wait_until_mu(self.core.get_rtio_counter_mu() + wait_mu)
        self.reported_wait_us.push(wall_wait_us)


class PreparedKernelFixedWaitScanXFragment(ExpFragment):
    """One ``p`` point that acquires an inner prepared scan over ``x``."""

    def build_fragment(self):
        self.setattr_device("core")
        self.scan_x = setattr_prepared_child_scan(
            self,
            "wait",
            KernelFixedWaitLeafFragment,
            scan_name="scan_x",
        )
        self.setattr_result("current_p", FloatChannel)

    def host_setup(self):
        x_points = np.linspace(0.0, 4.0, 5).tolist()
        self.scan_x.configure(
            ScanRequest.cartesian(
                [(self.wait.x, x_points)],
                metadata={"analysis_note": "prepared kernel fixed-wall-wait scan over x"},
                execution_policy=ExecutionPolicy(max_points_per_batch=2),
            )
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.scan_x.acquire()
        self.current_p.push(self.wait.p.get())


class PreparedKernelNestedWaitFragment(ExpFragment):
    """Top-level fragment launching one prepared fixed-wait child scan over ``p``."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("outer", FloatParam, "outer", default=1.0)
        self.scan_p = setattr_prepared_child_scan(
            self,
            "scan_x",
            PreparedKernelFixedWaitScanXFragment,
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
                [(self.scan_x.wait.p, p_points)],
                metadata={"analysis_note": "prepared kernel fixed-wall-wait scan over p"},
                execution_policy=ExecutionPolicy(max_points_per_batch=2),
            )
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.scan_p.acquire()
        self.completed.push(self.outer.get())


HostRuntimePreparedKernelNestedWait = make_fragment_prepared_scan_exp(
    PreparedKernelNestedWaitFragment,
    lambda fragment: ScanRequest.explicit(
        [fragment.outer],
        [[1.0]],
        metadata={"demo_name": "host_runtime_prepared_kernel_nested_wait"},
    ),
)

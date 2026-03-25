"""Prepared nested-kernel host-runtime example.

This is the compiler-friendly counterpart to
``host_runtime_nested_p_variation.py``. The scan shape is intentionally similar:

1. the top-level runtime executes exactly one root point,
2. that root point launches a prepared scan over ``p``,
3. each ``p`` point launches a prepared scan over ``x``,
4. the root, child, and leaf point bodies all run on the core device.

The important difference is the API shape:

- structural setup happens in ``build_fragment()``,
- concrete scan requests are configured in ``host_setup()``,
- nested execution uses ``PreparedChildScan.acquire()``.

This makes the nested call graph compiler-visible while still letting the host choose
point batches at every level.

After a run, inspect datasets such as:

- ``ndscan.rid_<rid>.site.root.scan_p.points.param_0``
- ``ndscan.rid_<rid>.site.root.scan_p.scan_x.points.param_0``
- ``ndscan.rid_<rid>.site.root.scan_p.scan_x.points.channel_0``

to see the ``p`` and ``x`` hierarchy in the saved data.
"""

from __future__ import annotations

import numpy as np

from ndscan.experiment import (
    ExecutionPolicy,
    ExpFragment,
    FloatChannel,
    FloatParam,
    ScanRequest,
    kernel,
    make_fragment_prepared_scan_exp,
    setattr_prepared_child_scan,
)


class KernelLineFragment(ExpFragment):
    """Leaf fragment evaluated on the core device."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("p", FloatParam, "p", default=1.0)
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_param("offset", FloatParam, "offset", default=0.5)
        self.setattr_result("y", FloatChannel)

    @kernel
    def run_once(self):
        self.y.push(self.p.get() * self.x.get() + self.offset.get())


class PreparedScanXFragment(ExpFragment):
    """One ``p`` point that acquires an inner prepared scan over ``x``."""

    def build_fragment(self):
        self.setattr_device("core")
        self.scan_x = setattr_prepared_child_scan(
            self,
            "line",
            KernelLineFragment,
            scan_name="scan_x",
        )
        self.setattr_result("current_p", FloatChannel)

    def host_setup(self):
        x_points = np.linspace(0.0, 5.0, 6).tolist()
        self.scan_x.configure(
            ScanRequest.cartesian(
                [(self.line.x, x_points)],
                metadata={"analysis_note": "prepared kernel scan over x"},
                execution_policy=ExecutionPolicy(max_points_per_batch=3),
            )
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.scan_x.acquire()
        # Record which p value this child point represented.
        self.current_p.push(self.line.p.get())


class PreparedKernelNestedVariationFragment(ExpFragment):
    """Top-level fragment launching one prepared kernel child scan over ``p``."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("outer", FloatParam, "outer", default=1.0)
        self.scan_p = setattr_prepared_child_scan(
            self,
            "scan_x",
            PreparedScanXFragment,
            scan_name="scan_p",
        )
        self.setattr_result("completed", FloatChannel)

    def host_setup(self):
        # Prime the detached child before the outer kernel is first entered so its own
        # prepared inner scan has already fixed its compiler-visible shape.
        self.scan_p.prime()

        p_points = np.linspace(1.0, 5.0, 5).tolist()
        self.scan_p.configure(
            ScanRequest.cartesian(
                [(self.scan_x.line.p, p_points)],
                metadata={"analysis_note": "prepared kernel scan over p"},
                execution_policy=ExecutionPolicy(max_points_per_batch=2),
            )
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.scan_p.acquire()
        self.completed.push(self.outer.get())


HostRuntimePreparedKernelNested = make_fragment_prepared_scan_exp(
    PreparedKernelNestedVariationFragment,
    lambda fragment: ScanRequest.explicit(
        [fragment.outer],
        [[1.0]],
        metadata={"demo_name": "host_runtime_prepared_kernel_nested"},
    ),
)

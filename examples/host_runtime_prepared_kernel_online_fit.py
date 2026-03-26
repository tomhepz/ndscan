"""Prepared kernel child-scan example with host-side online fitting.

This example demonstrates the current intended split:

- the child scan acquires points on the core device via ``PreparedChildScan.acquire()``,
- the host still receives completed batches by RPC,
- the host runs online analysis after each batch boundary,
- the same analysis also produces final analysis results at the end.

The scientific model is deliberately simple: the leaf fragment emits a straight line
``y = 2 * x + 0.5``. The child scan declares a `CustomAnalysis` that fits the slope and
intercept both online and finally, and the parent kernel point reads those fixed
outputs back via ``line_scan.get_outputs()``.

This is also the shape we want to preserve if later additions make either of these
operations kernel-capable:

- point generation could gain an optional kernel implementation,
- line fitting could gain an optional kernel reducer,

but the user-facing concepts should still remain:

- `PointPolicy`,
- `CustomAnalysis`,
- `PreparedChildScan`.
"""

from __future__ import annotations

import numpy as np

from artiq.experiment import *
from ndscan.define import *
from ndscan.define import annotations
from ndscan.scan import *
from ndscan.runtime.api import *


def fit_line(xs, ys) -> tuple[float, float]:
    """Return the least-squares slope/intercept for ``y = m * x + b``."""

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    design = np.column_stack([xs, np.ones_like(xs)])
    (slope, intercept), _, _, _ = np.linalg.lstsq(design, ys, rcond=None)
    return float(slope), float(intercept)


class KernelLineWithOnlineFitFragment(ExpFragment):
    """Kernel leaf whose data is fitted online on the host."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    @kernel
    def run_once(self):
        self.y.push(2.0 * self.x.get() + 0.5)

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.x],
                self._fit_running_line,
                analysis_results=[
                    FloatChannel("fit_slope", "Running best-fit slope"),
                    FloatChannel("fit_intercept", "Running best-fit intercept"),
                ],
                online_fn=self._fit_running_line,
                online_analysis_identifier="running_line_fit",
            )
        ]

    def _fit_running_line(self, axis_values, result_values, analysis_results):
        xs = np.asarray(axis_values[self.x], dtype=float)
        ys = np.asarray(result_values[self.y], dtype=float)

        slope, intercept = fit_line(xs, ys)
        fit_xs = np.linspace(float(xs.min()), float(xs.max()), 50)
        fit_ys = slope * fit_xs + intercept

        analysis_results["fit_slope"].push(slope)
        analysis_results["fit_intercept"].push(intercept)
        return [
            annotations.curve_1d(
                x_axis=self.x,
                x_values=fit_xs,
                y_axis=self.y,
                y_values=fit_ys,
            )
        ]


class PreparedKernelOnlineFitFragment(ExpFragment):
    """Top-level kernel point that launches the prepared child scan."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("outer", FloatParam, "outer", default=1.0)
        self.line_scan = setattr_prepared_child_scan(
            self,
            "line",
            KernelLineWithOnlineFitFragment,
            scan_name="line_scan",
            expose_outputs=["fit_slope", "fit_intercept"],
        )
        self.setattr_result("completed", FloatChannel)
        self.setattr_result("fit_slope", FloatChannel)
        self.setattr_result("fit_intercept", FloatChannel)

    def host_setup(self):
        x_points = np.linspace(0.0, 5.0, 6).tolist()
        self.line_scan.configure(
            ScanRequest.cartesian(
                [(self.line.x, x_points)],
                metadata={"analysis_note": "host-side running line fit over kernel data"},
                execution_policy=ExecutionPolicy(max_points_per_batch=3),
            )
        )
        super().host_setup()

    @kernel
    def run_once(self):
        self.line_scan.acquire()
        fit_slope, fit_intercept = self.line_scan.get_outputs()
        self.fit_slope.push(fit_slope)
        self.fit_intercept.push(fit_intercept)
        self.completed.push(self.outer.get())


HostRuntimePreparedKernelOnlineFit = make_fragment_prepared_scan_exp(
    PreparedKernelOnlineFitFragment,
    lambda fragment: ScanRequest.explicit(
        [fragment.outer],
        [[1.0]],
        metadata={"demo_name": "host_runtime_prepared_kernel_online_fit"},
    ),
)

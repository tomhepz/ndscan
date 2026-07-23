"""Prepared-runtime demo tailored for the live site-tree viewer.

This example is meant to exercise the new runtime applet rather than any particular
lab workflow. It gives the viewer a few things to chew on at once:

- a root point that launches one nested outer scan,
- an outer scan zipped over two parameters, ``center`` and ``curvature``,
- two sibling child scan sites for each outer point, ``coarse_scan`` and
  ``fine_scan``,
- a quadratic peak fit on each child scan,
- a deliberate ``time.sleep()`` in the leaf point body so points arrive slowly enough
  to watch live.

Suggested viewer interactions:

1. watch ``scan_outer`` fill in live,
2. switch its x-axis between ``center`` and ``curvature``,
3. click an outer point,
4. use the child-site dropdown to switch between ``coarse_scan`` and ``fine_scan``.
"""

from __future__ import annotations

import time

import numpy as np
from artiq.experiment import *

from ndscan.define import *
from ndscan.define import annotations
from ndscan.runtime.api import *
from ndscan.scan import *


def fit_quadratic_peak(xs, ys) -> tuple[float, float, float]:
    """Fit ``y = ax^2 + bx + c`` and return vertex position, value, and scale."""

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    a, b, c = np.polyfit(xs, ys, deg=2)
    if abs(a) < 1e-12:
        peak_center = float(xs[np.argmax(ys)])
        peak_value = float(np.max(ys))
    else:
        peak_center = float(-b / (2.0 * a))
        peak_value = float(a * peak_center**2 + b * peak_center + c)
    return peak_center, peak_value, float(a)


class DelayedQuadraticPeakFragment(ExpFragment):
    """Leaf fragment producing a simple quadratic peak with a visible delay."""

    def build_fragment(self):
        self.setattr_param("center", FloatParam, "peak center", default=0.0)
        self.setattr_param("curvature", FloatParam, "peak curvature", default=0.2)
        self.setattr_param("x", FloatParam, "probe position", default=0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        time.sleep(0.08)
        x = self.x.get()
        center = self.center.get()
        curvature = self.curvature.get()
        self.y.push(3.0 - curvature * (x - center) ** 2)

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.x],
                self._analyse_peak,
                [
                    FloatChannel("fit_center", "Quadratic-fit center"),
                    FloatChannel("fit_peak", "Quadratic-fit peak value"),
                ],
            )
        ]

    def _analyse_peak(self, axis_values, result_values, analysis_results):
        xs = np.asarray(axis_values[self.x], dtype=float)
        ys = np.asarray(result_values[self.y], dtype=float)
        fit_center, fit_peak, fit_scale = fit_quadratic_peak(xs, ys)

        analysis_results["fit_center"].push(fit_center)
        analysis_results["fit_peak"].push(fit_peak)
        return [
            annotations.computed_curve(
                function_name="parabola",
                parameters={
                    "position": fit_center,
                    "scale": fit_scale,
                    "offset": fit_peak,
                },
                associated_channels=[self.y],
            )
        ]


class ZippedOuterPointFragment(ExpFragment):
    """One outer point that launches both a coarse and a fine subscan."""

    def build_fragment(self):
        self.setattr_param("center", FloatParam, "logical center", default=0.0)
        self.setattr_param("curvature", FloatParam, "logical curvature", default=0.2)

        self.coarse_scan = setattr_prepared_child_scan(
            self,
            "coarse",
            DelayedQuadraticPeakFragment,
            scan_name="coarse_scan",
            extra_metadata={"viewer_note": "wide quadratic peak scan"},
        )
        self.fine_scan = setattr_prepared_child_scan(
            self,
            "fine",
            DelayedQuadraticPeakFragment,
            scan_name="fine_scan",
            extra_metadata={"viewer_note": "zoomed quadratic peak scan"},
        )

        self.rebind_param(
            self.coarse.center,
            [self.center],
            lambda values: values[self.center],
            description="Drive coarse child center from the outer logical center",
        )
        self.rebind_param(
            self.coarse.curvature,
            [self.curvature],
            lambda values: values[self.curvature],
            description="Drive coarse child curvature from the outer logical curvature",
        )
        self.rebind_param(
            self.fine.center,
            [self.center],
            lambda values: values[self.center],
            description="Drive fine child center from the outer logical center",
        )
        self.rebind_param(
            self.fine.curvature,
            [self.curvature],
            lambda values: values[self.curvature],
            description="Drive fine child curvature from the outer logical curvature",
        )

        self.setattr_result("coarse_center", FloatChannel)
        self.setattr_result("fine_center", FloatChannel)
        self.setattr_result("fine_peak", FloatChannel)

    def run_once(self):
        self.coarse_scan.configure(
            ScanRequest.cartesian(
                [(self.coarse.x, np.linspace(-4.0, 4.0, 9).tolist())],
                metadata={"viewer_note": "coarse x scan"},
                execution_policy=ExecutionPolicy(max_points_per_batch=1),
            )
        )
        coarse_outputs = self.coarse_scan.execute()
        coarse_center = float(coarse_outputs["fit_center"])

        self.fine_scan.configure(
            ScanRequest.cartesian(
                [(
                    self.fine.x,
                    np.linspace(coarse_center - 1.0, coarse_center + 1.0, 7).tolist(),
                )],
                metadata={"viewer_note": "fine x scan around the coarse center"},
                execution_policy=ExecutionPolicy(max_points_per_batch=1),
            )
        )
        fine_outputs = self.fine_scan.execute()

        self.coarse_center.push(coarse_center)
        self.fine_center.push(float(fine_outputs["fit_center"]))
        self.fine_peak.push(float(fine_outputs["fit_peak"]))


class RuntimeViewerDemoFragment(ExpFragment):
    """Root fragment launching the zipped outer scan."""

    def build_fragment(self):
        self.outer_scan = setattr_prepared_child_scan(
            self,
            "outer_point",
            ZippedOuterPointFragment,
            scan_name="scan_outer",
            extra_metadata={
                "viewer_note": "zipped outer scan over center and curvature"
            },
        )
        self.setattr_result("latest_fine_center", FloatChannel)
        self.setattr_result("latest_fine_peak", FloatChannel)

    def run_once(self):
        center_points = [-1.8, -0.8, 0.2, 1.1, 2.0]
        curvature_points = [0.10, 0.16, 0.24, 0.33, 0.45]
        self.outer_scan.configure(
            ScanRequest.zipped(
                [
                    (self.outer_point.center, center_points),
                    (self.outer_point.curvature, curvature_points),
                ],
                metadata={"demo_name": "prepared_scan_live_viewer_demo"},
                execution_policy=ExecutionPolicy(max_points_per_batch=1),
            )
        )
        self.outer_scan.execute()
        result = self.outer_scan.inspect()
        self.latest_fine_center.push(float(result.values[self.outer_point.fine_center][-1]))
        self.latest_fine_peak.push(float(result.values[self.outer_point.fine_peak][-1]))


PreparedScanLiveViewerDemo = make_fragment_prepared_scan_exp(
    RuntimeViewerDemoFragment,
    lambda fragment: ScanRequest.single(
        metadata={"demo_name": "prepared_scan_live_viewer_demo"}
    ),
)

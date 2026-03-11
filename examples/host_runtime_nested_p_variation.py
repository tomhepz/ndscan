"""Nested host-runtime scan example.

This is the host-runtime rewrite of the classic "subscan inside a subscan" shape.

Key differences from the legacy ``SubscanExpFragment`` approach:

- no ``SubscanExpFragment`` subclasses,
- no ``setattr_subscan(...)``,
- nested scans are launched explicitly with ``run_host_scan(...)``,
- child scan sites are derived explicitly with ``make_child_scan_site(...)``.

The runtime model is intentionally simple:

1. the top-level experiment executes exactly one root point,
2. that root point launches a scan over ``p``,
3. each ``p`` point launches a scan over ``x``,
4. the scanned fragments declare default analyses, and the parent level consumes the
   returned ``analysis_results`` explicitly.

This keeps the recursive structure obvious and makes it easy to inspect how data flows
through the new runtime while still reusing ndscan's fragment-side analysis API.
"""

from __future__ import annotations

import numpy as np

from ndscan.experiment import (
    CustomAnalysis,
    ExpFragment,
    FloatChannel,
    FloatParam,
    OpaqueChannel,
    ScanRequest,
    annotations,
    make_child_scan_site,
    make_fragment_host_scan_exp,
    run_host_scan,
)


def fit_line_through_origin(xs, ys) -> float:
    """Return the least-squares slope for ``y = m * x``.

    The inner scan has a fixed zero intercept, so the fit can stay deliberately simple
    and avoid any dependence on the legacy default-analysis machinery.
    """

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return float(np.dot(xs, ys) / np.dot(xs, xs))


def fit_power_law_exponent(ps, ms) -> float:
    """Return the least-squares exponent for ``m = p**e``.

    We fit in log-space and deliberately avoid ``p = 0`` in the outer scan, because
    the transformed model requires strictly positive inputs.
    """

    ps = np.asarray(ps, dtype=float)
    ms = np.asarray(ms, dtype=float)
    log_p = np.log(ps)
    log_m = np.log(ms)
    return float(np.dot(log_p, log_m) / np.dot(log_p, log_p))


class LineFragment(ExpFragment):
    """Small leaf fragment producing ``y = (p**e) * x``."""

    def build_fragment(self):
        self.setattr_param("p", FloatParam, "p", default=2.0)
        self.setattr_param("e", FloatParam, "e", default=2.0)
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push((self.p.get() ** self.e.get()) * self.x.get())

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.x],
                self._analyse_gradient,
                [
                    FloatChannel("m", "Extracted slope"),
                    OpaqueChannel("fit_xs", save_by_default=False),
                    OpaqueChannel("fit_ys", save_by_default=False),
                ],
            )
        ]

    def _analyse_gradient(self, axis_values, result_values, analysis_results):
        xs = np.asarray(axis_values[self.x], dtype=float)
        ys = np.asarray(result_values[self.y], dtype=float)

        m = fit_line_through_origin(xs, ys)
        fit_xs = np.linspace(xs.min(), xs.max(), 50)
        fit_ys = m * fit_xs

        analysis_results["m"].push(m)
        analysis_results["fit_xs"].push(fit_xs)
        analysis_results["fit_ys"].push(fit_ys)
        return [
            annotations.curve_1d(
                x_axis=self.x,
                x_values=fit_xs,
                y_axis=self.y,
                y_values=fit_ys,
            )
        ]


class ScanXFragment(ExpFragment):
    """Scan ``x`` for one fixed choice of ``p`` and expose the fitted slope ``m``."""

    def build_fragment(self):
        self.setattr_fragment("line", LineFragment, detached=True)

        self.setattr_result("m", FloatChannel, description="Extracted slope")

    def run_once(self):
        x_points = np.linspace(0.0, 5.0, 6).tolist()
        x_request = ScanRequest.cartesian(
            [(self.line.x, x_points)],
            site=make_child_scan_site(
                "scan_x",
                extra_metadata={"analysis_note": "default-analysis slope fit"},
            ),
        )
        x_result = run_host_scan(self, self.line, x_request)
        self.m.push(x_result.analysis_results["m"])

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.line.p],
                self._analyse_exponent,
                [
                    FloatChannel("fit_e", "Extracted exponent"),
                    OpaqueChannel("fit_ps", save_by_default=False),
                    OpaqueChannel("fit_ms", save_by_default=False),
                ],
            )
        ]

    def _analyse_exponent(self, axis_values, result_values, analysis_results):
        ps = np.asarray(axis_values[self.line.p], dtype=float)
        ms = np.asarray(result_values[self.m], dtype=float)

        fit_e = fit_power_law_exponent(ps, ms)
        fit_ps = np.linspace(ps.min(), ps.max(), 100)
        fit_ms = fit_ps**fit_e

        analysis_results["fit_e"].push(fit_e)
        analysis_results["fit_ps"].push(fit_ps)
        analysis_results["fit_ms"].push(fit_ms)
        return []


class HowDoesPVaryFragment(ExpFragment):
    """Top-level fragment that studies how the extracted slope varies with ``p``.

    The fragment itself is not scanned by the top-level runtime; instead it runs one
    nested scan over ``p`` inside a single root point. This mirrors the old
    ``SubscanExpFragment`` usage style while exercising the new host-runtime recursion
    path directly.
    """

    def build_fragment(self):
        self.setattr_fragment("scan_x", ScanXFragment, detached=True)

        self.setattr_result("fit_e", FloatChannel, description="Extracted exponent")

    def run_once(self):
        # Start at p = 1 to keep the log-space fit well-defined.
        p_points = np.linspace(1.0, 5.0, 10).tolist()
        p_request = ScanRequest.cartesian(
            [(self.scan_x.line.p, p_points)],
            site=make_child_scan_site(
                "scan_p",
                extra_metadata={"analysis_note": "default-analysis exponent fit"},
            ),
        )
        p_result = run_host_scan(self, self.scan_x, p_request)
        self.fit_e.push(p_result.analysis_results["fit_e"])


HostRuntimeHowDoesPVary = make_fragment_host_scan_exp(
    HowDoesPVaryFragment,
    # The top-level runtime executes one root point. That point then launches the
    # nested scans inside ``HowDoesPVaryFragment.run_once()``.
    lambda fragment: ScanRequest.single(
        metadata={"demo_name": "host_runtime_nested_p_variation"}
    ),
)
HostRuntimeHowDoesPVary.__doc__ = "Host-runtime nested p-variation scan"

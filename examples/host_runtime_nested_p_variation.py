"""Nested host-runtime scan example.

This is the host-runtime rewrite of the classic "subscan inside a subscan" shape.

Key differences from the legacy ``SubscanExpFragment`` approach:

- no ``SubscanExpFragment`` subclasses,
- no ``setattr_subscan(...)``,
- no implicit default-analysis pipeline,
- nested scans are launched explicitly with ``run_host_scan(...)``,
- child scan sites are derived explicitly with ``make_child_scan_site(...)``.

The runtime model is intentionally simple:

1. the top-level experiment executes exactly one root point,
2. that root point launches a scan over ``p``,
3. each ``p`` point launches a scan over ``x``,
4. each level performs its own post-run analysis synchronously in plain Python.

This keeps the recursive structure obvious and makes it easy to inspect how data flows
through the new runtime before a dedicated analysis pipeline is added.
"""

from __future__ import annotations

import numpy as np

from ndscan.experiment import (
    ExpFragment,
    FloatChannel,
    FloatParam,
    OpaqueChannel,
    ScanRequest,
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


class ScanXFragment(ExpFragment):
    """Scan ``x`` for one fixed choice of ``p`` and extract the slope ``m``.

    This fragment owns the first nested scan site. The scanned child fragment is
    detached so its result channels are not treated as ordinary parent outputs; instead
    they belong to the nested scan launched in ``run_once()``.
    """

    def build_fragment(self):
        self.setattr_fragment("line", LineFragment)
        self.detach_fragment(self.line)

        self.setattr_result("m", FloatChannel, description="Extracted slope")
        self.setattr_result(
            "fit_xs",
            OpaqueChannel,
            save_by_default=False,
        )
        self.setattr_result(
            "fit_ys",
            OpaqueChannel,
            save_by_default=False,
        )

    def run_once(self):
        x_points = np.linspace(0.0, 5.0, 6).tolist()
        x_request = ScanRequest.cartesian(
            [(self.line.x, x_points)],
            site=make_child_scan_site(
                "scan_x",
                extra_metadata={"analysis_note": "manual slope fit"},
            ),
        )
        x_result = run_host_scan(self, self.line, x_request)

        xs = np.asarray(next(iter(x_result.coordinates.values())), dtype=float)
        ys = np.asarray(x_result.values[self.line.y], dtype=float)

        m = fit_line_through_origin(xs, ys)
        fit_xs = np.linspace(xs.min(), xs.max(), 50)
        fit_ys = m * fit_xs

        self.fit_xs.push(fit_xs)
        self.fit_ys.push(fit_ys)
        self.m.push(m)


class HowDoesPVaryFragment(ExpFragment):
    """Top-level fragment that studies how the extracted slope varies with ``p``.

    The fragment itself is not scanned by the top-level runtime; instead it runs one
    nested scan over ``p`` inside a single root point. This mirrors the old
    ``SubscanExpFragment`` usage style while exercising the new host-runtime recursion
    path directly.
    """

    def build_fragment(self):
        self.setattr_fragment("scan_x", ScanXFragment)
        self.detach_fragment(self.scan_x)

        self.setattr_result("fit_e", FloatChannel, description="Extracted exponent")
        self.setattr_result(
            "fit_ps",
            OpaqueChannel,
            save_by_default=False,
        )
        self.setattr_result(
            "fit_ms",
            OpaqueChannel,
            save_by_default=False,
        )

    def run_once(self):
        # Start at p = 1 to keep the log-space fit well-defined.
        p_points = np.linspace(1.0, 5.0, 10).tolist()
        p_request = ScanRequest.cartesian(
            [(self.scan_x.line.p, p_points)],
            site=make_child_scan_site(
                "scan_p",
                extra_metadata={"analysis_note": "manual exponent fit"},
            ),
        )
        p_result = run_host_scan(self, self.scan_x, p_request)

        ps = np.asarray(next(iter(p_result.coordinates.values())), dtype=float)
        ms = np.asarray(p_result.values[self.scan_x.m], dtype=float)

        fit_e = fit_power_law_exponent(ps, ms)
        fit_ps = np.linspace(ps.min(), ps.max(), 100)
        fit_ms = fit_ps**fit_e

        self.fit_ps.push(fit_ps)
        self.fit_ms.push(fit_ms)
        self.fit_e.push(fit_e)


HostRuntimeHowDoesPVary = make_fragment_host_scan_exp(
    HowDoesPVaryFragment,
    # The top-level runtime executes one root point. That point then launches the
    # nested scans inside ``HowDoesPVaryFragment.run_once()``.
    lambda fragment: ScanRequest.single(
        metadata={"demo_name": "host_runtime_nested_p_variation"}
    ),
)

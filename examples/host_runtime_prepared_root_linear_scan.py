"""Direct root ``PreparedScan`` example.

This shows the top-level prepared-scan contract directly, without going through the
``make_fragment_prepared_scan_exp(...)`` adapter:

- build the fragment structure in ``build()``,
- create the root ``PreparedScan`` in ``prepare()``,
- execute it in ``run()``,
- consume declared fixed outputs via ``scan.get_outputs()``.

The experiment scans ``x`` on the host, records ``y = slope * x``, fits the slope as a
default analysis, and writes that fitted slope into the regular dataset DB under
``prepared_root_fit_slope``.
"""

from __future__ import annotations

from artiq.language import EnvExperiment

from artiq.experiment import *
from ndscan.define import *
from ndscan.define import annotations
from ndscan.scan import *
from ndscan.runtime.api import *


def _fit_line_through_origin(xs, ys) -> float:
    numerator = sum(x * y for x, y in zip(xs, ys, strict=True))
    denominator = sum(x * x for x in xs)
    return numerator / denominator


class PreparedRootLinearResponseFragment(ExpFragment):
    """Simple linear fragment with one fitted scalar output."""

    def build_fragment(self):
        self.setattr_param("slope", FloatParam, "Slope", default=2.0)
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push(self.slope.get() * self.x.get())

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.x],
                self._analyse_gradient,
                [FloatChannel("m", "Extracted slope")],
            )
        ]

    def _analyse_gradient(self, axis_values, result_values, analysis_results):
        xs = axis_values[self.x]
        ys = result_values[self.y]
        slope = _fit_line_through_origin(xs, ys)
        analysis_results["m"].push(slope)
        return [
            annotations.curve_1d(
                x_axis=self.x,
                x_values=xs,
                y_axis=self.y,
                y_values=[slope * x for x in xs],
            )
        ]


class HostRuntimePreparedRootLinearScan(EnvExperiment):
    """Runnable direct-root example using ``prepare_scan(...)`` explicitly."""

    def build(self):
        self.fragment = PreparedRootLinearResponseFragment(self, [])
        self.scan = None

    def prepare(self):
        request = ScanRequest.explicit(
            [self.fragment.x],
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            metadata={"demo_name": "host_runtime_prepared_root_linear_scan"},
        )
        self.scan = prepare_scan(
            self,
            self.fragment,
            request=request,
            expose_outputs=["m"],
        )

    def run(self):
        self.scan.execute()
        (fit_slope,) = self.scan.get_outputs()
        self.fit_slope = float(fit_slope)
        self.set_dataset("prepared_root_fit_slope", float(fit_slope), archive=True)

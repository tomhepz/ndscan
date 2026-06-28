"""Prepared-runtime example for typical offline ``ndscan.results`` workflows.

The paired offline script shows how to:

- open one saved run,
- list the saved series and their metadata,
- inspect the automatic scan metadata,
- load scalar and array channels,
- inspect a nested child scan,
- and compare saved analysis with offline analysis.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from artiq.experiment import *

from ndscan.define import *
from ndscan.define import annotations
from ndscan.runtime.api import *
from ndscan.scan import *

TRACE_LENGTH = 32
TRACE_SAMPLE_TIMES_US = np.linspace(0.0, 31.0, TRACE_LENGTH)
TRACE_TEMPLATE = np.exp(-0.5 * ((np.arange(TRACE_LENGTH) - 12.0) / 4.0) ** 2)

CARRIER_FREQUENCY_MHZ = 100.0
ROOT_LOGICAL_FREQUENCY_POINTS_MHZ = np.linspace(-1.5, 1.5, 11).tolist()
DEFAULT_REPEAT_COUNT = 16
ROOT_MAX_POINTS_PER_BATCH = 4
REPEAT_MAX_POINTS_PER_BATCH = 4


def build_demo_config_blob() -> dict[str, object]:
    """Return one small user blob that an offline script can read back later."""

    return {
        "namespace": "results_api_demo",
        "version": 1,
        "carrier_frequency_mhz": CARRIER_FREQUENCY_MHZ,
        "trace_length": TRACE_LENGTH,
        "trace_sample_times_us": TRACE_SAMPLE_TIMES_US.tolist(),
        "repeat_count": DEFAULT_REPEAT_COUNT,
        "root_points_mhz": list(ROOT_LOGICAL_FREQUENCY_POINTS_MHZ),
        "summary": "Logical frequency scan with repeated noisy scalar/array readout",
    }


def underlying_mean_signal(drive_frequency_mhz: float) -> float:
    """Return the noiseless scalar signal for one physical drive frequency."""

    logical_frequency = drive_frequency_mhz - CARRIER_FREQUENCY_MHZ
    return 0.85 + 0.35 * logical_frequency


def simulate_shot(
    drive_frequency_mhz: float, rng: np.random.Generator
) -> tuple[float, np.ndarray]:
    """Return one noisy scalar shot value and one noisy trace."""

    mean_signal = underlying_mean_signal(drive_frequency_mhz)
    shot_signal = float(mean_signal + rng.normal(scale=0.08))
    shot_trace = 0.15 + shot_signal * TRACE_TEMPLATE + rng.normal(
        scale=0.03, size=TRACE_LENGTH
    )
    return shot_signal, np.asarray(shot_trace, dtype=float)
 

def mean_and_standard_error(samples) -> tuple[np.ndarray, np.ndarray]:
    """Return the sample mean and standard error along axis 0."""

    values = np.asarray(samples, dtype=float)
    mean = np.mean(values, axis=0)
    if values.shape[0] <= 1:
        error = np.zeros_like(mean, dtype=float)
    else:
        error = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return np.asarray(mean, dtype=float), np.asarray(error, dtype=float)


def fit_weighted_line(
    logical_frequencies_mhz, values, value_errors
) -> tuple[float, float, float, float]:
    """Return weighted least-squares line-fit parameters and uncertainties."""

    xs = np.asarray(logical_frequencies_mhz, dtype=float)
    ys = np.asarray(values, dtype=float)
    yerrs = np.asarray(value_errors, dtype=float)

    if len(xs) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    safe_errors = np.where(np.isfinite(yerrs) & (yerrs > 0.0), yerrs, 1.0)
    weights = 1.0 / safe_errors**2
    design = np.column_stack([xs, np.ones_like(xs)])

    lhs = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * ys)
    covariance = np.linalg.inv(lhs)
    slope, intercept = covariance @ rhs
    slope_error, intercept_error = np.sqrt(np.diag(covariance))

    return (
        float(slope),
        float(intercept),
        float(slope_error),
        float(intercept_error),
    )


def build_line_fit_artifact(
    logical_frequencies_mhz, values, value_errors
) -> tuple[dict[str, object], float, float, float, float]:
    """Return one serializable line-fit artifact and its key parameters."""

    logical_xs = np.asarray(logical_frequencies_mhz, dtype=float)
    slope, intercept, slope_error, intercept_error = fit_weighted_line(
        logical_xs, values, value_errors
    )

    fit_logical_xs = np.linspace(
        float(np.min(logical_xs)),
        float(np.max(logical_xs)),
        100,
    )
    fit_drive_xs = fit_logical_xs + CARRIER_FREQUENCY_MHZ
    fit_ys = slope * fit_logical_xs + intercept

    artifact = {
        "kind": "line_fit",
        "model_id": "straight_line",
        "provider": "results_api_demo",
        "parameters": {
            "slope": {"value": slope, "stderr": slope_error},
            "intercept": {"value": intercept, "stderr": intercept_error},
        },
        "logical_frequency_values_mhz": fit_logical_xs.tolist(),
        "drive_frequency_values_mhz": fit_drive_xs.tolist(),
        "fitted_y_values": fit_ys.tolist(),
        "num_points": int(len(logical_xs)),
    }
    return artifact, slope, intercept, slope_error, intercept_error


class TraceShotFragment(ExpFragment):
    """Leaf fragment producing one noisy scalar readout and one noisy trace."""

    def build_fragment(self):
        # This is the hardware-facing parameter for one repeated shot.
        self.setattr_param(
            "drive_frequency",
            FloatParam,
            "Drive frequency",
            CARRIER_FREQUENCY_MHZ * MHz,
            unit="MHz",
        )
        # Save one scalar and one array result so the offline side sees both cases.
        self.setattr_result("shot_signal", FloatChannel, description="One noisy shot value")
        self.setattr_result(
            "shot_trace",
            ArrayChannel,
            description="One noisy trace",
            shape=(TRACE_LENGTH,),
            dim_names=("sample",),
        )
        self._rng = np.random.default_rng(seed=12345)

    def run_once(self):
        # Each shot depends only on the current drive frequency for this point.
        shot_signal, shot_trace = simulate_shot(self.drive_frequency.get() / MHz, self._rng)
        self.shot_signal.push(shot_signal)
        self.shot_trace.push(shot_trace)

    def get_default_analyses(self):
        # The child scan publishes repeat-averaged scalar and array results.
        mean_signal = FloatChannel("mean_signal", "Mean shot signal")
        mean_trace = ArrayChannel(
            "mean_trace",
            "Mean shot trace",
            shape=(TRACE_LENGTH,),
            dim_names=("sample",),
        )
        return [
            CustomAnalysis(
                [],
                self._analyse_repeat_statistics,
                analysis_results=[
                    mean_signal,
                    FloatChannel(
                        "mean_signal_error",
                        "Standard error of the mean shot signal",
                        display_hints={"error_bar_for": mean_signal.path},
                    ),
                    IntChannel("num_shots", "Number of repeated shots"),
                    mean_trace,
                    ArrayChannel(
                        "mean_trace_error",
                        "Standard error of the mean shot trace",
                        shape=(TRACE_LENGTH,),
                        dim_names=("sample",),
                        display_hints={"error_bar_for": mean_trace.path},
                    ),
                ],
                online_fn=self._analyse_repeat_statistics,
                online_analysis_identifier="repeat_stats",
            )
        ]

    def _analyse_repeat_statistics(self, axis_values, result_values, analysis_results):
        del axis_values, analysis_results

        shot_signals = np.asarray(result_values[self.shot_signal], dtype=float)
        shot_traces = np.asarray(result_values[self.shot_trace], dtype=float)

        mean_signal, mean_signal_error = mean_and_standard_error(shot_signals)
        mean_trace, mean_trace_error = mean_and_standard_error(shot_traces)

        return AnalysisFeedback(
            outputs={
                "mean_signal": float(mean_signal),
                "mean_signal_error": float(mean_signal_error),
                "num_shots": int(len(shot_signals)),
                "mean_trace": np.asarray(mean_trace, dtype=float),
                "mean_trace_error": np.asarray(mean_trace_error, dtype=float),
            },
            artifacts={
                "repeat_statistics_summary": {
                    "kind": "repeat_statistics",
                    "num_shots": int(len(shot_signals)),
                    "mean_signal": float(mean_signal),
                    "mean_signal_error": float(mean_signal_error),
                    "trace_peak": float(np.max(mean_trace)),
                }
            },
        )


class ResultsApiDemoFragment(ExpFragment):
    """Root fragment scanned over logical frequency with a repeated-shot child scan."""

    def build_fragment(self):
        # The dashboard or code-first request scans this logical outer parameter.
        self.setattr_param(
            "logical_frequency",
            FloatParam,
            "Logical frequency",
            0.0 * MHz,
            unit="MHz",
        )
        # The child scan repeats one logical point and exposes a few fixed outputs.
        self.repeat_scan = setattr_prepared_child_scan(
            self,
            "shot",
            TraceShotFragment,
            scan_name="repeat_scan",
            expose_outputs=["mean_signal", "mean_signal_error", "num_shots"],
        )

        # Mirror the child's saved summaries onto the root site for easy plotting.
        self.setattr_result("mean_signal", FloatChannel, description="Mean shot signal")
        self.setattr_result(
            "mean_signal_error",
            FloatChannel,
            description="Standard error of the mean shot signal",
            display_hints={"error_bar_for": self.mean_signal.path},
        )
        self.setattr_result("num_shots", IntChannel, description="Repeated-shot count")
        self.setattr_result(
            "mean_trace",
            ArrayChannel,
            description="Mean shot trace",
            shape=(TRACE_LENGTH,),
            dim_names=("sample",),
        )
        self.setattr_result(
            "mean_trace_error",
            ArrayChannel,
            description="Standard error of the mean shot trace",
            shape=(TRACE_LENGTH,),
            dim_names=("sample",),
            display_hints={"error_bar_for": self.mean_trace.path},
        )

    def run_once(self):
        # Re-run the child leaf a fixed number of times at this outer point.
        repeat_request = ScanRequest.single(
            metadata={
                "repeat_config": {
                    "repeats": DEFAULT_REPEAT_COUNT,
                    "max_points_per_batch": REPEAT_MAX_POINTS_PER_BATCH,
                }
            },
            execution_policy=ExecutionPolicy(
                max_points_per_batch=REPEAT_MAX_POINTS_PER_BATCH
            ),
        ).with_repeats(repeats=DEFAULT_REPEAT_COUNT)

        self.repeat_scan.configure(repeat_request)
        outputs = self.repeat_scan.execute()
        inspection = self.repeat_scan.inspect()

        self.mean_signal.push(float(outputs["mean_signal"]))
        self.mean_signal_error.push(float(outputs["mean_signal_error"]))
        self.num_shots.push(int(outputs["num_shots"]))
        self.mean_trace.push(
            np.asarray(inspection.analysis_results["mean_trace"], dtype=float)
        )
        self.mean_trace_error.push(
            np.asarray(inspection.analysis_results["mean_trace_error"], dtype=float)
        )

    def get_default_analyses(self):
        # Save one simple fit so the offline script can inspect saved analysis output.
        return [
            CustomAnalysis(
                [self.logical_frequency],
                self._analyse_root_signal_line,
                [
                    FloatChannel("fit_slope", "Weighted line-fit slope"),
                    FloatChannel("fit_intercept", "Weighted line-fit intercept"),
                ],
                online_fn=self._analyse_root_signal_line,
                online_analysis_identifier="running_signal_fit",
            )
        ]

    def _analyse_root_signal_line(self, axis_values, result_values, analysis_results):
        del analysis_results

        logical_frequencies_mhz = (
            np.asarray(axis_values[self.logical_frequency], dtype=float) / MHz
        )
        mean_signals = np.asarray(result_values[self.mean_signal], dtype=float)
        mean_signal_errors = np.asarray(result_values[self.mean_signal_error], dtype=float)

        fit_artifact, slope, intercept, _, _ = build_line_fit_artifact(
            logical_frequencies_mhz,
            mean_signals,
            mean_signal_errors,
        )
        feedback = AnalysisFeedback(
            outputs={
                "fit_slope": slope,
                "fit_intercept": intercept,
            },
            artifacts={"signal_line_fit": fit_artifact},
        )
        if np.isfinite(slope) and np.isfinite(intercept):
            # The fit above is done in MHz for human readability, but the live viewer
            # plots the raw scanned parameter values, which are stored in Hz. Convert
            # the gradient back into raw x units before drawing the computed curve.
            feedback.annotations.append(
                annotations.computed_curve(
                    function_name="line",
                    parameters={"a": intercept, "b": slope / MHz},
                    associated_channels=[self.mean_signal],
                )
            )
        return feedback


def build_results_api_demo_request(fragment: ResultsApiDemoFragment) -> ScanRequest:
    """Return the root scan request for the results-API demo."""

    # Keep one fixed pseudoparam so the offline side can inspect it in the schema.
    carrier_frequency = ScanVariable(
        "carrier_frequency",
        description="Fixed carrier frequency",
        spec={"unit": "MHz", "scale": 1.0},
    )

    # The root request scans the logical axis and maps it onto the child parameter.
    request = ScanRequest.cartesian(
        [(fragment.logical_frequency, [value * MHz for value in ROOT_LOGICAL_FREQUENCY_POINTS_MHZ])],
        metadata={
            "demo_name": "prepared_scan_results_api_demo",
            "demo_config": build_demo_config_blob(),
        },
        execution_policy=ExecutionPolicy(max_points_per_batch=ROOT_MAX_POINTS_PER_BATCH),
    ).with_parameter_mappings(
        [
            ParameterMapping.single_target(
                fragment.shot.drive_frequency,
                [fragment.logical_frequency],
                lambda values: values[fragment.logical_frequency]
                + CARRIER_FREQUENCY_MHZ * MHz,
                description=(
                    "Map logical_frequency onto shot.drive_frequency using a fixed carrier"
                ),
            )
        ]
    )
    return replace(
        request,
        fixed_pseudoparams=(FixedPseudoparam(carrier_frequency, CARRIER_FREQUENCY_MHZ),),
    )


PreparedScanResultsApiDemo = make_fragment_prepared_scan_exp(
    ResultsApiDemoFragment,
    build_results_api_demo_request,
)

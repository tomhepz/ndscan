"""Host-runtime probability/frequency example using fragment-attached online analysis.

This is the "nicer" counterpart to ``host_runtime_probability_frequency.py``.

The key difference is where the repeated-shot reduction lives:

- the leaf fragment declares a `CustomAnalysis` that turns yes/no shots into
  probability, probability error, and shot count,
- the same analysis function is reused for both final and online execution,
- the nested repeat scan stops from the online analysis outputs rather than from a
  second hand-maintained accumulator in the parent fragment.

This keeps the runtime feedback path explicit while making the repeated-shot
interpretation reusable and local to the fragment that actually produces the shots.
"""

from __future__ import annotations

import math

import numpy as np

from ndscan.experiment import (
    CustomAnalysis,
    ExecutionPolicy,
    ExpFragment,
    FloatChannel,
    FloatParam,
    IntChannel,
    OpaqueChannel,
    RepeatPointSource,
    ScanRequest,
    SinglePointSource,
    annotations,
    make_fragment_host_scan_exp,
    run_subscan,
)


TRUE_FREQUENCY = 0.2


def underlying_probability(t: float) -> float:
    """Return the underlying yes-probability for one time point."""

    return 0.5 + 0.35 * math.sin(2.0 * math.pi * TRUE_FREQUENCY * t + 0.3)


def estimate_probability_from_shots(shots: list[float]) -> tuple[float, float]:
    """Return the observed probability and a simple Poisson-style shot-noise error."""

    num_shots = len(shots)
    if num_shots == 0:
        return 0.0, float("inf")

    num_successes = float(sum(shots))
    probability = num_successes / num_shots
    success_error = math.sqrt(num_successes) if num_successes > 0.0 else 1.0
    probability_error = success_error / num_shots
    return probability, probability_error


def fit_sine_frequency(ts, probabilities) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a sine-wave frequency by grid-searching a linearised sinusoid model."""

    ts = np.asarray(ts, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    candidate_frequencies = np.linspace(0.05, 0.5, 600)
    best_frequency = float(candidate_frequencies[0])
    best_coefficients = np.zeros(3)
    best_residual = float("inf")

    for frequency in candidate_frequencies:
        omega_t = 2.0 * math.pi * frequency * ts
        design = np.column_stack(
            [
                np.ones_like(ts),
                np.sin(omega_t),
                np.cos(omega_t),
            ]
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, probabilities, rcond=None)
        residual = np.linalg.norm(design @ coefficients - probabilities)
        if residual < best_residual:
            best_residual = float(residual)
            best_frequency = float(frequency)
            best_coefficients = coefficients

    fit_ts = np.linspace(float(ts.min()), float(ts.max()), 200)
    fit_omega_t = 2.0 * math.pi * best_frequency * fit_ts
    fit_probabilities = (
        best_coefficients[0]
        + best_coefficients[1] * np.sin(fit_omega_t)
        + best_coefficients[2] * np.cos(fit_omega_t)
    )
    return best_frequency, fit_ts, fit_probabilities


def make_online_precision_stopper(*, error_threshold: float, min_shots: int):
    """Return a stop predicate driven by the leaf fragment's online analysis."""

    def stop(feedback) -> bool:
        stats = feedback.online_analyses["repeat_stats"].outputs
        if stats.get("num_shots", 0) < min_shots:
            return False
        return stats["probability_error"] <= error_threshold

    return stop


class YesNoAtTimeWithAnalysisFragment(ExpFragment):
    """Leaf fragment that emits Bernoulli shots and analyses them over repeats."""

    def build_fragment(self):
        self.setattr_param("t", FloatParam, "t", default=0.0)
        self.setattr_result("hit", FloatChannel)
        self._rng = np.random.default_rng(seed=12345)

    def run_once(self):
        probability = underlying_probability(self.t.get())
        self.hit.push(1.0 if self._rng.random() < probability else 0.0)

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [],
                self._analyse_repeat_statistics,
                analysis_results=[
                    FloatChannel("probability", "Observed success probability"),
                    FloatChannel("probability_error", "Shot-noise probability error"),
                    IntChannel("num_shots", "Number of repeated shots"),
                ],
                online_fn=self._analyse_repeat_statistics,
                online_analysis_identifier="repeat_stats",
            )
        ]

    def _analyse_repeat_statistics(self, axis_values, result_values, analysis_results):
        del axis_values
        shots = list(result_values[self.hit])
        probability, probability_error = estimate_probability_from_shots(shots)
        analysis_results["probability"].push(probability)
        analysis_results["probability_error"].push(probability_error)
        analysis_results["num_shots"].push(len(shots))
        return []


class ProbabilityAtTimeViaAnalysisFragment(ExpFragment):
    """Estimate probability(t) by delegating repeat statistics to the leaf analysis."""

    def build_fragment(self):
        self.setattr_fragment("detector", YesNoAtTimeWithAnalysisFragment, detached=True)
        self.setattr_result("probability", FloatChannel)
        self.setattr_result("probability_error", FloatChannel)
        self.setattr_result("num_shots", IntChannel)

    def run_once(self):
        # A dummy ScanVariable would still be a valid choice if "repeat index" carried
        # scientific meaning for a particular experiment. Here it does not, so repeated
        # shots stay as execution policy rather than becoming a pseudoparam.
        repeat_request = ScanRequest(
            axes=(),
            point_source=RepeatPointSource(
                SinglePointSource(),
                stop_predicate=make_online_precision_stopper(
                    error_threshold=0.05,
                    min_shots=24,
                ),
                min_repeats=24,
                max_repeats=600,
                predicate_description="repeat_stats.probability_error <= 0.035",
            ),
            execution_policy=ExecutionPolicy(max_points_per_batch=16),
        )
        repeat_result = run_subscan(
            self,
            self.detector,
            repeat_request,
            name="repeat_scan",
            extra_metadata={
                "analysis_note": "leaf CustomAnalysis publishes repeat statistics online"
            },
        )
        self.probability.push(repeat_result.analysis_results["probability"])
        self.probability_error.push(repeat_result.analysis_results["probability_error"])
        self.num_shots.push(repeat_result.analysis_results["num_shots"])

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.detector.t],
                self._analyse_frequency,
                [
                    FloatChannel("fit_frequency", "Extracted sine frequency"),
                    OpaqueChannel("fit_ts", save_by_default=False),
                    OpaqueChannel("fit_probabilities", save_by_default=False),
                ],
            )
        ]

    def _analyse_frequency(self, axis_values, result_values, analysis_results):
        ts = np.asarray(axis_values[self.detector.t], dtype=float)
        probabilities = np.asarray(result_values[self.probability], dtype=float)

        fit_frequency, fit_ts, fit_probabilities = fit_sine_frequency(ts, probabilities)
        analysis_results["fit_frequency"].push(fit_frequency)
        analysis_results["fit_ts"].push(fit_ts)
        analysis_results["fit_probabilities"].push(fit_probabilities)
        return [
            annotations.curve_1d(
                x_axis=self.detector.t,
                x_values=fit_ts,
                y_axis=self.probability,
                y_values=fit_probabilities,
            )
        ]


class FrequencyFromProbabilityViaAnalysisFragment(ExpFragment):
    """Top-level fragment that scans t and reads the fitted frequency."""

    def build_fragment(self):
        self.setattr_fragment(
            "probability_scan", ProbabilityAtTimeViaAnalysisFragment, detached=True
        )
        self.setattr_result("fit_frequency", FloatChannel)

    def run_once(self):
        t_points = np.linspace(0.0, 12.0, 25).tolist()
        probability_request = ScanRequest.cartesian(
            [(self.probability_scan.detector.t, t_points)],
            execution_policy=ExecutionPolicy(max_points_per_batch=5),
        )
        probability_result = run_subscan(
            self,
            self.probability_scan,
            probability_request,
            name="probability_scan",
            extra_metadata={
                "analysis_note": "outer sine fit over probability(t), with repeat statistics on the leaf fragment"
            },
        )
        self.fit_frequency.push(probability_result.analysis_results["fit_frequency"])


HostRuntimeProbabilityFrequencyViaAnalysis = make_fragment_host_scan_exp(
    FrequencyFromProbabilityViaAnalysisFragment,
    lambda fragment: ScanRequest.single(
        metadata={"demo_name": "host_runtime_probability_frequency_via_analysis"}
    ),
)
HostRuntimeProbabilityFrequencyViaAnalysis.__doc__ = (
    "Host-runtime probability/frequency example using leaf-attached online analysis"
)

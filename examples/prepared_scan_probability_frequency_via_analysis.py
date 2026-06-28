"""Prepared-runtime probability/frequency example using fragment-attached online analysis.

This is the "nicer" counterpart to ``prepared_scan_probability_frequency.py``.

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

from artiq.experiment import *
from ndscan.define import *
from ndscan.define import annotations
from ndscan.fits import (
    artifact_parameter_value,
    build_builtin_model,
    fit_data_with_model,
    import_sensible_fitting,
)
from ndscan.scan import *
from ndscan.runtime.api import *


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


def fit_sine_frequency_artifact(ts, probabilities, probability_errors) -> dict[str, object]:
    """Fit a sinusoid with sensible-fitting and return the serializable artifact."""

    sensible = import_sensible_fitting()
    ts = np.asarray(ts, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    probability_errors = np.asarray(probability_errors, dtype=float)

    model = build_builtin_model("sinusoid").bound(
        amplitude=(0.0, 1.0),
        offset=(0.0, 1.0),
        frequency=(0.01, 1.0),
        phase=(-math.pi, math.pi),
    )
    data = sensible.FitData.normal(
        x=ts,
        y=probabilities,
        yerr=probability_errors,
        x_label="t",
        y_label="probability",
        label="probability_trace",
    )
    _, artifact = fit_data_with_model(
        model,
        data,
        model_id="sinusoid",
        source={
            "x_key": "axis_0",
            "y_key": "channel_probability",
        },
    )
    return artifact.to_dict()


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

    def _analyse_repeat_statistics(self, axis_values, result_values):
        del axis_values
        shots = list(result_values[self.hit])
        probability, probability_error = estimate_probability_from_shots(shots)
        return AnalysisFeedback(
            outputs={
                "probability": probability,
                "probability_error": probability_error,
                "num_shots": len(shots),
            }
        )


class ProbabilityAtTimeViaAnalysisFragment(ExpFragment):
    """Estimate probability(t) by delegating repeat statistics to the leaf analysis."""

    def build_fragment(self):
        self.setattr_fragment("detector", YesNoAtTimeWithAnalysisFragment, detached=True)
        self.repeat_scan = prepare_child_scan(
            self,
            self.detector,
            name="repeat_scan",
            extra_metadata={
                "analysis_note": "leaf CustomAnalysis publishes repeat statistics online"
            },
        )
        self.setattr_result("probability", FloatChannel)
        self.setattr_result("probability_error", FloatChannel)
        self.setattr_result("num_shots", IntChannel)

    def run_once(self):
        # A dummy ScanVariable would still be a valid choice if "repeat index" carried
        # scientific meaning for a particular experiment. Here it does not, so repeated
        # shots stay as execution policy rather than becoming a pseudoparam.
        repeat_request = ScanRequest(
            axes=(),
            point_policy=RepeatPointPolicy(
                SinglePointPolicy(),
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
        self.repeat_scan.configure(repeat_request)
        repeat_outputs = self.repeat_scan.execute()
        self.probability.push(repeat_outputs["probability"])
        self.probability_error.push(repeat_outputs["probability_error"])
        self.num_shots.push(repeat_outputs["num_shots"])

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.detector.t],
                self._analyse_frequency,
                [
                    FloatChannel("fit_frequency", "Extracted sine frequency"),
                ],
            )
        ]

    def _analyse_frequency(self, axis_values, result_values):
        ts = np.asarray(axis_values[self.detector.t], dtype=float)
        probabilities = np.asarray(result_values[self.probability], dtype=float)
        probability_errors = np.asarray(
            result_values[self.probability_error], dtype=float
        )

        fit_artifact = fit_sine_frequency_artifact(
            ts, probabilities, probability_errors
        )
        fit_frequency = float(artifact_parameter_value(fit_artifact, "frequency"))
        return AnalysisFeedback(
            outputs={"fit_frequency": fit_frequency},
            artifacts={"frequency_fit": fit_artifact},
            annotations=[
                annotations.artifact_curve(
                    artifact="frequency_fit",
                    x_axis=self.detector.t,
                    y_axis=self.probability,
                )
            ],
        )


class FrequencyFromProbabilityViaAnalysisFragment(ExpFragment):
    """Top-level fragment that scans t and reads the fitted frequency."""

    def build_fragment(self):
        self.setattr_fragment(
            "probability_scan", ProbabilityAtTimeViaAnalysisFragment, detached=True
        )
        self.probability_trace = prepare_child_scan(
            self,
            self.probability_scan,
            name="probability_scan",
            extra_metadata={
                "analysis_note": "outer sine fit over probability(t), with repeat statistics on the leaf fragment"
            },
        )
        self.setattr_result("fit_frequency", FloatChannel)

    def run_once(self):
        t_points = np.linspace(0.0, 12.0, 25).tolist()
        probability_request = ScanRequest.cartesian(
            [(self.probability_scan.detector.t, t_points)],
            execution_policy=ExecutionPolicy(max_points_per_batch=5),
        )
        self.probability_trace.configure(probability_request)
        self.fit_frequency.push(self.probability_trace.execute()["fit_frequency"])


PreparedScanProbabilityFrequencyViaAnalysis = make_fragment_prepared_scan_exp(
    FrequencyFromProbabilityViaAnalysisFragment,
    lambda fragment: ScanRequest.single(
        metadata={"demo_name": "prepared_scan_probability_frequency_via_analysis"}
    ),
)

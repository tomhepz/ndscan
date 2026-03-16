"""Host-runtime repeat-until-probability-precision example.

This example combines several of the new host-runtime ideas in one place:

- a leaf fragment that returns stochastic yes/no outcomes,
- a nested repeated-shot policy to gather shot statistics without a native repeat axis,
- early exit once the estimated probability error is small enough,
- an outer scan over time ``t``,
- a default analysis on that outer scan that fits a sine-wave frequency.

The structure mirrors a common lab workflow:

1. for one experimental setting ``t``, take repeated shots until the measured
   probability is precise enough,
2. treat that estimated probability as one point in a higher-level scan,
3. fit a model to the higher-level curve.
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
    RepeatPointPolicy,
    ScanRequest,
    SinglePointPolicy,
    annotations,
    make_fragment_host_scan_exp,
    run_subscan,
)


TRUE_FREQUENCY = 0.2


def underlying_probability(t: float) -> float:
    """Return the underlying yes-probability for one time point."""

    return 0.5 + 0.35 * math.sin(2.0 * math.pi * TRUE_FREQUENCY * t + 0.3)


def estimate_probability_from_shots(shots: list[float]) -> tuple[float, float]:
    """Return the observed probability and a simple Poisson-style shot-noise error.

    The success probability is estimated from the success count divided by the number
    of shots. The uncertainty is taken from Poisson counting noise on the success
    count: ``sqrt(k) / n``. For ``k = 0``, use ``1 / n`` as a conservative floor.
    """

    num_shots = len(shots)
    if num_shots == 0:
        return 0.0, float("inf")

    num_successes = float(sum(shots))
    probability = num_successes / num_shots
    success_error = math.sqrt(num_successes) if num_successes > 0.0 else 1.0
    probability_error = success_error / num_shots
    return probability, probability_error


def make_probability_precision_stopper(
    hit_channel,
    *,
    error_threshold: float,
    min_shots: int,
):
    """Return a batch predicate for ``RepeatPointPolicy``.

    The returned closure looks at the full accumulated result series for the nested
    repeat scan and stops once the estimated probability error is below the requested
    threshold. This demonstrates the intended adaptive-runtime pattern: batch feedback
    already carries the scan state the next-point policy needs, so user code does not
    need to maintain a second shadow accumulator.
    """

    def stop(feedback) -> bool:
        all_shots = list(feedback.result_data[hit_channel])

        if len(all_shots) < min_shots:
            return False

        _, probability_error = estimate_probability_from_shots(all_shots)
        return probability_error <= error_threshold

    return stop


def fit_sine_frequency(ts, probabilities) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a sine-wave frequency by grid-searching the linearised sinusoid model.

    For each candidate frequency ``f``, solve the linear least-squares problem

    ``p(t) = c0 + c1 * sin(2 pi f t) + c2 * cos(2 pi f t)``.

    The best frequency is the one with the smallest residual norm.
    """

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


class YesNoAtTimeFragment(ExpFragment):
    """Leaf fragment that returns one stochastic yes/no outcome."""

    def build_fragment(self):
        self.setattr_param("t", FloatParam, "t", default=0.0)
        self.setattr_result("hit", FloatChannel)
        self._rng = np.random.default_rng(seed=12345)

    def run_once(self):
        probability = underlying_probability(self.t.get())
        self.hit.push(1.0 if self._rng.random() < probability else 0.0)


class ProbabilityAtTimeFragment(ExpFragment):
    """Estimate the yes-probability for one fixed time point ``t``.

    The fragment launches a nested repeated-shot scan with no scientific repeat axis of
    its own. The actual independent variable for the outer curve is ``t``.
    """

    def build_fragment(self):
        self.setattr_fragment("detector", YesNoAtTimeFragment, detached=True)
        self.setattr_result("probability", FloatChannel)
        self.setattr_result("probability_error", FloatChannel)
        self.setattr_result("num_shots", IntChannel)

    def run_once(self):
        stop_when_precise = make_probability_precision_stopper(
            self.detector.hit,
            error_threshold=0.035,
            min_shots=24,
        )
        # If a repeat count were itself scientifically meaningful, we could model it as
        # an explicit dummy ScanVariable and it would then appear in the datasets as a
        # pseudoparam. Here repetition is just execution policy, so we keep it out of
        # the scan coordinates entirely.
        repeat_request = ScanRequest(
            axes=(),
            point_policy=RepeatPointPolicy(
                SinglePointPolicy(),
                stop_predicate=stop_when_precise,
                min_repeats=24,
                max_repeats=256,
                predicate_description="probability error <= 0.035",
            ),
            execution_policy=ExecutionPolicy(max_points_per_batch=16),
        )
        repeat_result = run_subscan(
            self,
            self.detector,
            repeat_request,
            name="repeat_scan",
            extra_metadata={
                "analysis_note": "stop once the shot-noise error is small enough"
            },
        )

        probability, probability_error = estimate_probability_from_shots(
            repeat_result.values[self.detector.hit]
        )
        self.probability.push(probability)
        self.probability_error.push(probability_error)
        self.num_shots.push(len(repeat_result.values[self.detector.hit]))

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


class FrequencyFromProbabilityFragment(ExpFragment):
    """Top-level fragment that scans ``t`` and extracts the fitted frequency."""

    def build_fragment(self):
        self.setattr_fragment("probability_scan", ProbabilityAtTimeFragment, detached=True)
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
                "analysis_note": "fit a sine frequency to probability(t)"
            },
        )
        self.fit_frequency.push(probability_result.analysis_results["fit_frequency"])


HostRuntimeProbabilityFrequency = make_fragment_host_scan_exp(
    FrequencyFromProbabilityFragment,
    lambda fragment: ScanRequest.single(
        metadata={"demo_name": "host_runtime_probability_frequency"}
    ),
)
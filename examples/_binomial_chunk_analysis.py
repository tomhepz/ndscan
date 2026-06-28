"""Reusable Bernoulli/binomial repeat helpers for prepared-runtime examples.

These helpers are intentionally example-scoped rather than ndscan core API. They are
small enough to keep the examples readable, but common enough that repeating the same
count/error bookkeeping in every demo was starting to obscure the interesting parts of
the scan structure.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from ndscan.define.default_analysis import AnalysisFeedback, CustomAnalysis
from ndscan.define.result_channels import FloatChannel, IntChannel


def estimate_probability_from_counts(
    num_successes: float, num_shots: int
) -> tuple[float, float]:
    """Return a conservative binomial probability estimate and standard error."""

    if num_shots == 0:
        return 0.0, float("inf")
    probability = float(num_successes) / float(num_shots)
    error = math.sqrt(max(probability * (1.0 - probability), 0.0) / num_shots)
    error = max(error, 0.5 / num_shots)
    return probability, error


def build_binomial_chunk_analysis(
    channel,
    *,
    prefix: str,
    probability_label: str,
    probability_error_label: str,
    shots_label: str,
    successes_label: str,
    online_analysis_identifier: str | None = None,
) -> CustomAnalysis:
    """Build a small analysis that reduces repeated Bernoulli shots into chunk stats."""

    probability_name = f"{prefix}_probability"
    probability_error_name = f"{prefix}_probability_error"

    def analyse(axis_values, result_values, analysis_results):
        del axis_values, analysis_results
        shots = list(result_values[channel])
        num_shots = len(shots)
        num_successes = int(sum(shots))
        probability, probability_error = estimate_probability_from_counts(
            float(num_successes), num_shots
        )
        return AnalysisFeedback(
            outputs={
                probability_name: probability,
                probability_error_name: probability_error,
                "num_shots": num_shots,
                "num_successes": num_successes,
            }
        )

    kwargs = {}
    if online_analysis_identifier is not None:
        kwargs["online_fn"] = analyse
        kwargs["online_analysis_identifier"] = online_analysis_identifier

    return CustomAnalysis(
        [],
        analyse,
        analysis_results=[
            FloatChannel(probability_name, probability_label),
            FloatChannel(probability_error_name, probability_error_label),
            IntChannel("num_shots", shots_label),
            IntChannel("num_successes", successes_label),
        ],
        **kwargs,
    )


def make_binomial_repeat_stop_predicate(
    channel,
    *,
    error_threshold: float,
    min_shots: int,
    max_shots: int,
):
    """Stop repeated Bernoulli shots once one logical point is precise enough."""

    def stop(feedback) -> bool:
        shots = feedback.result_data[channel]
        num_shots = len(shots)
        if num_shots < min_shots:
            return False
        _, probability_error = estimate_probability_from_counts(float(sum(shots)), num_shots)
        return probability_error <= error_threshold or num_shots >= max_shots

    return stop


def make_binomial_chunk_stop_predicate(
    num_successes_channel,
    num_shots_channel,
    *,
    error_threshold: float,
    min_total_shots: int,
    max_total_shots: int,
):
    """Stop repeated chunk observations once combined chunk statistics are precise enough."""

    def stop(feedback) -> bool:
        num_successes = sum(int(value) for value in feedback.result_data[num_successes_channel])
        num_shots = sum(int(value) for value in feedback.result_data[num_shots_channel])
        if num_shots < min_total_shots:
            return False
        _, probability_error = estimate_probability_from_counts(float(num_successes), num_shots)
        return probability_error <= error_threshold or num_shots >= max_total_shots

    return stop


def aggregate_binomial_chunk_statistics(
    x_values: Sequence[float],
    num_successes: Sequence[float | int],
    num_shots: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse repeated chunk-level statistics into one estimate per unique x value."""

    successes_by_x = defaultdict(float)
    shots_by_x = defaultdict(int)
    for x_value, successes_chunk, shots_chunk in zip(
        x_values, num_successes, num_shots, strict=True
    ):
        key = float(x_value)
        successes_by_x[key] += float(successes_chunk)
        shots_by_x[key] += int(shots_chunk)

    unique_x = np.asarray(sorted(shots_by_x), dtype=float)
    probabilities = np.empty(len(unique_x), dtype=float)
    probability_errors = np.empty(len(unique_x), dtype=float)
    total_shots = np.empty(len(unique_x), dtype=int)
    for index, x_value in enumerate(unique_x):
        total = shots_by_x[float(x_value)]
        probability, probability_error = estimate_probability_from_counts(
            successes_by_x[float(x_value)],
            total,
        )
        probabilities[index] = probability
        probability_errors[index] = probability_error
        total_shots[index] = total
    return unique_x, probabilities, probability_errors, total_shots

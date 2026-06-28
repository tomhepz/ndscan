"""Interleaved spectroscopy patterns for the prepared runtime.

This module collects the main ways to make one spectroscopy feature "improve
everywhere at once" instead of finishing one probe-frequency point before moving to
the next.

The three concrete examples exported here are:

- ``PreparedScanInterleavedSpectroscopy``:
  one flat scan over repeated raw Bernoulli shots. This is the lightest-weight form,
  but the site only stores the raw ``survived`` points.
- ``PreparedScanInterleavedSpectroscopySubscan``:
  the same repeated-frequency pattern, but living behind a prepared child scan. This
  keeps the parent/child ownership model intact while still revisiting the same child
  probe frequencies round-robin.
- ``PreparedScanChunkedInterleavedSpectroscopy``:
  the preferred lab-facing pattern. Each outer point runs a *small fixed repeat
  chunk* in a prepared child scan, publishes ``p``/``p_err``/counts as ordinary outer
  point channels, and then revisits the same outer frequency later. This gives the
  viewer sensible y-axis streams immediately and leaves room for later aggregation of
  repeated outer points.

If you need more freedom than nested scans make comfortable, the escape hatch is still
to flatten the problem into one explicit point list over all relevant coordinates and
then use global randomisation. The new ``RepeatPointPolicy(schedule="interleaved")``
added to ndscan covers the common nested cases without forcing that flatter style.

Request-builder layout
----------------------
This file now shows the same interleaved-repeat patterns in two styles:

- the "nice" style used by the exported demo experiments:
  build the base ``ScanRequest`` first, then apply ``.with_repeats(...)``
- the "terse" style:
  build the same request directly from ``RepeatPointPolicy(...)``

The higher-level form is what normal example readers should copy. The lower-level form
is still kept nearby as a reference for people who want to see the underlying point-
policy construction explicitly.
"""

from __future__ import annotations

import math
import time

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
from ndscan.runtime.api import *
from ndscan.scan import ExplicitPointPolicy, RepeatPointPolicy, ScanRequest, SinglePointPolicy
from ndscan.scan.request import ExecutionPolicy

from examples._binomial_chunk_analysis import (
    aggregate_binomial_chunk_statistics,
    build_binomial_chunk_analysis,
    estimate_probability_from_counts,
    make_binomial_chunk_stop_predicate,
    make_binomial_repeat_stop_predicate,
)

__all__ = [
    "POINT_DELAY_S",
    "DEMO_FREQUENCY_POINTS",
    "TARGET_PROBABILITY_ERROR",
    "CHUNKED_TARGET_PROBABILITY_ERROR",
    "MIN_SHOTS_PER_FREQUENCY",
    "MAX_SHOTS_PER_FREQUENCY",
    "CHUNK_SHOTS_PER_POINT",
    "MIN_TOTAL_SHOTS_PER_FREQUENCY",
    "MAX_TOTAL_SHOTS_PER_FREQUENCY",
    "TRUE_LINE_CENTER",
    "TRUE_DIP_SIGMA",
    "TRUE_SURVIVAL_BASELINE",
    "TRUE_DIP_DEPTH",
    "survival_probability",
    "estimate_probability_from_counts",
    "aggregate_frequency_statistics",
    "aggregate_chunked_frequency_statistics",
    "fit_gaussian_dip_artifact",
    "InterleavedSpectroscopyFragment",
    "ChunkedInterleavedSpectroscopyFragment",
    "InterleavedSpectroscopySubscanRootFragment",
    "BernoulliSpectroscopyLeafFragment",
    "build_interleaved_demo_request",
    "build_interleaved_demo_request_terse",
    "build_chunked_demo_request",
    "build_chunked_demo_request_terse",
    "PreparedScanInterleavedSpectroscopy",
    "PreparedScanInterleavedSpectroscopySubscan",
    "PreparedScanChunkedInterleavedSpectroscopy",
]


TRUE_LINE_CENTER = 0.22
TRUE_DIP_SIGMA = 0.17
TRUE_SURVIVAL_BASELINE = 0.92
TRUE_DIP_DEPTH = 0.68

POINT_DELAY_S = 0.01
DEMO_FREQUENCY_POINTS = np.linspace(-0.45, 0.85, 13).tolist()
TARGET_PROBABILITY_ERROR = 0.02
CHUNKED_TARGET_PROBABILITY_ERROR = 0.01
MIN_SHOTS_PER_FREQUENCY = 12
MAX_SHOTS_PER_FREQUENCY = 100
CHUNK_SHOTS_PER_POINT = 55
MIN_TOTAL_SHOTS_PER_FREQUENCY = 16
MAX_TOTAL_SHOTS_PER_FREQUENCY = 400

_FREQUENCY_STATISTICS_ARTIFACT = "frequency_statistics"
_ONLINE_ANALYSIS_IDENTIFIER = "aggregated_spectroscopy"


def survival_probability(probe_frequency: float) -> float:
    """Return the Bernoulli survival probability for one probe frequency."""

    detuning = (probe_frequency - TRUE_LINE_CENTER) / TRUE_DIP_SIGMA
    dip = TRUE_DIP_DEPTH * math.exp(-0.5 * detuning**2)
    return float(np.clip(TRUE_SURVIVAL_BASELINE - dip, 1e-4, 1.0 - 1e-4))


def aggregate_frequency_statistics(
    probe_frequencies, bernoulli_shots
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse repeated Bernoulli shots into one estimate per probe frequency."""

    ones = np.ones(len(bernoulli_shots), dtype=int)
    return aggregate_binomial_chunk_statistics(
        probe_frequencies,
        bernoulli_shots,
        ones,
    )


def aggregate_chunked_frequency_statistics(
    probe_frequencies, num_successes, num_shots
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse repeated chunk-level observations into one estimate per frequency."""

    return aggregate_binomial_chunk_statistics(
        probe_frequencies,
        num_successes,
        num_shots,
    )


def frequency_statistics_artifact(
    probe_frequencies, probabilities, probability_errors, total_shots
) -> dict[str, object]:
    """Return a small serialisable artifact describing the aggregated line state."""

    return {
        "kind": "frequency_statistics",
        "frequencies": [float(value) for value in probe_frequencies],
        "probabilities": [float(value) for value in probabilities],
        "probability_errors": [float(value) for value in probability_errors],
        "total_shots": [int(value) for value in total_shots],
    }


def fit_gaussian_dip_artifact(
    probe_frequencies, survival_probabilities, survival_probability_errors
) -> dict[str, object]:
    """Fit a Gaussian dip and return the canonical serialisable artifact."""

    sensible = import_sensible_fitting()
    probe_frequencies = np.asarray(probe_frequencies, dtype=float)
    survival_probabilities = np.asarray(survival_probabilities, dtype=float)
    survival_probability_errors = np.asarray(
        survival_probability_errors,
        dtype=float,
    )

    x_min = float(np.min(probe_frequencies))
    x_max = float(np.max(probe_frequencies))
    span = max(x_max - x_min, 1e-6)

    model = build_builtin_model("gaussian_dip").bound(
        x0=(x_min, x_max),
        y0=(0.0, 1.0),
        a=(0.0, 1.0),
        sigma=(0.03, span),
    )
    data = sensible.FitData.normal(
        x=probe_frequencies,
        y=survival_probabilities,
        yerr=survival_probability_errors,
        x_label="probe_frequency",
        y_label="survival_probability",
        label="interleaved_spectral_dip",
    )
    _, artifact = fit_data_with_model(
        model,
        data,
        model_id="gaussian_dip",
        source={
            "x_key": "axis_0",
            "y_key": "channel_survived",
        },
    )
    return artifact.to_dict()


def _interleaved_frequency_points() -> list[tuple[float]]:
    """Return the base logical frequency points used by the interleaved demos."""

    return [(value,) for value in DEMO_FREQUENCY_POINTS]


def _build_repeat_chunk_request_terse() -> ScanRequest:
    """Return the fixed-size repeat-chunk request in the low-level point-policy style."""

    return ScanRequest(
        axes=(),
        point_policy=RepeatPointPolicy(
            SinglePointPolicy(),
            repeats=CHUNK_SHOTS_PER_POINT,
        ),
        execution_policy=ExecutionPolicy(max_points_per_batch=CHUNK_SHOTS_PER_POINT),
    )


def _build_repeat_chunk_request() -> ScanRequest:
    """Return the fixed-size repeat-chunk request in the recommended request style."""

    return ScanRequest.single(
        execution_policy=ExecutionPolicy(max_points_per_batch=CHUNK_SHOTS_PER_POINT),
    ).with_repeats(repeats=CHUNK_SHOTS_PER_POINT)


class InterleavedSpectroscopyFragment(ExpFragment):
    """One flat scan over repeated raw Bernoulli probe-frequency shots."""

    def build_fragment(self):
        self.setattr_param("probe_frequency", FloatParam, "probe frequency", default=0.0)
        self.setattr_result("survived", FloatChannel)
        self._rng = np.random.default_rng(seed=20260401)

    def run_once(self):
        time.sleep(POINT_DELAY_S)
        p_survive = survival_probability(self.probe_frequency.get())
        self.survived.push(1.0 if self._rng.random() < p_survive else 0.0)

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.probe_frequency],
                self._analyse_interleaved_spectroscopy,
                [
                    FloatChannel("fit_center", "Fitted line centre"),
                    FloatChannel("fit_center_error", "One-sigma error on fitted centre"),
                    FloatChannel("fit_depth", "Fitted dip depth"),
                    FloatChannel("fit_fwhm", "Fitted line width"),
                    IntChannel(
                        "num_unique_frequencies",
                        "Number of unique frequency points seen so far",
                    ),
                    IntChannel(
                        "min_shots_per_frequency",
                        "Minimum repeat count across frequency points",
                    ),
                    IntChannel(
                        "max_shots_per_frequency",
                        "Maximum repeat count across frequency points",
                    ),
                ],
                online_fn=self._analyse_interleaved_spectroscopy,
                online_analysis_identifier="spectral_dip_live",
            )
        ]

    def _analyse_interleaved_spectroscopy(
        self, axis_values, result_values, analysis_results
    ):
        del analysis_results
        probe_frequencies, survival_probabilities, survival_probability_errors, shot_counts = (
            aggregate_frequency_statistics(
                axis_values[self.probe_frequency],
                result_values[self.survived],
            )
        )

        outputs: dict[str, object] = {
            "fit_center": float("nan"),
            "fit_center_error": float("nan"),
            "fit_depth": float("nan"),
            "fit_fwhm": float("nan"),
            "num_unique_frequencies": int(len(probe_frequencies)),
            "min_shots_per_frequency": int(np.min(shot_counts)) if len(shot_counts) else 0,
            "max_shots_per_frequency": int(np.max(shot_counts)) if len(shot_counts) else 0,
        }
        analysis_annotations = []

        artifacts = {}
        if len(probe_frequencies) >= 4:
            try:
                fit_artifact = fit_gaussian_dip_artifact(
                    probe_frequencies,
                    survival_probabilities,
                    survival_probability_errors,
                )
            except Exception:
                fit_artifact = None
            if fit_artifact is not None:
                fit_center_stderr = fit_artifact["parameters"]["x0"].get("stderr")
                outputs.update(
                    {
                        "fit_center": float(artifact_parameter_value(fit_artifact, "x0")),
                        "fit_center_error": float("nan")
                        if fit_center_stderr is None
                        else float(fit_center_stderr),
                        "fit_depth": float(artifact_parameter_value(fit_artifact, "a")),
                        "fit_fwhm": float(artifact_parameter_value(fit_artifact, "fwhm")),
                    }
                )
                artifacts["spectral_dip_fit"] = fit_artifact
                analysis_annotations.append(
                    annotations.artifact_curve(
                        artifact="spectral_dip_fit",
                        x_axis=self.probe_frequency,
                        y_axis=self.survived,
                    )
                )

        return AnalysisFeedback(
            outputs=outputs,
            artifacts=artifacts,
            annotations=analysis_annotations,
        )


# Nice vs terse:
# - ``build_interleaved_demo_request()`` is the form normal users should copy.
# - ``build_interleaved_demo_request_terse()`` shows the equivalent explicit
#   ``RepeatPointPolicy(...)`` construction for readers who want the plumbing.
def build_interleaved_demo_request_terse(
    fragment: InterleavedSpectroscopyFragment,
) -> ScanRequest:
    """Create the flat repeated-shot demo request in the low-level point-policy style."""

    return ScanRequest(
        axes=(fragment.probe_frequency,),
        point_policy=RepeatPointPolicy(
            ExplicitPointPolicy(1, _interleaved_frequency_points()),
            stop_predicate=make_binomial_repeat_stop_predicate(
                fragment.survived,
                error_threshold=TARGET_PROBABILITY_ERROR,
                min_shots=MIN_SHOTS_PER_FREQUENCY,
                max_shots=MAX_SHOTS_PER_FREQUENCY,
            ),
            min_repeats=1,
            predicate_description="binomial shot error <= threshold",
            schedule="interleaved",
        ),
        metadata={"demo_name": "prepared_scan_interleaved_spectroscopy"},
        execution_policy=ExecutionPolicy(max_points_per_batch=1),
    )


def build_interleaved_demo_request(fragment: InterleavedSpectroscopyFragment) -> ScanRequest:
    """Create the flat repeated-shot demo request in the recommended request style."""

    return ScanRequest.explicit(
        [fragment.probe_frequency],
        _interleaved_frequency_points(),
        metadata={"demo_name": "prepared_scan_interleaved_spectroscopy"},
        execution_policy=ExecutionPolicy(max_points_per_batch=1),
    ).with_repeats(
        stop_predicate=make_binomial_repeat_stop_predicate(
            fragment.survived,
            error_threshold=TARGET_PROBABILITY_ERROR,
            min_shots=MIN_SHOTS_PER_FREQUENCY,
            max_shots=MAX_SHOTS_PER_FREQUENCY,
        ),
        min_repeats=1,
        predicate_description="binomial shot error <= threshold",
        schedule="interleaved",
    )


PreparedScanInterleavedSpectroscopy = make_fragment_prepared_scan_exp(
    InterleavedSpectroscopyFragment,
    build_interleaved_demo_request,
)


class BernoulliSpectroscopyLeafFragment(ExpFragment):
    """Leaf fragment returning one Bernoulli yes/no spectroscopy shot."""

    def build_fragment(self):
        self.setattr_param("probe_frequency", FloatParam, "probe frequency", default=0.0)
        self.setattr_result("survived", FloatChannel)
        self._rng = np.random.default_rng(seed=20260401)

    def run_once(self):
        time.sleep(POINT_DELAY_S)
        p_survive = survival_probability(self.probe_frequency.get())
        self.survived.push(1.0 if self._rng.random() < p_survive else 0.0)

    def get_default_analyses(self):
        return [
            build_binomial_chunk_analysis(
                self.survived,
                prefix="survival",
                probability_label="Observed survival probability",
                probability_error_label=(
                    "Binomial standard error on survival probability"
                ),
                shots_label="Number of shots in this repeat chunk",
                successes_label="Number of successes in this repeat chunk",
            )
        ]


class ChunkedInterleavedSpectroscopyFragment(ExpFragment):
    """Interleave frequency chunks while exposing chunk-level p/p_err as point data."""

    def build_fragment(self):
        self.setattr_param("probe_frequency", FloatParam, "probe frequency", default=0.0)
        self.repeat_scan = setattr_prepared_child_scan(
            self,
            "detector",
            BernoulliSpectroscopyLeafFragment,
            scan_name="repeat_scan",
            extra_metadata={
                "analysis_note": "fixed repeat chunks for one probe frequency"
            },
        )
        self.rebind_param(
            self.detector.probe_frequency,
            [self.probe_frequency],
            lambda values: values[self.probe_frequency],
            description="Drive the leaf probe frequency from the outer scan point",
        )

        self.setattr_result("survival_probability", FloatChannel)
        self.setattr_result(
            "survival_probability_error",
            FloatChannel,
            display_hints={"error_bar_for": self.survival_probability.path},
        )
        self.setattr_result("num_shots", IntChannel)
        self.setattr_result("num_successes", IntChannel)

    def run_once(self):
        repeat_request = _build_repeat_chunk_request()
        self.repeat_scan.configure(repeat_request)
        repeat_outputs = self.repeat_scan.execute()
        self.survival_probability.push(repeat_outputs["survival_probability"])
        self.survival_probability_error.push(repeat_outputs["survival_probability_error"])
        self.num_shots.push(repeat_outputs["num_shots"])
        self.num_successes.push(repeat_outputs["num_successes"])

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.probe_frequency],
                self._analyse_chunked_spectroscopy,
                [
                    FloatChannel("fit_center", "Fitted line centre"),
                    FloatChannel("fit_center_error", "One-sigma error on fitted centre"),
                    FloatChannel("fit_depth", "Fitted dip depth"),
                    FloatChannel("fit_fwhm", "Fitted line width"),
                    IntChannel(
                        "num_unique_frequencies",
                        "Number of unique frequency points seen so far",
                    ),
                    IntChannel(
                        "min_total_shots_per_frequency",
                        "Minimum combined shots across frequencies",
                    ),
                    IntChannel(
                        "max_total_shots_per_frequency",
                        "Maximum combined shots across frequencies",
                    ),
                ],
                online_fn=self._analyse_chunked_spectroscopy,
                online_analysis_identifier=_ONLINE_ANALYSIS_IDENTIFIER,
            )
        ]

    def _analyse_chunked_spectroscopy(self, axis_values, result_values, analysis_results):
        del analysis_results
        probe_frequencies, survival_probabilities, survival_probability_errors, total_shots = (
            aggregate_chunked_frequency_statistics(
                axis_values[self.probe_frequency],
                result_values[self.num_successes],
                result_values[self.num_shots],
            )
        )

        outputs: dict[str, object] = {
            "fit_center": float("nan"),
            "fit_center_error": float("nan"),
            "fit_depth": float("nan"),
            "fit_fwhm": float("nan"),
            "num_unique_frequencies": int(len(probe_frequencies)),
            "min_total_shots_per_frequency": int(np.min(total_shots))
            if len(total_shots)
            else 0,
            "max_total_shots_per_frequency": int(np.max(total_shots))
            if len(total_shots)
            else 0,
        }
        artifacts = {
            _FREQUENCY_STATISTICS_ARTIFACT: frequency_statistics_artifact(
                probe_frequencies,
                survival_probabilities,
                survival_probability_errors,
                total_shots,
            )
        }
        analysis_annotations = []

        if len(probe_frequencies) >= 4:
            try:
                fit_artifact = fit_gaussian_dip_artifact(
                    probe_frequencies,
                    survival_probabilities,
                    survival_probability_errors,
                )
            except Exception:
                fit_artifact = None
            if fit_artifact is not None:
                fit_center_stderr = fit_artifact["parameters"]["x0"].get("stderr")
                outputs.update(
                    {
                        "fit_center": float(artifact_parameter_value(fit_artifact, "x0")),
                        "fit_center_error": float("nan")
                        if fit_center_stderr is None
                        else float(fit_center_stderr),
                        "fit_depth": float(artifact_parameter_value(fit_artifact, "a")),
                        "fit_fwhm": float(artifact_parameter_value(fit_artifact, "fwhm")),
                    }
                )
                artifacts["spectral_dip_fit"] = fit_artifact
                analysis_annotations.append(
                    annotations.artifact_curve(
                        artifact="spectral_dip_fit",
                        x_axis=self.probe_frequency,
                        y_axis=self.survival_probability,
                    )
                )

        return AnalysisFeedback(
            outputs=outputs,
            artifacts=artifacts,
            annotations=analysis_annotations,
        )


# Nice vs terse:
# - ``build_chunked_demo_request()`` is the form normal users should copy.
# - ``build_chunked_demo_request_terse()`` shows the equivalent explicit
#   ``RepeatPointPolicy(...)`` construction for readers who want the plumbing.
def build_chunked_demo_request_terse(
    fragment: ChunkedInterleavedSpectroscopyFragment,
) -> ScanRequest:
    """Create the chunked demo request in the low-level point-policy style."""

    return ScanRequest(
        axes=(fragment.probe_frequency,),
        point_policy=RepeatPointPolicy(
            ExplicitPointPolicy(1, _interleaved_frequency_points()),
            stop_predicate=make_binomial_chunk_stop_predicate(
                fragment.num_successes,
                fragment.num_shots,
                error_threshold=CHUNKED_TARGET_PROBABILITY_ERROR,
                min_total_shots=MIN_TOTAL_SHOTS_PER_FREQUENCY,
                max_total_shots=MAX_TOTAL_SHOTS_PER_FREQUENCY,
            ),
            min_repeats=1,
            predicate_description="combined chunk statistics error <= threshold",
            schedule="interleaved",
        ),
        metadata={"demo_name": "prepared_scan_chunked_interleaved_spectroscopy"},
        execution_policy=ExecutionPolicy(max_points_per_batch=1),
    )


def build_chunked_demo_request(fragment: ChunkedInterleavedSpectroscopyFragment) -> ScanRequest:
    """Create the chunked demo request in the recommended request style."""

    return ScanRequest.explicit(
        [fragment.probe_frequency],
        _interleaved_frequency_points(),
        metadata={"demo_name": "prepared_scan_chunked_interleaved_spectroscopy"},
        execution_policy=ExecutionPolicy(max_points_per_batch=1),
    ).with_repeats(
        stop_predicate=make_binomial_chunk_stop_predicate(
            fragment.num_successes,
            fragment.num_shots,
            error_threshold=CHUNKED_TARGET_PROBABILITY_ERROR,
            min_total_shots=MIN_TOTAL_SHOTS_PER_FREQUENCY,
            max_total_shots=MAX_TOTAL_SHOTS_PER_FREQUENCY,
        ),
        min_repeats=1,
        predicate_description="combined chunk statistics error <= threshold",
        schedule="interleaved",
    )


PreparedScanChunkedInterleavedSpectroscopy = make_fragment_prepared_scan_exp(
    ChunkedInterleavedSpectroscopyFragment,
    build_chunked_demo_request,
)


class InterleavedSpectroscopySubscanRootFragment(ExpFragment):
    """Launch one prepared child scan that revisits probe frequencies round-robin."""

    def build_fragment(self):
        self.scan_frequency = setattr_prepared_child_scan(
            self,
            "spectroscopy_point",
            InterleavedSpectroscopyFragment,
            scan_name="scan_frequency",
            extra_metadata={
                "viewer_note": "interleaved repeated probe-frequency visits inside a prepared child scan"
            },
        )
        self.setattr_result("fit_center", FloatChannel)
        self.setattr_result("fit_center_error", FloatChannel)
        self.setattr_result("fit_depth", FloatChannel)
        self.setattr_result("fit_fwhm", FloatChannel)
        self.setattr_result("num_unique_frequencies", IntChannel)
        self.setattr_result("min_shots_per_frequency", IntChannel)
        self.setattr_result("max_shots_per_frequency", IntChannel)

    def run_once(self):
        self.scan_frequency.configure(build_interleaved_demo_request(self.spectroscopy_point))
        outputs = self.scan_frequency.execute()
        self.fit_center.push(outputs["fit_center"])
        self.fit_center_error.push(outputs["fit_center_error"])
        self.fit_depth.push(outputs["fit_depth"])
        self.fit_fwhm.push(outputs["fit_fwhm"])
        self.num_unique_frequencies.push(outputs["num_unique_frequencies"])
        self.min_shots_per_frequency.push(outputs["min_shots_per_frequency"])
        self.max_shots_per_frequency.push(outputs["max_shots_per_frequency"])


PreparedScanInterleavedSpectroscopySubscan = make_fragment_prepared_scan_exp(
    InterleavedSpectroscopySubscanRootFragment,
    lambda fragment: ScanRequest.single(
        metadata={"demo_name": "prepared_scan_interleaved_spectroscopy_subscan"}
    ),
)

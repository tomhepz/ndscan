"""Nested prepared-runtime spectroscopy example with repeated Bernoulli shots.

This example exercises the newer analysis path all the way up a three-level stack:

- a leaf fragment returns yes/no outcomes with a field-dependent Gaussian dip in the
  survival probability,
- repeated shots at one probe frequency are reduced online into ``p`` and ``p_err``,
- a frequency scan fits the dip centre for one field value,
- a field scan fits centre versus field to extract the line shift per field unit.

The example is intentionally structured to be useful in the runtime viewer:

- ``scan_field`` shows fitted line centres versus field,
- clicking one field point opens ``scan_frequency`` with the matching Gaussian-dip fit,
- clicking one frequency point opens ``repeat_scan`` with the raw Bernoulli shots.
"""

from __future__ import annotations

import math

import numpy as np
import time

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
from ndscan.scan import *


TRUE_CENTER_AT_ZERO = 0.35
TRUE_SHIFT_PER_FIELD = 0.28
TRUE_DIP_SIGMA = 0.18
TRUE_SURVIVAL_BASELINE = 0.93
TRUE_DIP_DEPTH = 0.55


def survival_probability(field: float, probe_frequency: float) -> float:
    """Return the Bernoulli success probability for one field/frequency point."""

    centre = TRUE_CENTER_AT_ZERO + TRUE_SHIFT_PER_FIELD * field
    detuning = (probe_frequency - centre) / TRUE_DIP_SIGMA
    dip = TRUE_DIP_DEPTH * math.exp(-0.5 * detuning**2)
    return float(np.clip(TRUE_SURVIVAL_BASELINE - dip, 1e-4, 1.0 - 1e-4))


def estimate_survival_from_shots(shots: list[float]) -> tuple[float, float]:
    """Return binomial survival probability and its standard error."""

    num_shots = len(shots)
    if num_shots == 0:
        return 0.0, float("inf")
    probability = float(sum(shots)) / float(num_shots)
    error = math.sqrt(max(probability * (1.0 - probability), 0.0) / num_shots)
    error = max(error, 0.5 / num_shots)
    return probability, error


def fit_gaussian_dip_artifact(
    probe_frequencies, survival_probabilities, survival_probability_errors
) -> dict[str, object]:
    """Fit one Gaussian dip in the survival probability."""

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
        label="spectral_dip",
    )
    _, artifact = fit_data_with_model(
        model,
        data,
        model_id="gaussian_dip",
        source={
            "x_key": "axis_0",
            "y_key": "channel_survival_probability",
        },
    )
    return artifact.to_dict()


def fit_line_artifact(
    fields, fitted_centres, centre_errors: list[float] | np.ndarray | None = None
) -> dict[str, object]:
    """Fit a straight line to centre versus field."""

    sensible = import_sensible_fitting()
    fields = np.asarray(fields, dtype=float)
    fitted_centres = np.asarray(fitted_centres, dtype=float)
    fit_data_kwargs = {
        "x": fields,
        "y": fitted_centres,
        "x_label": "field",
        "y_label": "fit_center",
        "label": "centre_vs_field",
    }
    if centre_errors is not None:
        centre_errors = np.asarray(centre_errors, dtype=float)
        finite = np.isfinite(centre_errors) & (centre_errors > 0.0)
        if finite.all():
            fit_data_kwargs["yerr"] = centre_errors

    data = sensible.FitData.normal(**fit_data_kwargs)
    _, artifact = fit_data_with_model(
        build_builtin_model("straight_line"),
        data,
        model_id="straight_line",
        source={
            "x_key": "axis_0",
            "y_key": "channel_fit_center",
        },
    )
    return artifact.to_dict()


def make_online_precision_stopper(*, error_threshold: float, min_shots: int):
    """Stop repeated shots once the online probability error is small enough."""

    def stop(feedback) -> bool:
        stats = feedback.online_analyses["repeat_stats"].outputs
        if stats.get("num_shots", 0) < min_shots:
            return False
        return stats["survival_probability_error"] <= error_threshold

    return stop


class YesNoSpectralDipFragment(ExpFragment):
    """Leaf fragment returning Bernoulli shots from a field-shifted spectral dip."""

    def build_fragment(self):
        self.setattr_param("field", FloatParam, "field", default=0.0)
        self.setattr_param("probe_frequency", FloatParam, "probe frequency", default=0.0)
        self.setattr_result("survived", FloatChannel)
        self._rng = np.random.default_rng(seed=20260401)

    def run_once(self):
        p_survive = survival_probability(self.field.get(), self.probe_frequency.get())
        time.sleep(0.05)
        self.survived.push(1.0 if self._rng.random() < p_survive else 0.0)

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [],
                self._analyse_repeat_statistics,
                analysis_results=[
                    FloatChannel("survival_probability", "Observed survival probability"),
                    FloatChannel(
                        "survival_probability_error",
                        "Binomial standard error on survival probability",
                    ),
                    IntChannel("num_shots", "Number of repeated Bernoulli shots"),
                ],
                online_fn=self._analyse_repeat_statistics,
                online_analysis_identifier="repeat_stats",
            )
        ]

    def _analyse_repeat_statistics(self, axis_values, result_values):
        del axis_values
        shots = list(result_values[self.survived])
        probability, probability_error = estimate_survival_from_shots(shots)
        return AnalysisFeedback(
            outputs={
                "survival_probability": probability,
                "survival_probability_error": probability_error,
                "num_shots": len(shots),
            }
        )


class SpectroscopyAtFieldFragment(ExpFragment):
    """Run a frequency scan for one field value and fit the Gaussian dip centre."""

    def build_fragment(self):
        self.setattr_param("field", FloatParam, "field", default=0.0)
        self.repeat_scan = setattr_prepared_child_scan(
            self,
            "detector",
            YesNoSpectralDipFragment,
            scan_name="repeat_scan",
            extra_metadata={
                "analysis_note": "online Bernoulli reduction for repeated shots"
            },
        )
        self.rebind_param(
            self.detector.field,
            [self.field],
            lambda values: values[self.field],
            description="Drive the leaf fragment's field from the outer field scan",
        )

        self.setattr_result("survival_probability", FloatChannel)
        self.setattr_result("survival_probability_error", FloatChannel)
        self.setattr_result("num_shots", IntChannel)

    def run_once(self):
        repeat_request = ScanRequest(
            axes=(),
            point_policy=RepeatPointPolicy(
                SinglePointPolicy(),
                stop_predicate=make_online_precision_stopper(
                    error_threshold=0.07,
                    min_shots=20,
                ),
                min_repeats=20,
                max_repeats=120,
                predicate_description=(
                    "repeat_stats.survival_probability_error <= 0.07"
                ),
            ),
            execution_policy=ExecutionPolicy(max_points_per_batch=16),
        )
        self.repeat_scan.configure(repeat_request)
        repeat_outputs = self.repeat_scan.execute()
        self.survival_probability.push(repeat_outputs["survival_probability"])
        self.survival_probability_error.push(repeat_outputs["survival_probability_error"])
        self.num_shots.push(repeat_outputs["num_shots"])

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.detector.probe_frequency],
                self._analyse_spectral_dip,
                [
                    FloatChannel("fit_center", "Fitted line centre"),
                    FloatChannel("fit_center_error", "One-sigma error on fitted centre"),
                    FloatChannel("fit_depth", "Fitted dip depth"),
                    FloatChannel("fit_fwhm", "Fitted line width"),
                ],
            )
        ]

    def _analyse_spectral_dip(self, axis_values, result_values, analysis_results):
        del analysis_results
        probe_frequencies = np.asarray(axis_values[self.detector.probe_frequency], dtype=float)
        survival_probabilities = np.asarray(result_values[self.survival_probability], dtype=float)
        survival_probability_errors = np.asarray(
            result_values[self.survival_probability_error],
            dtype=float,
        )

        fit_artifact = fit_gaussian_dip_artifact(
            probe_frequencies,
            survival_probabilities,
            survival_probability_errors,
        )
        fit_center = float(artifact_parameter_value(fit_artifact, "x0"))
        fit_depth = float(artifact_parameter_value(fit_artifact, "a"))
        fit_fwhm = float(artifact_parameter_value(fit_artifact, "fwhm"))
        fit_center_error = float(
            fit_artifact["parameters"]["x0"].get("stderr", float("nan"))
        )
        return AnalysisFeedback(
            outputs={
                "fit_center": fit_center,
                "fit_center_error": fit_center_error,
                "fit_depth": fit_depth,
                "fit_fwhm": fit_fwhm,
            },
            artifacts={"spectral_dip_fit": fit_artifact},
            annotations=[
                annotations.artifact_curve(
                    artifact="spectral_dip_fit",
                    x_axis=self.detector.probe_frequency,
                    y_axis=self.survival_probability,
                ),
                annotations.artifact_location(
                    artifact="spectral_dip_fit",
                    axis=self.detector.probe_frequency,
                    parameter="x0",
                    associated_channels=[self.survival_probability],
                ),
            ],
        )


class FieldShiftSpectroscopyFragment(ExpFragment):
    """Scan field values, fit the line centre at each one, then fit the shift."""

    def build_fragment(self):
        self.frequency_scan = setattr_prepared_child_scan(
            self,
            "frequency_point",
            SpectroscopyAtFieldFragment,
            scan_name="scan_frequency",
            extra_metadata={
                "analysis_note": "fit Gaussian dip centre at each field value"
            },
        )
        self.setattr_result("fit_center", FloatChannel)
        self.setattr_result("fit_center_error", FloatChannel)
        self.setattr_result("fit_depth", FloatChannel)

    def run_once(self):
        probe_points = np.linspace(-0.5, 1.2, 13).tolist()
        self.frequency_scan.configure(
            ScanRequest.cartesian(
                [(self.frequency_point.detector.probe_frequency, probe_points)],
                execution_policy=ExecutionPolicy(max_points_per_batch=1),
            )
        )
        outputs = self.frequency_scan.execute()
        self.fit_center.push(outputs["fit_center"])
        self.fit_center_error.push(outputs["fit_center_error"])
        self.fit_depth.push(outputs["fit_depth"])

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.frequency_point.field],
                self._analyse_field_shift,
                [
                    FloatChannel("fit_shift_per_field", "Fitted line shift per field"),
                    FloatChannel("fit_centre_at_zero_field", "Fitted zero-field centre"),
                ],
            )
        ]

    def _analyse_field_shift(self, axis_values, result_values, analysis_results):
        del analysis_results
        fields = np.asarray(axis_values[self.frequency_point.field], dtype=float)
        fitted_centres = np.asarray(result_values[self.fit_center], dtype=float)
        centre_errors = np.asarray(result_values[self.fit_center_error], dtype=float)

        fit_artifact = fit_line_artifact(fields, fitted_centres, centre_errors)
        shift_per_field = float(artifact_parameter_value(fit_artifact, "m"))
        centre_at_zero_field = float(artifact_parameter_value(fit_artifact, "b"))
        return AnalysisFeedback(
            outputs={
                "fit_shift_per_field": shift_per_field,
                "fit_centre_at_zero_field": centre_at_zero_field,
            },
            artifacts={"field_shift_fit": fit_artifact},
            annotations=[
                annotations.artifact_curve(
                    artifact="field_shift_fit",
                    x_axis=self.frequency_point.field,
                    y_axis=self.fit_center,
                )
            ],
        )


class FieldShiftDemoRootFragment(ExpFragment):
    """Root fragment launching the field scan and exposing the final slope."""

    def build_fragment(self):
        self.field_scan = setattr_prepared_child_scan(
            self,
            "field_point",
            FieldShiftSpectroscopyFragment,
            scan_name="scan_field",
            extra_metadata={
                "viewer_note": "field scan over field-shifted Gaussian-dip spectroscopy"
            },
        )
        self.setattr_result("fit_shift_per_field", FloatChannel)
        self.setattr_result("fit_centre_at_zero_field", FloatChannel)

    def run_once(self):
        field_points = np.linspace(-1.5, 1.5, 5).tolist()
        self.field_scan.configure(
            ScanRequest.cartesian(
                [(self.field_point.frequency_point.field, field_points)],
                execution_policy=ExecutionPolicy(max_points_per_batch=1),
            )
        )
        outputs = self.field_scan.execute()
        self.fit_shift_per_field.push(outputs["fit_shift_per_field"])
        self.fit_centre_at_zero_field.push(outputs["fit_centre_at_zero_field"])


PreparedScanFieldShiftSpectroscopy = make_fragment_prepared_scan_exp(
    FieldShiftDemoRootFragment,
    lambda fragment: ScanRequest.single(
        metadata={"demo_name": "prepared_scan_field_shift_spectroscopy"}
    ),
)

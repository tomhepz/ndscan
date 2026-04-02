"""Prepared-runtime migration of a legacy counts/chunk/frequency/field stack.

This example is the modern counterpart to an older pattern that looked roughly like:

- one physical shot fragment returns raw counts,
- an explicit ``shot_index`` subscan repeats that shot and reduces the counts into
  bright-state probabilities,
- one scan over frequency fits the spectral centre,
- one outer scan over field fits the centre shift.

The prepared-runtime version keeps the same ownership model while using the newer,
cleaner pieces:

- repeated shots are execution policy via ``ScanRequest.single().with_repeats(...)``
  rather than a scientific ``shot_index`` scan axis,
- the repeated-shot reduction lives in the shot fragment's default analysis, close to
  the raw counts channel it interprets,
- higher-level fits return ``AnalysisFeedback`` with artifact-backed annotations rather
  than opaque sampled ``fit_xs`` / ``fit_ys`` arrays,
- prepared child scans preserve the nested site history in the runtime viewer.

The simulated shot now follows the same path as the old mock readout:

- draw one fluorescence image,
- sum counts over ROI boxes,
- threshold those ROI sums into bright/dark classification.

The chunk fragment republishes both pooled ROI-average probabilities and per-group
probabilities, so the migrated example still feels recognisably close to the original
code while using the current runtime API.
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
from ndscan.scan import *

__all__ = [
    "NUM_GROUPS",
    "NUM_ROIS",
    "POINT_DELAY_S",
    "DEFAULT_SHOTS_PER_CHUNK",
    "DURATION_POINTS",
    "FREQUENCY_POINTS",
    "FIELD_POINTS",
    "IMAGE_SHAPE",
    "DEFAULT_ROIS",
    "DEFAULT_RABI_FREQUENCY",
    "TRUE_ZERO_FIELD_FREQUENCY",
    "TRUE_FREQUENCY_SHIFT_PER_CURRENT",
    "SingleShotBase",
    "SingleShotCountsFragment",
    "CountsShotChunkFragment",
    "FrequencyCalibrationAtFieldFragment",
    "CountsFieldCalibrationRootFragment",
    "HostRuntimeCountsShotChunkTimeScan",
    "HostRuntimeCountsFrequencyCalibration",
    "HostRuntimeCountsFieldCalibration",
]


NUM_GROUPS = 8
NUM_ROIS = 1
PRIMARY_ROI_INDEX = 0

POINT_DELAY_S = 0.002
DEFAULT_SHOTS_PER_CHUNK = 40

DURATION_POINTS = np.linspace(0.0, 1.2, 21).tolist()
FREQUENCY_POINTS = np.linspace(5.0, 15.0, 20).tolist()
FIELD_POINTS = np.linspace(0.0, 10.0, 6).tolist()

DEFAULT_THRESHOLD = 2000
DEFAULT_RABI_FREQUENCY = 1.0
DEFAULT_PULSE_DURATION = 0.48

IMAGE_SHAPE = (64, 64)
IMAGE_BACKGROUND_MEAN = 200.0
IMAGE_BRIGHT_COUNTS_MEAN = 1500.0
IMAGE_SPOT_SIGMA = 1.4

TRUE_ZERO_FIELD_FREQUENCY = 10.0
TRUE_FREQUENCY_SHIFT_PER_CURRENT = 0.13
DEFAULT_ROIS = (
    ((15, 18, 5, 8),),
    ((15, 18, 11, 14),),
    ((15, 18, 17, 20),),
    ((15, 18, 23, 26),),
    ((15, 18, 29, 32),),
    ((15, 18, 35, 38),),
    ((15, 18, 41, 44),),
    ((15, 18, 47, 50),),
)


def _pooled_probability_name() -> str:
    return "pooled_probability"


def _pooled_probability_error_name() -> str:
    return "pooled_probability_error"


def _pooled_num_shots_name() -> str:
    return "pooled_num_shots"


def _pooled_num_successes_name() -> str:
    return "pooled_num_successes"


def _group_probability_name() -> str:
    return "group_probability"


def _group_probability_error_name() -> str:
    return "group_probability_error"


def _estimate_probability_from_counts(
    num_successes: float, num_shots: int
) -> tuple[float, float]:
    """Return a conservative binomial estimate and standard error."""

    if num_shots <= 0:
        return 0.0, float("inf")
    probability = float(num_successes) / float(num_shots)
    error = math.sqrt(max(probability * (1.0 - probability), 0.0) / num_shots)
    error = max(error, 0.5 / num_shots)
    return probability, error


def _primary_probability_name() -> str:
    return _pooled_probability_name()


def _primary_probability_error_name() -> str:
    return _pooled_probability_error_name()


def _p_bright_detuned_rabi(
    *,
    coil_current: float,
    probe_frequency: float,
    pulse_duration: float,
    rabi_frequency: float,
) -> float:
    """Return the old mock-model bright probability from detuned Rabi driving."""

    f0 = TRUE_ZERO_FIELD_FREQUENCY + TRUE_FREQUENCY_SHIFT_PER_CURRENT * coil_current
    omega = 2.0 * math.pi * max(0.0, rabi_frequency)
    delta = 2.0 * math.pi * (probe_frequency - f0)
    omega_eff = math.hypot(omega, delta)
    if omega_eff == 0.0:
        return 0.0
    probability = (omega / omega_eff) ** 2 * (
        math.sin(0.5 * omega_eff * max(0.0, pulse_duration)) ** 2
    )
    return float(np.clip(probability, 0.0, 1.0))


def _gaussian2d(shape, x0: float, y0: float, sigma: float) -> np.ndarray:
    """Return a normalised 2D Gaussian PSF."""

    yy, xx = np.indices(shape)
    gaussian = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))
    total = float(np.sum(gaussian))
    if total > 0.0:
        gaussian /= total
    return gaussian


def _image_from_probabilities_and_locations(
    locations,
    *,
    rng: np.random.Generator,
    shape=IMAGE_SHAPE,
    bright_counts_mean: float = IMAGE_BRIGHT_COUNTS_MEAN,
    background_mean: float = IMAGE_BACKGROUND_MEAN,
    sigma: float = IMAGE_SPOT_SIGMA,
) -> np.ndarray:
    """Draw one fluorescence image from spot locations and bright probabilities."""

    image = rng.poisson(background_mean, size=shape).astype(np.int32)
    for x0, y0, p_bright in locations:
        if rng.random() >= p_bright:
            continue
        amplitude = int(rng.poisson(bright_counts_mean))
        if amplitude <= 0:
            continue
        psf = _gaussian2d(shape, x0, y0, sigma)
        image += rng.poisson(amplitude * psf).astype(np.int32)
    return image


def _default_rois():
    """Return the default ``(group, roi)`` ROI layout."""

    return [[tuple(bounds) for bounds in group] for group in DEFAULT_ROIS]


def _normalise_rois(raw_rois) -> list[list[tuple[int, int, int, int]]] | None:
    """Return validated ROI bounds or ``None`` when the layout is incompatible."""

    if raw_rois is None:
        return None
    try:
        rois = [
            [tuple(int(value) for value in bounds) for bounds in roi_group]
            for roi_group in raw_rois
        ]
    except Exception:
        return None

    if len(rois) != NUM_GROUPS:
        return None
    if any(len(group) != NUM_ROIS for group in rois):
        return None
    if any(len(bounds) != 4 for group in rois for bounds in group):
        return None
    return rois


def _spot_locations_from_probabilities(rois, probabilities) -> list[tuple[float, float, float]]:
    """Map one ``(group, roi)`` probability table to ROI-centred spot locations."""

    locations = []
    for group_index, roi_group in enumerate(rois):
        for roi_index, (y0, y1, x0, x1) in enumerate(roi_group):
            x_centre = 0.5 * (float(x0) + float(x1 - 1))
            y_centre = 0.5 * (float(y0) + float(y1 - 1))
            locations.append((x_centre, y_centre, float(probabilities[group_index, roi_index])))
    return locations


def _sum_counts_in_rois(image: np.ndarray, rois) -> np.ndarray:
    """Return one fixed-shape ``(group, roi)`` counts array from an image and ROI boxes."""

    counts = np.empty((len(rois), len(rois[0])), dtype=np.int32)
    for group_index, roi_group in enumerate(rois):
        for roi_index, (y0, y1, x0, x1) in enumerate(roi_group):
            counts[group_index, roi_index] = int(np.sum(image[y0:y1, x0:x1]))
    return counts


def _build_chunk_analysis_channels() -> list[ResultChannel]:
    """Return the repeated-shot analysis outputs for one counts chunk."""

    pooled_probability = ArrayChannel(
        _pooled_probability_name(),
        "Pooled bright probability",
        element_type="float",
        shape=(NUM_ROIS,),
        dim_names=("roi",),
    )
    return [
        pooled_probability,
        ArrayChannel(
            _pooled_probability_error_name(),
            "Pooled bright probability error",
            element_type="float",
            shape=(NUM_ROIS,),
            dim_names=("roi",),
            display_hints={"error_bar_for": pooled_probability.path},
        ),
        ArrayChannel(
            _group_probability_name(),
            "Group bright probability",
            element_type="float",
            shape=(NUM_GROUPS, NUM_ROIS),
            dim_names=("group", "roi"),
        ),
        ArrayChannel(
            _group_probability_error_name(),
            "Group bright probability error",
            element_type="float",
            shape=(NUM_GROUPS, NUM_ROIS),
            dim_names=("group", "roi"),
            display_hints={"error_bar_for": _group_probability_name()},
        ),
        ArrayChannel(
            _pooled_num_shots_name(),
            "Pooled number of Bernoulli trials",
            element_type="int",
            shape=(NUM_ROIS,),
            dim_names=("roi",),
        ),
        ArrayChannel(
            _pooled_num_successes_name(),
            "Pooled number of bright outcomes",
            element_type="int",
            shape=(NUM_ROIS,),
            dim_names=("roi",),
        ),
    ]


def _fit_frequency_peak_artifact(
    probe_frequencies, probabilities, probability_errors
) -> dict[str, object]:
    """Fit one peaked frequency response with a Gaussian-with-offset model."""

    sensible = import_sensible_fitting()
    probe_frequencies = np.asarray(probe_frequencies, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    probability_errors = np.asarray(probability_errors, dtype=float)

    x_min = float(np.min(probe_frequencies))
    x_max = float(np.max(probe_frequencies))
    span = max(x_max - x_min, 1e-6)

    model = build_builtin_model("gaussian_with_offset").bound(
        x0=(x_min, x_max),
        y0=(0.0, 1.0),
        a=(0.0, 1.0),
        sigma=(0.03, span),
    )
    data = sensible.FitData.normal(
        x=probe_frequencies,
        y=probabilities,
        yerr=probability_errors,
        x_label="probe_frequency",
        y_label=_primary_probability_name(),
        label="counts_frequency_peak",
    )
    _, artifact = fit_data_with_model(
        model,
        data,
        model_id="gaussian_with_offset",
        source={
            "x_key": "axis_0",
            "y_key": f"channel_{_primary_probability_name()}",
        },
    )
    return artifact.to_dict()


def _fit_time_scan_artifact(
    pulse_durations, probabilities, probability_errors
) -> dict[str, object]:
    """Fit the duration scan with the built-in sinusoid model."""

    sensible = import_sensible_fitting()
    pulse_durations = np.asarray(pulse_durations, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    probability_errors = np.asarray(probability_errors, dtype=float)

    model = build_builtin_model("sinusoid").bound(
        amplitude=(0.0, 1.0),
        offset=(0.0, 1.0),
        frequency=(0.05, 2.0),
        phase=(-math.pi, math.pi),
    )
    data = sensible.FitData.normal(
        x=pulse_durations,
        y=probabilities,
        yerr=probability_errors,
        x_label="pulse_duration",
        y_label=_primary_probability_name(),
        label="counts_duration_scan",
    )
    _, artifact = fit_data_with_model(
        model,
        data,
        model_id="sinusoid",
        source={
            "x_key": "axis_0",
            "y_key": f"channel_{_primary_probability_name()}",
        },
    )
    return artifact.to_dict()


def _fit_line_artifact(
    xs, ys, y_errors: list[float] | np.ndarray | None = None
) -> dict[str, object]:
    """Fit a straight line to ``ys(x)`` and return the canonical artifact."""

    sensible = import_sensible_fitting()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    fit_data_kwargs = {
        "x": xs,
        "y": ys,
        "x_label": "coil_current",
        "y_label": "fit_center",
        "label": "field_calibration",
    }
    if y_errors is not None:
        y_errors = np.asarray(y_errors, dtype=float)
        finite = np.isfinite(y_errors) & (y_errors > 0.0)
        if finite.all():
            fit_data_kwargs["yerr"] = y_errors

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


class SingleShotBase(ExpFragment):
    """One physical attempt. Subclasses must expose their raw counts channel."""

    def get_counts_handle(self) -> ResultChannel:
        raise NotImplementedError


class SingleShotCountsFragment(SingleShotBase):
    """One simulated attempt returning ROI counts extracted from a mock image."""

    def build_fragment(self):
        self.setattr_param(
            "coil_current",
            FloatParam,
            "Coil current",
            default=0.0,
            unit="A",
        )
        self.setattr_param(
            "probe_frequency",
            FloatParam,
            "Probe frequency",
            default=TRUE_ZERO_FIELD_FREQUENCY,
            unit="MHz",
        )
        self.setattr_param(
            "pulse_duration",
            FloatParam,
            "Pulse duration",
            default=DEFAULT_PULSE_DURATION,
            unit="us",
            min=0.0,
        )
        self.setattr_param(
            "rabi_frequency",
            FloatParam,
            "Rabi frequency",
            default=DEFAULT_RABI_FREQUENCY,
            unit="MHz",
            min=0.01,
        )
        self.setattr_param(
            "threshold_counts",
            IntParam,
            "Threshold counts",
            default=DEFAULT_THRESHOLD,
            min=1,
        )

        self.setattr_result(
            "counts",
            ArrayChannel,
            "ROI counts",
            element_type="int",
            shape=(NUM_GROUPS, NUM_ROIS),
            dim_names=("group", "roi"),
            unit="cts",
            scale=1.0,
        )
        self._rng = np.random.default_rng(seed=20260402)

    def get_counts_handle(self) -> ResultChannel:
        return self.counts

    def run_once(self):
        if POINT_DELAY_S > 0.0:
            time.sleep(POINT_DELAY_S)
        rois = _normalise_rois(self.get_dataset("rois", default=None))
        if rois is None:
            rois = _default_rois()
            self.set_dataset("rois", rois, broadcast=True)

        bright_probability = _p_bright_detuned_rabi(
            coil_current=self.coil_current.get(),
            probe_frequency=self.probe_frequency.get(),
            pulse_duration=self.pulse_duration.get(),
            rabi_frequency=self.rabi_frequency.get(),
        )
        probabilities = np.full((NUM_GROUPS, NUM_ROIS), bright_probability, dtype=float)

        image = _image_from_probabilities_and_locations(
            _spot_locations_from_probabilities(rois, probabilities),
            rng=self._rng,
        )
        self.set_dataset("last_image", image, broadcast=True)
        self.counts.push(_sum_counts_in_rois(image, rois))

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [],
                self._analyse_thresholded_counts_chunk,
                analysis_results=_build_chunk_analysis_channels(),
                online_fn=self._analyse_thresholded_counts_chunk,
                online_analysis_identifier="thresholded_counts_chunk",
            )
        ]

    def _analyse_thresholded_counts_chunk(self, axis_values, result_values, analysis_results):
        del axis_values, analysis_results

        counts = np.asarray(result_values[self.get_counts_handle()])
        if counts.size == 0:
            return AnalysisFeedback()

        threshold = int(self.threshold_counts.get())
        bright = counts >= threshold
        num_shots = int(bright.shape[0])
        num_successes = np.sum(bright, axis=0).astype(int)

        outputs: dict[str, object] = {}
        pooled_probabilities = np.empty((NUM_ROIS,), dtype=float)
        pooled_probability_errors = np.empty((NUM_ROIS,), dtype=float)
        pooled_num_shots = np.empty((NUM_ROIS,), dtype=int)
        pooled_num_successes = np.empty((NUM_ROIS,), dtype=int)
        group_probabilities = np.empty((NUM_GROUPS, NUM_ROIS), dtype=float)
        group_probability_errors = np.empty((NUM_GROUPS, NUM_ROIS), dtype=float)

        for roi_index in range(NUM_ROIS):
            pooled_successes = int(np.sum(num_successes[:, roi_index]))
            pooled_shots = int(NUM_GROUPS * num_shots)
            pooled_probability, pooled_error = _estimate_probability_from_counts(
                pooled_successes,
                pooled_shots,
            )
            pooled_probabilities[roi_index] = pooled_probability
            pooled_probability_errors[roi_index] = pooled_error
            pooled_num_shots[roi_index] = pooled_shots
            pooled_num_successes[roi_index] = pooled_successes

            for group_index in range(NUM_GROUPS):
                probability, probability_error = _estimate_probability_from_counts(
                    int(num_successes[group_index, roi_index]),
                    num_shots,
                )
                group_probabilities[group_index, roi_index] = probability
                group_probability_errors[group_index, roi_index] = probability_error

        outputs[_pooled_probability_name()] = pooled_probabilities
        outputs[_pooled_probability_error_name()] = pooled_probability_errors
        outputs[_pooled_num_shots_name()] = pooled_num_shots
        outputs[_pooled_num_successes_name()] = pooled_num_successes
        outputs[_group_probability_name()] = group_probabilities
        outputs[_group_probability_error_name()] = group_probability_errors

        return AnalysisFeedback(outputs=outputs)


class CountsShotChunkFragment(ExpFragment):
    """Repeat one counts-producing shot and republish chunk statistics as point data."""

    def build_fragment(self):
        self.repeat_scan = setattr_prepared_child_scan(
            self,
            "shot",
            SingleShotCountsFragment,
            scan_name="repeat_scan",
            extra_metadata={
                "analysis_note": "threshold counts into chunk-level bright probabilities"
            },
        )
        self.setattr_param(
            "shots_per_chunk",
            IntParam,
            "Shots per chunk",
            default=DEFAULT_SHOTS_PER_CHUNK,
            min=1,
        )

        self._chunk_output_channels: dict[str, ResultChannel] = {}
        self._pooled_probability_channel = self.setattr_result(
            _pooled_probability_name(),
            ArrayChannel,
            "Pooled bright probability",
            element_type="float",
            shape=(NUM_ROIS,),
            dim_names=("roi",),
        )
        self._pooled_probability_error_channel = self.setattr_result(
            _pooled_probability_error_name(),
            ArrayChannel,
            "Pooled bright probability error",
            element_type="float",
            shape=(NUM_ROIS,),
            dim_names=("roi",),
            display_hints={"error_bar_for": self._pooled_probability_channel.path},
        )
        self._group_probability_channel = self.setattr_result(
            _group_probability_name(),
            ArrayChannel,
            "Group bright probability",
            element_type="float",
            shape=(NUM_GROUPS, NUM_ROIS),
            dim_names=("group", "roi"),
        )
        self._group_probability_error_channel = self.setattr_result(
            _group_probability_error_name(),
            ArrayChannel,
            "Group bright probability error",
            element_type="float",
            shape=(NUM_GROUPS, NUM_ROIS),
            dim_names=("group", "roi"),
            display_hints={"error_bar_for": self._group_probability_channel.path},
        )
        self._pooled_num_shots_channel = self.setattr_result(
            _pooled_num_shots_name(),
            ArrayChannel,
            "Pooled number of Bernoulli trials",
            element_type="int",
            shape=(NUM_ROIS,),
            dim_names=("roi",),
        )
        self._pooled_num_successes_channel = self.setattr_result(
            _pooled_num_successes_name(),
            ArrayChannel,
            "Pooled number of bright outcomes",
            element_type="int",
            shape=(NUM_ROIS,),
            dim_names=("roi",),
        )

        self._chunk_output_channels[_pooled_probability_name()] = (
            self._pooled_probability_channel
        )
        self._chunk_output_channels[_pooled_probability_error_name()] = (
            self._pooled_probability_error_channel
        )
        self._chunk_output_channels[_group_probability_name()] = (
            self._group_probability_channel
        )
        self._chunk_output_channels[_group_probability_error_name()] = (
            self._group_probability_error_channel
        )
        self._chunk_output_channels[_pooled_num_shots_name()] = (
            self._pooled_num_shots_channel
        )
        self._chunk_output_channels[_pooled_num_successes_name()] = (
            self._pooled_num_successes_channel
        )

    def run_once(self):
        shots_per_chunk = int(self.shots_per_chunk.get())
        self.repeat_scan.configure(
            ScanRequest.single(
                execution_policy=ExecutionPolicy(
                    max_points_per_batch=min(shots_per_chunk, 16)
                )
            ).with_repeats(repeats=shots_per_chunk)
        )
        outputs = self.repeat_scan.execute()
        for name, channel in self._chunk_output_channels.items():
            channel.push(outputs[name])

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.shot.pulse_duration],
                self._analyse_time_scan,
                [
                    FloatChannel("fit_pi_time", "Fitted pi time"),
                    FloatChannel("fit_pi_time_error", "Fitted pi time error"),
                    FloatChannel(
                        "fit_rabi_frequency",
                        "Fitted oscillation frequency",
                    ),
                ],
            ),
            CustomAnalysis(
                [self.shot.probe_frequency],
                self._analyse_frequency_scan,
                [
                    FloatChannel("fit_center", "Fitted spectral centre"),
                    FloatChannel("fit_center_error", "Fitted spectral centre error"),
                    FloatChannel("fit_depth", "Fitted spectral dip depth"),
                    FloatChannel("fit_fwhm", "Fitted spectral FWHM"),
                ],
            ),
        ]

    def _analyse_time_scan(self, axis_values, result_values, analysis_results):
        del analysis_results

        pulse_durations = np.asarray(axis_values[self.shot.pulse_duration], dtype=float)
        probabilities = np.asarray(
            result_values[self._pooled_probability_channel],
            dtype=float,
        )[:, PRIMARY_ROI_INDEX]
        probability_errors = np.asarray(
            result_values[self._pooled_probability_error_channel],
            dtype=float,
        )[:, PRIMARY_ROI_INDEX]

        fit_artifact = _fit_time_scan_artifact(
            pulse_durations,
            probabilities,
            probability_errors,
        )
        fit_frequency = float(artifact_parameter_value(fit_artifact, "frequency"))
        fit_frequency_error = fit_artifact["parameters"]["frequency"].get("stderr")
        fit_pi_time = float("nan")
        fit_pi_time_error = float("nan")
        if np.isfinite(fit_frequency) and fit_frequency > 0.0:
            fit_pi_time = 0.5 / fit_frequency
            if fit_frequency_error is not None and np.isfinite(float(fit_frequency_error)):
                fit_pi_time_error = 0.5 * float(fit_frequency_error) / fit_frequency**2

        return AnalysisFeedback(
            outputs={
                "fit_pi_time": fit_pi_time,
                "fit_pi_time_error": fit_pi_time_error,
                "fit_rabi_frequency": fit_frequency,
            },
            artifacts={"time_scan_fit": fit_artifact},
            annotations=[
                annotations.artifact_curve(
                    artifact="time_scan_fit",
                    x_axis=self.shot.pulse_duration,
                    y_axis=self._pooled_probability_channel,
                    y_indices=[PRIMARY_ROI_INDEX],
                )
            ],
        )

    def _analyse_frequency_scan(self, axis_values, result_values, analysis_results):
        del analysis_results

        probe_frequencies = np.asarray(axis_values[self.shot.probe_frequency], dtype=float)
        probabilities = np.asarray(
            result_values[self._pooled_probability_channel],
            dtype=float,
        )[:, PRIMARY_ROI_INDEX]
        probability_errors = np.asarray(
            result_values[self._pooled_probability_error_channel],
            dtype=float,
        )[:, PRIMARY_ROI_INDEX]

        fit_artifact = _fit_frequency_peak_artifact(
            probe_frequencies,
            probabilities,
            probability_errors,
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
            artifacts={"frequency_scan_fit": fit_artifact},
            annotations=[
                annotations.artifact_curve(
                    artifact="frequency_scan_fit",
                    x_axis=self.shot.probe_frequency,
                    y_axis=self._pooled_probability_channel,
                    y_indices=[PRIMARY_ROI_INDEX],
                )
            ],
        )


class FrequencyCalibrationAtFieldFragment(ExpFragment):
    """For one field value, scan frequency and fit the spectral centre."""

    def build_fragment(self):
        self.frequency_scan = setattr_prepared_child_scan(
            self,
            "measurement_point",
            CountsShotChunkFragment,
            scan_name="scan_frequency",
            extra_metadata={
                "viewer_note": "frequency scan over one counts-based shot chunk"
            },
        )
        self.setattr_result("fit_center", FloatChannel)
        self.setattr_result(
            "fit_center_error",
            FloatChannel,
            display_hints={"error_bar_for": self.fit_center.path},
        )
        self.setattr_result("fit_depth", FloatChannel)

    def run_once(self):
        self.frequency_scan.configure(build_frequency_scan_request(self.measurement_point))
        outputs = self.frequency_scan.execute()
        self.fit_center.push(outputs["fit_center"])
        self.fit_center_error.push(outputs["fit_center_error"])
        self.fit_depth.push(outputs["fit_depth"])

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [self.measurement_point.shot.coil_current],
                self._analyse_field_calibration,
                [
                    FloatChannel("fit_shift_per_current", "Fitted frequency shift per current"),
                    FloatChannel(
                        "fit_center_at_zero_current",
                        "Fitted zero-current spectral centre",
                    ),
                ],
            )
        ]

    def _analyse_field_calibration(self, axis_values, result_values, analysis_results):
        del analysis_results

        currents = np.asarray(axis_values[self.measurement_point.shot.coil_current], dtype=float)
        centres = np.asarray(result_values[self.fit_center], dtype=float)
        centre_errors = np.asarray(result_values[self.fit_center_error], dtype=float)

        fit_artifact = _fit_line_artifact(currents, centres, centre_errors)
        return AnalysisFeedback(
            outputs={
                "fit_shift_per_current": float(artifact_parameter_value(fit_artifact, "m")),
                "fit_center_at_zero_current": float(
                    artifact_parameter_value(fit_artifact, "b")
                ),
            },
            artifacts={"field_calibration_fit": fit_artifact},
            annotations=[
                annotations.artifact_curve(
                    artifact="field_calibration_fit",
                    x_axis=self.measurement_point.shot.coil_current,
                    y_axis=self.fit_center,
                )
            ],
        )


class CountsFieldCalibrationRootFragment(ExpFragment):
    """Root fragment launching the field scan and exposing the fitted shift."""

    def build_fragment(self):
        self.field_scan = setattr_prepared_child_scan(
            self,
            "field_point",
            FrequencyCalibrationAtFieldFragment,
            scan_name="scan_field",
            extra_metadata={
                "viewer_note": "counts-based field calibration with nested frequency scans"
            },
        )
        self.setattr_result("fit_shift_per_current", FloatChannel)
        self.setattr_result("fit_center_at_zero_current", FloatChannel)

    def run_once(self):
        self.field_scan.configure(build_field_scan_request(self.field_point))
        outputs = self.field_scan.execute()
        self.fit_shift_per_current.push(outputs["fit_shift_per_current"])
        self.fit_center_at_zero_current.push(outputs["fit_center_at_zero_current"])


def build_time_scan_request(fragment: CountsShotChunkFragment) -> ScanRequest:
    """Return the direct duration-scan request for the chunk fragment."""

    return ScanRequest.cartesian(
        [(fragment.shot.pulse_duration, DURATION_POINTS)],
        metadata={"demo_name": "host_runtime_counts_shot_chunk_time_scan"},
        execution_policy=ExecutionPolicy(max_points_per_batch=1),
    )


def build_frequency_scan_request(fragment: CountsShotChunkFragment) -> ScanRequest:
    """Return the frequency-scan request for one field value."""

    return ScanRequest.cartesian(
        [(fragment.shot.probe_frequency, FREQUENCY_POINTS)],
        execution_policy=ExecutionPolicy(max_points_per_batch=1),
    )


def build_field_scan_request(fragment: FrequencyCalibrationAtFieldFragment) -> ScanRequest:
    """Return the field-scan request for the outer calibration stage."""

    return ScanRequest.cartesian(
        [(fragment.measurement_point.shot.coil_current, FIELD_POINTS)],
        execution_policy=ExecutionPolicy(max_points_per_batch=1),
    )


HostRuntimeCountsShotChunkTimeScan = make_fragment_prepared_scan_exp(
    CountsShotChunkFragment,
    build_time_scan_request,
)


HostRuntimeCountsFrequencyCalibration = make_fragment_prepared_scan_exp(
    FrequencyCalibrationAtFieldFragment,
    lambda fragment: ScanRequest.single(
        metadata={"demo_name": "host_runtime_counts_frequency_calibration"}
    ),
)


HostRuntimeCountsFieldCalibration = make_fragment_prepared_scan_exp(
    CountsFieldCalibrationRootFragment,
    lambda fragment: ScanRequest.single(
        metadata={"demo_name": "host_runtime_counts_field_calibration"}
    ),
)

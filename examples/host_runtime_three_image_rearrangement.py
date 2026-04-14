"""Three-image rearrangement/spectroscopy repeat example.

This example is a deliberately simple foundation for the more complicated
multi-image ROI-statistics workflows used in the lab:

- ``image0`` records the initially loaded array,
- ``image1`` records a low-entropy rearranged version of that array,
- ``image2`` records the final bright/dark spectroscopy outcome.

The example intentionally keeps the analysis lightweight:

- one shot produces three fluorescence images,
- each image is reduced to one fixed-shape ``(group, roi)`` counts array,
- a first-pass analysis thresholds those counts over repeated visits to the same
  logical point and publishes per-image occupancy probabilities.

That keeps the leaf reusable for later examples that layer:

- repeated-shot reduction,
- conditional binomial statistics,
- and higher-level fits.
"""

from __future__ import annotations

import math
import time

import numpy as np

from examples._roi_condition_stats import (
    Occupied,
    conditional_binomial,
    counts_to_occupancy_stack,
    parse_condition_syntax,
)
from artiq.experiment import *
from ndscan.define import *
from ndscan.runtime.api import *
from ndscan.scan import *

__all__ = [
    "NUM_GROUPS",
    "NUM_IMAGES",
    "POINT_DELAY_S",
    "DEFAULT_NUM_SHOTS",
    "NUM_ROIS",
    "NUM_ROIS_BY_IMAGE",
    "IMAGE_SHAPE",
    "IMAGE_SHAPES",
    "DEFAULT_THRESHOLD_COUNTS",
    "ThreeImageRearrangementShotFragment",
    "ThreeImageRearrangementStatisticsFragment",
    "ThreeImageRearrangementDashboardFragment",
    "build_three_image_rearrangement_statistics_request",
    "HostRuntimeThreeImageRearrangement",
    "HostRuntimeThreeImageRearrangementDashboard",
]


NUM_GROUPS = 6
NUM_IMAGES = 3

POINT_DELAY_S = 0.02

IMAGE_SHAPES = (
    (68, 92),
    (80, 100),
    (76, 96),
)
DEFAULT_NUM_SHOTS = 48

DEFAULT_INITIAL_LOAD_PROBABILITY = 0.5
DEFAULT_RESONANCE_FREQUENCY = 10.0
DEFAULT_SPECTROSCOPY_WIDTH = 0.22
DEFAULT_SPECTROSCOPY_CONTRAST = 0.92
DEFAULT_THRESHOLD_COUNTS = 3500

IMAGE_BACKGROUND_MEAN = 180.0
IMAGE_BRIGHT_COUNTS_MEAN = 1400.0
IMAGE_SPOT_SIGMA = 1.35

GROUP_X_PITCH = 12
GROUP_X_START = 10
ROI_LAYOUT_SPECS_BY_IMAGE = (
    (
        {"y_centre": 22, "height": 4, "width": 4},
        {"y_centre": 46, "height": 6, "width": 5},
    ),
    (
        {"y_centre": 18, "height": 4, "width": 4},
        {"y_centre": 40, "height": 6, "width": 5},
        {"y_centre": 62, "height": 5, "width": 6},
    ),
    (
        {"y_centre": 20, "height": 5, "width": 4},
        {"y_centre": 42, "height": 5, "width": 6},
        {"y_centre": 60, "height": 4, "width": 6},
    ),
)
NUM_ROIS_BY_IMAGE = tuple(len(specs) for specs in ROI_LAYOUT_SPECS_BY_IMAGE)
NUM_ROIS = max(NUM_ROIS_BY_IMAGE)
IMAGE_SHAPE = IMAGE_SHAPES[0]
COMMON_IMAGE12_ROIS = min(NUM_ROIS_BY_IMAGE[1], NUM_ROIS_BY_IMAGE[2])
if COMMON_IMAGE12_ROIS < 2:
    raise ValueError(
        "Images 1 and 2 must share at least two ROI indices for the built-in pair statistics"
    )
DEBUG_DATASET_PREFIX = "debug.imaging"


def _build_rois_for_image(
    image_index: int,
) -> list[list[tuple[int, int, int, int]]]:
    """Return one ``(group, roi)`` layout for one image-specific trap geometry."""

    image_height, image_width = IMAGE_SHAPES[image_index]
    rois: list[list[tuple[int, int, int, int]]] = []
    for group_index in range(NUM_GROUPS):
        x_centre = GROUP_X_START + group_index * GROUP_X_PITCH
        roi_group = []
        for roi_spec in ROI_LAYOUT_SPECS_BY_IMAGE[image_index]:
            width = int(roi_spec["width"])
            height = int(roi_spec["height"])
            y_centre = float(roi_spec["y_centre"])
            x0 = int(round(x_centre - width / 2))
            x1 = x0 + width
            y0 = int(round(y_centre - height / 2))
            y1 = y0 + height
            if x0 < 0 or y0 < 0 or x1 > image_width or y1 > image_height:
                raise ValueError(
                    f"ROI box {(y0, y1, x0, x1)} for image {image_index} exceeds image bounds"
                )
            roi_group.append((y0, y1, x0, x1))
        rois.append(roi_group)
    return rois


DEFAULT_ROIS_IMAGE0 = _build_rois_for_image(0)
DEFAULT_ROIS_IMAGE1 = _build_rois_for_image(1)
DEFAULT_ROIS_IMAGE2 = _build_rois_for_image(2)


def _num_rois_for_image(image_index: int) -> int:
    return NUM_ROIS_BY_IMAGE[image_index]


def _debug_dataset_name(name: str) -> str:
    return f"{DEBUG_DATASET_PREFIX}.{name}"


def _occupancy_probability_name(image_index: int) -> str:
    return f"occupancy_probability_image{image_index}"


def _occupancy_probability_error_name(image_index: int) -> str:
    return f"occupancy_probability_error_image{image_index}"


def _pair_loaded_probability_name() -> str:
    return "pair_loaded_probability_image1"


def _pair_loaded_probability_error_name() -> str:
    return "pair_loaded_probability_error_image1"


def _pair_loaded_probability_by_group_name() -> str:
    return "pair_loaded_probability_image1_by_group"


def _pair_loaded_probability_by_group_error_name() -> str:
    return "pair_loaded_probability_error_image1_by_group"


def _bright_pair_probability_name() -> str:
    return "bright_pair_probability_image2_given_pair_image1"


def _bright_pair_probability_error_name() -> str:
    return "bright_pair_probability_error_image2_given_pair_image1"


def _bright_pair_probability_by_group_name() -> str:
    return "bright_pair_probability_image2_given_pair_image1_by_group"


def _bright_pair_probability_by_group_error_name() -> str:
    return "bright_pair_probability_error_image2_given_pair_image1_by_group"


def _bright_given_bright_probability_by_trap_name() -> str:
    return "bright_probability_image2_given_bright_image1_by_trap"


def _bright_given_bright_probability_by_trap_error_name() -> str:
    return "bright_probability_error_image2_given_bright_image1_by_trap"


def _estimate_probability_from_counts(
    num_successes: np.ndarray, num_shots: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative binomial estimate and standard error per array element."""

    if num_shots <= 0:
        shape = np.shape(num_successes)
        return np.zeros(shape, dtype=float), np.full(shape, float("inf"), dtype=float)
    probability = np.asarray(num_successes, dtype=float) / float(num_shots)
    error = np.sqrt(np.maximum(probability * (1.0 - probability), 0.0) / num_shots)
    error = np.maximum(error, 0.5 / num_shots)
    return probability, error


def _gaussian2d(shape, x0: float, y0: float, sigma: float) -> np.ndarray:
    yy, xx = np.indices(shape)
    gaussian = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))
    total = float(np.sum(gaussian))
    if total > 0.0:
        gaussian /= total
    return gaussian


def _render_image_from_occupancy(
    rois,
    occupied: np.ndarray,
    *,
    rng: np.random.Generator,
    shape=IMAGE_SHAPE,
    bright_counts_mean: float = IMAGE_BRIGHT_COUNTS_MEAN,
    background_mean: float = IMAGE_BACKGROUND_MEAN,
    sigma: float = IMAGE_SPOT_SIGMA,
) -> np.ndarray:
    """Render one fluorescence image from a boolean ``(group, roi)`` occupancy table."""

    image = rng.poisson(background_mean, size=shape).astype(np.int32)
    for group_index, roi_group in enumerate(rois):
        for roi_index, (y0, y1, x0, x1) in enumerate(roi_group):
            if not occupied[group_index, roi_index]:
                continue
            x_centre = 0.5 * (float(x0) + float(x1 - 1))
            y_centre = 0.5 * (float(y0) + float(y1 - 1))
            amplitude = int(rng.poisson(bright_counts_mean))
            if amplitude <= 0:
                continue
            psf = _gaussian2d(shape, x_centre, y_centre, sigma)
            image += rng.poisson(amplitude * psf).astype(np.int32)
    return image


def _sum_counts_in_rois(image: np.ndarray, rois) -> np.ndarray:
    """Return one fixed-shape ``(group, roi)`` counts array from an image."""

    counts = np.empty((len(rois), len(rois[0])), dtype=np.int32)
    for group_index, roi_group in enumerate(rois):
        for roi_index, (y0, y1, x0, x1) in enumerate(roi_group):
            counts[group_index, roi_index] = int(np.sum(image[y0:y1, x0:x1]))
    return counts


def _threshold_counts_to_occupancy(counts: np.ndarray, threshold: int) -> np.ndarray:
    """Infer a boolean occupancy table from ROI counts."""

    return np.asarray(counts, dtype=np.int32) >= int(threshold)


def _plan_left_compaction(
    occupied: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Return the target compacted occupancy and a simple move list.

    The move list entries are ``(roi_index, source_group, target_group)``. This keeps
    the example honest about the control flow a real experiment would follow:

    1. analyse the first image,
    2. derive a rearrangement plan from that measured occupancy,
    3. apply the plan,
    4. take the next image.
    """

    compacted = np.zeros_like(occupied, dtype=bool)
    moves: list[tuple[int, int, int]] = []
    for roi_index in range(occupied.shape[1]):
        source_groups = np.flatnonzero(occupied[:, roi_index]).tolist()
        for target_group, source_group in enumerate(source_groups):
            compacted[target_group, roi_index] = True
            if source_group != target_group:
                moves.append((roi_index, int(source_group), int(target_group)))
    return compacted, moves


def _spectroscopy_dark_probability(
    probe_frequency: float,
    *,
    group_index: int,
    roi_index: int,
    num_rois: int,
    resonance_frequency: float,
    width: float,
    contrast: float,
) -> float:
    """Return one simulated dark-state probability for the spectroscopy image."""

    row_offset = 0.18 * (float(roi_index) - 0.5 * float(max(num_rois - 1, 0)))
    group_offset = 0.035 * (group_index - 0.5 * (NUM_GROUPS - 1))
    centre = resonance_frequency + row_offset + group_offset
    detuning = (probe_frequency - centre) / max(width, 1e-6)
    return float(contrast * math.exp(-0.5 * detuning**2))


def _project_occupancy_to_target_rois(
    occupied: np.ndarray,
    target_num_rois: int,
) -> np.ndarray:
    """Project one ``(group, roi)`` occupancy table onto a target ROI count.

    This deliberately keeps the example simple: ROI indices are matched by position,
    overlapping indices are preserved, extra source ROIs are dropped, and extra target
    ROIs start empty.
    """

    projected = np.zeros((occupied.shape[0], int(target_num_rois)), dtype=bool)
    shared = min(occupied.shape[1], int(target_num_rois))
    if shared > 0:
        projected[:, :shared] = occupied[:, :shared]
    return projected


def _build_repeat_statistics_channels() -> list[ResultChannel]:
    """Return the repeated-shot outputs published by the example."""

    channels: list[ResultChannel] = []
    for image_index in range(NUM_IMAGES):
        image_num_rois = _num_rois_for_image(image_index)
        probability = ArrayChannel(
            _occupancy_probability_name(image_index),
            f"Image {image_index} occupancy probability",
            element_type="float",
            shape=(NUM_GROUPS, image_num_rois),
            dim_names=("group", "roi"),
        )
        channels.extend(
            [
                probability,
                ArrayChannel(
                    _occupancy_probability_error_name(image_index),
                    f"Image {image_index} occupancy probability error",
                    element_type="float",
                    shape=(NUM_GROUPS, image_num_rois),
                    dim_names=("group", "roi"),
                    display_hints={"error_bar_for": probability.path},
                ),
            ]
        )

    pair_loaded = FloatChannel(
        _pair_loaded_probability_name(),
        "Probability both traps in one group are occupied in image 1",
    )
    channels.extend(
        [
            pair_loaded,
            FloatChannel(
                _pair_loaded_probability_error_name(),
                "Error on pair-loaded probability in image 1",
                display_hints={"error_bar_for": pair_loaded.path},
            ),
            ArrayChannel(
                _pair_loaded_probability_by_group_name(),
                "Per-group probability both traps are occupied in image 1",
                element_type="float",
                shape=(NUM_GROUPS,),
                dim_names=("group",),
            ),
            ArrayChannel(
                _pair_loaded_probability_by_group_error_name(),
                "Per-group error on pair-loaded probability in image 1",
                element_type="float",
                shape=(NUM_GROUPS,),
                dim_names=("group",),
                display_hints={"error_bar_for": _pair_loaded_probability_by_group_name()},
            ),
        ]
    )

    bright_pair = FloatChannel(
        _bright_pair_probability_name(),
        "Probability a rearranged pair is bright in image 2",
    )
    channels.extend(
        [
            bright_pair,
            FloatChannel(
                _bright_pair_probability_error_name(),
                "Error on bright-pair probability in image 2",
                display_hints={"error_bar_for": bright_pair.path},
            ),
            ArrayChannel(
                _bright_pair_probability_by_group_name(),
                "Per-group probability a rearranged pair is bright in image 2",
                element_type="float",
                shape=(NUM_GROUPS,),
                dim_names=("group",),
            ),
            ArrayChannel(
                _bright_pair_probability_by_group_error_name(),
                "Per-group error on bright-pair probability in image 2",
                element_type="float",
                shape=(NUM_GROUPS,),
                dim_names=("group",),
                display_hints={"error_bar_for": _bright_pair_probability_by_group_name()},
            ),
        ]
    )

    channels.extend(
        [
            ArrayChannel(
                _bright_given_bright_probability_by_trap_name(),
                "Per-trap probability of brightness in image 2 given brightness in image 1",
                element_type="float",
                shape=(NUM_GROUPS, COMMON_IMAGE12_ROIS),
                dim_names=("group", "roi"),
            ),
            ArrayChannel(
                _bright_given_bright_probability_by_trap_error_name(),
                "Per-trap error on brightness in image 2 given brightness in image 1",
                element_type="float",
                shape=(NUM_GROUPS, COMMON_IMAGE12_ROIS),
                dim_names=("group", "roi"),
                display_hints={
                    "error_bar_for": _bright_given_bright_probability_by_trap_name()
                },
            ),
        ]
    )
    return channels


def _clone_result_channel(fragment: ExpFragment, channel: ResultChannel) -> ResultChannel:
    """Create one matching result channel on ``fragment``."""

    common_kwargs = {
        "display_hints": dict(channel.display_hints),
        "save_by_default": channel.save_by_default,
    }
    if isinstance(channel, ArrayChannel):
        return fragment.setattr_result(
            channel.path,
            ArrayChannel,
            channel.description,
            element_type=channel.element_type,
            shape=channel.shape,
            dim_names=channel.dim_names,
            min=channel.min,
            max=channel.max,
            unit=channel.unit,
            scale=channel.scale,
            **common_kwargs,
        )
    if isinstance(channel, FloatChannel):
        return fragment.setattr_result(
            channel.path,
            FloatChannel,
            channel.description,
            min=channel.min,
            max=channel.max,
            unit=channel.unit,
            scale=channel.scale,
            **common_kwargs,
        )
    if isinstance(channel, IntChannel):
        return fragment.setattr_result(
            channel.path,
            IntChannel,
            channel.description,
            min=channel.min,
            max=channel.max,
            unit=channel.unit,
            scale=channel.scale,
            **common_kwargs,
        )
    raise TypeError(f"Unsupported result channel type: {type(channel)!r}")


class ThreeImageRearrangementShotFragment(ExpFragment):
    """One simulated shot with initial load, rearrangement, and final spectroscopy."""

    def build_fragment(self):
        self.setattr_param(
            "probe_frequency",
            FloatParam,
            "Probe frequency",
            default=DEFAULT_RESONANCE_FREQUENCY,
            unit="MHz",
        )
        self.setattr_param(
            "initial_load_probability",
            FloatParam,
            "Initial load probability",
            default=DEFAULT_INITIAL_LOAD_PROBABILITY,
            min=0.0,
            max=1.0,
        )
        self.setattr_param(
            "resonance_frequency",
            FloatParam,
            "Resonance frequency",
            default=DEFAULT_RESONANCE_FREQUENCY,
            unit="MHz",
        )
        self.setattr_param(
            "spectroscopy_width",
            FloatParam,
            "Spectroscopy width",
            default=DEFAULT_SPECTROSCOPY_WIDTH,
            unit="MHz",
            min=0.01,
        )
        self.setattr_param(
            "spectroscopy_contrast",
            FloatParam,
            "Spectroscopy contrast",
            default=DEFAULT_SPECTROSCOPY_CONTRAST,
            min=0.0,
            max=1.0,
        )
        self.setattr_param(
            "threshold_counts",
            IntParam,
            "Threshold counts",
            default=DEFAULT_THRESHOLD_COUNTS,
            min=1,
        )

        self.setattr_result(
            "counts_image0",
            ArrayChannel,
            "Initial-load ROI counts",
            element_type="int",
            shape=(NUM_GROUPS, _num_rois_for_image(0)),
            dim_names=("group", "roi"),
            unit="cts",
            scale=1.0,
        )
        self.setattr_result(
            "counts_image1",
            ArrayChannel,
            "Rearranged ROI counts",
            element_type="int",
            shape=(NUM_GROUPS, _num_rois_for_image(1)),
            dim_names=("group", "roi"),
            unit="cts",
            scale=1.0,
        )
        self.setattr_result(
            "counts_image2",
            ArrayChannel,
            "Spectroscopy ROI counts",
            element_type="int",
            shape=(NUM_GROUPS, _num_rois_for_image(2)),
            dim_names=("group", "roi"),
            unit="cts",
            scale=1.0,
        )
        self.setattr_result(
            "image0",
            ArrayChannel,
            "Initial-load fluorescence image",
            element_type="int",
            shape=IMAGE_SHAPES[0],
            dim_names=("y", "x"),
            display_hints={"priority": -5},
            unit="cts",
            scale=1.0,
        )
        self.setattr_result(
            "image1",
            ArrayChannel,
            "Rearranged fluorescence image",
            element_type="int",
            shape=IMAGE_SHAPES[1],
            dim_names=("y", "x"),
            display_hints={"priority": -5},
            unit="cts",
            scale=1.0,
        )
        self.setattr_result(
            "image2",
            ArrayChannel,
            "Spectroscopy fluorescence image",
            element_type="int",
            shape=IMAGE_SHAPES[2],
            dim_names=("y", "x"),
            display_hints={"priority": -5},
            unit="cts",
            scale=1.0,
        )

        self._rng = np.random.default_rng(seed=20260410)
        self._rois_by_image = {
            0: [[tuple(bounds) for bounds in group] for group in DEFAULT_ROIS_IMAGE0],
            1: [[tuple(bounds) for bounds in group] for group in DEFAULT_ROIS_IMAGE1],
            2: [[tuple(bounds) for bounds in group] for group in DEFAULT_ROIS_IMAGE2],
        }

    def host_setup(self):
        for image_index, rois in self._rois_by_image.items():
            self.set_dataset(
                _debug_dataset_name(f"rois_image{image_index}"),
                rois,
                broadcast=True,
                archive=False,
            )
        self.set_dataset(
            _debug_dataset_name("schema_version"),
            1,
            broadcast=True,
            archive=False,
        )
        super().host_setup()

    def run_once(self):
        if POINT_DELAY_S > 0.0:
            time.sleep(POINT_DELAY_S)

        # Stage 1: prepare/load an initial random array and acquire the first image.
        loaded0 = (
            self._rng.random((NUM_GROUPS, _num_rois_for_image(0)))
            < float(self.initial_load_probability.get())
        )
        image0 = _render_image_from_occupancy(
            self._rois_by_image[0],
            loaded0,
            rng=self._rng,
            shape=IMAGE_SHAPES[0],
        )
        counts0 = _sum_counts_in_rois(image0, self._rois_by_image[0])

        # Stage 2: infer occupancy from image0 and derive the rearrangement actions
        # from the measured counts rather than the hidden true occupancy.
        inferred0 = _threshold_counts_to_occupancy(
            counts0, int(self.threshold_counts.get())
        )
        occupied1, rearrangement_moves = _plan_left_compaction(inferred0)
        occupied1 = _project_occupancy_to_target_rois(
            occupied1,
            _num_rois_for_image(1),
        )

        # Stage 3: apply the rearrangement plan and verify it with image1.
        image1 = _render_image_from_occupancy(
            self._rois_by_image[1],
            occupied1,
            rng=self._rng,
            shape=IMAGE_SHAPES[1],
        )
        counts1 = _sum_counts_in_rois(image1, self._rois_by_image[1])

        # Stage 4: run spectroscopy on the rearranged array and read out image2.
        occupied2_input = _project_occupancy_to_target_rois(
            occupied1,
            _num_rois_for_image(2),
        )
        dark_probability = np.empty((NUM_GROUPS, _num_rois_for_image(2)), dtype=float)
        for group_index in range(NUM_GROUPS):
            for roi_index in range(_num_rois_for_image(2)):
                dark_probability[group_index, roi_index] = _spectroscopy_dark_probability(
                    float(self.probe_frequency.get()),
                    group_index=group_index,
                    roi_index=roi_index,
                    num_rois=_num_rois_for_image(2),
                    resonance_frequency=float(self.resonance_frequency.get()),
                    width=float(self.spectroscopy_width.get()),
                    contrast=float(self.spectroscopy_contrast.get()),
                )

        occupied2 = occupied2_input & (
            self._rng.random((NUM_GROUPS, _num_rois_for_image(2))) >= dark_probability
        )
        image2 = _render_image_from_occupancy(
            self._rois_by_image[2],
            occupied2,
            rng=self._rng,
            shape=IMAGE_SHAPES[2],
        )
        counts2 = _sum_counts_in_rois(image2, self._rois_by_image[2])

        debug_values = {
            "image0": image0,
            "image1": image1,
            "image2": image2,
            "counts_image0": counts0,
            "counts_image1": counts1,
            "counts_image2": counts2,
            "threshold_counts": int(self.threshold_counts.get()),
            "occupied_image0": inferred0.astype(np.int8),
            "occupied_image1": _threshold_counts_to_occupancy(
                counts1, int(self.threshold_counts.get())
            ).astype(np.int8),
            "occupied_image2": _threshold_counts_to_occupancy(
                counts2, int(self.threshold_counts.get())
            ).astype(np.int8),
            "rearranged_target_image1": occupied1.astype(np.int8),
            "rearrangement_moves": rearrangement_moves,
        }
        for name, value in debug_values.items():
            self.set_dataset(
                _debug_dataset_name(name),
                value,
                broadcast=True,
                archive=False,
            )

        self.counts_image0.push(counts0)
        self.counts_image1.push(counts1)
        self.counts_image2.push(counts2)
        self.image0.push(image0)
        self.image1.push(image1)
        self.image2.push(image2)

    def get_default_analyses(self):
        return [
            CustomAnalysis(
                [],
                self._analyse_image_statistics,
                analysis_results=_build_repeat_statistics_channels(),
                online_fn=self._analyse_image_statistics,
                online_analysis_identifier="image_statistics",
            )
        ]

    def _analyse_image_statistics(self, axis_values, result_values, analysis_results):
        del axis_values, analysis_results

        threshold = int(self.threshold_counts.get())
        counts_by_image = [
            np.asarray(result_values[channel])
            for channel in (self.counts_image0, self.counts_image1, self.counts_image2)
        ]

        outputs: dict[str, object] = {}
        for image_index, counts in enumerate(counts_by_image):
            if counts.size == 0:
                shape = (NUM_GROUPS, _num_rois_for_image(image_index))
                outputs[_occupancy_probability_name(image_index)] = np.zeros(shape, dtype=float)
                outputs[_occupancy_probability_error_name(image_index)] = np.full(
                    shape, float("inf"), dtype=float
                )
                continue
            occupied = counts >= threshold
            probability, probability_error = _estimate_probability_from_counts(
                np.sum(occupied, axis=0),
                occupied.shape[0],
            )
            outputs[_occupancy_probability_name(image_index)] = probability
            outputs[_occupancy_probability_error_name(image_index)] = probability_error

        occupancy_stack = counts_to_occupancy_stack(counts_by_image, threshold=threshold)
        pair_image1 = parse_condition_syntax("1[0,1]")
        bright_pair_image2 = parse_condition_syntax("2[0,1]")

        pair_loaded = conditional_binomial(occupancy_stack, event=pair_image1)
        bright_pair = conditional_binomial(
            occupancy_stack,
            given=pair_image1,
            event=bright_pair_image2,
        )

        outputs[_pair_loaded_probability_name()] = pair_loaded.pooled_probability
        outputs[_pair_loaded_probability_error_name()] = pair_loaded.pooled_probability_error
        outputs[_pair_loaded_probability_by_group_name()] = pair_loaded.probability_by_group
        outputs[_pair_loaded_probability_by_group_error_name()] = (
            pair_loaded.probability_error_by_group
        )

        outputs[_bright_pair_probability_name()] = bright_pair.pooled_probability
        outputs[_bright_pair_probability_error_name()] = bright_pair.pooled_probability_error
        outputs[_bright_pair_probability_by_group_name()] = bright_pair.probability_by_group
        outputs[_bright_pair_probability_by_group_error_name()] = (
            bright_pair.probability_error_by_group
        )

        bright_given_bright_by_trap = np.empty((NUM_GROUPS, COMMON_IMAGE12_ROIS), dtype=float)
        bright_given_bright_error_by_trap = np.empty((NUM_GROUPS, COMMON_IMAGE12_ROIS), dtype=float)
        for roi_index in range(COMMON_IMAGE12_ROIS):
            bright_given_bright = conditional_binomial(
                occupancy_stack,
                given=Occupied(1, roi_index),
                event=Occupied(2, roi_index),
            )
            bright_given_bright_by_trap[:, roi_index] = (
                bright_given_bright.probability_by_group
            )
            bright_given_bright_error_by_trap[:, roi_index] = (
                bright_given_bright.probability_error_by_group
            )

        outputs[_bright_given_bright_probability_by_trap_name()] = (
            bright_given_bright_by_trap
        )
        outputs[_bright_given_bright_probability_by_trap_error_name()] = (
            bright_given_bright_error_by_trap
        )
        return AnalysisFeedback(outputs=outputs)


class ThreeImageRearrangementStatisticsFragment(ExpFragment):
    """Repeat one three-image shot and republish occupancy statistics as point data."""

    def build_fragment(self):
        self.repeat_scan = setattr_prepared_child_scan(
            self,
            "shot",
            ThreeImageRearrangementShotFragment,
            scan_name="repeat_scan",
            extra_metadata={
                "analysis_note": "threshold three-image ROI counts into occupancy probabilities"
            },
        )
        self.setattr_param(
            "shots_per_point",
            IntParam,
            "Shots per point",
            default=DEFAULT_NUM_SHOTS,
            min=1,
        )
        self._stat_channels: dict[str, ResultChannel] = {}
        for channel in _build_repeat_statistics_channels():
            self._stat_channels[channel.path] = _clone_result_channel(self, channel)

    def run_once(self):
        num_shots = int(self.shots_per_point.get())
        self.repeat_scan.configure(
            ScanRequest.single(
                execution_policy=ExecutionPolicy(max_points_per_batch=min(num_shots, 16))
            ).with_repeats(repeats=num_shots)
        )
        outputs = self.repeat_scan.execute()
        for name, channel in self._stat_channels.items():
            channel.push(outputs[name])


class ThreeImageRearrangementDashboardFragment(ThreeImageRearrangementStatisticsFragment):
    """Dashboard-friendly wrapper exposing the key scan/control parameters by default."""

    def get_always_shown_params(self):
        shown = super().get_always_shown_params()
        shown += [
            self.shot.probe_frequency,
            self.shot.initial_load_probability,
            self.shot.resonance_frequency,
            self.shot.spectroscopy_width,
            self.shot.spectroscopy_contrast,
            self.shot.threshold_counts,
        ]
        return shown


def build_three_image_rearrangement_statistics_request(
    fragment: ThreeImageRearrangementStatisticsFragment,
) -> ScanRequest:
    """Run one logical point whose data comes from repeated raw three-image shots."""

    del fragment

    return ScanRequest.single(
        metadata={"demo_name": "host_runtime_three_image_rearrangement"},
    )


HostRuntimeThreeImageRearrangement = make_fragment_prepared_scan_exp(
    ThreeImageRearrangementStatisticsFragment,
    build_three_image_rearrangement_statistics_request,
)


HostRuntimeThreeImageRearrangementDashboard = make_fragment_prepared_dashboard_scan_exp(
    ThreeImageRearrangementDashboardFragment
)

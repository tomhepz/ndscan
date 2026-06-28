"""Example offline plotting script for a saved three-image rearrangement snapshot.

Edit the variables in the configuration section below, then run the file directly:

    python examples/plot_three_image_rearrangement_snapshot.py

The script demonstrates three things:

- average the saved raw image channels over the repeated-shot child site and draw the
  ROI boxes from the persisted imaging blob
- show threshold-debug histograms from the raw saved ROI counts
- plot one saved live-analysis statistic alongside the same statistic recomputed
  offline from the raw repeated-shot counts
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from examples._roi_condition_stats import (
    conditional_binomial,
    counts_to_occupancy_stack,
    parse_condition_syntax,
)
from examples.lab_offline_results_helpers import LabNdscanRun
from ndscan.results.scan_site_reader import ScanSiteData

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set this to the prepared-runtime HDF5 file you want to inspect.
SNAPSHOT_PATH = "/home/lab/artiq-files/dnamic-lab/results/2026-04-15/12/000002761-ThreeImageRearrangementDashboardFragment.h5"

# Site containing the raw repeated-shot image/count channels.
IMAGE_SITE_PATH = ("repeat_scan",)

# Site containing the 1D aggregated statistics you want to plot. For the
# three-image rearrangement example this is usually the root site.
LINE_PLOT_SITE_PATH = ()

# Persisted site blob describing the imaging schema.
IMAGING_BLOB_NAME = "lab.imaging_readout"

# Saved live-analysis statistic to compare against an offline recomputation.
# Examples:
# - x: "shot/probe_frequency"
# - y: "bright_pair_probability_image2_given_pair_image1"
COMPARISON_X = "shot/probe_frequency"
COMPARISON_SAVED_Y = "bright_pair_probability_image2_given_pair_image1"

# Offline recomputation rule, using the same boolean syntax as the live helper.
# Here:
# - given a bright pair in image 1
# - what is the probability of a bright pair in image 2
OFFLINE_GIVEN = "1[0,1]"
OFFLINE_EVENT = "2[0,1]"

# Histogram debug settings for the raw thresholding step. One figure is produced per
# image; subplot columns are groups and rows are ROI indices.
HISTOGRAM_BINS = 48


def _image_index_from_logical_name(name: str) -> int:
    if not name.startswith("image"):
        raise ValueError(f"Expected imaging blob key like 'image1', got {name!r}")
    return int(name[len("image") :])


def _sorted_image_entries(
    blob: Mapping[str, object],
) -> list[tuple[int, str, Mapping[str, object]]]:
    images = blob.get("images")
    if not isinstance(images, Mapping):
        raise ValueError(
            f"Blob {blob.get('namespace', '<unknown>')!r} does not define an "
            "'images' mapping"
        )

    entries: list[tuple[int, str, Mapping[str, object]]] = []
    for logical_name, image_spec in images.items():
        if not isinstance(image_spec, Mapping):
            raise ValueError(f"Blob entry {logical_name!r} is not a mapping")
        entries.append(
            (_image_index_from_logical_name(str(logical_name)), str(logical_name), image_spec)
        )
    return sorted(entries, key=lambda item: item[0])


@dataclass(frozen=True)
class ThreeImageReadout:
    """Script-local interpretation of the example's ``lab.imaging_readout`` blob."""

    site: ScanSiteData
    blob_name: str
    blob: Mapping[str, object]

    @classmethod
    def from_site(
        cls,
        site: ScanSiteData,
        *,
        blob_name: str,
    ) -> "ThreeImageReadout":
        blob = site.require_metadata_blob(blob_name)
        if not isinstance(blob, Mapping):
            raise ValueError(f"Metadata blob {blob_name!r} is not a mapping")
        return cls(site=site, blob_name=blob_name, blob=blob)

    def image_entries(self) -> list[tuple[int, str, Mapping[str, object]]]:
        return _sorted_image_entries(self.blob)

    def threshold_value(self) -> int:
        occupancy_rule = self.blob.get("occupancy_rule")
        if not isinstance(occupancy_rule, Mapping):
            raise ValueError("Imaging blob does not define an 'occupancy_rule' mapping")

        threshold_fqn = occupancy_rule.get("threshold_parameter_fqn")
        if not isinstance(threshold_fqn, str):
            raise ValueError(
                "Imaging blob occupancy_rule does not define 'threshold_parameter_fqn'"
            )

        matches = [
            entry["value"]
            for entry in self.site.fixed_parameters.values()
            if isinstance(entry, Mapping)
            and isinstance(entry.get("param"), Mapping)
            and entry["param"].get("fqn") == threshold_fqn
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Could not resolve unique threshold parameter {threshold_fqn!r} on "
                f"site {'/'.join(self.site.path) or '<root>'}"
            )
        return int(matches[0])

    def image_channel(self, image_spec: Mapping[str, object]) -> str:
        return str(image_spec["image_channel"])

    def counts_channel(self, image_spec: Mapping[str, object]) -> str:
        return str(image_spec["counts_channel"])

    def average_image_payloads(self) -> list[dict[str, object]]:
        return [
            {
                "name": logical_name,
                "image_channel": self.image_channel(image_spec),
                "average_image": np.asarray(
                    np.mean(self.site.series(self.image_channel(image_spec)), axis=0),
                    dtype=float,
                ),
                "rois": image_spec["rois"],
            }
            for _, logical_name, image_spec in self.image_entries()
        ]

    def counts_by_image_for_segment(self, start_index: int, stop_index: int) -> list[np.ndarray]:
        segment_data = self.site.slice_raw_points(start_index, stop_index)
        return [
            np.asarray(
                segment_data[
                    self.site.require_channel_storage_key(self.counts_channel(image_spec))
                ]
            )
            for _, _, image_spec in self.image_entries()
        ]

    def threshold_histogram_payloads(self) -> list[dict[str, object]]:
        threshold = self.threshold_value()
        payloads: list[dict[str, object]] = []
        for image_index, logical_name, image_spec in self.image_entries():
            counts = np.asarray(
                self.site.series(self.counts_channel(image_spec))
            )
            if counts.ndim != 3:
                raise ValueError(
                    f"Counts channel {self.counts_channel(image_spec)!r} for "
                    f"{logical_name} has shape {counts.shape}; expected "
                    "(points, group, roi)"
                )
            num_points, num_groups, num_rois = counts.shape
            samples_by_roi_group: list[list[np.ndarray]] = []
            bright_fraction_by_roi_group = np.empty((num_rois, num_groups), dtype=float)
            for roi_index in range(num_rois):
                roi_samples = []
                for group_index in range(num_groups):
                    samples = np.asarray(counts[:, group_index, roi_index], dtype=float)
                    roi_samples.append(samples)
                    bright_fraction_by_roi_group[roi_index, group_index] = float(
                        np.mean(samples >= threshold)
                    )
                samples_by_roi_group.append(roi_samples)
            payloads.append(
                {
                    "image_index": image_index,
                    "name": logical_name,
                    "counts_channel": self.counts_channel(image_spec),
                    "num_points": num_points,
                    "num_groups": num_groups,
                    "num_rois": num_rois,
                    "samples_by_roi_group": samples_by_roi_group,
                    "bright_fraction_by_roi_group": bright_fraction_by_roi_group,
                    "threshold": threshold,
                }
            )
        return payloads


def build_saved_vs_recomputed_probability_payload(
    parent_site: ScanSiteData,
    readout: ThreeImageReadout,
    *,
    x: str | None,
    saved_y: str,
    given_syntax: str,
    event_syntax: str,
) -> dict[str, Any]:
    x_path = parent_site.choose_default_x_path() if x is None else x
    if x_path is None:
        raise ValueError(
            f"Site {'/'.join(parent_site.path) or '<root>'} does not have a default "
            "x path"
        )

    x_values = np.asarray(parent_site.series(x_path), dtype=float)
    saved_y_values = np.asarray(parent_site.series(saved_y), dtype=float)

    threshold = readout.threshold_value()
    given_condition = parse_condition_syntax(given_syntax)
    event_condition = parse_condition_syntax(event_syntax)

    recomputed_y_values = np.empty(parent_site.num_points, dtype=float)
    recomputed_y_errors = np.empty(parent_site.num_points, dtype=float)
    num_selected = np.empty(parent_site.num_points, dtype=int)
    num_successes = np.empty(parent_site.num_points, dtype=int)

    for point_index in range(parent_site.num_points):
        segments = readout.site.segments_for_parent_point(point_index)
        if len(segments) != 1:
            raise ValueError(
                f"Expected exactly one repeat segment for parent point {point_index}, "
                f"got {len(segments)}"
            )
        segment = segments[0]
        counts_by_image = readout.counts_by_image_for_segment(
            segment.start_index,
            segment.stop_index,
        )
        occupancy = counts_to_occupancy_stack(counts_by_image, threshold=threshold)
        result = conditional_binomial(
            occupancy,
            given=given_condition,
            event=event_condition,
        )
        recomputed_y_values[point_index] = result.pooled_probability
        recomputed_y_errors[point_index] = result.pooled_probability_error
        num_selected[point_index] = result.pooled_num_selected
        num_successes[point_index] = result.pooled_num_successes

    order = np.argsort(x_values)
    x_sorted = np.asarray(x_values[order], dtype=float)
    saved_y_sorted = np.asarray(saved_y_values[order], dtype=float)
    recomputed_sorted = np.asarray(recomputed_y_values[order], dtype=float)

    return {
        "x_path": x_path,
        "saved_y_path": saved_y,
        "given_syntax": given_syntax,
        "event_syntax": event_syntax,
        "threshold": threshold,
        "x_values": x_sorted,
        "saved_y_values": saved_y_sorted,
        "recomputed_y_values": recomputed_sorted,
        "recomputed_y_errors": np.asarray(recomputed_y_errors[order], dtype=float),
        "difference": saved_y_sorted - recomputed_sorted,
        "num_selected": np.asarray(num_selected[order], dtype=int),
        "num_successes": np.asarray(num_successes[order], dtype=int),
    }


def plot_average_images_with_rois(
    readout: ThreeImageReadout,
) -> tuple[plt.Figure, np.ndarray]:
    payloads = readout.average_image_payloads()
    if not payloads:
        raise ValueError("No images were defined in the imaging blob")

    figure, axes = plt.subplots(
        1,
        len(payloads),
        figsize=(4.5 * len(payloads), 4.8),
        constrained_layout=True,
    )
    if len(payloads) == 1:
        axes = np.asarray([axes])

    for axis, payload in zip(axes, payloads, strict=True):
        average_image = np.asarray(payload["average_image"], dtype=float)
        axis.imshow(average_image, origin="upper", interpolation="nearest", cmap="magma")
        for group_rois in payload["rois"]:
            for y0, y1, x0, x1 in group_rois:
                axis.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor="white",
                        linewidth=1.2,
                    )
                )
        axis.set_title(str(payload["name"]))
        axis.set_xlabel("x (pixels)")
        axis.set_ylabel("y (pixels)")
        axis.set_aspect("equal")

    return figure, axes


def plot_saved_vs_recomputed_probability(
    parent_site: ScanSiteData,
    readout: ThreeImageReadout,
    *,
    x: str | None,
    saved_y: str,
    given_syntax: str,
    event_syntax: str,
) -> tuple[plt.Figure, np.ndarray]:
    payload = build_saved_vs_recomputed_probability_payload(
        parent_site,
        readout,
        x=x,
        saved_y=saved_y,
        given_syntax=given_syntax,
        event_syntax=event_syntax,
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )

    axes[0].plot(
        payload["x_values"],
        payload["saved_y_values"],
        marker="o",
        linewidth=1.6,
        label=f"Saved live result: {payload['saved_y_path']}",
    )
    axes[0].plot(
        payload["x_values"],
        payload["recomputed_y_values"],
        marker="x",
        linestyle="--",
        linewidth=1.4,
        label=(
            "Offline recompute: "
            f"P({payload['event_syntax']} | {payload['given_syntax']})"
        ),
    )
    axes[0].set_ylabel("probability")
    axes[0].set_title("/".join(parent_site.path) or "root")
    axes[0].legend()

    axes[1].axhline(0.0, color="0.4", linewidth=1.0, linestyle=":")
    axes[1].plot(
        payload["x_values"],
        payload["difference"],
        marker="o",
        color="tab:green",
        linewidth=1.2,
    )
    axes[1].set_xlabel(str(payload["x_path"]))
    axes[1].set_ylabel("saved - offline")

    figure.suptitle(
        "Saved live statistic vs offline recomputation "
        f"(threshold = {payload['threshold']})"
    )
    return figure, axes


def plot_threshold_histograms(
    readout: ThreeImageReadout,
    *,
    bins: int = 48,
) -> list[tuple[plt.Figure, np.ndarray]]:
    """Plot one histogram grid per image.

    Grid rows are ROI indices and columns are groups. Counts are pooled over the
    repeated child site's flat point list, so for a parent scan this includes both
    scan points and repeats.
    """

    payloads = readout.threshold_histogram_payloads()
    if not payloads:
        raise ValueError("No images were found for threshold histogram plotting")

    figures_and_axes: list[tuple[plt.Figure, np.ndarray]] = []
    for payload in payloads:
        num_rois = int(payload["num_rois"])
        num_groups = int(payload["num_groups"])
        figure, axes = plt.subplots(
            num_rois,
            num_groups,
            figsize=(2.0 * num_groups, 1.75 * num_rois),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes_grid = np.asarray(axes, dtype=object).reshape(num_rois, num_groups)

        samples_by_roi_group = payload["samples_by_roi_group"]
        bright_fraction = np.asarray(payload["bright_fraction_by_roi_group"], dtype=float)
        threshold = float(payload["threshold"])

        for roi_index in range(num_rois):
            for group_index in range(num_groups):
                axis = axes_grid[roi_index, group_index]
                samples = np.asarray(
                    samples_by_roi_group[roi_index][group_index],
                    dtype=float,
                )
                axis.hist(samples, bins=bins, color="0.55", alpha=0.85)
                axis.axvline(
                    threshold,
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.2,
                )
                axis.text(
                    0.03,
                    0.93,
                    f"p={bright_fraction[roi_index, group_index]:.2f}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.75,
                    },
                )
                if roi_index == 0:
                    axis.set_title(f"group {group_index}", fontsize=9)
                if group_index == 0:
                    axis.set_ylabel(f"roi {roi_index}\ncount")
                if roi_index == num_rois - 1:
                    axis.set_xlabel("ROI counts")

        figure.suptitle(
            f"{payload['name']} threshold histograms "
            f"(threshold = {payload['threshold']}, points = {payload['num_points']})"
        )
        figures_and_axes.append((figure, axes_grid))
    return figures_and_axes


def main() -> None:
    run = LabNdscanRun.open(SNAPSHOT_PATH)

    image_readout = ThreeImageReadout.from_site(
        run.site(IMAGE_SITE_PATH).site,
        blob_name=IMAGING_BLOB_NAME,
    )
    image_figure, _ = plot_average_images_with_rois(image_readout)
    image_figure.suptitle("/".join(image_readout.site.path) or "root")

    plot_threshold_histograms(
        image_readout,
        bins=HISTOGRAM_BINS,
    )

    line_plot_site = run.site(LINE_PLOT_SITE_PATH).site
    plot_saved_vs_recomputed_probability(
        line_plot_site,
        image_readout,
        x=COMPARISON_X,
        saved_y=COMPARISON_SAVED_Y,
        given_syntax=OFFLINE_GIVEN,
        event_syntax=OFFLINE_EVENT,
    )

    plt.show()


if __name__ == "__main__":
    main()

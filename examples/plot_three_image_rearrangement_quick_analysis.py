"""Short lab-book-style offline analysis for the three-image example.

This is intentionally a normal Python script rather than a CLI. Edit the variables at
the top, run the file, and then add whatever ad hoc fits, annotations, extra points, or
figure styling are useful for the lab book.

The script only uses the generic helpers in ``lab_offline_results_helpers.py``:

- open a prepared-runtime HDF5 file,
- pull saved series arrays by semantic path,
- split one array-valued probability by group and plot several traces.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples._roi_condition_stats import (
    conditional_binomial,
    counts_to_occupancy_stack,
    parse_condition_syntax,
)
from examples.lab_offline_results_helpers import LabNdscanRun
from examples.plot_three_image_rearrangement_snapshot import ThreeImageReadout

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SNAPSHOT_PATH = "/home/lab/artiq-files/dnamic-lab/results/2026-04-15/12/000002761-ThreeImageRearrangementDashboardFragment.h5"

# The root site contains the parent scan axis and republished statistics.
SITE_PATH = ()

# The repeated child site contains the raw counts/images needed for ad hoc conditions.
IMAGE_SITE_PATH = ("repeat_scan",)
IMAGING_BLOB_NAME = "lab.imaging_readout"

X_PATH = "shot/probe_frequency"

AVERAGE_Y_PATH = "bright_pair_probability_image2_given_pair_image1"
AVERAGE_YERR_PATH = "bright_pair_probability_error_image2_given_pair_image1"

GROUP_Y_PATH = "bright_pair_probability_image2_given_pair_image1_by_group"
GROUP_YERR_PATH = "bright_pair_probability_error_image2_given_pair_image1_by_group"
GROUP_INDICES = range(4)

# Optional x-axis transform for lab-book plots. For example, set X_OFFSET to the
# carrier frequency and X_SCALE to 1e3 to plot detuning in kHz.
X_OFFSET = None
X_SCALE = 1.0
X_LABEL = X_PATH

# Edit freely for lab-book markers, cooling points, fit centres, etc.
VERTICAL_MARKERS = [
    # (10.0, "resonance"),
]

# Ad hoc offline-only condition. This does not need to have been saved by the live
# analysis. Leave ADHOC_GIVEN empty/None to compute an unconditional probability.
ADHOC_GIVEN = "1[0]"
ADHOC_EVENT = "2[0]"
ADHOC_GROUP_INDICES = range(4)


def _plot_x(raw_x: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_x, dtype=float)
    if X_OFFSET is not None:
        values = values - float(X_OFFSET)
    return values * float(X_SCALE)


def _apply_common_axis_markup(axis: plt.Axes, *, show_xlabel: bool = True) -> None:
    for x_value, label in VERTICAL_MARKERS:
        axis.axvline(_plot_x(np.asarray([x_value]))[0], color="0.25", linestyle="--", lw=1)
        if label:
            axis.text(
                _plot_x(np.asarray([x_value]))[0],
                0.98,
                label,
                transform=axis.get_xaxis_transform(),
                ha="right",
                va="top",
                rotation=90,
                fontsize=8,
            )
    if show_xlabel:
        axis.set_xlabel(X_LABEL)
    axis.set_ylim(-0.05, 1.05)


def plot_average_trace(site) -> tuple[plt.Figure, plt.Axes]:
    x_values = np.asarray(site.series(X_PATH), dtype=float)
    y_values = np.asarray(site.series(AVERAGE_Y_PATH), dtype=float)
    y_errors = np.asarray(site.series(AVERAGE_YERR_PATH), dtype=float)
    order = np.argsort(x_values)

    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.errorbar(
        _plot_x(x_values[order]),
        y_values[order],
        yerr=y_errors[order],
        marker="o",
        linestyle="",
        color="k",
        capsize=2,
        label="average",
    )
    axis.set_ylabel("probability")
    axis.set_title(AVERAGE_Y_PATH)
    _apply_common_axis_markup(axis)
    axis.legend()
    return figure, axis


def plot_group_traces(site) -> tuple[plt.Figure, plt.Axes]:
    x_values = np.asarray(site.series(X_PATH), dtype=float)
    order = np.argsort(x_values)
    x_plot = _plot_x(x_values[order])

    group_values = site.split_array_series(
        GROUP_Y_PATH,
        axis=0,
        indices=GROUP_INDICES,
    )
    group_errors = site.split_array_series(
        GROUP_YERR_PATH,
        axis=0,
        indices=GROUP_INDICES,
    )

    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for group_index in GROUP_INDICES:
        axis.errorbar(
            x_plot,
            np.asarray(group_values[group_index])[order],
            yerr=np.asarray(group_errors[group_index])[order],
            marker="o",
            linestyle="",
            capsize=2,
            label=f"group {group_index}",
        )

    axis.set_ylabel("probability")
    axis.set_title(GROUP_Y_PATH)
    _apply_common_axis_markup(axis)
    axis.legend()
    return figure, axis


def _condition_label(given_syntax: str | None, event_syntax: str) -> str:
    if given_syntax is None or given_syntax.strip() == "":
        return f"P({event_syntax})"
    return f"P({event_syntax} | {given_syntax})"


def _parse_optional_condition(condition_syntax: str | None):
    if condition_syntax is None or condition_syntax.strip() == "":
        return None
    return parse_condition_syntax(condition_syntax)


def build_adhoc_conditional_payload(
    parent_site,
    readout: ThreeImageReadout,
    *,
    x: str,
    given_syntax: str | None,
    event_syntax: str,
) -> dict[str, object]:
    """Recompute an arbitrary condition from raw repeated-shot counts."""

    x_values = np.asarray(parent_site.series(x), dtype=float)
    threshold = readout.threshold_value()
    given_condition = _parse_optional_condition(given_syntax)
    event_condition = parse_condition_syntax(event_syntax)

    pooled_probability = np.empty(parent_site.site.num_points, dtype=float)
    pooled_error = np.empty(parent_site.site.num_points, dtype=float)
    probability_by_group = []
    error_by_group = []
    num_selected = np.empty(parent_site.site.num_points, dtype=int)
    num_successes = np.empty(parent_site.site.num_points, dtype=int)

    for point_index in range(parent_site.site.num_points):
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
        pooled_probability[point_index] = result.pooled_probability
        pooled_error[point_index] = result.pooled_probability_error
        probability_by_group.append(result.probability_by_group)
        error_by_group.append(result.probability_error_by_group)
        num_selected[point_index] = result.pooled_num_selected
        num_successes[point_index] = result.pooled_num_successes

    order = np.argsort(x_values)
    return {
        "x_path": x,
        "given_syntax": given_syntax,
        "event_syntax": event_syntax,
        "threshold": threshold,
        "x_values": np.asarray(x_values[order], dtype=float),
        "pooled_probability": np.asarray(pooled_probability[order], dtype=float),
        "pooled_error": np.asarray(pooled_error[order], dtype=float),
        "probability_by_group": np.asarray(probability_by_group, dtype=float)[order, :],
        "error_by_group": np.asarray(error_by_group, dtype=float)[order, :],
        "num_selected": np.asarray(num_selected[order], dtype=int),
        "num_successes": np.asarray(num_successes[order], dtype=int),
    }


def plot_adhoc_condition(
    site, readout: ThreeImageReadout
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    payload = build_adhoc_conditional_payload(
        site,
        readout,
        x=X_PATH,
        given_syntax=ADHOC_GIVEN,
        event_syntax=ADHOC_EVENT,
    )
    x_plot = _plot_x(payload["x_values"])

    figure, (pooled_axis, group_axis) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.6),
        sharex=True,
        constrained_layout=True,
        height_ratios=(1.0, 1.8),
    )
    pooled_axis.errorbar(
        x_plot,
        payload["pooled_probability"],
        yerr=payload["pooled_error"],
        marker="o",
        linestyle="",
        color="k",
        capsize=2,
        label="pooled groups",
    )
    pooled_axis.set_ylabel("pooled probability")
    pooled_axis.set_title(_condition_label(ADHOC_GIVEN, ADHOC_EVENT))
    _apply_common_axis_markup(pooled_axis, show_xlabel=False)
    pooled_axis.legend()

    for group_index in ADHOC_GROUP_INDICES:
        group_axis.errorbar(
            x_plot,
            payload["probability_by_group"][:, group_index],
            yerr=payload["error_by_group"][:, group_index],
            marker="o",
            linestyle="",
            capsize=2,
            alpha=0.85,
            label=f"group {group_index}",
        )

    group_axis.set_ylabel("per-group probability")
    _apply_common_axis_markup(group_axis)
    group_axis.legend()
    return figure, (pooled_axis, group_axis)


def main() -> None:
    run = LabNdscanRun.open(SNAPSHOT_PATH)
    site = run.site(SITE_PATH)
    readout = ThreeImageReadout.from_site(
        run.site(IMAGE_SITE_PATH).site,
        blob_name=IMAGING_BLOB_NAME,
    )

    plot_average_trace(site)
    plot_group_traces(site)
    plot_adhoc_condition(site, readout)

    # Add ad hoc analysis here, e.g. scipy fits, excluded points, annotations, theory
    # curves, or additional saved/recomputed series.

    plt.show()


if __name__ == "__main__":
    main()

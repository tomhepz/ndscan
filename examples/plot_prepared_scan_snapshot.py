"""Tiny pyplot helper for prepared-runtime HDF5 snapshots.

Usage:

    python examples/plot_prepared_scan_snapshot.py <snapshot.h5> [--site scan_p/scan_x]

The script is intentionally small, but it supports one very useful interactive action:

- click a plotted point in the chosen site,
- and the matching child scan sites are replotted below for that point.
- click one of those child plots, and the next level is expanded beneath it.

This is especially useful for nested repeated-shot scans such as
``prepared_scan_probability_frequency.py``:

- select ``--site probability_scan``,
- click a point on the outer probability-vs-time curve,
- and the lower panel will show the underlying repeated-shot yes/no outcomes for that
  specific outer point.

It also works for deeper trees such as ``prepared_scan_nested_p_variation.py``:

- start at the root site,
- click the single root point to reveal ``scan_p``,
- then click a ``scan_p`` point to reveal the matching ``scan_x`` data beneath it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ndscan.results.scan_site_reader import ScanSiteData, read_scan_site_snapshot


@dataclass(frozen=True)
class _VisibleSitePanel:
    site: ScanSiteData
    raw_points: dict[str, list]
    global_point_indices: np.ndarray
    selected_point_index: int | None = None


def _site_path_from_argument(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part for part in value.split("/") if part)


def _default_plot_x_values_and_label(
    site: ScanSiteData, raw_points: dict[str, list]
) -> tuple[np.ndarray, str]:
    plot_choices = site.describe_plot_choices()
    default_x = plot_choices.x.default
    if default_x is None:
        first_series = next(iter(raw_points.values()), [])
        return np.arange(len(first_series)), "point_index"

    storage_key = default_x.storage_key
    if storage_key is None:
        first_series = next(iter(raw_points.values()), [])
        return np.arange(len(first_series)), default_x.label

    values = raw_points.get(storage_key)
    if values is None:
        first_series = next(iter(raw_points.values()), [])
        return np.arange(len(first_series)), default_x.label

    return np.asarray(values), default_x.label


def _label_for_channel(site, key: str) -> str:
    schema = site.channels[key]
    return schema.get("description") or schema.get("path") or key


def _binary_channel_summary(values: np.ndarray) -> str | None:
    """Return a compact binomial-style summary for binary raw points."""

    if values.size == 0:
        return None
    if not np.all(np.isin(values, [0, 1])):
        return None

    successes = int(values.sum())
    num_shots = int(values.size)
    probability = successes / num_shots
    standard_error = np.sqrt(probability * (1.0 - probability) / num_shots)
    return (
        f"n={num_shots}, successes={successes}, "
        f"p={probability:.3f}, stderr={standard_error:.3f}"
    )


def _x_data_for_site(site: ScanSiteData, raw_points: dict[str, list]):
    """Return default x data and label for one site's raw point arrays."""

    return _default_plot_x_values_and_label(site, raw_points)


def _merge_segment_raw_points(
    site: ScanSiteData, parent_point_index: int
) -> tuple[dict[str, list], np.ndarray]:
    """Return merged child raw points and global point indices for one parent point."""

    segments = site.segments_for_parent_point(parent_point_index)
    merged_raw_points = {key: [] for key in site.raw_points}
    global_point_indices = []

    for segment in segments:
        segment_data = site.slice_raw_points(segment.start_index, segment.stop_index)
        for key, values in segment_data.items():
            merged_raw_points[key].extend(values)
        global_point_indices.extend(range(segment.start_index, segment.stop_index))

    return merged_raw_points, np.asarray(global_point_indices, dtype=int)


def _build_visible_site_panels(snapshot, selection_chain):
    """Return all detail panels implied by the current selection chain."""

    selected_points = dict(selection_chain)
    panels = []
    for parent_path, parent_point_index in selection_chain:
        for child_site in snapshot.child_sites(parent_path):
            raw_points, global_point_indices = _merge_segment_raw_points(
                child_site, parent_point_index
            )
            panels.append(
                _VisibleSitePanel(
                    site=child_site,
                    raw_points=raw_points,
                    global_point_indices=global_point_indices,
                    selected_point_index=selected_points.get(child_site.path),
                )
            )
    return panels


def _update_selection_chain(snapshot, root_path, selection_chain, clicked_site_path, point_index):
    """Update the active drill-down branch after a point pick."""

    if clicked_site_path == root_path:
        return [(root_path, point_index)]

    clicked_site = snapshot.get_site(clicked_site_path)
    parent_path = clicked_site.parent_path
    if parent_path is None:
        return [(clicked_site_path, point_index)]

    chain_paths = [path for path, _ in selection_chain]
    if parent_path not in chain_paths:
        root_selection = next(
            (entry for entry in selection_chain if entry[0] == root_path),
            None,
        )
        if root_selection is None:
            return selection_chain
        return [root_selection, (clicked_site_path, point_index)]

    parent_index = chain_paths.index(parent_path)
    return selection_chain[: parent_index + 1] + [(clicked_site_path, point_index)]


def _plot_child_raw_points(
    plot_axis,
    site: ScanSiteData,
    raw_points: dict[str, list],
    global_point_indices: np.ndarray,
    *,
    selected_point_index: int | None = None,
):
    """Plot one child site's raw points for a selected parent point."""

    plot_axis.clear()
    plot_axis.set_axis_on()

    channel_keys = list(site.channels.keys())
    if global_point_indices.size == 0:
        plot_axis.text(0.5, 0.5, "No child data for the selected parent point", ha="center", va="center")
        plot_axis.set_title("/".join(site.path))
        plot_axis.set_axis_off()
        return

    if not channel_keys:
        plot_axis.text(0.5, 0.5, "No saved child channels", ha="center", va="center")
        plot_axis.set_title("/".join(site.path))
        plot_axis.set_axis_off()
        return

    x_values, x_label = _x_data_for_site(site, raw_points)
    order = np.argsort(x_values)
    point_order = global_point_indices[order]
    plotted_binary_summary = None
    for channel_key in channel_keys:
        y_values = np.asarray(raw_points[channel_key])
        label = _label_for_channel(site, channel_key)
        if np.all(np.isin(y_values, [0, 1])):
            plot_axis.step(x_values[order], y_values[order], where="mid", label=label)
            scatter = plot_axis.scatter(x_values[order], y_values[order], s=20, picker=True)
            if plotted_binary_summary is None:
                plotted_binary_summary = _binary_channel_summary(y_values)
        else:
            plot_axis.plot(x_values[order], y_values[order], marker="o", label=label)
            scatter = plot_axis.scatter(
                x_values[order],
                y_values[order],
                s=30,
                alpha=0.0,
                picker=True,
            )

        scatter._ndscan_point_order = point_order
        scatter._ndscan_site_path = site.path

    title = "/".join(site.path) if site.path else "root"
    if plotted_binary_summary is not None:
        title += "\n" + plotted_binary_summary
    plot_axis.set_title(title)
    plot_axis.set_xlabel(x_label)
    if len(channel_keys) > 1:
        plot_axis.legend()

    if selected_point_index is not None:
        matches = np.nonzero(global_point_indices == selected_point_index)[0]
        if len(matches):
            plot_axis.axvline(x_values[matches[0]], color="tab:red", alpha=0.6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument(
        "--site",
        default="",
        help="Slash-separated site path, e.g. scan_p/scan_x. Defaults to the root site.",
    )
    args = parser.parse_args()

    snapshot = read_scan_site_snapshot(args.snapshot)
    site = snapshot.get_site(_site_path_from_argument(args.site))

    channel_keys = list(site.channels.keys())
    if not channel_keys:
        raise SystemExit("Selected site has no saved channels to plot")

    figure = plt.figure(figsize=(10, 3 * len(channel_keys)), constrained_layout=True)
    selection_chain = []

    def rebuild_figure():
        x_values, x_label = _default_plot_x_values_and_label(site, site.raw_points)

        direct_child_sites = snapshot.child_sites(site.path)
        visible_panels = _build_visible_site_panels(snapshot, selection_chain)
        show_instruction_axis = bool(direct_child_sites) and not selection_chain
        num_detail_axes = len(visible_panels) if visible_panels else (1 if show_instruction_axis else 0)

        figure.clear()
        all_axes = np.atleast_1d(
            figure.subplots(
                nrows=len(channel_keys) + num_detail_axes,
                sharex=False,
                squeeze=False,
            )
        ).reshape(-1)
        figure.set_size_inches(10, 3 * (len(channel_keys) + num_detail_axes))

        axes = all_axes[: len(channel_keys)]
        detail_axes = all_axes[len(channel_keys) :]
        order = np.argsort(x_values)
        selected_root_point = dict(selection_chain).get(site.path)

        for plot_axis, channel_key in zip(axes, channel_keys):
            y_values = np.asarray(site.raw_points[channel_key])
            plot_axis.plot(x_values[order], y_values[order], color="tab:blue", alpha=0.7)
            scatter = plot_axis.scatter(
                x_values[order],
                y_values[order],
                s=40,
                color="tab:blue",
                picker=True,
            )
            scatter._ndscan_point_order = order
            scatter._ndscan_site_path = site.path
            plot_axis.set_ylabel(_label_for_channel(site, channel_key))

            selected_line = plot_axis.axvline(
                x_values[order][0] if len(order) else 0.0,
                color="tab:red",
                alpha=0.0,
            )
            if selected_root_point is not None:
                selected_line.set_xdata([x_values[selected_root_point], x_values[selected_root_point]])
                selected_line.set_alpha(0.6)

        axes[-1].set_xlabel(x_label)
        axes[0].set_title("/".join(site.path) if site.path else "root")

        if show_instruction_axis:
            detail_axes[0].text(
                0.5,
                0.5,
                "Click a point above to inspect child-site data",
                ha="center",
                va="center",
            )
            detail_axes[0].set_axis_off()
        else:
            for plot_axis, panel in zip(detail_axes, visible_panels, strict=False):
                _plot_child_raw_points(
                    plot_axis,
                    panel.site,
                    panel.raw_points,
                    panel.global_point_indices,
                    selected_point_index=panel.selected_point_index,
                )

    def on_pick(event):
        artist = event.artist
        point_order = getattr(artist, "_ndscan_point_order", None)
        site_path = getattr(artist, "_ndscan_site_path", None)
        if point_order is None or site_path is None or len(event.ind) == 0:
            return

        picked_sorted_index = int(event.ind[0])
        point_index = int(point_order[picked_sorted_index])

        selection_chain[:] = _update_selection_chain(
            snapshot, site.path, selection_chain, site_path, point_index
        )
        rebuild_figure()
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("pick_event", on_pick)
    rebuild_figure()

    plt.show()


if __name__ == "__main__":
    main()

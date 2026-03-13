"""Tiny pyplot helper for host-runtime HDF5 snapshots.

Usage:

    python examples/plot_host_runtime_snapshot.py <snapshot.h5> [--site scan_p/scan_x]

The script is intentionally small, but it supports one very useful interactive action:

- click a plotted point in the chosen site,
- and any immediate child scan sites are replotted below for the matching parent point.

This is especially useful for nested repeated-shot scans such as
``host_runtime_probability_frequency.py``:

- select ``--site probability_scan``,
- click a point on the outer probability-vs-time curve,
- and the lower panel will show the underlying repeated-shot yes/no outcomes for that
  specific outer point.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from ndscan.results.scan_site_reader import HostRuntimeSiteData, read_host_runtime_snapshot


def _site_path_from_argument(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part for part in value.split("/") if part)


def _label_for_x(site, kind: str, key: str) -> str:
    if kind == "pseudoparam":
        schema = site.pseudoparams[key]["variable"]
        return schema.get("description") or schema["name"]

    schema = site.parameters[key]["param"]
    return schema.get("description") or schema["fqn"].split(".")[-1]


def _label_for_channel(site, key: str) -> str:
    schema = site.channels[key]
    return schema.get("description") or schema.get("path") or key


def _binary_channel_summary(values: np.ndarray) -> str | None:
    """Return a compact binomial-style summary for binary point data."""

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


def _x_data_for_site(site: HostRuntimeSiteData, point_data: dict[str, list]):
    """Return default x data and label for one site's point arrays."""

    x_kind, x_key = site.choose_default_x_key()
    if x_kind is None or x_key is None:
        first_series = next(iter(point_data.values()), [])
        return np.arange(len(first_series)), "point index"

    if x_kind == "pseudoparam":
        return np.asarray(point_data[x_key]), _label_for_x(site, x_kind, x_key)

    return np.asarray(point_data[x_key]), _label_for_x(site, x_kind, x_key)


def _plot_child_point_data(plot_axis, site: HostRuntimeSiteData, point_data: dict[str, list]):
    """Plot one child site's point data for a selected parent point."""

    plot_axis.clear()

    x_values, x_label = _x_data_for_site(site, point_data)
    channel_keys = list(site.channels.keys())
    if not channel_keys:
        plot_axis.text(0.5, 0.5, "No saved child channels", ha="center", va="center")
        plot_axis.set_axis_off()
        return

    order = np.argsort(x_values)
    plotted_binary_summary = None
    for channel_key in channel_keys:
        y_values = np.asarray(point_data[channel_key])
        label = _label_for_channel(site, channel_key)
        if np.all(np.isin(y_values, [0, 1])):
            plot_axis.step(x_values[order], y_values[order], where="mid", label=label)
            plot_axis.scatter(x_values[order], y_values[order], s=20)
            if plotted_binary_summary is None:
                plotted_binary_summary = _binary_channel_summary(y_values)
        else:
            plot_axis.plot(x_values[order], y_values[order], marker="o", label=label)

    title = "/".join(site.path) if site.path else "root"
    if plotted_binary_summary is not None:
        title += "\n" + plotted_binary_summary
    plot_axis.set_title(title)
    plot_axis.set_xlabel(x_label)
    if len(channel_keys) > 1:
        plot_axis.legend()


def _show_selected_subpoints(detail_axes, snapshot, site: HostRuntimeSiteData, point_index: int):
    """Redraw detail axes for the child-site data beneath one selected parent point."""

    child_sites = snapshot.child_sites(site.path)
    if not child_sites:
        return

    for plot_axis, child_site in zip(detail_axes, child_sites, strict=False):
        segments = child_site.segments_for_parent_point(point_index)
        if not segments:
            plot_axis.clear()
            plot_axis.text(
                0.5,
                0.5,
                f"No child segment for parent point {point_index}",
                ha="center",
                va="center",
            )
            plot_axis.set_title("/".join(child_site.path))
            plot_axis.set_axis_off()
            continue

        merged_point_data = {
            key: []
            for key in child_site.point_data
        }
        for segment in segments:
            segment_data = child_site.slice_point_data(
                segment.start_index, segment.stop_index
            )
            for key, values in segment_data.items():
                merged_point_data[key].extend(values)

        _plot_child_point_data(plot_axis, child_site, merged_point_data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument(
        "--site",
        default="",
        help="Slash-separated site path, e.g. scan_p/scan_x. Defaults to the root site.",
    )
    args = parser.parse_args()

    snapshot = read_host_runtime_snapshot(args.snapshot)
    site = snapshot.get_site(_site_path_from_argument(args.site))
    child_sites = snapshot.child_sites(site.path)

    x_kind, x_key = site.choose_default_x_key()
    if x_kind is None or x_key is None:
        x_values = np.arange(site.metadata["state.num_points"])
        x_label = "point index"
    else:
        x_values = np.asarray(site.point_data[x_key])
        x_label = _label_for_x(site, x_kind, x_key)

    channel_keys = list(site.channels.keys())
    if not channel_keys:
        raise SystemExit("Selected site has no saved channels to plot")

    num_detail_axes = max(1, len(child_sites)) if child_sites else 0
    figure, all_axes = plt.subplots(
        nrows=len(channel_keys) + num_detail_axes,
        sharex=False,
        figsize=(10, 3 * (len(channel_keys) + num_detail_axes)),
        constrained_layout=True,
    )
    all_axes = np.atleast_1d(all_axes)
    axes = all_axes[: len(channel_keys)]
    detail_axes = all_axes[len(channel_keys) :] if child_sites else np.array([])

    order = np.argsort(x_values)
    selected_marker_lines = []

    for plot_axis, channel_key in zip(axes, channel_keys):
        y_values = np.asarray(site.point_data[channel_key])
        plot_axis.plot(x_values[order], y_values[order], color="tab:blue", alpha=0.7)
        scatter = plot_axis.scatter(
            x_values[order],
            y_values[order],
            s=40,
            color="tab:blue",
            picker=True,
        )
        scatter._ndscan_point_order = order
        plot_axis.set_ylabel(_label_for_channel(site, channel_key))
        selected_marker_lines.append(
            plot_axis.axvline(x_values[order][0] if len(order) else 0.0, color="tab:red", alpha=0.0)
        )

    axes[-1].set_xlabel(x_label)
    axes[0].set_title("/".join(site.path) if site.path else "root")

    if child_sites:
        for plot_axis in detail_axes:
            plot_axis.text(
                0.5,
                0.5,
                "Click a point above to inspect child-site data",
                ha="center",
                va="center",
            )
            plot_axis.set_axis_off()

        def on_pick(event):
            artist = event.artist
            point_order = getattr(artist, "_ndscan_point_order", None)
            if point_order is None or len(event.ind) == 0:
                return

            picked_sorted_index = int(event.ind[0])
            parent_point_index = int(point_order[picked_sorted_index])
            selected_x = x_values[parent_point_index]

            for line in selected_marker_lines:
                line.set_xdata([selected_x, selected_x])
                line.set_alpha(0.6)

            _show_selected_subpoints(detail_axes, snapshot, site, parent_point_index)
            figure.canvas.draw_idle()

        figure.canvas.mpl_connect("pick_event", on_pick)

    plt.show()


if __name__ == "__main__":
    main()

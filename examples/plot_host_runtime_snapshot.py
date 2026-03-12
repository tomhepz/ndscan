"""Tiny pyplot helper for host-runtime HDF5 snapshots.

Usage:

    python examples/plot_host_runtime_snapshot.py <snapshot.h5> [--site scan_p/scan_x]

The script is intentionally simple. It plots the first pseudoparam or parameter it can
find against all saved channels for the chosen site.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from ndscan.results.scan_site_reader import read_host_runtime_snapshot


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

    figure, axes = plt.subplots(
        nrows=len(channel_keys),
        sharex=True,
        figsize=(10, 3 * len(channel_keys)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    order = np.argsort(x_values)
    for plot_axis, channel_key in zip(axes, channel_keys):
        y_values = np.asarray(site.point_data[channel_key])
        plot_axis.plot(x_values[order], y_values[order], marker="o")
        plot_axis.set_ylabel(_label_for_channel(site, channel_key))

    axes[-1].set_xlabel(x_label)
    axes[0].set_title("/".join(site.path) if site.path else "root")
    plt.show()


if __name__ == "__main__":
    main()

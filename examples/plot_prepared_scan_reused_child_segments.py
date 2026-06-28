"""Offline walkthrough for the reused-child-segment example.

Run ``prepared_scan_reused_child_segments.py`` first, then edit the path below.

This file focuses on one thing:

- the same child site can produce more than one segment for one parent point.
"""

# %%
from __future__ import annotations

import pprint

import matplotlib.pyplot as plt
import numpy as np

from ndscan.results import read_scan_site_snapshot

# %%

# Change this to the HDF5 file you want to inspect.
SNAPSHOT_PATH = "/home/lab/artiq-files/dnamic-lab/results/2026-04-23/00/000000000-ReusedChildSegmentsParentFragment.h5"

ROOT_SITE_PATH = ()
CHILD_SITE_PATH = ("child_scan",)

snapshot = read_scan_site_snapshot(SNAPSHOT_PATH)
root_site = snapshot.get_site(ROOT_SITE_PATH)
child_site = snapshot.get_site(CHILD_SITE_PATH)

# %% Print the sites so it is clear what was saved.
print("Available site paths:")
print(sorted(snapshot.sites))
print()
print("Child sites of the root site:")
print([site.path for site in snapshot.child_sites(ROOT_SITE_PATH)])

# %% Print the child segments grouped by parent point.
print()
print("Child segments grouped by parent point:")
print("---------------------")
for parent_point_index in range(root_site.num_points):
    segments = child_site.segments_for_parent_point(parent_point_index)
    print(f"parent point {parent_point_index}:")
    for segment in segments:
        print(
            f"  segment {segment.index}: "
            f"start={segment.start_index}, stop={segment.stop_index}, "
            f"parent={segment.parent_point_index}"
        )
print("---------------------")

# %% Show the saved per-segment analysis payloads.
print()
print("Per-segment analysis payloads:")
print("---------------------")
for segment in child_site.segments():
    print(f"segment {segment.index}:")
    pprint.pp(child_site.analysis_for_segment(segment.index), sort_dicts=False)
print("---------------------")

# %% Plot the root totals so the two child phases are visible at the root site too.
outer_values = np.asarray(root_site.series("outer"), dtype=float)
first_totals = np.asarray(root_site.series("first_total"), dtype=float)
second_totals = np.asarray(root_site.series("second_total"), dtype=float)
combined_totals = np.asarray(root_site.series("combined_total"), dtype=float)

order = np.argsort(outer_values)

figure, axis = plt.subplots(figsize=(7, 4))
axis.plot(outer_values[order], first_totals[order], marker="o", label="first child run")
axis.plot(outer_values[order], second_totals[order], marker="o", label="second child run")
axis.plot(
    outer_values[order],
    combined_totals[order],
    marker="o",
    linestyle="--",
    label="combined total",
)
axis.set_xlabel("outer")
axis.set_ylabel("saved root result")
axis.set_title("Root results from two child segments per parent point")
axis.legend()
figure.tight_layout()

# %% Plot the flat child point stream and shade each saved segment.
child_y_key = child_site.require_channel_storage_key("y")
child_y_values = np.asarray(child_site.series("y"), dtype=float)
flat_point_index = np.arange(len(child_y_values))

figure, axis = plt.subplots(figsize=(8, 4))
axis.plot(flat_point_index, child_y_values, marker="o", linestyle="", label="child y")

for segment in child_site.segments():
    axis.axvspan(
        segment.start_index - 0.5,
        segment.stop_index - 0.5,
        alpha=0.15,
        label=f"parent {segment.parent_point_index}" if segment.index < 2 else None,
    )
    midpoint = 0.5 * (segment.start_index + segment.stop_index - 1)
    axis.text(
        midpoint,
        float(np.max(child_y_values)) + 0.5,
        f"s{segment.index}\np{segment.parent_point_index}",
        ha="center",
        va="bottom",
    )

axis.set_xlabel("flat child point index")
axis.set_ylabel("child y")
axis.set_title("One child site with multiple segments per parent point")
figure.tight_layout()

# %% Slice the raw child points back out for each parent point.
print()
print("Raw child y values for each parent point:")
print("---------------------")
for parent_point_index in range(root_site.num_points):
    segments = child_site.segments_for_parent_point(parent_point_index)
    print(f"parent point {parent_point_index}:")
    for segment in segments:
        raw = child_site.slice_raw_points(segment.start_index, segment.stop_index)
        print(f"  segment {segment.index} y values:", raw[child_y_key])
print("---------------------")

plt.show()

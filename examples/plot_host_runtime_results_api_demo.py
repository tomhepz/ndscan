"""Offline walkthrough of the generic ``ndscan.results`` API.

Run ``host_runtime_results_api_demo.py`` first to produce a snapshot, then edit the
path below and run this file in an interactive session.

The point of this file is to show the common offline workflows:

- open a saved run,
- discover what series exist,
- inspect the automatic scan metadata,
- load scalar and array channels,
- inspect one or two user blobs,
- and inspect one nested child scan segment.
"""

# %%
from __future__ import annotations

import pprint

import matplotlib.pyplot as plt
import numpy as np

from ndscan.results import read_host_runtime_snapshot, series_slices_along_axis

# %%

# Change this to the HDF5 file you want to inspect.
SNAPSHOT_PATH = "/home/lab/artiq-files/dnamic-lab/results/2026-04-22/00/000002776-ResultsApiDemoFragment.h5"

# These are the two sites created by the example experiment.
ROOT_SITE_PATH = ()
REPEAT_SITE_PATH = ("repeat_scan",)

# Pick one outer point when looking at the nested child scan.
HIGHLIGHT_PARENT_POINT_INDEX = 4

# Pick a few array indices when turning one array channel into several scalar traces.
TRACE_SAMPLE_INDICES = (0, 8, 16, 24, 31)
NUM_RAW_TRACES_TO_PLOT = 5

snapshot = read_host_runtime_snapshot(SNAPSHOT_PATH)
root_site = snapshot.get_site(ROOT_SITE_PATH)
repeat_site = snapshot.get_site(REPEAT_SITE_PATH)

# %% Open the run and see which sites exist.
print("Top-level snapshot metadata:")
print("---------------------")
pprint.pp(snapshot.top_level_metadata, sort_dicts=False)
print()
print("Available site paths:")
print(sorted(snapshot.sites))
print()
print("Child sites of the root site:")
print([site.path for site in snapshot.child_sites(ROOT_SITE_PATH)])
print("---------------------")

# %% List the saved series on the root site.
print()
print("Root site series descriptions:")
print("---------------------")
for item in root_site.describe_series():
    print(
        f"{item.kind:14} {item.path:28} "
        f"shape={item.shape!s:18} dtype={item.dtype!s:8} "
        f"label={item.label!r}"
    )
print("---------------------")

# %% Ask ndscan which x/y series make sense to plot by default.
print()
print("Root plot choices:")
print("---------------------")
root_plot_choices = root_site.describe_plot_choices()
print("default x path:", root_plot_choices.x.default_path)
print("default y path:", root_plot_choices.y.default_path)
print("default x shortcut:", root_site.choose_default_x_path())
print("advanced default x source:", root_site.choose_default_x_source())
print("---------------------")

# %% These are the automatic scan metadata blobs that ndscan persists for every run.
print()
print("Root automatic scan metadata:")
print("---------------------")
print("scan.point_policy")
pprint.pp(root_site.metadata["scan.point_policy"], sort_dicts=False)
print()
print("scan.parameter_mappings")
pprint.pp(root_site.metadata["scan.parameter_mappings"], sort_dicts=False)
print()
print("scan.fixed_pseudoparams")
pprint.pp(root_site.metadata["scan.fixed_pseudoparams"], sort_dicts=False)
print("---------------------")

# %% User blobs are the small bits of custom metadata your experiment chose to save.
print()
print("Root user metadata blobs:")
print("---------------------")
print("available blob names:", root_site.available_metadata_blob_names())
print()
print("demo_config")
pprint.pp(root_site.require_metadata_blob("demo_config"), sort_dicts=False)
print("---------------------")

# %% This is the saved final and online analysis attached to the root site.
print()
print("Root saved analysis payloads:")
print("---------------------")
print("final analysis outputs:", sorted(root_site.analysis_outputs))
print("final analysis artifacts:", sorted(root_site.analysis_artifacts))
print("online analysis ids:", sorted(root_site.online_analysis_results))
print("---------------------")

# %% Build one simple scalar plot from x/y/error series.
root_series = {item.path: item for item in root_site.describe_series()}

# Use the default x choice and scale it back into display units.
x_path = root_plot_choices.x.default_path
x_scale = root_series[x_path].scale if root_series[x_path].scale not in (None, 0) else 1.0
x_values = np.asarray(root_site.series(x_path), dtype=float) / float(x_scale)
y_values = np.asarray(root_site.series("mean_signal"), dtype=float)
y_errors = np.asarray(root_site.series("mean_signal_error"), dtype=float)

# Sort by x before plotting.
order = np.argsort(x_values)
x_values = x_values[order]
y_values = y_values[order]
y_errors = y_errors[order]

figure, axis = plt.subplots(figsize=(7, 4))
axis.errorbar(
    x_values,
    y_values,
    yerr=y_errors,
    marker="o",
    linestyle="",
    capsize=2,
    label="saved root channel",
)

# Plot the saved fit artifact if it exists.
line_fit_artifact = root_site.analysis_artifacts.get("signal_line_fit")
if line_fit_artifact is not None:
    axis.plot(
        np.asarray(line_fit_artifact["logical_frequency_values_mhz"], dtype=float),
        np.asarray(line_fit_artifact["fitted_y_values"], dtype=float),
        label="saved final analysis artifact",
    )

# Also show one offline fit done directly from the saved series.
offline_polyfit = np.polyfit(
    x_values,
    y_values,
    deg=1,
    w=1.0 / np.maximum(y_errors, 1e-9),
)
offline_fit_x = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
offline_fit_y = np.polyval(offline_polyfit, offline_fit_x)
axis.plot(
    offline_fit_x,
    offline_fit_y,
    linestyle="--",
    label="offline np.polyfit",
)

axis.set_xlabel(root_series[x_path].label)
axis.set_ylabel(root_series["mean_signal"].label)
axis.set_title("Root scan: saved mean signal with saved and offline fits")
axis.legend()
figure.tight_layout()

# %% Load one array-valued channel and inspect one outer point.
demo_config = root_site.require_metadata_blob("demo_config")
sample_times_us = np.asarray(demo_config["trace_sample_times_us"], dtype=float)
selected_point_index = min(HIGHLIGHT_PARENT_POINT_INDEX, root_site.num_points - 1)

mean_trace = np.asarray(root_site.series("mean_trace"), dtype=float)
mean_trace_error = np.asarray(root_site.series("mean_trace_error"), dtype=float)

figure, axis = plt.subplots(figsize=(7, 4))
axis.errorbar(
    sample_times_us,
    mean_trace[selected_point_index],
    yerr=mean_trace_error[selected_point_index],
    marker="o",
    linestyle="-",
    capsize=2,
)
axis.set_xlabel("trace sample time / us")
axis.set_ylabel(root_series["mean_trace"].label)
axis.set_title(f"Root array channel at parent point {selected_point_index}")
figure.tight_layout()

# %% Split one array channel into several scalar traces.
sample_value_series = series_slices_along_axis(
    root_site,
    "mean_trace",
    axis=0,
    indices=TRACE_SAMPLE_INDICES,
)

figure, axis = plt.subplots(figsize=(7, 4))
for sample_index, series_values in sample_value_series.items():
    axis.plot(
        x_values,
        np.asarray(series_values, dtype=float)[order],
        marker="o",
        linestyle="",
        label=f"sample {sample_index}",
    )
axis.set_xlabel(root_series[x_path].label)
axis.set_ylabel("trace value at fixed sample")
axis.set_title("A few array slices from root.mean_trace")
axis.legend()
figure.tight_layout()

# %% The repeat site is just another site, so the same discovery methods work there too.
print()
print("Repeat-site series descriptions:")
print("---------------------")
for item in repeat_site.describe_series():
    print(
        f"{item.kind:14} {item.path:28} "
        f"shape={item.shape!s:18} dtype={item.dtype!s:8} "
        f"label={item.label!r}"
    )
print("---------------------")

print()
print("Repeat-site plot choices:")
print("---------------------")
repeat_plot_choices = repeat_site.describe_plot_choices()
print("default x path:", repeat_plot_choices.x.default_path)
print("default y path:", repeat_plot_choices.y.default_path)
print("default x shortcut:", repeat_site.choose_default_x_path())
print("---------------------")

# %% Show one child-site user blob and the saved online analysis ids.
print()
print("Repeat-site user metadata blobs:")
print("---------------------")
print("available blob names:", repeat_site.available_metadata_blob_names())
print()
print("repeat_config")
pprint.pp(repeat_site.require_metadata_blob("repeat_config"), sort_dicts=False)
print("---------------------")

print()
print("Repeat-site saved analysis payloads:")
print("---------------------")
print("final analysis outputs:", sorted(repeat_site.analysis_outputs))
print("online analysis ids:", sorted(repeat_site.online_analysis_results))
print("---------------------")

# %% Recover the child segment for one outer point and inspect its raw saved series.
segments = repeat_site.segments_for_parent_point(selected_point_index)
print()
print(f"Segments for parent point {selected_point_index}:")
print("---------------------")
pprint.pp(segments, sort_dicts=False)

if not segments:
    print("No repeat-site segments for the selected parent point.")
else:
    segment = segments[0]

    # This is the child's saved final analysis for this one segment.
    print()
    print("Final analysis payload for this segment:")
    pprint.pp(repeat_site.analysis_for_segment(segment.index), sort_dicts=False)

    # This is the raw per-shot data slice for the same segment.
    raw_segment = repeat_site.slice_raw_points(segment.start_index, segment.stop_index)
    shot_signal_key = repeat_site.require_channel_storage_key("shot/shot_signal")
    shot_trace_key = repeat_site.require_channel_storage_key("shot/shot_trace")

    raw_shot_signal = np.asarray(raw_segment[shot_signal_key], dtype=float)
    raw_shot_trace = np.asarray(raw_segment[shot_trace_key], dtype=float)

    figure, (signal_axis, trace_axis) = plt.subplots(1, 2, figsize=(10, 4))

    signal_axis.plot(
        np.arange(len(raw_shot_signal)),
        raw_shot_signal,
        marker="o",
        linestyle="",
    )
    signal_axis.set_xlabel("repeat point_index")
    signal_axis.set_ylabel("shot/shot_signal")
    signal_axis.set_title(f"Raw repeated shots for parent point {selected_point_index}")

    for trace_index, trace in enumerate(raw_shot_trace[:NUM_RAW_TRACES_TO_PLOT]):
        trace_axis.plot(sample_times_us, trace, label=f"shot {trace_index}")
    trace_axis.set_xlabel("trace sample time / us")
    trace_axis.set_ylabel("shot/shot_trace")
    trace_axis.set_title("A few raw repeated traces")
    trace_axis.legend()
    figure.tight_layout()

# %% Show the plots in an interactive session.
plt.show()

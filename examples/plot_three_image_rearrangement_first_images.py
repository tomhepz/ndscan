"""Plot the first few saved images from one ndscan image channel.

This is intentionally a very small example. It only uses the generic ndscan results
reader, numpy, and matplotlib. Change the variables below, then run the file.
"""

# %%
from __future__ import annotations

import pprint

import matplotlib.pyplot as plt
import numpy as np

from ndscan.results import read_host_runtime_snapshot

# %%

SNAPSHOT_PATH = "/home/lab/artiq-files/dnamic-lab/results/2026-04-15/12/000002761-ThreeImageRearrangementDashboardFragment.h5"
IMAGE_SITE_PATH = ("repeat_scan",)
IMAGE_CHANNEL = "shot/image0"
NUM_IMAGES_TO_PLOT = 3

ROOT_SITE_PATH = ()
X_PATH = "shot/probe_frequency"
TRAP_PROBABILITY_CHANNEL = "bright_probability_image2_given_bright_image1_by_trap"
TRAP_ERROR_CHANNEL = "bright_probability_error_image2_given_bright_image1_by_trap"
GROUPS_TO_PLOT = (0, 1, 2, 3)
ROI_INDEX = 1

HIGHLIGHT_PARENT_POINT_INDEX = 10
SUBSCAN_SERIES = "shot/counts_image2"
SUBSCAN_GROUP_INDEX = 0
SUBSCAN_ROI_INDEX = 1

snapshot = read_host_runtime_snapshot(SNAPSHOT_PATH)

# %% Show available series on the image site
site = snapshot.get_site(IMAGE_SITE_PATH)
root_site = snapshot.get_site(ROOT_SITE_PATH)

print("Available series on the image site:")
print("---------------------")
for item in site.describe_series():
    print(
        f"{item['kind']:14} {item['path']:28} "
        f"shape={item['shape']} dtype={item['dtype']}"
    )
print("---------------------")

# %% Show structured metadata blobs on the image site
print()
print("Metadata blobs on the image site:")
print("---------------------")
for name, blob in site.metadata_blobs().items():
    print(name)
    pprint.pp(blob, sort_dicts=False)
print("---------------------")

# %% Plot the first N images from the image channel
images = site.series(IMAGE_CHANNEL)
num_to_plot = min(NUM_IMAGES_TO_PLOT, len(images))

print()
print(f"Loaded {len(images)} images from {IMAGE_CHANNEL!r}.")
print(f"Image array shape: {images.shape}")
print(f"Plotting the first {num_to_plot}.")

figure, axes = plt.subplots(1, num_to_plot, figsize=(4 * num_to_plot, 4))
axes = np.atleast_1d(axes)

for image_index, axis in enumerate(axes):
    image_plot = axis.imshow(images[image_index], cmap="magma", origin="upper")
    axis.set_title(f"{IMAGE_CHANNEL}\nimage {image_index}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.colorbar(image_plot, ax=axis)

figure.tight_layout()
plt.show()

# %% Average the images of this image channel
figure, axis = plt.subplots()
image_plot = axis.imshow(np.mean(images, axis=0), cmap="magma", origin="upper")
axis.set_title(f"Average {IMAGE_CHANNEL}")
axis.set_xlabel("x")
axis.set_ylabel("y")
figure.colorbar(image_plot, ax=axis)

# %% Plot a live-analysis probability channel against scan frequency
print()
print("Available series on the root site:")
print("---------------------")
for item in root_site.describe_series():
    print(
        f"{item['kind']:14} {item['path']:60} "
        f"shape={item['shape']} dtype={item['dtype']}"
    )
print("---------------------")

x_values = root_site.series(X_PATH, dtype=float)

trap_probability = root_site.series(TRAP_PROBABILITY_CHANNEL, dtype=float)
trap_error = root_site.series(TRAP_ERROR_CHANNEL, dtype=float)

sort_order = np.argsort(x_values)
x_values = x_values[sort_order]
trap_probability = trap_probability[sort_order]
trap_error = trap_error[sort_order]

group_probability = trap_probability[:, GROUPS_TO_PLOT, ROI_INDEX]
group_error = trap_error[:, GROUPS_TO_PLOT, ROI_INDEX]

average_probability = np.mean(group_probability, axis=1)
average_error = np.sqrt(np.sum(group_error**2, axis=1)) / len(GROUPS_TO_PLOT)

figure, (average_axis, group_axis) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(6.4, 5.6),
)

average_axis.errorbar(
    x_values,
    average_probability,
    yerr=average_error,
    marker="o",
    linestyle="",
    color="k",
    capsize=2,
)
average_axis.set_title(f"{TRAP_PROBABILITY_CHANNEL}, roi {ROI_INDEX}")
average_axis.set_ylabel("mean over selected groups")
average_axis.set_ylim(-0.05, 1.05)

for plot_index, group_index in enumerate(GROUPS_TO_PLOT):
    group_axis.errorbar(
        x_values,
        group_probability[:, plot_index],
        yerr=group_error[:, plot_index],
        marker="o",
        linestyle="",
        capsize=2,
        label=f"group {group_index}",
    )

group_axis.set_xlabel(X_PATH)
group_axis.set_ylabel("group probability")
group_axis.set_ylim(-0.05, 1.05)
group_axis.legend()

figure.tight_layout()
plt.show()

# %% Highlight one parent scan point and plot the child subscan segment it launched
parent_x = root_site.series(X_PATH, dtype=float)
parent_probability = root_site.series(TRAP_PROBABILITY_CHANNEL, dtype=float)
parent_error = root_site.series(TRAP_ERROR_CHANNEL, dtype=float)

parent_group_probability = parent_probability[:, GROUPS_TO_PLOT, ROI_INDEX]
parent_group_error = parent_error[:, GROUPS_TO_PLOT, ROI_INDEX]
parent_average_probability = np.mean(parent_group_probability, axis=1)
parent_average_error = (
    np.sqrt(np.sum(parent_group_error**2, axis=1)) / len(GROUPS_TO_PLOT)
)

parent_point_index = min(HIGHLIGHT_PARENT_POINT_INDEX, root_site.num_points - 1)
segments = site.segments_for_parent_point(parent_point_index)

if not segments:
    print(
        f"No child segment on {site.path} for parent point {parent_point_index}. "
        "This site may not be a segmented child scan."
    )
else:
    segment = segments[0]
    subscan_values = site.series(SUBSCAN_SERIES)
    subscan_y = subscan_values[
        segment.start_index:segment.stop_index,
        SUBSCAN_GROUP_INDEX,
        SUBSCAN_ROI_INDEX,
    ]
    subscan_x = np.arange(segment.length)

    order = np.argsort(parent_x)
    figure, (parent_axis, subscan_axis) = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    parent_axis.errorbar(
        parent_x[order],
        parent_average_probability[order],
        yerr=parent_average_error[order],
        marker="o",
        linestyle="",
        color="k",
        capsize=2,
    )
    parent_axis.scatter(
        [parent_x[parent_point_index]],
        [parent_average_probability[parent_point_index]],
        color="tab:red",
        zorder=10,
        label=f"point {parent_point_index}",
    )
    parent_axis.set_xlabel(X_PATH)
    parent_axis.set_ylabel("mean probability")
    parent_axis.set_title("Parent scan site")
    parent_axis.set_ylim(-0.05, 1.05)
    parent_axis.legend()

    subscan_axis.plot(subscan_x, subscan_y, marker="o", linestyle="")
    subscan_axis.set_xlabel("repeat index in selected child segment")
    subscan_axis.set_ylabel(
        f"{SUBSCAN_SERIES}[group={SUBSCAN_GROUP_INDEX}, roi={SUBSCAN_ROI_INDEX}]"
    )
    subscan_axis.set_title(f"Child site {site.path}, segment {segment.index}")

    figure.tight_layout()
    plt.show()

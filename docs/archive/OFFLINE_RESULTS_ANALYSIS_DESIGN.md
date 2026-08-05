# Offline Results Analysis Design

This note describes the current architecture for offline analysis of prepared-runtime
`ndscan` results.

It is motivated by a common lab workflow:

- run an experiment through prepared-runtime `ndscan`
- save raw per-point series and live-analysis outputs to the ARTIQ HDF5 results file
- later reopen that file in a Python analysis environment
- either trust the saved live-analysis outputs, or recompute more complicated
  quantities from raw data
- plot the results with lab-specific helper code, ideally using the same structural
  view of the data as the live viewer

The goal is not to put lab-specific analysis into `ndscan`. The goal is to make
`ndscan` own the generic offline object model and persistence hooks needed for
lab/helper libraries to build on top.

## Goals

The offline results path should support:

- opening prepared-runtime HDF5 results without depending on ARTIQ execution code
- reconstructing the same site-tree structure used by the live viewer
- exposing enough metadata to reproduce saved live analysis
- letting downstream analysis either trust saved live outputs or recompute from raw
  saved channels
- keeping domain-specific physics, statistics, and figure logic outside `ndscan`

The user writing an experiment remains responsible for saving enough raw data and
metadata to make their own analysis reproducible. `ndscan` provides the generic storage
and access mechanism, but does not judge whether a lab-specific metadata blob is
complete enough.

The offline results path should not require:

- the legacy flat `ndscan.show` data model
- lab-specific assumptions in `ndscan` core
- scraping ad hoc `debug.*` datasets for reproducible offline analysis

## Current State

Prepared-runtime now has the desired core shape.

### Shared Live/Offline Object Model

Offline HDF5 reading reconstructs a site tree in:

- [ndscan/results/scan_site_reader.py](/home/lab/artiq-files/install/ndscan/ndscan/results/scan_site_reader.py)

The main public objects are:

- `ScanSiteSnapshot`
- `ScanSiteData`
- `ScanSiteSegment`
- `ScanSiteSegmentAnalysis`

The live viewer also converts live flat dataset mappings into the same
`ScanSiteSnapshot`/`ScanSiteData` structure via:

- [ndscan/plots/runtime/live.py](/home/lab/artiq-files/install/ndscan/ndscan/plots/runtime/live.py)

This is the key architectural seam:

- live applet datasets -> common site-tree representation
- HDF5 results file -> common site-tree representation

### Public `ndscan.results` API

The intended public API surface is exported from:

- [ndscan/results/__init__.py](/home/lab/artiq-files/install/ndscan/ndscan/results/__init__.py)

The core entry point is:

```python
from ndscan.results import read_scan_site_snapshot

snapshot = read_scan_site_snapshot("result.h5")
```

The core object model is:

```python
site = snapshot.get_site(())
child_site = snapshot.get_site(("repeat_scan",))
children = snapshot.child_sites(())
```

The main user-facing `ScanSiteData` methods for analysis scripts are:

```python
site.available_series_paths()
site.describe_series()
site.describe_plot_choices()
site.series("shot/probe_frequency")
site.series("some_result_channel")
site.series("point_index")
site.series("acquired_at_unix")

site.available_metadata_blob_names()
site.metadata_blobs()
site.require_metadata_blob("lab.imaging_readout")

site.segments()
site.segments_for_parent_point(parent_point_index)
site.slice_raw_points(start_index, stop_index)
```

For analysis scripts and user-facing examples, use the word `series`: a series can be
a parameter stream, pseudoparameter stream, result channel, runtime stream such as
`acquired_at_unix`, synthetic `point_index`, or point metadata stream. The word
`points` is still used internally for the persisted storage schema and by
`slice_raw_points(...)`, which deliberately returns storage-keyed raw point arrays for
segment-level work.

`describe_series()` now returns `SeriesDescription` dataclasses exposing display-oriented
metadata such as labels, units, scales, and whether a saved series is numeric.
`describe_plot_choices()` returns a `PlotChoices` dataclass exposing the numeric
plot-choice pool plus viewer-like default x/y/z selections by public series path.

The intended layering is:

- `describe_plot_choices()` is the public plotting-oriented entry point
- `choose_default_x_path()` is a lightweight convenience shortcut
- `choose_default_x_source()` is an advanced schema-level helper and should generally
  not be needed in ordinary plotting scripts

### Series Helpers

Keep generic pure-Python helpers small. Offline scripts should usually use
`site.series(...)` plus normal NumPy slicing, sorting, and averaging directly.

## Repository Boundary

The intended split across repositories is:

- `artiq`
  - experiment execution substrate
  - scheduler, datasets, HDF5 results file container

- `ndscan`
  - prepared-runtime execution and persistence
  - scan-site schema
  - live/offline site-tree result model
  - generic metadata-blob persistence and reading
  - generic result-navigation and plot-trace extraction helpers

- lab experiment/control repository
  - experiment fragments
  - hardware control libraries
  - experiment-specific decisions about what raw data and metadata to save

- lab helper repository
  - physics/statistics helper code
  - domain-specific reanalysis
  - publication/lab-style plotting helpers
  - lab-specific typed wrappers around persisted metadata blobs

The important rule is:

- `ndscan` owns the generic schema-to-object layer
- the lab helper repository owns the semantics of lab-specific metadata blobs

## Persisted Metadata Blobs

### Why Blobs Are Needed

If saved live analysis is supposed to be reproducible offline, then the final HDF5 file
must contain:

- the raw data used by that analysis
- enough metadata to explain how the raw data was interpreted

Sometimes that metadata is naturally an `ndscan` parameter. When it is fixed but still
essential for interpretation, it should still be persisted somewhere in the site tree.

Examples:

- ROI geometry
- threshold rule
- mapping from logical image roles to saved channel paths
- internal analysis configuration that is not intended to be a user-facing scan
  parameter

### Where Blobs Live

Prepared-runtime persists arbitrary site metadata as `extra.*` entries via:

- `ScanSite.extra_metadata`
- `ScanSiteDatasetWriter.publish_metadata(...)`

This is the right place for structured, versioned metadata blobs that are constant for
the whole site.

By contrast, a `debug.*` prefix is appropriate for live applets and ephemeral visual
debugging, but not as the source of truth for reproducible offline analysis.

### Blob Access

Offline code can list, retrieve, and inspect blobs generically:

```python
snapshot = read_scan_site_snapshot(path)
site = snapshot.get_site(("repeat_scan",))

print(site.available_metadata_blob_names())
print(site.metadata_blobs())

blob = site.require_metadata_blob("lab.imaging_readout")
```

`ndscan` decodes the stored value and returns ordinary Python objects. It does not
interpret the domain-specific meaning of the blob.

### Shape Of A Blob

`ndscan` should not impose a camera-specific or lab-specific schema in core. It should
only support storing and recovering structured blobs cleanly.

A good blob should be:

- namespaced
- versioned
- self-contained
- explicit about which saved channel paths it refers to

For example:

```python
{
    "namespace": "lab.imaging_readout",
    "version": 1,
    "images": {
        "image0": {
            "image_channel": "shot/image0",
            "counts_channel": "shot/counts_image0",
            "shape": [68, 92],
            "rois": [[[y0, y1, x0, x1], ...], ...],
        },
        "image1": {
            "image_channel": "shot/image1",
            "counts_channel": "shot/counts_image1",
            "shape": [80, 100],
            "rois": ...,
        },
    },
    "occupancy_rule": {
        "kind": "threshold_counts",
        "threshold_parameter_path": "threshold_counts",
    },
}
```

The `image_channel` / `counts_channel` entries are semantic channel paths:

- they tell later analysis code which saved `ndscan` channel path corresponds to each
  logical role
- they should refer to stable channel paths like `shot/image0`, not storage-order keys
  like `channel_3`

### What Should Not Go Into A Site Blob

A site blob is the right tool only for metadata that is constant for that site
invocation.

Do not use a site blob for:

- values that vary point-by-point
- large raw data arrays already saved as channels
- saved analysis outputs

Use instead:

- series / point metadata for point-varying values
- result channels for raw data arrays
- analysis outputs / artifacts for derived results

## Minimal Offline Plotting Pattern

The simplest offline scripts should be readable top-to-bottom and use plain numpy and
matplotlib. For example:

```python
import matplotlib.pyplot as plt
import numpy as np

from ndscan.results import read_scan_site_snapshot

snapshot = read_scan_site_snapshot("result.h5")

image_site = snapshot.get_site(("repeat_scan",))
root_site = snapshot.get_site(())

print("Image-site series:")
for item in image_site.describe_series():
    print(f"{item.kind:14} {item.path:30} shape={item.shape}")

print("Image-site metadata blobs:")
for name, blob in image_site.metadata_blobs().items():
    print(name, blob)

images = image_site.series("shot/image0")
plt.figure()
plt.imshow(images[0], cmap="magma", origin="upper")

x = root_site.series("shot/probe_frequency", dtype=float)
y = root_site.series("bright_pair_probability_image2_given_pair_image1", dtype=float)
yerr = root_site.series(
    "bright_pair_probability_error_image2_given_pair_image1",
    dtype=float,
)

order = np.argsort(x)
plt.figure()
plt.errorbar(x[order], y[order], yerr=yerr[order], marker="o", linestyle="")
plt.xlabel("shot/probe_frequency")
plt.ylabel("probability")
plt.show()
```

For parent/child scans, the same object model exposes the child segment launched by a
specific parent point:

```python
parent_point_index = 10
segments = image_site.segments_for_parent_point(parent_point_index)
segment = segments[0]

counts = image_site.series("shot/counts_image2")
child_counts = counts[segment.start_index:segment.stop_index, 0, 1]
repeat_index = np.arange(segment.length)

plt.figure()
plt.plot(repeat_index, child_counts, marker="o", linestyle="")
plt.xlabel("repeat index in selected child segment")
plt.ylabel("shot/counts_image2[group=0, roi=1]")
```

`segments_for_parent_point(...)` returns a list because one parent point can launch the
same child site more than once. Each child execution becomes a separate segment on that
site, all linked back to the same `parent_point_index`.

This does **not** mean one child site should be reused for arbitrarily different child
requests. A site path has one scalar metadata/schema record but an append-only flat
point stream. Reusing the same child site is only safe when the child executions keep
the same logical schema, for example the same scanned parameter set and the same result
channels. If later executions scan genuinely different things, they should usually be
modelled as different child scan sites rather than extra segments on one site.

A concrete example of this style is:

- [examples/plot_three_image_rearrangement_first_images.py](/home/lab/artiq-files/install/ndscan/examples/plot_three_image_rearrangement_first_images.py)
- [examples/plot_prepared_scan_reused_child_segments.py](/home/lab/artiq-files/install/ndscan/examples/plot_prepared_scan_reused_child_segments.py)

## Lab-Helper Responsibilities

The lab helper repository should own:

- parsing and validating lab-specific blobs such as `"lab.imaging_readout"`
- typed wrappers over `ScanSiteData`
- thresholding and ROI re-summing logic
- conditional binomial statistics
- domain-specific figure construction

For example, the lab helper repository can define a wrapper like:

```python
readout = ImagingReadoutSite.from_site(site)
readout.images["image0"]
readout.counts["image2"]
readout.rois["image1"]
readout.threshold_rule
```

That object is too lab-specific for `ndscan` core, but it is a good fit for a lab helper
repository built on the generic `ndscan` result model.

## Live Plotter Alignment

The live plotter already follows the right overall pattern:

1. transport-specific input:
   - live flat dataset mapping
2. common result object model:
   - `ScanSiteSnapshot`-shaped site tree via
     [ndscan/plots/runtime/live.py](/home/lab/artiq-files/install/ndscan/ndscan/plots/runtime/live.py)
3. plot-specific projection:
   - viewer logic in
     [ndscan/plots/runtime/viewer.py](/home/lab/artiq-files/install/ndscan/ndscan/plots/runtime/viewer.py)

The offline path mirrors that:

1. transport-specific input:
   - HDF5 file
2. common result object model:
   - `ScanSiteSnapshot` via
     [ndscan/results/scan_site_reader.py](/home/lab/artiq-files/install/ndscan/ndscan/results/scan_site_reader.py)
3. plot-specific projection:
   - normal NumPy/matplotlib code, or lab/helper plotting code built on the same
     object model

Prepared-runtime HDF5 files can also be opened directly in the runtime viewer:

```bash
ndscan_runtime_show result.h5
```

This uses `read_scan_site_snapshot(...)` and then feeds the resulting
`ScanSiteSnapshot` into the same `RuntimePlotViewer` widget used for live data. The
legacy `ndscan_show` command remains the right tool for pre-prepared-runtime files that
use the older `axes` / `channels` / `points.channel_*` schema.

The remaining long-term opportunity is factoring more pure-Python trace extraction out
of the Qt viewer so live and offline plotting can share more logic.

## Progress Against The Original Plan

### Phase 1: Persist enough metadata to reproduce live analysis

Status: done for the three-image example.

- Structured `extra.*` metadata is persisted.
- `site.require_metadata_blob(...)`, `site.available_metadata_blob_names()`, and
  `site.metadata_blobs()` recover it offline.
- `examples/prepared_scan_three_image_rearrangement.py` persists a versioned
  `lab.imaging_readout` blob alongside raw image/count channels.
- Tests prove the blob survives round-trip into `ScanSiteData`.

### Phase 2: Prove offline reproducibility

Status: proven in examples/tests, with lab-specific code still intentionally outside
`ndscan` core.

- HDF5 snapshots can be reopened via `read_scan_site_snapshot(...)`.
- Saved live analysis outputs can be plotted directly.
- Raw counts plus metadata can be used to recompute saved statistics in tests.
- Lab-specific image/blob/statistics logic remains in examples for now and should move
  to the lab helper repository when ready.

### Phase 3: Grow reusable offline helpers

Status: started.

- `site.series(...)`
- `site.describe_series(...)`

Only add more helpers when repeated real analysis scripts show a stable pattern.

### Phase 4: Unify live and offline plotting logic further

Status: future work.

This is the longer-term "plot in the same way" step. It should wait until the current
offline API has been used in real scripts and the useful common patterns are clearer.

## Current Recommendation

Stop adding broad API features for now and use the current results model in real
analysis scripts.

Let those scripts determine whether the next useful `ndscan` feature should be:

- joining multiple compatible HDF5 files as one virtual site
- dataframe/export helpers
- segment-aware recomputation helpers
- HDF5 compression/storage policy for large array channels
- factoring live-viewer trace extraction into reusable pure-Python helpers

The current public API is intentionally small enough to stabilize before choosing the
next extension.

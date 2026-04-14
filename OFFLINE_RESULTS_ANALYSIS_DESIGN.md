# Offline Results Analysis Design

This note describes the intended architecture for offline analysis of prepared-runtime
`ndscan` results.

It is motivated by a common lab workflow:

- run an experiment through prepared-runtime `ndscan`
- save raw point data and live-analysis outputs to the ARTIQ HDF5 results file
- later reopen that file in a Python analysis environment
- either trust the saved live-analysis outputs, or recompute more complicated
  quantities from raw data
- plot the results with lab-specific helper code, ideally using the same structural
  view of the data as the live viewer

The goal is not to put all lab-specific analysis into `ndscan`. The goal is to make
`ndscan` own the generic offline object model and the persistence hooks needed for
lab/helper libraries to build on top.

## Goals

The offline results path should support:

- opening prepared-runtime HDF5 results without depending on ARTIQ execution code
- reconstructing the same site-tree structure used by the live viewer
- exposing enough metadata to reproduce how saved live analysis was produced (the burden of this is on the user writing the metadata)
- letting downstream analysis either:
  - trust persisted live-analysis outputs, or
  - recompute from raw saved channels
- keeping domain-specific physics/statistics/figure logic outside `ndscan`

It should not require:

- the legacy flat `ndscan.show` data model
- lab-specific assumptions in `ndscan` core
- scraping ad hoc `debug.*` datasets for reproducible offline analysis

## Current Situation

Prepared-runtime already has the beginnings of the right shape.

### Shared live/offline object model

Offline HDF5 reading already reconstructs a site tree in:

- [ndscan/results/scan_site_reader.py](/home/lab/artiq-files/install/ndscan/ndscan/results/scan_site_reader.py)

The main objects are:

- [HostRuntimeSnapshot](/home/lab/artiq-files/install/ndscan/ndscan/results/scan_site_reader.py#L349)
- [HostRuntimeSiteData](/home/lab/artiq-files/install/ndscan/ndscan/results/scan_site_reader.py#L206)

The live viewer already converges on the same shape:

- [ndscan/plots/runtime/live.py](/home/lab/artiq-files/install/ndscan/ndscan/plots/runtime/live.py)

That module converts a live flat dataset mapping into the same
`HostRuntimeSnapshot`/`HostRuntimeSiteData` structure.

This is the key existing architectural seam. It means:

- live applet datasets -> common site-tree representation
- HDF5 results file -> common site-tree representation

### What is still missing

The missing pieces are:

- a clean way to persist structured, versioned site metadata blobs needed for later
  analysis
- helper APIs for reading those blobs back in a decoded and easily retrievable way
- reusable pure-Python analysis/series extraction helpers above the site tree
- a clear separation between generic `ndscan` result handling and lab-specific
  analysis logic

## Main Design Decision

The common representation should be the prepared-runtime site tree:

- one `HostRuntimeSnapshot` for the whole results file
- one `HostRuntimeSiteData` per scan site
- point arrays, segment metadata, analysis outputs, annotations, and artifacts all
  attached to that tree

Everything else should build on top of that:

- live plotting
- offline plotting
- lab-specific reanalysis
- helper libraries that inspect raw nested scan results

This means the offline stack should not start from:

- a raw `h5py` dictionary of datasets
- the legacy `ndscan.show` / `plots.model.hdf5` flat model

It should start from:

- `read_host_runtime_snapshot(...)`

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
  - generic result-navigation / plot-trace extraction helpers

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

### Why blobs are needed

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

### Where the blob should live

Prepared-runtime already persists arbitrary site metadata as `extra.*` entries via:

- [ScanSite.extra_metadata](/home/lab/artiq-files/install/ndscan/ndscan/schema/scan_site.py#L41)
- [ScanSiteDatasetWriter.publish_metadata()](/home/lab/artiq-files/install/ndscan/ndscan/runtime/persistence.py#L94)

This is the right place for structured, versioned metadata blobs that are constant for
the whole site.

By contrast, a `debug.*` prefix is appropriate for live applets and ephemeral visual
debugging, but not as the source of truth for reproducible offline analysis.

### Shape of a blob

`ndscan` should not impose a camera-specific or lab-specific schema in core. It should
only support storing and recovering structured blobs cleanly.

It is the user's responsibility to make sure a blob contains enough information to
reproduce any saved live statistics that depend on it. `ndscan` should provide the
mechanism for persisting and recovering the blob, but should not try to judge whether a
lab-specific blob is "complete enough" for offline reanalysis.

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
        "load": {
            "image_channel": "image0",
            "counts_channel": "counts_image0",
            "shape": [68, 92],
            "rois": [[[y0, y1, x0, x1], ...], ...],
        },
        "rearranged": {
            "image_channel": "image1",
            "counts_channel": "counts_image1",
            "shape": [80, 100],
            "rois": ...,
        },
        "spectroscopy": {
            "image_channel": "image2",
            "counts_channel": "counts_image2",
            "shape": [76, 96],
            "rois": ...,
        },
    },
    "occupancy_rule": {
        "kind": "threshold_counts",
        "threshold_source": {
            "kind": "fixed_parameter",
            "fqn": "fragment.threshold_counts",
        },
    },
}
```

The `image_channel` / `counts_channel` entries are the semantic-to-channel mapping:

- they tell later analysis code which saved `ndscan` channel path corresponds to each
  logical role
- they should refer to stable channel paths like `image0`, not storage-order keys like
  `channel_3`

### What should not go into a site blob

A site blob is the right tool only for metadata that is constant for that site
invocation.

Do not use a site blob for:

- values that vary point-by-point
- large raw data arrays already saved as channels
- saved analysis outputs

Use instead:

- point streams / point metadata for point-varying values
- result channels for raw data arrays
- analysis outputs / artifacts for derived results

## Proposed ndscan Responsibilities

`ndscan` should provide the following generic pieces.

### 1. HDF5 -> site-tree reader

Already present:

- [read_host_runtime_snapshot()](/home/lab/artiq-files/install/ndscan/ndscan/results/scan_site_reader.py#L372)

This remains the standard offline entry point.

### 2. Structured site-blob retrieval helpers

Add helpers on `HostRuntimeSiteData` or nearby utility functions so offline code can do
something like:

```python
snapshot = read_host_runtime_snapshot(path)
site = snapshot.get_site(("repeat_scan",))
blob = site.require_extra_blob("lab.imaging_readout")
```

This helper should:

- read a namespaced blob from site metadata
- decode JSON automatically
- raise a clear error if missing

It should not interpret the domain-specific meaning of the blob, nor should it try to
validate whether the blob is sufficient to reproduce a user's lab-specific statistics.

### 3. Pure-Python result navigation helpers

`HostRuntimeSnapshot` / `HostRuntimeSiteData` already expose site navigation and segment
helpers. This should be extended, when needed, with generic convenience methods for:

- selecting child-site slices for one parent point
- mapping channel paths to point arrays
- extracting common 1D/2D trace inputs for plotting

This layer should stay domain-neutral.

### 4. Reusable plot-trace extraction seam

Longer term, some of the pure-Python logic currently embedded in the runtime viewer
should move into reusable helpers so both:

- the live Qt viewer
- offline matplotlib/lab plotting

can use the same series extraction logic from the same site-tree model.

## Proposed Lab-Helper Responsibilities

The lab helper repository should own:

- parsing and validating lab-specific blobs such as `"lab.imaging_readout"`
- typed wrappers over `HostRuntimeSiteData`
- thresholding and ROI re-summing logic
- conditional binomial statistics
- domain-specific figure construction

For example, the lab helper repository can define a wrapper like:

```python
readout = ImagingReadoutSite.from_site(site)
readout.images["load"]
readout.counts["spectroscopy"]
readout.rois["rearranged"]
readout.threshold_rule
```

That object is probably too lab-specific for `ndscan` core, but it is a good fit for a
lab helper repository built on the generic `ndscan` result model.

## How This Ties Into the Live Plotter

The live plotter already follows the right overall pattern:

1. transport-specific input:
   - live flat dataset mapping
2. common result object model:
   - `HostRuntimeSnapshot`-shaped site tree via
     [ndscan/plots/runtime/live.py](/home/lab/artiq-files/install/ndscan/ndscan/plots/runtime/live.py)
3. plot-specific projection:
   - viewer logic in
     [ndscan/plots/runtime/viewer.py](/home/lab/artiq-files/install/ndscan/ndscan/plots/runtime/viewer.py)

The offline path should mirror that:

1. transport-specific input:
   - HDF5 file
2. common result object model:
   - `HostRuntimeSnapshot` via
     [ndscan/results/scan_site_reader.py](/home/lab/artiq-files/install/ndscan/ndscan/results/scan_site_reader.py)
3. plot-specific projection:
   - either the same pure-Python trace extraction helpers as the live viewer, or
     lab/helper plotting code built on the same object model

So yes: the common representation should be shared. The part that still needs work is
factoring more of the plot-trace extraction logic out of the Qt viewer and into
reusable pure-Python helpers.

## Example Offline Workflow

The intended offline usage should look like:

```python
from ndscan.results import read_host_runtime_snapshot

snapshot = read_host_runtime_snapshot("result.h5")
site = snapshot.get_site(("repeat_scan",))

blob = site.require_extra_blob("lab.imaging_readout")

# Lab-helper code starts here.
readout = ImagingReadoutSite(site, blob)

counts2 = readout.counts["spectroscopy"]
occupied2 = threshold_counts(counts2, readout.threshold_rule)
figure = plot_survival(readout, occupied2)
```

The important point is:

- `ndscan` gets the analysis code to a clean semantic object model
- the lab helper repository decides what to do with that object model

## Ordered Plan

### Phase 1: Persist enough metadata to reproduce live analysis

1. Add a clear convention for structured site metadata blobs in `ndscan`.
2. Add reader support so arbitrary structured site blobs are recovered and decoded
   cleanly from the HDF5/site-tree model.
3. Update
   [examples/host_runtime_three_image_rearrangement.py](/home/lab/artiq-files/install/ndscan/examples/host_runtime_three_image_rearrangement.py)
   to persist a versioned imaging-readout blob alongside the raw image/count channels.
4. Add a results-reader test proving that the blob survives round-trip into
   `HostRuntimeSiteData`.

### Phase 2: Prove offline reproducibility

5. In the lab helper repository, replace raw `load_hdf5_file(...)` usage for `ndscan`
   runs with `read_host_runtime_snapshot(...)`.
6. Add a small lab-helper wrapper for the imaging blob used by the three-image example.
7. Add one offline round-trip test:
   - open HDF5
   - read blob
   - recompute one saved live statistic from raw counts/images
   - compare against the persisted `ndscan` output

This is the first major milestone. Once it works, the storage contract is good enough
for real offline analysis.

### Phase 3: Grow reusable offline helpers

8. Add generic result-navigation helpers in `ndscan.results` where repeated patterns
   emerge.
9. Move any broadly useful schema-level helpers out of examples and into proper
   packages, but keep lab-specific semantics in the lab helper repository.
10. Keep the lab helper repository responsible for physics/statistics/domain logic.

### Phase 4: Unify live and offline plotting logic further

11. Identify pure-Python trace extraction logic currently trapped inside the runtime
    viewer.
12. Factor that logic into reusable helpers that accept `HostRuntimeSiteData` plus plot
    state/configuration.
13. Use those helpers from:
    - the live Qt viewer
    - offline plotting tools
    - lab-helper plotting code where appropriate

This is the longer-term "plot in the same way" step.

## Immediate Next Step

The next concrete step should be:

1. implement structured site-blob retrieval in `ndscan.results`
2. persist an imaging-readout blob in
   [examples/host_runtime_three_image_rearrangement.py](/home/lab/artiq-files/install/ndscan/examples/host_runtime_three_image_rearrangement.py)
3. add one test proving offline reconstruction of that blob from the saved HDF5

That is the smallest step that moves the architecture forward without overcommitting to
lab-specific abstractions inside `ndscan` core.

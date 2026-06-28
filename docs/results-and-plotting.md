# Results And Plotting

The prepared runtime writes a scan-site tree. Offline readers and live plotting both
adapt that tree into the same result model.

## Offline Model

`read_scan_site_snapshot(path)` reads a prepared-runtime HDF5 result file and returns a
`ScanSiteSnapshot`.

`ScanSiteSnapshot`
: File-level object containing top-level HDF5 metadata and a mapping of site paths to
`ScanSiteData`.

`ScanSiteData`
: One scan site. It exposes schemas, raw point streams, analysis outputs/artifacts,
annotations, segment metadata, and convenience methods for common series access.

The public package `ndscan.results` re-exports the normal entry points.

## Live Model

The runtime applet receives live ARTIQ dataset updates and builds an equivalent
`ScanSiteSnapshot` in memory. The viewer should not need to care whether the source is a
live dataset map or an HDF5 file.

## Plotting Role

`ndscan.plots.runtime` is a convenience layer over the scan-site schema. It lives in this
repository because it needs to understand the persisted schema and result-channel
metadata, but it should remain isolated from execution.

The viewer is expected to:

- show one scan site per column,
- let the user navigate child sites through parent point selection,
- choose x/y/z series from numeric point streams,
- render final and online analysis outputs where useful,
- work from both live snapshots and offline snapshots.

## Analysis Artifacts

Final analysis outputs are values intended for programmatic consumption. Artifacts are
richer payloads intended for viewers or notebooks. Online analysis uses parallel
`analysis.online_*` streams.

Lab-specific artifact interpretation should live outside core ndscan unless it is a
general schema feature.

## Current Pressure Points

The result model is conceptually stable, but two implementation areas still need
consolidation:

- `ndscan.results.scan_site_reader` and `ndscan.plots.runtime.live` duplicate decoding
  and extraction logic.
- `ndscan.plots.runtime.viewer` contains many independent responsibilities and should be
  split once behavior is stable.

# Scan-Site Schema

The scan-site schema is the persisted dataset contract for prepared-runtime results.
The current revision is `8`, defined by `SCAN_SITE_SCHEMA_REVISION` in
`ndscan.schema.scan_site`.

Each scan site is written under a prefix such as:

```text
ndscan.rid_<rid>.site.root.
ndscan.rid_<rid>.site.root.subscans.child_name.
ndscan.rid_<rid>.site.root.subscans.child_name.subscans.grandchild_name.
```

Child scan sites live below the reserved `subscans` key of their parent site. Human
readable child names live in `site.path`; using the reserved `subscans` namespace means
child names such as `points` or `segments` cannot collide with per-site datasets. The
root site has path `[]`. Child sites have non-empty paths and record their parent path.
Dataset-prefix path components preserve ASCII letters, digits, and `_`; other UTF-8
bytes are encoded as `~xx`, and an empty component is encoded as `~empty`.

## Required Metadata

Each site writes:

`ndscan_schema_revision`
: Current scan-site schema revision.

`site.source_id`
: Source/run identifier derived from the ARTIQ scheduler RID.

`site.path`
: JSON/list path for this site.

`site.parent_path`
: JSON/list parent path, only for child sites.

`site.fragment_fqn`
: Fully qualified fragment class/name for the site.

`site.start_unix_time`
: Unix timestamp for the site invocation when available.

`state.completed`
: Boolean completion flag.

`state.num_points`
: Number of completed points currently written.

## Scan Description

`scan.point_policy`
: Description of the point policy used for this site.

`scan.axes`
: Ordered list of logical scan axes. Each entry contains:

  - `index`: zero-based logical axis index,
  - `axis_key`: runtime axis key, e.g. `axis_0`,
  - `storage_key`: point-stream key, e.g. `param_0` or `pseudoparam_0`,
  - `kind`: `parameter` or `pseudoparam`,
  - `path`: public series path accepted by result readers,
  - `description`, `unit`, `scale`, `type`: display/analysis hints copied from the
    source schema when available.

  This is the canonical once-per-site source for axis order. The full parameter and
  pseudoparam schemas still live in `scan.parameters` and `scan.pseudoparams`.

`scan.pseudoparams`
: Logical scanned variables that produce `points.pseudoparam_*` streams.

`scan.fixed_pseudoparams`
: Logical variables fixed for the whole site.

`scan.parameters`
: Concrete fragment parameters whose values are recorded point-by-point.

`scan.fixed_parameters`
: Concrete fragment parameters fixed for the whole site.

`scan.channels`
: Saved result channel descriptions.

`scan.parameter_mappings`
: Parameter mappings used to derive concrete parameters from scan variables.

## Point Streams

Point streams are append-only arrays below `points.`:

`points.pseudoparam_<name>`
: Logical pseudoparam value for each completed point.

`points.param_<name>`
: Concrete fragment parameter value for each completed point.

`points.channel_<name>`
: Result-channel value for each completed point.

`points.metadata.<name>`
: Runtime point metadata.

`points.acquired_at_unix`
: Optional point acquisition timestamps.

The point index is implicit in the array position. Readers may expose an artificial
`__point_index__` choice for plotting and analysis.

## Batches

Batches record which flat point rows were completed and published together. They are
execution metadata, not scan coordinates.

`batches.start_index`
: Start point index for each completed execution batch.

`batches.start_unix_time`
: Host-side Unix timestamp taken once when the completed batch is published.

The point interval for batch `i` is:

```text
[batches.start_index[i], batches.start_index[i + 1])
```

For the final batch, the stop index is `state.num_points`. The batch length is therefore
derived from the interval rather than stored. Per-point acquisition timestamps, when
available, stay in `points.acquired_at_unix`.

## Segments

Segmented sites represent repeated child-scan invocations or other logical runs inside
one flat site.

`segments.start_index`
: Start point index for each segment.

`segments.start_unix_time`
: Optional per-segment Unix timestamp.

`segments.parent_point_index`
: Parent point index for child sites.

`state.current_segment`
: Current open segment, or `-1` when no segment is open.

`segments.analysis.final_feedback`
: JSON payload containing final segment analysis outputs, artifacts, and annotations.

## Analysis

`analysis.outputs`
: Schema for final analysis outputs.

`analysis.output.<name>`
: Final analysis output value.

`analysis.artifact.<name>`
: Final analysis artifact payload.

`analysis.online`
: Schema for online analysis outputs.

`analysis.online_result.<analysis>.<name>`
: Online analysis output stream.

`analysis.online_artifact.<analysis>.<name>`
: Online analysis artifact stream.

`analysis.annotations`
: Final static annotations.

`analysis.online_annotation.<analysis>`
: Online annotation stream.

## Extra Metadata

Extra site metadata is written below `extra.`. Core ndscan readers preserve these values
but do not assign lab-specific meaning to them.

## Reader API

Use `read_scan_site_snapshot(path)` from `ndscan.results` to read a final or preview HDF5
file. It returns a `ScanSiteSnapshot` containing `ScanSiteData` objects for each site.

`ScanSiteData.axes` exposes the ordered logical axis metadata. Use
`ScanSiteData.axis_paths()` for public series paths and
`ScanSiteData.axis_storage_keys()` for the corresponding `points.*` storage keys.
`ScanSiteData.batches()` exposes the completed execution batches as flat point-index
intervals.

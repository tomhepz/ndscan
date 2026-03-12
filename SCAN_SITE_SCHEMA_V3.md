# Scan-Site Schema v3 Proposal

This document proposes a cleaned-up scan-site schema for the new host runtime.

It is a design note, not an implemented change.

The goals are:

- keep one canonical flat representation for root scans and nested scan sites,
- make the naming scheme more regular and easier to read,
- remove keys that only exist to explain other keys,
- make the format easy to consume without any ARTIQ runtime dependency,
- make the contract explicit around JSON, HDF5, and non-ragged arrays.

## Scope

This proposal only covers the new scan-site layout written by the host runtime.

It does not try to redesign:

- the legacy `entry_point.py` dataset layout,
- the legacy `subscan.py` compatibility exports,
- any dashboard-side internal model beyond what a reader can infer from the stored data.

## Design Principles

### 1. A scan site is self-contained

Every site should contain enough metadata to be understood on its own:

- what fragment it scanned,
- what axes and channels exist,
- what point source created the points,
- how nested segments map back to parent points,
- what final analysis outputs and annotations were produced.

### 2. Readers should not need ARTIQ

The stable contract should be:

- HDF5 scalars and arrays for numeric data,
- UTF-8 JSON strings for composite metadata,
- no PYON-specific semantics required by offline readers.

In practice, the current implementation already behaves more like "HDF5 + JSON" than
"raw PYON". Schema v3 should make that explicit.

### 3. Fixed field names are better than indirection

Keys such as `segment_fields` and `segment_state_fields` are not carrying new
information. They only tell the reader where the real data lives.

If the schema itself defines the dataset names, those keys should be removed.

### 4. Structural ids should be machine-stable, names should live in metadata

Point datasets are naturally positional:

- `points.axis_0`
- `points.channel_0`

That is fine. The human meaning belongs in metadata:

- axis `path`, `param.fqn`, `description`
- channel `path`, `description`, `type`, `unit`, ...

What should be avoided is mixing positional ids in some places with free-form names in
others without a clear mapping.

### 5. The site tree should be explicit, not mirrored into parent metadata

Nested scan sites should be discovered as scan sites in their own right.

The parent site should not need to list all child sites explicitly. Readers should be
able to discover all sites by scanning dataset prefixes and reading:

- `site.path`
- `site.parent_path`

This scales better when:

- one run launches multiple different child scans,
- one child scan is repeated across many parent points,
- the same fragment type is scanned in more than one role.

## Proposed Prefix Convention

Keep the current structural prefix convention:

```text
ndscan.rid_<rid>.site.root.
ndscan.rid_<rid>.site.root.<child_name>.
ndscan.rid_<rid>.site.root.<child_name>.<grandchild_name>.
```

This part is already good:

- it is human-readable,
- it is hierarchical,
- it does not require a parent schema to discover child sites.

Repeated execution of the same logical child scan should reuse the same site prefix and
append new segments.

Distinct logical scans, even of the same fragment type, should use distinct sibling
site names.

## Proposed v3 Dataset Layout

For a given site prefix `<site>`, the stable v3 datasets are:

### Required identity and structure

```text
<site>ndscan_schema_revision
<site>site.path
<site>site.fragment_fqn
<site>site.source_id
<site>scan.axes
<site>scan.channels
<site>scan.point_source
```

### Optional nested-site identity

```text
<site>site.parent_path
```

### Point data

```text
<site>points.axis_0
<site>points.axis_1
...
<site>points.channel_0
<site>points.channel_1
...
```

### Optional segmentation

```text
<site>segments.start_index
<site>segments.parent_point_index
```

### Optional live-state / acquisition-state

```text
<site>state.current_segment
<site>state.num_points
<site>state.completed
```

### Optional analysis outputs

```text
<site>analysis.online
<site>analysis.online_result.<name>
<site>analysis.online_annotation.<name>
<site>analysis.outputs
<site>analysis.output.<name>
<site>analysis.annotations
```

### Optional user-defined metadata

```text
<site>extra.<name>
```

## Meaning of Each Dataset Group

### `ndscan_schema_revision`

Integer schema revision for this site.

This keeps compatibility with existing root-discovery code, which already scans for
`ndscan_schema_revision`.

### `site.*`

Identity of the scan site itself.

- `site.path`: JSON list of structural site path components relative to `root`.
- `site.parent_path`: JSON list of the parent site path, if this is nested.
- `site.fragment_fqn`: fully-qualified fragment class name.
- `site.source_id`: stable run/source identifier.

Example:

```json
site.path = ["scan_p", "scan_x"]
site.parent_path = ["scan_p"]
site.fragment_fqn = "host_runtime_nested_p_variation.LineFragment"
site.source_id = "rid_2460"
```

### `scan.*`

Static scan description.

- `scan.axes`: JSON object keyed by `axis_<n>`.
- `scan.channels`: JSON object keyed by `channel_<n>`.
- `scan.point_source`: JSON object describing the point source kind and parameters.
- `scan.parameter_mappings`: optional JSON object keyed by `mapping_<n>`.

Example:

```json
scan.axes = {
  "axis_0": {
    "path": "scan_x/line",
    "param": {
      "fqn": "host_runtime_nested_p_variation.LineFragment.x",
      "description": "x",
      "type": "float",
      "default": "0.0"
    }
  }
}
```

Logical runtime-only axes are represented explicitly rather than being forced into the
fragment-parameter schema:

```json
scan.axes = {
  "axis_0": {
    "path": "",
    "variable": {
      "name": "laser_frequency",
      "description": "Logical frequency axis",
      "type": "float",
      "spec": {}
    }
  }
}
```

```json
scan.channels = {
  "channel_0": {
    "path": "scan_x/line/y",
    "description": "",
    "type": "float",
    "scale": 1.0,
    "unit": ""
  }
}
```

Using objects keyed by `axis_0` / `channel_0` removes the redundant `"key"` field and
makes the mapping to `points.axis_0` / `points.channel_0` direct.

Parameter mappings record how logical axes and existing parameter values are converted
into concrete parameter-store updates before each point runs:

```json
scan.parameter_mappings = {
  "mapping_0": {
    "description": "Offset the physical drive from the logical axis",
    "targets": [
      {
        "path": "",
        "param": {
          "fqn": "example.HardwareDriveFragment.drive",
          "description": "drive",
          "type": "float"
        }
      }
    ],
    "dependencies": [
      {
        "kind": "axis",
        "axis": "axis_0"
      }
    ]
  }
}
```

### `points.*`

Flat append-only point datasets.

The leading axis is always point index.

Allowed shapes:

- scalar per point: dataset shape `(num_points,)`
- fixed-shape vector/matrix per point: dataset shape `(num_points, ...)`

Not allowed in the canonical schema:

- ragged arrays,
- arbitrary Python objects,
- non-rectangular nested lists.

If a channel produces data that cannot satisfy this contract, it should not be written
to `points.channel_*` in the canonical schema.

### `segments.*`

Logical segmentation of one flat point stream.

- `segments.start_index[i]` is the flat point index where segment `i` begins.
- `segments.parent_point_index[i]` is the parent site point index that launched
  segment `i`, when the site is nested.

These datasets exist only for segmented sites.

If `segments.parent_point_index` is absent, the site is segmented but not nested under
parent points.

### `state.*`

Live-state convenience values.

- `state.current_segment`: currently open segment index, or `-1` when idle.
- `state.num_points`: current flat point count.
- `state.completed`: whether the site is currently complete.

These are useful for live subscribers and dataset-driven plotting. They are not needed
to reconstruct the final offline result if the run completed normally.

Readers should therefore treat them as optional convenience state, not as core schema.

### `analysis.*`

Everything related to fragment-defined or runtime-defined analyses.

- `analysis.online`: JSON object describing online/in-loop analyses.
- `analysis.online_result.<name>`: latest batch-updated result object for online
  analysis `<name>`.
- `analysis.online_annotation.<name>`: latest batch-updated annotation list for online
  analysis `<name>`.
- `analysis.outputs`: JSON object describing final analysis output channels.
- `analysis.output.<name>`: final value for analysis output `<name>`.
- `analysis.annotations`: JSON list of produced annotations.

This naming is intentionally regular:

- analysis schema lives under `analysis.*`,
- live online-analysis values live under `analysis.online_result.*`,
- live online-analysis annotations live under `analysis.online_annotation.*`,
- analysis values live under `analysis.output.*`,
- there is no longer a mismatch between `online_analyses` and `analysis_results`.

### `extra.*`

Caller-supplied metadata that is meaningful to the experiment author but not part of
the stable scan-site contract.

Examples:

- `extra.demo_name`
- `extra.analysis_note`
- `extra.sample_id`

This keeps the reserved schema namespace small and prevents user metadata from being
mixed with structural keys like `fragment_fqn` or `completed`.

## Removed v2 Keys

Schema v3 should remove these keys from the stable contract:

### `runtime_flavour`

Reason:

- useful for debugging,
- not required by a reader that already keys off schema revision,
- couples stored data to implementation flavour rather than schema contract.

If debugging metadata is still desired, it can live under `extra.runtime_flavour`, not
the stable schema.

### `segment_fields`

Reason:

- it only points to `starts` and `parent_point_indices`,
- the schema itself should define those names directly.

Replaced by fixed datasets:

- `segments.start_index`
- `segments.parent_point_index`

### `segment_state_fields`

Reason:

- same issue as `segment_fields`,
- it only points to `current_segment`.

Replaced by fixed dataset:

- `state.current_segment`

### Top-level free-form user keys

Examples:

- `demo_name`
- `analysis_note`

Reason:

- they collide conceptually with reserved keys,
- they make it harder for generic readers to know which fields are schema and which are
  experiment-specific.

Replaced by:

- `extra.demo_name`
- `extra.analysis_note`

## Naming Mapping From Current v2

Current v2 keys and the proposed v3 replacements:

| Current key | Proposed v3 key |
| --- | --- |
| `fragment_fqn` | `site.fragment_fqn` |
| `source_id` | `site.source_id` |
| `site_path` | `site.path` |
| `parent_site_path` | `site.parent_path` |
| `axes` | `scan.axes` |
| `channels` | `scan.channels` |
| `point_source` | `scan.point_source` |
| `online_analyses` | `analysis.online` |
| `analysis_results` | `analysis.outputs` |
| `analysis_result.<name>` | `analysis.output.<name>` |
| `annotations` | `analysis.annotations` |
| `starts` | `segments.start_index` |
| `parent_point_indices` | `segments.parent_point_index` |
| `current_segment` | `state.current_segment` |
| `num_points` | `state.num_points` |
| `completed` | `state.completed` |
| `runtime_flavour` | removed or `extra.runtime_flavour` |
| `segment_fields` | removed |
| `segment_state_fields` | removed |
| user top-level keys | `extra.<name>` |

## Serialization Contract

Schema v3 should make the following guarantees:

### Composite metadata

Composite metadata is always stored as JSON text using only vanilla JSON-compatible
types:

- object
- array
- string
- number
- boolean
- null

This includes:

- `site.path`
- `site.parent_path`
- `scan.axes`
- `scan.channels`
- `scan.point_source`
- `analysis.online`
- `analysis.outputs`
- `analysis.annotations`

Offline readers should not be required to decode PYON.

### Point and analysis datasets

Point datasets and `analysis.output.*` datasets must be HDF5-storable as either:

- scalar values,
- 1-D arrays,
- fixed-shape higher-rank arrays.

Ragged arrays are out of contract for the canonical schema.

### Unknown keys

Readers should ignore unknown keys below a site prefix.

This allows future additions under namespaced groups like:

- `extra.*`
- `state.*`

without breaking older readers.

## Reader Model

A non-ARTIQ reader should only need this algorithm:

1. Find every site prefix by scanning for keys ending in `ndscan_schema_revision`.
2. For each prefix:
   - read `site.path`,
   - read `site.parent_path` if present,
   - read `scan.axes`,
   - read `scan.channels`,
   - load all `points.axis_*` and `points.channel_*` datasets.
3. If `segments.start_index` exists:
   - split the flat point arrays into segments,
   - use `segments.parent_point_index` to map nested segments back to parent points.
4. Read `analysis.output.*` and `analysis.annotations` if needed.

That is enough to build:

- a generic offline analysis tool,
- a nested-scan visualiser,
- a plain Python export pipeline.

## Multiple Child Scans Per Run

The parent site should not directly enumerate child site names in its own metadata.

Instead:

- each child site has its own prefix,
- hierarchy is reconstructed from `site.path` / `site.parent_path`,
- repeated invocations of the same logical child scan use one site plus many segments.

This naturally supports:

- one parent running several different child scans,
- the same child fragment type used in several roles,
- one child scan repeated for many parent points.

The only requirement is that sibling site path components must be unique within the
parent site.

## Recommended Implementation Notes

If schema v3 is adopted, the write-side changes should stay mostly local to:

- `ndscan/experiment/scan_site.py` for key names and grouping,
- `ndscan/experiment/host_runtime.py` for metadata content shape.

The schema should not require any parent site to know its children ahead of time.

## Summary

The main v3 changes are:

- keep the current hierarchical site prefix convention,
- group metadata into `site.*`, `scan.*`, `analysis.*`, `segments.*`, `state.*`,
- move user metadata under `extra.*`,
- replace metadata lists with id-keyed objects where appropriate,
- drop `segment_fields` and `segment_state_fields`,
- define the contract as HDF5 + JSON, not PYON-dependent,
- forbid ragged point data in the canonical flat schema.

That keeps the recursive scan-site model, but makes the schema smaller, more regular,
and easier for generic code to consume.

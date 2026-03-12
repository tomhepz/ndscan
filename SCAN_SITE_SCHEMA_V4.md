# Scan-Site Schema v4

This document describes the current implemented scan-site schema written by the new
host runtime.

It supersedes the earlier v3 proposal. The most important changes are:

- logical runtime-only scan variables are now `pseudoparam_*`
- actual varying fragment parameters are now `param_*`
- result channels remain `channel_*`
- repeated acquisition does not introduce a native repeat axis

## Scope

This schema applies only to the new host runtime.

It does not describe:

- the legacy `entry_point.py` top-level dataset layout
- the legacy `subscan.py` compatibility exports

## Design Principles

### 1. One site is self-contained

Every scan site should contain enough metadata to be understood on its own.

### 2. Readers should not need ARTIQ

Composite metadata is written as JSON-compatible values. Offline readers should be
able to work from HDF5 scalars, arrays, and JSON strings alone.

### 3. Point-like data is split by role

- `pseudoparam_*`: logical runtime-only quantities
- `param_*`: actual varying fragment parameters
- `channel_*`: results

### 4. Repetition is not a native axis

If a point is repeated, the repeated observations simply appear as repeated points in
the site. A repeat index only exists if the user explicitly models one as a
`ScanVariable`.

## Prefix Convention

Top-level site:

```text
ndscan.rid_<rid>.site.root.
```

Nested sites:

```text
ndscan.rid_<rid>.site.root.<child_name>.
ndscan.rid_<rid>.site.root.<child_name>.<grandchild_name>.
```

## Required Datasets

For one site prefix `<site>`, the runtime may write:

### Identity and structure

```text
<site>ndscan_schema_revision
<site>site.path
<site>site.fragment_fqn
<site>site.source_id
<site>state.completed
<site>state.num_points
```

### Optional site identity

```text
<site>site.parent_path
<site>site.start_unix_time
```

### Scan description

```text
<site>scan.point_source
<site>scan.pseudoparams
<site>scan.parameters
<site>scan.channels
<site>scan.parameter_mappings
```

### Point data

```text
<site>points.pseudoparam_0
<site>points.pseudoparam_1
...
<site>points.param_0
<site>points.param_1
...
<site>points.channel_0
<site>points.channel_1
...
<site>points.acquired_at_unix
```

### Segmentation for nested/repeated child sites

```text
<site>segments.start_index
<site>segments.start_unix_time
<site>segments.parent_point_index
<site>state.current_segment
```

### Analysis

```text
<site>analysis.online
<site>analysis.online_result.<name>
<site>analysis.online_annotation.<name>
<site>analysis.outputs
<site>analysis.output.<name>
<site>analysis.annotations
```

### User metadata

```text
<site>extra.<name>
```

## Dataset Semantics

### `scan.pseudoparams`

Logical scan variables declared with `ScanVariable`.

Example:

```json
{
  "pseudoparam_0": {
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

### `scan.parameters`

Actual fragment parameters whose installed values varied during the scan.

Each entry includes:

- `path`
- `param`
- `is_scanned`
- `scan_role`

`scan_role` is currently:

- `"direct"` for directly scanned real fragment parameters
- `"derived"` for mapping targets

Example:

```json
{
  "param_0": {
    "path": "",
    "param": {
      "fqn": "example.Fragment.logical_drive",
      "description": "logical drive",
      "type": "float",
      "default": "0.0"
    },
    "is_scanned": true,
    "scan_role": "direct"
  },
  "param_1": {
    "path": "hardware",
    "param": {
      "fqn": "example.HardwareFragment.drive",
      "description": "drive",
      "type": "float",
      "default": "0.0"
    },
    "is_scanned": false,
    "scan_role": "derived"
  }
}
```

### `scan.channels`

Saved result channels, keyed by `channel_<n>`.

### `scan.parameter_mappings`

Optional metadata describing per-point parameter mappings.

Dependencies can refer to:

- a `pseudoparam_*`
- a `param_*`
- an unscanned fragment parameter

Example:

```json
{
  "mapping_0": {
    "description": "Offset child drive from the wrapper's logical axis",
    "targets": [
      {
        "path": "hardware",
        "param": { "...": "..." }
      }
    ],
    "dependencies": [
      {
        "kind": "parameter",
        "key": "param_0"
      }
    ]
  }
}
```

## Point Data Semantics

### `points.pseudoparam_*`

Append-only arrays of logical runtime-only scan quantities.

### `points.param_*`

Append-only arrays of actual installed fragment parameter values.

This includes:

- direct scanned parameters
- parameter-mapping targets

### `points.channel_*`

Append-only arrays of saved result values.

### `points.acquired_at_unix`

Host-recorded Unix timestamp for each completed point.

## Repeated Points

Repeated points do not create a native repeat axis.

If a point policy repeats one logical point multiple times, then:

- `points.param_*` / `points.pseudoparam_*` simply repeat the same values
- `points.channel_*` records the repeated observations
- analyses can aggregate identical parameter values if that is scientifically useful

If a repeat index is scientifically meaningful, it should be introduced explicitly as a
`ScanVariable`. In that case it appears as a `pseudoparam_*`.

## Nested Sites and Segments

Nested sites are discovered structurally:

- `site.path`
- `site.parent_path`

Repeated child-site invocations are segmented by:

- `segments.start_index`
- `segments.parent_point_index`
- optionally `segments.start_unix_time`

`state.current_segment` is live-state information for a currently open segment.

## Analysis Datasets

### Online analysis

Batch-updated snapshots:

- `analysis.online`
- `analysis.online_result.<name>`
- `analysis.online_annotation.<name>`

These are latest-value state, not append-only histories.

### Final analysis

End-of-run outputs:

- `analysis.outputs`
- `analysis.output.<name>`
- `analysis.annotations`

## Minimal Reader Contract

An offline reader can reconstruct a site by:

1. discovering prefixes with `ndscan_schema_revision == 4`
2. reading:
   - `site.path`
   - `site.parent_path` if present
   - `scan.pseudoparams`
   - `scan.parameters`
   - `scan.channels`
3. loading all `points.pseudoparam_*`, `points.param_*`, and `points.channel_*`
4. reading segment datasets if present
5. reading online/final analysis datasets if needed

No ARTIQ runtime object model is required.

## Notes

- The older `SCAN_SITE_SCHEMA_V3.md` file is now historical.
- The legacy runtime still uses different dataset naming and should not be mixed with
  this schema.

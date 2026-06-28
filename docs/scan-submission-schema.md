# Scan Submission Schema

The scan-submission schema is the dashboard/transport format for prepared scans. It is
not the runtime model itself. The worker compiles it to:

- a `ScanRequest`, and
- fixed concrete parameter overrides.

The top-level ARTIQ argument payload key is `scan_submission`.

## Top-Level Shape

```json
{
  "scan_submission": {
    "version": 1,
    "mode": {"type": "grid"},
    "entries": [],
    "execution": {},
    "metadata": {}
  }
}
```

Inside the `scan_submission` object:

`version`
: Integer schema revision. Currently `1`.

`mode`
: Top-level scan mode. Currently `grid` or `gpo`.

`entries`
: Parameter and pseudoparam rows.

`execution`
: Execution settings such as `max_points_per_batch`.

`metadata`
: User metadata copied to the resulting `ScanRequest`.

## Entry Targets

Entries can target real fragment parameters:

```json
{
  "id": "detuning",
  "kind": "param",
  "target": {"fqn": "example.Fragment.detuning", "path": ""},
  "mode": {"type": "scan", "generator": {"type": "linear", "range": {}}}
}
```

or pseudoparams:

```json
{
  "id": "logical_frequency",
  "kind": "pseudoparam",
  "description": "logical frequency",
  "mode": {"type": "fixed", "value": 1000000.0}
}
```

`id` is the stable symbol used by groups, text rebinds, and generated pseudoparam
names. It must be unique within one scan submission.

## Entry Modes

`fixed`
: Use one fixed value.

`scan`
: Grid-mode finite generator. Valid only when the top-level mode is `grid`.

`gpo_scan`
: Optimiser dimension with lower/upper bounds. Valid only when the top-level mode is
`gpo`.

`rebind`
: Text expression that maps one or more row symbols onto a concrete parameter. Valid
only for real parameter targets.

## Grid Mode

Grid mode compiles finite scanned rows into point policies. Rows can be grouped so
multiple dimensions advance together. Ungrouped scanned rows are combined as a Cartesian
product. Fixed rows become fixed overrides or fixed pseudoparams.

Global randomisation is represented by `mode.randomise_order_globally`.

## GPO Mode

GPO mode compiles `gpo_scan` rows into optimiser dimensions and uses an objective saved
result channel.

The supported objective shape is:

```json
{
  "kind": "channel",
  "target": {"path": "result_channel_path"}
}
```

Backend options live under `mode.backend`.

## Compiler API

Use:

- `ScanSubmissionSpec.from_dict(...)` for validation and typed access,
- `compile_scan_submission_spec(fragment, spec)`,
- `compile_scan_submission_schema(fragment, mapping)`.

The compiler is in `ndscan.submission.scan_submission_schema`.

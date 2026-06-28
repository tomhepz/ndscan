# Host Scan Submission Schema

This document proposes the first concrete dashboard submission schema for the
prepared-runtime path.

It is intentionally shaped around the current row-oriented UI:

- the user sees a list of parameters,
- each row declares how that parameter is driven,
- a top-level mode decides whether the run is a normal grid scan or a Gaussian
  process optimisation (GPO),
- pseudoparameters can be added at the top and then used exactly like ordinary
  numeric rows.

This document is a proposal, not a description of code already implemented.

## Goals

The schema should:

- fit naturally into the existing `ndscan_params` dashboard payload,
- preserve the current per-parameter editing workflow,
- support pseudoparameters in both grid and GPO modes,
- compile cleanly into:
  - `overrides`,
  - `ScanVariable`s,
  - `ParameterMapping`s,
  - `PointPolicy`,
  - `ExecutionPolicy`,
  - `ScanRequest`.

## Non-Goals

The first version does not try to cover:

- arbitrary explicit point tables,
- text-defined multi-step strategies beyond rebind expressions,
- no-axis `repeat`/`time_series` compatibility with the legacy runner,
- full parity with every old generator in every mode,
- rebind chains between derived symbols.

## Outer `ndscan_params` Envelope

The existing envelope stays the same:

```python
{
    "instances": ...,
    "schemata": ...,
    "always_shown": ...,
    "overrides": ...,
    "scan": ...,       # legacy mode only
    "scan_submission": ...,  # new runtime only
}
```

Rules:

- `scan` and `scan_submission` are mutually exclusive.
- `overrides` remains shared between legacy and host modes.
- `scan_submission` is only interpreted by the new prepared-runtime adapter.

## Top-Level `scan_submission`

```python
{
    "version": 1,
    "mode": {
        "type": "grid" | "gpo",
        # when type == "gpo":
        "objective": {...},
        "backend": {...},
    },
    "entries": [...],
    "metadata": {...},
    "execution": {...},
}
```

Fields:

- `version`
  - integer schema revision for the scan-submission format
- `mode`
  - top-level behaviour:
    - `{"type": "grid"}` for ordinary scans
    - `{"type": "gpo", ...}` for Gaussian process optimisation
- `entries`
  - parameter rows and pseudoparameter rows
- `metadata`
  - optional request metadata copied through to `ScanRequest.metadata`
- `execution`
  - runtime scheduling knobs that belong in `ExecutionPolicy`
- top-level mode metadata lives inside the `mode` object itself

## Entries

Each row in the host editor lowers to one `entry`.

```python
{
    "id": "detuning",
    "kind": "param" | "pseudoparam",
    "target": {"fqn": "...", "path": "*"},  # required for kind="param"
    "description": "Logical drive",          # optional, pseudoparams mainly
    "default": 0.0,                          # optional, pseudoparams mainly
    "mode": {
        "type": "fixed" | "scan" | "gpo_scan" | "rebind",
        ...
    },
}
```

Rules:

- `id` is required and must be unique within one `scan_submission`.
- `id` is the identifier used in rebind expressions.
- `kind="param"` refers to a real fragment parameter selected by `(fqn, path)`.
- `kind="pseudoparam"` declares a runtime-only logical variable that will compile to a
  `ScanVariable`.
- pseudoparameters should be creatable from a dedicated top-of-editor action, but once
  created they follow the same row model as ordinary parameters.

## Row Modes

The available row modes depend on the top-level `scan_submission.mode.type`.

### Grid Mode

When `scan_submission.mode.type == "grid"`, each row may use:

- `fixed`
- `scan`
- `rebind`

### GPO Mode

When `scan_submission.mode.type == "gpo"`, each row may use:

- `fixed`
- `gpo_scan`
- `rebind`

This keeps the UI simple:

- in ordinary scans, rows either scan directly or derive from scanned values,
- in optimiser mode, rows either define bounded optimisation dimensions, remain fixed,
  or derive from optimisation dimensions.

## `fixed`

```python
{
    "mode": {
        "type": "fixed",
        "value": 1.23
    }
}
```

Semantics:

- for `kind="param"`:
  - compile to a fixed override
- for `kind="pseudoparam"`:
  - compile to a constant symbol available to rebind expressions
  - not a scan axis

## `scan`

```python
{
    "mode": {
        "type": "scan",
        "generator": {
            "type": "linear" | "centre_span" | "list" | "refining" | ...,
            "range": {...},
        },
        "group": "pulse" | None
    }
}
```

Semantics:

- only valid when `scan_submission.mode.type == "grid"`
- declares a directly scanned variable
- `group` is the scan group previously discussed as a zip group
- rows with the same non-empty `group` are zipped together
- rows in different groups are orthogonal, i.e. combined by Cartesian product

`mode.generator` should reuse the existing legacy generator-style schema shape wherever
possible so the current scan widgets can be reused.

## `gpo_scan`

```python
{
    "mode": {
        "type": "gpo_scan",
        "lower": -1.0,
        "upper": 1.0
    }
}
```

Semantics:

- only valid when `scan_submission.mode.type == "gpo"`
- declares one bounded optimiser dimension
- this lowers to a logical optimisation axis in the GP input space

First-version restriction:

- `gpo_scan` is numeric only
- explicit finite lower and upper bounds are required

## `rebind`

```python
{
    "mode": {
        "type": "rebind",
        "expr": "0.2 + 0.05 * drive"
    }
}
```

Semantics:

- intended primarily for `kind="param"`
- the parameter is not driven directly by a scan or optimiser dimension
- its value is computed from other symbols and lowered to a `ParameterMapping`

First-version restriction:

- rebind expressions may reference only non-rebound symbols:
  - scanned rows
  - optimiser-dimension rows
  - fixed pseudoparameters
- rebind expressions may not reference other rebind rows

This keeps the dependency graph one layer deep and avoids cycle handling in the first
compiler.

## Scan Groups in Grid Mode

When `scan_submission.mode.type == "grid"`, active scanned rows are partitioned by
`entry.mode.group`.

Rules:

- scanned rows with `group = null` or omitted are placed in singleton groups
- scanned rows with the same non-empty `group` are zipped together
- different groups are combined orthogonally

Compiler lowering:

- one singleton group -> ordinary 1D axis
- one multi-entry group -> `ZipPointPolicy`
- product across groups -> `ProductPointPolicy`

Each unique group therefore behaves like one logical axis of an ND scan, even if that
axis drives multiple concrete parameters together.

### Shortest-Length Rule

Zip groups should follow ordinary zip semantics:

- the group stops when the shortest member stops
- longer generators are truncated

This means equal lengths are not required for correctness. However, the dashboard
should warn when a zip group mixes generators with different finite lengths so the user
understands that some points will be dropped.

### Recommended First-Version Restrictions

The cleanest initial implementation is:

- allow only generators whose finite point sequence can be materialised up front inside
  zip groups
- warn on length mismatch
- reject genuinely open-ended or level-driven generators in zip groups if they do not
  yet have a clean lowering to a finite point list

If needed, the first implementation can be narrower still and support only:

- `linear`
- finite `centre_span`
- `list`
- bool/enum-as-list

with later extension to refinement-style generators once their prepared-runtime lowering is
finalised.

## Gaussian Process Optimisation Mode

When `scan_submission.mode.type == "gpo"`, the scan request is built from:

- all rows with `mode.type == "gpo_scan"` as the GP input dimensions,
- all rows with `mode.type == "fixed"` as constants,
- all rows with `mode.type == "rebind"` as downstream parameter mappings.

This means:

- the GP lives in logical variable space,
- rebound parameters may depend on those dimensions,
- rebound parameters are not themselves GP dimensions.

That separation keeps the optimiser state clean and makes pseudoparameter-driven
reparameterisations natural.

## Top-Level GPO Metadata

```python
{
    "mode": {
        "type": "gpo",
        "objective": {
            "kind": "channel",
            "target": {"path": "probability"}
        },
        "backend": {
            "kind": "nubo",
            "batch_size": 4,
            "max_batches": 20,
            "initial_design_size": 8,
            "acquisition": "ucb"
        }
    },
}
```

Fields:

- `objective`
  - the scalar quantity to optimise
- `backend`
  - optimiser configuration that lowers to the existing ask/tell backend path

### Objective Selection

The first schema only needs scalar channel objectives:

```python
{
    "kind": "channel",
    "target": {"path": "probability"}
}
```

Future extensions can add:

- analysis-result objectives
- explicit noise/error channel pairing

but these are not required for the first compiler pass.

### First-Version GPO Restrictions

- every `mode.type == "gpo_scan"` row must be numeric
- every `mode.type == "gpo_scan"` row must have explicit finite lower/upper bounds
- rebinds are allowed, but only downstream of those dimensions
- scan-group metadata is irrelevant in GPO mode and should be ignored or rejected

In practical UI terms, this matches the intended "min/max over a chosen set of
parameters" model.

## Execution

```python
{
    "max_points_per_batch": 32
}
```

This lowers directly to `ExecutionPolicy`.

The first submission schema does not need to expose:

- preview HDF5 settings
- site metadata
- nested scan site placement

Those remain runtime/code concerns for now.

## Compiler Lowering Summary

Given a validated `scan_submission`, the compiler should:

1. Resolve each `kind="param"` target against the fragment tree.
2. Convert fixed real params into `overrides`.
3. Convert fixed/scanned/GPO-scanned pseudoparam entries into `ScanVariable`s or
   constants as appropriate.
4. Convert rebind entries into `ParameterMapping`s.
5. Lower:
   - `scan_submission.mode.type == "grid"` to zipped groups + Cartesian product
   - `scan_submission.mode.type == "gpo"` to the optimiser point policy
6. Produce:
   - `ScanRequest`
   - `overrides`

### Direct vs logical axes

When lowering directly driven `kind="param"` rows:

- if the target resolves to exactly one concrete handle:
  - the compiler may use the real `ParamHandle` directly as a scan axis
- if the target selector resolves to multiple handles:
  - the compiler should create one logical `ScanVariable`
  - and an identity `ParameterMapping` onto all matching handles

This keeps wildcard path selection compatible with the prepared runtime's core identity
model.

## Example: Grid Mode with Scan Groups and Rebind

```python
{
    "version": 1,
    "mode": {
        "type": "grid"
    },
    "entries": [
        {
            "id": "t",
            "kind": "param",
            "target": {"fqn": "pkg.Exp.t", "path": "*"},
            "mode": {
                "type": "scan",
                "generator": {
                    "type": "linear",
                    "range": {
                        "start": 0.0,
                        "stop": 10.0,
                        "num_points": 11,
                        "randomise_order": False
                    }
                },
                "group": None
            },
        },
        {
            "id": "drive",
            "kind": "pseudoparam",
            "description": "Logical drive",
            "default": 0.0,
            "mode": {
                "type": "scan",
                "generator": {
                    "type": "linear",
                    "range": {
                        "start": -1.0,
                        "stop": 1.0,
                        "num_points": 21,
                        "randomise_order": False
                    }
                },
                "group": "pair"
            },
        },
        {
            "id": "phase",
            "kind": "param",
            "target": {"fqn": "pkg.Exp.phase", "path": "*"},
            "mode": {
                "type": "scan",
                "generator": {
                    "type": "linear",
                    "range": {
                        "start": 0.0,
                        "stop": 180.0,
                        "num_points": 21,
                        "randomise_order": False
                    }
                },
                "group": "pair"
            },
        },
        {
            "id": "amp",
            "kind": "param",
            "target": {"fqn": "pkg.Exp.amp", "path": "*"},
            "mode": {
                "type": "rebind",
                "expr": "0.5 + 0.2 * drive"
            },
        }
    ],
    "execution": {"max_points_per_batch": 32}
}
```

Meaning:

- `drive` and `phase` are zipped together as one logical axis
- `t` is orthogonal to that zipped pair
- `amp` is derived from `drive`

## Example: GPO Mode with Pseudoparameters and Rebind

```python
{
    "version": 1,
    "mode": {
        "type": "gpo",
        "objective": {
            "kind": "channel",
            "target": {"path": "cost"}
        },
        "backend": {
            "kind": "nubo",
            "batch_size": 4,
            "max_batches": 20,
            "initial_design_size": 8,
            "acquisition": "ucb"
        }
    },
    "entries": [
        {
            "id": "x",
            "kind": "pseudoparam",
            "default": 0.0,
            "mode": {
                "type": "gpo_scan",
                "lower": -3.0,
                "upper": 3.0
            },
        },
        {
            "id": "y",
            "kind": "pseudoparam",
            "default": 0.0,
            "mode": {
                "type": "gpo_scan",
                "lower": -2.0,
                "upper": 2.0
            },
        },
        {
            "id": "physical_x",
            "kind": "param",
            "target": {"fqn": "pkg.Exp.x", "path": "*"},
            "mode": {"type": "rebind", "expr": "x"}
        },
        {
            "id": "physical_y",
            "kind": "param",
            "target": {"fqn": "pkg.Exp.y", "path": "*"},
            "mode": {"type": "rebind", "expr": "2.0 * y"}
        }
    ],
    "execution": {"max_points_per_batch": 4}
}
```

Meaning:

- GP lives in logical `(x, y)` space
- real experiment parameters are driven by rebind mappings
- no explicit point table is required in the schema

## Immediate Next Implementation Targets

The first code steps implied by this schema are:

1. scan-submission dataclasses / validation
2. compiler from `scan_submission` to `ScanRequest + overrides`
3. AST-based rebind expression compiler

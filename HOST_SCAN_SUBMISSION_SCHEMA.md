# Host Scan Submission Schema

This document proposes the first concrete dashboard submission schema for the
host-runtime path.

It is intentionally narrower than the full host runtime:

- it does not try to expose every `ScanRequest` field directly,
- it keeps the existing parameter-row UI model as much as possible,
- it adds only the new concepts needed to reach:
  - pseudoparameters,
  - rebind expressions,
  - zip-groups over finite scan variables,
  - Bayesian optimisation.

This document is a proposal, not a description of code already implemented.

## Goals

The schema should:

- fit naturally into the existing `ndscan_params` dashboard payload,
- preserve the current per-parameter editing workflow,
- separate legacy scan submission from host-runtime submission,
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
- full parity with every old generator inside zipped groups,
- UI formula chaining through arbitrary intermediate derived quantities.

## Outer `ndscan_params` Envelope

The existing envelope stays the same:

```python
{
    "instances": ...,
    "schemata": ...,
    "always_shown": ...,
    "overrides": ...,
    "scan": ...,       # legacy mode only
    "host_scan": ...,  # new runtime only
}
```

Rules:

- `scan` and `host_scan` are mutually exclusive.
- `overrides` remains shared between legacy and host modes.
- `host_scan` is only interpreted by the new host-runtime adapter.

## Top-Level `host_scan`

```python
{
    "version": 1,
    "entries": [...],
    "strategy": {...},
    "execution": {...},
}
```

Fields:

- `version`
  - integer schema revision for the host submission format
- `entries`
  - parameter rows and pseudoparameter rows
- `strategy`
  - how active scanned variables are combined
- `execution`
  - runtime scheduling knobs that belong in `ExecutionPolicy`

## Entries

Each row in the host editor lowers to one `entry`.

```python
{
    "id": "detuning",
    "kind": "param" | "pseudoparam",
    "target": {"fqn": "...", "path": "*"},  # required for kind="param"
    "description": "Logical drive",          # optional, pseudoparams mainly
    "default": 0.0,                          # optional, pseudoparams mainly
    "mode": {...}
}
```

Rules:

- `id` is required and must be unique within one `host_scan`.
- `id` is the identifier used in rebind expressions and strategy dimension lists.
- `kind="param"` refers to a real fragment parameter selected by `(fqn, path)`.
- `kind="pseudoparam"` declares a runtime-only logical variable that will compile to a
  `ScanVariable`.

## Entry Modes

### Fixed

```python
{
    "kind": "fixed",
    "value": 1.23
}
```

Semantics:

- for `kind="param"`:
  - compile to a normal fixed override
- for `kind="pseudoparam"`:
  - compile to a constant available to rebind expressions
  - not a scan axis

### Scan

```python
{
    "kind": "scan",
    "generator": {
        "type": "linear" | "centre_span" | "list" | ...,
        "range": {...}
    },
    "zip_group": "pulse" | None
}
```

Semantics:

- declares a scanned variable
- `zip_group` is optional
- entries with the same non-empty `zip_group` are zipped together
- entries in different groups are orthogonal, i.e. combined by Cartesian product

`generator` reuses the existing legacy generator schema shape, so current numeric scan
widgets can be reused where possible.

### Rebind

```python
{
    "kind": "rebind",
    "expr": "0.2 + 0.05 * drive"
}
```

Semantics:

- only valid for `kind="param"`
- this parameter is not directly scanned
- its value is computed from other variables and lowered to a `ParameterMapping`

First-version restriction:

- rebind expressions may reference only entry `id`s with:
  - `kind="scan"`
  - or `kind="fixed"` on pseudoparameters
- rebind expressions may not reference other rebind entries

This keeps the dependency graph one layer deep and avoids cycle handling in the first
compiler.

## Zip-Group Semantics

The `strategy.kind = "grid"` mode uses the active scanned entries and partitions them by
`zip_group`.

Rules:

- scanned entries with `zip_group = null` or omitted are placed in singleton groups
- scanned entries with the same `zip_group` are zipped together
- different groups are combined orthogonally

Compiler lowering:

- one singleton group -> ordinary 1D axis
- one multi-entry group -> `ZipPointPolicy`
- product across groups -> `ProductPointPolicy`

Example:

```python
entries = [
    {"id": "a", "kind": "param", "target": ..., "mode": {"kind": "scan", "generator": G1}},
    {"id": "b", "kind": "param", "target": ..., "mode": {"kind": "scan", "generator": G2, "zip_group": "g"}},
    {"id": "c", "kind": "param", "target": ..., "mode": {"kind": "scan", "generator": G3, "zip_group": "g"}},
]
```

Meaning:

- `b` and `c` are zipped
- `a` is orthogonal to that zipped pair
- overall result is an ND scan over:
  - `a`
  - `(b, c)` together

### Zip-Group Restrictions

First version:

- zipped entries must lower to a finite ordered point list
- all entries in a zip group must have the same number of points

In practice, that means zip groups should initially support only generators that are
obviously finite and level-free, such as:

- `linear`
- finite `centre_span`
- `list`
- bool/enum-as-list

The following should be rejected in zip groups for the first version:

- `refining`
- `centre_span_refining`
- `expanding`

Those generators can still be used outside zip groups where their standalone semantics
are well-defined.

## Strategy

The first host submission schema only needs two top-level strategies:

```python
{"kind": "grid"}
```

and

```python
{
    "kind": "bayesopt",
    "dimensions": ["detuning", "drive"],
    "backend": {...},
    "objective": {...}
}
```

### Grid

`{"kind": "grid"}` means:

- gather all entries with `mode.kind == "scan"`
- partition by `zip_group`
- zip within groups
- take the Cartesian product across groups

This is the default host-submission mode.

### Bayesian Optimisation

```python
{
    "kind": "bayesopt",
    "dimensions": ["detuning", "drive"],
    "backend": {
        "kind": "nubo",
        "batch_size": 4,
        "max_batches": 20,
        "initial_design_size": 8,
        "acquisition": "ucb"
    },
    "objective": {
        "kind": "channel",
        "target": {"path": "probability"}
    }
}
```

Semantics:

- `dimensions` selects which scanned entry ids form the GP input space
- rebound parameters may depend on those dimensions, but are not dimensions
- the backend lowers to the existing ask/tell optimiser path

### Bayesian Optimisation Restrictions

First version:

- every BO dimension must come from an entry with:
  - `mode.kind == "scan"`
  - numeric type
  - explicit finite lower/upper bounds
- no zipped groups are used in BO mode
- no non-numeric lists/bool/enum dimensions
- no refining/expanding generators

Practical first-version rule:

- only `linear` scan entries should be accepted as BO dimensions

This aligns with the current UI idea that BO dimensions come from a min/max style
selection.

## Objective Selection

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

Given a validated `host_scan`, the compiler should:

1. Resolve each `kind="param"` target against the fragment tree.
2. Convert fixed real params into `overrides`.
3. Convert fixed/scanned pseudoparam entries into `ScanVariable`s and constants.
4. Convert scan entries into logical scan variables.
5. Convert rebind entries into `ParameterMapping`s.
6. Lower:
   - `strategy.kind == "grid"` to zipped groups + Cartesian product
   - `strategy.kind == "bayesopt"` to the optimiser point policy
7. Produce:
   - `ScanRequest`
   - `overrides`

### Direct vs logical axes

When lowering scanned `kind="param"` entries:

- if the target resolves to exactly one concrete handle:
  - the compiler may use the real `ParamHandle` directly as a scan axis
- if the target selector resolves to multiple handles:
  - the compiler should create one logical `ScanVariable`
  - and an identity `ParameterMapping` onto all matching handles

This keeps wildcard path selection compatible with the host runtime's core identity
model.

## Example: Grid with Pseudoparam and Rebind

```python
{
    "version": 1,
    "entries": [
        {
            "id": "t",
            "kind": "param",
            "target": {"fqn": "pkg.Exp.t", "path": "*"},
            "mode": {
                "kind": "scan",
                "generator": {
                    "type": "linear",
                    "range": {"start": 0.0, "stop": 10.0, "num_points": 11,
                              "randomise_order": False}
                }
            }
        },
        {
            "id": "drive",
            "kind": "pseudoparam",
            "description": "Logical drive",
            "default": 0.0,
            "mode": {
                "kind": "scan",
                "generator": {
                    "type": "linear",
                    "range": {"start": -1.0, "stop": 1.0, "num_points": 21,
                              "randomise_order": False}
                },
                "zip_group": "pair"
            }
        },
        {
            "id": "phase",
            "kind": "param",
            "target": {"fqn": "pkg.Exp.phase", "path": "*"},
            "mode": {
                "kind": "scan",
                "generator": {
                    "type": "linear",
                    "range": {"start": 0.0, "stop": 180.0, "num_points": 21,
                              "randomise_order": False}
                },
                "zip_group": "pair"
            }
        },
        {
            "id": "amp",
            "kind": "param",
            "target": {"fqn": "pkg.Exp.amp", "path": "*"},
            "mode": {
                "kind": "rebind",
                "expr": "0.5 + 0.2 * drive"
            }
        }
    ],
    "strategy": {"kind": "grid"},
    "execution": {"max_points_per_batch": 32}
}
```

Meaning:

- `drive` and `phase` are zipped together
- `t` is orthogonal to that zipped pair
- `amp` is derived from `drive`

## Example: Bayesian Optimisation with Rebind

```python
{
    "version": 1,
    "entries": [
        {
            "id": "x",
            "kind": "pseudoparam",
            "default": 0.0,
            "mode": {
                "kind": "scan",
                "generator": {
                    "type": "linear",
                    "range": {"start": -3.0, "stop": 3.0, "num_points": 21,
                              "randomise_order": False}
                }
            }
        },
        {
            "id": "y",
            "kind": "pseudoparam",
            "default": 0.0,
            "mode": {
                "kind": "scan",
                "generator": {
                    "type": "linear",
                    "range": {"start": -2.0, "stop": 2.0, "num_points": 21,
                              "randomise_order": False}
                }
            }
        },
        {
            "id": "physical_x",
            "kind": "param",
            "target": {"fqn": "pkg.Exp.x", "path": "*"},
            "mode": {"kind": "rebind", "expr": "x"}
        },
        {
            "id": "physical_y",
            "kind": "param",
            "target": {"fqn": "pkg.Exp.y", "path": "*"},
            "mode": {"kind": "rebind", "expr": "2.0 * y"}
        }
    ],
    "strategy": {
        "kind": "bayesopt",
        "dimensions": ["x", "y"],
        "backend": {
            "kind": "nubo",
            "batch_size": 4,
            "max_batches": 20,
            "initial_design_size": 8,
            "acquisition": "ucb"
        },
        "objective": {
            "kind": "channel",
            "target": {"path": "cost"}
        }
    },
    "execution": {"max_points_per_batch": 4}
}
```

Meaning:

- GP lives in logical `(x, y)` space
- real experiment parameters are driven by rebind mappings
- no explicit point table is required in the schema

## Immediate Next Implementation Targets

The first code steps implied by this schema are:

1. host submission dataclasses / validation
2. compiler from `host_scan` to `ScanRequest + overrides`
3. AST-based rebind expression compiler
4. grid mode with zip-group lowering
5. BO mode with bounded linear dimensions only

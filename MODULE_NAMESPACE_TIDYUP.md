# Module Namespace Tidy-Up Proposal

This note proposes a cleaner module tree for `ndscan`.

The goal is not to rename everything immediately. The goal is to make the major
concepts explicit so future work, especially around the host runtime and dashboard
submission, has a clearer home.

## Why

The current `ndscan.experiment` package has accumulated several responsibilities:

- experiment-definition API
- legacy execution/runtime code
- host-runtime execution code
- host-scan schema compilation
- logical scan mappings and expression compilation
- scan-site persistence details

That makes the package hard to navigate and makes it unclear where new code should
live.

The most important design choice is:

- split by responsibility
- not by version labels like `v1` / `v2`

Names like `legacy`, `host`, `submission`, and `define` age much better than
`v1` / `v2`.

## Principles

### 1. Keep a friendly facade

User experiment code should still be able to write:

```python
from ndscan.experiment import *
```

So `ndscan.experiment` should become a facade/re-export layer, not the place where
all implementation detail lives forever.

### 2. Separate ontology from execution

These are different concerns:

- what a fragment/parameter/result channel is
- how a scan request is submitted/compiled
- how a runtime executes that request

They should not be mixed in one large package namespace.

### 3. Keep the dashboard out of the runtime

Qt/dashboard code should emit declarative submission objects.

Compilation from dashboard state into runtime objects should live in a
submission-oriented layer, not inside the runtime loop itself.

### 4. Keep offline results reading independent

Reading HDF5 snapshots should stay separate from experiment/runtime code as much as
possible.

## Proposed Target Tree

If the tree were designed from scratch, a sensible layout would be:

```text
ndscan/
    __init__.py

    define/
        __init__.py
        fragment.py
        parameters.py
        result_channels.py
        annotations.py
        analysis.py

    submission/
        __init__.py
        host_scan_schema.py
        legacy_scan_schema.py
        compile_host_scan.py
        compile_legacy_scan.py
        expression.py

    runtime/
        __init__.py

        host/
            __init__.py
            request.py
            point_policy.py
            scan_mapping.py
            session.py
            analysis.py
            scan_site.py
            optimisation.py

        legacy/
            __init__.py
            entry_point.py
            scan_runner.py
            scan_generator.py
            subscan.py

    results/
        __init__.py
        scan_site_reader.py
        arguments.py
        pyplot.py
        tools.py

    dashboard/
        __init__.py
        argument_editor.py
        scan_options.py
        param_tree_dialog.py
        utils.py

    plots/
        __init__.py
        model/
        widgets/
        legacy/
        host/

    compat/
        __init__.py
        experiment.py
```

## Meaning of the Top-Level Areas

### `define/`

Stable experiment-building ontology:

- fragments
- parameters
- result channels
- annotations
- default analysis declarations

This is the layer experiment authors conceptually program against.

### `submission/`

Everything that turns a declarative request into runtime objects:

- schema validation
- schema-to-request compilation
- tiny expression language for text mappings
- future dashboard-facing request dataclasses

This is where the new `host_scan_schema.py` really belongs conceptually.

### `runtime/host/`

Execution of the new host-driven scan model:

- `ScanRequest`
- `ExecutionPolicy`
- `PointPolicy`
- point execution/session orchestration
- request-level scan mappings
- host BO backends
- scan-site persistence for this runtime

### `runtime/legacy/`

The older runtime path, clearly named as such:

- `entry_point.py`
- `scan_runner.py`
- `scan_generator.py`
- `subscan.py`

This avoids pretending the two execution models are one coherent implementation.

### `results/`

Offline reading and tooling:

- HDF5 snapshot readers
- plotting helpers
- argument/result tools

### `dashboard/`

Qt submission UI only.

The dashboard should depend on `submission/` and selected definition schema, not on
runtime internals directly.

## Proposed Incremental Mapping From Today

This is the pragmatic version, starting from the current tree.

### Keep `ndscan.experiment` as a facade

Do not break public imports early.

Instead:

- move implementation modules underneath clearer internal namespaces
- re-export the stable public symbols from `ndscan.experiment`

### First extraction

Move schema compilation out of `host_runtime.py`.

Status:

- this has already started via `ndscan/experiment/host_scan_schema.py`

This is the right first move because schema compilation is conceptually separate from
scan execution and will keep growing as dashboard work lands.

### Second extraction

Create a more obvious submission cluster around:

- `expression.py`
- `scan_mapping.py`
- `host_scan_schema.py`

That could be either:

```text
ndscan/experiment/submission/
```

or eventually:

```text
ndscan/submission/
```

### Third extraction

Group legacy runtime code explicitly:

```text
ndscan/experiment/legacy/
```

or later:

```text
ndscan/runtime/legacy/
```

This gives the codebase a clear place to put bug fixes without implying new runtime
features should also go there.

### Fourth extraction

Group host-runtime execution files:

- `host_runtime.py`
- `point_policy.py`
- host-side analysis helpers
- BO backends
- scan-site persistence

That cluster could become:

```text
ndscan/experiment/host/
```

before any larger top-level package split.

## Recommended Intermediate Tree

If we want a realistic near-term target without moving every public import, this is a
good intermediate structure:

```text
ndscan/experiment/
    __init__.py            # facade/re-exports

    define/
        fragment.py
        parameters.py
        result_channels.py
        annotations.py
        default_analysis.py

    submission/
        expression.py
        host_scan_schema.py
        scan_mapping.py

    host/
        runtime.py
        point_policy.py
        optimisation.py
        scan_site.py
        _host_analysis.py

    legacy/
        entry_point.py
        scan_generator.py
        scan_runner.py
        subscan.py
```

This is likely the best balance between:

- conceptual cleanliness
- migration effort
- keeping imports stable

## File-Level Mapping Suggestion

Current -> proposed home:

- `experiment/fragment.py` -> `experiment/define/fragment.py`
- `experiment/parameters.py` -> `experiment/define/parameters.py`
- `experiment/result_channels.py` -> `experiment/define/result_channels.py`
- `experiment/annotations.py` -> `experiment/define/annotations.py`
- `experiment/default_analysis.py` -> `experiment/define/default_analysis.py`

- `experiment/expression.py` -> `experiment/submission/expression.py`
- `experiment/scan_mapping.py` -> `experiment/submission/scan_mapping.py`
- `experiment/host_scan_schema.py` -> `experiment/submission/host_scan_schema.py`

- `experiment/host_runtime.py` -> `experiment/host/runtime.py`
- `experiment/point_policy.py` -> `experiment/host/point_policy.py`
- `experiment/optimisation.py` -> `experiment/host/optimisation.py`
- `experiment/scan_site.py` -> `experiment/host/scan_site.py`
- `experiment/_host_analysis.py` -> `experiment/host/analysis.py`

- `experiment/entry_point.py` -> `experiment/legacy/entry_point.py`
- `experiment/scan_generator.py` -> `experiment/legacy/scan_generator.py`
- `experiment/scan_runner.py` -> `experiment/legacy/scan_runner.py`
- `experiment/subscan.py` -> `experiment/legacy/subscan.py`

## Explicit Non-Goals

This proposal does not require:

- removing `ndscan.experiment`
- rewriting public APIs immediately
- unifying legacy and host runtime concepts artificially
- renaming everything in one big breaking commit

## Recommended Next Moves

In order:

1. Keep `host_scan_schema.py` as the submission/compiler home and avoid adding new
   schema logic back into `host_runtime.py`.
2. Move `expression.py` and `scan_mapping.py` into the same submission cluster.
3. Move legacy runtime files under a `legacy/` namespace.
4. Move host runtime files under a `host/` namespace.
5. Keep `ndscan.experiment` as a facade throughout.

## Short Version

If only one sentence is remembered, it should be:

> Split by responsibility (`define`, `submission`, `host`, `legacy`, `results`,
> `dashboard`), and keep `ndscan.experiment` as a compatibility facade.

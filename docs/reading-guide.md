# Reading Guide

This guide is for people who want to understand or modify ndscan without first holding
the whole architecture in their head. It points to the active prepared-runtime path.
Treat `ndscan.legacy` and `docs/archive/` as background unless you are deliberately
working on compatibility.

## First Pass

Read these in order:

1. [concepts.md](concepts.md)
2. [module-map.md](module-map.md)
3. [prepared-runtime.md](prepared-runtime.md)
4. [scan-site-schema.md](scan-site-schema.md)
5. [child-scans-and-kernels.md](child-scans-and-kernels.md)

That gives the vocabulary, the package boundaries, the execution model, the persisted
data contract, and the nested-scan/kernel rules.

## Code Reading Order

### 1. Fragment-Facing API

Start with `ndscan.define`:

- `ndscan/define/fragment.py`
- `ndscan/define/parameters.py`
- `ndscan/define/result_channels.py`
- `ndscan/define/default_analysis.py`
- `ndscan/define/annotations.py`

This is the layer experiment authors see. The important ideas are:

- fragments compose experiment structure,
- parameter handles read/write parameter stores,
- result channels describe output streams,
- sinks decide where pushed result values go at runtime.

### 2. Scan Semantics

Then read `ndscan.scan`:

- `ndscan/scan/request.py`
- `ndscan/scan/point_policy.py`
- `ndscan/scan/mapping.py`
- `ndscan/scan/optimisation.py`

This layer describes what should be scanned. It should not know how ARTIQ executes a
point or how datasets are written.

Keep these distinctions in mind:

- `ScanRequest` is declarative.
- `PointPolicy` chooses points and receives feedback.
- `ParameterMapping` turns logical coordinates into concrete fragment parameters.
- `ExecutionPolicy` controls batching/preview behaviour, not the scan trajectory.

### 3. Prepared Runtime

Then follow the execution path through `ndscan.runtime`:

- `ndscan/runtime/api.py`: public imports and convenience surface.
- `ndscan/runtime/adapters.py`: dashboard/code entry points.
- `ndscan/runtime/prepared.py`: reusable `PreparedScan` and `PreparedChildScan`
  handles.
- `ndscan/runtime/program.py`: binds a request to one fragment tree.
- `ndscan/runtime/executors.py`: host and resident-kernel execution loops.
- `ndscan/runtime/context.py`: current point/run context for child scans and previews.
- `ndscan/runtime/persistence.py`: writes the scan-site dataset schema.
- `ndscan/runtime/analysis.py`: default/online analysis execution.

The runtime should be read as a pipeline:

```text
ScanRequest
  -> ScanProgramBuilder
  -> ScanProgram
  -> ScanProgramRunner
  -> HostExecutor or KernelStreamingExecutor
  -> PointObservation batches
  -> ScanSiteDatasetWriter
```

`program.py` owns binding and metadata. `executors.py` owns point execution.
`persistence.py` owns dataset writes. Keeping those responsibilities separate is the
main way to stay oriented.

### 4. Child Scans

For subscans, read these together:

- [child-scans-and-kernels.md](child-scans-and-kernels.md)
- `ndscan/runtime/prepared.py`
- `ndscan/runtime/context.py`
- `ndscan/runtime/executors.py`
- `ndscan/schema/scan_site.py`

The key idea is that a child scan is not a special runtime. It is another scan site
using the same request/program/executor/writer path, with extra context that links its
segments back to parent point indices.

### 5. Saved Results and Plotting

For data analysis and plotting, read:

- [scan-site-schema.md](scan-site-schema.md)
- [results-and-plotting.md](results-and-plotting.md)
- `ndscan/results/scan_site_reader.py`
- `ndscan/results/series.py`
- `ndscan/results/pyplot.py`
- `ndscan/plots/runtime/live.py`
- `ndscan/plots/runtime/viewer.py`

The persisted HDF5/live dataset tree is the contract. The reader and plotting layers
should derive convenient views from that contract rather than requiring the runtime to
write duplicate convenience datasets.

### 6. Dashboard Submission

For dashboard or transport payloads, read:

- [scan-submission-schema.md](scan-submission-schema.md)
- `ndscan/submission/scan_submission_schema.py`
- `ndscan/submission/expression.py`

This path turns a declarative submission payload into `ScanRequest` plus parameter
overrides. It should stay separate from the runtime execution loop.

## Read By Task

If you want to add a new parameter type:

- read `ndscan/define/parameters.py`,
- check how stores expose `RpcType`, `to_rpc_type()`, and `set_from_rpc()`,
- check `ndscan/runtime/program.py` and `ndscan/runtime/executors.py` for assumptions
  about parameter RPC values.

If you want to add a new point policy:

- read `ndscan/scan/point_policy.py`,
- implement `axis_count`, `next_batch()`, `is_finished()`, and `describe()`,
- use `observe_batch()` only if the policy needs feedback.

If you want to change what is saved:

- read [scan-site-schema.md](scan-site-schema.md),
- update `ndscan/schema/scan_site.py` if the physical layout changes,
- update `ndscan/runtime/persistence.py` for writes,
- update `ndscan/results/scan_site_reader.py` and `ndscan/plots/runtime/live.py` for
  reads,
- add tests before treating the schema as settled.

If you want to change child-scan behaviour:

- read [child-scans-and-kernels.md](child-scans-and-kernels.md),
- start in `ndscan/runtime/prepared.py`,
- then inspect `ndscan/runtime/context.py` and `ndscan/runtime/executors.py`,
- verify both host-parent and resident-kernel-parent cases.

If you want to change plotting:

- start with `ndscan/results/scan_site_reader.py`,
- then `ndscan/plots/runtime/live.py`,
- then the viewer modules.

The plotting code should consume `ScanSiteSnapshot`/`ScanSiteData` where possible
instead of parsing raw dataset keys independently.

## Patterns To Recognise

### Handle vs Store

`ParamHandle` is what fragment code uses. `ParamStore` is where the value lives. Several
handles can point to one store, and the prepared runtime can temporarily swap stores for
scanned parameters.

### Channel vs Sink

`ResultChannel` describes an output series. Its current sink decides where pushed values
go. The runtime swaps sinks during point execution so it can collect one value per saved
channel.

### Request vs Program

`ScanRequest` is user/runtime-independent intent. `ScanProgram` is a request bound to a
specific fragment instance, result channels, parameter mappings, analysis, and writer.

### Point Policy vs Executor

The point policy chooses where to go next. The executor only runs the points it is given.
This separation is what keeps adaptive scan logic outside the ARTIQ execution path.

### Host, Kernel, Portable, RPC

ARTIQ decorators matter:

- `@host_only` runs only on the host.
- `@kernel` compiles for the core device.
- `@portable` can be used from both sides if ARTIQ can compile it.
- `@rpc` is a host call made from kernel code.

The resident-kernel path is designed so kernel code calls RPCs for data/results and then
returns to the same kernel. RPCs should not call back into another kernel.

### Context Variables

`runtime.context` tracks the current scan point and root run. This is how child scans
know which parent point they belong to without passing parent indices through every user
function.

## What Not To Start With

Avoid starting with:

- `ndscan.legacy`,
- `docs/archive/`,
- plotting viewer internals,
- optimiser internals,
- generated kernel strings in `executors.py`.

Those pieces make more sense after the main request/program/executor/writer path is
clear.

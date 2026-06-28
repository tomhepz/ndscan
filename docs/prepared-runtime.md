# Prepared Runtime

The prepared runtime is the canonical execution path for new ndscan work.

Its job is to run a fragment over batches of resolved points, record the complete
scan-site dataset, run analysis at defined boundaries, and optionally publish live
preview snapshots. It should not contain scan-navigation policy.

## Execution Flow

The common path is:

1. Define a fragment using `ndscan.define`.
2. Create a `ScanRequest` directly, or compile a `ScanSubmissionSpec`.
3. Prepare a `PreparedScan` or `PreparedChildScan`.
4. Build a `ScanProgram` from the fragment and request.
5. Ask the point policy for batches.
6. Resolve logical point values through parameter mappings.
7. Execute the batch with either `HostExecutor` or `KernelStreamingExecutor`.
8. Append point observations to the scan-site writer.
9. Run online analysis and feed batch feedback back to the point policy.
10. Flush/publish preview state at the configured boundaries.
11. Run final analysis and mark the scan site complete.

## Core Runtime Types

`PreparedScan`
: Root executable handle. It owns setup, execution, cleanup, retry behavior, and output
selection for a top-level scan.

`PreparedChildScan`
: Child-scan handle used inside fragments. It follows the same prepared-runtime path as
root scans, but writes a child scan site and can be acquired from host or kernel code.

`ScanProgram`
: Bound execution plan in `runtime.program`. It contains the fragment, bound axes,
varying concrete parameters, channels, mappings, point source, analysis adapter, and
observation transport.

`ScanProgramBuilder`
: Validates and binds a fragment/request pair in `runtime.runner`, using
`runtime.binding` for axes, parameter mappings, and concrete point values.

`ScanProgramRunner`
: Owns the run loop in `runtime.runner`: retry boundaries, executor selection, preview
coordination, and batch finalisation.

`ScanOutputs`
: Stable named output surface returned by prepared scans.

`ScanInspection`
: Rich host-only inspection artifact for tests, diagnostics, and exploratory code.

## Executor Choice

The runtime has two execution backends:

- `HostExecutor` invokes already-resolved point bodies from the host.
- `KernelStreamingExecutor` enters one resident kernel and uses RPCs to exchange
  batches/results with the host.

The important design point is that both executors use the same `ScanProgram`, point
policy, scan-site writer, and analysis path. There is no separate host scan semantic
model and kernel scan semantic model.

## Batch Boundary

The prepared runtime treats a completed batch as the main boundary for host-side work.
At that boundary it:

- records observations in memory,
- appends datasets,
- runs online analysis,
- feeds the point policy,
- flushes persistence,
- writes preview snapshots when due.

This keeps per-point kernel/host traffic low while still allowing adaptive scans to
observe results and choose future points.

## Entry Points

Code-first scans use `make_fragment_prepared_scan_exp()` or construct `PreparedScan`
directly.

Dashboard-driven scans use `make_fragment_prepared_dashboard_scan_exp()`. The dashboard
payload key is `scan_submission`, which is compiled into a `ScanRequest` before the
runtime starts.

Vanilla ARTIQ `EnvExperiment` code can also instantiate fragments and prepared scans
directly, so dashboard, code-first, and embedded use all converge on the same prepared
runtime.

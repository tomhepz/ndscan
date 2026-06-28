# Concepts

This is the current vocabulary for the prepared-runtime ndscan path.

## Fragment

A fragment is the composable experiment unit. It owns parameters, result channels,
subfragments, lifecycle hooks, and the point body to execute. Fragment-facing APIs live
mostly in `ndscan.define`.

## Parameter

A parameter is a concrete fragment input. It has a schema, a store, a current value, and
usually a stable fully qualified name plus a fragment path. Prepared scans may vary
parameters directly or set them indirectly through mappings.

## Pseudoparam

A pseudoparam is a logical scan input that is not itself a concrete fragment parameter.
It is useful when the user wants to scan in an experimental coordinate and map that
coordinate onto one or more real parameters. The persisted schema uses
`pseudoparam_*` point streams for these values.

## Result Channel

A result channel is a named output stream produced by a fragment. Saved channels become
`channel_*` point streams in the scan-site dataset.

## Scan Request

`ScanRequest` is the runtime-independent description of a scan. It contains:

- the point policy,
- fixed pseudoparams,
- parameter mappings,
- execution policy,
- preview policy,
- scan-site placement,
- user metadata.

Code-first prepared scans should generally construct `ScanRequest` objects directly.

## Scan Submission

`ScanSubmissionSpec` is the dashboard/transport-facing declarative schema. It is
compiled by `compile_scan_submission_spec()` or `compile_scan_submission_schema()` into
`ScanRequest` plus fixed parameter overrides.

The transport key used in ARTIQ argument payloads is `scan_submission`.

## Point Policy

A point policy decides which logical points to run next and receives batch-level
feedback. Static grids, zipped scans, explicit points, repeats, gradient descent, and
ask/tell optimisers all fit behind this interface. The runtime asks for batches and
reports observations; it does not contain scan-navigation logic.

## Prepared Scan

`PreparedScan` is the executable handle created after a fragment and `ScanRequest` are
known. It prepares a stable execution shape and then runs through the common prepared
runtime.

## Scan Program

`ScanProgram` is the fragment-bound execution plan. It binds logical scan axes,
concrete varying parameters, result channels, parameter mappings, persistence, and
analysis. `ScanProgramBuilder` builds it; `ScanProgramRunner` runs it.

## Executor

Executors run resolved batches for a `ScanProgram`.

- `HostExecutor` runs point bodies from the host. The name is literal.
- `KernelStreamingExecutor` enters one resident kernel region and receives batches from
  host RPCs without launching nested kernels.

Both executors share the same `ScanProgram` and persistence path.

## Scan Site

A scan site is one persisted node in a scan tree. The root scan is site `()`. Child scans
have child paths and record segments that link child points back to parent points.

The scan-site schema is described in [scan-site-schema.md](scan-site-schema.md).

## Snapshot

`ScanSiteSnapshot` is the offline/live result model. It contains `ScanSiteData` objects
for the root site and all child sites in one result file or live dataset view.

# Module Map

The prepared-runtime path is split by responsibility.

## `ndscan.define`

Fragment-facing experiment APIs:

- fragments and subfragments,
- parameter types and handles,
- result channels,
- annotations,
- default analysis declarations.

This is the layer experiment authors use to define parameterised, composable ARTIQ
experiments.

## `ndscan.scan`

Runtime-independent scan semantics:

- `ScanRequest`,
- point policies,
- parameter mappings,
- optimisation helpers.

This layer should not know about dashboard widgets or persistence details.

## `ndscan.submission`

Submission-front-end compilation:

- safe expression compilation for text rebinds,
- `ScanSubmissionSpec`,
- `compile_scan_submission_schema()`.

This layer turns dashboard/transport payloads into `ScanRequest` plus fixed parameter
overrides.

## `ndscan.runtime`

Prepared execution:

- adapters for code/dashboard `EnvExperiment` entry points,
- `PreparedScan` and `PreparedChildScan`,
- `ScanProgram`, `ScanProgramBuilder`, `ScanProgramRunner`,
- `HostExecutor` and `KernelStreamingExecutor`,
- persistence and preview snapshots,
- default-analysis execution.

This layer runs batches and records observations. It does not decide how scan points are
chosen beyond asking the point policy for the next batch.

## `ndscan.schema`

Small schema helpers shared by runtime/results/plotting. The current scan-site schema
revision is defined in `ndscan.schema.scan_site`.

## `ndscan.results`

Offline result reading and analysis convenience:

- `read_scan_site_snapshot()`,
- `ScanSiteSnapshot`,
- `ScanSiteData`,
- series helpers.

This layer reads persisted HDF5 files without requiring the experiment runtime.

## `ndscan.plots.runtime`

Live/offline plotting convenience over the scan-site model. It is coupled to the
scan-site schema, but should stay outside the execution runtime.

## `ndscan.dashboard`

Dashboard argument editing and submission payload construction. The prepared path uses
the `scan_submission` payload key and `ScanSubmissionBackend`.

## `ndscan.legacy`

The older ndscan scan/subscan runtime. It remains present for compatibility and for
some helper logic that has not yet been moved. New prepared-runtime work should avoid
adding new dependencies on this package unless it is a deliberate compatibility bridge.

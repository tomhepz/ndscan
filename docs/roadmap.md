# Roadmap

This file lists current consolidation work for the prepared-runtime path. Historical
implementation plans live in [archive/](archive/).

## Documentation

- Keep [concepts.md](concepts.md), [module-map.md](module-map.md), and
  [prepared-runtime.md](prepared-runtime.md) as the authoritative architecture docs.
- Keep [scan-site-schema.md](scan-site-schema.md) aligned with
  `SCAN_SITE_SCHEMA_REVISION`.
- Keep [scan-submission-schema.md](scan-submission-schema.md) aligned with
  `ScanSubmissionSpec`.

## Schema And Results

- Remove duplication between `ndscan.results.scan_site_reader` and
  `ndscan.plots.runtime.live`.
- Add focused tests around any scan-site schema revision change.
- Decide whether metadata blobs/artifacts need a small shared helper API or should stay
  plain dictionaries.

## Runtime

- Keep `ScanProgram` as the single bound execution plan for host and kernel executors.
- Avoid adding scan-navigation logic to the runtime; put it behind point policies.
- Keep `runtime.runner` as the scan-site lifecycle owner and `runtime.executors` as
  point-invocation backends.

## Submission

- Split `scan_submission_schema.py` into smaller modules if it continues to grow:
  typed specs, validation/transport, grid compilation, GPO compilation, generator bridge.
- Move expression compilation out of `ndscan.submission` if lower-level scan modules need
  it directly.
- Compile dashboard finite generators to native point policies instead of relying on the
  legacy generator bridge.

## Plotting

- Split `ndscan.plots.runtime.viewer` into pure choice/data helpers, array selection,
  annotation/artifact rendering, Qt controls, and top-level viewer composition.
- Keep plotting isolated from runtime execution.

## Legacy Boundary

- Avoid new prepared-runtime dependencies on `ndscan.legacy`.
- Move shared helpers out of legacy modules when they are needed by the prepared path.

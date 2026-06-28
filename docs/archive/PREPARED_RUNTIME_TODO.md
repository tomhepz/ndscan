# Prepared Runtime TODO

This document captures follow-up work for the prepared-runtime scan model after the first
dashboard/submission implementation pass.

## Point policies

- Add native 1D point policies for the common scan shapes currently inherited from the
  legacy generator model:
  - `LinearPointPolicy1D`
  - `CentreSpanPointPolicy1D`
  - `ExpandingPointPolicy1D`
  - `RefiningPointPolicy1D`
- Prefer these native point policies as the prepared-runtime core abstraction rather than
  continuing to materialise old `ScanGenerator` output into value lists.
- Keep legacy `ScanGenerator` classes as a compatibility layer for the old runtime/UI,
  not as the long-term core for the prepared runtime.

## Point-policy capabilities

- Add explicit point-policy capability/introspection for finiteness and
  materialisability.
- Do not rely on a single boolean alone; distinguish at least:
  - static finite
  - dynamic finite
  - unbounded
- Add a way to query the total number of points when that is actually knowable.
- Add a way to query whether the full point set can be materialised up front.

## Randomisation

- Implement global point-order randomisation as a wrapper over finite materialisable
  point policies rather than baking it into every policy.
- Candidate shape:
  - `RandomisedFinitePointPolicy(inner, seed=...)`
- Reject global randomisation cleanly for adaptive or unbounded policies that cannot be
  materialised.

## Host schema compiler

- Compile scan-submission schema directly to native point policies instead of routing
  through legacy generator classes.
- Remove the new-path dependence on `_materialise_scan_mode_values()` using
  `scan_generator.py` once native host policies exist for the supported scan modes.
- Keep the dashboard and worker-side schema model unchanged where possible; this should
  be a compiler/runtime cleanup, not a submission-schema redesign.

## Code-first ergonomics

- Continue improving the code-first `ScanRequest` helpers for common finite scans.
- `ScanRequest.linear(...)` now exists; consider whether matching helpers are warranted
  for:
  - centre/span scans
  - expanding scans
  - finite shuffled scans
- Keep code-first APIs handle-based (`ParamHandle`, `ScanVariable`,
  `ParameterMapping`) rather than introducing selector-string semantics there.

## Design notes

- `ParamHandle` should remain a concrete singular handle, not a wildcard selector.
- Wildcard targeting belongs in submission/compiler layers, not in the runtime handle
  abstraction.
- Runtime-side parameter mappings are still the right place for transformed rebinding:
  - they are visible to the execution engine,
  - can be validated centrally,
  - and run before `device_setup()` / `run_once()`.

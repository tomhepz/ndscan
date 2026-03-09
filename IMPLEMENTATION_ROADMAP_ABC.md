# NDScan Implementation Roadmap (A/B/C)

This document is a practical implementation guide for splitting the current prototype work into reviewable PRs, starting from `master`.

It combines:
- the high-level 3-PR plan (`A`, `B`, `C`)
- a detailed implementation guide for `A`
- updated naming: use `ScanSiteDatasetWriter` (not `SubscanDatasetWriter`)

## Scope and philosophy

- Preserve existing behavior where possible.
- Keep kernel functionality working as-is for existing modes.
- Allow new features to be host-first, with clear early errors for unsupported kernel combinations.
- Keep commits small, testable, and reviewable.

---

## PR split overview

## A) Point engine + flat/live dataset schema per scan site

Goal:
- Decouple point choice from runner execution.
- Support better point streams (`grid`, `zip`, `point_list`, and adaptive hooks).
- Use consistent flat schema for all scan sites.
- Stream live subscan data while keeping archive-safe flat storage.

Primary files:
- `ndscan/experiment/scan_strategy_specs.py` (new)
- `ndscan/experiment/scan_point_strategies.py` (new)
- `ndscan/experiment/point_source.py` (new)
- `ndscan/experiment/scan_runner.py`
- `ndscan/experiment/subscan.py`
- `ndscan/experiment/entry_point.py`
- `ndscan/experiment/scan_site_dataset_writer.py` (new)

## B) UI changes to consume and visualize these point streams

Goal:
- Plot flat subscan data and live preview reliably.
- Keep selected-point view stable and separate from live-running view.
- Add scan strategy submission in UI (at least `grid`, `zip`, `point_list`).

Primary files:
- `ndscan/plots/model/subscan.py`
- `ndscan/plots/model/subscriber.py`
- `ndscan/plots/xy_1d.py`
- `ndscan/plots/plot_widgets.py`
- `ndscan/plots/container_widgets.py`
- dashboard argument editing modules

## C) Parameter transform API (`setattr_param_transform()` family)

Goal:
- Allow parameter values to be computed from other parameters at runtime.
- Support code-defined transforms and optional UI formula transforms.
- Keep transform activation/conflict semantics explicit.

Primary files:
- `ndscan/experiment/fragment.py`
- `ndscan/experiment/relation_specs.py`
- `ndscan/experiment/relation_expressions.py`
- dashboard serialization/parsing for relations

---

## Detailed plan for A

## Current state on `master`

1. Point generation is tightly coupled to `generate_points(...)` and axis generators.
2. `ScanRunner` consumes iterators (`set_points`) rather than a first-class point source contract.
3. Top-level has separate scan vs no-axes/continuous flow.
4. Subscan writes array payload channels, not a canonical flat per-site stream.
5. Metadata/schema writing is duplicated across top-level and subscan paths.

## Conceptual flaws in current state

1. Point stream policy is not pluggable.
- Harder to support tandem (`zip`), explicit shot lists, and adaptive drivers cleanly.

2. Ragged subscan data is fragile for archival.
- HDF5 archival expects rectangular arrays.
- List-of-lists behavior causes persistence risks.

3. Site schema is inconsistent.
- Top-level and subscan data layout are not uniformly represented.
- Consumers (plot tools/offline tools) must special-case paths.

4. Lifecycle and writing responsibilities are mixed.
- Subscan runtime owns orchestration and dataset serialization details in one class.

## Implementation flaws in code structure

1. Strategy parsing and row validation are duplicated.
2. Retry/observe/point completion logic is partly duplicated between host/kernel paths.
3. Metadata conversion (`json` vs native scalar) is repeated.
4. Dataset key conventions are spread across modules.

## What A is aiming for

1. A runner accepts point sources rather than being tied to one generator pipeline.
2. Strategy logic is modular and testable.
3. Every scan site can emit a consistent flat schema:
- metadata
- `points.axis_*`
- `points.channel_*`
- optional `points.param_*`
- optional `points.acquired_at`
- segmentation via `starts` (+ optional `start_timestamps`)
4. Subscan supports:
- preview stream (`subscan_preview.*`) for live UI
- flat stream (`subscan_flat.*`) for canonical/archive-safe storage
5. Existing modes stay compatible.

---

## A architecture sketch

```python
# point_source.py
@dataclass(frozen=True)
class PointObservation:
    point_index: int
    axis_values: tuple[Any, ...]
    result_values: dict[str, Any]
    axis_by_param: dict[tuple[str, str], Any] | None = None
    acquired_at: float | None = None


class PointSource:
    def next_point(self): ...
    def take_points(self, max_points: int) -> list[tuple[Any, ...]]: ...
    def observe(self, observation: PointObservation) -> None: ...
    def preferred_batch_size(self, default: int) -> int: return default
```

```python
# scan_runner.py
def run(
    self,
    fragment: ExpFragment,
    spec: ScanSpec,
    axis_sinks: list[ResultSink],
    param_sinks: list[tuple[ParamHandle, ResultSink]] | None = None,
    acquired_at_sink: ResultSink | None = None,
    point_source: PointSource | None = None,
) -> None:
    if point_source is None:
        point_source = StrategyPointSource(spec.generators, spec.options, spec.strategy)
    self.set_point_source(point_source)
    ...
```

```python
# scan_site_dataset_writer.py
class ScanSiteDatasetWriter:
    def begin_site(self, site_meta: dict[str, Any]) -> None: ...
    def reset_preview(self) -> None: ...
    def append_point(
        self,
        axis_values: dict[str, Any],
        channel_values: dict[str, Any],
        param_values: dict[str, Any] | None = None,
        acquired_at: float | None = None,
    ) -> None: ...
    def append_segment_start(self, index: int, start_timestamp: float | None = None) -> None: ...
    def set_completed(self) -> None: ...
```

---

## A commit plan (recommended)

## A1: Strategy spec parsing extraction

Implement:
- `scan_strategy_specs.py` with:
  - `get_scan_strategy_kind(...)`
  - `parse_scan_strategy(...)`
  - `extract_point_list_rows(...)`

Checks:
- unit tests for accepted/rejected strategy shapes and row widths

## A2: Point composition strategy module

Implement:
- `scan_point_strategies.py` with:
  - `_generate_grid_points(...)`
  - `_generate_zip_points(...)`
  - `_generate_rows(...)`
  - `generate_points_for_strategy(...)`

Compatibility:
- keep `scan_generator.generate_points(...)` as wrapper during transition

Checks:
- parity test for existing grid behavior
- zip/pairing tests
- point-list tests

## A3: PointSource contract and strategy-backed source

Implement:
- `point_source.py` with `PointSource`, `IteratorPointSource`, `StrategyPointSource`, `PointObservation`

Checks:
- source chunking behavior
- observation callback forwarding

## A4: Runner accepts point sources and emits observations

Implement:
- `ScanRunner.run(..., point_source=...)`
- host/kernel runners consume `PointSource` uniformly
- push `acquired_at` and call `observe(...)` after successful point completion

Checks:
- runner-level tests for observation payload
- no regression in existing scan tests

## A5: Introduce `ScanSiteDatasetWriter`

Implement:
- new module `scan_site_dataset_writer.py`
- encapsulate dataset key writing and metadata serialization per site
- first use from subscan path (minimal invasive rollout)

Checks:
- existing subscan tests still pass
- new tests for segmentation and timestamp fields

## A6: Flat + live subscan schema consolidation

Implement:
- preview + flat stream handling via writer
- consistent `starts` segmentation and optional `start_timestamps`
- ensure no ragged archive writes

Checks:
- ragged subscan archival safety tests
- preview reset across repeated subscan runs

## A7: Top-level integration with site writer (optional in A, but recommended)

Implement:
- use `ScanSiteDatasetWriter` for top-level points and metadata too
- this reduces key drift between top-level and subscan

Checks:
- entrypoint tests for schema keys and channel streams

---

## Dataset schema target for A

For each site (top-level or subscan-flat):

```text
<site>ndscan_schema_revision
<site>source_id
<site>completed
<site>start_timestamp
<site>fragment_fqn
<site>axes
<site>channels
<site>strategy

<site>points.axis_0
<site>points.axis_1
...
<site>points.param_<name>       # optional
<site>points.channel_<name>
<site>points.acquired_at        # optional

<site>starts                    # subscan segmented sites
<site>start_timestamps          # optional
```

Live preview remains:

```text
ndscan.rid_<rid>.subscan_preview.<site>.points.*
ndscan.rid_<rid>.subscan_preview.<site>.completed
```

---

## Explicit boundaries for A

1. No plotting behavior changes in A.
2. No dashboard UI changes in A.
3. No parameter transform API changes in A.
4. Keep kernel behavior for existing paths unchanged.
5. New strategy/feature kernel limitations must fail early with clear errors.

---

## Suggested tests to run per step

```bash
python -m unittest -v test.test_experiment_scan_strategy_specs
python -m unittest -v test.test_experiment_scan_point_strategies
python -m unittest -v test.test_experiment_point_source
python -m unittest -v test.test_experiment_scan_generator
python -m unittest -v test.test_experiment_subscan
python -m unittest -v test.test_experiment_entrypoint
```

If kernel emulator tests are available in your environment:

```bash
python -m unittest -v test.test_experiment_kernel
```

---

## Reviewability guidance

1. Keep each commit focused on one abstraction.
2. Prefer new modules over inflating `subscan.py` and `entry_point.py`.
3. Add contract tests whenever a key/schema contract changes.
4. Document host-only behavior explicitly in commit message and docstring.

---

## Notes for PR descriptions

For PR A, keep messaging simple:
- “Decouple point stream policy from runner execution.”
- “Introduce consistent flat schema for scan sites.”
- “Improve archival robustness for subscan data.”
- “No UI or relation API changes in this PR.”


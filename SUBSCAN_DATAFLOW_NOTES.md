# Subscan Dataflow Notes (Handoff for Agent Work)

## Current Project Status

Baseline characterization is now in place (no behavior change yet):

- Bulk subscan aggregate push semantics are locked by test:
  - [test_experiment_subscan.py](/home/lab/artiq-files/install/ndscan/test/test_experiment_subscan.py)
  - `BulkPushCharacterisationCase.test_aggregate_channels_receive_one_push_per_subscan`
- Ragged subscan current storage shape (list-of-lists) is locked by test:
  - [test_experiment_subscan.py](/home/lab/artiq-files/install/ndscan/test/test_experiment_subscan.py)
  - `RaggedSubscanDatasetCase.test_ragged_subscan_is_written_as_list_of_lists`
- Slash-to-underscore naming collision risk is locked by test:
  - [test_utils.py](/home/lab/artiq-files/install/ndscan/test/test_utils.py)
  - `ShortenTest.test_shorten_can_collide_after_slash_to_underscore`

## Locked Decisions (for this mini-project)

- Changes should be atomic, tested, and understandable commit-by-commit.
- Rollout is parallel-first:
  - keep legacy list-of-lists output while adding new paths.
- Preview stream and flattened persistent stream are separate concerns.
- New preview/flattened channel identifiers should be deterministic IDs, not
  sanitized path names.
- Avoid name flattening ambiguity from `"/" -> "_"` in new roots.

## Planned Phases

1. Baseline characterization tests (done).
2. Add sink primitives only:
   - `TeeSink`
   - resettable dataset sink behavior
3. Wire live preview stream:
   - tee child per-point sinks (not aggregate list-at-end sinks)
   - emit preview metadata root
4. Add flattened persistent datasets in parallel:
   - append-only flattened arrays
   - `segments.start` + `segments.len`
5. Add reconstruction helpers and consumer migration path.

## Preview vs Flat Roots

- `subscan_preview.<id>.*`:
  - live in-progress view
  - resettable per outer-point subscan run
  - debugging/applet responsiveness
- `subscan_flat.<id>.*`:
  - persistent flattened archive
  - append-only data across all outer points
  - reconstructed via segment metadata

## Future Shape Constraints (Recorded)

- Flattened storage must support non-grid scans and dynamic subscan lengths.
- It must allow tandem scan traces (multiple handles advanced per point without
  Cartesian expansion).
- Segment indexing should delimit subscan spans for each outer-point context,
  instead of assuming fixed-length rows.
- Reconstruction should be done by slicing contiguous spans (`starts`) from
  append-only arrays.

## Known Naming Trap

The current code path can produce doubled underscores and possible collisions:

- `SubscanExpFragment` currently uses `name_prefix="_"`:
  - [subscan.py:539](/home/lab/artiq-files/install/ndscan/ndscan/experiment/subscan.py:539)
- Subscan channel names are then built as `name_prefix + "channel_" + ...`:
  - [subscan.py:360](/home/lab/artiq-files/install/ndscan/ndscan/experiment/subscan.py:360)
- Top-level path shortening then applies `replace("/", "_")`:
  - [entry_point.py:351](/home/lab/artiq-files/install/ndscan/ndscan/experiment/entry_point.py:351)

This is why names like `float_frag_scan__channel_result` appear in kernel tests,
and why sanitization-only schemes are risky.

## Delimiter Semantics and Naming Policy

Why this keeps biting:

- `.` is the ARTIQ dataset tree separator (dataset namespace hierarchy).
- `/` is ndscan-internal path structure (fragment/channel/pathspec semantics).
- `_` is just a normal character in user-facing names.

So any projection like `"/" -> "_"` is lossy, and can collide with naturally
occurring underscores in real names.

Policy for new schemas (`subscan_preview`, `subscan_flat`):

- Machine keys should use stable deterministic IDs, not path sanitization.
- Human-readable labels/paths should live in metadata (`path`, schema maps, etc.).
- Optional compromise for readability is `<slug>__<short_id>` (never slug alone).
- Current chosen compromise: root IDs are `human_slug__stablehash`.
- Legacy aggregate subscan channels (`scan_axis_*`, `scan_channel_*`) are kept for
  compatibility but are non-archived by default to avoid ragged HDF5 failures.
- `subscan_preview.*` is broadcast-only (non-archived), because nested ragged
  previews can otherwise fail HDF5 writes at end-of-run.

This keeps names robust while preserving readability where needed.

This note captures the minimum context needed to understand where ndscan currently:

1. runs subscans,
2. buffers subscan data in memory, and
3. appends whole subscan rows to datasets as list-of-lists.

It also highlights where to hook changes for incremental subscan output.

## Why This Matters

From [updates.md](/home/lab/artiq-files/install/ndscan/updates.md):

- You want subscan data visible incrementally (not only at subscan completion).
- You care about ragged subscans (variable points per subscan).
- You want to avoid writing subscan results only as "one big list per outer point".

## Core Files To Read First

- [subscan.py](/home/lab/artiq-files/install/ndscan/ndscan/experiment/subscan.py)
- [scan_runner.py](/home/lab/artiq-files/install/ndscan/ndscan/experiment/scan_runner.py)
- [result_channels.py](/home/lab/artiq-files/install/ndscan/ndscan/experiment/result_channels.py)
- [entry_point.py](/home/lab/artiq-files/install/ndscan/ndscan/experiment/entry_point.py)
- Plot-side:
  - [subscriber.py](/home/lab/artiq-files/install/ndscan/ndscan/plots/model/subscriber.py)
  - [subscan.py (plot model)](/home/lab/artiq-files/install/ndscan/ndscan/plots/model/subscan.py)

## Sink Types (Current Behavior)

Defined in [result_channels.py](/home/lab/artiq-files/install/ndscan/ndscan/experiment/result_channels.py):

- `ArraySink`: in-memory append list (`push`, `get_all`, `clear`).
- `AppendingDatasetSink`: dataset-backed append (`set_dataset` on first value, then `append_to_dataset`).
- `ScalarDatasetSink`: dataset overwrite per push.
- `SingleUseSink`: point-completeness guard used by `ResultBatcher`.
- `LastValueSink`: keep only latest value.

Important: subscan internals currently use `ArraySink`, then bulk-push at subscan end.

## High-Level Data Path

## 1) Subscan setup rewires child channels into `ArraySink`

In `setup_subscan()`:

```python
# ndscan/experiment/subscan.py
child_result_sinks = {}
for channel in original_channels.values():
    sink = ArraySink()
    channel.set_sink(sink)
    child_result_sinks[channel] = sink
```

Also, scan-axis coordinate sinks are `ArraySink` in `Subscan.set_scan_spec()`:

```python
self._coordinate_sinks[param_handle] = ArraySink()
```

## 2) Scan runner pushes one point at a time, but only after point completeness check

In [scan_runner.py](/home/lab/artiq-files/install/ndscan/ndscan/experiment/scan_runner.py):

- `ResultBatcher` temporarily swaps channel sinks to `SingleUseSink`.
- After `run_once()`, `ensure_complete_and_push()` forwards exactly one value/channel to the original sinks.
- Axis coordinates are then pushed to axis sinks.

Host path:

```python
result_batcher.ensure_complete_and_push()
for sink, value in zip(self._axis_sinks, axis_values):
    sink.push(value)
```

Kernel path equivalent is `_point_completed()`.

## 3) Subscan completion bulk-pushes lists to parent channels

In `Subscan._push_coordinates()` and `_push_values()`:

```python
v = sink.get_all()
channel.push(v)  # or aggregate_result_channel.push(v)
sink.clear()
```

This is the key "append all at once" step for subscan rows.

## 4) Top-level scan channels use `AppendingDatasetSink`

In [entry_point.py](/home/lab/artiq-files/install/ndscan/ndscan/experiment/entry_point.py), top-level scanned channels are assigned:

```python
sink = AppendingDatasetSink(self, self.dataset_prefix + "points.channel_" + name)
channel.set_sink(sink)
```

So when a subscan parent channel receives `v` where `v` is a list, each outer point appends one list.

Result: dataset shape becomes list-of-lists.

## Observed Dataset Shape (Host Repro)

For outer scan over `num_scan_points = [2, 4, 3]` and inner subscan over `value`:

- `points.axis_0 => [2, 4, 3]`
- `points.channel_scan_axis_0 => [[0.0, 1.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]`
- `points.channel_scan_channel_result => [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0]]`

That confirms ragged list-of-lists currently occur at the top-level dataset for subscan channels.

## Plot/Data Consumer Implications

- Top-level subscriber treats `points.*` as append or rewrite based on prefix comparison: [subscriber.py](/home/lab/artiq-files/install/ndscan/ndscan/plots/model/subscriber.py).
- Subscan plot model reads subscan data from selected top-level point (`SinglePointModel`) and always emits rewritten point data for that subscan view: [plots/model/subscan.py](/home/lab/artiq-files/install/ndscan/ndscan/plots/model/subscan.py).
- Existing flow assumes per-outer-point subscan payloads are complete arrays.

## Best Current Hook Points For Incremental Subscan Output

If implementing the incremental route discussed in `updates.md`:

1. `setup_subscan()` in [subscan.py](/home/lab/artiq-files/install/ndscan/ndscan/experiment/subscan.py):
   - Replace plain `ArraySink` with a tee-like sink (`ArraySink` + dataset sink) for child results and coordinates.
2. Add sink(s) capable of reset/clear semantics for in-progress datasets.
3. Emit subscan metadata similarly to `_broadcast_metadata()` in [entry_point.py](/home/lab/artiq-files/install/ndscan/ndscan/experiment/entry_point.py).
4. Keep current final aggregate push path initially for compatibility, then iterate.

## Code Pointers You Will Reopen Frequently

- Subscan channel/sink wiring:
  - [subscan.py:301](/home/lab/artiq-files/install/ndscan/ndscan/experiment/subscan.py:301)
  - [subscan.py:339](/home/lab/artiq-files/install/ndscan/ndscan/experiment/subscan.py:339)
  - [subscan.py:155](/home/lab/artiq-files/install/ndscan/ndscan/experiment/subscan.py:155)
  - [subscan.py:168](/home/lab/artiq-files/install/ndscan/ndscan/experiment/subscan.py:168)
- Point-completeness + per-point push:
  - [scan_runner.py:154](/home/lab/artiq-files/install/ndscan/ndscan/experiment/scan_runner.py:154)
  - [scan_runner.py:248](/home/lab/artiq-files/install/ndscan/ndscan/experiment/scan_runner.py:248)
  - [scan_runner.py:462](/home/lab/artiq-files/install/ndscan/ndscan/experiment/scan_runner.py:462)
- Dataset sink behavior:
  - [result_channels.py:88](/home/lab/artiq-files/install/ndscan/ndscan/experiment/result_channels.py:88)
  - [result_channels.py:110](/home/lab/artiq-files/install/ndscan/ndscan/experiment/result_channels.py:110)
  - [result_channels.py:138](/home/lab/artiq-files/install/ndscan/ndscan/experiment/result_channels.py:138)
- Top-level channel sink assignment:
  - [entry_point.py:346](/home/lab/artiq-files/install/ndscan/ndscan/experiment/entry_point.py:346)
  - [entry_point.py:355](/home/lab/artiq-files/install/ndscan/ndscan/experiment/entry_point.py:355)

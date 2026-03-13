**Purpose**

This note describes a plausible future path for supporting kernel-executed leaf
fragments within the new host runtime.

The intent is *not* to move the whole scan controller onto the core device.
Instead:

- the host still owns scan orchestration, batching, online/final analysis, previews,
  and adaptive point choice;
- the core device only executes a tight inner loop for a batch of already-chosen
  points;
- results return to the host at batch boundaries.

This matches the structure of the current host runtime well, and avoids recreating
the complexity of the legacy runtime.


**What ARTIQ Does Today**

ARTIQ itself runs experiments in a separate worker *process*, not a Python thread.
The master launches:

```text
python -m artiq.master.worker_impl ...
```

and communicates with the worker process over IPC.

The worker then executes:

- `prepare()`
- `run()`
- `analyze()`

synchronously in that worker process.

Relevant upstream files:

- `artiq.master.worker`
- `artiq.master.worker_impl`


**Are Kernel RPCs Allowed?**

Yes. Kernel code can call host RPCs.

The question is not whether they are allowed, but whether they are acceptable in the
hot per-point loop.

For a kernel-batched leaf executor, the goal is:

- choose a batch on the host,
- ship the batch to the core once,
- execute the batch in a tight loop on the core,
- return one batch of results,
- then do host-side feedback/analysis.

If each point in that inner loop performs host RPCs, then much of the benefit of
kernel batching is lost:

- latency returns to the hot loop,
- throughput becomes dominated by host/core round-trips,
- timing becomes less predictable,
- adaptive logic becomes harder to reason about.

So kernel RPCs are not forbidden; they are just the wrong tool for the proposed
fast-path.


**How the Legacy Kernel Scan Runner Works**

The legacy kernel runner in
[scan_runner.py](/home/lab/artiq-files/install/ndscan/ndscan/experiment/scan_runner.py)
already uses a hybrid host/core approach.

Important details:

1. The host chooses points and sends them to the kernel in chunks via a synchronous
   RPC:
   - `_get_param_values_chunk()`

2. The kernel iterates over the chunk in a generated `run_chunk()` loop.

3. Result channels still stream back to the host through RPC-based sinks.

4. After a point finishes, `_point_completed()` is called as an async RPC so the host
   can:
   - validate that all result channels were pushed,
   - forward result values through the original sinks,
   - record axis coordinates.

So the legacy kernel runner does *not* avoid host RPCs in the per-point path. It just
reduces some overhead by:

- fetching scan points in chunks, and
- using async RPCs for result forwarding.


**How Result Channels Work in the Legacy Kernel Path**

The current result-channel/sink model is host-oriented.

Examples in
[result_channels.py](/home/lab/artiq-files/install/ndscan/ndscan/experiment/result_channels.py):

- `ResultChannel.push(...)` is an async RPC.
- `NumericChannel.push(...)` is `@portable`, updates local cached state, and then calls
  `_push(...)`, which is an async RPC back to the host.
- `ResultBatcher` lives on the host and temporarily replaces sinks with
  `SingleUseSink`.

That means the kernel code can still call `self.result.push(...)`, but the actual sink
graph is still resolved on the host.

This is why the current sink model is not a good fit for an ideal kernel-batch fast
path: it assumes the host remains involved for each point's results.


**Why the Current Sink Model Is Not the Right Fast Path**

For a future kernel-leaf batch executor, the desired data path is:

1. Host computes one batch of actual parameter values.
2. Core executes all points in one loop.
3. Core writes results into fixed-shape buffers.
4. Host receives the completed buffers once per batch.

That is different from today's sink model, where each point's result is pushed through
host-facing result channels immediately.

The current model is still useful as the *declaration* surface:

- fragments declare `ResultChannel`s,
- callers know which results exist,
- metadata generation can still reuse channel schemas.

But the *collection* path for a kernel batch should likely bypass normal sinks and use
preallocated arrays/buffers instead.


**Proposed Architecture**

Add a dedicated leaf-batch executor abstraction:

```python
class BatchLeafExecutor:
    def execute_batch(self, points) -> list[PointObservation]:
        ...
```

Two implementations:

- `HostLeafExecutor`
- `KernelLeafExecutor`

The rest of the host runtime should stay the same:

- `PointSource` / point policy chooses the next batch
- host applies parameter mappings and builds actual point values
- executor runs the batch
- host writes datasets, runs online analysis, updates point policy, checks pause, and
  maybe writes a preview snapshot


**KernelLeafExecutor Shape**

Conceptually:

1. Host chooses a batch of logical points.
2. Host applies mappings and resolves actual fragment parameter values.
3. Host packs those values into fixed-type arrays.
4. Host calls one kernel method with:
   - input parameter arrays,
   - output result arrays,
   - batch length.
5. Kernel loops over the batch with no per-point host RPCs.
6. Host converts the returned arrays into `PointObservation`s.

Pseudo-shape:

```python
@kernel
def run_batch(self, param_a, param_b, out_y, n):
    self.device_setup()
    for i in range(n):
        self.param_a_store.set_value(param_a[i])
        self.param_b_store.set_value(param_b[i])
        self.run_once()
        out_y[i] = self.result_y_last
    self.device_cleanup()
```

This is intentionally only a sketch. The real design needs a precise contract for:

- parameter-store installation,
- result extraction,
- supported result types,
- retry/error handling.


**Supported Fragment Subset**

The first version of a kernel leaf executor should be strict.

Likely requirements:

- `run_once()` is kernel-safe.
- `device_setup()` / `device_cleanup()` are kernel-safe.
- result schemas are fixed and statically known.
- result values are scalar or fixed-shape arrays.
- no nested scans inside that kernel executor.
- no host RPC in the per-point hot loop.

This keeps the feature understandable and avoids recreating legacy complexity.


**What Stays on the Host**

The following should remain host-side even when a leaf executor uses the core device:

- point policy / adaptive point choice
- parameter mappings
- batch boundaries
- online analysis
- final analysis
- preview HDF5 snapshots
- pause/yield policy
- nested scan orchestration

This is the key architectural decision. Kernel support should accelerate the inner leaf
loop, not absorb the whole runtime.


**What This Means for Adaptive Scans**

Adaptive feedback still works naturally, but only at batch boundaries:

1. host chooses batch
2. core executes batch
3. host runs online analysis
4. point policy updates from `BatchFeedback`
5. host chooses next batch

This is a good match for the current batch-oriented host runtime.


**Non-Goals for a First Version**

The first kernel-leaf batching implementation should *not* try to support:

- arbitrary host callbacks during the point loop
- nested subscans on the core
- ragged/variable-length point results
- full parity with every legacy kernel-runner behavior

Those are exactly the kinds of features that made the old runtime hard to reason
about.


**Recommended Next Step**

Before implementing kernel-leaf batching, the best next design task would be to write
down:

- the supported fragment contract,
- the exact input/output array format,
- how result channels map to output buffers,
- and where retry/error handling belongs.

That would be enough to evaluate feasibility without yet changing the current host
runtime.

# Host Runtime: Dynamic vs Prepared Scans

## Purpose

This note captures the current design discussion around kernel execution in the newer
host-runtime direction.

The most important distinction is not:

- top-level scan vs subscan
- host scan vs kernel scan

It is:

- **dynamic scan invocation**
- **prepared/static scan invocation**

That distinction is what determines whether ARTIQ can compile once and then reuse the
compiled kernel efficiently.


## Quick Primer

For readers not already steeped in ARTIQ or compiler constraints, the key idea is:

- **dynamic** code decides what to do while it is already running
- **prepared/static** code fixes the executable shape first, and then runs it many
  times with different data

In ordinary Python, dynamic behavior is natural and cheap. A function can build a new
object, call another helper, inspect a rich result, and then decide what to do next.

In ARTIQ kernel code, that style is much more expensive because:

- kernels are ahead-of-time compiled
- kernel compilation can take many seconds
- kernels are not cached
- compiled code prefers a fixed call graph and a fixed data interface

So the practical performance rule is:

- if the executable structure keeps changing, recompilation pressure appears
- if the executable structure is fixed, values can still change cheaply at runtime

That is why this note talks about **execution shape** rather than whether all scan
points must be known in advance.


## Static vs Dynamic In A Familiar Programming Sense

This is not exactly the same as classic "static vs dynamic dispatch", but it is related
in spirit.

Dynamic style:

```python
def run_dynamic(kind, x):
    if kind == "ramsey":
        return run_ramsey_scan(x)
    if kind == "rabi":
        return run_rabi_scan(x)
    return run_sideband_scan(x)
```

Here the program decides at runtime which whole scan shape to execute.

Prepared/static style:

```python
class PreparedRamseyScan:
    def __init__(self, fixed_shape_config):
        self.config = fixed_shape_config

    def run(self, points):
        # Same executable shape every time, new point values allowed.
        return run_precompiled_ramsey(points, self.config)
```

Here the executable structure is chosen first, and then reused with different data.

That is closer to the distinction in this note than classic dynamic dispatch. The real
question is not just "which function do I call?", but:

- what whole executable region exists?
- what inputs does it expect?
- what outputs does it produce?
- can that region be reused without recompilation?


## Short Version

If a scan should run efficiently on the core device, then the *shape of the execution*
must be fixed before the kernel is entered.

That does **not** mean all point values must be known in advance.

It means things like:

- which fragment/subtree is being executed
- what result values come back
- what fields/state the compiled code touches
- what nested scan objects exist
- what the call graph looks like

must already be known.

Point values may still come from the host later via RPC. Batch contents may still be
chosen adaptively. The host may still remain the scan orchestrator.

So the right summary is:

- **prepared** means fixed executable structure
- **dynamic** means the host is still free to construct new scan structure ad hoc at
  runtime


## Why This Matters For ARTIQ

ARTIQ kernel execution is ahead-of-time compiled from a static call graph.

That has a few consequences:

1. Kernel compilation is expensive.
   In bad cases, compilation takes many seconds. If a fragment enters the kernel once
   per point or once per tiny nested scan, runtime becomes dominated by recompilation.

2. ARTIQ does not cache kernels.
   Re-entering a logically "same" kernel later is still costly. The legacy
   `KernelScanRunner` avoids this by entering one long-lived kernel region and then
   fetching more work over RPC.

3. The compiler wants static structure.
   Rich Python objects, dynamically constructed nested sessions, ad hoc result shapes,
   and arbitrary host-created control flow are not what the ARTIQ compiler is designed
   to make cheap.

4. Nested kernel-friendly scans need a predeclared ABI.
   A compiled kernel can naturally call another predeclared compiled block. It does not
   naturally call a host helper that constructs a fresh Python scan session and returns
   a rich Python result object.

This is why "if you want performance, you are not allowed to change your mind" is
mostly correct, but should be stated more precisely:

- you **may** still change the next points
- you **may** still choose batches adaptively on the host
- you **may** still stream work from the host by RPC
- you **may not** dynamically change the executable scan structure from inside the
  already-compiled kernel region


## Dynamic Scans

The current host runtime already has a good dynamic host-side scan API:

- `run_host_scan(...)`
- `run_subscan(...)`

The important property of these helpers is that they are convenient and expressive.

They let user code:

- construct a fresh `ScanRequest` at the point of use
- run it immediately
- get back a rich Python `HostScanRunResult`
- inspect coordinates, values, parameters, and analysis results directly

That is what makes examples like:

```python
x_request = ScanRequest.cartesian([(self.line.x, x_points)])
x_result = run_subscan(self, self.line, x_request, name="scan_x")
self.m.push(x_result.analysis_results["m"])
```

pleasant to write.

### What Dynamic Gives You

- no predeclaration in `build_fragment()`
- no separate "prepared scan" object
- easy construction of scan requests from current host-side data
- rich immediate Python return values
- natural composition for host-only nested scans

### Why Dynamic Is Hard To Call From Kernel Code

The same features that make dynamic scans pleasant on the host make them a poor fit
for compiled kernel use:

- they create fresh Python-level scan sessions dynamically
- they return rich Python objects (`HostScanRunResult`)
- they rely on host-side orchestration as part of the API itself

So a dynamic helper like `run_subscan(...)` is a good **host convenience API**, but it
is not a good **kernel execution ABI**.


### What Is Genuinely Dynamic?

Very few things are mathematically forced to be dynamic. But some things are
*practically dynamic* unless they are redesigned into a fixed ABI.

Typical examples:

- the fragment or subtree to scan is chosen at runtime
- the number or identity of outputs changes at runtime
- the scan dimensionality or structure changes at runtime
- the caller expects a rich Python result object and then makes more Python decisions
  from it
- arbitrary Python callbacks or closures are created dynamically and become part of the
  scan's behavior

These can often be made static only by introducing:

- a fixed superset interface
- a prepared object with predeclared outputs
- or a more interpreter-like protocol

That redesign may be worthwhile, but it is a real redesign.

### When Repeat Compilation Becomes Unavoidable

Repeat compilation pressure appears when all three of these are true:

- the scan structure stays dynamically shaped
- that dynamically shaped work still needs to execute as kernel code
- no fixed prepared ABI is introduced

That is the real tradeoff.

So the practical options are usually:

1. keep the dynamic part on the host
2. redesign it into a prepared/static interface
3. accept that recompilation will happen


## Prepared Scans

A prepared scan is the kernel-friendly counterpart.

Prepared does **not** mean:

- all points are fixed forever
- no adaptive behavior is allowed
- the host is no longer involved

Prepared **does** mean:

- the scan target/subtree is predeclared
- the executable shape is stable
- the output interface is fixed
- the compiler can see the relevant fields and call graph ahead of time

Conceptually, a prepared scan object would be something like:

- `PreparedScan`
- `PreparedChildScan`

with separate phases:

1. prepare/build the scan object once
2. configure it from the host as needed
3. execute it many times without recompilation

This is very similar in spirit to what legacy `SubscanExpFragment` was doing for
kernel-capable nested scans: configuration is separate from execution, and the
kernel-callable part is static enough for the compiler.


## Prepared Does Not Mean Fixed Points

This point is easy to miss and worth making explicit.

A prepared kernel-capable scan can still work like this:

1. enter a long-lived kernel execution region once
2. kernel asks host for the next batch by blocking RPC
3. host chooses the batch using `PointPolicy`
4. kernel executes the batch
5. host observes the results and chooses the next batch

So:

- batch contents may still be dynamic
- adaptive scans are still possible
- the host may still own the scan controller

What is fixed is the *execution interface*, not the numerical data.


## Recommended Structure

The cleanest structure for ndscan is:

### 1. One Semantic Scan Model

Keep one user-facing scan model:

- `ScanRequest`
- nested scans
- point policies
- parameter mappings
- analyses

This layer should describe *what* is being scanned, not *how* it executes.

### 2. One Host-Side Controller

The host should remain the overall controller for:

- point-policy decisions
- adaptive feedback
- dataset writing
- online/final analyses
- preview snapshots
- pause/scheduler interaction

### 3. Multiple Execution Backends

Execution backend is then an implementation detail:

- `HostExecutor`
- `KernelStreamingExecutor`
- maybe later `KernelAutonomousExecutor`

The first kernel backend should be the simple one:

- kernel stays resident
- host still chooses batches by RPC
- main goal is avoiding repeated compilation

### 3a. Optional Kernel-Capable Implementations Of The Same Concepts

One important design goal is that kernel support should not require a second semantic
API.

The better direction is:

- one semantic scan model
- one set of core concepts
- optional kernel-capable implementations of those concepts where that makes sense

In other words, kernel support should usually be an *implementation capability*, not a
different user-facing ontology.

The most useful examples are:

#### Point policy

- host implementation is always available
- some simple policies may also have a kernel-capable implementation

Examples of policies that could plausibly support this later:

- linear
- list
- cartesian
- zipped
- simple local adaptive policies

The important point is that this does **not** require a different public scan concept.
It just means some `PointPolicy` implementations are kernel-capable and some are not.

#### Analysis

- host-side analysis remains the default
- some simple analyses may also have kernel-capable reducers

Examples:

- fixed-shape numeric reductions
- least-squares fit of a straight line
- simple extrema or averages

Again, this should be treated as an optional implementation capability of the same
analysis concept, not as a separate "kernel analysis API" that user code must learn
from scratch.

#### Result transport

- host/default path: normal result-channel push/sink model
- optional fast kernel path: preallocated buffers or other kernel-friendly collection
  path

This is another place where it is tempting to invent a second system. The cleaner view
is:

- result channels are still the declaration surface
- result transport may vary by execution backend

So the execution backend chooses whether to use:

- the normal host-oriented push/sink path
- or a stricter kernel-friendly buffered transport path

### 3b. Why This Layering Matters

This layering keeps the architecture from splitting into two parallel worlds:

- host scans
- kernel scans

Instead, it becomes:

- one scan model
- one controller model
- several execution capabilities

That is a much better fit for the long-term goal of ndscan supporting:

- host-only scans
- kernel-resident execution
- maybe later kernel-autonomous execution
- maybe later kernel-capable analyses

without exposing a completely different user model for each.

### 4. Two Invocation Styles

These are the two user-facing invocation styles that naturally fall out:

#### Dynamic convenience

- `run_scan(...)`
- `run_child_scan(...)`

Properties:

- host-only convenience
- returns rich Python results
- can build requests ad hoc

#### Prepared execution

- `prepare_scan(...) -> PreparedScan`
- `prepare_child_scan(...) -> PreparedScan`

Properties:

- static enough for kernel execution
- configured before use
- run/acquire many times without recompilation
- results exposed through a fixed interface


## Why One API Surface Is Not Enough

In general, fewer APIs are better. But here the distinction is justified because the
two modes provide genuinely different capabilities.

### Dynamic API gives:

- ad hoc scan construction
- rich Python return objects
- minimal ceremony

### Prepared API gives:

- kernel-callable execution
- reusable compiled execution regions
- explicit static interface

Trying to force both into one literal function usually means one of two bad outcomes:

1. the API becomes so vague that users cannot tell when it is kernel-safe
2. the implementation becomes so complex that the runtime ends up with hidden special
   cases everywhere

So the right goal is:

- one semantic model
- one implementation core
- two invocation styles where the difference is real


## Root Scan vs Child Scan Is Not The Main Distinction

It is tempting to talk about:

- top-level scan
- subscan

but this is no longer the most useful split.

In the current host runtime, `run_subscan(...)` already just derives a child site and
then delegates into the same session/controller path as the root scan.

So the meaningful distinction is not:

- root
- nested

It is:

- dynamic
- prepared

Top-level scans are special mostly because ARTIQ needs an `EnvExperiment` entrypoint
and because the root owns run-wide policies such as preview cadence. Semantically, root
and nested scans should remain as similar as possible.


## What This Means For Kernel-Friendly Nested Scans

Suppose:

- `HowDoesPVaryFragment` runs a scan over `p`
- each `p` point runs a scan over `x`
- `LineFragment.run_once()` is `@kernel`

If only the leaf is kernel-capable, the ideal behavior is:

- the host still orchestrates the nested scan structure
- the `LineFragment` leaf execution uses one resident kernel session
- batches are still fed from the host by RPC
- the same leaf kernel session is reused across repeated nested invocations

That already solves the important problem:

- no repeated compilation of the leaf point body

If later a parent fragment such as `ScanXFragment.run_once()` also needs to be `@kernel`
while still invoking a nested scan efficiently, that is where prepared child scans
become necessary. A dynamic helper returning `HostScanRunResult` is simply the wrong
shape for compiled code.


## Why A Prepared Scan Is Still Not "Kernel Autonomy"

Prepared scans and kernel-autonomous scans are separate ideas.

A prepared scan says:

- the executable structure is static enough for the compiler
- the kernel can stay resident
- the interface between host and kernel is stable

It does **not** say:

- the kernel must generate points itself
- the host must disappear from orchestration
- all adaptive logic must move to the core device

This distinction matters because the first performance goal is usually just:

- avoid recompiling kernels

That is already solved by a kernel-streaming/prepared approach where:

- the host remains the controller
- the host still chooses batches
- the kernel stays resident and executes them

Only later, if it is worth the complexity, should ndscan add:

- kernel-autonomous point generation for a subset of point policies
- kernel-capable analyses/reducers for a subset of analyses

So the natural order of ambition is:

1. prepared + kernel-streaming
2. prepared + kernel-autonomous
3. broader fusion of larger all-kernel subtrees


## Why Legacy Kernel Ideas Still Matter

For the specific problem of avoiding recompilation, the legacy runtime already contains
the right basic idea:

- one resident kernel region
- host feeding work by RPC
- nested kernel-capable execution using a prepared/static path rather than a dynamic
  host convenience helper

What the newer host runtime improves is not that kernel story by itself, but the place
where that story should live.

The cleaner end state is therefore:

- keep the newer scan semantics and controller structure
- recover the legacy kernel-residency idea as an execution backend
- avoid rebuilding the old top-level/subscan/host/kernel split as parallel semantic
  systems


## What This Means For The Current Runtime

The current host runtime has already won something important:

- a cleaner semantic/controller model than legacy ndscan

But it has not yet won the kernel residency story.

The right next step is therefore not to create a second independent runtime again. It
is to:

1. keep the newer semantic/controller structure
2. add a clean executor/backend seam
3. reintroduce the good legacy idea of one resident kernel execution region
4. support prepared nested scans for kernel use when needed

In other words:

- use the newer runtime as the semantic core
- steal the kernel-residency idea from the legacy path


## Suggested Naming Direction

The current names are slightly misleading because they mix semantic structure and
implementation history.

In particular, `run_subscan(...)` sounds more legacy-specific than the actual current
design.

If naming is revisited, the clearest public split would be:

- `run_scan(...)`
- `run_child_scan(...)`
- `prepare_scan(...)`
- `prepare_child_scan(...)`
- `PreparedScan`

This makes the important distinction explicit:

- dynamic vs prepared

rather than:

- host vs kernel
- top-level vs subscan


## Naming Principles

If naming is revisited, the following principles should help:

1. Public names should describe semantics, not implementation backend.
   Prefer:
   - `scan`
   - `child`
   - `prepared`
   - `dashboard`
   over:
   - `host`
   - `kernel`
   - `subscan`

2. Internal names may describe execution strategy.
   Executor/backend names such as:
   - `HostExecutor`
   - `KernelStreamingExecutor`
   - `KernelAutonomousExecutor`
   are useful because they make the execution distinction explicit in the right place.

3. "Subscan" is legacy-loaded and slightly misleading.
   In the newer model, the more meaningful distinction is usually:
   - root vs child site
   - dynamic vs prepared invocation
   not:
   - scan vs subscan

4. The same user-facing scan model should survive backend changes.
   A user should ideally think:
   - "I am running a scan"
   and only secondarily:
   - "this scan is executing through a kernel-streaming backend"


## Final Summary

The right mental model is:

- dynamic scans are the rich host-side convenience surface
- prepared scans are the static execution surface that keeps ARTIQ happy

Prepared scans are not about fixing all point values ahead of time.
They are about fixing the *shape* of execution so the compiler can compile once and the
runtime can reuse that compiled region efficiently.

So the practical rule is:

- if you want flexibility, use dynamic scans
- if you want kernel performance, accept that the executable scan structure must be
  prepared ahead of time

That tradeoff is reasonable, and making it explicit in the API is better than hiding it
behind unclear names or runtime heuristics.

The companion architectural principle is:

- kernel support should be added as optional execution capabilities of the same scan
  concepts

not as an entirely different semantic runtime.

# Prepared Scans vs Legacy Subscans

This note is discussion material for a possible `ndscan v2` direction.

The short version is:

- the new prepared-scan model has an API that looks closer to legacy
  `setattr_subscan(...)`
- but it has execution mechanics much closer to what `SubscanExpFragment` was trying
  to achieve
- and it does so inside a cleaner overall runtime model

So the claim is not merely that the new code is "different". The claim is that it now
seems to preserve the good part of both legacy approaches while dropping much of the
old conceptual split.


## Short Thesis

The prepared-scan model appears to be a better foundation than either legacy subscan
API taken as-is because it gives us:

- composition instead of a special subclass-heavy mental model
- a fixed prepared execution shape for kernel-capable scans
- one request model for root and nested scans
- one batch-oriented runtime for host and kernel execution
- one stable output contract

The important shift is this:

- a scan is described by a `ScanRequest`
- a scan is executed through a prepared handle
- the host still chooses batches
- the executor backend is an implementation detail

That is a better long-term story than:

- "scan submissions" as one kind of runtime
- "kernel scans" as another kind of runtime
- "subscans" as a third special thing


## The Two Legacy Subscan Ideas

### `setattr_subscan(...)`

The attractive part of legacy `setattr_subscan(...)` was its API shape.

It felt natural in ordinary Python:

- create a handle for a child scan
- call it from the parent
- get results back directly

That made subscans feel like composition rather than like a separate fragment taxonomy.

But the model it taught was still mostly host-oriented:

- execution, setup, and result collection were bundled together
- the main return shape was a rich Python result object
- the natural way to think about it was "run some helper and get arrays/dicts back"
- kernel support was something that had to be layered on afterwards

So it was pleasant, but it pushed the design toward "subscan as Python helper call".

### `SubscanExpFragment`

The attractive part of `SubscanExpFragment` was its execution shape.

It captured several things that matter for ARTIQ performance:

- the nested scan structure is fixed before kernel entry
- the compiler sees a stable call path
- the nested scan can be reused without recompiling for every tiny invocation
- repeated kernel subscans can become practical

That part was right.

But the surface model was heavier:

- a special fragment kind
- inheritance as a more prominent pattern
- a stronger distinction between "normal fragment" and "scan-driving fragment"
- a weaker feeling that root scans and child scans are really the same concept

So it had better kernel mechanics, but less pleasant overall structure.


## Why The Prepared-Scan Model Looks Better

The current model looks like it combines the useful part of both older approaches:

- composition and handles, like the simpler legacy host style
- prepared compiler-visible execution shape, like the better legacy kernel style

without forcing us to preserve the old conceptual split.

### Better than `setattr_subscan(...)`

The main improvement is that the primary contract is no longer a rich host-only result
object.

The core shape is now:

- `configure(request)`
- `execute()`
- `outputs()`
- `get_outputs()`
- `inspect()`

That is a much better foundation for something that is supposed to work both:

- on the host
- and through a kernel-resident execution backend

The host can still provide rich inspection data, but it is no longer the thing that
defines the scan model.

### Better than `SubscanExpFragment`

The main improvement here is that we seem to get the same fixed prepared execution
shape without making the main abstraction a special scan-driving fragment subclass.

The compiler-facing facts are still right:

- prepared objects exist before kernel entry
- nested execution uses a fixed callable path
- host-chosen batches still flow through a stable ABI
- repeated recompilation is avoided

But the user-facing API remains handle- and composition-oriented.

So the new model feels less like "there are special scan fragments" and more like
"there is one scan runtime with prepared handles".

### The key phrase

If the old choice was roughly:

- nice API but host-centric
- or good kernel mechanics but heavier structure

then the current model is aiming for:

- nice API outside
- prepared static execution inside


## What "Prepared" Means Here

"Prepared" does not mean "all points are known in advance".

It means:

- the executable scan structure is fixed
- the output contract is fixed
- the host may still choose future batches adaptively
- the executor backend, if kernel-backed, can remain resident

So a prepared scan is still compatible with:

- adaptive point choice
- Bayesian optimisation
- online analysis
- nested scans
- host-side persistence after each batch

Prepared is about static execution shape, not about removing host control.


## Root And Child Are The Same Concept

One of the most important benefits of the newer design is that root and child scans now
look like the same idea instead of two separate runtimes.

The model is:

- `PreparedScan` at the root
- `PreparedChildScan` for nested scans
- `ScanRequest` for both
- `ScanOutputs` for both
- host-side `ScanInspection` for both

This is a much cleaner story than:

- one runner model for the top level
- another runner model for subscans
- special-case return types depending on where the scan lives

The root scan is just a root prepared scan. The child scan is just a nested prepared
scan.


## The Current Creation Routes Are Really One Thing

There are several ways to arrive at a prepared scan today, but they are all just front
doors into the same runtime concept.

### 1. Explicit code path

This is the clearest expression of the model:

```python
request = ScanRequest.explicit([...], [...])
scan = prepare_scan(self, self.fragment, request=request, expose_outputs=["fit_m"])
scan.execute()
fit_m, = scan.get_outputs()
```

Here the user explicitly says:

- this is the fragment being scanned
- this is the request
- this is the declared stable output surface

### 2. Explicit child path

Nested scans use the same idea:

```python
self.child_scan = self.setattr_prepared_child_scan(
    "child_scan",
    ChildFragment,
    expose_outputs=["fit_m"],
)
```

and later:

```python
self.child_scan.configure(request)
self.child_scan.execute()
m, = self.child_scan.get_outputs()
```

That is not a different nested runtime. It is the same prepared-scan contract applied
inside the tree.

### 3. Code convenience wrapper

The helper:

```python
MyExperiment = make_fragment_prepared_scan_exp(
    MyFragment,
    request_factory,
)
```

is not a different scan model.

It is just an `EnvExperiment` convenience wrapper that:

- instantiates the fragment
- calls the request factory
- constructs a root `PreparedScan`
- executes it

This is useful because it removes boilerplate, but it should be understood as thin
sugar over the same prepared-scan runtime.

### 4. Dashboard convenience wrapper

Likewise:

```python
MyDashboardExperiment = make_fragment_prepared_dashboard_scan_exp(MyFragment)
```

is not a different scan model.

It simply takes a request description from the UI path, compiles it into the internal
runtime form, and then runs the same root prepared scan.


## Code Path vs Dashboard Path

The desired design rule seems to be:

- code and dashboard may differ in how the request is authored
- they should not differ in the runtime they enter afterwards

That means:

- code creates a `ScanRequest` directly
- the dashboard compiles a UI/schema payload into a `ScanRequest`
- both then execute through `PreparedScan`

So the dashboard path is conceptually:

- UI payload
- `compile_scan_submission_schema(...)`
- `ScanRequest`
- `PreparedScan`
- executor selection

not:

- UI scan system
- separate dashboard runtime
- special "scan submission" execution species

This is a very good kind of separation:

- different front doors
- same runtime object underneath


## What Happens For A Kernel-Capable Scan

The key point is that the same prepared scan is used whether execution ends up on the
host or on the kernel backend.

The host still owns:

- point choice
- batch boundaries
- online/final analyses
- persistence and HDF5 flushing
- result-site bookkeeping

The kernel backend only changes how the prepared point body executes.

So when a scan is kernel-eligible, the flow is still:

- build a `PreparedScan`
- configure a `ScanRequest`
- execute the scan

and then internally:

- the runtime selects the kernel executor
- one resident kernel execution region is entered
- the kernel asks the host for the next batch by RPC
- batches are streamed until completion

That is the important performance property:

- one prepared scan object
- one execution contract
- one resident kernel region
- host-fed batches at every level

This is why the newer model can look like legacy host composition on the outside while
delivering the kind of prepared kernel behavior that previously pushed the design
toward `SubscanExpFragment`.


## Why The Helper Still Matters

It would be a mistake to conclude that the convenience wrapper should disappear just
because the core prepared-scan model is now explicit.

The helper still removes real boilerplate:

- fragment construction
- request wiring
- `EnvExperiment` scaffolding
- root prepared-scan setup

So the right layering seems to be:

### Explicit low-level runtime API

- `prepare_scan(...)`
- `prepare_child_scan(...)`
- `setattr_prepared_child_scan(...)`

### Thin convenience wrappers

- `make_fragment_prepared_scan_exp(...)`
- `make_fragment_prepared_dashboard_scan_exp(...)`

That gives us:

- one real runtime concept
- one explicit low-level API
- one thin convenience layer

which is better than forcing every example to spell out the boilerplate by hand.


## Philosophically: Should Everything Be A Scan?

Inside the ndscan runtime, probably yes.

A one-point request is just the degenerate scan case.

That gives one model for:

- point selection
- batching
- persistence
- outputs
- root/nested composition
- host/kernel backend choice

The main obvious exception is the manual escape hatch:

- a hand-written fixed kernel mini-procedure

That is still a useful pattern, but it is not necessarily an ndscan scan. It does not
need to be forced into the same runtime if it is really just a manual compiled helper.

So the likely rule is:

- within ndscan runtime concepts, everything is a scan
- outside that, fully manual kernel procedures remain allowed


## Honest Caveats

This model still has a few rough edges.

### The transport naming is not fully cleaned up

The dashboard submission path still contains names like `scan_submission` in some places.
That is more historical naming than conceptual truth.

### Convenience layers can still hide the core model

If examples only show the wrapper path, users can miss that the real runtime concept is
`PreparedScan`.

### Host inspection is still richer than stable outputs

That is acceptable and useful, but it is important that `ScanInspection` remains
clearly secondary to `ScanOutputs`.


## Practical Conclusion

The prepared-scan model now seems to support a cleaner statement of what ndscan is:

- one scan model
- one request model
- one batch model
- one output model
- multiple execution backends

That is a better basis for a future `v2` than preserving either legacy subscan API
shape as the main user model.

The right conclusion does not seem to be:

- keep both legacy ideas exactly as they were

but rather:

- keep the useful ideas they discovered
- express them through `ScanRequest` and prepared scans
- let host and kernel be backend choices rather than different scan species

If that framing is right, then the new runtime is not merely "another subscan API".
It is a cleaner unification of:

- composition
- prepared compiler-visible execution
- host-directed adaptive batching
- root/nested convergence
- code/dashboard convergence

and that feels like the strongest argument for building `ndscan v2` on top of it.

# ndscan Status

Date: 2026-04-20

This note is a checkpoint for the current `ndscan` work, with an emphasis on what is
effectively complete, what is intentionally being left to lab-side code, and what the
next useful `ndscan` tasks are.

## Current Position

The prepared-runtime path is now in a good state for real use. The main structural
pieces are in place:

- prepared-runtime scan-site data is saved in HDF5 in a form that can be reopened
  offline
- the live viewer and the offline reader now share the same site-tree object model
- offline scripts can access saved data through a reasonably clean `ndscan.results`
  API
- structured metadata blobs can be saved by experiments and recovered offline
- completed prepared-runtime runs can be reopened in the runtime viewer

The main remaining work on the `ndscan` side is no longer basic architecture. It is
mostly:

- consolidating what is already there
- tightening a few user-facing plotting/runtime features
- avoiding unnecessary expansion of the API before more real scripts have used it

## Completed Recently

### Offline Results / Reopening Runs

These todo items are effectively done:

- Create library for opening the results file and plotting in the same way
- Check we can reopen the live viewer on completed runs

What exists now:

- `ndscan.results.read_host_runtime_snapshot(...)`
- `HostRuntimeSnapshot`
- `HostRuntimeSite`
- `site.available_series_paths()`
- `site.describe_series()`
- `site.describe_plot_choices()`
- `site.series(...)`
- `site.available_metadata_blob_names()`
- `site.metadata_blobs()`
- `site.require_metadata_blob(...)`
- `site.segments_for_parent_point(...)`
- one nontrivial array-splitting helper in `ndscan.results.series`
- `ndscan_runtime_show result.h5` for prepared-runtime HDF5 snapshots

There are also example scripts showing the intended offline usage pattern, including:

- simple top-to-bottom plotting from saved series
- reopening saved images
- comparing saved live-analysis outputs against offline recomputation
- selecting a parent scan point and plotting the child subscan segment it launched

### Simulation / Three-Image Rearrangement Example

This todo item has progressed substantially:

- Continue increasing complexity in the simulation to level we currently have with
  multiple images, groups, atoms, etc…

The three-image example now covers a more realistic experimental shape:

- multiple images
- multiple groups
- ROI/count/image channels
- saved raw images
- saved live-analysis outputs
- structured imaging metadata blob
- offline scripts that consume the saved data

It is not the final form of the simulation, but it is now rich enough to exercise the
results path seriously.

### Kernel vs PC Timing / Overhead

This todo item has progressed meaningfully:

- Can we benchmark kernel vs PC execution time to see if the library has many
  bottlenecks

What exists now:

- a local/mock prepared-runtime overhead benchmark
- a real-core benchmark path that measures host-observed time spent in `core.run(...)`
- a nested TTL benchmark case
- a nested fixed-wait benchmark case with known core-side wait time

The important conclusion is:

- for the tested prepared-kernel shapes, we have good engineering confidence that the
  runtime is using one resident kernel entry per prepared scan shape, not recompiling
  per point
- the benchmark measures the host-observed `core.run(...)` boundary, not a full
  decomposition of all time spent on host vs device internals

That is enough to inform design decisions, even though it is not a full profiler.

## Decisions To Hold Fixed

These are the architectural decisions we should now treat as settled unless real usage
shows they are wrong.

### Results API Language

For user-facing offline analysis, the right word is `series`, not `point data`.

Reason:

- users want to think in terms of “plot this against this”
- the same interface should work for parameter streams, pseudoparameters, result
  channels, point index, timestamps, and other saved per-point streams

Internally, the persisted schema still uses `points.*`, which is fine.

### Metadata Blob Boundary

This is now the intended split:

- `ndscan` is responsible for storing and recovering structured metadata blobs
- experiment code is responsible for saving enough raw data and metadata to make its
  own analysis reproducible
- lab/helper code is responsible for interpreting the blob semantics

In particular:

- ROI/image metadata may be transported and recovered through `ndscan`
- `ndscan` should not grow camera-specific or lab-specific interpretation logic for
  those blobs

### Use Before Expanding

The current `ndscan.results` API is good enough to use in real scripts.

We should not add broad new API layers until actual analysis scripts show that a common
pattern is genuinely missing.

The recent addition of richer `describe_series()` metadata and
`describe_plot_choices()` should reduce pressure to add more convenience wrappers for
basic plotting, because users can now recover labels/units/default axes without hiding
the actual NumPy data manipulation. These descriptions now come back as small
dataclasses rather than loose dicts, which should make repeated use in scripts less
error-prone.

The intended default-axis story is now:

- `describe_plot_choices()` for the full user-facing plotting view
- `choose_default_x_path()` as the lightweight shortcut
- `choose_default_x_source()` only for advanced schema-level introspection

## Not Fully Done Yet

These items are real but do not need large new architecture.

### ROI/Image Schema Framework

This item is only partly done:

- Create some kind of framework for relating the ROIs to the image consistently and in
  code to handle them per experiment

What is done:

- the generic blob transport and recovery path exists
- the three-image example shows how an experiment can persist ROI/image metadata

What is not done:

- a stable lab-side typed wrapper or convention for imaging metadata across many
  experiment types

This should mostly live outside `ndscan`.

### Streamlining the Simulation Workflow

This has improved, but there is not yet one small, obvious “recommended workflow”
document or helper layer.

That is now mostly a packaging/documentation problem, not a core runtime problem.

## Recommended Next ndscan Tasks

These are the next tasks worth doing on the `ndscan` side, in order.

### 1. Stabilise The Current Offline Path

Use the current offline API in a few real scripts and do not expand it yet.

Concrete rule:

- prefer small example scripts and lab-side helper code
- only add new `ndscan.results` helpers when the same friction appears repeatedly

This is the highest-value next step because it is the best way to find out which parts
of the current API are genuinely missing.

### 2. Write Down The Benchmark Meaning

Add a short benchmark note describing:

- what each benchmark column means
- what is measured inside `core.run(...)`
- what “outside” time means
- what confidence we have about single resident kernel entry for the tested shapes
- what the benchmark does not tell us

This should be short and practical. The goal is to avoid re-deriving the same
interpretation later.

### 3. Choose One User-Facing Live Viewer Improvement

The architecture is good enough that it makes sense to return to a direct user-facing
plotting feature.

Best candidates:

- asymmetric error bars in live plots
- suggestion of next point from the GUI plot
- improve the live viewer layout to be more space efficient

Recommended first pick:

- asymmetric error bars

Reason:

- it is concrete
- it improves scientific readability immediately
- it is smaller and less design-heavy than GUI-driven point suggestion

### 4. Lightly Consolidate The Benchmark / Example Files

The benchmarking work is useful now, but it should not sprawl.

Do:

- keep one benchmark script
- keep the nested TTL and nested fixed-wait examples
- avoid building a large benchmarking framework unless a second round of performance
  work really needs it

## Deferred / Outside ndscan For Now

These are either not `ndscan` tasks or should not be the next `ndscan` tasks:

- Test Fast SLM with linux PC
- Order SMA cables if needed for amplifiers
- Look at improving current photodiodes in setup w/ sampler feedback
- Push ARTIQ main VCD file generation fix with a pull request
- Modify Artiq master scheduler to do equal time sharing

These may still matter, but they are not the right next move for finishing the current
`ndscan` work.

## Lower-Priority ndscan Work

These items remain open, but they should wait until the current offline/runtime work
has been used a bit more:

- Allow suggestion of next point from the GUI plot
- Make live plotter understand binomial statistics in isolation
- Improve bayesian optimisation submission dashboard to expose more options
- Improve cornerplot/bayesian optimisation live viewer to have optional Gaussian
  process fit and use saved online artifacts where sensible
- Improve submission dashboard GUI for other scan types & repeats
- Improve looks of GUI live plot viewer to make it more space efficient
- Consider histogram live plotter
- Plot uncertainty range on plots from sensible fit
- Can reparameterisations happen on kernel

## Summary

The important point is that the big architectural items are now mostly done:

- prepared-runtime offline reading exists
- prepared-runtime runs can be reopened in the runtime viewer
- experiments can persist structured metadata blobs
- offline scripts can consume the data cleanly
- kernel-overhead benchmarking now exists and is informative

The next `ndscan` work should therefore be disciplined:

- use the current results API in real scripts
- document the benchmark meaning briefly
- implement one concrete viewer improvement
- avoid drifting back into broad architecture unless real usage forces it

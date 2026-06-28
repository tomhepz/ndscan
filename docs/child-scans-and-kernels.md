# Child Scans And Kernels

Child scans use the same prepared-runtime concepts as root scans. They differ mainly in
where they are invoked and how their scan-site path is derived.

## Child Scan Sites

A child scan writes a child scan site below its parent site. The child site records:

- `site.parent_path`,
- segment start indices,
- parent point indices,
- optional segment-level final analysis feedback.

This makes nested scans visible in the same result tree as root scans.

## Prepared Child Scan

`PreparedChildScan` is the fragment-facing child-scan handle. It prepares a child
fragment/request pair and provides an `acquire()` path for use inside parent fragments.

The child request inherits the parent scan context where appropriate, but the child has
its own site, point policy, writer, and analysis path.

## Kernel Streaming

The kernel-streaming executor exists to minimise kernel compilations:

1. Compile and enter one resident kernel region.
2. Request a batch of host-resolved point values through RPC.
3. Execute those points on the kernel side.
4. Return completed result buffers through RPC.
5. Resume the same kernel loop for the next batch.

The design intentionally avoids launching a new kernel from inside a host RPC that was
called by an existing kernel.

## Host Work At Batch Boundaries

Even in resident-kernel execution, the host owns:

- point-policy state,
- adaptive feedback,
- persistence,
- online analysis,
- preview savepoints,
- retry/restart decisions.

That is why the executor boundary is batch-oriented rather than fully autonomous on the
core device.

## What Is Intrinsically Complex

Some complexity in child scans is real ARTIQ complexity rather than accidental
abstraction:

- child fragments must be visible to the compiler,
- fixed output shapes/types matter for kernel calls,
- host setup may need to happen before the outer kernel is entered,
- transient kernel errors may require host context restart,
- host/device/kernel methods cannot be freely nested.

Refactoring this area should preserve those constraints explicitly rather than hiding
them behind a generic abstraction.

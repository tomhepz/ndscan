"""Host-runtime fixed-parameter traceability example.

This example is intentionally small and explicit. It exists to show where
``scan.fixed_parameters`` ends up for:

- a root scan site, and
- a nested child scan site launched with ``run_subscan(...)``.

The important detail is that both fragments use parameters that are *not* scanned for
that site, but are still used in ``run_once()``:

- the root fragment keeps ``multiplier`` fixed,
- the child fragment keeps ``gain`` and ``offset`` fixed.

After a run, inspect:

- ``ndscan.rid_<rid>.site.root.scan.fixed_parameters``
- ``ndscan.rid_<rid>.site.root.inner.scan.fixed_parameters``

The root entry should contain only the parent's fixed parameter. The child site should
contain only the child's fixed parameters.
"""

from __future__ import annotations

from ndscan.experiment import (
    ExpFragment,
    FloatChannel,
    FloatParam,
    ScanRequest,
    make_fragment_host_scan_exp,
    run_subscan,
)


class FixedLeafFragment(ExpFragment):
    """Leaf fragment with fixed parameters that still affect every point result."""

    def build_fragment(self):
        self.setattr_param("gain", FloatParam, "gain", 2.0)
        self.setattr_param("offset", FloatParam, "offset", 1.0)
        self.setattr_param("x", FloatParam, "x", 0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push(self.gain.get() * self.x.get() + self.offset.get())


class FixedParameterParentFragment(ExpFragment):
    """Root fragment whose nested child scan gets its own fixed-parameter metadata."""

    def build_fragment(self):
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_param("multiplier", FloatParam, "multiplier", 3.0)
        self.setattr_fragment("child", FixedLeafFragment, detached=True)
        self.setattr_result("total", FloatChannel)

    def run_once(self):
        child_result = run_subscan(
            self,
            self.child,
            ScanRequest.explicit(
                [self.child.x],
                [[self.outer.get()], [self.outer.get() + 1.0]],
            ),
            name="inner",
        )
        self.total.push(self.multiplier.get() * sum(child_result.values[self.child.y]))


HostRuntimeFixedParametersNested = make_fragment_host_scan_exp(
    FixedParameterParentFragment,
    lambda fragment: ScanRequest.explicit(
        [fragment.outer],
        [[10.0], [20.0]],
        metadata={"demo_name": "host_runtime_fixed_parameters_nested"},
    ),
)
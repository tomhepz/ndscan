"""Host-runtime 2D recursive-refinement grid example.

This example shows how the new point-policy layer can build a structured 2D scan
without going through the legacy one-generator-per-axis path.

Each axis uses a 1D recursive midpoint policy:

- emit the interval endpoints,
- then add midpoints breadth-first,
- stop after a fixed depth.

The two axes are then combined with ``ProductPointPolicy`` to form a rectangular 2D
grid whose density increases in a deterministic refinement order.
"""

from __future__ import annotations

from artiq.experiment import *
from ndscan.define import *
from ndscan.define import annotations
from ndscan.scan import *
from ndscan.runtime.api import *


class BowlSurfaceFragment(ExpFragment):
    """Return a simple smooth 2D surface over ``x`` and ``y``."""

    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_param("y", FloatParam, "y", default=0.0)
        self.setattr_result("z", FloatChannel)

    def run_once(self):
        x = self.x.get()
        y = self.y.get()
        self.z.push((x - 0.5) ** 2 + 0.5 * (y + 1.0) ** 2)


HostRuntimeRecursiveGrid2D = make_fragment_prepared_scan_exp(
    BowlSurfaceFragment,
    lambda fragment: ScanRequest(
        axes=(fragment.x, fragment.y),
        point_policy=ProductPointPolicy(
            [
                RecursiveMidpointPointPolicy1D(-2.0, 2.0, max_depth=2),
                RecursiveMidpointPointPolicy1D(-3.0, 1.0, max_depth=2),
            ]
        ),
        execution_policy=ExecutionPolicy(max_points_per_batch=8),
        metadata={"demo_name": "host_runtime_recursive_grid_2d"},
    ),
)

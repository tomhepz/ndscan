"""Explicit-point host-runtime scan.

This example demonstrates the more flexible point-selection model directly: the
fragment still sees ordinary parameters, but the scan request supplies a hand-written
list of full 2D points instead of expanding one generator per axis into a Cartesian
product.
"""

from __future__ import annotations

from artiq.experiment import *
from ndscan.define import *
from ndscan.define import annotations
from ndscan.scan import *
from ndscan.runtime.api import *


class ExplicitPointFragment(ExpFragment):
    """Return a simple planar response for arbitrary 2D points."""

    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_param("y", FloatParam, "y", default=0.0)
        self.setattr_result("z", FloatChannel)

    def run_once(self):
        x = self.x.get()
        y = self.y.get()
        self.z.push(x + 2.0 * y)


HostRuntimeExplicitPointScan = make_fragment_prepared_scan_exp(
    ExplicitPointFragment,
    lambda fragment: ScanRequest.explicit(
        [fragment.x, fragment.y],
        [
            (-2.0, 0.0),
            (-1.0, 1.0),
            (0.0, 1.5),
            (1.0, 1.0),
            (2.0, 0.0),
            (1.0, -1.0),
            (0.0, -1.5),
            (-1.0, -1.0),
        ],
        metadata={"demo_name": "host_runtime_explicit_points"},
    ),
)

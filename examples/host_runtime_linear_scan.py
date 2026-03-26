"""Simple 1D host-runtime scan.

This is the smallest useful example of the new host-only runtime:

- one root fragment,
- one scanned parameter,
- one saved result channel,
- one bare ``EnvExperiment`` export that ARTIQ can run directly.

The canonical data lands in the root scan site under
``ndscan.rid_<rid>.site.root.*``.
"""

from __future__ import annotations

from artiq.experiment import *
from ndscan.define import *
from ndscan.define import annotations
from ndscan.scan import *
from ndscan.runtime.api import *


class LinearResponseFragment(ExpFragment):
    """Return ``y = slope * x + offset`` for the current point."""

    def build_fragment(self):
        self.setattr_param("slope", FloatParam, "Slope", default=2.0)
        self.setattr_param("offset", FloatParam, "Offset", default=1.0)
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push(self.slope.get() * self.x.get() + self.offset.get())


HostRuntimeLinearScan = make_fragment_prepared_scan_exp(
    LinearResponseFragment,
    lambda fragment: ScanRequest.cartesian(
        [(fragment.x, [0.5 * i - 2.0 for i in range(11)])],
        metadata={"demo_name": "host_runtime_linear_scan"},
    ),
)

"""Zipped/tandem host-runtime scan.

This example shows the simplest non-Cartesian scan shape supported by the new runner:
two parameters scanned together point-by-point using ``ScanRequest.zipped(...)``.
"""

from __future__ import annotations

from artiq.experiment import *
from ndscan.define import *
from ndscan.define import annotations
from ndscan.scan import *
from ndscan.runtime.api import *


class TandemResponseFragment(ExpFragment):
    """Combine two scanned parameters into a pair of outputs."""

    def build_fragment(self):
        self.setattr_param("left", FloatParam, "Left", default=0.0)
        self.setattr_param("right", FloatParam, "Right", default=0.0)
        self.setattr_result("sum", FloatChannel)
        self.setattr_result("difference", FloatChannel)

    def run_once(self):
        left = self.left.get()
        right = self.right.get()
        self.sum.push(left + right)
        self.difference.push(left - right)


HostRuntimeZipScan = make_fragment_prepared_scan_exp(
    TandemResponseFragment,
    lambda fragment: ScanRequest.zipped(
        [
            (fragment.left, [-2.0, -1.0, 0.0, 1.0, 2.0]),
            (fragment.right, [0.5, 1.0, 1.5, 2.0, 2.5]),
        ],
        metadata={"demo_name": "host_runtime_zip_scan"},
    ),
)

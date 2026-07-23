"""Dashboard-driven prepared-runtime example for text rebinds.

Open this experiment in the ARTIQ dashboard and:

- leave ``logical_drive`` as a scanned row,
- switch ``drive`` to ``Rebind``,
- enter an expression such as ``logical_drive + 0.5``.

The dashboard then serialises that into ``scan_submission`` transport, and the worker
compiles the text expression into the same runtime ``ParameterMapping`` path used by
the code-first examples.
"""

from __future__ import annotations

from artiq.experiment import *

from ndscan.define import *
from ndscan.runtime.api import *
from ndscan.scan import *


class DashboardMappedDriveFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("logical_drive", FloatParam, "logical drive", 0.0)
        self.setattr_param("drive", FloatParam, "drive", 0.0)
        self.setattr_result("result", FloatChannel)

    def get_always_shown_params(self):
        return [self.logical_drive, self.drive]

    def run_once(self):
        self.result.push(2.0 * self.drive.get())


PreparedScanParameterMappingDashboard = make_fragment_prepared_dashboard_scan_exp(
    DashboardMappedDriveFragment
)

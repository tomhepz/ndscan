"""Dashboard-driven host-runtime zipped/tandem scan.

Open this in the dashboard, set both ``left`` and ``right`` to scan, and give them the
same non-empty scan group to zip them together point-by-point.
"""

from __future__ import annotations

from ndscan.experiment import make_fragment_prepared_dashboard_scan_exp

from host_runtime_zip_scan import TandemResponseFragment


class TandemResponseDashboardFragment(TandemResponseFragment):
    def get_always_shown_params(self):
        return super().get_always_shown_params() + [self.left, self.right]


HostRuntimeZipScanDashboard = make_fragment_prepared_dashboard_scan_exp(
    TandemResponseDashboardFragment
)

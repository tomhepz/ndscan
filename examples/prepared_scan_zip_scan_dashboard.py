"""Dashboard-driven prepared-runtime zipped/tandem scan.

Open this in the dashboard, set both ``left`` and ``right`` to scan, and give them the
same non-empty scan group to zip them together point-by-point.
"""

from __future__ import annotations

from prepared_scan_zip_scan import TandemResponseFragment

from ndscan.runtime.api import make_fragment_prepared_dashboard_scan_exp


class TandemResponseDashboardFragment(TandemResponseFragment):
    def get_always_shown_params(self):
        return super().get_always_shown_params() + [self.left, self.right]


PreparedScanZipScanDashboard = make_fragment_prepared_dashboard_scan_exp(
    TandemResponseDashboardFragment
)

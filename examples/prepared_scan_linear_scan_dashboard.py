"""Dashboard-driven prepared-runtime scan.

This is the counterpart to ``prepared_scan_linear_scan.py``:

- ``prepared_scan_linear_scan.py`` is code-first and already contains the full
  ``ScanRequest`` in Python,
- this file exposes the same fragment through the dashboard submission path so the
  dashboard can build ``scan_submission`` transport and the worker compiles that into a
  ``ScanRequest`` at run time.

The initial dashboard slice supports simple grid-mode parameter scans.  ``x`` is shown
by default so the experiment opens with something immediately editable.
"""

from __future__ import annotations

from ndscan.runtime.api import make_fragment_prepared_dashboard_scan_exp

from prepared_scan_linear_scan import LinearResponseFragment


class LinearResponseDashboardFragment(LinearResponseFragment):
    def get_always_shown_params(self):
        return super().get_always_shown_params() + [self.x]


PreparedScanLinearScanDashboard = make_fragment_prepared_dashboard_scan_exp(
    LinearResponseDashboardFragment
)

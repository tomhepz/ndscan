"""Dashboard-driven host-runtime scan.

This is the counterpart to ``host_runtime_linear_scan.py``:

- ``host_runtime_linear_scan.py`` is code-first and already contains the full
  ``ScanRequest`` in Python,
- this file exposes the same fragment through the dashboard submission path so the
  dashboard can build ``host_scan`` transport and the worker compiles that into a
  ``ScanRequest`` at run time.

The initial dashboard slice supports simple grid-mode parameter scans.  ``x`` is shown
by default so the experiment opens with something immediately editable.
"""

from __future__ import annotations

from ndscan.experiment import make_fragment_host_dashboard_scan_exp

from host_runtime_linear_scan import LinearResponseFragment


class LinearResponseDashboardFragment(LinearResponseFragment):
    def get_always_shown_params(self):
        return super().get_always_shown_params() + [self.x]


HostRuntimeLinearScanDashboard = make_fragment_host_dashboard_scan_exp(
    LinearResponseDashboardFragment
)

"""Public facade for the prepared runtime."""

from __future__ import annotations

import sys

from ..scan.mapping import ParameterMapping, ScanVariable
from ..submission.host_scan_schema import (
    HostScanSchemaError,
    HostScanSpec,
    compile_host_scan_schema,
    compile_host_scan_spec,
)
from .adapters import *
from .context import *
from .prepared import *
from .program import *
from . import adapters as _adapters
from . import context as _context
from . import prepared as _prepared
from . import program as _program

__all__ = [
    "ExecutionPolicy",
    "PreviewPolicy",
    "HostScanSchemaError",
    "HostScanSpec",
    "compile_host_scan_spec",
    "compile_host_scan_schema",
    "ScanVariable",
    "ParameterMapping",
    "ScanRequest",
    "PreparedScan",
    "ActiveScanContext",
    "current_scan_context",
    "make_child_scan_site",
    "BoundScanAxis",
    "BoundResultChannel",
    "PointObservation",
    "ScanOutputs",
    "ScanInspection",
    "PreparedChildScan",
    "prepare_scan",
    "prepare_child_scan",
    "setattr_prepared_child_scan",
    "make_fragment_prepared_scan_exp",
    "make_fragment_prepared_dashboard_scan_exp",
]

# Keep the base adapter classes available for docs only.
if "sphinx" in sys.modules:
    __all__.append("PreparedScanExperiment")
    __all__.append("PreparedDashboardScanExperiment")
    PreparedScanExperiment = _adapters.PreparedScanExperiment
    PreparedDashboardScanExperiment = _adapters.PreparedDashboardScanExperiment

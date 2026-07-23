"""Public facade for the prepared runtime."""

from __future__ import annotations

import sys

from ..scan.mapping import ParameterMapping, ScanVariable
from ..scan.request import ExecutionPolicy, PreviewPolicy, ScanRequest
from ..submission.scan_submission_schema import (
    ScanSubmissionSchemaError,
    ScanSubmissionSpec,
    compile_scan_submission_schema,
    compile_scan_submission_spec,
)
from . import adapters as _adapters
from .adapters import *
from .context import *
from .prepared import *
from .program import *

__all__ = [
    "ExecutionPolicy",
    "PreviewPolicy",
    "ScanSubmissionSchemaError",
    "ScanSubmissionSpec",
    "compile_scan_submission_spec",
    "compile_scan_submission_schema",
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

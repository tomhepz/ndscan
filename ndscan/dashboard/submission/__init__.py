"""Dashboard submission backend selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import DashboardSubmissionBackend, SubmittedOverride, SubmittedScanAxis
from .host import HostSubmissionBackend
from .legacy import (
    LegacyScanOptionsState,
    LegacySubmissionBackend,
    LegacySubmissionState,
)

__all__ = [
    "DashboardSubmissionBackend",
    "SubmittedOverride",
    "SubmittedScanAxis",
    "LegacyScanOptionsState",
    "LegacySubmissionState",
    "LegacySubmissionBackend",
    "HostSubmissionBackend",
    "select_submission_backend",
]


def select_submission_backend(params: Mapping[str, Any]) -> DashboardSubmissionBackend:
    """Select the dashboard submission backend from the stored transport payload."""

    if "host_scan" in params:
        return HostSubmissionBackend(params)
    return LegacySubmissionBackend()

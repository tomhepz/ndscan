"""Public helpers for reading and navigating prepared-runtime result files."""

from .scan_site_reader import (
    HostRuntimeSegmentFinalAnalysis,
    HostRuntimeSite,
    HostRuntimeSiteSegment,
    HostRuntimeSnapshot,
    read_host_runtime_snapshot,
)
from .series import (
    average_series,
    build_1d_errorbar_payload,
    build_1d_series_payload,
    format_series_selector,
    parse_series_selector,
    series_dict,
    series_for_selector,
    series_slices_along_axis,
    slice_series,
)

__all__ = [
    "HostRuntimeSegmentFinalAnalysis",
    "HostRuntimeSite",
    "HostRuntimeSiteSegment",
    "HostRuntimeSnapshot",
    "average_series",
    "build_1d_errorbar_payload",
    "build_1d_series_payload",
    "format_series_selector",
    "parse_series_selector",
    "series_dict",
    "series_for_selector",
    "series_slices_along_axis",
    "read_host_runtime_snapshot",
    "slice_series",
]

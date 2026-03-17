"""Legacy dashboard submission backend.

This module owns the current ``scan`` / ``overrides`` transport format used by the old
runtime path.  Keeping that serialization logic here means the rest of the dashboard
can treat "submission" as an abstract concern rather than directly mutating legacy
schema keys everywhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ...utils import NoAxesMode
from .common import DashboardSubmissionBackend, SubmittedOverride, SubmittedScanAxis

__all__ = [
    "LegacyScanOptionsState",
    "LegacySubmissionState",
    "LegacySubmissionBackend",
]


@dataclass(slots=True)
class LegacyScanOptionsState:
    """Global scan controls from the legacy dashboard UI."""

    num_repeats: int = 1
    num_repeats_per_point: int = 1
    no_axes_mode: str = NoAxesMode.single.name
    randomise_order_globally: bool = False
    skip_on_persistent_transitory_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_repeats": self.num_repeats,
            "num_repeats_per_point": self.num_repeats_per_point,
            "no_axes_mode": self.no_axes_mode,
            "randomise_order_globally": self.randomise_order_globally,
            "skip_on_persistent_transitory_error": (
                self.skip_on_persistent_transitory_error
            ),
        }


@dataclass(slots=True)
class LegacySubmissionState:
    """Mutable accumulation target for legacy dashboard widgets."""

    overrides: list[SubmittedOverride] = field(default_factory=list)
    axes: list[SubmittedScanAxis] = field(default_factory=list)
    scan_options: LegacyScanOptionsState | None = None

    def add_override(self, *, fqn: str, path: str, value: Any) -> None:
        self.overrides.append(SubmittedOverride(fqn=fqn, path=path, value=value))

    def add_scan_axis(
        self,
        *,
        fqn: str,
        path: str,
        axis_type: str,
        axis_range: Mapping[str, Any],
    ) -> None:
        self.axes.append(
            SubmittedScanAxis(
                fqn=fqn,
                path=path,
                type=axis_type,
                range=dict(axis_range),
            )
        )

    def set_scan_options(self, options: LegacyScanOptionsState) -> None:
        self.scan_options = options


class LegacySubmissionBackend(DashboardSubmissionBackend):
    """Adapter for the original dashboard submission format."""

    supports_editing = True

    def is_scannable(self, params: Mapping[str, Any]) -> bool:
        return "scan" in params

    def initial_scan_options_state(
        self, params: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        return params.get("scan", None)

    def iter_configured_entries(
        self, params: Mapping[str, Any]
    ) -> Iterable[tuple[str, str]]:
        for axis in params.get("scan", {}).get("axes", []):
            yield axis["fqn"], axis["path"]
        for fqn, overrides in params.get("overrides", {}).items():
            for override in overrides:
                yield fqn, override["path"]

    def new_submission_state(self) -> LegacySubmissionState:
        return LegacySubmissionState()

    def apply_submission_state(
        self,
        params: dict[str, Any],
        state: LegacySubmissionState,
    ) -> None:
        params["overrides"] = {}
        for override in state.overrides:
            params["overrides"].setdefault(override.fqn, []).append(
                {"path": override.path, "value": override.value}
            )

        if state.scan_options is None and not state.axes:
            params.pop("scan", None)
            return

        scan = {}
        if state.scan_options is not None:
            scan.update(state.scan_options.to_dict())
        scan["axes"] = [
            {
                "fqn": axis.fqn,
                "path": axis.path,
                "type": axis.type,
                "range": dict(axis.range),
            }
            for axis in state.axes
        ]
        params["scan"] = scan

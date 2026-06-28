"""Dashboard submission backends.

The dashboard should not know runtime semantics directly.  It should edit a
submission-shaped model and let a backend module serialize that model into the transport
dict carried through ARTIQ submission arguments.

The legacy ndscan scan schema and the new prepared-runtime schema are intentionally treated
as separate backends.  That keeps the old path contained here, so removing it later is
mostly a matter of deleting this package and its editor wiring rather than untangling
legacy conditionals from the runtime.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "SubmittedOverride",
    "SubmittedScanAxis",
    "DashboardSubmissionBackend",
]


@dataclass(frozen=True)
class SubmittedOverride:
    """One fixed parameter override chosen in the dashboard."""

    fqn: str
    path: str
    value: Any


@dataclass(frozen=True)
class SubmittedScanAxis:
    """One legacy-style scan axis produced by a dashboard row widget."""

    fqn: str
    path: str
    type: str
    range: dict[str, Any]


class DashboardSubmissionBackend(Protocol):
    """Backend-specific bridge between the editor and submission transport dicts."""

    supports_editing: bool

    def is_scannable(self, params: Mapping[str, Any]) -> bool:
        """Return whether the backend should expose scan controls in the editor."""

    def initial_scan_options_state(self, params: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Return the backend-specific scan-options payload, if any."""

    def iter_configured_entries(
        self, params: Mapping[str, Any]
    ) -> Iterable[tuple[str, str]]:
        """Yield ``(fqn, path)`` pairs already present in the submission state."""

    def new_submission_state(self):
        """Create an empty mutable submission state for this backend."""

    def apply_submission_state(self, params: dict[str, Any], state) -> None:
        """Write the editor state back into the mutable transport dict."""

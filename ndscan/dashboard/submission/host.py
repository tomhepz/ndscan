"""Host-runtime dashboard submission backend.

This backend is intentionally narrower than the worker-side ``HostScanSpec`` model.
It implements the first useful dashboard-editable subset:

- grid mode only,
- real fragment parameters only,
- row modes ``fixed``, finite ``scan``, and ``rebind``.

That is enough to give the host runtime a clean "dashboard submission path" distinct
from the code-first ``make_fragment_host_scan_exp()`` path, without pretending that the
full host schema already has a matching dashboard UI. More advanced host specs such as
GPO or pseudoparameters are rejected up front so the editor does not silently drop
information it cannot display.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ...experiment.host_scan_schema import (
    HostScanEntry,
    HostScanFixedModeSpec,
    HostScanGeneratorSpec,
    HostScanGridModeSpec,
    HostScanParamTargetSpec,
    HostScanRebindModeSpec,
    HostScanScanModeSpec,
    HostScanSchemaError,
    HostScanSpec,
)
from .common import DashboardSubmissionBackend

__all__ = ["HostSubmissionState", "HostSubmissionBackend"]


_EDITABLE_GENERATOR_TYPES = {"linear", "centre_span", "expanding", "list"}


@dataclass(slots=True)
class HostSubmissionState:
    """Mutable accumulation target for editable host-runtime dashboard rows."""

    entries: list[HostScanEntry] = field(default_factory=list)

    def add_override(self, *, fqn: str, path: str, value: Any) -> None:
        self.entries.append(
            HostScanEntry(
                id=_entry_id_for_target(fqn, path),
                kind="param",
                target=HostScanParamTargetSpec(fqn=fqn, path=path),
                mode=HostScanFixedModeSpec(value=value),
            )
        )

    def add_scan_axis(
        self,
        *,
        fqn: str,
        path: str,
        axis_type: str,
        axis_range: Mapping[str, Any],
        scan_group: str | None = None,
    ) -> None:
        if axis_type not in _EDITABLE_GENERATOR_TYPES:
            raise ValueError(
                f"Dashboard host backend does not support generator type {axis_type!r}"
            )
        self.entries.append(
            HostScanEntry(
                id=_entry_id_for_target(fqn, path),
                kind="param",
                target=HostScanParamTargetSpec(fqn=fqn, path=path),
                mode=HostScanScanModeSpec(
                    generator=HostScanGeneratorSpec(
                        type=axis_type,
                        range=dict(axis_range),
                    ),
                    group=scan_group,
                ),
            )
        )

    def add_rebind(self, *, fqn: str, path: str, expression: str) -> None:
        self.entries.append(
            HostScanEntry(
                id=_entry_id_for_target(fqn, path),
                kind="param",
                target=HostScanParamTargetSpec(fqn=fqn, path=path),
                mode=HostScanRebindModeSpec(expr=expression),
            )
        )


class HostSubmissionBackend(DashboardSubmissionBackend):
    """Adapter for the editable subset of ``host_scan`` transport payloads."""

    def __init__(self, params: Mapping[str, Any] | None = None):
        self._base_spec = _editable_host_spec(params or {})
        self.supports_editing = self._base_spec is not None

    def is_scannable(self, params: Mapping[str, Any]) -> bool:
        del params
        return True

    def initial_scan_options_state(
        self, params: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        del params
        return None

    def iter_configured_entries(
        self, params: Mapping[str, Any]
    ) -> Iterable[tuple[str, str]]:
        seen = set[tuple[str, str]]()
        spec = _load_host_spec(params)
        if spec is not None and _is_editable_host_spec(spec):
            for entry in spec.entries:
                assert entry.target is not None
                key = (entry.target.fqn, entry.target.path)
                seen.add(key)
                yield key
        for fqn, overrides in params.get("overrides", {}).items():
            for override in overrides:
                key = (fqn, override["path"])
                if key in seen:
                    continue
                seen.add(key)
                yield key

    def new_submission_state(self) -> HostSubmissionState:
        return HostSubmissionState()

    def apply_submission_state(
        self,
        params: dict[str, Any],
        state: HostSubmissionState,
    ) -> None:
        if self._base_spec is None:
            raise NotImplementedError(
                "Dashboard host-scan editing is only implemented for simple grid-mode "
                "parameter rows"
            )

        spec = HostScanSpec(
            version=self._base_spec.version,
            mode=HostScanGridModeSpec(),
            entries=tuple(_ensure_unique_entry_ids(state.entries)),
            execution=self._base_spec.execution,
            metadata=self._base_spec.metadata,
        )
        params["host_scan"] = spec.to_dict()
        params["overrides"] = {}
        params.pop("scan", None)

    def find_entry(
        self,
        params: Mapping[str, Any],
        *,
        fqn: str,
        path: str,
    ) -> HostScanEntry | None:
        spec = _load_host_spec(params)
        if spec is None or not _is_editable_host_spec(spec):
            return None
        for entry in spec.entries:
            if entry.target is None:
                continue
            if entry.target.fqn == fqn and entry.target.path == path:
                return entry
        return None

    def symbol_name_for_target(self, *, fqn: str, path: str) -> str:
        return _entry_id_for_target(fqn, path)


def _load_host_spec(params: Mapping[str, Any]) -> HostScanSpec | None:
    schema = params.get("host_scan", None)
    if schema is None:
        return None
    try:
        spec = HostScanSpec.from_dict(schema)
    except HostScanSchemaError:
        return None
    return spec


def _is_editable_host_spec(spec: HostScanSpec) -> bool:
    if not isinstance(spec.mode, HostScanGridModeSpec):
        return False
    for entry in spec.entries:
        if entry.kind != "param":
            return False
        if entry.target is None:
            return False
        if isinstance(entry.mode, HostScanFixedModeSpec):
            continue
        if isinstance(entry.mode, HostScanScanModeSpec):
            if entry.mode.generator.type not in _EDITABLE_GENERATOR_TYPES:
                return False
            continue
        if isinstance(entry.mode, HostScanRebindModeSpec):
            continue
        return False
    return True


def _editable_host_spec(params: Mapping[str, Any]) -> HostScanSpec | None:
    """Return the editable base spec for the dashboard host backend.

    There are two distinct failure modes when a dashboard opens a host-runtime
    experiment:

    - the stored ``host_scan`` payload is malformed or stale (e.g. cached from an old
      version of the experiment),
    - the stored payload is valid, but uses host features the current dashboard does
      not yet know how to edit.

    The first case should not brick the editor; we can safely fall back to a fresh
    empty grid spec and let the next save replace the stale transport data.  The
    second case *must* stay non-editable so we do not silently drop advanced semantics
    such as GPO or pseudoparameters.
    """

    spec = _load_host_spec(params)
    if spec is None:
        return HostScanSpec(mode=HostScanGridModeSpec())
    if _is_editable_host_spec(spec):
        return spec
    return None


def _entry_id_for_target(fqn: str, path: str) -> str:
    """Return a stable schema identifier for a real parameter row.

    These ids become the user-visible symbols for dashboard-entered rebind
    expressions, so prefer short names derived from the parameter name and path over a
    fully qualified identifier. Duplicates are still resolved deterministically by
    ``_ensure_unique_entry_ids()`` when the host transport dict is written back out.
    """

    pieces = []
    if path not in {"", "*"}:
        pieces.extend(part for part in re.split(r"[^0-9A-Za-z_]+", path) if part)
    pieces.append(fqn.split(".")[-1])
    text = "_".join(pieces)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    if not text or text[0].isdigit():
        text = "param_" + text
    return text


def _ensure_unique_entry_ids(entries: Iterable[HostScanEntry]) -> list[HostScanEntry]:
    """Deduplicate auto-generated ids while preserving deterministic order."""

    seen = dict[str, int]()
    result: list[HostScanEntry] = []
    for entry in entries:
        count = seen.get(entry.id, 0)
        seen[entry.id] = count + 1
        if count == 0:
            result.append(entry)
            continue
        result.append(
            HostScanEntry(
                id=f"{entry.id}_{count + 1}",
                kind=entry.kind,
                target=entry.target,
                mode=entry.mode,
                description=entry.description,
                default=entry.default,
            )
        )
    return result

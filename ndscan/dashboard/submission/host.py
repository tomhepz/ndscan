"""Host-runtime dashboard submission backend.

This backend is intentionally narrower than the worker-side ``HostScanSpec`` model.
It implements the first useful dashboard-editable subset:

- grid and GPO top-level modes,
- real fragment parameters and pseudoparams,
- grid row modes ``fixed``, finite ``scan``, and ``rebind``,
- GPO row modes ``fixed``, ``gpo_scan``, and ``rebind`` for real parameters,
- GPO row modes ``fixed`` and ``gpo_scan`` for pseudoparams.

That is enough to give the host runtime a clean "dashboard submission path" distinct
from the code-first ``make_fragment_host_scan_exp()`` path, without pretending that the
full host schema already has a matching dashboard UI. Specs outside this subset are
still rejected up front so the editor does not silently drop information it cannot
display.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ...experiment.host_scan_schema import (
    HostScanChannelObjectiveSpec,
    HostScanChannelTargetSpec,
    HostScanEntry,
    HostScanFixedModeSpec,
    HostScanGeneratorSpec,
    HostScanGpoModeSpec,
    HostScanGpoScanModeSpec,
    HostScanGridModeSpec,
    HostScanNuboBackendSpec,
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
class HostSubmissionGpoSettings:
    """Dashboard-side transport settings for the future GPO editor."""

    objective_channel_path: str
    batch_size: int | None = None
    initial_design_size: int = 1
    max_batches: int | None = None
    acquisition: str = "ucb"
    minimise: bool = True

    def to_mode_spec(self) -> HostScanGpoModeSpec:
        return HostScanGpoModeSpec(
            objective=HostScanChannelObjectiveSpec(
                target=HostScanChannelTargetSpec(path=self.objective_channel_path)
            ),
            backend=HostScanNuboBackendSpec(
                batch_size=self.batch_size,
                initial_design_size=self.initial_design_size,
                max_batches=self.max_batches,
                acquisition=self.acquisition,
                minimise=self.minimise,
            ),
        )


@dataclass(slots=True)
class HostSubmissionState:
    """Mutable accumulation target for editable host-runtime dashboard rows."""

    mode_type: str = "grid"
    gpo_settings: HostSubmissionGpoSettings | None = None
    entries: list[HostScanEntry] = field(default_factory=list)

    def set_grid_mode(self) -> None:
        self.mode_type = "grid"
        self.gpo_settings = None

    def set_gpo_mode(
        self,
        *,
        objective_channel_path: str,
        batch_size: int | None = None,
        initial_design_size: int = 1,
        max_batches: int | None = None,
        acquisition: str = "ucb",
        minimise: bool = True,
    ) -> None:
        self.mode_type = "gpo"
        self.gpo_settings = HostSubmissionGpoSettings(
            objective_channel_path=objective_channel_path,
            batch_size=batch_size,
            initial_design_size=initial_design_size,
            max_batches=max_batches,
            acquisition=acquisition,
            minimise=minimise,
        )

    def add_override(
        self, *, fqn: str, path: str, value: Any, entry_id: str | None = None
    ) -> None:
        self.entries.append(
            HostScanEntry(
                id=entry_id or _entry_id_for_target(fqn, path),
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
        entry_id: str | None = None,
        scan_group: str | None = None,
    ) -> None:
        if axis_type not in _EDITABLE_GENERATOR_TYPES:
            raise ValueError(
                f"Dashboard host backend does not support generator type {axis_type!r}"
            )
        self.entries.append(
            HostScanEntry(
                id=entry_id or _entry_id_for_target(fqn, path),
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

    def add_rebind(
        self,
        *,
        fqn: str,
        path: str,
        expression: str,
        entry_id: str | None = None,
    ) -> None:
        self.entries.append(
            HostScanEntry(
                id=entry_id or _entry_id_for_target(fqn, path),
                kind="param",
                target=HostScanParamTargetSpec(fqn=fqn, path=path),
                mode=HostScanRebindModeSpec(expr=expression),
            )
        )

    def add_gpo_axis(
        self,
        *,
        fqn: str,
        path: str,
        lower: float,
        upper: float,
        entry_id: str | None = None,
    ) -> None:
        self.entries.append(
            HostScanEntry(
                id=entry_id or _entry_id_for_target(fqn, path),
                kind="param",
                target=HostScanParamTargetSpec(fqn=fqn, path=path),
                mode=HostScanGpoScanModeSpec(lower=lower, upper=upper),
            )
        )

    def add_pseudoparam_fixed(self, *, entry_id: str, value: Any) -> None:
        self.entries.append(
            HostScanEntry(
                id=entry_id,
                kind="pseudoparam",
                mode=HostScanFixedModeSpec(value=value),
            )
        )

    def add_pseudoparam_scan(
        self,
        *,
        entry_id: str,
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
                id=entry_id,
                kind="pseudoparam",
                mode=HostScanScanModeSpec(
                    generator=HostScanGeneratorSpec(
                        type=axis_type,
                        range=dict(axis_range),
                    ),
                    group=scan_group,
                ),
            )
        )

    def add_pseudoparam_gpo_axis(
        self,
        *,
        entry_id: str,
        lower: float,
        upper: float,
    ) -> None:
        self.entries.append(
            HostScanEntry(
                id=entry_id,
                kind="pseudoparam",
                mode=HostScanGpoScanModeSpec(lower=lower, upper=upper),
            )
        )

    def mode_spec(self) -> HostScanGridModeSpec | HostScanGpoModeSpec:
        if self.mode_type == "grid":
            return HostScanGridModeSpec()
        if self.mode_type == "gpo":
            if self.gpo_settings is None:
                raise ValueError("GPO submission state requires gpo_settings")
            return self.gpo_settings.to_mode_spec()
        raise ValueError(f"Unsupported host submission mode {self.mode_type!r}")


class HostSubmissionBackend(DashboardSubmissionBackend):
    """Adapter for the editable subset of ``host_scan`` transport payloads."""

    def __init__(self, params: Mapping[str, Any] | None = None):
        params = params or {}
        self._params = params
        self._base_spec = _editable_host_spec(params)
        self._param_symbol_names = _build_param_symbol_name_map(params, self._base_spec)
        self.supports_editing = self._base_spec is not None

    def is_scannable(self, params: Mapping[str, Any]) -> bool:
        del params
        return True

    def initial_scan_options_state(
        self, params: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        del params
        return None

    def initial_mode_state(self) -> dict[str, Any]:
        if self._base_spec is None:
            return {"mode_type": "grid"}
        mode = self._base_spec.mode
        if isinstance(mode, HostScanGridModeSpec):
            return {"mode_type": "grid"}
        assert isinstance(mode, HostScanGpoModeSpec)
        return {
            "mode_type": "gpo",
            "objective_channel_path": mode.objective.target.path,
            "batch_size": mode.backend.batch_size,
            "initial_design_size": mode.backend.initial_design_size,
            "max_batches": mode.backend.max_batches,
            "acquisition": mode.backend.acquisition,
            "minimise": mode.backend.minimise,
        }

    def available_result_channels(self) -> tuple[dict[str, Any], ...]:
        channels = _params_channels(self._params)
        return tuple(channels[path] for path in sorted(channels))

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

    def iter_configured_pseudoparams(
        self, params: Mapping[str, Any]
    ) -> Iterable[HostScanEntry]:
        spec = _load_host_spec(params)
        if spec is None or not _is_editable_host_spec(spec):
            return ()
        return tuple(entry for entry in spec.entries if entry.kind == "pseudoparam")

    def new_submission_state(self) -> HostSubmissionState:
        state = HostSubmissionState()
        if self._base_spec is None:
            return state
        mode = self._base_spec.mode
        if isinstance(mode, HostScanGridModeSpec):
            state.set_grid_mode()
            return state
        assert isinstance(mode, HostScanGpoModeSpec)
        state.set_gpo_mode(
            objective_channel_path=mode.objective.target.path,
            batch_size=mode.backend.batch_size,
            initial_design_size=mode.backend.initial_design_size,
            max_batches=mode.backend.max_batches,
            acquisition=mode.backend.acquisition,
            minimise=mode.backend.minimise,
        )
        return state

    def apply_submission_state(
        self,
        params: dict[str, Any],
        state: HostSubmissionState,
    ) -> None:
        if self._base_spec is None:
            raise NotImplementedError(
                "Dashboard host-scan editing is only implemented for simple grid-mode "
                "parameter and pseudoparam rows"
            )

        spec = HostScanSpec(
            version=self._base_spec.version,
            mode=state.mode_spec(),
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
        return self._param_symbol_names.get((fqn, path), _entry_id_for_target(fqn, path))


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
    if not isinstance(spec.mode, (HostScanGridModeSpec, HostScanGpoModeSpec)):
        return False
    in_grid_mode = isinstance(spec.mode, HostScanGridModeSpec)
    for entry in spec.entries:
        if entry.kind == "param":
            if entry.target is None:
                return False
            if isinstance(entry.mode, HostScanFixedModeSpec):
                continue
            if isinstance(entry.mode, HostScanScanModeSpec):
                if not in_grid_mode:
                    return False
                if entry.mode.generator.type not in _EDITABLE_GENERATOR_TYPES:
                    return False
                continue
            if isinstance(entry.mode, HostScanGpoScanModeSpec):
                if in_grid_mode:
                    return False
                continue
            if isinstance(entry.mode, HostScanRebindModeSpec):
                continue
            return False
        if entry.kind == "pseudoparam":
            if entry.target is not None:
                return False
            if isinstance(entry.mode, HostScanFixedModeSpec):
                continue
            if isinstance(entry.mode, HostScanScanModeSpec):
                if not in_grid_mode:
                    return False
                if entry.mode.generator.type not in _EDITABLE_GENERATOR_TYPES:
                    return False
                continue
            if isinstance(entry.mode, HostScanGpoScanModeSpec):
                if in_grid_mode:
                    return False
                continue
            return False
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
    outside the currently supported dashboard subset.
    """

    spec = _load_host_spec(params)
    if spec is None:
        return HostScanSpec(mode=HostScanGridModeSpec())
    if _is_editable_host_spec(spec):
        return spec
    return None


    
def _params_channels(params: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    channels = params.get("channels", {})
    if not isinstance(channels, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for path, desc in channels.items():
        if not isinstance(path, str) or not isinstance(desc, Mapping):
            continue
        result[path] = dict(desc)
    return result


def _entry_id_for_target(fqn: str, path: str) -> str:
    """Return a stable schema identifier for a real parameter row.

    These ids become the user-visible symbols for dashboard-entered rebind
    expressions, so prefer short names derived from the parameter name and path over a
    fully qualified identifier. Duplicates are still resolved deterministically by
    ``_ensure_unique_entry_ids()`` when the host transport dict is written back out.
    """

    suffix = "all" if path == "*" else path
    text = f"{fqn}_{suffix}" if suffix else fqn
    return _normalise_symbol_identifier(text)


def _normalise_symbol_identifier(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    if not text or text[0].isdigit():
        text = "param_" + text
    return text


def _build_param_symbol_name_map(
    params: Mapping[str, Any],
    base_spec: HostScanSpec | None,
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}

    if base_spec is not None:
        for entry in base_spec.entries:
            if entry.kind != "param" or entry.target is None:
                continue
            result[(entry.target.fqn, entry.target.path)] = entry.id

    default_names = _default_param_symbol_names(params)
    for key, value in default_names.items():
        result.setdefault(key, value)
    return result


def _default_param_symbol_names(params: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    schemata = params.get("schemata", {})
    instances = params.get("instances", {})
    if not isinstance(schemata, Mapping) or not isinstance(instances, Mapping):
        return {}
    if not schemata:
        return {}

    occurrence_count = Counter[str]()
    targets: list[tuple[str, str]] = []
    for path, fqns in instances.items():
        if not isinstance(path, str):
            continue
        if not isinstance(fqns, Iterable):
            continue
        for fqn in fqns:
            if not isinstance(fqn, str):
                continue
            targets.append((fqn, path))
            occurrence_count[fqn] += 1
    for fqn, count in occurrence_count.items():
        if count > 1:
            targets.append((fqn, "*"))

    raw_names = _choose_minimal_unique_symbol_names(targets)
    return _dedupe_symbol_names(raw_names)


def _choose_minimal_unique_symbol_names(
    targets: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    targets = tuple(sorted(targets))
    token_lists = {target: _symbol_candidate_tokens(*target) for target in targets}
    widths = {target: 1 for target in targets}
    unresolved = set(targets)

    while unresolved:
        counts = Counter(
            tuple(token_lists[target][: widths[target]]) for target in unresolved
        )
        next_unresolved = set()
        progress = False
        for target in unresolved:
            key = tuple(token_lists[target][: widths[target]])
            if counts[key] == 1:
                continue
            if widths[target] < len(token_lists[target]):
                widths[target] += 1
                progress = True
            next_unresolved.add(target)
        if not next_unresolved or not progress:
            unresolved = next_unresolved
            break
        unresolved = next_unresolved

    return {
        target: _normalise_symbol_identifier(
            "_".join(reversed(token_lists[target][: widths[target]]))
        )
        for target in targets
    }


def _symbol_candidate_tokens(fqn: str, path: str) -> list[str]:
    fqn_parts = [_normalise_symbol_identifier(part) for part in fqn.split(".") if part]
    if fqn_parts:
        param_name = fqn_parts[-1]
        context = list(reversed(fqn_parts[:-1]))
    else:
        param_name = _normalise_symbol_identifier(fqn)
        context = []

    if path == "*":
        context.insert(0, "all")
    elif path:
        path_suffix = _normalise_symbol_identifier(path)
        if path_suffix:
            context.insert(0, path_suffix)

    tokens = [param_name]
    for token in context:
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _dedupe_symbol_names(
    names: Mapping[tuple[str, str], str]
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    counts: dict[str, int] = {}
    for key in sorted(names):
        base = names[key]
        count = counts.get(base, 0)
        counts[base] = count + 1
        if count == 0:
            result[key] = base
        else:
            result[key] = f"{base}_{count + 1}"
    return result


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

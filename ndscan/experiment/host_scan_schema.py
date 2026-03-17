"""Compilation of declarative host-scan schemas into runtime objects.

This module intentionally keeps submission-schema lowering separate from the host scan
execution loop. The runtime should execute an already-validated ``ScanRequest``; the
work of interpreting dashboard-style dictionaries belongs here.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .expression import ExpressionCompileError
from .fragment import ExpFragment
from .parameters import ParamHandle, ParamStore
from .point_policy import (
    AskTellOptimiserPointPolicy,
    ExplicitPointPolicy,
    PointPolicy,
    ProductPointPolicy,
    SinglePointPolicy,
    ZipPointPolicy,
)
from .result_channels import ResultChannel
from .scan_generator import GENERATORS
from .scan_mapping import FixedPseudoparam, ParameterMapping, ScanVariable
from .utils import path_matches_spec

if TYPE_CHECKING:
    from .host_runtime import ExecutionPolicy, ScanRequest

__all__ = [
    "HostScanSchemaError",
    "compile_host_scan_schema",
]


class HostScanSchemaError(ValueError):
    """Raised when a dict-based host scan schema cannot be compiled."""


def compile_host_scan_schema(
    fragment: ExpFragment,
    schema: Mapping[str, Any],
) -> tuple["ScanRequest", dict[str, list[tuple[str, ParamStore]]]]:
    """Compile a dict-based host scan schema to ``ScanRequest`` plus overrides.

    This is the bridge between the proposed dashboard submission schema and the
    code-first host runtime. The compiler intentionally supports a narrower subset than
    the full runtime:

    - top-level ``grid`` or ``gpo`` mode,
    - row modes ``fixed``, ``scan``, ``gpo_scan``, and ``rebind``,
    - zipped groups in grid mode via ``mode.group``,
    - fixed overrides for real parameters,
    - pseudoparameters as runtime-only scan variables.
    """

    if not isinstance(schema, Mapping):
        raise HostScanSchemaError("host scan schema must be a mapping")

    version = schema.get("version", 1)
    if version != 1:
        raise HostScanSchemaError(f"Unsupported host scan schema version: {version!r}")

    mode = _schema_mapping(schema.get("mode", None), "mode")
    mode_type = _schema_string(mode, "type")
    entries = _schema_sequence(schema.get("entries", ()), "entries")
    metadata = _optional_mapping(schema.get("metadata", {}), "metadata")
    execution_policy = _compile_execution_policy_from_schema(
        _optional_mapping(schema.get("execution", {}), "execution")
    )

    compiled_entries = _compile_schema_entries(fragment, entries)

    if mode_type == "grid":
        return _compile_grid_schema_request(
            compiled_entries,
            metadata=metadata,
            execution_policy=execution_policy,
        )
    if mode_type == "gpo":
        return _compile_gpo_schema_request(
            fragment,
            compiled_entries,
            mode,
            metadata=metadata,
            execution_policy=execution_policy,
        )
    raise HostScanSchemaError(f"Unsupported host scan mode: {mode_type!r}")


@dataclass(frozen=True)
class _CompiledSchemaEntry:
    entry_id: str
    kind: str
    mode_type: str
    source: ParamHandle | ScanVariable | None
    pathspec: str | None
    targets: tuple[ParamHandle, ...] = ()
    values: tuple[Any, ...] = ()
    bounds: tuple[float, float] | None = None
    group: str | None = None
    override: tuple[str, tuple[str, ParamStore]] | None = None
    parameter_mappings: tuple[ParameterMapping, ...] = ()
    description: str = ""
    expression: str | None = None
    defines_symbol: bool = False
    symbol_value: Any = None
    fixed_pseudoparam: FixedPseudoparam | None = None


def _compile_execution_policy_from_schema(
    execution: Mapping[str, Any],
) -> "ExecutionPolicy":
    if execution:
        unknown = set(execution.keys()) - {"max_points_per_batch"}
        if unknown:
            raise HostScanSchemaError(
                "Unsupported execution settings: " + ", ".join(sorted(unknown))
            )
    from .host_runtime import ExecutionPolicy

    max_points_per_batch = execution.get("max_points_per_batch", None)
    return ExecutionPolicy(max_points_per_batch=max_points_per_batch)


def _compile_schema_entries(
    fragment: ExpFragment,
    entries: Sequence[Any],
) -> list[_CompiledSchemaEntry]:
    compiled: list[_CompiledSchemaEntry] = []
    seen_ids = set[str]()
    claimed_targets = set[tuple[int, str]]()

    for index, raw_entry in enumerate(entries):
        entry = _schema_mapping(raw_entry, f"entries[{index}]")
        entry_id = _schema_string(entry, "id")
        if not entry_id.isidentifier():
            raise HostScanSchemaError(
                f"Entry id {entry_id!r} is not a valid Python identifier"
            )
        if entry_id in seen_ids:
            raise HostScanSchemaError(f"Duplicate host scan entry id: {entry_id!r}")
        seen_ids.add(entry_id)

        kind = _schema_string(entry, "kind")
        if kind not in {"param", "pseudoparam"}:
            raise HostScanSchemaError(
                f"entries[{index}].kind must be 'param' or 'pseudoparam', got {kind!r}"
            )

        mode = _schema_mapping(entry.get("mode", None), f"entries[{index}].mode")
        mode_type = _schema_string(mode, "type")

        description = str(entry.get("description", ""))
        default_value = entry.get("default", None)

        if kind == "param":
            target = _schema_mapping(entry.get("target", None), f"entries[{index}].target")
            fqn = _schema_string(target, "fqn")
            pathspec = _schema_string(target, "path")
            handles = _resolve_schema_param_handles(fragment, fqn, pathspec)
            if not handles:
                raise HostScanSchemaError(
                    f"No parameter handles matched target fqn={fqn!r}, path={pathspec!r}"
                )
            overlap = {(_mapping_target_key(handle)) for handle in handles} & claimed_targets
            if overlap:
                raise HostScanSchemaError(
                    f"Parameter target for entry {entry_id!r} overlaps an earlier entry"
                )
            claimed_targets.update(_mapping_target_key(handle) for handle in handles)

            compiled.append(
                _compile_param_entry(
                    entry_id,
                    handles,
                    pathspec,
                    mode,
                    mode_type,
                    description=description,
                )
            )
            continue

        compiled.append(
            _compile_pseudoparam_entry(
                entry_id,
                mode,
                mode_type,
                description=description,
                default_value=default_value,
            )
        )

    symbols = _collect_schema_symbols(compiled)
    return [
        _attach_rebind_mapping(entry, symbols)
        if entry.mode_type == "rebind"
        else entry
        for entry in compiled
    ]


def _compile_param_entry(
    entry_id: str,
    handles: Sequence[ParamHandle],
    pathspec: str,
    mode: Mapping[str, Any],
    mode_type: str,
    *,
    description: str,
) -> _CompiledSchemaEntry:
    first_handle = handles[0]
    if mode_type == "fixed":
        if "value" not in mode:
            raise HostScanSchemaError(f"Fixed entry {entry_id!r} is missing mode.value")
        value = mode["value"]
        store = first_handle.parameter.make_store(
            (first_handle.parameter.fqn, pathspec), value
        )
        return _CompiledSchemaEntry(
            entry_id=entry_id,
            kind="param",
            mode_type=mode_type,
            source=None,
            pathspec=pathspec,
            targets=tuple(handles),
            override=(first_handle.parameter.fqn, (pathspec, store)),
            description=description,
            defines_symbol=True,
            symbol_value=value,
        )

    if mode_type == "scan":
        values = _materialise_scan_mode_values(mode, entry_id)
        source, mappings = _compile_entry_axis_source(
            entry_id,
            handles,
            values,
            description=description or first_handle.parameter.description,
        )
        return _CompiledSchemaEntry(
            entry_id=entry_id,
            kind="param",
            mode_type=mode_type,
            source=source,
            pathspec=pathspec,
            targets=tuple(handles),
            values=values,
            group=_optional_string(mode.get("group", None), f"{entry_id}.mode.group"),
            parameter_mappings=mappings,
            description=description,
            defines_symbol=True,
            symbol_value=source,
        )

    if mode_type == "gpo_scan":
        lower, upper = _compile_gpo_bounds(mode, entry_id)
        source, mappings = _compile_entry_axis_source(
            entry_id,
            handles,
            (lower, upper),
            description=description or first_handle.parameter.description,
        )
        return _CompiledSchemaEntry(
            entry_id=entry_id,
            kind="param",
            mode_type=mode_type,
            source=source,
            pathspec=pathspec,
            targets=tuple(handles),
            bounds=(lower, upper),
            parameter_mappings=mappings,
            description=description,
            defines_symbol=True,
            symbol_value=source,
        )

    if mode_type == "rebind":
        return _CompiledSchemaEntry(
            entry_id=entry_id,
            kind="param",
            mode_type=mode_type,
            source=None,
            pathspec=pathspec,
            targets=tuple(handles),
            description=description,
            expression=_schema_string(mode, "expr"),
        )

    raise HostScanSchemaError(
        f"Unsupported row mode for parameter entry {entry_id!r}: {mode_type!r}"
    )


def _compile_pseudoparam_entry(
    entry_id: str,
    mode: Mapping[str, Any],
    mode_type: str,
    *,
    description: str,
    default_value: Any,
) -> _CompiledSchemaEntry:
    if mode_type == "fixed":
        if "value" not in mode:
            raise HostScanSchemaError(
                f"Fixed pseudoparam entry {entry_id!r} is missing mode.value"
            )
        value = mode["value"]
        variable = ScanVariable(
            name=entry_id,
            description=description,
            type=_infer_scan_variable_type(value if value is not None else default_value),
            spec={},
        )
        return _CompiledSchemaEntry(
            entry_id=entry_id,
            kind="pseudoparam",
            mode_type=mode_type,
            source=None,
            pathspec=None,
            description=description,
            defines_symbol=True,
            symbol_value=value,
            fixed_pseudoparam=FixedPseudoparam(variable=variable, value=value),
        )

    if mode_type == "scan":
        values = _materialise_scan_mode_values(mode, entry_id)
        variable = ScanVariable(
            name=entry_id,
            description=description,
            type=_infer_scan_variable_type(values[0] if values else default_value),
            spec=_describe_scan_variable_values(values),
        )
        return _CompiledSchemaEntry(
            entry_id=entry_id,
            kind="pseudoparam",
            mode_type=mode_type,
            source=variable,
            pathspec=None,
            values=values,
            group=_optional_string(mode.get("group", None), f"{entry_id}.mode.group"),
            description=description,
            defines_symbol=True,
            symbol_value=variable,
        )

    if mode_type == "gpo_scan":
        lower, upper = _compile_gpo_bounds(mode, entry_id)
        variable = ScanVariable(
            name=entry_id,
            description=description,
            type="float",
            spec={"min": lower, "max": upper},
        )
        return _CompiledSchemaEntry(
            entry_id=entry_id,
            kind="pseudoparam",
            mode_type=mode_type,
            source=variable,
            pathspec=None,
            bounds=(lower, upper),
            description=description,
            defines_symbol=True,
            symbol_value=variable,
        )

    if mode_type == "rebind":
        raise HostScanSchemaError(
            f"Pseudoparameter entry {entry_id!r} cannot use row mode 'rebind'"
        )

    raise HostScanSchemaError(
        f"Unsupported row mode for pseudoparam entry {entry_id!r}: {mode_type!r}"
    )


def _compile_entry_axis_source(
    entry_id: str,
    handles: Sequence[ParamHandle],
    values: Sequence[Any],
    *,
    description: str,
) -> tuple[ParamHandle | ScanVariable, tuple[ParameterMapping, ...]]:
    if len(handles) == 1:
        return handles[0], ()

    variable = ScanVariable(
        name=entry_id,
        description=description,
        type=_infer_scan_variable_type(values[0] if values else None),
        spec=_describe_scan_variable_values(values),
    )
    targets = tuple(handles)
    mapping = ParameterMapping(
        targets=targets,
        dependencies=(variable,),
        evaluate=lambda dependency_values, variable=variable, targets=targets: {
            target: dependency_values[variable] for target in targets
        },
        description=f"{entry_id} identity mapping",
    )
    return variable, (mapping,)


def _collect_schema_symbols(
    compiled_entries: Sequence[_CompiledSchemaEntry],
) -> dict[str, ParamHandle | ScanVariable | Any]:
    symbols: dict[str, ParamHandle | ScanVariable | Any] = {}
    for entry in compiled_entries:
        if entry.defines_symbol:
            symbols[entry.entry_id] = entry.symbol_value
    return symbols


def _attach_rebind_mapping(
    entry: _CompiledSchemaEntry,
    symbols: Mapping[str, ParamHandle | ScanVariable | Any],
) -> _CompiledSchemaEntry:
    assert entry.mode_type == "rebind"
    assert entry.expression is not None
    try:
        mapping = ParameterMapping.from_text(
            targets=entry.targets,
            expression=entry.expression,
            symbols=symbols,
            description=entry.description,
        )
    except (ExpressionCompileError, TypeError, ValueError) as exc:
        raise HostScanSchemaError(
            f"Invalid rebind expression for entry {entry.entry_id!r}: {exc}"
        ) from exc

    return _CompiledSchemaEntry(
        entry_id=entry.entry_id,
        kind=entry.kind,
        mode_type=entry.mode_type,
        source=entry.source,
        pathspec=entry.pathspec,
        targets=entry.targets,
        values=entry.values,
        bounds=entry.bounds,
        group=entry.group,
        override=entry.override,
        parameter_mappings=(mapping,),
        description=entry.description,
        expression=entry.expression,
    )


def _compile_grid_schema_request(
    compiled_entries: Sequence[_CompiledSchemaEntry],
    *,
    metadata: Mapping[str, Any],
    execution_policy: "ExecutionPolicy",
) -> tuple["ScanRequest", dict[str, list[tuple[str, ParamStore]]]]:
    from .host_runtime import ScanRequest

    overrides = _collect_compiled_overrides(compiled_entries)
    fixed_pseudoparams = tuple(
        entry.fixed_pseudoparam
        for entry in compiled_entries
        if entry.fixed_pseudoparam is not None
    )
    scanned_entries = [entry for entry in compiled_entries if entry.mode_type == "scan"]
    parameter_mappings = tuple(
        mapping
        for entry in compiled_entries
        for mapping in entry.parameter_mappings
    )
    if not scanned_entries:
        return (
            ScanRequest(
                axes=(),
                point_policy=SinglePointPolicy(),
                metadata=metadata,
                execution_policy=execution_policy,
                parameter_mappings=parameter_mappings,
                fixed_pseudoparams=fixed_pseudoparams,
            ),
            overrides,
        )

    groups: OrderedDict[str, list[_CompiledSchemaEntry]] = OrderedDict()
    for entry in scanned_entries:
        group_name = entry.group if entry.group else f"__singleton__:{entry.entry_id}"
        groups.setdefault(group_name, []).append(entry)

    axes = tuple(
        entry.source
        for group_entries in groups.values()
        for entry in group_entries
        if entry.source is not None
    )
    child_policies = [_compile_group_point_policy(group_entries) for group_entries in groups.values()]
    point_policy: PointPolicy
    if len(child_policies) == 1:
        point_policy = child_policies[0]
    else:
        point_policy = ProductPointPolicy(child_policies)

    return (
        ScanRequest(
            axes=axes,
            point_policy=point_policy,
            metadata=metadata,
            execution_policy=execution_policy,
            parameter_mappings=parameter_mappings,
            fixed_pseudoparams=fixed_pseudoparams,
        ),
        overrides,
    )


def _compile_group_point_policy(
    group_entries: Sequence[_CompiledSchemaEntry],
) -> PointPolicy:
    axis_values = [entry.values for entry in group_entries]
    lengths = {len(values) for values in axis_values}
    if len(lengths) == 1:
        return ZipPointPolicy(axis_values)
    return ExplicitPointPolicy(
        len(group_entries),
        list(zip(*axis_values)),
    )


def _compile_gpo_schema_request(
    fragment: ExpFragment,
    compiled_entries: Sequence[_CompiledSchemaEntry],
    mode: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    execution_policy: "ExecutionPolicy",
) -> tuple["ScanRequest", dict[str, list[tuple[str, ParamStore]]]]:
    from .host_runtime import ExecutionPolicy, ScanRequest

    overrides = _collect_compiled_overrides(compiled_entries)
    fixed_pseudoparams = tuple(
        entry.fixed_pseudoparam
        for entry in compiled_entries
        if entry.fixed_pseudoparam is not None
    )
    gpo_entries = [entry for entry in compiled_entries if entry.mode_type == "gpo_scan"]
    if not gpo_entries:
        raise HostScanSchemaError("GPO mode requires at least one gpo_scan entry")

    backend = _schema_mapping(mode.get("backend", None), "mode.backend")
    if _schema_string(backend, "kind") != "nubo":
        raise HostScanSchemaError(
            "Only backend.kind == 'nubo' is supported for dict-based GPO mode"
        )

    objective = _schema_mapping(mode.get("objective", None), "mode.objective")
    if _schema_string(objective, "kind") != "channel":
        raise HostScanSchemaError(
            "Only channel objectives are supported for dict-based GPO mode"
        )
    objective_target = _schema_mapping(objective.get("target", None), "mode.objective.target")
    objective_path = _schema_string(objective_target, "path")
    objective_channel_key = _resolve_saved_channel_key(fragment, objective_path)

    bounds = [[], []]
    for entry in gpo_entries:
        assert entry.bounds is not None
        bounds[0].append(entry.bounds[0])
        bounds[1].append(entry.bounds[1])

    try:
        from .optimisation import (
            NuboBatchBayesianOptimisationBackend,
            extract_scalar_channel_objective,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optional Bayesian optimisation dependencies are not installed"
        ) from exc

    batch_size = int(backend.get("batch_size", execution_policy.max_points_per_batch or 1))
    backend_instance = NuboBatchBayesianOptimisationBackend(
        bounds=bounds,
        batch_size=batch_size,
        initial_design_size=int(backend.get("initial_design_size", 1)),
        max_batches=backend.get("max_batches", None),
        acquisition_name=str(backend.get("acquisition", "ucb")),
        minimise=bool(backend.get("minimise", True)),
    )
    point_policy = AskTellOptimiserPointPolicy(
        backend_instance,
        extract_scalar_channel_objective(objective_channel_key),
    )
    parameter_mappings = tuple(
        mapping
        for entry in compiled_entries
        for mapping in entry.parameter_mappings
    )
    resolved_execution_policy = execution_policy
    if execution_policy.max_points_per_batch is None:
        resolved_execution_policy = ExecutionPolicy(max_points_per_batch=batch_size)
    elif execution_policy.max_points_per_batch == 1 and batch_size != 1:
        resolved_execution_policy = ExecutionPolicy(max_points_per_batch=batch_size)

    return (
        ScanRequest(
            axes=tuple(entry.source for entry in gpo_entries if entry.source is not None),
            point_policy=point_policy,
            metadata=metadata,
            execution_policy=resolved_execution_policy,
            parameter_mappings=parameter_mappings,
            fixed_pseudoparams=fixed_pseudoparams,
        ),
        overrides,
    )


def _collect_compiled_overrides(
    compiled_entries: Sequence[_CompiledSchemaEntry],
) -> dict[str, list[tuple[str, ParamStore]]]:
    overrides: dict[str, list[tuple[str, ParamStore]]] = {}
    for entry in compiled_entries:
        if entry.override is None:
            continue
        fqn, pair = entry.override
        overrides.setdefault(fqn, []).append(pair)
    return overrides


def _resolve_schema_param_handles(
    fragment: ExpFragment,
    fqn: str,
    pathspec: str,
) -> list[ParamHandle]:
    handles = []
    stack = [fragment]
    while stack:
        current = stack.pop(0)
        if path_matches_spec(current._fragment_path, pathspec):
            for name, param in current._free_params.items():
                if param.fqn == fqn:
                    handles.append(getattr(current, name))
        stack.extend(current._subfragments)
    return handles


def _resolve_saved_channel_key(fragment: ExpFragment, path: str) -> str:
    channel_dict = dict[str, ResultChannel]()
    fragment._collect_result_channels(channel_dict)
    saved_channels = [channel for channel in channel_dict.values() if channel.save_by_default]
    for index, channel in enumerate(saved_channels):
        if channel.path == path:
            return f"channel_{index}"
    raise HostScanSchemaError(f"No saved result channel matched path {path!r}")


def _materialise_scan_mode_values(mode: Mapping[str, Any], entry_id: str) -> tuple[Any, ...]:
    generator = _schema_mapping(mode.get("generator", None), "mode.generator")
    generator_type = _schema_string(generator, "type")
    generator_range = _schema_mapping(generator.get("range", None), "mode.generator.range")
    generator_class = GENERATORS.get(generator_type, None)
    if generator_class is None:
        raise HostScanSchemaError(
            f"Unsupported scan generator type for entry {entry_id!r}: {generator_type!r}"
        )

    if generator_type in {"refining", "centre_span_refining"}:
        raise NotImplementedError(
            f"Generator type {generator_type!r} is not yet supported in dict-based host scans"
        )

    generator_instance = generator_class(**dict(generator_range))
    rng = np.random.RandomState(random.getrandbits(32))
    if generator_type in {"linear", "centre_span", "list"}:
        return tuple(generator_instance.points_for_level(0, rng))
    if generator_type == "expanding":
        values = []
        level = 0
        while generator_instance.has_level(level):
            values.extend(generator_instance.points_for_level(level, rng))
            level += 1
            if level > 1_000_000:
                raise HostScanSchemaError(
                    f"Refusing to materialise more than 1,000,000 expanding levels for {entry_id!r}"
                )
        return tuple(values)

    raise HostScanSchemaError(
        f"Generator type {generator_type!r} is not yet supported in dict-based host scans"
    )


def _compile_gpo_bounds(mode: Mapping[str, Any], entry_id: str) -> tuple[float, float]:
    if "lower" not in mode or "upper" not in mode:
        raise HostScanSchemaError(
            f"GPO entry {entry_id!r} requires both mode.lower and mode.upper"
        )
    lower = float(mode["lower"])
    upper = float(mode["upper"])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise HostScanSchemaError(f"GPO bounds for entry {entry_id!r} must be finite")
    if lower >= upper:
        raise HostScanSchemaError(
            f"GPO bounds for entry {entry_id!r} must satisfy lower < upper"
        )
    return lower, upper


def _schema_mapping(obj: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise HostScanSchemaError(f"{name} must be a mapping")
    return obj


def _optional_mapping(obj: Any, name: str) -> Mapping[str, Any]:
    if obj is None:
        return {}
    return _schema_mapping(obj, name)


def _schema_sequence(obj: Any, name: str) -> Sequence[Any]:
    if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes, bytearray)):
        raise HostScanSchemaError(f"{name} must be a sequence")
    return obj


def _schema_string(obj: Mapping[str, Any], key: str) -> str:
    value = obj.get(key, None)
    if not isinstance(value, str):
        raise HostScanSchemaError(f"{key} must be a string")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HostScanSchemaError(f"{name} must be a string or null")
    return value


def _infer_scan_variable_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return "int"
    if isinstance(value, (float, np.floating)) or value is None:
        return "float"
    if isinstance(value, str):
        return "string"
    return "float"


def _describe_scan_variable_values(values: Sequence[Any]) -> dict[str, Any]:
    if not values:
        return {}
    first = values[0]
    if isinstance(first, bool):
        return {}
    if isinstance(first, (int, float, np.integer, np.floating)):
        return {"min": min(values), "max": max(values)}
    return {}


def _mapping_target_key(handle: ParamHandle) -> tuple[int, str]:
    return (id(handle.owner), handle.name)

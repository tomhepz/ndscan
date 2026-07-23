"""Bind scan requests to concrete fragment parameters and execution points.

The classes in :mod:`ndscan.runtime.program` describe the runtime objects. This module
does the work of making those objects: validating axes and parameter mappings,
installing temporary per-point stores, and resolving a logical point into the exact
fragment parameter values that the executors must run.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from ..define.fragment import ExpFragment
from ..define.parameters import ParamHandle
from ..define.utils import is_kernel
from ..scan.mapping import ParameterMapping, ScanVariable
from ..scan.point_policy import BasePoint
from ..scan.request import ScanRequest
from .program import (
    BoundScanAxis,
    BoundScanParameter,
    _BoundParameterMapping,
    _mapping_target_key,
    _ResolvedExecutionPoint,
    _TransientParamBinding,
)


def _axis_source_key(source: ParamHandle | ScanVariable) -> tuple[Any, ...]:
    if isinstance(source, ParamHandle):
        return ("param", id(source.owner), source.name)
    return ("variable", source.name)


def _build_bound_axes(
    axes: Sequence[ParamHandle | ScanVariable],
) -> list[BoundScanAxis]:
    """Assign stable runtime/storage keys to the requested logical axes."""

    bound_axes = []
    seen = set[tuple[Any, ...]]()
    next_pseudoparam_index = 0
    next_param_index = 0
    for index, source in enumerate(axes):
        key = _axis_source_key(source)
        if key in seen:
            if isinstance(source, ParamHandle):
                raise ValueError(
                    f"Scan axis '{source.owner._stringize_path()}/{source.name}' "
                    "was specified more than once"
                )
            raise ValueError(
                f"Scan variable '{source.name}' was specified more than once"
            )
        seen.add(key)

        if isinstance(source, ParamHandle):
            if source._store is None:
                raise ValueError(
                    f"Parameter handle '{source.name}' is not bound to a store yet; "
                    "initialise fragment parameters before building a scan submission"
                )
            bound_axes.append(
                BoundScanAxis(
                    source=source,
                    key=f"axis_{index}",
                    point_key=f"param_{next_param_index}",
                    path=source.owner._stringize_path(),
                    schema=source.parameter.describe(),
                    identity=source._store.identity,
                    param_store=source._store,
                )
            )
            next_param_index += 1
            continue

        bound_axes.append(
            BoundScanAxis(
                source=source,
                key=f"axis_{index}",
                point_key=f"pseudoparam_{next_pseudoparam_index}",
                path="",
                schema=source.describe(),
                identity=(source.fqn, ""),
                param_store=None,
            )
        )
        next_pseudoparam_index += 1
    return bound_axes


def _build_bound_parameters(
    axes: Sequence[BoundScanAxis],
    parameter_mappings: Sequence[_BoundParameterMapping],
) -> list[BoundScanParameter]:
    """Return the concrete fragment parameters that vary point-by-point."""

    parameters = []
    seen = set[tuple[int, str]]()
    next_index = 0

    for axis in axes:
        if not isinstance(axis.source, ParamHandle):
            continue
        key = _mapping_target_key(axis.source)
        if key in seen:
            continue
        seen.add(key)
        parameters.append(
            BoundScanParameter(
                handle=axis.source,
                key=f"param_{next_index}",
                path=axis.source.owner._stringize_path(),
                param_schema=axis.source.parameter.describe(),
                is_scanned=True,
                scan_role="direct",
            )
        )
        next_index += 1

    for mapping in parameter_mappings:
        for target in mapping.targets:
            key = _mapping_target_key(target)
            if key in seen:
                continue
            seen.add(key)
            parameters.append(
                BoundScanParameter(
                    handle=target,
                    key=f"param_{next_index}",
                    path=target.owner._stringize_path(),
                    param_schema=target.parameter.describe(),
                    is_scanned=False,
                    scan_role="derived",
                )
            )
            next_index += 1

    return parameters


def _collect_parameter_mappings(
    fragment: ExpFragment,
    request: ScanRequest,
    axes: Sequence[BoundScanAxis],
) -> list[_BoundParameterMapping]:
    """Validate and order parameter mappings for one fragment/request pair."""

    mappings = tuple(fragment._parameter_mappings) + tuple(request.parameter_mappings)
    if not mappings:
        return []

    axis_sources = {axis.source for axis in axes}
    scanned_param_targets = {
        _mapping_target_key(axis.source)
        for axis in axes
        if isinstance(axis.source, ParamHandle)
    }
    producers = {}
    for mapping in mappings:
        for target in mapping.targets:
            key = _mapping_target_key(target)
            if key in scanned_param_targets:
                raise ValueError(
                    f"Parameter '{target.owner._stringize_path()}/{target.name}' "
                    "cannot be both a direct scan axis and a mapping target"
                )
            if key in producers:
                raise ValueError(
                    f"Parameter '{target.owner._stringize_path()}/{target.name}' "
                    "is produced by more than one parameter mapping"
                )
            if target._store is None:
                raise ValueError(
                    "Cannot map unbound parameter "
                    f"'{target.owner._stringize_path()}/{target.name}'"
                )
            producers[key] = mapping

    for mapping in mappings:
        for dependency in mapping.dependencies:
            if isinstance(dependency, ScanVariable) and dependency not in axis_sources:
                raise ValueError(
                    "Scan variable dependency "
                    f"'{dependency.name}' is not present in the scan axes"
                )
            if isinstance(dependency, ParamHandle) and dependency._store is None:
                raise ValueError(
                    "Parameter dependency "
                    f"'{dependency.owner._stringize_path()}/{dependency.name}' "
                    "is not bound to a store"
                )

    ordered = _order_parameter_mappings(mappings)
    return [
        _BoundParameterMapping(
            mapping=mapping,
            targets=tuple(mapping.targets),
            dependencies=tuple(mapping.dependencies),
        )
        for mapping in ordered
    ]


def _order_parameter_mappings(
    mappings: Sequence[ParameterMapping],
) -> list[ParameterMapping]:
    """Order mappings so derived parameters are available to dependants."""

    producers = {}
    for mapping in mappings:
        for target in mapping.targets:
            producers[_mapping_target_key(target)] = mapping

    dependencies = {mapping: set() for mapping in mappings}
    for mapping in mappings:
        for dependency in mapping.dependencies:
            if not isinstance(dependency, ParamHandle):
                continue
            producer = producers.get(_mapping_target_key(dependency), None)
            if producer is not None and producer is not mapping:
                dependencies[mapping].add(producer)

    ordered = []
    ready = [mapping for mapping in mappings if not dependencies[mapping]]
    while ready:
        mapping = ready.pop(0)
        ordered.append(mapping)
        for other in mappings:
            if mapping in dependencies[other]:
                dependencies[other].remove(mapping)
                if (
                    not dependencies[other]
                    and other not in ordered
                    and other not in ready
                ):
                    ready.append(other)

    if len(ordered) != len(mappings):
        raise ValueError("Parameter mappings contain a dependency cycle")
    return ordered


def _fragment_uses_kernel_execution(fragment: ExpFragment) -> bool:
    return any(
        is_kernel(method)
        for method in (
            fragment.device_setup,
            fragment.run_once,
            fragment.device_cleanup,
        )
    )


def _missing_kernel_core_devices(fragment: ExpFragment) -> tuple[str, ...]:
    missing = set[str]()
    for method in (
        fragment.device_setup,
        fragment.run_once,
        fragment.device_cleanup,
    ):
        if not is_kernel(method):
            continue
        core_name = method.artiq_embedded.core_name
        if core_name is not None and not hasattr(fragment, core_name):
            missing.add(core_name)
    return tuple(sorted(missing))


def _can_use_kernel_streaming_executor(
    fragment: ExpFragment,
    axes: Sequence[BoundScanAxis],
    parameters: Sequence[BoundScanParameter],
) -> bool:
    del axes, parameters
    return is_kernel(fragment.run_once)


def _fragment_tree_needs_param_initialisation(fragment: ExpFragment) -> bool:
    for name in fragment._free_params.keys():
        if getattr(fragment, name)._store is None:
            return True
    for subfragment in fragment._subfragments:
        if _fragment_tree_needs_param_initialisation(subfragment):
            return True
    return False


def _collect_varying_parameter_handles(
    fragment: ExpFragment,
    request: ScanRequest,
) -> list[ParamHandle]:
    """Return parameter handles whose values vary point-by-point for this request.

    Directly scanned parameters and mapping targets are treated the same here. Both are
    concrete values chosen for one resolved execution point, so both need transient
    stores that are insulated from the fragment's default-value stores.
    """

    handles = []
    seen = set[tuple[int, str]]()

    def add(handle: ParamHandle) -> None:
        key = (id(handle.owner), handle.name)
        if key in seen:
            return
        seen.add(key)
        handles.append(handle)

    for axis in request.axes:
        if isinstance(axis, ParamHandle):
            add(axis)

    mappings = tuple(fragment._parameter_mappings) + tuple(request.parameter_mappings)
    for mapping in mappings:
        for target in mapping.targets:
            add(target)

    return handles


def _install_varying_parameter_stores(
    fragment: ExpFragment,
    request: ScanRequest,
) -> list[_TransientParamBinding]:
    """Temporarily isolate point-varying parameters from their default stores.

    Fragment parameter handles normally share stores according to defaults/overrides.
    During a scan, any parameter that changes point-by-point needs a fresh store so
    setting the next point does not mutate the default-value store that should be
    restored after execution.
    """

    bindings = []
    for handle in _collect_varying_parameter_handles(fragment, request):
        affected_handles = tuple(handle.owner._get_all_handles_for_param(handle.name))
        original_stores = tuple(
            bound_handle._store for bound_handle in affected_handles
        )
        if any(store is None for store in original_stores):
            raise ValueError(
                "Cannot vary parameter "
                f"'{handle.owner._stringize_path()}/{handle.name}' before its stores "
                "are initialised"
            )

        scan_store = type(handle._store)(handle._store.identity, handle.get())
        for bound_handle in affected_handles:
            bound_handle.set_store(scan_store)

        bindings.append(_TransientParamBinding(affected_handles, original_stores))
    return bindings


def _resolve_execution_point(
    point: BasePoint,
    axes: Sequence[BoundScanAxis],
    parameters: Sequence[BoundScanParameter],
    parameter_mappings: Sequence[_BoundParameterMapping],
) -> _ResolvedExecutionPoint:
    """Apply axes and mappings to produce the concrete state for one point."""

    axis_map = OrderedDict(
        (axis.key, value) for axis, value in zip(axes, point.axis_values, strict=True)
    )
    pseudoparam_map = OrderedDict()

    dependency_values = {}
    for axis, value in zip(axes, point.axis_values, strict=True):
        if axis.param_store is not None:
            axis.param_store.set_value(value)
            dependency_values[axis.source] = axis.param_store.get_value()
        else:
            pseudoparam_map[axis.point_key] = value
            dependency_values[axis.source] = value

    for mapping in parameter_mappings:
        for dependency in mapping.dependencies:
            if (
                isinstance(dependency, ParamHandle)
                and dependency not in dependency_values
            ):
                dependency_values[dependency] = dependency.get()

        updates = mapping.mapping.compute(dependency_values)
        for target, value in updates.items():
            if target._store is None:
                raise ValueError(
                    "Cannot apply parameter mapping to unbound parameter "
                    f"'{target.name}'"
                )
            target._store.set_value(value)
            dependency_values[target] = target.get()

    parameter_values = OrderedDict(
        (parameter.key, parameter.handle.get()) for parameter in parameters
    )
    rpc_parameter_values = tuple(
        parameter.handle._store.to_rpc_type(parameter_values[parameter.key])
        for parameter in parameters
    )
    return _ResolvedExecutionPoint(
        point=point,
        axis_values=axis_map,
        pseudoparam_values=pseudoparam_map,
        parameter_values=parameter_values,
        rpc_parameter_values=rpc_parameter_values,
    )


def _resolve_execution_batch(
    points: Sequence[BasePoint],
    axes: Sequence[BoundScanAxis],
    parameters: Sequence[BoundScanParameter],
    parameter_mappings: Sequence[_BoundParameterMapping],
) -> list[_ResolvedExecutionPoint]:
    return [
        _resolve_execution_point(point, axes, parameters, parameter_mappings)
        for point in points
    ]

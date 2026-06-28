"""Binding and program-layer types for the prepared runtime.

This module is the bridge between a user-facing :class:`ScanRequest` and the execution
backends in ``runtime.executors``. It owns the host-side meaning of a scan:

- which logical axes exist,
- which concrete fragment parameters are installed for each point,
- which result channels are collected,
- which metadata is written once per scan site.

The code here deliberately does not run fragments. It prepares plain Python objects
that both the host executor and resident-kernel executor can consume.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ..define.fragment import ExpFragment
from ..define.parameters import ParamHandle, ParamStore
from ..define.result_channels import ResultChannel
from ..define.utils import is_kernel
from ..scan.mapping import ParameterMapping, ScanVariable
from ..scan.point_policy import (
    BasePoint,
    BatchFeedback,
    PointPolicy,
)
from ..scan.request import ExecutionPolicy, ScanRequest
from .analysis import ScanAnalysisEngine
from .persistence import ScanSiteDatasetWriter

__all__ = [
    "BoundScanAxis",
    "BoundResultChannel",
    "PointObservation",
    "ScanOutputs",
    "ScanInspection",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanOutputs:
    """Stable named output surface for completed scans."""

    values: OrderedDict[str, Any]

    @classmethod
    def empty(cls) -> "ScanOutputs":
        return cls(OrderedDict())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ScanOutputs":
        return cls(OrderedDict(mapping.items()))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.values.keys())

    def __len__(self) -> int:
        return len(self.values)

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def as_tuple(self, *names: str) -> tuple[Any, ...]:
        selected = self.names if not names else names
        return tuple(self.values[name] for name in selected)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class BoundScanAxis:
    """Runtime-bound scan input metadata.

    A scan axis may be either a real fragment parameter or a logical ``ScanVariable``.
    The runtime treats both as axis coordinates, but persistence keeps them separate:
    real parameters become ``points.param_*`` streams, while logical variables become
    ``points.pseudoparam_*`` streams.
    """

    source: ParamHandle | ScanVariable
    key: str
    point_key: str
    path: str
    schema: dict[str, Any]
    identity: tuple[str, str]
    param_store: ParamStore | None = None

    def metadata(self) -> dict[str, Any]:
        if isinstance(self.source, ParamHandle):
            return {
                "path": self.path,
                "param": self.schema,
            }
        return {
            "path": self.path,
            "variable": self.schema,
        }


@dataclass(frozen=True)
class BoundScanParameter:
    """Runtime-bound actual fragment parameter recorded point-by-point.

    This is always a concrete ``ParamHandle``. Direct scan axes and mapping targets both
    appear here because both produce installed fragment parameter values that matter for
    offline reconstruction.
    """

    handle: ParamHandle
    key: str
    path: str
    param_schema: dict[str, Any]
    is_scanned: bool
    scan_role: str

    @property
    def identity(self) -> tuple[str, str]:
        return self.handle._store.identity

    def metadata(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "param": self.param_schema,
            "is_scanned": self.is_scanned,
            "scan_role": self.scan_role,
        }


@dataclass(frozen=True)
class _BoundParameterMapping:
    """Validated parameter mapping ready for point-by-point execution."""

    mapping: ParameterMapping
    targets: tuple[ParamHandle, ...]
    dependencies: tuple[ParamHandle | ScanVariable, ...]


@dataclass(frozen=True)
class BoundResultChannel:
    """Runtime-bound result channel metadata."""

    channel: ResultChannel
    key: str


@dataclass(frozen=True)
class PointObservation:
    """Result of one successfully completed point."""

    point_index: int
    axis_values: OrderedDict[str, Any]
    channel_values: OrderedDict[str, Any]
    pseudoparam_values: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    parameter_values: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    point_metadata: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    acquired_at_unix: float | None = None


@dataclass(frozen=True)
class _ResolvedExecutionPoint:
    """Concrete point installation plan shared by host and kernel executors.

    Point policies only choose logical axis values. Resolving a point applies direct
    parameter axes and parameter mappings, producing the actual parameter values that
    must be installed before ``device_setup()`` / ``run_once()`` execute.
    """

    point: BasePoint
    axis_values: OrderedDict[str, Any]
    pseudoparam_values: OrderedDict[str, Any]
    parameter_values: OrderedDict[str, Any]
    rpc_parameter_values: tuple[Any, ...]


@dataclass
class _TransientParamBinding:
    """Temporary rebinding of one per-point-varying parameter onto a fresh store."""

    handles: tuple[ParamHandle, ...]
    original_stores: tuple[ParamStore, ...]

    def restore(self) -> None:
        for handle, store in zip(self.handles, self.original_stores, strict=True):
            handle.set_store(store)


@dataclass
class ScanRuntimeStats:
    """Lightweight runtime counters/timings for executor bring-up and profiling."""

    batch_count: int = 0
    point_count: int = 0
    executor_entry_count: int = 0
    first_executor_entry_elapsed_s: float | None = None
    total_executor_elapsed_s: float = 0.0
    total_batch_finalize_elapsed_s: float = 0.0


@dataclass
class ScanInspection:
    """Host-only inspection artifact for one completed prepared scan execution."""

    coordinates: OrderedDict[str, list[Any]]
    parameters: OrderedDict[str, list[Any]]
    values: dict[ResultChannel, list[Any]]
    analysis_results: dict[str, Any]
    analysis_artifacts: dict[str, Any]
    online_analysis_results: dict[str, dict[str, Any]]
    online_analysis_artifacts: dict[str, dict[str, Any]]
    annotations: list[dict[str, Any]]
    online_analysis_annotations: dict[str, list[dict[str, Any]]]
    runtime_stats: ScanRuntimeStats
    point_metadata: OrderedDict[str, list[Any]]
    site_prefix: str

    @classmethod
    def empty(
        cls,
        axes: Sequence[BoundScanAxis],
        parameters: Sequence[BoundScanParameter],
        channels: Sequence[BoundResultChannel],
        site_prefix: str,
        *,
        initial_annotations: Sequence[dict[str, Any]] = (),
    ) -> "ScanInspection":
        return cls(
            coordinates=OrderedDict((axis.identity, []) for axis in axes),
            parameters=OrderedDict((parameter.identity, []) for parameter in parameters),
            values={binding.channel: [] for binding in channels},
            analysis_results={},
            analysis_artifacts={},
            online_analysis_results={},
            online_analysis_artifacts={},
            annotations=list(initial_annotations),
            online_analysis_annotations={},
            runtime_stats=ScanRuntimeStats(),
            point_metadata=OrderedDict(),
            site_prefix=site_prefix,
        )

    def record(
        self,
        observation: PointObservation,
        axes: Sequence[BoundScanAxis],
        parameters: Sequence[BoundScanParameter],
        channels: Sequence[BoundResultChannel],
    ) -> None:
        for axis in axes:
            if axis.param_store is not None:
                self.coordinates[axis.identity].append(
                    observation.axis_values[axis.key]
                )
            else:
                self.coordinates[axis.identity].append(
                    observation.pseudoparam_values[axis.point_key]
                )

        for parameter in parameters:
            self.parameters[parameter.identity].append(
                observation.parameter_values[parameter.key]
            )

        for binding in channels:
            self.values[binding.channel].append(observation.channel_values[binding.key])

        if observation.acquired_at_unix is not None:
            self.point_metadata.setdefault("point.acquired_at_unix", []).append(
                observation.acquired_at_unix
            )
        for key, value in observation.point_metadata.items():
            self.point_metadata.setdefault(key, []).append(value)

    def record_batch(
        self,
        observations: Sequence[PointObservation],
        axes: Sequence[BoundScanAxis],
        parameters: Sequence[BoundScanParameter],
        channels: Sequence[BoundResultChannel],
    ) -> None:
        for observation in observations:
            self.record(observation, axes, parameters, channels)

    def outputs(self) -> ScanOutputs:
        return ScanOutputs.from_mapping(self.analysis_results)


class _PointBatchSource:
    """Host-side adapter around a point policy.

    The runtime asks this object for concrete batches and feeds observations back after
    each completed batch. This keeps adaptive scan logic outside the fragment executor:
    the executor only sees "run these points", not "decide where to go next".
    """

    def __init__(self, point_policy: PointPolicy, execution_policy: ExecutionPolicy):
        self._point_policy = point_policy
        self._execution_policy = execution_policy

    @property
    def point_policy(self) -> PointPolicy:
        return self._point_policy

    def describe(self) -> dict[str, Any]:
        return self._point_policy.describe()

    def effective_batch_size(self) -> int:
        request_limit = self._execution_policy.max_points_per_batch
        if request_limit is None:
            request_limit = 1

        preferred = self._point_policy.preferred_batch_size(request_limit)
        if preferred <= 0:
            raise ValueError("preferred_batch_size() must return a positive integer")
        return min(request_limit, preferred)

    def next_batch(self):
        requested_size = self.effective_batch_size()
        batch = self._point_policy.next_batch(requested_size)
        if batch:
            return batch
        if self._point_policy.is_finished():
            return []
        raise RuntimeError(
            f"{type(self._point_policy).__name__} returned no points before finishing"
        )

    def has_more_work(self) -> bool:
        return not self._point_policy.is_finished()

    def observe_batch(self, feedback: BatchFeedback) -> None:
        self._point_policy.observe_batch(feedback)

    def kernel_batch_source(self):
        return None


class _AnalysisAdapter:
    """Host-side adapter around the analysis engine.

    Batch publication calls analysis through this small interface so host execution,
    kernel streaming, and prepared child scans all use the same finalisation path.
    """

    def __init__(self, engine: ScanAnalysisEngine):
        self._engine = engine

    @property
    def engine(self) -> ScanAnalysisEngine:
        return self._engine

    def metadata(self) -> dict[str, Any]:
        return self._engine.metadata()

    def initial_annotations(self) -> list[dict[str, Any]]:
        return self._engine.initial_annotations()

    def observe_batch(
        self,
        completed_batch: Sequence[PointObservation],
        result: ScanInspection,
        site_writer: ScanSiteDatasetWriter,
    ):
        return self._engine.observe_batch(completed_batch, result, site_writer)

    def execute_final(
        self, result: ScanInspection, site_writer: ScanSiteDatasetWriter
    ) -> None:
        self._engine.execute_final(result, site_writer)

    def kernel_reducer(self):
        return None


class _ObservationTransport:
    """Host-side persistence/preview adapter for completed observations.

    The runtime calls this with semantic events such as "append observations" and
    "finish segment". The dataset writer below this layer owns the exact scan-site
    dataset keys.
    """

    def __init__(self, site_writer: ScanSiteDatasetWriter):
        self._site_writer = site_writer

    @property
    def site_writer(self) -> ScanSiteDatasetWriter:
        return self._site_writer

    @property
    def prefix(self) -> str:
        return self._site_writer.prefix

    @property
    def next_point_index(self) -> int:
        return self._site_writer.next_point_index

    def register_preview(self, preview) -> None:
        if preview is not None:
            preview.register_writer(self._site_writer)

    def unregister_preview(self, preview) -> None:
        if preview is not None:
            preview.unregister_writer(self._site_writer)

    def publish_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        extra_metadata: Mapping[str, Any],
        start_unix_time: float,
    ) -> None:
        self._site_writer.publish_metadata(
            metadata,
            extra_metadata=extra_metadata,
            start_unix_time=start_unix_time,
        )

    def start_segment(
        self, *, parent_point_index: int | None, start_unix_time: float
    ) -> None:
        self._site_writer.start_segment(
            parent_point_index=parent_point_index,
            start_unix_time=start_unix_time,
        )

    def append_observations(self, observations: Sequence[PointObservation]) -> None:
        self._site_writer.append_observations(observations)

    def flush(self) -> None:
        self._site_writer.flush()

    def finish_segment(self) -> None:
        self._site_writer.finish_segment()

    def set_completed(self, completed: bool) -> None:
        self._site_writer.set_completed(completed)

    def close(self) -> None:
        self._site_writer.close()

    def maybe_write_preview(self, preview) -> None:
        if preview is not None:
            preview.maybe_write_preview()

    def write_completion_preview(self, preview) -> None:
        if preview is not None:
            preview.write_completion_preview()

    def kernel_transport(self):
        return None


def _make_batch_feedback(
    observations: Sequence[PointObservation],
    result: ScanInspection,
    axes: Sequence[BoundScanAxis],
    parameters: Sequence[BoundScanParameter],
    channels: Sequence[BoundResultChannel],
    online_analyses: dict[str, Any],
) -> BatchFeedback:
    return BatchFeedback(
        observations=tuple(observations),
        axis_data={
            axis.source: tuple(result.coordinates[axis.identity]) for axis in axes
        },
        parameter_data={
            parameter.handle: tuple(result.parameters[parameter.identity])
            for parameter in parameters
        },
        result_data={
            binding.channel: tuple(result.values[binding.channel]) for binding in channels
        },
        online_analyses=online_analyses,
    )


def _publish_completed_batch(
    completed_batch: Sequence[PointObservation],
    result: ScanInspection,
    *,
    axes: Sequence[BoundScanAxis],
    parameters: Sequence[BoundScanParameter],
    channels: Sequence[BoundResultChannel],
    point_source: _PointBatchSource,
    analysis: _AnalysisAdapter,
    transport: _ObservationTransport,
    preview,
) -> None:
    if not completed_batch:
        return

    # This is the one place a completed batch becomes visible to the rest of ndscan:
    # persist raw point streams, update the in-memory inspection object, run online
    # analysis, feed the result back to the point policy, then write any preview.
    # Keeping this order stable matters for adaptive scans: policies observe the same
    # accumulated state that readers and plots will see after the batch is flushed.
    started_at = time.perf_counter()
    transport.append_observations(completed_batch)
    result.record_batch(completed_batch, axes, parameters, channels)
    online_analyses = analysis.observe_batch(
        completed_batch,
        result,
        transport.site_writer,
    )
    point_source.observe_batch(
        _make_batch_feedback(
            completed_batch,
            result,
            axes,
            parameters,
            channels,
            online_analyses,
        )
    )
    transport.flush()
    transport.maybe_write_preview(preview)
    result.runtime_stats.batch_count += 1
    result.runtime_stats.point_count += len(completed_batch)
    result.runtime_stats.total_batch_finalize_elapsed_s += (
        time.perf_counter() - started_at
    )


class ScanProgram:
    """Validated, fragment-bound scan submission plan.

    ``ScanProgram`` is the prepared runtime's "compiled" host object. It is built once
    for a request/site and then consumed by an execution backend. It contains no ARTIQ
    kernel code itself, which keeps the binding/metadata logic testable without running
    the core device.
    """

    def __init__(
        self,
        fragment: ExpFragment,
        request: ScanRequest,
        axes: Sequence[BoundScanAxis],
        parameters: Sequence[BoundScanParameter],
        channels: Sequence[BoundResultChannel],
        parameter_mappings: Sequence[_BoundParameterMapping],
        site_writer: ScanSiteDatasetWriter,
        analysis_engine: ScanAnalysisEngine,
    ):
        self.fragment = fragment
        self.request = request
        self.axes = tuple(axes)
        self.parameters = tuple(parameters)
        self.channels = tuple(channels)
        self.parameter_mappings = tuple(parameter_mappings)
        self.point_source = _PointBatchSource(
            request.point_policy, request.execution_policy
        )
        self.analysis = _AnalysisAdapter(analysis_engine)
        self.transport = _ObservationTransport(site_writer)

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "site.fragment_fqn": self.fragment.fqn,
            "scan.point_policy": self.point_source.describe(),
            "scan.axes": [
                _axis_metadata(index, axis)
                for index, axis in enumerate(self.axes)
            ],
            "scan.parameters": {
                parameter.key: parameter.metadata()
                for parameter in self.parameters
            },
            "scan.pseudoparams": {
                axis.point_key: axis.metadata()
                for axis in self.axes
                if isinstance(axis.source, ScanVariable)
            },
            "scan.fixed_pseudoparams": {
                pseudoparam.name: {
                    "variable": pseudoparam.variable.describe(),
                    "value": _parameter_value_for_metadata(pseudoparam.value),
                }
                for pseudoparam in self.request.fixed_pseudoparams
            },
            "scan.fixed_parameters": _collect_fixed_parameter_metadata(
                self.fragment,
                self.parameters,
            ),
            "scan.channels": {
                binding.key: binding.channel.describe()
                for binding in self.channels
            },
        }
        if self.parameter_mappings:
            metadata["scan.parameter_mappings"] = {
                f"mapping_{index}": mapping.mapping.describe(
                    {axis.source: axis.point_key for axis in self.axes}
                )
                for index, mapping in enumerate(self.parameter_mappings)
            }
        metadata.update(self.analysis.metadata())
        return metadata


def _parameter_value_for_metadata(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _join_metadata_path(base: str, name: str) -> str:
    base = base.strip("/")
    name = name.strip("/")
    if base and name:
        return base + "/" + name
    if name:
        return name
    return base


def _axis_schema_display_metadata(
    *,
    kind: str,
    axis: BoundScanAxis,
) -> dict[str, Any]:
    if kind == "parameter":
        spec = axis.schema.get("spec", {})
        name = str(axis.schema.get("fqn", axis.point_key)).rsplit(".", 1)[-1]
        path = _join_metadata_path(axis.path, name)
        description = axis.schema.get("description")
        value_type = axis.schema.get("type", "float")
    else:
        spec = axis.schema.get("spec", {})
        name = str(axis.schema.get("name", axis.point_key))
        path = _join_metadata_path(axis.path, name)
        description = axis.schema.get("description")
        value_type = axis.schema.get("type", "float")

    unit = spec.get("unit") if isinstance(spec, Mapping) else None
    scale = spec.get("scale") if isinstance(spec, Mapping) else None
    return {
        "path": path,
        "description": description,
        "unit": unit,
        "scale": scale,
        "type": value_type,
    }


def _axis_metadata(index: int, axis: BoundScanAxis) -> dict[str, Any]:
    kind = "parameter" if isinstance(axis.source, ParamHandle) else "pseudoparam"
    return {
        "index": index,
        "axis_key": axis.key,
        "storage_key": axis.point_key,
        "kind": kind,
        **_axis_schema_display_metadata(kind=kind, axis=axis),
    }


def _collect_fixed_parameter_metadata(
    fragment: ExpFragment,
    varying_parameters: Sequence[BoundScanParameter],
) -> dict[str, Any]:
    varying_keys = {
        _mapping_target_key(parameter.handle) for parameter in varying_parameters
    }
    fixed_parameters = []

    def walk(current: ExpFragment) -> None:
        # Detached subfragments own their own scan sites. Including their parameters in
        # the parent site's fixed-parameter metadata would make nested scans look as if
        # their child parameters belonged to the parent point stream.
        for name, param in current._free_params.items():
            handle = getattr(current, name)
            key = _mapping_target_key(handle)
            if key in varying_keys or handle._store is None:
                continue
            fixed_parameters.append(
                {
                    "path": current._stringize_path(),
                    "param": param.describe(),
                    "value": _parameter_value_for_metadata(handle._store.get_value()),
                }
            )
        for child in current._subfragments:
            if child in current._detached_subfragments:
                continue
            walk(child)

    walk(fragment)
    return {
        f"fixed_param_{index}": entry for index, entry in enumerate(fixed_parameters)
    }


def _axis_source_key(source: ParamHandle | ScanVariable) -> tuple[Any, ...]:
    if isinstance(source, ParamHandle):
        return ("param", id(source.owner), source.name)
    return ("variable", source.name)


def _mapping_target_key(handle: ParamHandle) -> tuple[int, str]:
    return (id(handle.owner), handle.name)


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
                    f"Cannot map unbound parameter '{target.owner._stringize_path()}/{target.name}'"
                )
            producers[key] = mapping

    for mapping in mappings:
        for dependency in mapping.dependencies:
            if isinstance(dependency, ScanVariable) and dependency not in axis_sources:
                raise ValueError(
                    f"Scan variable dependency '{dependency.name}' is not present in the scan axes"
                )
            if isinstance(dependency, ParamHandle) and dependency._store is None:
                raise ValueError(
                    f"Parameter dependency '{dependency.owner._stringize_path()}/{dependency.name}' "
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
    """Topologically order mappings so derived parameters are available to dependants."""

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
    if not _fragment_uses_kernel_execution(fragment):
        return False
    if not axes:
        return False
    return bool(parameters)


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

    for mapping in tuple(fragment._parameter_mappings) + tuple(request.parameter_mappings):
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
        original_stores = tuple(bound_handle._store for bound_handle in affected_handles)
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
                    f"Cannot apply parameter mapping to unbound parameter '{target.name}'"
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


def _apply_resolved_parameter_values(
    parameters: Sequence[BoundScanParameter],
    point: _ResolvedExecutionPoint,
) -> None:
    for parameter in parameters:
        parameter.handle._store.set_value(point.parameter_values[parameter.key])

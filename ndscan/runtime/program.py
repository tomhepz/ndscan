"""Program-layer data model and batch publication for the prepared runtime.

This module owns the host-side objects produced by binding a user-facing
:class:`ScanRequest` to a concrete fragment tree:

- which logical axes exist,
- which concrete fragment parameters are installed for each point,
- which result channels are collected,
- which metadata is written once per scan site.

The code here deliberately does not run fragments and does not decide point values.
``runtime.binding`` resolves point values; ``runtime.runner`` and ``runtime.executors``
consume the program objects defined here.
"""

from __future__ import annotations

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

    When reading the runtime loop, think of this object as the bundle of everything the
    runner needs to know but should not rediscover every batch: the fragment to call,
    the bound scan axes, the concrete varying parameters, result channels, point source,
    analysis engine, and persistence transport.
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


def _mapping_target_key(handle: ParamHandle) -> tuple[int, str]:
    return (id(handle.owner), handle.name)

"""A minimal parallel host-only runtime for ndscan fragments.

This module is intentionally narrower than the legacy runtime:

- the scan infrastructure runs on the host,
- code-first requests rather than dashboard parsing,
- one flat scan-site dataset layout,
- point sources as the only point-selection abstraction.

The goal is not to replace the old runtime in one shot. The goal is to establish a
small execution core that is easy to read, extend, and eventually reuse from both
top-level scans and subscans.

Fragments are still free to call ``@kernel`` helpers from host methods. What makes this
runtime "host-only" is that the *scan loop* itself is driven from the host rather than
by a dedicated kernel-side runner.

Direct ``@kernel`` point bodies remain a separate concern. The loop structure here is
host-side, but the per-point result collection contract still expects values to be
visible to the host collector by the time a point returns.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import reduce
from typing import Any

from artiq.coredevice.exceptions import RTIOUnderflow
from artiq.language import EnvExperiment, HasEnvironment, kernel, portable

from ..utils import merge_no_duplicates
from .annotations import AnnotationContext
from .default_analysis import AnalysisFeedback
from .fragment import ExpFragment, RestartKernelTransitoryError, TransitoryError
from .parameters import ParamHandle, ParamStore
from .point_source import (
    BatchFeedback,
    CartesianPointSource,
    ExplicitPointSource,
    PointSource,
    SinglePointSource,
    ZipPointSource,
)
from .result_channels import LastValueSink, ResultChannel, SingleUseSink
from .scan_mapping import ParameterMapping, ScanVariable
from .scan_runner import describe_analyses, filter_default_analyses
from .scan_site import ScanSite, ScanSiteDatasetWriter
from .utils import is_kernel

__all__ = [
    "ExecutionPolicy",
    "ScanVariable",
    "ParameterMapping",
    "ScanRequest",
    "ActiveScanContext",
    "current_scan_context",
    "make_child_scan_site",
    "BoundScanAxis",
    "BoundResultChannel",
    "PointObservation",
    "HostScanRunResult",
    "HostScanSession",
    "run_host_scan",
    "run_subscan",
    "HostScanExperiment",
    "make_fragment_host_scan_exp",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveScanContext:
    """Execution context for the point currently being run.

    This is what lets nested scans feel like regular recursive runtime use rather than
    a second, separate subscan framework. A child scan can inspect the current context
    to derive its scan-site path and record which parent point a segment belongs to.
    """

    site_path: tuple[str, ...]
    point_index: int


_active_scan_context: ContextVar[tuple[ActiveScanContext, ...]] = ContextVar(
    "_active_scan_context", default=()
)


def current_scan_context() -> ActiveScanContext | None:
    """Return the currently executing parent scan context, if any."""

    stack = _active_scan_context.get()
    return stack[-1] if stack else None


@contextmanager
def _push_scan_context(context: ActiveScanContext):
    stack = _active_scan_context.get()
    token = _active_scan_context.set(stack + (context,))
    try:
        yield
    finally:
        _active_scan_context.reset(token)


def make_child_scan_site(
    name: str,
    *,
    segmented: bool = True,
    extra_metadata: Mapping[str, Any] | None = None,
) -> ScanSite:
    """Create a child scan site nested under the current executing point.

    This helper is the new runtime's first subscan composition primitive. It keeps the
    recursive shape explicit:

    - the parent point currently being executed determines where the child site lives,
    - the child scan still runs through the same ``HostScanSession`` code path as a
      root scan,
    - the resulting child site records both its own path and its parent-site metadata.

    A child scan site only makes sense while a parent scan point is being executed, so
    calling this helper outside an active scan context is an error.
    """

    parent = current_scan_context()
    if parent is None:
        raise RuntimeError(
            "make_child_scan_site() can only be used while a parent scan point is active"
        )

    return ScanSite(
        path=parent.site_path + (name,),
        parent_path=parent.site_path,
        segmented=segmented,
        extra_metadata={} if extra_metadata is None else extra_metadata,
    )


@dataclass(frozen=True)
class ExecutionPolicy:
    """Host-runtime scheduling and flush policy for one scan request.

    The current host runtime only needs one execution knob: the maximum number of
    points to execute before closing a batch boundary. The point policy can still ask
    for a smaller natural batch size, but the execution policy owns the hard cap.

    Keeping these controls out of ``ScanRequest`` leaves room for more runtime-only
    settings later without turning the request itself into a mixed scan-and-scheduler
    object.
    """

    max_points_per_batch: int | None = None

    def __post_init__(self) -> None:
        if self.max_points_per_batch is not None and self.max_points_per_batch <= 0:
            raise ValueError("max_points_per_batch must be positive when specified")


@dataclass(frozen=True)
class ScanRequest:
    """User-facing host-runtime scan request.

    The first implementation is deliberately code-first: callers provide either real
    ``ParamHandle`` scan axes or logical ``ScanVariable`` axes, together with a
    ``PointSource`` describing the point strategy. A future dashboard adapter can
    resolve selector syntax or text formulas into the same objects without changing
    the runtime core again.

    Runtime scheduling choices live in ``execution_policy`` rather than in the request
    itself. That keeps the request focused on the scan shape while still letting the
    runner batch, flush, and pause at well-defined boundaries.
    """

    axes: tuple[ParamHandle | ScanVariable, ...]
    point_source: PointSource
    site: ScanSite = field(default_factory=ScanSite)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    parameter_mappings: tuple[ParameterMapping, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.execution_policy, ExecutionPolicy):
            raise TypeError("execution_policy must be an ExecutionPolicy instance")
        for mapping in self.parameter_mappings:
            if not isinstance(mapping, ParameterMapping):
                raise TypeError(
                    "parameter_mappings must contain ParameterMapping instances"
                )

    def with_site(self, site: ScanSite) -> "ScanRequest":
        """Return this request with a different scan-site placement.

        ``ScanRequest`` is immutable, but nested scans often want to reuse the same
        point strategy while changing only where the resulting data is published.
        Keeping this as a tiny value-level transformation makes the higher-level
        nested helper read naturally without introducing a second request type.
        """

        return ScanRequest(
            axes=self.axes,
            point_source=self.point_source,
            site=site,
            metadata=self.metadata,
            execution_policy=self.execution_policy,
            parameter_mappings=self.parameter_mappings,
        )

    def with_parameter_mappings(
        self, parameter_mappings: Sequence[ParameterMapping]
    ) -> "ScanRequest":
        """Return this request with extra parameter mappings appended.

        This is the code-first counterpart of the future GUI formula path: ad hoc
        reparameterisations stay attached to the request rather than having to mutate
        the fragment tree.
        """

        return ScanRequest(
            axes=self.axes,
            point_source=self.point_source,
            site=self.site,
            metadata=self.metadata,
            execution_policy=self.execution_policy,
            parameter_mappings=self.parameter_mappings + tuple(parameter_mappings),
        )

    @classmethod
    def single(
        cls,
        *,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
        execution_policy: ExecutionPolicy | None = None,
    ) -> "ScanRequest":
        return cls(
            axes=(),
            point_source=SinglePointSource(),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
            execution_policy=ExecutionPolicy() if execution_policy is None else execution_policy,
        )

    @classmethod
    def cartesian(
        cls,
        axes: Sequence[tuple[ParamHandle | ScanVariable, Sequence[Any]]],
        *,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
        execution_policy: ExecutionPolicy | None = None,
    ) -> "ScanRequest":
        return cls(
            axes=tuple(handle for handle, _ in axes),
            point_source=CartesianPointSource([values for _, values in axes]),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
            execution_policy=ExecutionPolicy() if execution_policy is None else execution_policy,
        )

    @classmethod
    def zipped(
        cls,
        axes: Sequence[tuple[ParamHandle | ScanVariable, Sequence[Any]]],
        *,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
        execution_policy: ExecutionPolicy | None = None,
    ) -> "ScanRequest":
        return cls(
            axes=tuple(handle for handle, _ in axes),
            point_source=ZipPointSource([values for _, values in axes]),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
            execution_policy=ExecutionPolicy() if execution_policy is None else execution_policy,
        )

    @classmethod
    def explicit(
        cls,
        axes: Sequence[ParamHandle | ScanVariable],
        points: Sequence[Sequence[Any]],
        *,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
        execution_policy: ExecutionPolicy | None = None,
    ) -> "ScanRequest":
        return cls(
            axes=tuple(axes),
            point_source=ExplicitPointSource(len(axes), points),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
            execution_policy=ExecutionPolicy() if execution_policy is None else execution_policy,
        )


@dataclass(frozen=True)
class BoundScanAxis:
    """Runtime-bound scan input metadata.

    ``key`` remains the internal point-policy position key (``axis_<n>``), while
    ``point_key`` is the persisted scan-site key. For direct fragment-parameter scan
    inputs that becomes ``param_<n>``; for runtime-only logical scan variables it
    becomes ``pseudoparam_<n>``.
    """

    source: ParamHandle | ScanVariable
    key: str
    point_key: str
    path: str
    schema: dict[str, Any]
    identity: tuple[str, str]
    param_store: ParamStore | None = None

    def metadata(self) -> dict[str, Any]:
        """Return serialisable metadata for this concrete bound axis."""

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
    """Runtime-bound actual fragment parameter recorded point-by-point."""

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
    """Result of one successfully completed point.

    ``acquired_at_unix`` is recorded on the host once the point body has returned and
    all point results have been collected successfully.

    ``axis_values`` are kept as the internal point-policy view of the executed point,
    keyed by positional ``axis_<n>`` names. Persisted scan-site data is split more
    explicitly into ``pseudoparam_values`` for logical runtime-only scan variables and
    ``parameter_values`` for actual fragment parameters whose installed values varied
    for this point.
    """

    point_index: int
    axis_values: OrderedDict[str, Any]
    channel_values: OrderedDict[str, Any]
    pseudoparam_values: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    parameter_values: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    acquired_at_unix: float | None = None


@dataclass
class HostScanRunResult:
    """In-memory copy of the data produced by a host-runtime scan.

    Keeping a small in-memory mirror of the flat site is useful for tests and for the
    first analysis hooks. The canonical on-disk/broadcast representation remains the
    scan-site datasets written by ``ScanSiteDatasetWriter``.
    """

    coordinates: OrderedDict[tuple[str, str], list[Any]]
    parameters: OrderedDict[tuple[str, str], list[Any]]
    values: dict[ResultChannel, list[Any]]
    online_analysis_results: dict[str, Any]
    online_analysis_annotations: dict[str, list[dict[str, Any]]]
    analysis_results: dict[str, Any]
    annotations: list[dict[str, Any]]
    site_prefix: str

    @classmethod
    def empty(
        cls,
        axes: Sequence[BoundScanAxis],
        parameters: Sequence[BoundScanParameter],
        channels: Sequence[BoundResultChannel],
        site_prefix: str,
        initial_annotations: Sequence[dict[str, Any]] = (),
    ) -> "HostScanRunResult":
        return cls(
            coordinates=OrderedDict(
                (axis.identity, []) for axis in axes
            ),
            parameters=OrderedDict((param.identity, []) for param in parameters),
            values={binding.channel: [] for binding in channels},
            online_analysis_results={},
            online_analysis_annotations={},
            analysis_results={},
            annotations=list(initial_annotations),
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
            self.coordinates[axis.identity].append(observation.axis_values[axis.key])
        for param in parameters:
            self.parameters[param.identity].append(observation.parameter_values[param.key])
        for channel in channels:
            self.values[channel.channel].append(observation.channel_values[channel.key])

    def record_batch(
        self,
        observations: Sequence[PointObservation],
        axes: Sequence[BoundScanAxis],
        parameters: Sequence[BoundScanParameter],
        channels: Sequence[BoundResultChannel],
    ) -> None:
        """Record a completed batch of observations into the in-memory mirror."""
        for observation in observations:
            self.record(observation, axes, parameters, channels)


class _TemporaryAnalysisResultSinks:
    """Temporarily bind analysis result channels to in-memory last-value sinks.

    Default analyses are declared in terms of ordinary ``ResultChannel`` instances.
    Running them through temporary ``LastValueSink`` objects keeps the execution step
    separate from dataset publication: the analysis writes to channels exactly as it
    would in the legacy runtime, and the host runtime decides afterwards which values
    should be persisted to the scan site.
    """

    def __init__(self, channels: Mapping[str, ResultChannel]):
        self._channels = dict(channels)
        self._original_sinks = dict[ResultChannel, Any]()
        self._temporary_sinks = dict[str, LastValueSink]()

    def __enter__(self) -> dict[str, LastValueSink]:
        for name, channel in self._channels.items():
            self._original_sinks[channel] = channel.sink
            sink = LastValueSink()
            channel.set_sink(sink)
            self._temporary_sinks[name] = sink
        return self._temporary_sinks

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for channel, original_sink in self._original_sinks.items():
            channel.set_sink(original_sink)
        self._original_sinks.clear()
        self._temporary_sinks.clear()


class _HostScanAnalysisPlan:
    """Selected default analyses for one concrete host-runtime scan program.

    The fragment-side analysis API already splits naturally into two phases:

    - declaration/description, which determines metadata before the scan starts,
    - execution, which consumes the completed run result after the last point.

    This helper keeps those phases together without mixing them into the point
    execution loop itself.
    """

    def __init__(
        self,
        analyses,
        analysis_results: Mapping[str, ResultChannel],
        annotation_context: AnnotationContext,
    ):
        self._analyses = tuple(analyses)
        self._analysis_results = dict(analysis_results)
        self._annotation_context = annotation_context
        self._metadata = describe_analyses(self._analyses, self._annotation_context)
        self._metadata["analysis_results"] = {
            name: channel.describe() for name, channel in self._analysis_results.items()
        }

    @classmethod
    def build(
        cls,
        fragment: ExpFragment,
        axes: Sequence[BoundScanAxis],
        channels: Sequence[BoundResultChannel],
    ) -> "_HostScanAnalysisPlan":
        analysable_axes = [axis for axis in axes if axis.param_store is not None]
        analyses = filter_default_analyses(fragment, analysable_axes)

        axis_keys = {
            axis.param_store.identity: axis.point_key
            for axis in axes
            if axis.param_store is not None
        }
        # AnnotationContext expects bare channel names and adds the "channel_" prefix
        # itself when serialising coordinate references.
        channel_names = {
            binding.channel: binding.key.removeprefix("channel_") for binding in channels
        }
        analysis_results = reduce(
            lambda x, y: merge_no_duplicates(x, y, kind="analysis result"),
            (analysis.get_analysis_results() for analysis in analyses),
            {},
        )
        exported_analysis_channels = set(analysis_results.values())

        context = AnnotationContext(
            lambda handle: axis_keys[handle._store.identity],
            lambda channel: channel_names[channel],
            lambda channel: channel in exported_analysis_channels,
        )
        return cls(analyses, analysis_results, context)

    def metadata(self) -> dict[str, Any]:
        metadata = {}
        if self._metadata["annotations"]:
            metadata["analysis.annotations"] = list(self._metadata["annotations"])
        if self._metadata["online_analyses"]:
            metadata["analysis.online"] = dict(self._metadata["online_analyses"])
        if self._metadata["analysis_results"]:
            metadata["analysis.outputs"] = dict(self._metadata["analysis_results"])
        return metadata

    def initial_annotations(self) -> list[dict[str, Any]]:
        return list(self._metadata["annotations"])

    def execute(
        self,
        run_result: HostScanRunResult,
        site_writer: ScanSiteDatasetWriter,
    ) -> None:
        if not self._analyses:
            return

        axis_data = dict(run_result.coordinates)
        result_data = dict(run_result.values)

        with _TemporaryAnalysisResultSinks(self._analysis_results) as sinks:
            annotations = []
            for analysis in self._analyses:
                annotations.extend(
                    analysis.execute(axis_data, result_data, self._annotation_context)
                )

            analysis_results = {
                name: sink.get_last() for name, sink in sinks.items()
            }
        feedback = AnalysisFeedback(
            outputs=analysis_results,
            annotations=annotations,
        )

        for name, value in feedback.outputs.items():
            site_writer.set_analysis_result(name, value)
        run_result.analysis_results = dict(feedback.outputs)

        if feedback.annotations:
            site_writer.set_annotations(feedback.annotations)
            run_result.annotations = list(feedback.annotations)

    def observe_batch(
        self,
        observations: Sequence[PointObservation],
        run_result: HostScanRunResult,
        site_writer: ScanSiteDatasetWriter,
    ) -> dict[str, AnalysisFeedback]:
        """Handle one completed execution batch.

        Online analyses are re-evaluated on the accumulated scan data after each
        published batch. This keeps the semantics simple:

        - batch-local point writes happen first,
        - online analyses see exactly the data that is now visible to readers,
        - point policies can then react to the same batch-level analysis outputs.
        """
        del observations

        if not self._analyses:
            return {}

        axis_data = dict(run_result.coordinates)
        result_data = dict(run_result.values)
        raw_online_results = reduce(
            lambda x, y: merge_no_duplicates(x, y, kind="online analysis result"),
            (
                analysis.execute_online(axis_data, result_data, self._annotation_context)
                for analysis in self._analyses
            ),
            {},
        )
        online_results = {
            name: self._normalise_online_feedback(value)
            for name, value in raw_online_results.items()
        }
        for name, feedback in online_results.items():
            site_writer.set_online_analysis_result(name, feedback.outputs)
            site_writer.set_online_analysis_annotations(name, feedback.annotations)
        run_result.online_analysis_results = {
            name: feedback.outputs for name, feedback in online_results.items()
        }
        run_result.online_analysis_annotations = {
            name: list(feedback.annotations) for name, feedback in online_results.items()
        }
        return online_results

    def _normalise_online_feedback(self, value: AnalysisFeedback | dict[str, Any]) -> AnalysisFeedback:
        """Return the structured online-analysis payload for one analysis.

        Older online analyses returned only a dict of outputs. Newer code can return an
        ``AnalysisFeedback`` directly so outputs and annotations travel through the
        runtime together.
        """

        if isinstance(value, AnalysisFeedback):
            return value
        return AnalysisFeedback(outputs=dict(value))


@dataclass
class _ScanAxisBinding:
    """Temporary rebinding of one logical scan axis onto a dedicated store.

    Default-backed parameter stores are recomputed during host setup, so using them
    directly as scan axes would cause the scanned value to be reset on every point.
    The host runtime therefore gives each scanned axis its own temporary store for the
    duration of the session and restores the original stores afterwards.
    """

    handles: tuple[ParamHandle, ...]
    original_stores: tuple[ParamStore, ...]

    def restore(self) -> None:
        for handle, store in zip(self.handles, self.original_stores, strict=True):
            handle.set_store(store)


class _PointResultCollector:
    """Temporarily redirects selected result channels to ``SingleUseSink`` instances.

    This keeps the "one complete set of values per point" contract of the legacy
    runtime, while avoiding any coupling between point execution and dataset writing.
    """

    def __init__(self, channels: Sequence[BoundResultChannel]):
        self._channels = tuple(channels)
        self._original_sinks = dict[ResultChannel, Any]()

    def install(self) -> None:
        for binding in self._channels:
            channel = binding.channel
            self._original_sinks[channel] = channel.sink
            channel.set_sink(SingleUseSink())

    def discard_current(self) -> None:
        for binding in self._channels:
            binding.channel.sink.reset()

    def finish_point(self) -> OrderedDict[str, Any]:
        values = OrderedDict()
        for binding in self._channels:
            sink = binding.channel.sink
            if not sink.is_set():
                raise ValueError(
                    f"Missing value for result channel '{binding.channel.path}' "
                    "(push() not called for current point)"
                )
            values[binding.key] = sink.get()
        for binding in self._channels:
            binding.channel.sink.reset()
        return values

    def remove(self) -> None:
        self.discard_current()
        for binding in self._channels:
            binding.channel.set_sink(self._original_sinks[binding.channel])
        self._original_sinks.clear()


class _HostPointExecutor:
    """Execute already-resolved points against a host-only fragment."""

    def __init__(
        self,
        fragment: ExpFragment,
        axes: Sequence[BoundScanAxis],
        parameters: Sequence[BoundScanParameter],
        channels: Sequence[BoundResultChannel],
        parameter_mappings: Sequence[_BoundParameterMapping],
        site_path: tuple[str, ...],
        *,
        max_rtio_underflow_retries: int,
        max_transitory_error_retries: int,
    ):
        self._fragment = fragment
        self._axes = tuple(axes)
        self._parameters = tuple(parameters)
        self._channels = tuple(channels)
        self._parameter_mappings = tuple(parameter_mappings)
        self._site_path = site_path
        self._collector = _PointResultCollector(channels)
        self._runner = _PointInvocationRunner(
            fragment,
            fragment,
            max_rtio_underflow_retries=max_rtio_underflow_retries,
            max_transitory_error_retries=max_transitory_error_retries,
        )

    def install(self) -> None:
        self._collector.install()

    def remove(self) -> None:
        self._collector.remove()

    def execute_point(self, point, site_point_index: int) -> PointObservation | None:
        """Execute one point.

        Returns a completed observation on success. Returns ``None`` when a
        ``RestartKernelTransitoryError`` asks the caller to leave and re-enter the host
        setup context before retrying the same point.

        The per-point installation order is:

        1. direct values for any scanned ``ParamHandle`` axes,
        2. declared ``ParameterMapping`` updates derived from the current logical state,
        3. ``device_setup()`` / ``run_once()`` / ``device_cleanup()``.

        Keeping mappings here means both wrapper-fragment reparameterisations and ad
        hoc request-level mappings feed through the same runtime path.
        """

        axis_map = OrderedDict(
            (axis.key, value) for axis, value in zip(self._axes, point.axis_values, strict=True)
        )
        pseudoparam_map = OrderedDict()
        for axis, value in zip(self._axes, point.axis_values, strict=True):
            if axis.param_store is not None:
                axis.param_store.set_value(value)
            else:
                pseudoparam_map[axis.point_key] = value

        dependency_values = {
            axis.source: value
            for axis, value in zip(self._axes, point.axis_values, strict=True)
        }
        for mapping in self._parameter_mappings:
            for dependency in mapping.dependencies:
                if isinstance(dependency, ParamHandle) and dependency not in dependency_values:
                    dependency_values[dependency] = dependency.get()

            updates = mapping.mapping.compute(dependency_values)
            for target, value in updates.items():
                if target._store is None:
                    raise ValueError(
                        f"Cannot apply parameter mapping to unbound parameter '{target.name}'"
                    )
                target._store.set_value(value)
                dependency_values[target] = value

        parameter_map = OrderedDict(
            (parameter.key, parameter.handle.get()) for parameter in self._parameters
        )

        with _push_scan_context(ActiveScanContext(self._site_path, site_point_index)):
            if not self._runner.run():
                return None

        channel_values = self._collector.finish_point()
        return PointObservation(
            point_index=site_point_index,
            axis_values=axis_map,
            pseudoparam_values=pseudoparam_map,
            parameter_values=parameter_map,
            channel_values=channel_values,
            acquired_at_unix=time.time(),
        )


class _PointInvocationRunner(HasEnvironment):
    """Run one point body, including the kernel-wrapper path when needed.

    This is the per-point equivalent of the legacy ``_FragmentRunner``. The host
    runtime still owns the scan loop, but an individual point may enter the core if the
    fragment implementation chooses to express ``device_setup()``/``run_once()`` that
    way.
    """

    def build(
        self,
        fragment: ExpFragment,
        max_rtio_underflow_retries: int,
        max_transitory_error_retries: int,
    ):
        self.fragment = fragment
        self.max_rtio_underflow_retries = max_rtio_underflow_retries
        self.max_transitory_error_retries = max_transitory_error_retries
        self._requires_kernel_wrapper = any(
            is_kernel(method)
            for method in (
                fragment.device_setup,
                fragment.run_once,
                fragment.device_cleanup,
            )
        )
        if self._requires_kernel_wrapper:
            self.setattr_device("core")

    def run(self) -> bool:
        if self._requires_kernel_wrapper:
            return self._run_on_kernel()
        return self._run()

    @kernel
    def _run_on_kernel(self):
        return self._run()

    @portable
    def _run(self):
        num_underflows = 0
        num_transitory_errors = 0
        try:
            while True:
                try:
                    self.fragment.device_setup()
                    self.fragment.run_once()
                    return True
                except RTIOUnderflow:
                    num_underflows += 1
                    if num_underflows > self.max_rtio_underflow_retries:
                        raise
                    logger.warning(
                        "Ignoring RTIOUnderflow while executing point (%s/%s)",
                        num_underflows,
                        self.max_rtio_underflow_retries,
                    )
                except RestartKernelTransitoryError:
                    num_transitory_errors += 1
                    if num_transitory_errors > self.max_transitory_error_retries:
                        raise
                    logger.info(
                        "Restarting host setup after transitory error (%s/%s)",
                        num_transitory_errors,
                        self.max_transitory_error_retries,
                    )
                    return False
                except TransitoryError:
                    num_transitory_errors += 1
                    if num_transitory_errors > self.max_transitory_error_retries:
                        raise
                    logger.info(
                        "Retrying point after transitory error (%s/%s)",
                        num_transitory_errors,
                        self.max_transitory_error_retries,
                    )
        finally:
            self.fragment.device_cleanup()


class HostScanProgram:
    """Validated, fragment-bound host scan plan."""

    def __init__(
        self,
        fragment: ExpFragment,
        request: ScanRequest,
        axes: Sequence[BoundScanAxis],
        parameters: Sequence[BoundScanParameter],
        channels: Sequence[BoundResultChannel],
        parameter_mappings: Sequence[_BoundParameterMapping],
        site_writer: ScanSiteDatasetWriter,
        analysis_plan: _HostScanAnalysisPlan,
    ):
        self.fragment = fragment
        self.request = request
        self.axes = tuple(axes)
        self.parameters = tuple(parameters)
        self.channels = tuple(channels)
        self.parameter_mappings = tuple(parameter_mappings)
        self.point_source = request.point_source
        self.site_writer = site_writer
        self.analysis_plan = analysis_plan

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "site.fragment_fqn": self.fragment.fqn,
            "scan.point_source": self.point_source.describe(),
            "scan.parameters": {
                parameter.key: parameter.metadata()
                for parameter in self.parameters
            },
            "scan.pseudoparams": {
                axis.point_key: axis.metadata()
                for axis in self.axes
                if isinstance(axis.source, ScanVariable)
            },
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
        metadata.update(self.analysis_plan.metadata())
        return metadata


def _axis_source_key(source: ParamHandle | ScanVariable) -> tuple[Any, ...]:
    if isinstance(source, ParamHandle):
        return ("param", id(source.owner), source.name)
    return ("variable", source.name)


def _mapping_target_key(handle: ParamHandle) -> tuple[int, str]:
    return (id(handle.owner), handle.name)


def _build_bound_axes(
    axes: Sequence[ParamHandle | ScanVariable],
) -> list[BoundScanAxis]:
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
                    "initialise fragment parameters before building a host scan"
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
    """Return the actual fragment parameters that vary point-to-point."""

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
    """Merge, validate, and dependency-order all parameter mappings."""

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
    """Topologically order mappings by target/dependency relationships."""

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
                if not dependencies[other] and other not in ordered and other not in ready:
                    ready.append(other)

    if len(ordered) != len(mappings):
        raise ValueError("Parameter mappings contain a dependency cycle")
    return ordered


class HostScanProgramBuilder:
    """Bind a code-first ``ScanRequest`` to a concrete fragment instance."""

    def __init__(self, owner: HasEnvironment):
        self._owner = owner

    def build(self, fragment: ExpFragment, request: ScanRequest) -> HostScanProgram:
        if is_kernel(fragment.device_setup) or is_kernel(fragment.run_once):
            raise NotImplementedError(
                "The host runtime currently expects host-side device_setup()/run_once() "
                "methods. Host methods may still call @kernel helpers internally."
            )

        if request.point_source.axis_count != len(request.axes):
            raise ValueError(
                "Point source dimensionality does not match the number of requested axes"
            )

        axes = _build_bound_axes(request.axes)

        channel_dict = dict[str, ResultChannel]()
        fragment._collect_result_channels(channel_dict)
        channels = [
            BoundResultChannel(channel=channel, key=f"channel_{index}")
            for index, channel in enumerate(
                channel for channel in channel_dict.values() if channel.save_by_default
            )
        ]

        parameter_mappings = _collect_parameter_mappings(fragment, request, axes)
        parameters = _build_bound_parameters(axes, parameter_mappings)
        site_writer = ScanSiteDatasetWriter(self._owner, request.site)
        analysis_plan = _HostScanAnalysisPlan.build(fragment, axes, channels)
        return HostScanProgram(
            fragment,
            request,
            axes,
            parameters,
            channels,
            parameter_mappings,
            site_writer,
            analysis_plan,
        )


class HostScanProgramRunner:
    """Own the host-only scan loop.

    The runner is intentionally linear:

    1. prepare the fragment once,
    2. publish metadata once,
    3. repeatedly enter host setup,
    4. execute one batch of points,
    5. publish that completed batch,
    6. then consider pause, restart request, or completion.
    """

    def __init__(
        self,
        owner: HasEnvironment,
        program: HostScanProgram,
        *,
        max_rtio_underflow_retries: int,
        max_transitory_error_retries: int,
    ):
        self._program = program
        self._fragment = program.fragment
        self._executor = _HostPointExecutor(
            program.fragment,
            program.axes,
            program.parameters,
            program.channels,
            program.parameter_mappings,
            program.request.site.path,
            max_rtio_underflow_retries=max_rtio_underflow_retries,
            max_transitory_error_retries=max_transitory_error_retries,
        )
        self._scheduler = owner.get_device("scheduler")

    def run(self) -> HostScanRunResult:
        self._fragment.prepare()
        site_start_unix_time = time.time()
        self._program.site_writer.publish_metadata(
            self._program.metadata(),
            extra_metadata=self._program.request.metadata,
            start_unix_time=site_start_unix_time,
        )
        if self._program.request.site.segmented:
            parent = current_scan_context()
            self._program.site_writer.start_segment(
                parent_point_index=None if parent is None else parent.point_index,
                start_unix_time=time.time(),
            )

        result = HostScanRunResult.empty(
            self._program.axes,
            self._program.parameters,
            self._program.channels,
            self._program.site_writer.prefix,
            initial_annotations=self._program.analysis_plan.initial_annotations(),
        )

        self._executor.install()
        current_batch = list()
        batch_offset = 0

        try:
            while True:
                if batch_offset >= len(current_batch):
                    if self._program.point_source.is_finished():
                        break
                    current_batch = self._next_batch()
                    batch_offset = 0

                self._fragment.recompute_param_defaults()

                restart_host_context = False
                completed_batch: list[PointObservation] = []

                self._fragment.host_setup()
                try:
                    while batch_offset < len(current_batch):
                        observation = self._executor.execute_point(
                            current_batch[batch_offset],
                            self._program.site_writer.next_point_index,
                        )
                        if observation is None:
                            restart_host_context = True
                            break

                        completed_batch.append(observation)
                        batch_offset += 1
                finally:
                    self._fragment.host_cleanup()

                self._finish_completed_batch(completed_batch, result)

                if restart_host_context:
                    continue

                current_batch = []
                batch_offset = 0
                if self._should_pause_after_batch():
                    self._pause_after_batch()
        finally:
            self._executor.remove()

        if self._program.request.site.segmented:
            self._program.site_writer.finish_segment()
        # Finish the point stream before analyses run; later buffered implementations
        # should make this flush any still-pending point data.
        self._program.site_writer.flush()
        self._program.analysis_plan.execute(result, self._program.site_writer)
        self._program.site_writer.set_completed(True)
        self._program.site_writer.close()
        return result

    def _next_batch(self):
        requested_size = self._effective_batch_size()
        batch = self._program.point_source.next_batch(requested_size)
        if batch:
            return batch
        if self._program.point_source.is_finished():
            return []
        raise RuntimeError(
            f"{type(self._program.point_source).__name__} returned no points before finishing"
        )

    def _effective_batch_size(self) -> int:
        """Return the point count upper bound for the next execution batch.

        The execution policy owns the hard upper bound because batching is
        fundamentally a runtime/persistence choice. The point policy may still ask for
        a smaller natural batch size when, for example, it wants one ask/tell step per
        batch.
        """

        request_limit = self._program.request.execution_policy.max_points_per_batch
        if request_limit is None:
            request_limit = 1

        preferred = self._program.point_source.preferred_batch_size(request_limit)
        if preferred <= 0:
            raise ValueError("preferred_batch_size() must return a positive integer")
        return min(request_limit, preferred)

    def _finish_completed_batch(
        self,
        completed_batch: Sequence[PointObservation],
        result: HostScanRunResult,
    ) -> None:
        """Publish one completed batch at the runtime boundary.

        The order here is intentional:

        1. persist raw point observations,
        2. update the in-memory mirror,
        3. run future batch-level online analyses,
        4. let the point policy observe the completed batch,
        5. flush pending writer state before pause/restart/completion decisions.

        Keeping that boundary explicit makes later online analysis and writer-side
        buffering extensions much easier to reason about.
        """

        if not completed_batch:
            return

        self._program.site_writer.append_observations(completed_batch)
        result.record_batch(
            completed_batch,
            self._program.axes,
            self._program.parameters,
            self._program.channels,
        )
        online_analyses = self._program.analysis_plan.observe_batch(
            completed_batch, result, self._program.site_writer
        )
        self._program.point_source.observe_batch(
            BatchFeedback(
                observations=tuple(completed_batch),
                axis_data={
                    axis.source: tuple(result.coordinates[axis.identity])
                    for axis in self._program.axes
                },
                parameter_data={
                    parameter.handle: tuple(result.parameters[parameter.identity])
                    for parameter in self._program.parameters
                },
                result_data={
                    binding.channel: tuple(result.values[binding.channel])
                    for binding in self._program.channels
                },
                online_analyses=online_analyses,
            )
        )
        self._program.site_writer.flush()

    def _has_more_work(self) -> bool:
        return not self._program.point_source.is_finished()

    def _should_pause_after_batch(self) -> bool:
        """Return whether the scheduler wants to pause after the current batch.

        Scheduler interaction is intentionally aligned with published batch boundaries.
        A completed batch is the first point where it is safe to:

        - expose written data to readers,
        - re-enter host setup later if needed,
        - let the scheduler interrupt long-running scans without splitting a logical
          ask/tell step in half.
        """

        if not self._has_more_work():
            return False
        return self._scheduler.check_pause()

    def _pause_after_batch(self) -> None:
        """Yield control to the scheduler after a completed batch boundary."""

        self._scheduler.pause()


class HostScanSession:
    """Prepared host-only scan execution.

    This is the main entry point for direct code use and for thin ``EnvExperiment``
    adapters.
    """

    def __init__(
        self,
        owner: HasEnvironment,
        fragment: ExpFragment,
        request: ScanRequest,
        *,
        overrides: dict[str, list[tuple[str, ParamStore]]] | None = None,
        max_rtio_underflow_retries: int = 3,
        max_transitory_error_retries: int = 10,
    ):
        if _fragment_tree_needs_param_initialisation(fragment):
            fragment.init_params(overrides={} if overrides is None else overrides)
        self._axis_bindings = _install_scan_axis_stores(
            [axis for axis in request.axes if isinstance(axis, ParamHandle)]
        )
        builder = HostScanProgramBuilder(owner)
        self.program = builder.build(fragment, request)
        self._runner = HostScanProgramRunner(
            owner,
            self.program,
            max_rtio_underflow_retries=max_rtio_underflow_retries,
            max_transitory_error_retries=max_transitory_error_retries,
        )

    def run(self) -> HostScanRunResult:
        try:
            return self._runner.run()
        finally:
            for binding in self._axis_bindings:
                binding.restore()


def run_host_scan(
    owner: HasEnvironment,
    fragment: ExpFragment,
    request: ScanRequest,
    *,
    overrides: dict[str, list[tuple[str, ParamStore]]] | None = None,
    max_rtio_underflow_retries: int = 3,
    max_transitory_error_retries: int = 10,
) -> HostScanRunResult:
    """Convenience helper to prepare and run a host-only scan in one call."""

    session = HostScanSession(
        owner,
        fragment,
        request,
        overrides=overrides,
        max_rtio_underflow_retries=max_rtio_underflow_retries,
        max_transitory_error_retries=max_transitory_error_retries,
    )
    return session.run()


def run_subscan(
    owner: HasEnvironment,
    fragment: ExpFragment,
    request: ScanRequest,
    *,
    name: str,
    segmented: bool = True,
    extra_metadata: Mapping[str, Any] | None = None,
    overrides: dict[str, list[tuple[str, ParamStore]]] | None = None,
    max_rtio_underflow_retries: int = 3,
    max_transitory_error_retries: int = 10,
) -> HostScanRunResult:
    """Run ``request`` as a child scan nested under the current parent point.

    This is deliberately only a convenience wrapper. The nested scan still runs
    through the same ``HostScanSession`` and ``HostScanProgramRunner`` path as a root
    scan; the helper merely derives the structural child scan site and reuses the
    caller's request unchanged otherwise.

    ``request.site`` remains meaningful for site-local options such as:

    - ``dataset_prefix`` overrides,
    - request-specific site metadata.

    Any ``extra_metadata`` passed here is merged on top of the request site's own
    extra metadata before the child site is created.
    """

    base_site = request.site
    child_site = make_child_scan_site(
        name,
        segmented=segmented,
        extra_metadata={
            **dict(base_site.extra_metadata),
            **({} if extra_metadata is None else dict(extra_metadata)),
        },
    )
    if base_site.dataset_prefix is not None:
        child_site = ScanSite(
            path=child_site.path,
            parent_path=child_site.parent_path,
            dataset_prefix=base_site.dataset_prefix,
            segmented=child_site.segmented,
            extra_metadata=child_site.extra_metadata,
        )

    return run_host_scan(
        owner,
        fragment,
        request.with_site(child_site),
        overrides=overrides,
        max_rtio_underflow_retries=max_rtio_underflow_retries,
        max_transitory_error_retries=max_transitory_error_retries,
    )


def _fragment_tree_needs_param_initialisation(fragment: ExpFragment) -> bool:
    """Return whether any free parameter handle in the fragment tree is unbound.

    Nested scans often target fragments that are already part of an initialised parent
    tree. Re-running ``init_params()`` in that situation would overwrite the current
    stores and lose the parent scan point's parameter values.

    The new host runtime therefore initialises parameters lazily: only if it sees an
    unbound free parameter handle anywhere in the target fragment tree does it call
    ``init_params()``.
    """

    for name in fragment._free_params.keys():
        if getattr(fragment, name)._store is None:
            return True
    for subfragment in fragment._subfragments:
        if _fragment_tree_needs_param_initialisation(subfragment):
            return True
    return False


def _install_scan_axis_stores(
    axis_handles: Sequence[ParamHandle],
) -> list[_ScanAxisBinding]:
    """Bind each requested scan axis to a dedicated temporary store.

    The rebinding happens once per session. All handles corresponding to the same local
    parameter on the same fragment are moved together so direct use and rebound use see
    the same scanned value.
    """

    bindings = []
    seen = set[tuple[int, str]]()
    for handle in axis_handles:
        key = (id(handle.owner), handle.name)
        if key in seen:
            raise ValueError(
                f"Scan axis '{handle.owner._stringize_path()}/{handle.name}' was specified more than once"
            )
        seen.add(key)

        affected_handles = tuple(handle.owner._get_all_handles_for_param(handle.name))
        original_stores = tuple(bound_handle._store for bound_handle in affected_handles)
        if any(store is None for store in original_stores):
            raise ValueError(
                f"Cannot scan parameter '{handle.name}' before its stores are initialised"
            )

        scan_store = type(handle._store)(handle._store.identity, handle.get())
        for bound_handle in affected_handles:
            bound_handle.set_store(scan_store)

        bindings.append(_ScanAxisBinding(affected_handles, original_stores))
    return bindings


class HostScanExperiment(EnvExperiment):
    """Thin ``EnvExperiment`` adapter for the new host-only runtime.

    This keeps the new runtime runnable from ARTIQ without dragging the dashboard
    argument format into the first implementation. A code-defined request factory builds
    the ``ScanRequest`` from the fragment instance.
    """

    def build(
        self,
        fragment_init,
        request_factory,
        *,
        max_rtio_underflow_retries: int = 3,
        max_transitory_error_retries: int = 10,
    ) -> None:
        self.fragment = fragment_init()
        self._request_factory = request_factory
        self._max_rtio_underflow_retries = max_rtio_underflow_retries
        self._max_transitory_error_retries = max_transitory_error_retries
        self._session = None

    def prepare(self) -> None:
        request = self._request_factory(self.fragment)
        self._session = HostScanSession(
            self,
            self.fragment,
            request,
            max_rtio_underflow_retries=self._max_rtio_underflow_retries,
            max_transitory_error_retries=self._max_transitory_error_retries,
        )

    def run(self) -> None:
        self._session.run()


def make_fragment_host_scan_exp(
    fragment_class: type[ExpFragment],
    request_factory,
    *args,
    max_rtio_underflow_retries: int = 3,
    max_transitory_error_retries: int = 10,
) -> type[HostScanExperiment]:
    """Create a runnable ``EnvExperiment`` for the new host-only runtime.

    Example::

        MyHostScan = make_fragment_host_scan_exp(
            MyFragment,
            lambda fragment: ScanRequest.zipped(
                [
                    (fragment.detuning, [-1.0, 0.0, 1.0]),
                    (fragment.amplitude, [0.2, 0.4, 0.6]),
                ]
            ),
        )

    The request factory is intentionally passed the fragment instance so callers can
    build requests directly from fragment handles.
    """

    class FragmentHostScanShim(HostScanExperiment):
        def build(self):
            super().build(
                lambda: fragment_class(self, [], *args),
                request_factory,
                max_rtio_underflow_retries=max_rtio_underflow_retries,
                max_transitory_error_retries=max_transitory_error_retries,
            )

    # Present the generated experiment as a normal top-level class to ARTIQ's
    # discovery/examine machinery rather than as a nested local shim.
    FragmentHostScanShim.__name__ = fragment_class.__name__
    FragmentHostScanShim.__qualname__ = fragment_class.__name__
    FragmentHostScanShim.__module__ = fragment_class.__module__
    FragmentHostScanShim.__doc__ = fragment_class.__doc__
    return FragmentHostScanShim

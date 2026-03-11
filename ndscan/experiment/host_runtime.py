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
from .fragment import ExpFragment, RestartKernelTransitoryError, TransitoryError
from .parameters import ParamHandle, ParamStore
from .point_source import (
    CartesianPointSource,
    ExplicitPointSource,
    PointSource,
    SinglePointSource,
    ZipPointSource,
)
from .result_channels import LastValueSink, ResultChannel, SingleUseSink
from .scan_runner import describe_analyses, filter_default_analyses
from .scan_site import ScanSite, ScanSiteDatasetWriter
from .utils import is_kernel

__all__ = [
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
class ScanRequest:
    """User-facing host-runtime scan request.

    The first implementation is deliberately code-first: callers provide the actual
    parameter handles to scan and a ``PointSource`` describing the point strategy. A
    future dashboard adapter can resolve selector syntax into the same object without
    changing the runtime core again.
    """

    axes: tuple[ParamHandle, ...]
    point_source: PointSource
    site: ScanSite = field(default_factory=ScanSite)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def single(
        cls, *, site: ScanSite | None = None, metadata: Mapping[str, Any] | None = None
    ) -> "ScanRequest":
        return cls(
            axes=(),
            point_source=SinglePointSource(),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def cartesian(
        cls,
        axes: Sequence[tuple[ParamHandle, Sequence[Any]]],
        *,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ScanRequest":
        return cls(
            axes=tuple(handle for handle, _ in axes),
            point_source=CartesianPointSource([values for _, values in axes]),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def zipped(
        cls,
        axes: Sequence[tuple[ParamHandle, Sequence[Any]]],
        *,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ScanRequest":
        return cls(
            axes=tuple(handle for handle, _ in axes),
            point_source=ZipPointSource([values for _, values in axes]),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def explicit(
        cls,
        axes: Sequence[ParamHandle],
        points: Sequence[Sequence[Any]],
        *,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ScanRequest":
        return cls(
            axes=tuple(axes),
            point_source=ExplicitPointSource(len(axes), points),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True)
class BoundScanAxis:
    """Runtime-bound scan axis metadata."""

    handle: ParamHandle
    key: str
    path: str
    param_schema: dict[str, Any]
    param_store: ParamStore


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
    """

    point_index: int
    axis_values: OrderedDict[str, Any]
    channel_values: OrderedDict[str, Any]
    acquired_at_unix: float | None = None


@dataclass
class HostScanRunResult:
    """In-memory copy of the data produced by a host-runtime scan.

    Keeping a small in-memory mirror of the flat site is useful for tests and for the
    first analysis hooks. The canonical on-disk/broadcast representation remains the
    scan-site datasets written by ``ScanSiteDatasetWriter``.
    """

    coordinates: OrderedDict[tuple[str, str], list[Any]]
    values: dict[ResultChannel, list[Any]]
    analysis_results: dict[str, Any]
    annotations: list[dict[str, Any]]
    site_prefix: str

    @classmethod
    def empty(
        cls,
        axes: Sequence[BoundScanAxis],
        channels: Sequence[BoundResultChannel],
        site_prefix: str,
        initial_annotations: Sequence[dict[str, Any]] = (),
    ) -> "HostScanRunResult":
        return cls(
            coordinates=OrderedDict(
                ((axis.param_schema["fqn"], axis.path), []) for axis in axes
            ),
            values={binding.channel: [] for binding in channels},
            analysis_results={},
            annotations=list(initial_annotations),
            site_prefix=site_prefix,
        )

    def record(self, observation: PointObservation, axes: Sequence[BoundScanAxis], channels: Sequence[BoundResultChannel]) -> None:
        for axis in axes:
            self.coordinates[(axis.param_schema["fqn"], axis.path)].append(
                observation.axis_values[axis.key]
            )
        for channel in channels:
            self.values[channel.channel].append(observation.channel_values[channel.key])


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
        analyses = filter_default_analyses(fragment, axes)

        axis_indices = {
            axis.param_store.identity: index for index, axis in enumerate(axes)
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
            lambda handle: axis_indices[handle._store.identity],
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

        for name, value in analysis_results.items():
            site_writer.set_analysis_result(name, value)
        run_result.analysis_results = analysis_results

        if annotations:
            site_writer.set_annotations(annotations)
            run_result.annotations = annotations


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
        channels: Sequence[BoundResultChannel],
        site_path: tuple[str, ...],
        *,
        max_rtio_underflow_retries: int,
        max_transitory_error_retries: int,
    ):
        self._fragment = fragment
        self._axes = tuple(axes)
        self._channels = tuple(channels)
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
        """

        axis_map = OrderedDict(
            (axis.key, value) for axis, value in zip(self._axes, point.axis_values, strict=True)
        )
        for axis, value in zip(self._axes, point.axis_values, strict=True):
            axis.param_store.set_value(value)

        with _push_scan_context(ActiveScanContext(self._site_path, site_point_index)):
            if not self._runner.run():
                return None

        channel_values = self._collector.finish_point()
        return PointObservation(
            point_index=site_point_index,
            axis_values=axis_map,
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
        channels: Sequence[BoundResultChannel],
        site_writer: ScanSiteDatasetWriter,
        analysis_plan: _HostScanAnalysisPlan,
    ):
        self.fragment = fragment
        self.request = request
        self.axes = tuple(axes)
        self.channels = tuple(channels)
        self.point_source = request.point_source
        self.site_writer = site_writer
        self.analysis_plan = analysis_plan

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "site.fragment_fqn": self.fragment.fqn,
            "scan.point_source": self.point_source.describe(),
            "scan.axes": {
                axis.key: {
                    "path": axis.path,
                    "param": axis.param_schema,
                }
                for axis in self.axes
            },
            "scan.channels": {
                binding.key: binding.channel.describe()
                for binding in self.channels
            },
        }
        metadata.update(self.analysis_plan.metadata())
        return metadata


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

        axes = []
        for index, handle in enumerate(request.axes):
            if handle._store is None:
                raise ValueError(
                    f"Parameter handle '{handle.name}' is not bound to a store yet; "
                    "initialise fragment parameters before building a host scan"
                )
            axes.append(
                BoundScanAxis(
                    handle=handle,
                    key=f"axis_{index}",
                    path=handle.owner._stringize_path(),
                    param_schema=handle.parameter.describe(),
                    param_store=handle._store,
                )
            )

        channel_dict = dict[str, ResultChannel]()
        fragment._collect_result_channels(channel_dict)
        channels = [
            BoundResultChannel(channel=channel, key=f"channel_{index}")
            for index, channel in enumerate(
                channel for channel in channel_dict.values() if channel.save_by_default
            )
        ]

        site_writer = ScanSiteDatasetWriter(self._owner, request.site)
        analysis_plan = _HostScanAnalysisPlan.build(fragment, axes, channels)
        return HostScanProgram(
            fragment, request, axes, channels, site_writer, analysis_plan
        )


class HostScanProgramRunner:
    """Own the host-only scan loop.

    The runner is intentionally linear:

    1. prepare the fragment once,
    2. publish metadata once,
    3. repeatedly enter host setup,
    4. execute points until pause, restart request, or completion,
    5. write each completed observation through the scan-site writer.
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
            program.channels,
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
            self._program.channels,
            self._program.site_writer.prefix,
            initial_annotations=self._program.analysis_plan.initial_annotations(),
        )

        self._executor.install()
        points = iter(self._program.point_source)
        current_point = next(points, None)

        try:
            while current_point is not None:
                self._fragment.recompute_param_defaults()

                restart_host_context = False
                should_pause = False

                self._fragment.host_setup()
                try:
                    while current_point is not None:
                        observation = self._executor.execute_point(
                            current_point,
                            self._program.site_writer.next_point_index,
                        )
                        if observation is None:
                            restart_host_context = True
                            break

                        self._program.site_writer.append_observation(observation)
                        self._program.point_source.observe(observation)
                        result.record(
                            observation, self._program.axes, self._program.channels
                        )

                        current_point = next(points, None)
                        if current_point is not None and self._scheduler.check_pause():
                            should_pause = True
                            break
                finally:
                    self._fragment.host_cleanup()

                if restart_host_context:
                    # ``flush()`` is currently a no-op, but this marks the intended
                    # boundary where future buffered writers should publish any
                    # completed points before the host context is re-entered.
                    self._program.site_writer.flush()
                    continue
                if should_pause:
                    # Pause points are another natural batching boundary: keep the
                    # semantics explicit now so buffered writers can hook in later.
                    self._program.site_writer.flush()
                    self._scheduler.pause()
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
        self._axis_bindings = _install_scan_axis_stores(request.axes)
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

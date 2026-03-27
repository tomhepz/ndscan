"""Execution backends and controller loop for the prepared runtime."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from artiq.coredevice.exceptions import RTIOUnderflow
from artiq.language import HasEnvironment, host_only, kernel, kernel_from_string, portable, rpc

from .analysis import HostScanAnalysisEngine
from .context import (
    ActiveScanContext,
    PreviewCoordinator,
    RunContext,
    _KernelParentScanContextProvider,
    _current_effective_scan_context,
    _push_kernel_parent_scan_context,
    _push_run_context,
    _push_scan_context,
    current_run_context,
)
from .persistence import ScanSiteDatasetWriter
from .program import (
    BoundResultChannel,
    BoundScanAxis,
    BoundScanParameter,
    HostScanProgram,
    PointObservation,
    ScanInspection,
    _BoundParameterMapping,
    _HostObservationTransport,
    _HostPointBatchSource,
    _HostAnalysisAdapter,
    _ResolvedExecutionPoint,
    _build_bound_axes,
    _build_bound_parameters,
    _can_use_kernel_streaming_executor,
    _collect_parameter_mappings,
    _fragment_tree_needs_param_initialisation,
    _install_scan_axis_stores,
    _missing_kernel_core_devices,
    _publish_completed_batch,
    _resolve_execution_batch,
    _resolve_execution_point,
)
from ..define.fragment import ExpFragment, RestartKernelTransitoryError, TransitoryError
from ..define.parameters import ParamHandle
from ..define.result_channels import ResultChannel, SingleUseSink
from ..define.utils import is_kernel
from ..scan.request import ScanRequest

logger = logging.getLogger(__name__)


def _execute_scan_request_inspection(
    owner: HasEnvironment,
    fragment: ExpFragment,
    request: ScanRequest,
    *,
    overrides: dict[str, list[tuple[str, Any]]] | None,
    run_context: RunContext | None,
    max_rtio_underflow_retries: int,
    max_transitory_error_retries: int,
) -> ScanInspection:
    if _fragment_tree_needs_param_initialisation(fragment):
        fragment.init_params(overrides={} if overrides is None else overrides)
    axis_bindings = _install_scan_axis_stores(
        [axis for axis in request.axes if isinstance(axis, ParamHandle)]
    )
    try:
        builder = HostScanProgramBuilder(owner)
        program = builder.build(fragment, request)
        runner = HostScanProgramRunner(
            owner,
            program,
            run_context=run_context,
            max_rtio_underflow_retries=max_rtio_underflow_retries,
            max_transitory_error_retries=max_transitory_error_retries,
        )
        return runner.run()
    finally:
        for binding in axis_bindings:
            binding.restore()


class _PointResultCollector:
    """Temporarily redirects selected result channels to ``SingleUseSink`` instances."""

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


class _ResidentKernelBatchState:
    """Shared host-side state for one resident kernel execution region."""

    def __init__(
        self,
        axes: Sequence[BoundScanAxis],
        parameters: Sequence[BoundScanParameter],
        parameter_mappings: Sequence[_BoundParameterMapping],
        collector: _PointResultCollector,
    ):
        self._axes = tuple(axes)
        self._parameters = tuple(parameters)
        self._parameter_mappings = tuple(parameter_mappings)
        self._collector = collector
        self._current_chunk = list[_ResolvedExecutionPoint]()
        self._completed_batch = list[PointObservation]()
        self._current_next_point_index = 0

    @property
    def current_next_point_index(self) -> int:
        return self._current_next_point_index

    @host_only
    def reset(self) -> None:
        self._current_chunk.clear()
        self._completed_batch.clear()
        self._current_next_point_index = 0

    @host_only
    def get_param_values_chunk(self, *, next_batch, next_point_index):
        if not self._current_chunk:
            batch = next_batch()
            if not batch:
                return tuple([] for _ in self._parameters)
            self._current_chunk = _resolve_execution_batch(
                batch,
                self._axes,
                self._parameters,
                self._parameter_mappings,
            )
            self._current_next_point_index = next_point_index()
            self._update_host_param_stores()

        values = tuple([] for _ in self._parameters)
        for point in self._current_chunk:
            for index, value in enumerate(point.rpc_parameter_values):
                values[index].append(value)
        return values

    @host_only
    def _update_host_param_stores(self) -> None:
        if not self._current_chunk:
            return
        point = self._current_chunk[0]
        for index, parameter in enumerate(self._parameters):
            parameter.handle._store.set_from_rpc(point.rpc_parameter_values[index])

    def retry_point(self) -> None:
        self._collector.discard_current()

    def point_completed(self) -> None:
        point = self._current_chunk.pop(0)
        observation = PointObservation(
            point_index=self._current_next_point_index,
            axis_values=point.axis_values,
            pseudoparam_values=point.pseudoparam_values,
            parameter_values=point.parameter_values,
            channel_values=self._collector.finish_point(),
            point_metadata=OrderedDict(point.point.metadata.items()),
            acquired_at_unix=time.time(),
        )
        self._completed_batch.append(observation)
        self._current_next_point_index += 1
        self._update_host_param_stores()

    @host_only
    def take_completed_batch(self) -> tuple[PointObservation, ...]:
        completed = tuple(self._completed_batch)
        self._completed_batch.clear()
        return completed


@dataclass(frozen=True)
class _BatchExecutionResult:
    """Result of executing one runtime batch through an execution backend."""

    observations: tuple[PointObservation, ...]
    restart_host_context: bool = False
    elapsed_s: float = 0.0
    executor_entries: int = 1


class HostExecutor:
    """Execute already-resolved points against a fragment from the host runtime."""

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

    def execute_batch(self, points, *, start_point_index: int) -> _BatchExecutionResult:
        completed = list[PointObservation]()
        restart_host_context = False
        started_at = time.perf_counter()

        self._fragment.host_setup()
        try:
            for offset, point in enumerate(points):
                observation = self._execute_point(
                    point, start_point_index=start_point_index + offset
                )
                if observation is None:
                    restart_host_context = True
                    break
                completed.append(observation)
        finally:
            self._fragment.host_cleanup()

        return _BatchExecutionResult(
            observations=tuple(completed),
            restart_host_context=restart_host_context,
            elapsed_s=time.perf_counter() - started_at,
        )

    def _execute_point(self, point, *, start_point_index: int) -> PointObservation | None:
        resolved = _resolve_execution_point(
            point,
            self._axes,
            self._parameters,
            self._parameter_mappings,
        )

        with _push_scan_context(ActiveScanContext(self._site_path, start_point_index)):
            if not self._runner.run():
                return None

        channel_values = self._collector.finish_point()
        return PointObservation(
            point_index=start_point_index,
            axis_values=resolved.axis_values,
            pseudoparam_values=resolved.pseudoparam_values,
            parameter_values=resolved.parameter_values,
            channel_values=channel_values,
            point_metadata=OrderedDict(point.metadata.items()),
            acquired_at_unix=time.time(),
        )


class KernelStreamingExecutor:
    """Execute a strict subset of scans through one resident kernel session."""

    _STATUS_PROCEED = 0
    _STATUS_RESTART_HOST_CONTEXT = 1
    _STATUS_PAUSE = 2
    _STATUS_COMPLETE = 3

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
        self._parameter_mappings = tuple(parameter_mappings)
        self._site_path = site_path
        self._collector = _PointResultCollector(channels)
        self._batch_state = _ResidentKernelBatchState(
            self._axes,
            self._parameters,
            self._parameter_mappings,
            self._collector,
        )
        self._runner = _ResidentKernelPointRunner(
            fragment,
            fragment,
            self,
        )
        self._runner.configure_runner(
            self._parameters,
            max_rtio_underflow_retries=max_rtio_underflow_retries,
            max_transitory_error_retries=max_transitory_error_retries,
        )

        self._result: ScanInspection | None = None
        self._next_batch = None
        self._finish_completed_batch = None
        self._should_pause_after_batch = None
        self._pause_after_batch = None
        self._has_more_work = None
        self._next_point_index = None

    def install(self) -> None:
        self._collector.install()

    def remove(self) -> None:
        self._collector.remove()

    def run_to_completion(
        self,
        result: ScanInspection,
        *,
        next_batch,
        finish_completed_batch,
        should_pause_after_batch,
        pause_after_batch,
        has_more_work,
        next_point_index,
    ) -> None:
        self._result = result
        self._next_batch = next_batch
        self._finish_completed_batch = finish_completed_batch
        self._should_pause_after_batch = should_pause_after_batch
        self._pause_after_batch = pause_after_batch
        self._has_more_work = has_more_work
        self._next_point_index = next_point_index

        try:
            while True:
                self._fragment.recompute_param_defaults()

                started_at = time.perf_counter()
                self._fragment.host_setup()
                try:
                    with _push_kernel_parent_scan_context(
                        _KernelParentScanContextProvider(
                            self._site_path,
                            lambda: self._batch_state.current_next_point_index,
                        )
                    ):
                        status = self._runner.acquire()
                finally:
                    self._fragment.host_cleanup()

                result.runtime_stats.executor_entry_count += 1
                elapsed_s = time.perf_counter() - started_at
                if result.runtime_stats.first_executor_entry_elapsed_s is None:
                    result.runtime_stats.first_executor_entry_elapsed_s = elapsed_s
                result.runtime_stats.total_executor_elapsed_s += elapsed_s

                if status == self._STATUS_COMPLETE:
                    return
                if status == self._STATUS_PAUSE:
                    self._pause_after_batch()
                    continue
                if status == self._STATUS_RESTART_HOST_CONTEXT:
                    continue
                raise RuntimeError(f"Unexpected kernel streaming executor status: {status}")
        finally:
            self._result = None
            self._next_batch = None
            self._finish_completed_batch = None
            self._should_pause_after_batch = None
            self._pause_after_batch = None
            self._has_more_work = None
            self._next_point_index = None
            self._batch_state.reset()

    @host_only
    def _get_param_values_chunk(self):
        return self._batch_state.get_param_values_chunk(
            next_batch=self._next_batch,
            next_point_index=self._next_point_index,
        )

    def _retry_point(self):
        self._batch_state.retry_point()

    def _point_completed(self):
        self._batch_state.point_completed()

    def _finish_chunk(self):
        completed_batch = self._batch_state.take_completed_batch()
        if completed_batch:
            self._finish_completed_batch(completed_batch, self._result)

        if self._should_pause_after_batch():
            return self._STATUS_PAUSE
        if not self._has_more_work():
            return self._STATUS_COMPLETE
        return self._STATUS_PROCEED

    def _finish_chunk_after_restart(self):
        completed_batch = self._batch_state.take_completed_batch()
        if completed_batch:
            self._finish_completed_batch(completed_batch, self._result)
        return self._STATUS_RESTART_HOST_CONTEXT


class _ResidentKernelPointRunner(HasEnvironment):
    """Configurable resident kernel loop shared by top-level and prepared scans."""

    def build(
        self,
        fragment: ExpFragment,
        owner: Any,
    ):
        self.fragment = fragment
        self.owner = owner
        self.max_rtio_underflow_retries = 0
        self.max_transitory_error_retries = 0
        self.setattr_device("core")
        self._run_chunk = kernel_from_string(["self"], "return self._STATUS_COMPLETE")

    @kernel
    def acquire(self) -> np.int32:
        try:
            while True:
                result = self._run_chunk(self)
                if result != self._STATUS_PROCEED:
                    return np.int32(result)
        finally:
            self.fragment.device_cleanup()
        assert False, "Execution never reaches here, return is just to pacify compiler."
        return np.int32(self._STATUS_COMPLETE)

    @portable
    def _run_point(self) -> np.int32:
        num_underflows = 0
        num_transitory_errors = 0
        while True:
            try:
                self.fragment.device_setup()
                self.fragment.run_once()
                self._point_completed()
                return np.int32(self._STATUS_PROCEED)
            except RTIOUnderflow:
                num_underflows += 1
                if num_underflows > self.max_rtio_underflow_retries:
                    raise
                logger.warning(
                    "Ignoring RTIOUnderflow while executing resident kernel point (%s/%s)",
                    num_underflows,
                    self.max_rtio_underflow_retries,
                )
                self._retry_point()
            except RestartKernelTransitoryError:
                num_transitory_errors += 1
                if num_transitory_errors > self.max_transitory_error_retries:
                    raise
                logger.info(
                    "Restarting host setup after resident-kernel transitory error (%s/%s)",
                    num_transitory_errors,
                    self.max_transitory_error_retries,
                )
                self._retry_point()
                return np.int32(self._finish_chunk_after_restart())
            except TransitoryError:
                num_transitory_errors += 1
                if num_transitory_errors > self.max_transitory_error_retries:
                    raise
                logger.info(
                    "Retrying resident kernel point after transitory error (%s/%s)",
                    num_transitory_errors,
                    self.max_transitory_error_retries,
                )
                self._retry_point()
        return np.int32(self._STATUS_PROCEED)

    _STATUS_PROCEED = np.int32(KernelStreamingExecutor._STATUS_PROCEED)
    _STATUS_RESTART_HOST_CONTEXT = np.int32(
        KernelStreamingExecutor._STATUS_RESTART_HOST_CONTEXT
    )
    _STATUS_PAUSE = np.int32(KernelStreamingExecutor._STATUS_PAUSE)
    _STATUS_COMPLETE = np.int32(KernelStreamingExecutor._STATUS_COMPLETE)

    @host_only
    def configure_runner(
        self,
        parameters: Sequence[BoundScanParameter],
        *,
        max_rtio_underflow_retries: int,
        max_transitory_error_retries: int,
    ) -> None:
        self.max_rtio_underflow_retries = max_rtio_underflow_retries
        self.max_transitory_error_retries = max_transitory_error_retries
        self._get_param_values_chunk.__func__.__annotations__ = {
            "return": tuple.__class_getitem__(
                tuple(
                    list[parameter.handle._store.RpcType] for parameter in parameters
                )
            )
        }
        for index, parameter in enumerate(parameters):
            setattr(
                self,
                f"_param_setter_{index}",
                parameter.handle._store.set_from_rpc,
            )
        self._run_chunk = self._build_run_chunk(len(parameters))

    def _build_run_chunk(self, num_parameters):
        param_decl = " ".join(f"p{idx}," for idx in range(num_parameters))
        code = ""
        code += f"({param_decl}) = self._get_param_values_chunk()\n"
        code += "if not p0:\n"
        code += "    return self._STATUS_COMPLETE\n"
        code += "for i in range(len(p0)):\n"
        for idx in range(num_parameters):
            code += f"    self._param_setter_{idx}(p{idx}[i])\n"
        code += "    point_result = self._run_point()\n"
        code += "    if point_result != self._STATUS_PROCEED:\n"
        code += "        return point_result\n"
        code += "return self._finish_chunk()"
        return kernel_from_string(["self"], code)

    @rpc
    def _get_param_values_chunk(self):
        return self.owner._get_param_values_chunk()

    @rpc(flags={"async"})
    def _retry_point(self):
        self.owner._retry_point()

    @rpc(flags={"async"})
    def _point_completed(self):
        self.owner._point_completed()

    @rpc
    def _finish_chunk(self) -> np.int32:
        return np.int32(self.owner._finish_chunk())

    @rpc
    def _finish_chunk_after_restart(self) -> np.int32:
        return np.int32(self.owner._finish_chunk_after_restart())


class _PointInvocationRunner(HasEnvironment):
    """Run one point body, including the kernel-wrapper path when needed."""

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


class HostScanProgramBuilder:
    """Bind a code-first ``ScanRequest`` to a concrete fragment instance."""

    def __init__(self, owner: HasEnvironment):
        self._owner = owner

    def build(self, fragment: ExpFragment, request: ScanRequest) -> HostScanProgram:
        if request.point_policy.axis_count != len(request.axes):
            raise ValueError(
                "Point policy dimensionality does not match the number of requested axes"
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
        missing_kernel_devices = _missing_kernel_core_devices(fragment)
        if missing_kernel_devices:
            names = ", ".join(f"'{name}'" for name in missing_kernel_devices)
            raise ValueError(
                "Fragments using @kernel in the prepared runtime must declare the "
                f"corresponding device attribute(s) {names} during build_fragment(); "
                "for ordinary kernels this usually means calling "
                "self.setattr_device('core')"
            )
        if _can_use_kernel_streaming_executor(fragment, axes, parameters):
            pass
        elif any(
            is_kernel(method)
            for method in (
                fragment.device_setup,
                fragment.run_once,
                fragment.device_cleanup,
            )
        ):
            raise NotImplementedError(
                "Direct @kernel point bodies currently require at least one concrete "
                "fragment parameter to vary point-to-point; pure pseudoparam scans "
                "with no mapped parameter targets are not yet supported"
            )
        site_writer = ScanSiteDatasetWriter(self._owner, request.site)
        analysis_engine = HostScanAnalysisEngine.build(fragment, axes, channels)
        return HostScanProgram(
            fragment,
            request,
            axes,
            parameters,
            channels,
            parameter_mappings,
            site_writer,
            analysis_engine,
        )


class HostScanProgramRunner:
    """Own the host-only scan loop."""

    def __init__(
        self,
        owner: HasEnvironment,
        program: HostScanProgram,
        *,
        run_context: RunContext | None,
        max_rtio_underflow_retries: int,
        max_transitory_error_retries: int,
    ):
        self._owner = owner
        self._program = program
        self._run_context = run_context
        self._fragment = program.fragment
        if _can_use_kernel_streaming_executor(
            program.fragment, program.axes, program.parameters
        ):
            self._executor = KernelStreamingExecutor(
                program.fragment,
                program.axes,
                program.parameters,
                program.channels,
                program.parameter_mappings,
                program.request.site.path,
                max_rtio_underflow_retries=max_rtio_underflow_retries,
                max_transitory_error_retries=max_transitory_error_retries,
            )
        else:
            self._executor = HostExecutor(
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

    def run(self) -> ScanInspection:
        run_context = self._resolved_run_context()
        preview = run_context.preview
        self._program.transport.register_preview(preview)

        try:
            with _push_run_context(run_context):
                self._fragment.prepare()
                site_start_unix_time = time.time()
                self._program.transport.publish_metadata(
                    self._program.metadata(),
                    extra_metadata=self._program.request.metadata,
                    start_unix_time=site_start_unix_time,
                )
                if self._program.request.site.segmented:
                    parent = _current_effective_scan_context()
                    self._program.transport.start_segment(
                        parent_point_index=None if parent is None else parent.point_index,
                        start_unix_time=time.time(),
                    )

                result = ScanInspection.empty(
                    self._program.axes,
                    self._program.parameters,
                    self._program.channels,
                    self._program.transport.prefix,
                    initial_annotations=self._program.analysis.initial_annotations(),
                )

                self._executor.install()
                try:
                    if isinstance(self._executor, KernelStreamingExecutor):
                        self._executor.run_to_completion(
                            result,
                            next_batch=self._next_batch,
                            finish_completed_batch=self._finish_completed_batch,
                            should_pause_after_batch=self._should_pause_after_batch,
                            pause_after_batch=self._pause_after_batch,
                            has_more_work=self._has_more_work,
                            next_point_index=lambda: self._program.transport.next_point_index,
                        )
                    else:
                        current_batch = list()
                        batch_offset = 0

                        while True:
                            if batch_offset >= len(current_batch):
                                if not self._program.point_source.has_more_work():
                                    break
                                current_batch = self._next_batch()
                                batch_offset = 0

                            self._fragment.recompute_param_defaults()

                            point_index = self._program.transport.next_point_index
                            batch_result = self._executor.execute_batch(
                                current_batch[batch_offset:],
                                start_point_index=point_index,
                            )
                            completed_batch = list(batch_result.observations)
                            restart_host_context = batch_result.restart_host_context
                            result.runtime_stats.executor_entry_count += (
                                batch_result.executor_entries
                            )
                            if (
                                result.runtime_stats.first_executor_entry_elapsed_s is None
                                and batch_result.executor_entries > 0
                            ):
                                result.runtime_stats.first_executor_entry_elapsed_s = (
                                    batch_result.elapsed_s
                                )
                            result.runtime_stats.total_executor_elapsed_s += (
                                batch_result.elapsed_s
                            )
                            batch_offset += len(completed_batch)

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
                    self._program.transport.finish_segment()
                self._program.transport.flush()
                self._program.analysis.execute_final(
                    result, self._program.transport.site_writer
                )
                self._program.transport.set_completed(True)
                self._program.transport.close()
                self._program.transport.write_completion_preview(preview)
                return result
        finally:
            self._program.transport.unregister_preview(preview)

    def _resolved_run_context(self) -> RunContext:
        if self._run_context is not None:
            return self._run_context

        inherited = current_run_context()
        preview_policy = self._program.request.execution_policy.preview_policy
        if inherited is not None:
            if preview_policy is not None:
                raise ValueError(
                    "Nested scans cannot configure their own preview policy; "
                    "preview cadence is owned by the root run"
                )
            self._run_context = inherited
            return inherited

        run_start_unix_time = time.time()
        preview = None
        if preview_policy is not None:
            preview = PreviewCoordinator(
                self._owner,
                preview_policy,
                run_start_unix_time=run_start_unix_time,
            )
        self._run_context = RunContext(
            run_start_unix_time=run_start_unix_time,
            preview=preview,
        )
        return self._run_context

    def _next_batch(self):
        return self._program.point_source.next_batch()

    def _finish_completed_batch(
        self,
        completed_batch: Sequence[PointObservation],
        result: ScanInspection,
    ) -> None:
        if not completed_batch:
            return

        _publish_completed_batch(
            completed_batch,
            result,
            axes=self._program.axes,
            parameters=self._program.parameters,
            channels=self._program.channels,
            point_source=self._program.point_source,
            analysis=self._program.analysis,
            transport=self._program.transport,
            preview=None if self._run_context is None else self._run_context.preview,
        )

    def _has_more_work(self) -> bool:
        return self._program.point_source.has_more_work()

    def _should_pause_after_batch(self) -> bool:
        if not self._has_more_work():
            return False
        return self._scheduler.check_pause()

    def _pause_after_batch(self) -> None:
        self._scheduler.pause()

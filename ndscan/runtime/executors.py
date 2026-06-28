"""Execution backends for the prepared runtime.

The prepared runtime has two execution shapes:

- ``HostExecutor`` resolves and runs each point from the host, entering kernels point by
  point when needed.
- ``KernelStreamingExecutor`` keeps one resident kernel active and exchanges batches of
  parameter values/results through RPCs.

Both backends publish the same ``PointObservation`` objects. Everything above this
module can therefore ignore whether a scan used host stepping or resident-kernel
streaming.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from artiq.coredevice.exceptions import RTIOUnderflow
from artiq.language import (
    HasEnvironment,
    host_only,
    kernel,
    kernel_from_string,
    portable,
    rpc,
)

from ..define.fragment import ExpFragment, RestartKernelTransitoryError, TransitoryError
from ..define.result_channels import ResultChannel, SingleUseSink
from ..define.utils import is_kernel
from .binding import _resolve_execution_batch, _resolve_execution_point
from .context import (
    ActiveScanContext,
    _KernelParentScanContextProvider,
    _push_kernel_parent_scan_context,
    _push_scan_context,
)
from .program import (
    BoundResultChannel,
    BoundScanAxis,
    BoundScanParameter,
    PointObservation,
    ScanInspection,
    _BoundParameterMapping,
    _ResolvedExecutionPoint,
)

logger = logging.getLogger(__name__)


class _PointResultCollector:
    """Temporarily redirect selected result channels to ``SingleUseSink`` instances.

    User fragments push results to their normal ``ResultChannel`` objects. During point
    execution we swap those sinks so the executor can tell whether every saved channel
    produced exactly one value for the current point.
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


class _ResidentKernelBatchState:
    """Shared host-side state for one resident kernel execution region.

    The resident kernel cannot hold arbitrary Python objects, so the host keeps the
    current resolved batch here. The kernel asks for RPC-friendly parameter arrays,
    runs points, and calls back when each point is complete.
    """

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

        # Return one array per concrete parameter. The generated kernel loop indexes
        # these arrays in lockstep, which avoids an RPC round trip for every point.
        values = tuple([] for _ in self._parameters)
        for point in self._current_chunk:
            for index, value in enumerate(point.rpc_parameter_values):
                values[index].append(value)
        return values

    @host_only
    def _update_host_param_stores(self) -> None:
        if not self._current_chunk:
            return
        # Host-side setup code may inspect parameter values before the next kernel
        # entry. Keep stores reflecting the first not-yet-run point in the chunk.
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
    """Execute points against a fragment from the host side.

    This backend is the general fallback. It may still enter a kernel for a point body,
    but it returns to the host between points/batches, so it does not minimise
    host/kernel crossings as aggressively as ``KernelStreamingExecutor``.
    """

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
    """Execute a strict subset of scans through one resident kernel session.

    The resident kernel calls host RPCs only at chunk boundaries and point-completion
    events. Those RPCs update host-side state and return to the same kernel; they do not
    call back into another kernel.
    """

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
        """Run until the point policy finishes, pausing/restarting host setup as needed."""

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
                    # While inside the resident kernel, nested prepared child scans need
                    # to know which parent point they are attached to. The provider gives
                    # host-side child-scan code a live view of the current point index.
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
    """Configurable resident kernel loop shared by root and child scans.

    ``configure_runner()`` builds a small kernel function specialised to the concrete
    parameter types. This is the part that reduces recompilation pressure: once the
    runner shape is compiled, new point values arrive through RPC arrays instead of by
    regenerating kernel code.
    """

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
        """Generate the resident-kernel inner loop for the current parameter layout."""

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

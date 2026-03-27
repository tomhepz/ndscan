"""Prepared scan handles for root and nested execution."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import reduce
from typing import Any

import numpy as np

from artiq.language import HasEnvironment, host_only, portable, rpc

from .context import (
    RunContext,
    _KernelParentScanContextProvider,
    _current_effective_scan_context,
    current_run_context,
    make_child_scan_site,
    _persistent_kernel_parent_scan_context,
)
from .executors import (
    HostScanProgramBuilder,
    KernelStreamingExecutor,
    _PointResultCollector,
    _ResidentKernelBatchState,
    _ResidentKernelPointRunner,
    _execute_scan_request_inspection,
    _publish_completed_batch,
)
from .program import (
    ScanInspection,
    ScanOutputs,
    _can_use_kernel_streaming_executor,
    _collect_parameter_mappings,
    _fragment_uses_kernel_execution,
    _fragment_tree_needs_param_initialisation,
    _install_scan_axis_stores,
    _build_bound_axes,
    _build_bound_parameters,
)
from ..define.fragment import ExpFragment, Fragment
from ..define.parameters import ParamHandle, ParamStore
from ..define.result_channels import FloatChannel, IntChannel, ResultChannel
from ..scan.mapping import ParameterMapping
from ..scan.request import ScanRequest
from ..schema.scan_site import ScanSite
from ..utils import merge_no_duplicates

__all__ = [
    "PreparedScan",
    "PreparedChildScan",
    "prepare_scan",
    "prepare_child_scan",
    "setattr_prepared_child_scan",
]


class _PreparedScanHandleBase:
    """Shared host-side prepared-scan behavior."""

    def __init__(
        self,
        owner: HasEnvironment,
        fragment: ExpFragment,
        *,
        expose_outputs: Sequence[str] | None = None,
    ):
        self._owner = owner
        self._fragment = fragment
        self._request: ScanRequest | None = None
        self._overrides: dict[str, list[tuple[str, ParamStore]]] | None = None
        self._last_result: ScanInspection | None = None
        self._last_request: ScanRequest | None = None
        self._exposed_output_channel_specs = _resolve_declared_scan_output_channels(
            fragment,
            expose_outputs=expose_outputs,
        )

    @host_only
    def configure(
        self,
        request: ScanRequest,
        *,
        overrides: dict[str, list[tuple[str, ParamStore]]] | None = None,
    ) -> None:
        self._request = request
        self._overrides = overrides
        self._after_configure()

    @host_only
    def _after_configure(self) -> None:
        pass

    @host_only
    def _prepare_execution_request(self, request: ScanRequest) -> ScanRequest:
        return request

    @host_only
    def _execute_inspection_request(self, request: ScanRequest) -> ScanInspection:
        raise NotImplementedError

    @host_only
    def _not_configured_message(self) -> str:
        return "Prepared scan has not been configured yet"

    @host_only
    def _not_executed_message(self) -> str:
        return "Prepared scan has not been executed yet"

    @host_only
    def execute(self) -> ScanOutputs:
        if self._request is None:
            raise RuntimeError(self._not_configured_message())
        self._last_request = self._prepare_execution_request(
            _clone_scan_request_for_execution(self._request)
        )
        self._last_result = self._execute_inspection_request(self._last_request)
        return self.outputs()

    @host_only
    def inspect(self) -> ScanInspection:
        if self._last_result is None:
            raise RuntimeError(self._not_executed_message())
        return self._last_result

    @host_only
    def outputs(self) -> ScanOutputs:
        return _scan_outputs_from_inspection(
            self.inspect(),
            explicit_output_channels=self._exposed_output_channel_specs,
        )

    @host_only
    def get_outputs(self) -> tuple[Any, ...]:
        if not self._exposed_output_channel_specs:
            raise AttributeError(
                f"{type(self).__name__} does not declare fixed tuple outputs"
            )
        return self.outputs().as_tuple()


class PreparedScan(_PreparedScanHandleBase):
    """Prepared root scan execution."""

    def __init__(
        self,
        owner: HasEnvironment,
        fragment: ExpFragment,
        request: ScanRequest | None = None,
        *,
        overrides: dict[str, list[tuple[str, ParamStore]]] | None = None,
        run_context: RunContext | None = None,
        max_rtio_underflow_retries: int = 3,
        max_transitory_error_retries: int = 10,
        expose_outputs: Sequence[str] | None = None,
    ):
        super().__init__(
            owner,
            fragment,
            expose_outputs=expose_outputs,
        )
        self._run_context = run_context
        self._max_rtio_underflow_retries = max_rtio_underflow_retries
        self._max_transitory_error_retries = max_transitory_error_retries
        if request is not None:
            self.configure(request, overrides=overrides)

    @host_only
    def _execute_inspection_request(self, request: ScanRequest) -> ScanInspection:
        return _execute_scan_request_inspection(
            self._owner,
            self._fragment,
            request,
            overrides=self._overrides,
            run_context=self._run_context,
            max_rtio_underflow_retries=self._max_rtio_underflow_retries,
            max_transitory_error_retries=self._max_transitory_error_retries,
        )


def prepare_scan(
    owner: HasEnvironment,
    fragment: ExpFragment,
    *,
    request: ScanRequest | None = None,
    overrides: dict[str, list[tuple[str, ParamStore]]] | None = None,
    run_context: RunContext | None = None,
    max_rtio_underflow_retries: int = 3,
    max_transitory_error_retries: int = 10,
    expose_outputs: Sequence[str] | None = None,
) -> PreparedScan:
    return PreparedScan(
        owner,
        fragment,
        request,
        overrides=overrides,
        run_context=run_context,
        max_rtio_underflow_retries=max_rtio_underflow_retries,
        max_transitory_error_retries=max_transitory_error_retries,
        expose_outputs=expose_outputs,
    )


def _prepare_child_scan_request(
    request: ScanRequest,
    *,
    name: str,
    segmented: bool,
    extra_metadata: Mapping[str, Any] | None,
) -> ScanRequest:
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
    return request.with_site(child_site)


def _clone_scan_request_for_execution(request: ScanRequest) -> ScanRequest:
    return ScanRequest(
        axes=request.axes,
        point_policy=deepcopy(request.point_policy),
        parameter_mappings=request.parameter_mappings,
        fixed_pseudoparams=request.fixed_pseudoparams,
        site=request.site,
        metadata=deepcopy(request.metadata),
        execution_policy=request.execution_policy,
    )


def _fragment_subtree_contains(root: Fragment, candidate: Fragment) -> bool:
    if root is candidate:
        return True
    for child in root._subfragments:
        if _fragment_subtree_contains(child, candidate):
            return True
    return False


def _mapping_targets_belong_to_fragment(
    mapping: ParameterMapping, fragment: ExpFragment
) -> bool:
    return all(_fragment_subtree_contains(fragment, target.owner) for target in mapping.targets)


def _collect_prepared_child_owner_mappings(
    owner: HasEnvironment, fragment: ExpFragment
) -> tuple[ParameterMapping, ...]:
    if not isinstance(owner, Fragment):
        return ()
    return tuple(
        mapping
        for mapping in owner._parameter_mappings
        if _mapping_targets_belong_to_fragment(mapping, fragment)
    )


def _merge_child_inherited_parameter_mappings(
    request: ScanRequest,
    inherited_parameter_mappings: Sequence[ParameterMapping],
) -> ScanRequest:
    if not inherited_parameter_mappings:
        return request
    return ScanRequest(
        axes=request.axes,
        point_policy=request.point_policy,
        site=request.site,
        metadata=request.metadata,
        execution_policy=request.execution_policy,
        parameter_mappings=tuple(inherited_parameter_mappings)
        + tuple(request.parameter_mappings),
        fixed_pseudoparams=request.fixed_pseudoparams,
    )


def _auto_detach_prepared_child_fragment(
    owner: HasEnvironment, fragment: ExpFragment
) -> None:
    if not isinstance(owner, Fragment):
        return
    if fragment not in owner._subfragments:
        return
    if fragment in owner._detached_subfragments:
        return
    if owner._building:
        owner.detach_fragment(fragment)
        return
    raise ValueError(
        "Prepared child scan targets that are direct subfragments must be detached "
        "during build_fragment(); use detached=True on setattr_fragment() or "
        "setattr_prepared_child_scan()"
    )


def _collect_default_analysis_result_channels(
    fragment: ExpFragment,
) -> dict[str, ResultChannel]:
    return reduce(
        lambda x, y: merge_no_duplicates(x, y, kind="analysis result"),
        (analysis.get_analysis_results() for analysis in fragment.get_default_analyses()),
        {},
    )


def _prepared_child_fixed_output_rpc_type(channel: ResultChannel):
    if isinstance(channel, FloatChannel):
        return float
    if isinstance(channel, IntChannel):
        return np.int32
    raise NotImplementedError(
        "Prepared child fixed outputs currently support only FloatChannel and "
        f"IntChannel analysis results, not {type(channel).__name__}"
    )


def _prepared_child_fixed_output_coercer(channel: ResultChannel):
    if isinstance(channel, FloatChannel):
        return float
    if isinstance(channel, IntChannel):
        return np.int32
    raise NotImplementedError(
        "Prepared child fixed outputs currently support only FloatChannel and "
        f"IntChannel analysis results, not {type(channel).__name__}"
    )


def _resolve_declared_scan_output_channels(
    fragment: ExpFragment,
    *,
    expose_outputs: Sequence[str] | None,
) -> dict[str, ResultChannel]:
    all_channels = _collect_default_analysis_result_channels(fragment)

    if expose_outputs is None:
        return {}

    if not expose_outputs:
        raise ValueError("expose_outputs must contain at least one output name")

    selected = dict[str, ResultChannel]()
    seen = set[str]()
    for result_name in expose_outputs:
        if result_name in seen:
            raise ValueError(
                f"Prepared child fixed output {result_name!r} was requested more than once"
            )
        seen.add(result_name)
        try:
            channel = all_channels[result_name]
        except KeyError as exc:
            raise ValueError(
                f"Prepared child fixed output {result_name!r} is not declared by any "
                f"default analysis on {type(fragment).__name__}"
            ) from exc
        _prepared_child_fixed_output_rpc_type(channel)
        selected[result_name] = channel
    return selected


def _scan_outputs_from_inspection(
    inspection: ScanInspection,
    *,
    explicit_output_channels: Mapping[str, ResultChannel],
) -> ScanOutputs:
    if explicit_output_channels:
        names = tuple(explicit_output_channels.keys())
        return ScanOutputs(
            OrderedDict((name, inspection.analysis_results[name]) for name in names)
        )
    return inspection.outputs()


def _make_prepared_child_get_outputs_methods(
    output_channels: Mapping[str, ResultChannel],
):
    if not output_channels:
        return {}

    output_items = tuple(output_channels.items())
    return_type = tuple.__class_getitem__(
        tuple(
            _prepared_child_fixed_output_rpc_type(channel)
            for _, channel in output_items
        )
    )
    coercers = tuple(
        _prepared_child_fixed_output_coercer(channel) for _, channel in output_items
    )

    def _get_outputs_value(self):
        return tuple(
            coercer(self._get_exposed_output_value(result_name))
            for (result_name, _), coercer in zip(output_items, coercers, strict=True)
        )

    _get_outputs_value.__annotations__ = {"return": return_type}
    _get_outputs_value = rpc(_get_outputs_value)

    def get_outputs(self):
        return self._get_outputs_value()

    get_outputs.__annotations__ = {"return": return_type}
    get_outputs = portable(get_outputs)

    return {
        "_get_outputs_value": _get_outputs_value,
        "get_outputs": get_outputs,
    }


class _PreparedChildKernelAcquireSession:
    """Host-side lifecycle owner for one active prepared child kernel acquire."""

    def __init__(
        self,
        owner: HasEnvironment,
        fragment: ExpFragment,
        request: ScanRequest,
    ):
        self._owner = owner
        self._fragment = fragment
        self._program = HostScanProgramBuilder(owner).build(fragment, request)
        self._run_context = current_run_context()
        if self._run_context is None:
            raise RuntimeError(
                "PreparedChildScan.acquire() can only be used while a parent scan run "
                "context is active"
            )
        self._preview = self._run_context.preview
        self._result: ScanInspection | None = None
        self._collector: _PointResultCollector | None = None
        self._batch_state: _ResidentKernelBatchState | None = None
        self._parent_provider: _KernelParentScanContextProvider | None = None
        self._host_setup_active = False

    @property
    def result(self) -> ScanInspection | None:
        return self._result

    def start(self) -> None:
        self._program.transport.register_preview(self._preview)

        try:
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

            self._result = ScanInspection.empty(
                self._program.axes,
                self._program.parameters,
                self._program.channels,
                self._program.transport.prefix,
                initial_annotations=self._program.analysis.initial_annotations(),
            )
            self._collector = _PointResultCollector(self._program.channels)
            self._collector.install()
            self._batch_state = _ResidentKernelBatchState(
                self._program.axes,
                self._program.parameters,
                self._program.parameter_mappings,
                self._collector,
            )

            self._fragment.recompute_param_defaults()
            self._fragment.host_setup()
            self._host_setup_active = True

            self._parent_provider = _KernelParentScanContextProvider(
                self._program.request.site.path,
                lambda: self._batch_state.current_next_point_index,
            )
            _persistent_kernel_parent_scan_context.append(self._parent_provider)
        except BaseException:
            self.cleanup(completed=False)
            raise

    def record_executor_entry(self) -> None:
        if self._result is not None:
            self._result.runtime_stats.executor_entry_count += 1

    def restart_host_context(self) -> None:
        if self._host_setup_active:
            self._fragment.host_cleanup()
        self._fragment.recompute_param_defaults()
        self._fragment.host_setup()
        self._host_setup_active = True

    def pause_after_batch(self) -> None:
        scheduler = self._owner.get_device("scheduler")
        scheduler.pause()

    def get_param_values_chunk(self):
        assert self._batch_state is not None
        return self._batch_state.get_param_values_chunk(
            next_batch=self._program.point_source.next_batch,
            next_point_index=lambda: self._program.transport.next_point_index,
        )

    def retry_point(self) -> None:
        if self._batch_state is not None:
            self._batch_state.retry_point()

    def point_completed(self) -> None:
        if self._batch_state is None:
            raise RuntimeError("Prepared child kernel batch state is not installed")
        self._batch_state.point_completed()

    def finish_chunk(self):
        self._finish_kernel_completed_batch()
        if not self._program.point_source.has_more_work():
            return KernelStreamingExecutor._STATUS_COMPLETE
        scheduler = self._owner.get_device("scheduler")
        if scheduler.check_pause():
            return KernelStreamingExecutor._STATUS_PAUSE
        return KernelStreamingExecutor._STATUS_PROCEED

    def finish_chunk_after_restart(self):
        self._finish_kernel_completed_batch()
        return KernelStreamingExecutor._STATUS_RESTART_HOST_CONTEXT

    def complete(self) -> ScanInspection | None:
        self.cleanup(completed=True)
        return self._result

    def abort(self) -> None:
        self.cleanup(completed=False)

    def cleanup(self, *, completed: bool) -> None:
        provider = self._parent_provider
        if provider is not None:
            if (
                not _persistent_kernel_parent_scan_context
                or _persistent_kernel_parent_scan_context[-1] is not provider
            ):
                raise RuntimeError(
                    "Prepared child scan kernel parent context stack is out of sync"
                )
            _persistent_kernel_parent_scan_context.pop()
            self._parent_provider = None

        if self._host_setup_active:
            self._fragment.host_cleanup()
            self._host_setup_active = False

        if self._collector is not None:
            self._collector.remove()
            self._collector = None

        try:
            if completed and self._result is not None:
                if self._program.request.site.segmented:
                    self._program.transport.finish_segment()
                self._program.transport.flush()
                self._program.analysis.execute_final(
                    self._result, self._program.transport.site_writer
                )
                self._program.transport.set_completed(True)
            self._program.transport.close()
        finally:
            self._program.transport.unregister_preview(self._preview)
            if self._batch_state is not None:
                self._batch_state.reset()
            self._batch_state = None

    def _finish_kernel_completed_batch(self) -> None:
        assert self._result is not None
        if self._batch_state is None:
            return

        completed_batch = self._batch_state.take_completed_batch()
        _publish_completed_batch(
            completed_batch,
            self._result,
            axes=self._program.axes,
            parameters=self._program.parameters,
            channels=self._program.channels,
            point_source=self._program.point_source,
            analysis=self._program.analysis,
            transport=self._program.transport,
            preview=self._preview,
        )


class PreparedChildScan(_PreparedScanHandleBase):
    """Prepared child-scan handle for the prepared runtime."""

    def __init__(
        self,
        owner: HasEnvironment,
        fragment: ExpFragment,
        *,
        name: str,
        segmented: bool = True,
        extra_metadata: Mapping[str, Any] | None = None,
        max_rtio_underflow_retries: int = 3,
        max_transitory_error_retries: int = 10,
        expose_outputs: Sequence[str] | None = None,
    ):
        self._name = name
        self._segmented = segmented
        self._extra_metadata = {} if extra_metadata is None else dict(extra_metadata)
        self._max_rtio_underflow_retries = max_rtio_underflow_retries
        self._max_transitory_error_retries = max_transitory_error_retries
        super().__init__(
            owner,
            fragment,
            expose_outputs=expose_outputs,
        )
        self._kernel_runner = None
        self._kernel_runner_ready = False
        self._kernel_support_error: str | None = None
        self._kernel_axis_bindings: list[Any] = []
        self._kernel_session: _PreparedChildKernelAcquireSession | None = None

        if _fragment_uses_kernel_execution(fragment):
            runner_class = type(
                f"_PreparedChildKernelRunner_{id(self)}",
                (_ResidentKernelPointRunner,),
                {},
            )
            self._kernel_runner = runner_class(self._fragment, self._fragment, self)
        else:
            self.acquire = self._acquire_host

    @host_only
    def _after_configure(self) -> None:
        self._refresh_kernel_acquire_runner()

    @host_only
    def _prepare_execution_request(self, request: ScanRequest) -> ScanRequest:
        request = _merge_child_inherited_parameter_mappings(
            request, self._current_inherited_parameter_mappings()
        )
        return _prepare_child_scan_request(
            request,
            name=self._name,
            segmented=self._segmented,
            extra_metadata=self._extra_metadata,
        )

    @host_only
    def _execute_inspection_request(self, request: ScanRequest) -> ScanInspection:
        return _execute_scan_request_inspection(
            self._owner,
            self._fragment,
            request,
            overrides=self._overrides,
            run_context=None,
            max_rtio_underflow_retries=self._max_rtio_underflow_retries,
            max_transitory_error_retries=self._max_transitory_error_retries,
        )

    @host_only
    def _not_configured_message(self) -> str:
        return f"Prepared child scan '{self._name}' has not been configured yet"

    @host_only
    def _not_executed_message(self) -> str:
        return f"Prepared child scan '{self._name}' has not been executed yet"

    @host_only
    def prime(self) -> None:
        self._fragment.host_setup()

    @portable
    def acquire(self) -> None:
        self._acquire_kernel()

    @portable
    def get_outputs(self):
        raise AttributeError(
            f"Prepared child scan '{self._name}' does not declare fixed tuple outputs"
        )

    @rpc
    def _acquire_host(self) -> None:
        if _fragment_uses_kernel_execution(self._fragment):
            raise NotImplementedError(
                "PreparedChildScan.acquire() should use the dedicated prepared kernel "
                "backend for child fragments using @kernel"
            )
        self.execute()

    @portable
    def _acquire_kernel(self) -> None:
        self._begin_kernel_acquire()
        try:
            while True:
                self._record_kernel_executor_entry()
                status = self._kernel_runner.acquire()
                if status == _ResidentKernelPointRunner._STATUS_COMPLETE:
                    self._complete_kernel_acquire()
                    return
                if status == _ResidentKernelPointRunner._STATUS_RESTART_HOST_CONTEXT:
                    self._restart_kernel_host_context()
                    continue
                if status == _ResidentKernelPointRunner._STATUS_PAUSE:
                    self._pause_kernel_after_batch()
                    continue
                raise RuntimeError("Unexpected prepared child kernel runner status")
        except Exception:
            self._abort_kernel_acquire()
            raise

    @host_only
    def _refresh_kernel_acquire_runner(self) -> None:
        if self._kernel_axis_bindings:
            for binding in self._kernel_axis_bindings:
                binding.restore()
            self._kernel_axis_bindings.clear()

        self._kernel_runner_ready = False
        self._kernel_support_error = None

        if self._request is None or self._kernel_runner is None:
            return

        if _fragment_tree_needs_param_initialisation(self._fragment):
            self._fragment.init_params(
                overrides={} if self._overrides is None else self._overrides
            )

        effective_request = _merge_child_inherited_parameter_mappings(
            self._request, self._current_inherited_parameter_mappings()
        )
        axis_handles = [
            axis for axis in effective_request.axes if isinstance(axis, ParamHandle)
        ]
        axis_bindings = _install_scan_axis_stores(axis_handles)
        try:
            axes = _build_bound_axes(effective_request.axes)
            parameter_mappings = _collect_parameter_mappings(
                self._fragment, effective_request, axes
            )
            parameters = _build_bound_parameters(axes, parameter_mappings)
            if not _can_use_kernel_streaming_executor(
                self._fragment, axes, parameters
            ):
                self._kernel_support_error = (
                    "PreparedChildScan.acquire() for child fragments using @kernel "
                    "currently requires at least one concrete fragment parameter to "
                    "vary point-to-point; pure pseudoparam scans with no mapped "
                    "parameter targets are not yet supported"
                )
                for binding in axis_bindings:
                    binding.restore()
                return

            self._kernel_runner.configure_runner(
                parameters,
                max_rtio_underflow_retries=self._max_rtio_underflow_retries,
                max_transitory_error_retries=self._max_transitory_error_retries,
            )
            self._kernel_runner_ready = True
            self._kernel_axis_bindings = axis_bindings
        except BaseException:
            for binding in axis_bindings:
                binding.restore()
            raise

    @rpc
    def _begin_kernel_acquire(self) -> None:
        if self._request is None:
            raise RuntimeError(
                f"Prepared child scan '{self._name}' has not been configured yet"
            )
        if not self._kernel_runner_ready or self._kernel_runner is None:
            raise NotImplementedError(
                self._kernel_support_error
                or "PreparedChildScan.acquire() does not have a kernel-capable "
                "prepared runner for this child scan"
            )
        if self._kernel_session is not None:
            raise RuntimeError("Prepared child scan kernel acquire is already active")

        request = self._prepare_execution_request(
            _clone_scan_request_for_execution(self._request),
        )
        self._kernel_session = _PreparedChildKernelAcquireSession(
            self._owner,
            self._fragment,
            request,
        )
        try:
            self._kernel_session.start()
        except BaseException:
            self._kernel_session = None
            raise

    @host_only
    def _current_inherited_parameter_mappings(self) -> tuple[ParameterMapping, ...]:
        return _collect_prepared_child_owner_mappings(self._owner, self._fragment)

    @rpc(flags={"async"})
    def _record_kernel_executor_entry(self) -> None:
        if self._kernel_session is not None:
            self._kernel_session.record_executor_entry()

    @rpc
    def _restart_kernel_host_context(self) -> None:
        assert self._kernel_session is not None
        self._kernel_session.restart_host_context()

    @rpc
    def _pause_kernel_after_batch(self) -> None:
        assert self._kernel_session is not None
        self._kernel_session.pause_after_batch()

    @rpc
    def _complete_kernel_acquire(self) -> None:
        assert self._kernel_session is not None
        session = self._kernel_session
        self._kernel_session = None
        self._last_result = session.complete()

    @rpc
    def _abort_kernel_acquire(self) -> None:
        if self._kernel_session is not None:
            session = self._kernel_session
            self._kernel_session = None
            session.abort()

    @host_only
    def _get_param_values_chunk(self):
        assert self._kernel_session is not None
        return self._kernel_session.get_param_values_chunk()

    def _retry_point(self):
        if self._kernel_session is not None:
            self._kernel_session.retry_point()

    def _point_completed(self):
        assert self._kernel_session is not None
        self._kernel_session.point_completed()

    @host_only
    def _finish_chunk(self):
        assert self._kernel_session is not None
        return self._kernel_session.finish_chunk()

    @host_only
    def _finish_chunk_after_restart(self):
        assert self._kernel_session is not None
        return self._kernel_session.finish_chunk_after_restart()

    @host_only
    def _get_exposed_output_value(self, result_name: str) -> Any:
        if result_name not in self._exposed_output_channel_specs:
            raise AttributeError(
                f"Prepared child scan '{self._name}' does not expose output "
                f"{result_name!r}"
            )
        if self._last_result is None:
            raise RuntimeError(
                f"Prepared child scan '{self._name}' has not been executed yet"
            )
        try:
            return self._last_result.analysis_results[result_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Prepared child scan '{self._name}' did not produce output "
                f"{result_name!r} in its most recent run"
            ) from exc


def prepare_child_scan(
    owner: HasEnvironment,
    fragment: ExpFragment,
    *,
    name: str,
    segmented: bool = True,
    extra_metadata: Mapping[str, Any] | None = None,
    max_rtio_underflow_retries: int = 3,
    max_transitory_error_retries: int = 10,
    expose_outputs: Sequence[str] | None = None,
) -> PreparedChildScan:
    _auto_detach_prepared_child_fragment(owner, fragment)
    prepared_class_namespace = _make_prepared_child_get_outputs_methods(
        _resolve_declared_scan_output_channels(
            fragment,
            expose_outputs=expose_outputs,
        )
    )
    prepared_class = type(
        f"_PreparedChildScan_{id(owner)}_{name.replace('/', '_')}",
        (PreparedChildScan,),
        prepared_class_namespace,
    )
    return prepared_class(
        owner,
        fragment,
        name=name,
        segmented=segmented,
        extra_metadata=extra_metadata,
        max_rtio_underflow_retries=max_rtio_underflow_retries,
        max_transitory_error_retries=max_transitory_error_retries,
        expose_outputs=expose_outputs,
    )


def setattr_prepared_child_scan(
    owner: Fragment,
    name: str,
    fragment_class: type[ExpFragment],
    *args,
    scan_name: str | None = None,
    segmented: bool = True,
    extra_metadata: Mapping[str, Any] | None = None,
    max_rtio_underflow_retries: int = 3,
    max_transitory_error_retries: int = 10,
    expose_outputs: Sequence[str] | None = None,
    **kwargs,
) -> PreparedChildScan:
    fragment = owner.setattr_fragment(name, fragment_class, *args, detached=True, **kwargs)
    return prepare_child_scan(
        owner,
        fragment,
        name=name if scan_name is None else scan_name,
        segmented=segmented,
        extra_metadata=extra_metadata,
        max_rtio_underflow_retries=max_rtio_underflow_retries,
        max_transitory_error_retries=max_transitory_error_retries,
        expose_outputs=expose_outputs,
    )

"""High-level prepared scan lifecycle.

Start here when reading the runtime path. ``ScanProgramBuilder`` binds a
``ScanRequest`` to a concrete fragment tree, and ``ScanProgramRunner`` executes that
bound program by choosing an executor, publishing batches, and closing the scan site.

The lower-level mechanics live elsewhere:

- :mod:`ndscan.runtime.binding` resolves axes, mappings, and point values.
- :mod:`ndscan.runtime.executors` contains the host and resident-kernel backends.
- :mod:`ndscan.runtime.program` defines the bound data model and batch publication.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from artiq.language import HasEnvironment

from ..define.fragment import ExpFragment
from ..define.result_channels import ResultChannel
from ..define.utils import is_kernel
from ..scan.request import ScanRequest
from .analysis import ScanAnalysisEngine
from .binding import (
    _build_bound_axes,
    _build_bound_parameters,
    _can_use_kernel_streaming_executor,
    _collect_parameter_mappings,
    _fragment_tree_needs_param_initialisation,
    _install_varying_parameter_stores,
    _missing_kernel_core_devices,
)
from .context import (
    PreviewCoordinator,
    RunContext,
    _current_effective_scan_context,
    _push_run_context,
    current_run_context,
)
from .executors import HostExecutor, KernelStreamingExecutor
from .persistence import ScanSiteDatasetWriter
from .program import (
    BoundResultChannel,
    PointObservation,
    ScanInspection,
    ScanProgram,
    _publish_completed_batch,
)

__all__ = [
    "ScanProgramBuilder",
    "ScanProgramRunner",
]


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
    """Execute one prepared request and return its host-side inspection object."""

    # This wrapper is the last bit of setup before the real scan loop. Parameter
    # defaults/overrides are installed if needed, then point-varying parameters are
    # moved onto temporary stores so the scan can mutate them without corrupting the
    # fragment's default stores.
    if _fragment_tree_needs_param_initialisation(fragment):
        fragment.init_params(overrides={} if overrides is None else overrides)
    varying_param_bindings = _install_varying_parameter_stores(fragment, request)
    try:
        # After this point the request is bound to this concrete fragment tree. The
        # runner below is where point batches actually execute.
        builder = ScanProgramBuilder(owner)
        program = builder.build(fragment, request)
        runner = ScanProgramRunner(
            owner,
            program,
            run_context=run_context,
            max_rtio_underflow_retries=max_rtio_underflow_retries,
            max_transitory_error_retries=max_transitory_error_retries,
        )
        return runner.run()
    finally:
        # Whatever happened during execution, put the fragment handles back onto their
        # original stores. Without this, a later scan or ordinary experiment run would
        # inherit the last scanned point as its "default" value.
        for binding in varying_param_bindings:
            binding.restore()


class ScanProgramBuilder:
    """Bind a code-first ``ScanRequest`` to a concrete fragment instance.

    The builder validates all cross-object relationships before execution starts:
    policy dimensionality, result-channel collection, parameter mappings, kernel device
    availability, and analysis binding.
    """

    def __init__(self, owner: HasEnvironment):
        self._owner = owner

    def build(self, fragment: ExpFragment, request: ScanRequest) -> ScanProgram:
        if request.point_policy.axis_count != len(request.axes):
            raise ValueError(
                "Point policy dimensionality does not match the number of "
                "requested axes"
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
        analysis_engine = ScanAnalysisEngine.build(fragment, axes, channels)
        return ScanProgram(
            fragment,
            request,
            axes,
            parameters,
            channels,
            parameter_mappings,
            site_writer,
            analysis_engine,
        )


class ScanProgramRunner:
    """Own the prepared scan loop for one root or child scan site.

    Local vocabulary while reading this class:

    - ``program`` is the already-bound scan plan: fragment, axes, result channels,
      point policy, analysis engine, and dataset writer.
    - ``transport`` is the program's small persistence adapter. The runner says
      "append these observations" or "mark completed"; the transport/writer know the
      actual dataset keys.
    - ``run_context`` is root-run shared state. It carries preview ownership so a root
      scan and all child scans write one coherent preview file.
    - ``preview`` is the optional HDF5 snapshot coordinator used by live plotting while
      the scan is still running.
    """

    def __init__(
        self,
        owner: HasEnvironment,
        program: ScanProgram,
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
        """Execute the program and publish its scan-site datasets."""

        # 1. Establish the root-run context.
        #
        # A root scan creates a RunContext; child scans inherit it. The important thing
        # inside it is the preview coordinator. Registering this site's transport means
        # "when a preview snapshot is due, flush this site's datasets before the HDF5 is
        # copied". If preview is None, the transport methods are no-ops for previewing.
        run_context = self._resolved_run_context()
        preview = run_context.preview
        self._program.transport.register_preview(preview)

        try:
            with _push_run_context(run_context):
                # 2. Prepare the fragment and write the once-per-site metadata before
                # any points appear. If a run dies halfway through, readers should still
                # know what was attempted and where this site lived in the scan tree.
                self._fragment.prepare()
                site_start_unix_time = time.time()
                self._program.transport.publish_metadata(
                    self._program.metadata(),
                    extra_metadata=self._program.request.metadata,
                    start_unix_time=site_start_unix_time,
                )
                if self._program.request.site.segmented:
                    # Child scans reuse one site prefix across parent points. A segment
                    # says "the following flat child points came from this parent
                    # point".
                    parent = _current_effective_scan_context()
                    parent_point_index = (
                        None if parent is None else parent.point_index
                    )
                    self._program.transport.start_segment(
                        parent_point_index=parent_point_index,
                        start_unix_time=time.time(),
                    )

                # 3. Keep an in-memory mirror of the data we write. Analysis, policy
                # feedback, and callers inspecting the result all read from this object,
                # so the persisted and in-memory views are updated at the same batch
                # boundary.
                result = ScanInspection.empty(
                    self._program.axes,
                    self._program.parameters,
                    self._program.channels,
                    self._program.transport.prefix,
                    initial_annotations=self._program.analysis.initial_annotations(),
                )

                # 4. Install result-channel capture and execute point batches. The two
                # executor types differ in how they cross the host/kernel boundary, but
                # both report completed points back through _finish_completed_batch().
                self._executor.install()
                try:
                    if isinstance(self._executor, KernelStreamingExecutor):
                        # The resident-kernel executor owns its own inner loop. We pass
                        # in small callbacks so it can ask for points, publish batches,
                        # and honour scheduler pauses without duplicating runtime state.
                        self._executor.run_to_completion(
                            result,
                            next_batch=self._next_batch,
                            finish_completed_batch=self._finish_completed_batch,
                            should_pause_after_batch=self._should_pause_after_batch,
                            pause_after_batch=self._pause_after_batch,
                            has_more_work=self._has_more_work,
                            next_point_index=(
                                lambda: self._program.transport.next_point_index
                            ),
                        )
                    else:
                        # Host execution is easier to read because each pass through
                        # this loop enters the fragment for the remaining suffix of the
                        # current policy batch, then returns completed observations.
                        current_batch = list()
                        batch_offset = 0

                        while True:
                            if batch_offset >= len(current_batch):
                                if not self._program.point_source.has_more_work():
                                    break
                                # The point policy decides where to go next. From here
                                # down, the runtime just tries to execute that batch.
                                current_batch = self._next_batch()
                                batch_offset = 0

                            # Defaults can depend on datasets or earlier host-side
                            # changes. Recompute them before setup so host code sees the
                            # same starting state a normal ARTIQ run would see.
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
                                result.runtime_stats.first_executor_entry_elapsed_s
                                is None
                                and batch_result.executor_entries > 0
                            ):
                                result.runtime_stats.first_executor_entry_elapsed_s = (
                                    batch_result.elapsed_s
                                )
                            result.runtime_stats.total_executor_elapsed_s += (
                                batch_result.elapsed_s
                            )
                            batch_offset += len(completed_batch)

                            # This is the batch boundary: write datasets, update the
                            # inspection object, run online analysis, and give feedback
                            # to the point policy before asking for more points.
                            self._finish_completed_batch(completed_batch, result)

                            if restart_host_context:
                                # A transitory error asked for host setup to be rebuilt.
                                # Keep any uncompleted suffix of the batch and re-enter
                                # the executor after recomputing defaults/setup.
                                continue

                            # A normal host-executor pass consumed the current batch.
                            # Only now do we honour scheduler pause requests, so the
                            # dataset is left at a clean batch boundary.
                            current_batch = []
                            batch_offset = 0
                            if self._should_pause_after_batch():
                                self._pause_after_batch()
                finally:
                    # Restore result-channel sinks even if the fragment or kernel
                    # raises. Leaving temporary sinks installed makes later runs very
                    # confusing to debug.
                    self._executor.remove()

                # 5. Finish the site. The raw points are already written; this section
                # closes the current segment, runs final analysis once, marks the site
                # completed, and gives preview readers a final consistent snapshot.
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
            # Nested scans and preview writes share the same coordinator. Unregister
            # even on failure so the next scan does not try to flush a dead writer.
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

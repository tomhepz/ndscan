"""A minimal host-directed runtime for ndscan fragments.

This module is intentionally narrower than the legacy runtime:

- the scan structure and point selection live on the host,
- code-first requests rather than dashboard parsing,
- one flat scan-site dataset layout,
- point policies as the only point-selection abstraction.

The goal is not to replace the old runtime in one shot. The goal is to establish a
small execution core that is easy to read, extend, and eventually reuse from both
top-level scans and subscans.

The runtime remains host-directed even when an executor enters the core:

- the host still owns the scan request and point policy,
- the host still owns batch boundaries, persistence, and analyses,
- a kernel executor is only an implementation detail for how a chosen point body runs.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy, deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from typing import Any
from weakref import WeakSet

import h5py
import numpy as np
from sipyco import pyon

from artiq import __version__ as artiq_version
from artiq.coredevice.exceptions import RTIOUnderflow
from artiq.language import (
    EnvExperiment,
    HasEnvironment,
    PYONValue,
    host_only,
    kernel,
    kernel_from_string,
    portable,
    rpc,
)

from ._host_analysis import HostScanAnalysisEngine
from .fragment import ExpFragment, Fragment, RestartKernelTransitoryError, TransitoryError
from .parameters import ParamHandle, ParamStore
from .point_policy import (
    BasePoint,
    BatchFeedback,
    CartesianPointPolicy,
    ExplicitPointPolicy,
    PointPolicy,
    ProductPointPolicy,
    SinglePointPolicy,
    ZipPointPolicy,
)
from .result_channels import FloatChannel, IntChannel, ResultChannel, SingleUseSink
from .scan_mapping import FixedPseudoparam, ParameterMapping, ScanVariable
from .scan_site import ScanSite, ScanSiteDatasetWriter
from .utils import is_kernel
from ..utils import PARAMS_ARG_KEY, merge_no_duplicates

__all__ = [
    "ExecutionPolicy",
    "PreviewPolicy",
    "HostScanSchemaError",
    "HostScanSpec",
    "compile_host_scan_spec",
    "compile_host_scan_schema",
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
    "PreparedChildScan",
    "prepare_child_scan",
    "setattr_prepared_child_scan",
    "run_host_scan",
    "run_subscan",
    "make_fragment_host_scan_exp",
    "make_fragment_host_dashboard_scan_exp",
]

# Hack: Only export the internal base experiment classes for Sphinx/autodoc.
# Otherwise ARTIQ's explorer may try to instantiate them for modules using
# ``from ndscan.experiment import *``, which fails because the base classes expect
# adapter-provided build arguments.
if "sphinx" in sys.modules:
    __all__.append("HostScanExperiment")
    __all__.append("HostDashboardScanExperiment")

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


@dataclass(frozen=True)
class _KernelParentScanContextProvider:
    """Host-side view of a point currently executing inside a resident kernel."""

    site_path: tuple[str, ...]
    point_index_getter: Any

    def snapshot(self) -> ActiveScanContext:
        return ActiveScanContext(self.site_path, int(self.point_index_getter()))


_active_scan_context: ContextVar[tuple[ActiveScanContext, ...]] = ContextVar(
    "_active_scan_context", default=()
)
_active_kernel_parent_scan_context: ContextVar[
    tuple[_KernelParentScanContextProvider, ...]
] = ContextVar("_active_kernel_parent_scan_context", default=())
_persistent_kernel_parent_scan_context: list[_KernelParentScanContextProvider] = []


@dataclass(frozen=True)
class PreviewPolicy:
    """Configuration for periodic preview HDF5 snapshots.

    Preview files are rewritten only on completed batch boundaries. The cadence is
    still time-based, but delaying the decision until a safe batch boundary keeps the
    snapshot self-consistent even for nested scans and future buffered writers.
    """

    path: str | None = None
    min_interval_s: float = 120.0
    write_on_completion: bool = False
    remove_on_completion: bool = True

    def __post_init__(self) -> None:
        if self.min_interval_s < 0.0:
            raise ValueError("min_interval_s must be non-negative")

    def resolve_path(self, owner: HasEnvironment) -> str:
        """Return the preview file path for this root run.

        When no explicit path is given, the preview file mirrors ARTIQ's canonical
        RID/class-name pattern and simply inserts `.preview` before the `.h5`
        suffix.
        """

        if self.path is not None:
            return self.path
        scheduler = owner.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        return f"{rid:09d}-{owner.__class__.__name__}.preview.h5"


class PreviewCoordinator:
    """Root-scoped coordination for preview HDF5 snapshots.

    This object is intentionally narrow. It does not own execution. It only owns:

    - the shared preview cadence for one root run,
    - the preview file path,
    - the set of scan-site writers that must be flushed before snapshotting.

    Nested scans reuse the same coordinator through a run context, so any completed
    batch anywhere in the active scan tree can trigger a preview once enough time has
    elapsed.
    """

    def __init__(
        self,
        owner: HasEnvironment,
        policy: PreviewPolicy,
        *,
        run_start_unix_time: float,
    ):
        self.policy = policy
        self._owner = owner
        self._run_start_unix_time = run_start_unix_time
        self._path = policy.resolve_path(owner)
        self._writers: WeakSet[ScanSiteDatasetWriter] = WeakSet()
        self._last_preview_monotonic = time.monotonic()
        self._write_in_progress = False

    def register_writer(self, writer: ScanSiteDatasetWriter) -> None:
        """Register a site writer whose buffered state must reach preview files."""

        self._writers.add(writer)

    def unregister_writer(self, writer: ScanSiteDatasetWriter) -> None:
        """Remove a site writer from the preview flush set."""

        self._writers.discard(writer)

    def maybe_write_preview(self) -> None:
        """Write a preview snapshot when the configured interval has elapsed."""

        if self._write_in_progress:
            return
        now = time.monotonic()
        if now - self._last_preview_monotonic < self.policy.min_interval_s:
            return
        self._write_preview(preview_complete=False, monotonic_time=now)

    def write_completion_preview(self) -> None:
        """Write a final preview snapshot after the run has finished.

        This still targets the separate preview file rather than ARTIQ's canonical
        results file. The final worker-managed HDF5 write remains the source of truth
        for the completed run artifact.
        """

        if self.policy.remove_on_completion:
            self.remove_preview()
            return
        if not self.policy.write_on_completion or self._write_in_progress:
            return
        self._write_preview(preview_complete=True, monotonic_time=time.monotonic())

    def remove_preview(self) -> None:
        """Remove the preview file after a successful completed run."""

        if self._write_in_progress:
            return
        try:
            os.remove(self._path)
        except FileNotFoundError:
            pass

    def _write_preview(self, *, preview_complete: bool, monotonic_time: float) -> None:
        self._write_in_progress = True
        try:
            for writer in tuple(self._writers):
                writer.flush()

            dataset_mgr = self._owner._HasEnvironment__dataset_mgr
            scheduler = self._owner.get_device("scheduler")
            directory = os.path.dirname(self._path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            tmp_path = self._path + ".tmp"
            with h5py.File(tmp_path, "w") as h5_file:
                dataset_mgr.write_hdf5(h5_file)
                h5_file["artiq_version"] = artiq_version
                h5_file["rid"] = getattr(scheduler, "rid", 0)
                expid = getattr(scheduler, "expid", None)
                if expid is not None:
                    h5_file["expid"] = pyon.encode(expid)
                h5_file["preview_time"] = time.time()
                h5_file["preview_complete"] = preview_complete
                h5_file["run_start_unix_time"] = self._run_start_unix_time
            os.replace(tmp_path, self._path)
            self._last_preview_monotonic = monotonic_time
        finally:
            self._write_in_progress = False


@dataclass(frozen=True)
class RunContext:
    """Root-scoped coordination state shared by nested host-runtime sessions."""

    run_start_unix_time: float
    preview: PreviewCoordinator | None = None


_active_run_context: ContextVar[RunContext | None] = ContextVar(
    "_active_run_context", default=None
)


class HostArgumentInterface(HasEnvironment):
    """Expose host-runtime submissions through the existing ndscan dashboard channel.

    This intentionally does less than the legacy argument interface:

    - it always publishes parameter metadata and fixed overrides,
    - it only publishes ``host_scan`` transport when the request was defined
      declaratively (``HostScanSpec`` or dict form),
    - it never attempts to reverse-engineer an arbitrary ``ScanRequest`` back into
      submission schema.

    That keeps the bridge one-way and predictable. Code-defined ``ScanRequest``
    objects can still benefit from dashboard-set fixed overrides, while declarative
    host-scan requests gain a transport payload that future dashboard editing can
    target.
    """

    def build(self, fragment: ExpFragment, default_request_spec: Any = None) -> None:
        instances = dict[str, list[str]]()
        self._schemata = dict[str, dict]()
        self._sample_instances = dict[str, Any]()
        always_shown_params = []
        result_channels = dict[str, ResultChannel]()

        fragment._collect_params(instances, self._schemata, self._sample_instances)
        fragment._collect_result_channels(result_channels)
        for handle in fragment.get_always_shown_params():
            path = handle.owner._stringize_path()
            try:
                param = handle.owner._free_params[handle.name]
                always_shown_params += [(param.fqn, path)]
            except KeyError:
                logger.debug(
                    "Parameter '%s' specified in get_always_shown_params() is not a "
                    "free parameter of fragment '%s'",
                    handle.name,
                    path,
                )

        desc: dict[str, Any] = {
            "instances": instances,
            "schemata": self._schemata,
            "always_shown": always_shown_params,
            "channels": {
                path: channel.describe()
                for path, channel in result_channels.items()
                if channel.save_by_default
            },
            "overrides": {},
        }
        default_transport = _host_request_transport_dict(default_request_spec)
        if default_transport is not None:
            desc["host_scan"] = default_transport

        self._params = self.get_argument(PARAMS_ARG_KEY, PYONValue(default=desc))

    def make_override_stores(self) -> dict[str, list[tuple[str, ParamStore]]]:
        stores = {}
        for fqn, specs in self._params.get("overrides", {}).items():
            try:
                store_type = self._sample_instances[fqn].StoreType
            except KeyError:
                raise KeyError(
                    "Experiment does not have parameters matching override for FQN "
                    f"{fqn!r}"
                )
            stores[fqn] = [
                (
                    spec["path"],
                    store_type(
                        (fqn, spec["path"]),
                        store_type.value_from_pyon(spec["value"]),
                    ),
                )
                for spec in specs
            ]
        return stores

    def resolve_request(
        self,
        fragment: ExpFragment,
        default_request_spec: Any | None,
    ) -> tuple["ScanRequest", dict[str, list[tuple[str, ParamStore]]]]:
        if "host_scan" in self._params:
            request, compiled_overrides = compile_host_scan_schema(
                fragment, self._params["host_scan"]
            )
        elif default_request_spec is None:
            raise ValueError(
                "No host_scan submission was provided for this dashboard-driven "
                "host scan experiment"
            )
        else:
            request, compiled_overrides = _resolve_host_scan_request_spec(
                fragment,
                default_request_spec,
            )
        return request, _merge_override_store_maps(
            compiled_overrides,
            self.make_override_stores(),
        )


def current_scan_context() -> ActiveScanContext | None:
    """Return the currently executing parent scan context, if any."""

    stack = _active_scan_context.get()
    return stack[-1] if stack else None


def _current_effective_scan_context() -> ActiveScanContext | None:
    """Return the active parent point context from host or resident-kernel execution."""

    context = current_scan_context()
    if context is not None:
        return context

    if _persistent_kernel_parent_scan_context:
        return _persistent_kernel_parent_scan_context[-1].snapshot()

    stack = _active_kernel_parent_scan_context.get()
    if not stack:
        return None
    return stack[-1].snapshot()


def current_run_context() -> RunContext | None:
    """Return the root-scoped runtime context for the current host scan tree."""

    return _active_run_context.get()


@contextmanager
def _push_scan_context(context: ActiveScanContext):
    stack = _active_scan_context.get()
    token = _active_scan_context.set(stack + (context,))
    try:
        yield
    finally:
        _active_scan_context.reset(token)


@contextmanager
def _push_run_context(context: RunContext):
    token = _active_run_context.set(context)
    try:
        yield
    finally:
        _active_run_context.reset(token)


@contextmanager
def _push_kernel_parent_scan_context(provider: _KernelParentScanContextProvider):
    stack = _active_kernel_parent_scan_context.get()
    token = _active_kernel_parent_scan_context.set(stack + (provider,))
    try:
        yield
    finally:
        _active_kernel_parent_scan_context.reset(token)


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

    parent = _current_effective_scan_context()
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

    ``preview_policy`` is root-run scoped. Nested scans inherit the active root
    preview coordinator rather than configuring a second independent cadence.
    """

    max_points_per_batch: int | None = None
    preview_policy: PreviewPolicy | None = None

    def __post_init__(self) -> None:
        if self.max_points_per_batch is not None and self.max_points_per_batch <= 0:
            raise ValueError("max_points_per_batch must be positive when specified")
        if self.preview_policy is not None and not isinstance(
            self.preview_policy, PreviewPolicy
        ):
            raise TypeError("preview_policy must be a PreviewPolicy instance")


@dataclass(frozen=True)
class ScanRequest:
    """User-facing host-runtime scan request.

    The first implementation is deliberately code-first: callers provide either real
    ``ParamHandle`` scan axes or logical ``ScanVariable`` axes, together with a
    ``PointPolicy`` describing the point strategy. A future dashboard adapter can
    resolve selector syntax or text formulas into the same objects without changing
    the runtime core again.

    Runtime scheduling choices live in ``execution_policy`` rather than in the request
    itself. That keeps the request focused on the scan shape while still letting the
    runner batch, flush, and pause at well-defined boundaries.
    """

    axes: tuple[ParamHandle | ScanVariable, ...]
    point_policy: PointPolicy
    site: ScanSite = field(default_factory=ScanSite)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    parameter_mappings: tuple[ParameterMapping, ...] = ()
    fixed_pseudoparams: tuple[FixedPseudoparam, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.execution_policy, ExecutionPolicy):
            raise TypeError("execution_policy must be an ExecutionPolicy instance")
        for mapping in self.parameter_mappings:
            if not isinstance(mapping, ParameterMapping):
                raise TypeError(
                    "parameter_mappings must contain ParameterMapping instances"
                )
        for pseudoparam in self.fixed_pseudoparams:
            if not isinstance(pseudoparam, FixedPseudoparam):
                raise TypeError(
                    "fixed_pseudoparams must contain FixedPseudoparam instances"
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
            point_policy=self.point_policy,
            site=site,
            metadata=self.metadata,
            execution_policy=self.execution_policy,
            parameter_mappings=self.parameter_mappings,
            fixed_pseudoparams=self.fixed_pseudoparams,
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
            point_policy=self.point_policy,
            site=self.site,
            metadata=self.metadata,
            execution_policy=self.execution_policy,
            parameter_mappings=self.parameter_mappings + tuple(parameter_mappings),
            fixed_pseudoparams=self.fixed_pseudoparams,
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
            point_policy=SinglePointPolicy(),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
            execution_policy=ExecutionPolicy() if execution_policy is None else execution_policy,
        )

    @classmethod
    def linear(
        cls,
        axis: ParamHandle | ScanVariable,
        *,
        start: float,
        stop: float,
        num_points: int,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
        execution_policy: ExecutionPolicy | None = None,
    ) -> "ScanRequest":
        """Return a simple 1D linear scan request.

        This is a small code-first convenience for the common case where an experiment
        wants one evenly spaced axis without manually materialising a point list. More
        complex shapes should continue to use ``cartesian()``, ``zipped()``, or
        ``explicit()`` directly.
        """

        if num_points < 2:
            raise ValueError("linear scans require at least 2 points")
        values = np.linspace(start=float(start), stop=float(stop), num=int(num_points))
        return cls.cartesian(
            [(axis, values.tolist())],
            site=site,
            metadata=metadata,
            execution_policy=execution_policy,
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
            point_policy=CartesianPointPolicy([values for _, values in axes]),
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
            point_policy=ZipPointPolicy([values for _, values in axes]),
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
            point_policy=ExplicitPointPolicy(len(axes), points),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
            execution_policy=ExecutionPolicy() if execution_policy is None else execution_policy,
        )

# Imported here rather than at module top because the schema compiler constructs
# ``ScanRequest`` and ``ExecutionPolicy`` instances from this module.
from .host_scan_schema import (
    HostScanSchemaError,
    HostScanGridModeSpec,
    HostScanSpec,
    compile_host_scan_schema,
    compile_host_scan_spec,
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
    point_metadata: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    acquired_at_unix: float | None = None


@dataclass(frozen=True)
class _ResolvedExecutionPoint:
    """Concrete point installation plan shared by host and kernel executors.

    ``point`` keeps the original point-policy choice around for index/metadata, while
    the remaining fields capture the fully resolved execution state after direct axis
    installation and any parameter mappings have been applied on the host.
    """

    point: BasePoint
    axis_values: OrderedDict[str, Any]
    pseudoparam_values: OrderedDict[str, Any]
    parameter_values: OrderedDict[str, Any]
    rpc_parameter_values: tuple[Any, ...]


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
        _apply_resolved_parameter_values(self._parameters, self._current_chunk[0])

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
    runtime_stats: "HostScanRuntimeStats"

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
            runtime_stats=HostScanRuntimeStats(),
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


class _HostPointBatchSource:
    """Host-side batch source/fallback point-feedback seam.

    Today this is just an adapter around ``PointPolicy`` plus the execution-policy
    batch limit. Later a kernel-capable point source can slot in behind the same
    runtime shape without changing the public scan API.
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
        """Optional future kernel-capable point source descriptor."""

        return None


class _HostAnalysisAdapter:
    """Host-side analysis seam.

    Today this forwards directly to ``HostScanAnalysisEngine``. The separate adapter
    makes it easier to later support optional kernel reducers without changing the
    controller/batch-publication flow.
    """

    def __init__(self, engine: HostScanAnalysisEngine):
        self._engine = engine

    @property
    def engine(self) -> HostScanAnalysisEngine:
        return self._engine

    def metadata(self) -> dict[str, Any]:
        return self._engine.metadata()

    def initial_annotations(self) -> list[dict[str, Any]]:
        return self._engine.initial_annotations()

    def observe_batch(
        self,
        completed_batch: Sequence[PointObservation],
        result: HostScanRunResult,
        site_writer: ScanSiteDatasetWriter,
    ):
        return self._engine.observe_batch(completed_batch, result, site_writer)

    def execute_final(
        self, result: HostScanRunResult, site_writer: ScanSiteDatasetWriter
    ) -> None:
        self._engine.execute_final(result, site_writer)

    def kernel_reducer(self):
        """Optional future kernel-capable reducer descriptor."""

        return None


class _HostObservationTransport:
    """Host-side persistence/preview seam for completed observations."""

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

    def register_preview(self, preview: PreviewCoordinator | None) -> None:
        if preview is not None:
            preview.register_writer(self._site_writer)

    def unregister_preview(self, preview: PreviewCoordinator | None) -> None:
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

    def maybe_write_preview(self, preview: PreviewCoordinator | None) -> None:
        if preview is not None:
            preview.maybe_write_preview()

    def write_completion_preview(self, preview: PreviewCoordinator | None) -> None:
        if preview is not None:
            preview.write_completion_preview()

    def kernel_transport(self):
        """Optional future kernel-side buffered transport descriptor."""

        return None


def _make_batch_feedback(
    observations: Sequence[PointObservation],
    result: HostScanRunResult,
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
    result: HostScanRunResult,
    *,
    axes: Sequence[BoundScanAxis],
    parameters: Sequence[BoundScanParameter],
    channels: Sequence[BoundResultChannel],
    point_source: _HostPointBatchSource,
    analysis: _HostAnalysisAdapter,
    transport: _HostObservationTransport,
    preview: PreviewCoordinator | None,
) -> None:
    if not completed_batch:
        return

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


@dataclass
class HostScanRuntimeStats:
    """Lightweight runtime counters/timings for executor bring-up and profiling.

    These stats intentionally live on the in-memory run result rather than in the scan
    site schema. They are primarily for tests, profiling, and future executor bring-up.
    """

    batch_count: int = 0
    point_count: int = 0
    executor_entry_count: int = 0
    first_executor_entry_elapsed_s: float | None = None
    total_executor_elapsed_s: float = 0.0
    total_batch_finalize_elapsed_s: float = 0.0


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


@dataclass(frozen=True)
class _BatchExecutionResult:
    """Result of executing one runtime batch through an execution backend."""

    observations: tuple[PointObservation, ...]
    restart_host_context: bool = False
    elapsed_s: float = 0.0
    executor_entries: int = 1


class HostExecutor:
    """Execute already-resolved points against a fragment from the host runtime.

    This is the current host-side execution backend. The controller still owns batch
    selection, persistence, analyses, and pause handling; the executor owns:

    - host setup/cleanup for one execution batch,
    - direct axis installation,
    - parameter mappings,
    - point-body invocation,
    - and point result collection.

    Later kernel-oriented executors should be able to implement the same
    ``execute_batch()`` contract with a different inner execution strategy.
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
        """Execute one runtime batch.

        Returns all successfully completed observations in order. When a point body
        raises ``RestartKernelTransitoryError``, the executor returns the observations
        completed so far together with ``restart_host_context=True`` so the controller
        can re-enter host setup before retrying the remainder of the batch.
        """

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

    The first kernel backend intentionally keeps the host-side scan controller in
    charge of point selection and batch boundaries. The executor owns the resident
    kernel loop and asks the host for the next already-chosen batch via RPC.

    Supported in v1:

    - direct axes, pseudoparams, and runtime parameter mappings as long as the host can
      precompute the concrete installed parameter values,
    - host-selected batches,
    - existing per-point result channel push path.
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

        self._result: HostScanRunResult | None = None
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
        result: HostScanRunResult,
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
        analysis_engine: HostScanAnalysisEngine,
    ):
        self.fragment = fragment
        self.request = request
        self.axes = tuple(axes)
        self.parameters = tuple(parameters)
        self.channels = tuple(channels)
        self.parameter_mappings = tuple(parameter_mappings)
        self.point_source = _HostPointBatchSource(
            request.point_policy, request.execution_policy
        )
        self.analysis = _HostAnalysisAdapter(analysis_engine)
        self.transport = _HostObservationTransport(site_writer)

        # Internal compatibility aliases while the runtime migrates toward the
        # capability seams above.
        self.point_policy = request.point_policy
        self.site_writer = self.transport.site_writer
        self.analysis_engine = self.analysis.engine

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "site.fragment_fqn": self.fragment.fqn,
            "scan.point_policy": self.point_source.describe(),
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
    """Return a JSON-friendly representation of a parameter value."""

    if isinstance(value, Enum):
        return value.name
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _collect_fixed_parameter_metadata(
    fragment: ExpFragment,
    varying_parameters: Sequence[BoundScanParameter],
) -> dict[str, Any]:
    """Return metadata for all real parameters that stayed fixed for this scan.

    The legacy runtime used ARTIQ's submission-time `expid` payload to preserve the
    non-scanned parameter state. Programmatic host-runtime scans do not naturally go
    through that path, so the scan site records the same information explicitly in its
    own schema.
    """

    varying_keys = {
        _mapping_target_key(parameter.handle) for parameter in varying_parameters
    }
    fixed_parameters = []

    def walk(current: ExpFragment) -> None:
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
    """Return required kernel driver attributes missing from ``fragment``.

    ARTIQ's ``@kernel`` decorator records the attribute name of the core driver it
    expects to use. For the default form ``@kernel``, this is ``"core"``. Real core
    execution can compile nested same-device kernel calls without reading that
    attribute at runtime, but host-side entry paths and DAX sim do still expect the
    attribute to exist on the object. Failing early here produces a much clearer error
    than letting execution fall through to ``AttributeError: ... has no attribute
    'core'`` later.
    """

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
    """Return whether the strict first kernel-streaming backend can execute this scan.

    The v1 backend is intentionally narrow:

    - the scanned fragment actually uses kernel execution,
    - the scan has at least one concrete fragment parameter that changes point-to-point.

    Logical scan variables and parameter mappings are now supported as long as the host
    can precompute concrete parameter values before each batch enters the resident
    kernel loop.
    """

    if not _fragment_uses_kernel_execution(fragment):
        return False
    if not axes:
        return False
    return bool(parameters)


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
                "Fragments using @kernel in the host runtime must declare the "
                f"corresponding device attribute(s) {names} during build_fragment(); "
                "for ordinary kernels this usually means calling "
                "self.setattr_device('core')"
            )
        if _fragment_uses_kernel_execution(fragment) and not _can_use_kernel_streaming_executor(
            fragment, axes, parameters
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

    def run(self) -> HostScanRunResult:
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

                result = HostScanRunResult.empty(
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
                # Finish the point stream before analyses run; later buffered
                # implementations should make this flush any still-pending point data.
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

    def _effective_batch_size(self) -> int:
        """Return the point count upper bound for the next execution batch.

        The execution policy owns the hard upper bound because batching is
        fundamentally a runtime/persistence choice. The point policy may still ask for
        a smaller natural batch size when, for example, it wants one ask/tell step per
        batch.
        """

        return self._program.point_source.effective_batch_size()

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
        5. flush pending writer state,
        6. maybe emit a preview snapshot,
        7. then let pause/restart/completion decisions happen.

        Keeping that boundary explicit makes later online analysis and writer-side
        buffering extensions much easier to reason about.
        """

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
            preview=(
                None if self._run_context is None else self._run_context.preview
            ),
        )

    def _has_more_work(self) -> bool:
        return self._program.point_source.has_more_work()

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
        run_context: RunContext | None = None,
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
            run_context=run_context,
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


def _prepare_child_scan_request(
    request: ScanRequest,
    *,
    name: str,
    segmented: bool,
    extra_metadata: Mapping[str, Any] | None,
) -> ScanRequest:
    """Return ``request`` with a structural child scan site nested under the parent."""

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
    """Return a fresh executable copy of ``request``.

    `ScanRequest` structure is immutable enough to reuse directly, but the point policy
    it contains is stateful. Prepared scans should therefore execute against a fresh
    point-policy instance each time while keeping the same fragment handles and
    mappings.
    """

    return ScanRequest(
        axes=request.axes,
        point_policy=deepcopy(request.point_policy),
        parameter_mappings=request.parameter_mappings,
        site=request.site,
        metadata=deepcopy(request.metadata),
        execution_policy=request.execution_policy,
    )


def _auto_detach_prepared_child_fragment(
    owner: HasEnvironment, fragment: ExpFragment
) -> None:
    """Detach ``fragment`` when it is a direct child being prepared as a scan target.

    Prepared child scans own the execution lifecycle of their scanned fragment. If the
    fragment remained attached to the parent's normal traversal, setup/cleanup would
    happen through two different paths. When ``prepare_child_scan()`` is called during
    ``build_fragment()``, detach the direct child automatically. Outside build time we
    can no longer change that relationship, so raise a clear error instead.
    """

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
    """Return all declared default-analysis result channels for ``fragment``."""

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


def _resolve_prepared_child_output_channels(
    fragment: ExpFragment,
    *,
    expose_analysis_results: bool,
    expose_outputs: Sequence[str] | None,
) -> tuple[dict[str, ResultChannel], dict[str, ResultChannel]]:
    """Resolve named fixed outputs for a prepared child scan.

    Returns ``(analysis_result_accessors, tuple_output_accessors)``.
    """

    all_channels = _collect_default_analysis_result_channels(fragment)

    if expose_outputs is None:
        if expose_analysis_results:
            return all_channels, {}
        return {}, {}

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
    return selected, selected


class _PreparedChildAnalysisResultAccessorBase:
    """Compiler-friendly fixed-output accessor for prepared child scans."""

    def __init__(self, owner: "PreparedChildScan", result_name: str):
        self._owner = owner
        self._result_name = result_name

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}@{hex(id(self))}: "
            f"{self._owner._name}.{self._result_name}>"
        )


class _PreparedChildFloatAnalysisResultAccessor(_PreparedChildAnalysisResultAccessorBase):
    @rpc
    def _get_value(self) -> float:
        return float(self._owner._get_exposed_analysis_result(self._result_name))

    @portable
    def get(self) -> float:
        return self._get_value()


class _PreparedChildIntAnalysisResultAccessor(_PreparedChildAnalysisResultAccessorBase):
    @rpc
    def _get_value(self) -> np.int32:
        return np.int32(self._owner._get_exposed_analysis_result(self._result_name))

    @portable
    def get(self) -> np.int32:
        return self._get_value()


def _make_prepared_child_analysis_result_accessor(
    owner: "PreparedChildScan",
    result_name: str,
    channel: ResultChannel,
):
    """Return a typed fixed-output accessor for one prepared child analysis result."""

    if isinstance(channel, FloatChannel):
        accessor_class = type(
            f"_PreparedChildFloatAnalysisResult_{id(owner)}_{result_name}",
            (_PreparedChildFloatAnalysisResultAccessor,),
            {},
        )
        return accessor_class(owner, result_name)
    if isinstance(channel, IntChannel):
        accessor_class = type(
            f"_PreparedChildIntAnalysisResult_{id(owner)}_{result_name}",
            (_PreparedChildIntAnalysisResultAccessor,),
            {},
        )
        return accessor_class(owner, result_name)
    _prepared_child_fixed_output_rpc_type(channel)
    raise AssertionError("unreachable")


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
            coercer(self._get_exposed_analysis_result(result_name))
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


class PreparedChildScan:
    """Prepared child-scan handle for the newer host runtime.

    This fixes the structural part of a nested scan up front:

    - which fragment is the child target,
    - what child-site name it uses,
    - and the default site-shaping options for that child.

    Concrete `ScanRequest` instances are still configured later.

    Two execution entry points exist:

    - ``run()`` keeps the rich host-only return value.
    - ``acquire()`` is a compiler-friendly execution entry that returns ``None`` and
      can therefore be called from kernels.
    - ``prime()`` explicitly primes the detached child subtree from host code before an
      outer kernel is first compiled.

    `acquire()` now supports two execution shapes:

    - host-executed child scans still run through a blocking host RPC
    - kernel-executed child scans use a dedicated prepared nested kernel runner that
      keeps the child scan loop inside the compiled call graph while still fetching
      point batches from the host

    The kernel path is still intentionally strict and currently shares the same
    executor eligibility rules as the top-level `KernelStreamingExecutor`.
    """

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
        expose_analysis_results: bool = False,
        expose_outputs: Sequence[str] | None = None,
    ):
        self._owner = owner
        self._fragment = fragment
        self._name = name
        self._segmented = segmented
        self._extra_metadata = {} if extra_metadata is None else dict(extra_metadata)
        self._max_rtio_underflow_retries = max_rtio_underflow_retries
        self._max_transitory_error_retries = max_transitory_error_retries
        self._expose_analysis_results = expose_analysis_results
        self._expose_outputs = None if expose_outputs is None else tuple(expose_outputs)

        self._request: ScanRequest | None = None
        self._overrides: dict[str, list[tuple[str, ParamStore]]] | None = None
        self._last_result: HostScanRunResult | None = None
        self._kernel_runner = None
        self._kernel_runner_ready = False
        self._kernel_support_error: str | None = None
        self._kernel_axis_bindings: list[_ScanAxisBinding] = []
        self._kernel_program: HostScanProgram | None = None
        self._kernel_result: HostScanRunResult | None = None
        self._kernel_run_context: RunContext | None = None
        self._kernel_preview = None
        self._kernel_collector: _PointResultCollector | None = None
        self._kernel_batch_state: _ResidentKernelBatchState | None = None
        self._kernel_parent_provider: _KernelParentScanContextProvider | None = None
        self._kernel_host_setup_active = False
        (
            self._exposed_analysis_channel_specs,
            self._exposed_output_channel_specs,
        ) = _resolve_prepared_child_output_channels(
            self._fragment,
            expose_analysis_results=expose_analysis_results,
            expose_outputs=expose_outputs,
        )

        analysis_results_class = type(
            f"_PreparedChildAnalysisResults_{id(self)}",
            (),
            {},
        )
        self.analysis_results = analysis_results_class()
        for result_name, channel in self._exposed_analysis_channel_specs.items():
            accessor = _make_prepared_child_analysis_result_accessor(
                self, result_name, channel
            )
            setattr(self.analysis_results, result_name, accessor)

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
    def configure(
        self,
        request: ScanRequest,
        *,
        overrides: dict[str, list[tuple[str, ParamStore]]] | None = None,
    ) -> None:
        self._request = request
        self._overrides = overrides
        self._refresh_kernel_acquire_runner()

    @host_only
    def prime(self) -> None:
        """Prime the detached child subtree from host code.

        This is the ergonomic entry point for the standard ndscan pattern where nested
        kernel-visible state is prepared in ``host_setup()`` before the outer kernel is
        first compiled. For detached child fragments, calling ``child_scan.prime()`` is
        equivalent to explicitly forwarding to ``child.host_setup()`` without exposing
        the detach detail at the call site.
        """

        self._fragment.host_setup()

    @host_only
    def run(self) -> HostScanRunResult:
        if self._request is None:
            raise RuntimeError(
                f"Prepared child scan '{self._name}' has not been configured yet"
            )

        self._last_result = run_host_scan(
            self._owner,
            self._fragment,
            _prepare_child_scan_request(
                _clone_scan_request_for_execution(self._request),
                name=self._name,
                segmented=self._segmented,
                extra_metadata=self._extra_metadata,
            ),
            overrides=self._overrides,
            max_rtio_underflow_retries=self._max_rtio_underflow_retries,
            max_transitory_error_retries=self._max_transitory_error_retries,
        )
        return self._last_result

    @portable
    def acquire(self) -> None:
        """Execute the prepared child scan without returning a rich Python result.

        This is the compiler-facing entry point. Host-executed child scans continue to
        bridge out to the host directly. Kernel-executed child scans use a dedicated
        prepared nested runner so the child loop stays inside the compiled call graph
        while point batches still come from the host.
        """
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
        self.run()

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

        if self._request is None or not _fragment_uses_kernel_execution(self._fragment):
            return

        if _fragment_tree_needs_param_initialisation(self._fragment):
            self._fragment.init_params(
                overrides={} if self._overrides is None else self._overrides
            )

        axis_handles = [
            axis for axis in self._request.axes if isinstance(axis, ParamHandle)
        ]
        axis_bindings = _install_scan_axis_stores(axis_handles)
        try:
            axes = _build_bound_axes(self._request.axes)
            parameter_mappings = _collect_parameter_mappings(
                self._fragment, self._request, axes
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
        if self._kernel_program is not None:
            raise RuntimeError("Prepared child scan kernel acquire is already active")

        request = _prepare_child_scan_request(
            _clone_scan_request_for_execution(self._request),
            name=self._name,
            segmented=self._segmented,
            extra_metadata=self._extra_metadata,
        )

        builder = HostScanProgramBuilder(self._owner)
        self._kernel_program = builder.build(self._fragment, request)
        self._kernel_run_context = current_run_context()
        if self._kernel_run_context is None:
            raise RuntimeError(
                "PreparedChildScan.acquire() can only be used while a parent scan run "
                "context is active"
            )
        self._kernel_preview = self._kernel_run_context.preview
        self._kernel_program.transport.register_preview(self._kernel_preview)

        try:
            self._fragment.prepare()
            site_start_unix_time = time.time()
            self._kernel_program.transport.publish_metadata(
                self._kernel_program.metadata(),
                extra_metadata=self._kernel_program.request.metadata,
                start_unix_time=site_start_unix_time,
            )
            if self._kernel_program.request.site.segmented:
                parent = _current_effective_scan_context()
                self._kernel_program.transport.start_segment(
                    parent_point_index=None if parent is None else parent.point_index,
                    start_unix_time=time.time(),
                )

            self._kernel_result = HostScanRunResult.empty(
                self._kernel_program.axes,
                self._kernel_program.parameters,
                self._kernel_program.channels,
                self._kernel_program.transport.prefix,
                initial_annotations=self._kernel_program.analysis.initial_annotations(),
            )
            self._kernel_collector = _PointResultCollector(self._kernel_program.channels)
            self._kernel_collector.install()
            self._kernel_batch_state = _ResidentKernelBatchState(
                self._kernel_program.axes,
                self._kernel_program.parameters,
                self._kernel_program.parameter_mappings,
                self._kernel_collector,
            )

            self._fragment.recompute_param_defaults()
            self._fragment.host_setup()
            self._kernel_host_setup_active = True

            self._kernel_parent_provider = _KernelParentScanContextProvider(
                self._kernel_program.request.site.path,
                lambda: self._kernel_batch_state.current_next_point_index,
            )
            _persistent_kernel_parent_scan_context.append(self._kernel_parent_provider)
        except BaseException:
            self._cleanup_kernel_acquire_state(completed=False)
            raise

    @rpc(flags={"async"})
    def _record_kernel_executor_entry(self) -> None:
        if self._kernel_result is not None:
            self._kernel_result.runtime_stats.executor_entry_count += 1

    @rpc
    def _restart_kernel_host_context(self) -> None:
        if self._kernel_host_setup_active:
            self._fragment.host_cleanup()
        self._fragment.recompute_param_defaults()
        self._fragment.host_setup()
        self._kernel_host_setup_active = True

    @rpc
    def _pause_kernel_after_batch(self) -> None:
        scheduler = self._owner.get_device("scheduler")
        scheduler.pause()

    @rpc
    def _complete_kernel_acquire(self) -> None:
        self._cleanup_kernel_acquire_state(completed=True)

    @rpc
    def _abort_kernel_acquire(self) -> None:
        self._cleanup_kernel_acquire_state(completed=False)

    @host_only
    def _cleanup_kernel_acquire_state(self, *, completed: bool) -> None:
        program = self._kernel_program
        result = self._kernel_result

        provider = self._kernel_parent_provider
        if provider is not None:
            if (
                not _persistent_kernel_parent_scan_context
                or _persistent_kernel_parent_scan_context[-1] is not provider
            ):
                raise RuntimeError(
                    "Prepared child scan kernel parent context stack is out of sync"
                )
            _persistent_kernel_parent_scan_context.pop()
            self._kernel_parent_provider = None

        if self._kernel_host_setup_active:
            self._fragment.host_cleanup()
            self._kernel_host_setup_active = False

        if self._kernel_collector is not None:
            self._kernel_collector.remove()
            self._kernel_collector = None

        try:
            if program is not None:
                if completed and result is not None:
                    if program.request.site.segmented:
                        program.transport.finish_segment()
                    program.transport.flush()
                    program.analysis.execute_final(result, program.transport.site_writer)
                    program.transport.set_completed(True)
                    self._last_result = result
                program.transport.close()
        finally:
            if program is not None:
                program.transport.unregister_preview(self._kernel_preview)
            self._kernel_program = None
            self._kernel_result = None
            self._kernel_run_context = None
            self._kernel_preview = None
            if self._kernel_batch_state is not None:
                self._kernel_batch_state.reset()
            self._kernel_batch_state = None

    @host_only
    def _effective_kernel_batch_size(self) -> int:
        assert self._kernel_program is not None
        return self._kernel_program.point_source.effective_batch_size()

    @host_only
    def _next_kernel_batch(self):
        assert self._kernel_program is not None
        return self._kernel_program.point_source.next_batch()

    @host_only
    def _get_param_values_chunk(self):
        assert self._kernel_program is not None
        assert self._kernel_batch_state is not None
        return self._kernel_batch_state.get_param_values_chunk(
            next_batch=self._next_kernel_batch,
            next_point_index=lambda: self._kernel_program.site_writer.next_point_index,
        )

    def _retry_point(self):
        if self._kernel_batch_state is not None:
            self._kernel_batch_state.retry_point()

    def _point_completed(self):
        assert self._kernel_program is not None
        assert self._kernel_result is not None
        if self._kernel_batch_state is None:
            raise RuntimeError("Prepared child kernel batch state is not installed")
        self._kernel_batch_state.point_completed()

    @host_only
    def _finish_kernel_completed_batch(self):
        assert self._kernel_program is not None
        assert self._kernel_result is not None
        if self._kernel_batch_state is None:
            return

        completed_batch = self._kernel_batch_state.take_completed_batch()
        _publish_completed_batch(
            completed_batch,
            self._kernel_result,
            axes=self._kernel_program.axes,
            parameters=self._kernel_program.parameters,
            channels=self._kernel_program.channels,
            point_source=self._kernel_program.point_source,
            analysis=self._kernel_program.analysis,
            transport=self._kernel_program.transport,
            preview=self._kernel_preview,
        )

    @host_only
    def _finish_chunk(self):
        assert self._kernel_program is not None
        self._finish_kernel_completed_batch()
        if not self._kernel_program.point_source.has_more_work():
            return KernelStreamingExecutor._STATUS_COMPLETE
        scheduler = self._owner.get_device("scheduler")
        if scheduler.check_pause():
            return KernelStreamingExecutor._STATUS_PAUSE
        return KernelStreamingExecutor._STATUS_PROCEED

    @host_only
    def _finish_chunk_after_restart(self):
        self._finish_kernel_completed_batch()
        return KernelStreamingExecutor._STATUS_RESTART_HOST_CONTEXT

    @host_only
    def last_result(self) -> HostScanRunResult | None:
        """Return the most recent host-side result object, if any."""

        return self._last_result

    @host_only
    def _get_exposed_analysis_result(self, result_name: str) -> Any:
        """Return one declared fixed output from the most recent child-scan run."""

        if result_name not in self._exposed_analysis_channel_specs:
            raise AttributeError(
                f"Prepared child scan '{self._name}' does not expose analysis result "
                f"{result_name!r}"
            )
        if self._last_result is None:
            raise RuntimeError(
                f"Prepared child scan '{self._name}' has not been run yet"
            )
        try:
            return self._last_result.analysis_results[result_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Prepared child scan '{self._name}' did not produce analysis result "
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
    expose_analysis_results: bool = False,
    expose_outputs: Sequence[str] | None = None,
) -> PreparedChildScan:
    """Return a prepared child-scan handle with fixed structural scan identity.

    When called during ``build_fragment()`` for a direct child fragment, the child is
    detached automatically so its lifecycle is owned exclusively by the prepared scan
    handle.
    """

    _auto_detach_prepared_child_fragment(owner, fragment)
    prepared_class_namespace = _make_prepared_child_get_outputs_methods(
        _resolve_prepared_child_output_channels(
            fragment,
            expose_analysis_results=expose_analysis_results,
            expose_outputs=expose_outputs,
        )[1]
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
        expose_analysis_results=expose_analysis_results,
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
    expose_analysis_results: bool = False,
    expose_outputs: Sequence[str] | None = None,
    **kwargs,
) -> PreparedChildScan:
    """Create, detach, and prepare a child fragment as a prepared child scan.

    This is the ergonomic companion to legacy ``setattr_subscan()``:

    - the child fragment becomes available as ``owner.<name>``
    - it is detached from ordinary traversal immediately
    - the returned handle owns the prepared child-scan lifecycle

    The structural setup belongs in ``build_fragment()``. Concrete scan requests and
    any compiler priming still belong in ``host_setup()`` or other host-side setup
    helpers invoked before the outer kernel is first compiled.
    """

    fragment = owner.setattr_fragment(name, fragment_class, *args, detached=True, **kwargs)
    return prepare_child_scan(
        owner,
        fragment,
        name=name if scan_name is None else scan_name,
        segmented=segmented,
        extra_metadata=extra_metadata,
        max_rtio_underflow_retries=max_rtio_underflow_retries,
        max_transitory_error_retries=max_transitory_error_retries,
        expose_analysis_results=expose_analysis_results,
        expose_outputs=expose_outputs,
    )


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
    prepared = prepare_child_scan(
        owner,
        fragment,
        name=name,
        segmented=segmented,
        extra_metadata=extra_metadata,
        max_rtio_underflow_retries=max_rtio_underflow_retries,
        max_transitory_error_retries=max_transitory_error_retries,
    )
    prepared.configure(request, overrides=overrides)
    return prepared.run()


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


def _resolve_execution_point(
    point: BasePoint,
    axes: Sequence[BoundScanAxis],
    parameters: Sequence[BoundScanParameter],
    parameter_mappings: Sequence[_BoundParameterMapping],
) -> _ResolvedExecutionPoint:
    """Resolve one logical point into concrete installed parameter values.

    This mirrors the host-runtime execution order exactly:

    1. install any directly scanned fragment-parameter axes,
    2. evaluate parameter mappings against the resulting logical/current state,
    3. snapshot the concrete varying fragment parameters for execution/recording.

    The function intentionally mutates the live host-side parameter stores while it
    resolves the point. That preserves the legacy/host-runtime semantics where mapping
    functions may observe current parameter handles directly, and where the next point
    begins from the stores left behind by the previous one.
    """

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
            if isinstance(dependency, ParamHandle) and dependency not in dependency_values:
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
    """Resolve a whole host-chosen batch into concrete parameter payloads."""

    return [
        _resolve_execution_point(point, axes, parameters, parameter_mappings)
        for point in points
    ]


def _apply_resolved_parameter_values(
    parameters: Sequence[BoundScanParameter],
    point: _ResolvedExecutionPoint,
) -> None:
    """Install one already-resolved point's concrete parameter values on the host."""

    for parameter in parameters:
        parameter.handle._store.set_value(point.parameter_values[parameter.key])


class HostScanExperiment(EnvExperiment):
    """Thin ``EnvExperiment`` adapter for the new host-only runtime.

    This is the code-first path. The experiment code supplies the request directly and
    the result behaves like a normal ARTIQ experiment rather than a dashboard-driven
    ndscan submission target.
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
        self._request_spec = (
            request_factory(self.fragment) if callable(request_factory) else request_factory
        )
        self._max_rtio_underflow_retries = max_rtio_underflow_retries
        self._max_transitory_error_retries = max_transitory_error_retries
        self._session = None

    def prepare(self) -> None:
        request, overrides = _resolve_host_scan_request_spec(
            self.fragment,
            self._request_spec,
        )
        self._session = HostScanSession(
            self,
            self.fragment,
            request,
            overrides=overrides,
            max_rtio_underflow_retries=self._max_rtio_underflow_retries,
            max_transitory_error_retries=self._max_transitory_error_retries,
        )

    def run(self) -> None:
        self._session.run()


class HostDashboardScanExperiment(EnvExperiment):
    """Dashboard-driven host-runtime adapter.

    This path publishes ndscan-style submission metadata via ``PARAMS_ARG_KEY`` and
    expects the submitted ``host_scan`` payload to be compiled into a ``ScanRequest``
    during ``prepare()``.
    """

    argument_ui = "ndscan"

    def build(
        self,
        fragment_init,
        *,
        default_request_spec: Any | None = None,
        max_rtio_underflow_retries: int = 3,
        max_transitory_error_retries: int = 10,
    ) -> None:
        self.fragment = fragment_init()
        if default_request_spec is None:
            default_request_spec = HostScanSpec(mode=HostScanGridModeSpec())
        self._default_request_spec = default_request_spec
        self._max_rtio_underflow_retries = max_rtio_underflow_retries
        self._max_transitory_error_retries = max_transitory_error_retries
        self._session = None
        self.args = HostArgumentInterface(self, self.fragment, self._default_request_spec)

    def prepare(self) -> None:
        request, overrides = self.args.resolve_request(
            self.fragment,
            self._default_request_spec,
        )
        self._session = HostScanSession(
            self,
            self.fragment,
            request,
            overrides=overrides,
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


def make_fragment_host_dashboard_scan_exp(
    fragment_class: type[ExpFragment],
    default_request_spec: HostScanSpec | Mapping[str, Any] | None = None,
    *args,
    max_rtio_underflow_retries: int = 3,
    max_transitory_error_retries: int = 10,
) -> type[HostDashboardScanExperiment]:
    """Create a dashboard-driven host-runtime experiment.

    Unlike ``make_fragment_host_scan_exp()``, this entrypoint does not take a request
    factory.  It publishes ndscan submission metadata to the dashboard and expects a
    submitted ``host_scan`` payload to be compiled into a ``ScanRequest`` before the
    run starts.

    ``default_request_spec`` is optional.  When provided as a ``HostScanSpec`` or dict
    transport payload, it becomes the initial ``host_scan`` value shown to the
    dashboard and also serves as a fallback when running headlessly.
    """

    class FragmentHostDashboardScanShim(HostDashboardScanExperiment):
        def build(self):
            super().build(
                lambda: fragment_class(self, [], *args),
                default_request_spec=default_request_spec,
                max_rtio_underflow_retries=max_rtio_underflow_retries,
                max_transitory_error_retries=max_transitory_error_retries,
            )

    FragmentHostDashboardScanShim.__name__ = fragment_class.__name__
    FragmentHostDashboardScanShim.__qualname__ = fragment_class.__name__
    FragmentHostDashboardScanShim.__module__ = fragment_class.__module__
    FragmentHostDashboardScanShim.__doc__ = fragment_class.__doc__
    return FragmentHostDashboardScanShim


def _resolve_host_scan_request_spec(
    fragment: ExpFragment,
    request_spec: Any,
) -> tuple[ScanRequest, dict[str, list[tuple[str, ParamStore]]]]:
    if isinstance(request_spec, ScanRequest):
        return request_spec, {}

    if isinstance(request_spec, HostScanSpec):
        return compile_host_scan_spec(fragment, request_spec)

    if isinstance(request_spec, Mapping):
        return compile_host_scan_schema(fragment, request_spec)

    if (
        isinstance(request_spec, tuple)
        and len(request_spec) == 2
        and isinstance(request_spec[0], ScanRequest)
        and isinstance(request_spec[1], Mapping)
    ):
        return request_spec[0], dict(request_spec[1])

    raise TypeError(
        "Host scan request must be a ScanRequest, a HostScanSpec, a dict schema, "
        "or a pair of (ScanRequest, overrides)"
    )


def _host_request_transport_dict(request_spec: Any) -> dict[str, Any] | None:
    if isinstance(request_spec, HostScanSpec):
        return request_spec.to_dict()
    if isinstance(request_spec, Mapping):
        return dict(request_spec)
    return None


def _merge_override_store_maps(
    *sources: Mapping[str, Sequence[tuple[str, ParamStore]]],
) -> dict[str, list[tuple[str, ParamStore]]]:
    """Merge override maps, letting later sources replace the same ``(fqn, path)``."""

    merged: dict[str, OrderedDict[str, ParamStore]] = {}
    for source in sources:
        for fqn, pairs in source.items():
            target = merged.setdefault(fqn, OrderedDict())
            for path, store in pairs:
                target[path] = store
    return {
        fqn: list(path_map.items())
        for fqn, path_map in merged.items()
    }

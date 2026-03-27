"""Execution context and preview coordination for the prepared runtime."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from weakref import WeakSet

import h5py
from sipyco import pyon

from artiq import __version__ as artiq_version
from artiq.language import HasEnvironment

from ..scan.request import PreviewPolicy
from ..schema.scan_site import ScanSite
from .persistence import ScanSiteDatasetWriter

__all__ = [
    "ActiveScanContext",
    "RunContext",
    "current_scan_context",
    "current_run_context",
    "make_child_scan_site",
]


@dataclass(frozen=True)
class ActiveScanContext:
    """Execution context for the point currently being run."""

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


class PreviewCoordinator:
    """Root-scoped coordination for preview HDF5 snapshots."""

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
        self._writers.add(writer)

    def unregister_writer(self, writer: ScanSiteDatasetWriter) -> None:
        self._writers.discard(writer)

    def maybe_write_preview(self) -> None:
        if self._write_in_progress:
            return
        now = time.monotonic()
        if now - self._last_preview_monotonic < self.policy.min_interval_s:
            return
        self._write_preview(preview_complete=False, monotonic_time=now)

    def write_completion_preview(self) -> None:
        if self.policy.remove_on_completion:
            self.remove_preview()
            return
        if not self.policy.write_on_completion or self._write_in_progress:
            return
        self._write_preview(preview_complete=True, monotonic_time=time.monotonic())

    def remove_preview(self) -> None:
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
    """Root-scoped coordination state shared by nested runtime sessions."""

    run_start_unix_time: float
    preview: PreviewCoordinator | None = None


_active_run_context: ContextVar[RunContext | None] = ContextVar(
    "_active_run_context", default=None
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
    if stack:
        return stack[-1].snapshot()
    return None


def current_run_context() -> RunContext | None:
    """Return the current root run context, if any."""

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
    extra_metadata: dict[str, Any] | None = None,
) -> ScanSite:
    """Create a child scan site nested under the current executing point."""

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

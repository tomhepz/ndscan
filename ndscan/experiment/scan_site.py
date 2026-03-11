"""Scan-site dataset layout helpers for the new host-only runtime.

The legacy runtime spreads scan dataset layout across several modules via sink setup.
This module makes the dataset layout explicit instead:

- ``ScanSite`` describes *where* a scan writes its data,
- ``ScanSiteDatasetWriter`` is the sole owner of scan-site dataset keys.

The writer is deliberately plain Python. It sits at the edge of the ARTIQ world by
accepting a ``HasEnvironment`` owner, but the policy for *when* points are written is
kept in the host runtime.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from artiq.language import HasEnvironment

from ..utils import SCHEMA_REVISION, SCHEMA_REVISION_KEY
from .result_channels import AppendingDatasetSink, ScalarDatasetSink
from .utils import dump_json, to_metadata_broadcast_type

if TYPE_CHECKING:
    from .host_runtime import PointObservation

__all__ = [
    "ScanSite",
    "make_scan_site_prefix",
    "ScanSiteDatasetWriter",
]


@dataclass(frozen=True)
class ScanSite:
    """Description of a scan site's place in the dataset tree.

    ``path`` is structural, not cosmetic. The intention is that a top-level scan lives
    at ``()`` and nested scan sites later use child paths like ``("cooling",)`` or
    ``("cooling", "probe")``.

    ``parent_path`` is optional metadata describing which other scan site this site is
    nested under. Child scan sites still have their own full ``path``; the parent path
    is kept separately so consumers do not need to reverse-engineer hierarchy from the
    dataset prefix.

    ``dataset_prefix`` is an escape hatch for callers that need an explicit prefix.
    The first host-only runtime uses it rarely; most callers should rely on the
    canonical ``ndscan.rid_<rid>.site.root...`` convention instead.
    """

    path: tuple[str, ...] = ()
    parent_path: tuple[str, ...] | None = None
    dataset_prefix: str | None = None
    segmented: bool = False
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


def _normalise_site_component(name: str) -> str:
    """Return a dataset-safe site path component.

    Fragment paths are already identifier-like in normal ndscan usage, so this is a
    conservative safety net rather than a naming scheme in its own right.
    """

    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return cleaned or "unnamed"


def make_scan_site_prefix(owner: HasEnvironment, site: ScanSite) -> str:
    """Return the dataset prefix used by ``ScanSiteDatasetWriter``.

    By default the new runtime writes under ``ndscan.rid_<rid>.site.root.`` and child
    sites extend that path with structural site components.
    """

    if site.dataset_prefix is not None:
        return site.dataset_prefix if site.dataset_prefix.endswith(".") else site.dataset_prefix + "."

    scheduler = owner.get_device("scheduler")
    rid = getattr(scheduler, "rid", 0)
    parts = ["ndscan", f"rid_{rid}", "site", "root"]
    parts.extend(_normalise_site_component(part) for part in site.path)
    return ".".join(parts) + "."


class ScanSiteDatasetWriter:
    """Write one scan site's metadata and append-only point data.

    The writer keeps only enough mutable state to manage append-only datasets and
    open/closed segment tracking. It does not know how points are chosen or how a
    fragment is executed.

    The first implementation writes each update through to the dataset manager
    immediately. ``flush()``/``close()`` are nevertheless part of the interface from
    the start: future batching should live here rather than leaking into the runtime or
    fragment APIs.
    """

    def __init__(self, owner: HasEnvironment, site: ScanSite):
        self._owner = owner
        self._site = site
        self.prefix = make_scan_site_prefix(owner, site)

        self._point_sinks = dict[str, AppendingDatasetSink]()
        self._analysis_result_sinks = dict[str, ScalarDatasetSink]()
        self._starts_sink = (
            self._make_appending_sink("starts") if site.segmented else None
        )
        self._parent_point_sink = (
            self._make_appending_sink("parent_point_indices")
            if site.segmented and site.parent_path is not None
            else None
        )
        self._current_segment_sink = (
            ScalarDatasetSink(owner, self.prefix + "current_segment") if site.segmented else None
        )

        self._next_point_index = self._get_existing_scalar("num_points", 0)
        self._num_segments = len(self._get_existing_array("starts")) if site.segmented else 0
        self._current_segment = (
            self._get_existing_scalar("current_segment", -1) if site.segmented else -1
        )

    def publish_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Publish site metadata before points start arriving."""

        self._push_scalar(SCHEMA_REVISION_KEY, SCHEMA_REVISION)

        source_prefix = self._owner.get_dataset("system_id", default="rid")
        scheduler = self._owner.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        self._push_scalar("source_id", f"{source_prefix}_{rid}")

        base_metadata = {
            "runtime_flavour": "host_runtime_v2",
            "site_path": list(self._site.path),
            "completed": False,
            "analysis_results": {},
            "annotations": [],
            "online_analyses": {},
        }
        if self._site.parent_path is not None:
            base_metadata["parent_site_path"] = list(self._site.parent_path)
        if self._site.segmented:
            base_metadata["segment_fields"] = {"starts": "starts"}
            if self._parent_point_sink is not None:
                base_metadata["segment_fields"]["parent_points"] = "parent_point_indices"
            base_metadata["segment_state_fields"] = {"current": "current_segment"}

        merged = dict(base_metadata)
        merged.update(metadata)
        merged.update(self._site.extra_metadata)

        for key, value in merged.items():
            self._push_scalar(key, value)

        if self._site.segmented:
            self._push_scalar("current_segment", -1)
        self._push_scalar("num_points", self._next_point_index)

    @property
    def next_point_index(self) -> int:
        """Return the flat point index that the next appended point will receive."""

        return self._next_point_index

    def start_segment(self, parent_point_index: int | None = None) -> None:
        """Mark the current point index as the start of a new logical segment."""

        if not self._site.segmented:
            return
        if self._current_segment != -1:
            return

        self._starts_sink.push(self._next_point_index)
        if self._parent_point_sink is not None:
            if parent_point_index is None:
                raise ValueError(
                    "Nested segmented scan sites require a parent point index"
                )
            self._parent_point_sink.push(parent_point_index)
        self._current_segment = self._num_segments
        self._num_segments += 1
        self._current_segment_sink.push(self._current_segment)

    def finish_segment(self) -> None:
        """Mark the current segment as closed.

        The points remain in the flat append-only arrays. ``current_segment`` only
        describes whether a segment is still open right now.
        """

        if not self._site.segmented:
            return
        self._current_segment = -1
        self._current_segment_sink.push(-1)

    def append_observation(self, observation: "PointObservation") -> None:
        """Append one completed point to the site's point datasets."""

        for key, value in observation.axis_values.items():
            self._get_point_sink(key).push(value)
        for key, value in observation.channel_values.items():
            self._get_point_sink(key).push(value)
        self._next_point_index += 1
        self._push_scalar("num_points", self._next_point_index)

    def set_completed(self, completed: bool = True) -> None:
        self._push_scalar("completed", completed)

    def set_annotations(self, annotations: list[dict[str, Any]]) -> None:
        self._push_scalar("annotations", annotations)

    def set_analysis_result(self, key: str, value: Any) -> None:
        sink = self._analysis_result_sinks.get(key, None)
        if sink is None:
            sink = ScalarDatasetSink(self._owner, self.prefix + "analysis_result." + key)
            self._analysis_result_sinks[key] = sink
        sink.push(value)

    def flush(self) -> None:
        """Flush any buffered writes to the dataset manager.

        This writer currently pushes every update immediately, so ``flush()`` is a
        no-op today. The method exists intentionally to reserve a clean extension point
        for future batching strategies such as "flush every N points" or "flush on a
        timer", without forcing callers to change shape later on.
        """

    def close(self) -> None:
        """Finalize the writer after the scan has finished.

        Today this is just ``flush()``. Keeping a separate method makes the intended
        lifecycle explicit and leaves room for future buffered implementations to do
        any end-of-run draining in one place.
        """
        self.flush()

    def _get_point_sink(self, point_key: str) -> AppendingDatasetSink:
        sink = self._point_sinks.get(point_key, None)
        if sink is None:
            sink = self._make_appending_sink("points." + point_key)
            self._point_sinks[point_key] = sink
        return sink

    def _make_appending_sink(self, relative_key: str) -> AppendingDatasetSink:
        sink = AppendingDatasetSink(self._owner, self.prefix + relative_key)
        existing = self._get_existing_array(relative_key)
        if existing:
            sink.last_value = existing[-1]
        return sink

    def _get_existing_array(self, relative_key: str) -> list[Any]:
        try:
            return self._owner.get_dataset(self.prefix + relative_key)
        except KeyError:
            return []

    def _get_existing_scalar(self, relative_key: str, default: Any) -> Any:
        try:
            return self._owner.get_dataset(self.prefix + relative_key)
        except KeyError:
            return default

    def _push_scalar(self, key: str, value: Any) -> None:
        ds_value = to_metadata_broadcast_type(value)
        self._owner.set_dataset(
            self.prefix + key,
            dump_json(value) if ds_value is None else ds_value,
            broadcast=True,
        )

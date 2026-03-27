"""Scan-site dataset writing helpers for the prepared runtime.

The persisted scan-site contract lives in ``ndscan.schema.scan_site``. This module
owns only the runtime-side writer implementation.

The writer is deliberately plain Python. It sits at the edge of the ARTIQ world by
accepting a ``HasEnvironment`` owner, but the policy for *when* points are written is
kept in the host runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from artiq.language import HasEnvironment

from ..define.result_channels import AppendingDatasetSink, ScalarDatasetSink
from ..schema.scan_site import (
    SCAN_SITE_SCHEMA_REVISION,
    ScanSite,
    make_scan_site_prefix,
)
from ..define.utils import dump_json, to_metadata_broadcast_type
from ..utils import SCHEMA_REVISION_KEY

if TYPE_CHECKING:
    from .program import PointObservation

__all__ = ["ScanSiteDatasetWriter"]


class ScanSiteDatasetWriter:
    """Write one scan site's metadata and append-only point data.

    The writer keeps only enough mutable state to manage append-only datasets and
    open/closed segment tracking. It does not know how points are chosen or how a
    fragment is executed.

    The first implementation writes each update through to the dataset manager
    immediately. ``flush()``/``close()`` are nevertheless part of the interface from
    the start: future batching should live here rather than leaking into the runtime or
    fragment APIs.

    Point-like datasets are intentionally split by role:

    - ``points.pseudoparam_*`` for logical runtime-only scan variables,
    - ``points.param_*`` for actual fragment parameters whose values varied,
    - ``points.channel_*`` for result channels.
    """

    def __init__(self, owner: HasEnvironment, site: ScanSite):
        self._owner = owner
        self._site = site
        self.prefix = make_scan_site_prefix(owner, site)

        self._point_sinks = dict[str, AppendingDatasetSink]()
        self._analysis_result_sinks = dict[str, ScalarDatasetSink]()
        self._starts_sink = (
            self._make_appending_sink("segments.start_index") if site.segmented else None
        )
        self._segment_start_time_sink = (
            self._make_appending_sink("segments.start_unix_time")
            if site.segmented
            else None
        )
        self._parent_point_sink = (
            self._make_appending_sink("segments.parent_point_index")
            if site.segmented and site.parent_path is not None
            else None
        )
        self._current_segment_sink = (
            ScalarDatasetSink(owner, self.prefix + "state.current_segment")
            if site.segmented
            else None
        )

        self._next_point_index = self._get_existing_scalar(
            ("state.num_points", "num_points"), 0
        )
        self._num_segments = (
            len(self._get_existing_array(("segments.start_index", "starts")))
            if site.segmented
            else 0
        )
        self._current_segment = (
            self._get_existing_scalar(("state.current_segment", "current_segment"), -1)
            if site.segmented
            else -1
        )

    def publish_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        extra_metadata: Mapping[str, Any] | None = None,
        start_unix_time: float | None = None,
    ) -> None:
        """Publish site metadata before points start arriving.

        ``start_unix_time`` is the wall-clock Unix timestamp for this concrete site
        invocation. Segmented sites also record per-segment start times separately.
        """

        self._push_scalar(SCHEMA_REVISION_KEY, SCAN_SITE_SCHEMA_REVISION)

        source_prefix = self._owner.get_dataset("system_id", default="rid")
        scheduler = self._owner.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        self._push_scalar("site.source_id", f"{source_prefix}_{rid}")

        base_metadata = {
            "site.path": list(self._site.path),
            "state.completed": False,
        }
        if self._site.parent_path is not None:
            base_metadata["site.parent_path"] = list(self._site.parent_path)

        merged = dict(base_metadata)
        merged.update(metadata)
        if start_unix_time is not None:
            merged["site.start_unix_time"] = start_unix_time

        for key, value in merged.items():
            self._push_scalar(key, value)

        extras = dict(extra_metadata or {})
        extras.update(self._site.extra_metadata)
        for key, value in extras.items():
            self._push_scalar("extra." + key, value)

        if self._site.segmented:
            self._push_scalar("state.current_segment", -1)
        self._push_scalar("state.num_points", self._next_point_index)

    @property
    def next_point_index(self) -> int:
        """Return the flat point index that the next appended point will receive."""

        return self._next_point_index

    def start_segment(
        self,
        parent_point_index: int | None = None,
        *,
        start_unix_time: float | None = None,
    ) -> None:
        """Mark the current point index as the start of a new logical segment.

        When provided, ``start_unix_time`` is appended alongside the segment start
        index so repeated nested runs can be related back to wall-clock time later.
        """

        if not self._site.segmented:
            return
        if self._current_segment != -1:
            return

        self._starts_sink.push(self._next_point_index)
        if self._segment_start_time_sink is not None and start_unix_time is not None:
            self._segment_start_time_sink.push(start_unix_time)
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
        """Append one completed point to the site's point datasets.

        Point acquisition timestamps are optional, but when present they are stored as
        the dedicated `points.acquired_at_unix` stream rather than being mixed into the
        ordinary axis/channel payload arrays.
        """
        self.append_observations((observation,))

    def append_observations(
        self, observations: Sequence["PointObservation"]
    ) -> None:
        """Append a completed execution batch to the site's point datasets.

        The first writer implementation still pushes individual point payloads through
        immediately, but batching them in one method keeps the runtime boundary
        explicit and lets future buffered writers update batch-level bookkeeping only
        once per completed batch.
        """

        if not observations:
            return

        for observation in observations:
            for key, value in observation.pseudoparam_values.items():
                self._get_point_sink(key).push(value)
            for key, value in observation.parameter_values.items():
                self._get_point_sink(key).push(value)
            for key, value in observation.channel_values.items():
                self._get_point_sink(key).push(value)
            for key, value in observation.point_metadata.items():
                self._get_point_sink("metadata." + key).push(value)
            if observation.acquired_at_unix is not None:
                self._get_point_sink("acquired_at_unix").push(
                    observation.acquired_at_unix
                )
            self._next_point_index += 1
        self._push_scalar("state.num_points", self._next_point_index)

    def set_completed(self, completed: bool = True) -> None:
        self._push_scalar("state.completed", completed)

    def set_annotations(self, annotations: list[dict[str, Any]]) -> None:
        self._push_scalar("analysis.annotations", annotations)

    def set_analysis_result(self, key: str, value: Any) -> None:
        sink = self._analysis_result_sinks.get(key, None)
        if sink is None:
            sink = ScalarDatasetSink(self._owner, self.prefix + "analysis.output." + key)
            self._analysis_result_sinks[key] = sink
        sink.push(value)

    def set_online_analysis_result(self, key: str, value: Any) -> None:
        """Publish the latest value for one online analysis.

        Online analyses are batch-updated, not append-only histories. Each dataset
        stores the latest result object for one named online analysis and is rewritten
        whenever that analysis is re-evaluated on accumulated scan data.
        """

        self._push_scalar("analysis.online_result." + key, value)

    def set_online_analysis_annotations(
        self, key: str, annotations: list[dict[str, Any]]
    ) -> None:
        """Publish the latest annotations for one online analysis.

        Online annotations are rewritten at each batch boundary just like the matching
        online outputs. This keeps the write-side contract symmetric and lets adaptive
        readers treat online analyses as "latest snapshot" state rather than as a
        second append-only history.
        """

        self._push_scalar("analysis.online_annotation." + key, annotations)

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

    def _get_existing_array(self, relative_keys: str | tuple[str, ...]) -> list[Any]:
        keys = (relative_keys,) if isinstance(relative_keys, str) else relative_keys
        for key in keys:
            try:
                return self._owner.get_dataset(self.prefix + key)
            except KeyError:
                continue
        return []

    def _get_existing_scalar(
        self, relative_keys: str | tuple[str, ...], default: Any
    ) -> Any:
        keys = (relative_keys,) if isinstance(relative_keys, str) else relative_keys
        for key in keys:
            try:
                return self._owner.get_dataset(self.prefix + key)
            except KeyError:
                continue
        return default

    def _push_scalar(self, key: str, value: Any) -> None:
        ds_value = to_metadata_broadcast_type(value)
        self._owner.set_dataset(
            self.prefix + key,
            dump_json(value) if ds_value is None else ds_value,
            broadcast=True,
        )

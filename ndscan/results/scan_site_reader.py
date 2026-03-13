"""Read host-runtime scan-site HDF5 snapshots without ARTIQ.

This module is intentionally small. It understands the current host-runtime scan-site
schema and returns plain Python dataclasses so offline tools can inspect preview or
final HDF5 snapshots without importing the experiment runtime itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = [
    "HostRuntimeSnapshot",
    "HostRuntimeSiteSegment",
    "HostRuntimeSiteData",
    "read_host_runtime_snapshot",
]


_STRUCTURED_KEYS = {
    "site.path",
    "site.parent_path",
    "scan.point_source",
    "scan.pseudoparams",
    "scan.parameters",
    "scan.fixed_parameters",
    "scan.channels",
    "scan.parameter_mappings",
    "analysis.online",
    "analysis.outputs",
    "analysis.annotations",
}


def _decode_dataset_value(key: str, raw: Any) -> Any:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, str):
        if any(key.endswith(suffix) for suffix in _STRUCTURED_KEYS) or ".analysis.online_result." in key:
            return json.loads(raw)
        if ".analysis.online_annotation." in key:
            return json.loads(raw)
    return raw


def _read_datasets_group(group: h5py.Group) -> dict[str, Any]:
    datasets = {}
    for key, dataset in group.items():
        datasets[key] = _decode_dataset_value(key, dataset[()])
    return datasets


def _find_site_prefixes(dataset_values: dict[str, Any]) -> list[str]:
    suffix = "site.path"
    prefixes = [
        key[: -len(suffix)]
        for key in dataset_values.keys()
        if key.endswith(suffix)
    ]
    return sorted(prefixes, key=lambda prefix: (prefix.count("."), prefix))


def _point_keys_for_prefix(dataset_values: dict[str, Any], prefix: str) -> dict[str, Any]:
    point_prefix = prefix + "points."
    return {
        key[len(point_prefix) :]: value
        for key, value in dataset_values.items()
        if key.startswith(point_prefix)
    }


def _analysis_outputs_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> dict[str, Any]:
    analysis_prefix = prefix + "analysis.output."
    return {
        key[len(analysis_prefix) :]: value
        for key, value in dataset_values.items()
        if key.startswith(analysis_prefix)
    }


def _online_results_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> dict[str, Any]:
    analysis_prefix = prefix + "analysis.online_result."
    return {
        key[len(analysis_prefix) :]: value
        for key, value in dataset_values.items()
        if key.startswith(analysis_prefix)
    }


def _online_annotations_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> dict[str, list[dict[str, Any]]]:
    analysis_prefix = prefix + "analysis.online_annotation."
    return {
        key[len(analysis_prefix) :]: value
        for key, value in dataset_values.items()
        if key.startswith(analysis_prefix)
    }


@dataclass(frozen=True)
class HostRuntimeSiteSegment:
    """One logical segment within a segmented scan site."""

    index: int
    start_index: int
    stop_index: int
    parent_point_index: int | None = None
    start_unix_time: float | None = None

    @property
    def length(self) -> int:
        return self.stop_index - self.start_index


@dataclass(frozen=True)
class HostRuntimeSiteData:
    """Offline view of one host-runtime scan site."""

    prefix: str
    path: tuple[str, ...]
    parent_path: tuple[str, ...] | None
    fragment_fqn: str
    pseudoparams: dict[str, Any]
    parameters: dict[str, Any]
    fixed_parameters: dict[str, Any]
    channels: dict[str, Any]
    point_data: dict[str, Any]
    analysis_outputs_schema: dict[str, Any]
    analysis_outputs: dict[str, Any]
    online_analysis_schema: dict[str, Any]
    online_analysis_results: dict[str, Any]
    online_analysis_annotations: dict[str, list[dict[str, Any]]]
    annotations: list[dict[str, Any]]
    segmented: bool
    metadata: dict[str, Any]

    @property
    def num_points(self) -> int:
        return int(self.metadata.get("state.num_points", 0))

    def choose_default_x_key(self) -> tuple[str | None, str | None]:
        """Return a simple default x-data choice for plotting.

        Preference order:
        1. first pseudoparam
        2. first directly scanned parameter
        3. first parameter
        4. no x-data available
        """

        if self.pseudoparams:
            return "pseudoparam", next(iter(self.pseudoparams))

        for key, schema in self.parameters.items():
            if schema.get("is_scanned", False):
                return "param", key

        if self.parameters:
            return "param", next(iter(self.parameters))

        return None, None

    def segments(self) -> list[HostRuntimeSiteSegment]:
        """Return the site's logical segments in flat point-index space."""

        if not self.segmented:
            return []

        starts = list(self.metadata.get("segments.start_index", []))
        parent_point_indices = self.metadata.get("segments.parent_point_index", [])
        start_unix_times = self.metadata.get("segments.start_unix_time", [])

        result = []
        for index, start_index in enumerate(starts):
            stop_index = (
                starts[index + 1] if index + 1 < len(starts) else self.num_points
            )
            parent_point_index = (
                int(parent_point_indices[index])
                if index < len(parent_point_indices)
                else None
            )
            start_unix_time = (
                float(start_unix_times[index])
                if index < len(start_unix_times)
                else None
            )
            result.append(
                HostRuntimeSiteSegment(
                    index=index,
                    start_index=int(start_index),
                    stop_index=int(stop_index),
                    parent_point_index=parent_point_index,
                    start_unix_time=start_unix_time,
                )
            )
        return result

    def segments_for_parent_point(
        self, parent_point_index: int
    ) -> list[HostRuntimeSiteSegment]:
        """Return all child segments launched from one parent point index."""

        return [
            segment
            for segment in self.segments()
            if segment.parent_point_index == parent_point_index
        ]

    def slice_point_data(
        self, start_index: int, stop_index: int
    ) -> dict[str, list[Any]]:
        """Return point data sliced to a sub-range of this site's flat arrays."""

        return {
            key: list(values[start_index:stop_index])
            for key, values in self.point_data.items()
        }


@dataclass(frozen=True)
class HostRuntimeSnapshot:
    """Offline view of one host-runtime HDF5 snapshot file."""

    path: Path
    top_level_metadata: dict[str, Any]
    sites: dict[tuple[str, ...], HostRuntimeSiteData]

    def get_site(self, path: tuple[str, ...] = ()) -> HostRuntimeSiteData:
        return self.sites[path]

    def child_sites(self, parent_path: tuple[str, ...] = ()) -> list[HostRuntimeSiteData]:
        """Return the immediate child sites of the given parent site path."""

        return sorted(
            (
                site
                for site in self.sites.values()
                if site.parent_path == parent_path
            ),
            key=lambda site: site.path,
        )


def read_host_runtime_snapshot(path: str | Path) -> HostRuntimeSnapshot:
    """Read a host-runtime preview or final HDF5 snapshot."""

    file_path = Path(path)
    with h5py.File(file_path, "r") as h5_file:
        datasets = _read_datasets_group(h5_file["datasets"])
        top_level_metadata = {
            key: _decode_dataset_value(key, dataset[()])
            for key, dataset in h5_file.items()
            if key != "datasets" and key != "archive"
        }

    sites = {}
    for prefix in _find_site_prefixes(datasets):
        path_value = tuple(datasets[prefix + "site.path"])
        parent_path = datasets.get(prefix + "site.parent_path")
        sites[path_value] = HostRuntimeSiteData(
            prefix=prefix,
            path=path_value,
            parent_path=None if parent_path is None else tuple(parent_path),
            fragment_fqn=datasets[prefix + "site.fragment_fqn"],
            pseudoparams=datasets.get(prefix + "scan.pseudoparams", {}),
            parameters=datasets.get(prefix + "scan.parameters", {}),
            fixed_parameters=datasets.get(prefix + "scan.fixed_parameters", {}),
            channels=datasets.get(prefix + "scan.channels", {}),
            point_data=_point_keys_for_prefix(datasets, prefix),
            analysis_outputs_schema=datasets.get(prefix + "analysis.outputs", {}),
            analysis_outputs=_analysis_outputs_for_prefix(datasets, prefix),
            online_analysis_schema=datasets.get(prefix + "analysis.online", {}),
            online_analysis_results=_online_results_for_prefix(datasets, prefix),
            online_analysis_annotations=_online_annotations_for_prefix(datasets, prefix),
            annotations=datasets.get(prefix + "analysis.annotations", []),
            segmented=(prefix + "segments.start_index") in datasets,
            metadata={
                key[len(prefix) :]: value
                for key, value in datasets.items()
                if key.startswith(prefix)
                and not key.startswith(prefix + "points.")
                and not key.startswith(prefix + "analysis.output.")
                and not key.startswith(prefix + "analysis.online_result.")
                and not key.startswith(prefix + "analysis.online_annotation.")
            },
        )

    return HostRuntimeSnapshot(
        path=file_path,
        top_level_metadata=top_level_metadata,
        sites=sites,
    )

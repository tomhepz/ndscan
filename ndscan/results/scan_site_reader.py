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
    "HostRuntimeSiteData",
    "read_host_runtime_snapshot",
]


_STRUCTURED_KEYS = {
    "site.path",
    "site.parent_path",
    "scan.point_source",
    "scan.pseudoparams",
    "scan.parameters",
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
class HostRuntimeSiteData:
    """Offline view of one host-runtime scan site."""

    prefix: str
    path: tuple[str, ...]
    parent_path: tuple[str, ...] | None
    fragment_fqn: str
    pseudoparams: dict[str, Any]
    parameters: dict[str, Any]
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


@dataclass(frozen=True)
class HostRuntimeSnapshot:
    """Offline view of one host-runtime HDF5 snapshot file."""

    path: Path
    top_level_metadata: dict[str, Any]
    sites: dict[tuple[str, ...], HostRuntimeSiteData]

    def get_site(self, path: tuple[str, ...] = ()) -> HostRuntimeSiteData:
        return self.sites[path]


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

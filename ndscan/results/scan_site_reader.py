"""Read host-runtime scan-site HDF5 snapshots without ARTIQ.

This module is intentionally small. It understands the current host-runtime scan-site
schema and returns plain Python dataclasses so offline tools can inspect preview or
final HDF5 snapshots without importing the experiment runtime itself.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = [
    "HostRuntimeSnapshot",
    "HostRuntimeSiteSegment",
    "HostRuntimeSegmentFinalAnalysis",
    "HostRuntimeSite",
    "read_host_runtime_snapshot",
]


_STRUCTURED_KEYS = {
    "site.path",
    "site.parent_path",
    "scan.point_policy",
    "scan.pseudoparams",
    "scan.fixed_pseudoparams",
    "scan.parameters",
    "scan.fixed_parameters",
    "scan.channels",
    "scan.parameter_mappings",
    "analysis.online",
    "analysis.outputs",
    "analysis.annotations",
}
_STRUCTURED_ARRAY_KEYS = {
    "segments.analysis.final_feedback",
}
_POINT_INDEX_PATH = "point_index"
_POINT_INDEX_ALIASES = {_POINT_INDEX_PATH, "__point_index__"}


def _join_path(base: str, name: str) -> str:
    base = base.strip("/")
    name = name.strip("/")
    if base and name:
        return base + "/" + name
    if name:
        return name
    return base


def _parameter_series_path(key: str, schema: Mapping[str, Any]) -> str:
    owner_path = str(schema.get("path", ""))
    param_schema = schema.get("param", {})
    if isinstance(param_schema, Mapping):
        fqn = param_schema.get("fqn")
        if isinstance(fqn, str) and fqn:
            return _join_path(owner_path, fqn.rsplit(".", 1)[-1])
    return owner_path or key


def _pseudoparam_series_path(key: str, schema: Mapping[str, Any]) -> str:
    base_path = str(schema.get("path", ""))
    variable_schema = schema.get("variable", {})
    if isinstance(variable_schema, Mapping):
        name = variable_schema.get("name")
        if isinstance(name, str) and name:
            return _join_path(base_path, name)
    return base_path or key


def _channel_series_path(key: str, schema: Mapping[str, Any]) -> str:
    path = schema.get("path")
    return str(path) if isinstance(path, str) and path else key


def _decode_json_string_list(values: list[Any]) -> list[Any]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            decoded.append(json.loads(value))
        else:
            decoded.append(value)
    return decoded


def _decode_dataset_value(key: str, raw: Any) -> Any:
    if isinstance(raw, np.ndarray):
        if raw.dtype.kind == "S":
            decoded = [item.decode("utf-8") for item in raw.tolist()]
            if any(key.endswith(suffix) for suffix in _STRUCTURED_ARRAY_KEYS):
                return _decode_json_string_list(decoded)
            return decoded
        if raw.dtype.kind == "U":
            decoded = raw.tolist()
            if any(key.endswith(suffix) for suffix in _STRUCTURED_ARRAY_KEYS):
                return _decode_json_string_list(decoded)
            return decoded
        if raw.dtype.kind == "O":
            decoded = []
            changed = False
            for item in raw.tolist():
                if isinstance(item, bytes):
                    decoded.append(item.decode("utf-8"))
                    changed = True
                elif isinstance(item, np.generic):
                    decoded.append(item.item())
                    changed = True
                else:
                    decoded.append(item)
            if any(key.endswith(suffix) for suffix in _STRUCTURED_ARRAY_KEYS):
                return _decode_json_string_list(decoded if changed else raw.tolist())
            return decoded if changed else raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, str):
        if (
            any(key.endswith(suffix) for suffix in _STRUCTURED_KEYS)
            or key.startswith("analysis.online_result.")
            or key.startswith("analysis.artifact.")
            or key.startswith("analysis.online_artifact.")
            or ".analysis.online_result." in key
            or ".analysis.artifact." in key
            or ".analysis.online_artifact." in key
        ):
            return json.loads(raw)
        if key.startswith("analysis.online_annotation.") or ".analysis.online_annotation." in key:
            return json.loads(raw)
        if key.startswith("extra.") or ".extra." in key:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
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


def _analysis_artifacts_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> dict[str, Any]:
    analysis_prefix = prefix + "analysis.artifact."
    return {
        key[len(analysis_prefix) :]: value
        for key, value in dataset_values.items()
        if key.startswith(analysis_prefix)
    }


def _online_artifacts_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> dict[str, Any]:
    analysis_prefix = prefix + "analysis.online_artifact."
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


def _segment_final_analysis_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> list["HostRuntimeSegmentFinalAnalysis"]:
    raw_feedback = dataset_values.get(prefix + "segments.analysis.final_feedback", [])
    return [HostRuntimeSegmentFinalAnalysis.from_dict(item) for item in raw_feedback]


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
class HostRuntimeSegmentFinalAnalysis:
    """Final analysis payload attached to one finished site segment."""

    outputs: dict[str, Any]
    artifacts: dict[str, Any]
    annotations: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | str | bytes) -> "HostRuntimeSegmentFinalAnalysis":
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            raw = json.loads(raw)
        return cls(
            outputs=dict(raw.get("outputs", {})),
            artifacts=dict(raw.get("artifacts", {})),
            annotations=list(raw.get("annotations", [])),
        )


@dataclass(frozen=True)
class HostRuntimeSite:
    """Offline view of one host-runtime scan site."""

    prefix: str
    path: tuple[str, ...]
    parent_path: tuple[str, ...] | None
    fragment_fqn: str
    pseudoparams: dict[str, Any]
    fixed_pseudoparams: dict[str, Any]
    parameters: dict[str, Any]
    fixed_parameters: dict[str, Any]
    channels: dict[str, Any]
    raw_points: dict[str, Any]
    analysis_outputs_schema: dict[str, Any]
    analysis_outputs: dict[str, Any]
    analysis_artifacts: dict[str, Any]
    online_analysis_schema: dict[str, Any]
    online_analysis_results: dict[str, Any]
    online_analysis_artifacts: dict[str, Any]
    online_analysis_annotations: dict[str, list[dict[str, Any]]]
    annotations: list[dict[str, Any]]
    segment_final_analysis: list[HostRuntimeSegmentFinalAnalysis]
    segmented: bool
    metadata: dict[str, Any]

    @property
    def num_points(self) -> int:
        return int(self.metadata.get("state.num_points", 0))

    def available_pseudoparam_paths(self) -> list[str]:
        """Return the semantic pseudoparameter paths available on this site."""

        return [
            _pseudoparam_series_path(key, schema)
            for key, schema in self.pseudoparams.items()
        ]

    def available_parameter_paths(self) -> list[str]:
        """Return the semantic parameter paths available on this site."""

        return [
            _parameter_series_path(key, schema)
            for key, schema in self.parameters.items()
        ]

    def available_channel_paths(self) -> list[str]:
        """Return the semantic channel paths available on this site."""

        return [
            _channel_series_path(key, schema)
            for key, schema in self.channels.items()
        ]

    def available_series_paths(self) -> list[str]:
        """Return all plottable per-point series paths for this site."""

        paths = []
        semantic_keys = set()
        for path, _, key, _ in self._semantic_series_entries():
            if key in self.raw_points:
                paths.append(path)
                semantic_keys.add(key)

        paths.extend(sorted(key for key in self.raw_points if key not in semantic_keys))
        if _POINT_INDEX_PATH not in paths:
            paths.append(_POINT_INDEX_PATH)
        return paths

    def _semantic_series_entries(self):
        schema_groups = (
            ("pseudoparam", self.pseudoparams, _pseudoparam_series_path),
            ("parameter", self.parameters, _parameter_series_path),
            ("channel", self.channels, _channel_series_path),
        )
        for kind, schemas, path_fn in schema_groups:
            for key, schema in schemas.items():
                yield path_fn(key, schema), kind, key, schema

    def _matching_series_keys_for_path(self, path: str) -> list[tuple[str, str]]:
        matches: list[tuple[str, str]] = []
        for key, schema in self.pseudoparams.items():
            if _pseudoparam_series_path(key, schema) == path:
                matches.append(("pseudoparam", key))
        for key, schema in self.parameters.items():
            if _parameter_series_path(key, schema) == path:
                matches.append(("param", key))
        for key, schema in self.channels.items():
            if _channel_series_path(key, schema) == path:
                matches.append(("channel", key))
        return matches

    def get_channel_storage_key(self, channel_path: str) -> str | None:
        """Return the storage key for one semantic channel path, if present."""

        for key, schema in self.channels.items():
            if _channel_series_path(key, schema) == channel_path:
                return key
        return None

    def require_channel_storage_key(self, channel_path: str) -> str:
        """Return the storage key for one semantic channel path.

        Prepared-runtime series arrays are stored under machine-oriented keys such as
        ``channel_0`` while the channel schema exposes stable semantic paths such as
        ``result`` or ``shot/image0``. This helper bridges between the two.
        """

        key = self.get_channel_storage_key(channel_path)
        if key is None:
            raise KeyError(
                f"Site {'/'.join(self.path) or '<root>'} does not define channel path "
                f"{channel_path!r}; available paths are {self.available_channel_paths()}"
            )
        return key

    def get_series_storage_key(self, path: str) -> str | None:
        """Return the saved storage key for one public series path, if present.

        The path may refer to a pseudoparameter, parameter, channel, or direct runtime
        series such as ``acquired_at_unix`` or ``metadata.decision_source``.
        """

        matches = self._matching_series_keys_for_path(path)
        if len(matches) > 1:
            raise KeyError(
                f"Site {'/'.join(self.path) or '<root>'} has ambiguous series path "
                f"{path!r}: {matches}"
            )
        if matches:
            return matches[0][1]
        if path in self.raw_points:
            return path
        return None

    def require_series_storage_key(self, path: str) -> str:
        """Return the saved storage key for one public series path."""

        key = self.get_series_storage_key(path)
        if key is None:
            raise KeyError(
                f"Site {'/'.join(self.path) or '<root>'} does not define series path "
                f"{path!r}; available paths are {self.available_series_paths()}"
            )
        return key

    def _series_for_path(self, path: str) -> Any:
        """Return one saved or synthetic series by public series path."""

        if path in _POINT_INDEX_ALIASES:
            return np.arange(self.num_points, dtype=np.int64)
        return self.raw_points[self.require_series_storage_key(path)]

    def series(self, path: str, *, dtype: Any | None = None) -> np.ndarray:
        """Return one saved series as a NumPy array by public series path.

        The path may name a scanned parameter, pseudoparameter, result channel, direct
        runtime series, or synthetic stream such as ``point_index``. This is the
        simplest API for offline plotting code that wants to treat all saved series
        uniformly.
        """

        values = np.asarray(self._series_for_path(path))
        if dtype is None:
            return values
        return values.astype(dtype, copy=False)

    def describe_series(self) -> list[dict[str, Any]]:
        """Return a compact description of all plottable per-point series."""

        descriptions = []
        semantic_keys = set()
        for path, kind, key, schema in self._semantic_series_entries():
            if key not in self.raw_points:
                continue
            semantic_keys.add(key)
            values = np.asarray(self.raw_points[key])
            description = {
                "path": path,
                "kind": kind,
                "key": key,
                "shape": tuple(int(size) for size in values.shape),
                "dtype": str(values.dtype),
            }
            if isinstance(schema, Mapping):
                if "shape" in schema:
                    description["point_shape"] = tuple(
                        int(size) for size in schema["shape"]
                    )
                if "dim_names" in schema:
                    description["dim_names"] = tuple(schema["dim_names"])
            descriptions.append(description)

        for key in sorted(self.raw_points):
            if key in semantic_keys:
                continue
            values = np.asarray(self.raw_points[key])
            if key in {"acquired_at_unix", _POINT_INDEX_PATH}:
                kind = "runtime"
            elif key.startswith("metadata."):
                kind = "point_metadata"
            else:
                kind = "raw_points"
            descriptions.append(
                {
                    "path": key,
                    "kind": kind,
                    "key": key,
                    "shape": tuple(int(size) for size in values.shape),
                    "dtype": str(values.dtype),
                }
            )

        if _POINT_INDEX_PATH not in self.raw_points:
            descriptions.append(
                {
                    "path": _POINT_INDEX_PATH,
                    "kind": "runtime",
                    "key": None,
                    "shape": (self.num_points,),
                    "dtype": str(np.dtype(np.int64)),
                }
            )
        return descriptions

    def choose_default_x_path(self) -> str | None:
        """Return a simple default x-data path for plotting."""

        kind, key = self.choose_default_x_key()
        if kind is None or key is None:
            return None
        if kind == "pseudoparam":
            return _pseudoparam_series_path(key, self.pseudoparams[key])
        return _parameter_series_path(key, self.parameters[key])

    def available_metadata_blob_names(self) -> list[str]:
        """Return names of all persisted ``extra.*`` metadata blobs on this site."""

        prefix = "extra."
        return sorted(
            key.removeprefix(prefix)
            for key in self.metadata
            if key.startswith(prefix)
        )

    def metadata_blobs(self) -> dict[str, Any]:
        """Return all persisted ``extra.*`` metadata blobs by public blob name."""

        return {
            name: self.metadata["extra." + name]
            for name in self.available_metadata_blob_names()
        }

    def require_metadata_blob(self, name: str) -> Any:
        """Return one required metadata blob by name.

        The returned value is whatever was recovered from the site metadata, typically a
        decoded JSON object for structured blobs.
        """

        key = "extra." + name
        try:
            return self.metadata[key]
        except KeyError as exc:
            raise KeyError(
                f"Site {'/'.join(self.path) or '<root>'} does not define metadata blob {name!r}"
            ) from exc

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

    def final_analysis_for_segment(
        self, segment_index: int
    ) -> HostRuntimeSegmentFinalAnalysis | None:
        """Return the persisted final analysis payload for one segment, if present."""

        if 0 <= segment_index < len(self.segment_final_analysis):
            return self.segment_final_analysis[segment_index]
        return None

    def slice_raw_points(
        self, start_index: int, stop_index: int
    ) -> dict[str, list[Any]]:
        """Return raw stored point arrays sliced to a sub-range of this site.

        The returned dictionary is keyed by storage keys such as ``param_0`` or
        ``channel_0``. Use :meth:`series` for normal semantic access by public path.
        """

        return {
            key: list(values[start_index:stop_index])
            for key, values in self.raw_points.items()
        }


@dataclass(frozen=True)
class HostRuntimeSnapshot:
    """Offline view of one host-runtime HDF5 snapshot file."""

    path: Path
    top_level_metadata: dict[str, Any]
    sites: dict[tuple[str, ...], HostRuntimeSite]

    def get_site(self, path: tuple[str, ...] = ()) -> HostRuntimeSite:
        return self.sites[path]

    def child_sites(self, parent_path: tuple[str, ...] = ()) -> list[HostRuntimeSite]:
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
        sites[path_value] = HostRuntimeSite(
            prefix=prefix,
            path=path_value,
            parent_path=None if parent_path is None else tuple(parent_path),
            fragment_fqn=datasets[prefix + "site.fragment_fqn"],
            pseudoparams=datasets.get(prefix + "scan.pseudoparams", {}),
            fixed_pseudoparams=datasets.get(prefix + "scan.fixed_pseudoparams", {}),
            parameters=datasets.get(prefix + "scan.parameters", {}),
            fixed_parameters=datasets.get(prefix + "scan.fixed_parameters", {}),
            channels=datasets.get(prefix + "scan.channels", {}),
            raw_points=_point_keys_for_prefix(datasets, prefix),
            analysis_outputs_schema=datasets.get(prefix + "analysis.outputs", {}),
            analysis_outputs=_analysis_outputs_for_prefix(datasets, prefix),
            analysis_artifacts=_analysis_artifacts_for_prefix(datasets, prefix),
            online_analysis_schema=datasets.get(prefix + "analysis.online", {}),
            online_analysis_results=_online_results_for_prefix(datasets, prefix),
            online_analysis_artifacts=_online_artifacts_for_prefix(datasets, prefix),
            online_analysis_annotations=_online_annotations_for_prefix(datasets, prefix),
            annotations=datasets.get(prefix + "analysis.annotations", []),
            segment_final_analysis=_segment_final_analysis_for_prefix(datasets, prefix),
            segmented=(prefix + "segments.start_index") in datasets,
            metadata={
                key[len(prefix) :]: value
                for key, value in datasets.items()
                if key.startswith(prefix)
                and not key.startswith(prefix + "points.")
                and not key.startswith(prefix + "analysis.output.")
                and not key.startswith(prefix + "analysis.artifact.")
                and not key.startswith(prefix + "analysis.online_result.")
                and not key.startswith(prefix + "analysis.online_artifact.")
                and not key.startswith(prefix + "analysis.online_annotation.")
                and not key.startswith(prefix + "segments.analysis.final_feedback")
            },
        )

    return HostRuntimeSnapshot(
        path=file_path,
        top_level_metadata=top_level_metadata,
        sites=sites,
    )

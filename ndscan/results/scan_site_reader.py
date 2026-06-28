"""Read prepared-runtime scan-site HDF5 snapshots without ARTIQ.

This module is intentionally small. It understands the current prepared-runtime scan-site
schema and returns plain Python dataclasses so offline tools can inspect preview or
final HDF5 snapshots without importing the experiment runtime itself.

The reader keeps the persisted data mostly intact. Convenience methods can derive
plotting choices, batch lengths, and segment ranges, but the dataclasses still expose
the raw point streams and metadata so analysis code is not locked into one plotting
view.
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
    "ScanSiteSnapshot",
    "ScanSiteBatch",
    "ScanSiteSegment",
    "ScanSiteSegmentAnalysis",
    "SeriesDescription",
    "PlotAxisChoices",
    "PlotChoices",
    "ScanSiteData",
    "read_scan_site_snapshot",
]


_STRUCTURED_KEYS = {
    "site.path",
    "site.parent_path",
    "scan.point_policy",
    "scan.axes",
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


def _strip_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _series_choice_label(path: str, description: str | None, unit: str | None) -> str:
    unit_suffix = f" / {unit}" if unit else ""
    if description is None or description == path:
        return f"{path}{unit_suffix}"
    return f"{path} ({description}{unit_suffix})"


def _is_numeric_array_channel_schema(schema: dict[str, Any]) -> bool:
    return schema.get("type") == "array" and schema.get("element_type") in {"float", "int"}


def _is_numeric_parameter_schema(schema: dict[str, Any]) -> bool:
    param_schema = schema.get("param", {})
    return param_schema.get("type", "float") in {"float", "int"}


def _is_numeric_pseudoparam_schema(schema: dict[str, Any]) -> bool:
    variable = schema.get("variable", {})
    return variable.get("type", "float") in {"float", "int"}


def _is_numeric_channel_schema(schema: dict[str, Any]) -> bool:
    return schema.get("type") in {"float", "int"} or _is_numeric_array_channel_schema(schema)


def _is_numeric_point_stream(values: Any) -> bool:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    if array.ndim != 1:
        return False
    try:
        np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def _point_stream_varies(values: Any) -> bool:
    if not _is_numeric_point_stream(values):
        return False
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return False
    return not np.allclose(array, array[0], equal_nan=True)


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
    """Convert one HDF5 dataset payload into normal Python containers."""

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
    """Return all site prefixes, with parents before descendants."""

    suffix = "site.path"
    prefixes = [
        key[: -len(suffix)]
        for key in dataset_values.keys()
        if key.endswith(suffix)
    ]
    return sorted(prefixes, key=lambda prefix: (prefix.count("."), prefix))


def _point_keys_for_prefix(dataset_values: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Extract the flat point streams for exactly one site."""

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


def _segment_analyses_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> list["ScanSiteSegmentAnalysis"]:
    raw_feedback = dataset_values.get(prefix + "segments.analysis.final_feedback", [])
    return [ScanSiteSegmentAnalysis.from_dict(item) for item in raw_feedback]


@dataclass(frozen=True)
class ScanSiteBatch:
    """One completed execution batch in flat point-index space."""

    index: int
    start_index: int
    stop_index: int
    start_unix_time: float | None

    @property
    def length(self) -> int:
        return self.stop_index - self.start_index


@dataclass(frozen=True)
class ScanSiteSegment:
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
class ScanSiteSegmentAnalysis:
    """Analysis payload attached to one finished site segment."""

    outputs: dict[str, Any]
    artifacts: dict[str, Any]
    annotations: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | str | bytes) -> "ScanSiteSegmentAnalysis":
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
class SeriesDescription:
    """User-facing description of one public per-point series."""

    path: str
    kind: str
    storage_key: str | None
    shape: tuple[int, ...]
    dtype: np.dtype
    label: str
    description: str | None
    unit: str | None
    scale: Any | None
    is_numeric: bool
    point_shape: tuple[int, ...] | None = None
    dim_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PlotAxisChoices:
    """Plotting choices for one axis."""

    choices: tuple[SeriesDescription, ...]
    default_path: str | None
    default: SeriesDescription | None


@dataclass(frozen=True)
class PlotChoices:
    """User-facing numeric plot choices plus viewer-like defaults."""

    x: PlotAxisChoices
    y: PlotAxisChoices
    z: PlotAxisChoices


@dataclass(frozen=True)
class ScanSiteData:
    """Offline view of one prepared-runtime scan site."""

    prefix: str
    path: tuple[str, ...]
    parent_path: tuple[str, ...] | None
    fragment_fqn: str
    axes: list[dict[str, Any]]
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
    segment_analyses: list[ScanSiteSegmentAnalysis]
    segmented: bool
    metadata: dict[str, Any]

    @property
    def num_points(self) -> int:
        return int(self.metadata.get("state.num_points", 0))

    def axis_paths(self) -> list[str]:
        """Return public series paths for the ordered logical scan axes."""

        return [
            str(axis.get("path") or axis.get("storage_key", ""))
            for axis in self.axes
        ]

    def axis_storage_keys(self) -> list[str]:
        """Return point-stream storage keys for the ordered logical scan axes."""

        return [
            str(axis["storage_key"])
            for axis in self.axes
            if "storage_key" in axis
        ]

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
                matches.append(("parameter", key))
        for key, schema in self.channels.items():
            if _channel_series_path(key, schema) == path:
                matches.append(("channel", key))
        return matches

    def _series_display_metadata(
        self,
        *,
        path: str,
        kind: str,
        schema: Mapping[str, Any] | None,
        values: Any,
    ) -> dict[str, Any]:
        description = None
        unit = None
        scale = None
        declared_numeric = False

        if kind == "pseudoparam" and isinstance(schema, Mapping):
            variable = schema.get("variable", {})
            if isinstance(variable, Mapping):
                description = _strip_optional_string(variable.get("description"))
                spec = variable.get("spec", {})
                if isinstance(spec, Mapping):
                    unit = _strip_optional_string(spec.get("unit"))
                    scale = spec.get("scale")
            declared_numeric = _is_numeric_pseudoparam_schema(dict(schema))
        elif kind == "parameter" and isinstance(schema, Mapping):
            param = schema.get("param", {})
            if isinstance(param, Mapping):
                description = _strip_optional_string(param.get("description"))
                spec = param.get("spec", {})
                if isinstance(spec, Mapping):
                    unit = _strip_optional_string(spec.get("unit"))
                    scale = spec.get("scale")
            declared_numeric = _is_numeric_parameter_schema(dict(schema))
        elif kind == "channel" and isinstance(schema, Mapping):
            description = _strip_optional_string(schema.get("description"))
            unit = _strip_optional_string(schema.get("unit"))
            scale = schema.get("scale")
            declared_numeric = _is_numeric_channel_schema(dict(schema))

        return {
            "description": description,
            "unit": unit,
            "scale": scale,
            "label": _series_choice_label(path, description, unit),
            "is_numeric": declared_numeric or _is_numeric_point_stream(values),
        }

    def _describe_series_entry(
        self,
        *,
        path: str,
        kind: str,
        storage_key: str | None,
        values: Any,
        schema: Mapping[str, Any] | None = None,
    ) -> SeriesDescription:
        array = np.asarray(values)
        display_metadata = self._series_display_metadata(
            path=path,
            kind=kind,
            schema=schema,
            values=values,
        )
        point_shape = None
        dim_names = None
        if isinstance(schema, Mapping):
            if "shape" in schema:
                point_shape = tuple(int(size) for size in schema["shape"])
            if "dim_names" in schema:
                dim_names = tuple(schema["dim_names"])
        return SeriesDescription(
            path=path,
            kind=kind,
            storage_key=storage_key,
            shape=tuple(int(size) for size in array.shape),
            dtype=np.dtype(array.dtype),
            label=display_metadata["label"],
            description=display_metadata["description"],
            unit=display_metadata["unit"],
            scale=display_metadata["scale"],
            is_numeric=bool(display_metadata["is_numeric"]),
            point_shape=point_shape,
            dim_names=dim_names,
        )

    def _describe_runtime_series_entry(
        self,
        *,
        path: str,
        storage_key: str | None,
        values: Any,
        kind: str,
    ) -> SeriesDescription:
        description = self._describe_series_entry(
            path=path,
            kind=kind,
            storage_key=storage_key,
            values=values,
        )
        if path == _POINT_INDEX_PATH:
            description = SeriesDescription(
                path=description.path,
                kind=description.kind,
                storage_key=description.storage_key,
                shape=description.shape,
                dtype=description.dtype,
                label=path,
                description="Point index",
                unit=description.unit,
                scale=description.scale,
                is_numeric=description.is_numeric,
                point_shape=description.point_shape,
                dim_names=description.dim_names,
            )
        return description

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

    def describe_series(self) -> list[SeriesDescription]:
        """Return a compact description of all plottable per-point series.

        Each entry includes the public path plus display-oriented metadata such as
        ``label``, ``description``, ``unit``, ``scale``, and ``is_numeric``.
        """

        descriptions = []
        semantic_keys = set()
        for path, kind, key, schema in self._semantic_series_entries():
            if key not in self.raw_points:
                continue
            semantic_keys.add(key)
            descriptions.append(
                self._describe_series_entry(
                    path=path,
                    kind=kind,
                    storage_key=key,
                    values=self.raw_points[key],
                    schema=schema,
                )
            )

        for key in sorted(self.raw_points):
            if key in semantic_keys:
                continue
            if key in {"acquired_at_unix", _POINT_INDEX_PATH}:
                kind = "runtime"
            elif key.startswith("metadata."):
                kind = "point_metadata"
            else:
                kind = "raw_points"
            descriptions.append(
                self._describe_runtime_series_entry(
                    path=key,
                    storage_key=key,
                    values=self.raw_points[key],
                    kind=kind,
                )
            )

        if _POINT_INDEX_PATH not in self.raw_points:
            descriptions.append(
                self._describe_runtime_series_entry(
                    path=_POINT_INDEX_PATH,
                    storage_key=None,
                    values=np.arange(self.num_points, dtype=np.int64),
                    kind="runtime",
                )
            )
        return descriptions

    def _numeric_plot_choice_descriptions(self) -> list[SeriesDescription]:
        choices: list[SeriesDescription] = []
        seen = set[str]()

        def add_choice(
            *,
            path: str,
            kind: str,
            storage_key: str | None,
            values: Any,
            schema: Mapping[str, Any] | None = None,
            declared_numeric: bool = False,
        ) -> None:
            if path in seen:
                return
            if values is None:
                if not declared_numeric:
                    return
            elif not _is_numeric_point_stream(values) and not declared_numeric:
                return
            if kind in {"pseudoparam", "parameter", "channel"}:
                description = self._describe_series_entry(
                    path=path,
                    kind=kind,
                    storage_key=storage_key,
                    values=values,
                    schema=schema,
                )
            else:
                description = self._describe_runtime_series_entry(
                    path=path,
                    storage_key=storage_key,
                    values=values,
                    kind=kind,
                )
            choices.append(description)
            seen.add(path)

        for key, schema in self.pseudoparams.items():
            add_choice(
                path=_pseudoparam_series_path(key, schema),
                kind="pseudoparam",
                storage_key=key,
                values=self.raw_points.get(key),
                schema=schema,
                declared_numeric=_is_numeric_pseudoparam_schema(schema),
            )

        for key, schema in self.parameters.items():
            if not schema.get("is_scanned", False):
                continue
            add_choice(
                path=_parameter_series_path(key, schema),
                kind="parameter",
                storage_key=key,
                values=self.raw_points.get(key),
                schema=schema,
                declared_numeric=_is_numeric_parameter_schema(schema),
            )

        for key, schema in self.parameters.items():
            if schema.get("is_scanned", False):
                continue
            add_choice(
                path=_parameter_series_path(key, schema),
                kind="parameter",
                storage_key=key,
                values=self.raw_points.get(key),
                schema=schema,
                declared_numeric=_is_numeric_parameter_schema(schema),
            )

        for key, schema in self.channels.items():
            add_choice(
                path=_channel_series_path(key, schema),
                kind="channel",
                storage_key=key,
                values=self.raw_points.get(key),
                schema=schema,
                declared_numeric=_is_numeric_channel_schema(schema),
            )

        for key in sorted(self.raw_points.keys()):
            if key in self.pseudoparams or key in self.parameters or key in self.channels:
                continue
            if key in {"acquired_at_unix", _POINT_INDEX_PATH}:
                kind = "runtime"
            elif key.startswith("metadata."):
                kind = "point_metadata"
            else:
                kind = "raw_points"
            add_choice(
                path=key,
                kind=kind,
                storage_key=key,
                values=self.raw_points[key],
            )

        add_choice(
            path=_POINT_INDEX_PATH,
            kind="runtime",
            storage_key=None,
            values=np.arange(self.num_points, dtype=np.int64),
            declared_numeric=True,
        )

        return choices

    def _default_plot_x_path(self, choices: list[SeriesDescription]) -> str | None:
        if not choices:
            return None
        choice_paths = [entry.path for entry in choices]
        non_point_index_paths = [path for path in choice_paths if path != _POINT_INDEX_PATH]
        if self.num_points == 0:
            return non_point_index_paths[0] if non_point_index_paths else choice_paths[0]

        preferred = []

        def add_paths(paths) -> None:
            for path in paths:
                if path in choice_paths and path not in preferred:
                    preferred.append(path)

        def varies(path: str) -> bool:
            return _point_stream_varies(self._series_for_path(path))

        add_paths(
            path
            for path in self.axis_paths()
            if path in choice_paths and varies(path)
        )
        add_paths(
            _pseudoparam_series_path(key, schema)
            for key, schema in self.pseudoparams.items()
            if varies(_pseudoparam_series_path(key, schema))
        )
        add_paths(
            _parameter_series_path(key, schema)
            for key, schema in self.parameters.items()
            if schema.get("is_scanned", False) and varies(_parameter_series_path(key, schema))
        )
        add_paths(
            _parameter_series_path(key, schema)
            for key, schema in self.parameters.items()
            if not schema.get("is_scanned", False)
            and varies(_parameter_series_path(key, schema))
        )
        if "acquired_at_unix" in choice_paths and varies("acquired_at_unix"):
            preferred.append("acquired_at_unix")
        if _POINT_INDEX_PATH in choice_paths:
            preferred.append(_POINT_INDEX_PATH)
        add_paths(
            _channel_series_path(key, schema)
            for key, schema in self.channels.items()
            if varies(_channel_series_path(key, schema))
        )
        add_paths(
            key
            for key in sorted(self.raw_points.keys())
            if key not in self.pseudoparams
            and key not in self.parameters
            and key not in self.channels
            and key != "acquired_at_unix"
            and varies(key)
        )
        add_paths(non_point_index_paths)
        return preferred[0] if preferred else choice_paths[0]

    def _default_plot_y_path(
        self, choices: list[SeriesDescription], *, x_path: str | None
    ) -> str | None:
        if not choices:
            return None
        choice_paths = [entry.path for entry in choices]
        excluded = {x_path} if x_path is not None else set()
        preferred = [
            _channel_series_path(key, schema)
            for key, schema in self.channels.items()
            if (
                key in self.raw_points and _is_numeric_point_stream(self.raw_points[key])
            )
            or _is_numeric_channel_schema(schema)
        ]
        for path in preferred:
            if path in choice_paths and path not in excluded:
                return path
        for path in choice_paths:
            if path not in excluded:
                return path
        return choice_paths[0]

    def _default_plot_z_path(
        self,
        choices: list[SeriesDescription],
        *,
        x_path: str | None,
        y_path: str | None,
    ) -> str | None:
        if not choices:
            return None
        choice_paths = [entry.path for entry in choices]
        excluded = {path for path in (x_path, y_path) if path is not None}
        preferred = [
            _channel_series_path(key, schema)
            for key, schema in self.channels.items()
            if (
                key in self.raw_points and _is_numeric_point_stream(self.raw_points[key])
            )
            or _is_numeric_channel_schema(schema)
        ]
        for path in preferred:
            if path in choice_paths and path not in excluded:
                return path
        for path in choice_paths:
            if path not in excluded:
                return path
        return choice_paths[0]

    def describe_plot_choices(self) -> PlotChoices:
        """Return user-facing numeric plot choices plus viewer-like defaults.

        Choices are addressed by the same public series paths accepted by
        :meth:`series`. The default x/y/z paths follow the current runtime-viewer
        preference order as closely as possible without depending on the GUI layer.
        """

        choices = self._numeric_plot_choice_descriptions()
        choices_by_path = {entry.path: entry for entry in choices}
        default_x_path = self._default_plot_x_path(choices)
        default_y_path = self._default_plot_y_path(choices, x_path=default_x_path)
        default_z_path = self._default_plot_z_path(
            choices,
            x_path=default_x_path,
            y_path=default_y_path,
        )

        all_choices = tuple(choices)
        return PlotChoices(
            x=PlotAxisChoices(
                choices=all_choices,
                default_path=default_x_path,
                default=choices_by_path.get(default_x_path),
            ),
            y=PlotAxisChoices(
                choices=all_choices,
                default_path=default_y_path,
                default=choices_by_path.get(default_y_path),
            ),
            z=PlotAxisChoices(
                choices=all_choices,
                default_path=default_z_path,
                default=choices_by_path.get(default_z_path),
            ),
        )

    def choose_default_x_path(self) -> str | None:
        """Return the default x-data path for plotting.

        This is a lightweight shortcut for
        ``site.describe_plot_choices().x.default_path``.
        """

        return self.describe_plot_choices().x.default_path

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

    def choose_default_x_source(self) -> tuple[str | None, str | None]:
        """Return the schema-level x-source heuristic for advanced callers.

        This helper only reports the preferred pseudoparameter/parameter schema entry.
        It does not account for richer plotting defaults such as
        ``acquired_at_unix`` or ``point_index`` when those are more useful for the
        visible data. Most callers should use :meth:`describe_plot_choices` or
        :meth:`choose_default_x_path` instead.

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
                return "parameter", key

        if self.parameters:
            return "parameter", next(iter(self.parameters))

        return None, None

    def batches(self) -> list[ScanSiteBatch]:
        """Return completed execution batches in flat point-index space."""

        starts = list(self.metadata.get("batches.start_index", []))
        start_unix_times = self.metadata.get("batches.start_unix_time", [])

        result = []
        for index, start_index in enumerate(starts):
            start_index = int(start_index)
            stop_index = (
                int(starts[index + 1])
                if index + 1 < len(starts)
                else self.num_points
            )
            start_unix_time = (
                float(start_unix_times[index])
                if index < len(start_unix_times)
                else None
            )
            result.append(
                ScanSiteBatch(
                    index=index,
                    start_index=start_index,
                    stop_index=stop_index,
                    start_unix_time=start_unix_time,
                )
            )
        return result

    def segments(self) -> list[ScanSiteSegment]:
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
                ScanSiteSegment(
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
    ) -> list[ScanSiteSegment]:
        """Return all child segments launched from one parent point index."""

        return [
            segment
            for segment in self.segments()
            if segment.parent_point_index == parent_point_index
        ]

    def analysis_for_segment(
        self, segment_index: int
    ) -> ScanSiteSegmentAnalysis | None:
        """Return the persisted analysis payload for one segment, if present."""

        if 0 <= segment_index < len(self.segment_analyses):
            return self.segment_analyses[segment_index]
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
class ScanSiteSnapshot:
    """Offline view of one prepared-runtime HDF5 snapshot file."""

    path: Path
    top_level_metadata: dict[str, Any]
    sites: dict[tuple[str, ...], ScanSiteData]

    def get_site(self, path: tuple[str, ...] = ()) -> ScanSiteData:
        return self.sites[path]

    def child_sites(self, parent_path: tuple[str, ...] = ()) -> list[ScanSiteData]:
        """Return the immediate child sites of the given parent site path."""

        return sorted(
            (
                site
                for site in self.sites.values()
                if site.parent_path == parent_path
            ),
            key=lambda site: site.path,
        )


def read_scan_site_snapshot(path: str | Path) -> ScanSiteSnapshot:
    """Read a prepared-runtime preview or final HDF5 snapshot."""

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
        sites[path_value] = ScanSiteData(
            prefix=prefix,
            path=path_value,
            parent_path=None if parent_path is None else tuple(parent_path),
            fragment_fqn=datasets[prefix + "site.fragment_fqn"],
            axes=list(datasets.get(prefix + "scan.axes", [])),
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
            segment_analyses=_segment_analyses_for_prefix(datasets, prefix),
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
                # Child sites physically live under ``subscans``. The parent's
                # metadata view should describe only the parent site, not every
                # descendant dataset nested below the same string prefix.
                and not key.startswith(prefix + "subscans.")
            },
        )

    return ScanSiteSnapshot(
        path=file_path,
        top_level_metadata=top_level_metadata,
        sites=sites,
    )

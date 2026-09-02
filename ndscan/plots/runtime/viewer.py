"""Prepared-runtime live plot viewer.

This module deliberately keeps the live plotting model close to the persisted runtime
schema:

- each column shows one scan site,
- clicking a point opens child sites to the right,
- x/y/z selectors choose among the site's numeric point streams,
- 2D modes are still driven by the real sampled points, even when an image is shown.

The code is organised in three layers:

- small schema/label helpers at the top,
- one column widget that renders one site,
- the top-level viewer that builds the recursive column layout.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg

from ..._qt import QtCore, QtGui, QtWidgets
from ...fits.oitg import FIT_OBJECTS
from ...fits.sensible import artifact_summary, curve_points_for_artifact
from ...results.scan_site_reader import (
    ScanSiteData,
    ScanSiteSegment,
    ScanSiteSnapshot,
)
from .. import colormaps
from .bo_mpl import BoCornerPlotWidget, site_supports_bo_corner_plot
from .fitting import FitBackend, FitRequest, FitResult, default_fit_backend
from .live import snapshot_from_live_values

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

_POINT_INDEX_KEY = "__point_index__"
_NO_GROUP_KEY = "__no_group__"
_ARRAY_SERIES_GROUP_KEY = "__array_series_group__"
_NO_FIT_MODEL_KEY = "__no_fit_model__"
_REPEAT_COMBINE_NONE = "none"
_REPEAT_COMBINE_STD = "mean_std"
_REPEAT_COMBINE_SEM = "mean_sem"
_FIT_TARGET_VISIBLE = "visible"
_FIT_TARGET_SELECTED_GROUP = "selected_group"
_PLOT_MODE_1D = "1d"
_PLOT_MODE_2D_SCATTER = "2d_scatter"
_PLOT_MODE_2D_IMAGE = "2d_image"
_PLOT_MODE_BO = "bo_dashboard"
_PLOT_MODE_CHOICES = (
    (_PLOT_MODE_1D, "1D"),
    (_PLOT_MODE_2D_SCATTER, "2D scatter"),
    (_PLOT_MODE_2D_IMAGE, "2D image"),
)
_REPEAT_COMBINE_CHOICES = (
    (_REPEAT_COMBINE_NONE, "none"),
    (_REPEAT_COMBINE_STD, "mean ± std"),
    (_REPEAT_COMBINE_SEM, "mean ± sem"),
)
_ARRAY_INDEX_ALL = "all"

_COLORBAR_PIXMAP_CACHE: QtGui.QPixmap | None = None
_DisplayPointKey = tuple[Any, ...]


@dataclass(frozen=True)
class _SitePlotData:
    """Raw point slice currently shown for one site column."""

    source_indices: list[int]
    raw_points: dict[str, list[Any]]
    mode_label: str


@dataclass(frozen=True)
class _ResampledImage:
    """Rendered image payload for 2D image mode."""

    image: np.ndarray
    rect: QtCore.QRectF
    mode_label: str


@dataclass(frozen=True)
class _ColumnSpec:
    """State needed to keep one viewer column in sync with the current site tree."""

    site: ScanSiteData
    child_site_options: list[ScanSiteData]
    selected_child_path: tuple[str, ...] | None
    parent_point_index: int | None


@dataclass(frozen=True)
class _AnnotationRenderSpec:
    """One annotation plus the output/artifact payload needed to render it."""

    source_kind: str
    annotation: dict[str, Any]
    outputs: dict[str, Any]
    artifacts: dict[str, Any]


@dataclass(frozen=True)
class _Aggregated1DSeries:
    """Displayed 1D series after collapsing repeated x-values.

    Repeated points are grouped by identical x values, and by group value as well when
    the plot is split into multiple 1D series. The aggregated points are what the
    viewer uses for the main foreground trace, while the underlying raw points remain
    visible in a faded background layer.
    """

    x: np.ndarray
    y: np.ndarray
    y_std: np.ndarray
    y_sem: np.ndarray
    source_indices: list[int]
    group_values: np.ndarray | None
    used_repeats: bool


@dataclass(frozen=True)
class _DisplayedPointSelection:
    """One visible plot marker selected by the user.

    ``source_index`` is the underlying logical scan point used for drilldown into child
    sites. ``display_key`` identifies the exact visible marker inside the current plot,
    which matters when one source point fans out into multiple plotted markers (for
    example array-channel ``all`` selections).
    """

    source_index: int
    display_key: _DisplayPointKey


@dataclass(frozen=True)
class _Rendered1DView:
    """1D data currently displayed by one column widget.

    ``display_*`` describes the foreground series that the viewer fits against and
    connects with lines. ``raw_*`` is the clickable background scatter used to preserve
    access to the original repeated points.
    """

    display_x: np.ndarray
    display_y: np.ndarray
    display_point_keys: list[_DisplayPointKey]
    display_source_indices: list[int]
    display_group_values: np.ndarray | None
    raw_x: np.ndarray
    raw_y: np.ndarray
    raw_point_keys: list[_DisplayPointKey]
    raw_source_indices: list[int]
    raw_group_values: np.ndarray | None
    repeats_aggregated: bool


@dataclass(frozen=True)
class _PreparedPlotArrays:
    """Numeric plot payload for one site after axis selection and masking."""

    plot_mode: str
    x_key: str
    y_key: str
    group_key: str | None
    z_key: str | None
    x: np.ndarray
    y: np.ndarray
    y_error: np.ndarray | None
    point_keys: list[_DisplayPointKey]
    source_indices: list[int]
    group_values: np.ndarray | None
    group_label_map: dict[float, str] | None
    group_label_name: str | None
    array_series_label_map: dict[float, str] | None
    array_series_label_name: str | None
    z: np.ndarray | None


@dataclass
class _SiteUiState:
    """Viewer-side UI state for one site path."""

    selected_point_index: int | None = None
    selected_display_point_key: _DisplayPointKey | None = None
    selected_child_path: tuple[str, ...] | None = None
    plot_mode: str | None = None
    x_key: str | None = None
    x_index_tokens: tuple[str, ...] | None = None
    y_key: str | None = None
    y_index_tokens: tuple[str, ...] | None = None
    z_key: str | None = None
    z_index_tokens: tuple[str, ...] | None = None
    group_key: str | None = None
    show_lines: bool = False
    repeat_combine_mode: str | None = None
    series_states: tuple[_SeriesUiState, ...] | None = None


@dataclass(frozen=True)
class _SeriesUiState:
    """One plotted dependent-series row in the 1D viewer."""

    y_key: str | None = None
    y_index_tokens: tuple[str, ...] | None = None
    group_key: str | None = None
    repeat_combine_mode: str | None = None


@dataclass(frozen=True)
class _VisibleFitSeries:
    """Displayed 1D series data used by the compact fit controls."""

    label: str
    x: np.ndarray
    y: np.ndarray
    group_values: np.ndarray | None


@dataclass(frozen=True)
class _ParsedArrayIndexSelection:
    """One parsed array-index selection for an ``ArrayChannel``."""

    fixed_indices: tuple[int, ...]
    ranged_dim: int | None
    ranged_indices: tuple[int, ...]
    series_labels: tuple[str, ...] | None


@dataclass(frozen=True)
class _AxisSeriesValues:
    """Scalar series extracted from one selected point stream."""

    series_values: list[np.ndarray]
    series_labels: list[str] | None


def _clear_error_bar_item(item: pg.ErrorBarItem) -> None:
    """Hide one error-bar item completely so it contributes no plot bounds."""

    item.setData(
        x=None,
        y=None,
        top=None,
        bottom=None,
    )


def _annotation_artifact_map_for_display(
    site: ScanSiteData, parent_point_index: int | None
) -> tuple[str | None, dict[str, Any]]:
    """Return the artifacts most relevant to the currently displayed site slice."""

    if site.segmented:
        if parent_point_index is None:
            target_segments = (
                []
                if (segment := _current_or_latest_segment(site)) is None
                else [segment]
            )
        else:
            target_segments = site.segments_for_parent_point(parent_point_index)
    else:
        target_segments = []

    segment_artifacts = dict[str, Any]()
    for segment in target_segments:
        feedback = site.analysis_for_segment(segment.index)
        if feedback is not None:
            segment_artifacts.update(feedback.artifacts)

    if bool(site.metadata.get("state.completed", False)):
        if segment_artifacts:
            return ("final", segment_artifacts)
        return ("final", site.analysis_artifacts)

    active_segment = _current_or_latest_segment(site) if site.segmented else None
    showing_active_segment = (
        not site.segmented
        or parent_point_index is None
        or (
            active_segment is not None
            and active_segment.parent_point_index == parent_point_index
        )
    )
    if showing_active_segment:
        online_artifacts = dict[str, Any]()
        for name in sorted(site.online_analysis_artifacts):
            online_artifacts.update(site.online_analysis_artifacts.get(name, {}))
        if online_artifacts:
            return ("online", online_artifacts)
        if site.segmented and active_segment is not None:
            return (None, {})

    if segment_artifacts:
        return ("final", segment_artifacts)
    return ("final", site.analysis_artifacts)


def _format_artifact_readout(
    source_kind: str | None,
    artifacts: dict[str, Any],
) -> str:
    """Return a compact multiline summary of the active analysis artifacts."""

    if not artifacts:
        return "analysis: none"

    prefix = "analysis"
    if source_kind == "online":
        prefix = "analysis (online)"
    elif source_kind == "final":
        prefix = "analysis (final)"

    lines = [prefix + ":"]
    for name in sorted(artifacts):
        artifact = artifacts[name]
        kind = artifact.get("kind")
        if kind == "model_fit":
            model_name = str(artifact.get("model_name", artifact.get("model_id", "fit")))
            lines.append(f"{name}: {model_name}")
            summary = artifact_summary(artifact)
            if summary:
                lines.append(f"  {summary}")
            stats = dict(artifact.get("stats", {}))
            success = stats.get("success", None)
            message = str(stats.get("message", "")).strip()
            if success is not None or message:
                status_parts = []
                if success is not None:
                    status_parts.append("ok" if bool(success) else "failed")
                if message:
                    status_parts.append(message)
                lines.append("  status: " + "; ".join(status_parts))
            continue

        lines.append(f"{name}: {kind or 'artifact'}")
    return "\n".join(lines)


def _schema_description_label(description: str | None, fallback: str) -> str:
    """Return a readable schema label, falling back to a machine name when needed."""
    description = (description or "").strip()
    return description if description else fallback


def _machine_and_human_label(machine_label: str, human_label: str) -> str:
    if machine_label == _POINT_INDEX_KEY:
        return "point_index"
    return (
        machine_label
        if human_label == machine_label
        else f"{machine_label} ({human_label})"
    )


def _append_unit_label(label: str, unit: str | None) -> str:
    unit = (unit or "").strip()
    return f"{label} / {unit}" if unit else label


def _unique_choice_labels(choices: list[tuple[str, str]]) -> list[tuple[str, str]]:
    counts = dict[str, int]()
    for _, label in choices:
        counts[label] = counts.get(label, 0) + 1

    disambiguated = []
    for key, label in choices:
        if counts[label] > 1:
            disambiguated.append((key, f"{label} ({key})"))
        else:
            disambiguated.append((key, label))
    return disambiguated


def _pseudoparam_choice_label(key: str, schema: dict[str, Any]) -> str:
    variable = schema.get("variable", schema)
    machine_label = variable.get("name", key)
    human_label = _schema_description_label(
        variable.get("description"), variable.get("name", key)
    )
    human_label = _append_unit_label(human_label, variable.get("spec", {}).get("unit"))
    return _machine_and_human_label(machine_label, human_label)


def _parameter_machine_label(key: str, schema: dict[str, Any]) -> str:
    param = schema.get("param", schema)
    fqn = param.get("fqn", "")
    local_name = fqn.rsplit(".", 1)[-1] if fqn else key
    path = schema.get("path", "").strip("/")
    return f"{path}/{local_name}" if path else local_name


def _parameter_choice_label(key: str, schema: dict[str, Any]) -> str:
    param = schema.get("param", schema)
    machine_label = _parameter_machine_label(key, schema)
    human_label = _schema_description_label(param.get("description"), key)
    human_label = _append_unit_label(human_label, param.get("spec", {}).get("unit"))
    return _machine_and_human_label(machine_label, human_label)


def _channel_choice_label(key: str, schema: dict[str, Any]) -> str:
    machine_label = schema.get("path", key)
    human_label = _schema_description_label(schema.get("description"), key)
    human_label = _append_unit_label(human_label, schema.get("unit"))
    return _machine_and_human_label(machine_label, human_label)


def _array_channel_shape(schema: dict[str, Any]) -> tuple[int, ...] | None:
    if schema.get("type") != "array":
        return None
    shape = schema.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) == 0:
        return None
    try:
        return tuple(int(size) for size in shape)
    except (TypeError, ValueError):
        return None


def _array_channel_dim_names(schema: dict[str, Any]) -> tuple[str, ...] | None:
    shape = _array_channel_shape(schema)
    if shape is None:
        return None
    raw = schema.get("dim_names")
    if isinstance(raw, (list, tuple)) and len(raw) == len(shape):
        return tuple(str(name) for name in raw)
    return tuple(f"dim{index}" for index in range(len(shape)))


def _is_numeric_array_channel_schema(schema: dict[str, Any]) -> bool:
    return schema.get("type") == "array" and schema.get("element_type") in {"float", "int"}


def _default_array_index_tokens(schema: dict[str, Any]) -> tuple[str, ...] | None:
    shape = _array_channel_shape(schema)
    if shape is None:
        return None
    return tuple("0" for _ in shape)


def _normalise_array_index_tokens(
    schema: dict[str, Any],
    tokens: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...] | None:
    default_tokens = _default_array_index_tokens(schema)
    if default_tokens is None:
        return None
    if tokens is None:
        return default_tokens
    cleaned = [str(token).strip() for token in tokens[: len(default_tokens)]]
    while len(cleaned) < len(default_tokens):
        cleaned.append(default_tokens[len(cleaned)])
    return tuple(token if token else default_tokens[index] for index, token in enumerate(cleaned))


def _format_array_index_suffix(
    schema: dict[str, Any],
    tokens: tuple[str, ...] | list[str] | None,
) -> str:
    normalised = _normalise_array_index_tokens(schema, tokens)
    dim_names = _array_channel_dim_names(schema)
    if normalised is None or dim_names is None:
        return ""
    return "[" + ", ".join(
        f"{dim_name}={token}" for dim_name, token in zip(dim_names, normalised, strict=True)
    ) + "]"


def _axis_label_with_indices(
    label: str,
    *,
    site: ScanSiteData,
    key: str,
    index_tokens: tuple[str, ...] | None,
) -> str:
    schema = site.channels.get(key)
    if not isinstance(schema, dict) or _array_channel_shape(schema) is None:
        return label
    return label + " " + _format_array_index_suffix(schema, index_tokens)


def _parse_array_index_selection(
    schema: dict[str, Any],
    tokens: tuple[str, ...] | list[str] | None,
) -> _ParsedArrayIndexSelection:
    shape = _array_channel_shape(schema)
    dim_names = _array_channel_dim_names(schema)
    normalised = _normalise_array_index_tokens(schema, tokens)
    if shape is None or dim_names is None or normalised is None:
        raise ValueError("Array-channel selection requires an array schema")

    fixed_indices = []
    ranged_dim = None
    ranged_indices = ()
    ranged_labels = None
    for dim_index, (size, dim_name, token) in enumerate(
        zip(shape, dim_names, normalised, strict=True)
    ):
        token = token.strip()
        if token in {"", "0"}:
            fixed_indices.append(0)
            continue
        if token.lower() in {_ARRAY_INDEX_ALL, ":"}:
            indices = tuple(range(size))
        elif ":" in token:
            start_text, stop_text = token.split(":", 1)
            try:
                start = 0 if start_text == "" else int(start_text)
                stop = size if stop_text == "" else int(stop_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {dim_name} index range {token!r}; use N, all, or start:stop"
                ) from exc
            if start < 0 or stop < 0 or start > size or stop > size or stop <= start:
                raise ValueError(
                    f"{dim_name} range {token!r} is outside 0:{size}"
                )
            indices = tuple(range(start, stop))
        else:
            try:
                index = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {dim_name} index {token!r}; use N, all, or start:stop"
                ) from exc
            if index < 0 or index >= size:
                raise ValueError(f"{dim_name} index {index} is outside 0:{size - 1}")
            fixed_indices.append(index)
            continue

        if ranged_dim is not None:
            raise ValueError("Only one array index can be ranged at a time")
        ranged_dim = dim_index
        ranged_indices = indices
        ranged_labels = tuple(f"{dim_name}={index}" for index in indices)
        fixed_indices.append(0)

    return _ParsedArrayIndexSelection(
        fixed_indices=tuple(fixed_indices),
        ranged_dim=ranged_dim,
        ranged_indices=ranged_indices,
        series_labels=ranged_labels,
    )


def _extract_axis_series_values(
    values: list[Any],
    *,
    schema: dict[str, Any] | None,
    index_tokens: tuple[str, ...] | None,
) -> _AxisSeriesValues:
    if schema is None or _array_channel_shape(schema) is None:
        return _AxisSeriesValues(
            series_values=[np.asarray(values, dtype=float)],
            series_labels=None,
        )

    shape = _array_channel_shape(schema)
    assert shape is not None
    parsed = _parse_array_index_selection(schema, index_tokens)
    if parsed.ranged_dim is None:
        series_values = [[]]
        for raw in values:
            array = np.asarray(raw, dtype=float)
            if tuple(array.shape) != shape:
                raise ValueError(
                    f"Expected raw point arrays with shape {shape}, got {tuple(array.shape)}"
                )
            series_values[0].append(float(array[parsed.fixed_indices]))
        return _AxisSeriesValues(
            series_values=[np.asarray(series_values[0], dtype=float)],
            series_labels=None,
        )

    series_values = [[] for _ in parsed.ranged_indices]
    for raw in values:
        array = np.asarray(raw, dtype=float)
        if tuple(array.shape) != shape:
            raise ValueError(
                f"Expected raw point arrays with shape {shape}, got {tuple(array.shape)}"
            )
        for series_index, ranged_index in enumerate(parsed.ranged_indices):
            index_tuple = list(parsed.fixed_indices)
            index_tuple[parsed.ranged_dim] = ranged_index
            series_values[series_index].append(float(array[tuple(index_tuple)]))
    return _AxisSeriesValues(
        series_values=[np.asarray(series, dtype=float) for series in series_values],
        series_labels=list(parsed.series_labels or ()),
    )


def _ranged_dim_name_for_selection(
    schema: dict[str, Any] | None,
    index_tokens: tuple[str, ...] | None,
) -> str | None:
    if schema is None or _array_channel_shape(schema) is None:
        return None
    parsed = _parse_array_index_selection(schema, index_tokens)
    if parsed.ranged_dim is None:
        return None
    dim_names = _array_channel_dim_names(schema)
    if dim_names is None:
        return None
    return dim_names[parsed.ranged_dim]


def _annotation_index_tuple(raw: Any) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return None
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError):
        return None


def _selected_fixed_array_indices(
    schema: dict[str, Any] | None,
    index_tokens: tuple[str, ...] | None,
) -> tuple[int, ...] | None:
    if schema is None or _array_channel_shape(schema) is None:
        return None
    parsed = _parse_array_index_selection(schema, index_tokens)
    if parsed.ranged_dim is not None:
        return None
    return parsed.fixed_indices


def _generic_point_stream_choice_label(key: str) -> str:
    if key == "acquired_at_unix":
        return _machine_and_human_label(key, "acquired at / s")
    if key.startswith("metadata."):
        return _machine_and_human_label(key, key.removeprefix("metadata."))
    return key


def _is_numeric_point_stream(values: Any) -> bool:
    if len(values) == 0:
        return False
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
    """Return whether a numeric 1D stream contains more than one distinct value."""

    if not _is_numeric_point_stream(values):
        return False
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return False
    return not np.allclose(array, array[0], equal_nan=True)


def _is_numeric_parameter_schema(schema: dict[str, Any]) -> bool:
    """Return whether a persisted parameter schema describes a numeric stream."""

    param_schema = schema.get("param", {})
    return param_schema.get("type", "float") in {"float", "int"}


def _is_numeric_pseudoparam_schema(schema: dict[str, Any]) -> bool:
    """Return whether a persisted pseudoparam schema describes a numeric stream."""

    variable = schema.get("variable", {})
    return variable.get("type", "float") in {"float", "int"}


def _is_numeric_channel_schema(schema: dict[str, Any]) -> bool:
    """Return whether a persisted channel schema describes a numeric stream."""

    return schema.get("type") in {"float", "int"} or _is_numeric_array_channel_schema(schema)


def _x_axis_choices(site: ScanSiteData) -> list[tuple[str, str]]:
    """Return all numeric point streams that make sense as plot axes.

    The ordering is intentional:

    - pseudoparams first,
    - scanned parameters next,
    - mapped/fixed parameters after that,
    - saved channels,
    - finally any other numeric metadata-like point streams.
    """
    choices = []
    seen = set[str]()

    def add_choice(
        key: str,
        label: str,
        *,
        declared_numeric: bool = False,
    ) -> None:
        if key in seen:
            return
        values = site.raw_points.get(key)
        if values is None:
            if not declared_numeric:
                return
        elif not _is_numeric_point_stream(values) and not declared_numeric:
            return
        choices.append((key, label))
        seen.add(key)

    for key, schema in site.pseudoparams.items():
        add_choice(
            key,
            _pseudoparam_choice_label(key, schema),
            declared_numeric=_is_numeric_pseudoparam_schema(schema),
        )

    for key, schema in site.parameters.items():
        if schema.get("is_scanned", False):
            add_choice(
                key,
                _parameter_choice_label(key, schema),
                declared_numeric=_is_numeric_parameter_schema(schema),
            )

    for key, schema in site.parameters.items():
        if not schema.get("is_scanned", False):
            add_choice(
                key,
                _parameter_choice_label(key, schema),
                declared_numeric=_is_numeric_parameter_schema(schema),
            )

    for key, schema in site.channels.items():
        add_choice(
            key,
            _channel_choice_label(key, schema),
            declared_numeric=_is_numeric_channel_schema(schema),
        )

    for key in sorted(site.raw_points.keys()):
        add_choice(key, _generic_point_stream_choice_label(key))

    choices.append((_POINT_INDEX_KEY, _machine_and_human_label(_POINT_INDEX_KEY, "point_index")))
    return _unique_choice_labels(choices)


def _default_x_choices(site: ScanSiteData) -> list[tuple[str, str]]:
    """Return the default x-axis choices for a site column."""
    return _x_axis_choices(site)


def _default_y_choices(site: ScanSiteData) -> list[tuple[str, str]]:
    """Return the default y-axis choices for a site column."""
    return _x_axis_choices(site)


def _default_z_choices(site: ScanSiteData) -> list[tuple[str, str]]:
    """Return the default z-axis choices for a site column."""
    return _x_axis_choices(site)


def _group_by_choices(
    site: ScanSiteData, selected_x_key: str | None
) -> list[tuple[str, str]]:
    """Return available 1D grouping axes.

    For now grouping is intentionally limited to directly scanned parameters. Those are
    the most natural low-cardinality dimensions for splitting a 1D slice into one series
    per scan setting.
    """
    choices = [(_NO_GROUP_KEY, "none")]
    for key, schema in site.parameters.items():
        if not schema.get("is_scanned", False):
            continue
        if key == selected_x_key:
            continue
        values = site.raw_points.get(key)
        if values is None or not _is_numeric_point_stream(values):
            continue
        choices.append((key, _parameter_choice_label(key, schema)))
    return _unique_choice_labels(choices)


def _array_group_choice(
    *,
    dim_name: str | None,
) -> tuple[str, str] | None:
    if not dim_name:
        return None
    return (_ARRAY_SERIES_GROUP_KEY, dim_name)


def _fit_model_choices(fit_backend: FitBackend) -> list[tuple[str, str]]:
    """Return fit-model choices for the viewer-side fit controls."""
    return [(_NO_FIT_MODEL_KEY, "none")] + [
        (model.model_id, model.label) for model in fit_backend.models()
    ]


def _fit_target_choices(group_key: str | None) -> list[tuple[str, str]]:
    """Return which subset of the visible 1D data should be fit."""
    choices = [(_FIT_TARGET_VISIBLE, "visible data")]
    if group_key is not None:
        choices.append((_FIT_TARGET_SELECTED_GROUP, "selected group"))
    return choices


def _repeat_combine_choices() -> list[tuple[str, str]]:
    """Return how repeated identical x-values should be combined in 1D."""

    return list(_REPEAT_COMBINE_CHOICES)


def _plot_mode_choices(site: ScanSiteData) -> list[tuple[str, str]]:
    """Return the plot modes available for one site."""

    choices = list(_PLOT_MODE_CHOICES)
    if site_supports_bo_corner_plot(site):
        choices.append((_PLOT_MODE_BO, "BO dashboard"))
    return choices


def _choice_label_map(choices: list[tuple[str, str]]) -> dict[str, str]:
    return {key: label for key, label in choices}


def _choice_keys(choices: list[tuple[str, str]]) -> list[str]:
    return [key for key, _ in choices]


def _default_x_preferred_keys(
    site: ScanSiteData, choices: list[tuple[str, str]]
) -> list[str]:
    """Return x-axis preference order, favouring streams that actually vary.

    For ordinary scans, the scanned parameter usually varies and should stay the
    default x-axis. For repeated-shot sites, inherited/mapped parameters are often
    constant across the visible slice, so ``acquired_at_unix`` or ``point_index`` is
    a more useful default than a flat line at one constant parameter value.
    """

    choice_keys = _choice_keys(choices)
    non_point_index_keys = [key for key in choice_keys if key != _POINT_INDEX_KEY]
    if getattr(site, "num_points", 0) == 0:
        return non_point_index_keys

    preferred = []

    def add_keys(keys) -> None:
        for key in keys:
            if key in choice_keys and key not in preferred:
                preferred.append(key)

    def varies(key: str) -> bool:
        return _point_stream_varies(site.raw_points.get(key))

    add_keys(key for key in site.pseudoparams if varies(key))
    add_keys(
        key
        for key, schema in site.parameters.items()
        if schema.get("is_scanned", False) and varies(key)
    )
    add_keys(
        key
        for key, schema in site.parameters.items()
        if not schema.get("is_scanned", False) and varies(key)
    )
    if "acquired_at_unix" in choice_keys and varies("acquired_at_unix"):
        preferred.append("acquired_at_unix")
    if _POINT_INDEX_KEY in choice_keys:
        preferred.append(_POINT_INDEX_KEY)
    add_keys(key for key in site.channels if varies(key))
    add_keys(
        key
        for key in sorted(site.raw_points.keys())
        if key not in site.pseudoparams
        and key not in site.parameters
        and key not in site.channels
        and key != "acquired_at_unix"
        and varies(key)
    )
    add_keys(non_point_index_keys)
    return preferred


def _select_default_choice(
    choices: list[tuple[str, str]],
    *,
    preferred_keys: list[str] | None = None,
    excluded_keys: set[str] | None = None,
) -> str | None:
    if not choices:
        return None
    choice_keys = _choice_keys(choices)
    excluded_keys = set() if excluded_keys is None else set(excluded_keys)
    if preferred_keys is not None:
        for key in preferred_keys:
            if key in choice_keys and key not in excluded_keys:
                return key
    for key in choice_keys:
        if key not in excluded_keys:
            return key
    return choice_keys[0]


def _channel_choice_keys(site: ScanSiteData) -> list[str]:
    return [
        key
        for key in site.channels.keys()
        if (
            key in site.raw_points
            and _is_numeric_point_stream(site.raw_points[key])
        )
        or _is_numeric_channel_schema(site.channels[key])
    ]


def _error_bar_channel_key(site: ScanSiteData, value_key: str | None) -> str | None:
    """Return the error-bar channel paired with one plotted y-channel, if any."""

    if value_key is None:
        return None
    value_schema = site.channels.get(value_key)
    if not isinstance(value_schema, dict):
        return None
    value_path = value_schema.get("path")
    if not value_path:
        return None

    for key, schema in site.channels.items():
        if not isinstance(schema, dict):
            continue
        if schema.get("display_hints", {}).get("error_bar_for") == value_path:
            return key
    return None


def _color_brushes(z: np.ndarray) -> list[Any]:
    """Map one scalar z-value per point to scatter brushes."""
    finite = np.isfinite(z)
    if not finite.any():
        return [pg.mkBrush("#1f77b4")] * len(z)

    z_min = float(np.min(z[finite]))
    z_max = float(np.max(z[finite]))
    if z_max > z_min:
        scaled = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    else:
        scaled = np.full_like(z, 0.5, dtype=float)
    scaled = np.where(finite, scaled, 0.0)
    colors = colormaps.plasma.map(scaled, mode="qcolor")
    return [pg.mkBrush(color) for color in colors]


def _format_colorbar_value(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.4g}"


def _alpha_color(color: QtGui.QColor | str, alpha: int) -> QtGui.QColor:
    """Return one QColor with the requested alpha applied."""

    rgba = QtGui.QColor(color)
    rgba.setAlpha(max(0, min(255, int(alpha))))
    return rgba


def _darker_color(color: QtGui.QColor | str, factor: int = 135, alpha: int | None = None) -> QtGui.QColor:
    """Return a slightly darker version of one color for outlines/error bars."""

    rgba = QtGui.QColor(color).darker(factor)
    if alpha is not None:
        rgba.setAlpha(max(0, min(255, int(alpha))))
    return rgba


def _series_error_bar_pen(color: QtGui.QColor | str) -> QtGui.QPen:
    """Return a readable error-bar pen that stays close to the series color."""

    return pg.mkPen(_darker_color(color, factor=145, alpha=220), width=1.3)


def _aggregate_repeated_1d_points(
    x: np.ndarray,
    y: np.ndarray,
    source_indices: list[int],
    group_values: np.ndarray | None,
) -> _Aggregated1DSeries:
    """Collapse repeated x-values into one mean/std point per x (and group).

    The grouping key is:

    - ``x`` when the plot is a single series
    - ``(group, x)`` when ``group by`` is active

    The first raw point in each bucket is kept as a representative source index so the
    rest of the viewer can keep using site-level point identities without inventing a
    second identifier space for aggregated points.
    """

    grouped = dict[tuple[float | None, float], list[int]]()
    for index, xi in enumerate(x):
        group_value = None if group_values is None else float(group_values[index])
        grouped.setdefault((group_value, float(xi)), []).append(index)

    aggregated_x = []
    aggregated_y = []
    aggregated_std = []
    aggregated_sem = []
    aggregated_sources = []
    aggregated_group_values = [] if group_values is not None else None

    for (group_value, x_value), indices in grouped.items():
        y_values = np.asarray([y[index] for index in indices], dtype=float)
        y_std = float(np.std(y_values))
        aggregated_x.append(x_value)
        aggregated_y.append(float(np.mean(y_values)))
        aggregated_std.append(y_std)
        aggregated_sem.append(y_std / np.sqrt(len(y_values)))
        aggregated_sources.append(int(source_indices[indices[0]]))
        if aggregated_group_values is not None:
            aggregated_group_values.append(float(group_value))

    return _Aggregated1DSeries(
        x=np.asarray(aggregated_x, dtype=float),
        y=np.asarray(aggregated_y, dtype=float),
        y_std=np.asarray(aggregated_std, dtype=float),
        y_sem=np.asarray(aggregated_sem, dtype=float),
        source_indices=aggregated_sources,
        group_values=(
            None
            if aggregated_group_values is None
            else np.asarray(aggregated_group_values, dtype=float)
        ),
        used_repeats=len(aggregated_x) < len(x),
    )


def _format_readout_value(value: float) -> str:
    """Format point and cursor readouts compactly for the small viewer labels."""
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.6g}"


def _colorbar_pixmap(width: int = 18, height: int = 180) -> QtGui.QPixmap:
    """Return a cached vertical plasma gradient used by the lightweight colorbar."""
    global _COLORBAR_PIXMAP_CACHE
    if _COLORBAR_PIXMAP_CACHE is not None:
        return _COLORBAR_PIXMAP_CACHE

    colors = colormaps.plasma.map(np.linspace(1.0, 0.0, height), mode="byte")
    rgba = np.repeat(colors[:, None, :], width, axis=1).copy(order="C")
    image = QtGui.QImage(
        rgba.data,
        width,
        height,
        width * 4,
        QtGui.QImage.Format.Format_RGBA8888,
    )
    _COLORBAR_PIXMAP_CACHE = QtGui.QPixmap.fromImage(image.copy())
    return _COLORBAR_PIXMAP_CACHE


def _centres_to_edges(values: np.ndarray) -> np.ndarray:
    """Convert sorted bin centres into image edges."""
    if len(values) == 1:
        step = 1.0
        return np.array([values[0] - 0.5 * step, values[0] + 0.5 * step], dtype=float)
    midpoints = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate(([first], midpoints, [last]))


def _grid_resampled_image(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> _ResampledImage | None:
    """Create an exact image if the sampled points already lie on a rectangular grid."""
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    if len(unique_x) < 2 or len(unique_y) < 2:
        return None
    if len(unique_x) * len(unique_y) > max(len(x) * 4, 16_384):
        return None

    x_lookup = {float(value): index for index, value in enumerate(unique_x)}
    y_lookup = {float(value): index for index, value in enumerate(unique_y)}
    sums = np.zeros((len(unique_y), len(unique_x)), dtype=float)
    counts = np.zeros((len(unique_y), len(unique_x)), dtype=int)
    for xi, yi, zi in zip(x, y, z, strict=True):
        x_index = x_lookup.get(float(xi))
        y_index = y_lookup.get(float(yi))
        if x_index is None or y_index is None:
            return None
        sums[y_index, x_index] += float(zi)
        counts[y_index, x_index] += 1

    if np.count_nonzero(counts) < max(4, int(0.5 * counts.size)):
        return None

    image = np.full_like(sums, np.nan, dtype=float)
    mask = counts > 0
    image[mask] = sums[mask] / counts[mask]
    if not mask.any():
        return None

    x_edges = _centres_to_edges(unique_x.astype(float))
    y_edges = _centres_to_edges(unique_y.astype(float))
    rect = QtCore.QRectF(
        float(x_edges[0]),
        float(y_edges[0]),
        float(x_edges[-1] - x_edges[0]),
        float(y_edges[-1] - y_edges[0]),
    )
    return _ResampledImage(image=image, rect=rect, mode_label="exact grid image")


def _nearest_resampled_image(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    max_bins: int = 64,
    row_chunk: int = 8,
) -> _ResampledImage | None:
    """Fallback 2D image using nearest-neighbour resampling on a modest regular grid."""
    if len(x) == 0:
        return None
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_min == x_max or y_min == y_max:
        return None

    x_bins = min(max_bins, max(24, int(np.sqrt(len(x)))))
    y_bins = min(max_bins, max(24, int(np.sqrt(len(y)))))
    x_grid = np.linspace(x_min, x_max, x_bins)
    y_grid = np.linspace(y_min, y_max, y_bins)
    image = np.empty((len(y_grid), len(x_grid)), dtype=float)

    for start in range(0, len(y_grid), row_chunk):
        stop = min(start + row_chunk, len(y_grid))
        xx, yy = np.meshgrid(x_grid, y_grid[start:stop], indexing="xy")
        qx = xx.reshape(-1, 1)
        qy = yy.reshape(-1, 1)
        d2 = (qx - x.reshape(1, -1)) ** 2 + (qy - y.reshape(1, -1)) ** 2
        nearest = np.argmin(d2, axis=1)
        image[start:stop, :] = z[nearest].reshape(stop - start, len(x_grid))

    rect = QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
    return _ResampledImage(image=image, rect=rect, mode_label="nearest image")


def _current_or_latest_segment(
    site: ScanSiteData,
) -> ScanSiteSegment | None:
    """Return the currently active child segment, or the last finished one."""
    segments = site.segments()
    if not segments:
        return None
    current_index = int(site.metadata.get("state.current_segment", -1))
    if 0 <= current_index < len(segments):
        return segments[current_index]
    return segments[-1]


def _annotation_value_from_spec(
    site: ScanSiteData,
    render_spec: _AnnotationRenderSpec,
    spec: dict[str, Any] | None,
) -> Any | None:
    """Resolve one annotation value spec against the current site snapshot.

    Runtime annotations are already persisted in the same stringly-typed form used by
    the older plot model:

    - ``fixed`` embeds the value directly,
    - ``analysis_result`` points at final scalar analysis outputs,
    - ``online_result`` points at the latest online-analysis outputs for one named
      analysis.

    The runtime viewer does not maintain a second reactive annotation data model, so it
    resolves those references directly each time the column is redrawn.
    """

    if not isinstance(spec, dict):
        return None

    kind = spec.get("kind")
    if kind == "fixed":
        return spec.get("value")
    if kind == "analysis_result":
        name = spec.get("name")
        if name in render_spec.outputs:
            return render_spec.outputs.get(name)
        return site.analysis_outputs.get(name)
    if kind == "online_result":
        analysis_name = spec.get("analysis_name")
        result_key = spec.get("result_key")
        if analysis_name is None or result_key is None:
            return None
        return site.online_analysis_results.get(analysis_name, {}).get(result_key)
    return None


def _annotation_curve_data(value: Any) -> np.ndarray | None:
    """Return one numeric 1D array for a curve annotation source."""

    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or len(array) == 0:
        return None
    return array


def _annotation_scalar_value(value: Any) -> float | None:
    """Return one scalar float for marker-style annotations."""

    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 0:
        return None
    scalar = float(array)
    if not np.isfinite(scalar):
        return None
    return scalar


def _annotation_pen(source_kind: str, *, width: float = 2.0) -> QtGui.QPen:
    """Return a consistent pen for final vs online annotations."""

    if source_kind == "online":
        return pg.mkPen(
            "#9467bd",
            width=width,
            style=QtCore.Qt.PenStyle.DashLine,
        )
    return pg.mkPen("#2ca02c", width=width)


def _annotation_specs_for_display(
    site: ScanSiteData, parent_point_index: int | None
) -> list[_AnnotationRenderSpec]:
    """Choose which persisted annotations are meaningful for the visible site slice.

    The current schema keeps only one latest final-annotation list per site and one
    latest online-annotation list per online analysis. For a live segmented child site,
    online annotations are the best match for the currently active segment; for older
    selections we can only fall back to the latest final annotations.
    """

    if site.segmented:
        if parent_point_index is None:
            target_segments = (
                []
                if (segment := _current_or_latest_segment(site)) is None
                else [segment]
            )
        else:
            target_segments = site.segments_for_parent_point(parent_point_index)
    else:
        target_segments = []

    segment_final_specs = []
    for segment in target_segments:
        feedback = site.analysis_for_segment(segment.index)
        if feedback is None:
            continue
        segment_final_specs.extend(
            [
                _AnnotationRenderSpec(
                    source_kind="final",
                    annotation=annotation,
                    outputs=feedback.outputs,
                    artifacts=feedback.artifacts,
                )
                for annotation in feedback.annotations
            ]
        )

    if bool(site.metadata.get("state.completed", False)):
        if segment_final_specs:
            return segment_final_specs
        return [
            _AnnotationRenderSpec(
                source_kind="final",
                annotation=annotation,
                outputs=site.analysis_outputs,
                artifacts=site.analysis_artifacts,
            )
            for annotation in site.annotations
        ]

    active_segment = _current_or_latest_segment(site) if site.segmented else None
    showing_active_segment = (
        not site.segmented
        or parent_point_index is None
        or (
            active_segment is not None
            and active_segment.parent_point_index == parent_point_index
        )
    )
    if showing_active_segment:
        online_annotations = []
        for name in sorted(site.online_analysis_annotations):
            online_annotations.extend(
                [
                    _AnnotationRenderSpec(
                        source_kind="online",
                        annotation=annotation,
                        outputs=site.online_analysis_results.get(name, {}),
                        artifacts=site.online_analysis_artifacts.get(name, {}),
                    )
                    for annotation in site.online_analysis_annotations[name]
                ]
            )
        if online_annotations:
            return online_annotations
        if site.segmented and active_segment is not None:
            # A newly started child segment should not inherit the previous segment's
            # final fit/annotation overlay while fresh points are arriving. If there is
            # no online annotation yet and no segment-final feedback for this exact
            # segment, render nothing until the current segment produces its own state.
            return []

    if segment_final_specs:
        return segment_final_specs

    return [
        _AnnotationRenderSpec(
            source_kind="final",
            annotation=annotation,
            outputs=site.analysis_outputs,
            artifacts=site.analysis_artifacts,
        )
        for annotation in site.annotations
    ]


def _concat_slices(
    site: ScanSiteData, segments: list[ScanSiteSegment], mode_label: str
) -> _SitePlotData:
    """Merge one or more segment slices into the raw points for a displayed column."""
    source_indices = []
    merged = dict[str, list[Any]]()
    for segment in segments:
        source_indices.extend(range(segment.start_index, segment.stop_index))
        sliced = site.slice_raw_points(segment.start_index, segment.stop_index)
        for key, values in sliced.items():
            merged.setdefault(key, []).extend(values)
    return _SitePlotData(source_indices=source_indices, raw_points=merged, mode_label=mode_label)


def _site_plot_data(
    site: ScanSiteData, parent_point_index: int | None
) -> _SitePlotData:
    """Select the site data relevant for the current column state.

    Root/unsegmented sites show all their points. Segmented child sites either show:

    - the current/latest segment when no parent point is selected, or
    - all segments associated with the chosen parent point.
    """
    if parent_point_index is None:
        if site.segmented:
            segment = _current_or_latest_segment(site)
            if segment is None:
                return _SitePlotData([], {}, "no child segment yet")
            return _concat_slices(
                site,
                [segment],
                f"showing segment {segment.index}",
            )
        return _SitePlotData(
            source_indices=list(range(site.num_points)),
            raw_points={key: list(values) for key, values in site.raw_points.items()},
            mode_label="showing all points",
        )

    if not site.segmented:
        return _SitePlotData([], {}, f"site is not segmented for parent point {parent_point_index}")

    segments = site.segments_for_parent_point(parent_point_index)
    if not segments:
        return _SitePlotData([], {}, f"no child segments for parent point {parent_point_index}")
    return _concat_slices(
        site,
        segments,
        f"showing parent point {parent_point_index}",
    )


def _descends_from(path: tuple[str, ...], parent: tuple[str, ...]) -> bool:
    if len(path) <= len(parent):
        return False
    return path[: len(parent)] == parent


def _run_id_text_for_site(site: ScanSiteData) -> str | None:
    """Return a short run identifier for one site, if available."""

    source_id = site.metadata.get("site.source_id")
    if not source_id:
        return None
    source_id = str(source_id)
    suffix = source_id.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return f"RID {suffix}"
    return source_id


def _wrap_overlay_text(
    text: str,
    *,
    max_chars: int = 72,
    max_lines: int = 2,
) -> str:
    """Return a compact overlay summary wrapped to a small number of lines."""

    compact = " ".join(str(text).split())
    wrapped = textwrap.wrap(
        compact,
        width=max_chars,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)

    kept = wrapped[: max_lines]
    last_line = kept[-1]
    if len(last_line) >= max_chars:
        last_line = last_line[: max_chars - 1].rstrip()
    kept[-1] = last_line.rstrip(" .,;:") + "…"
    return "\n".join(kept)


class _PopupAwareComboBox(QtWidgets.QComboBox):
    """QComboBox that remembers whether its popup is currently open."""

    def __init__(self):
        super().__init__()
        self._popup_visible = False

    def showPopup(self) -> None:
        self._popup_visible = True
        super().showPopup()

    def hidePopup(self) -> None:
        super().hidePopup()
        if self._popup_visible:
            self._popup_visible = False

    def is_popup_visible(self) -> bool:
        return self._popup_visible


class _ArrayIndexWidget(QtWidgets.QWidget):
    """Compact editable array-index selector shown next to array-valued axes."""

    selection_changed = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._schema: dict[str, Any] | None = None
        self._edits: list[QtWidgets.QLineEdit] = []
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.setLayout(layout)
        self.hide()

    def set_state(
        self,
        *,
        schema: dict[str, Any] | None,
        index_tokens: tuple[str, ...] | None,
    ) -> None:
        self._schema = schema if isinstance(schema, dict) else None
        self._rebuild_inputs(index_tokens)

    def current_tokens(self) -> tuple[str, ...] | None:
        if self._schema is None or _array_channel_shape(self._schema) is None:
            return None
        return tuple(edit.text().strip() or "0" for edit in self._edits)

    def has_array_schema(self) -> bool:
        return self._schema is not None and _array_channel_shape(self._schema) is not None

    def _rebuild_inputs(self, index_tokens: tuple[str, ...] | None) -> None:
        layout = self.layout()
        assert isinstance(layout, QtWidgets.QHBoxLayout)
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._edits.clear()

        if self._schema is None or _array_channel_shape(self._schema) is None:
            self.hide()
            return

        tokens = _normalise_array_index_tokens(self._schema, index_tokens)
        dim_names = _array_channel_dim_names(self._schema) or ()
        if tokens is None:
            self.hide()
            return

        for dim_name, token in zip(dim_names, tokens, strict=True):
            label = QtWidgets.QLabel(dim_name)
            label.setStyleSheet("color: #555; font-size: 9px;")
            layout.addWidget(label)

            edit = QtWidgets.QLineEdit(token)
            edit.setPlaceholderText("0")
            edit.setMaximumWidth(66)
            edit.setToolTip("Use N, all, or start:stop")
            edit.editingFinished.connect(self._emit_selection_changed)
            self._edits.append(edit)
            layout.addWidget(edit)

        layout.addStretch(1)
        self.show()

    def _emit_selection_changed(self) -> None:
        self.selection_changed.emit(self.current_tokens())


class _SeriesControlRow(QtWidgets.QWidget):
    """One compact 1D dependent-series control row."""

    state_changed = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.setLayout(layout)

        self._y_label = QtWidgets.QLabel("y")
        layout.addWidget(self._y_label)

        self._y_combo = _PopupAwareComboBox()
        self._y_combo.currentIndexChanged.connect(self._emit_state_changed)
        layout.addWidget(self._y_combo, 2)

        self._y_index_label = QtWidgets.QLabel("idx")
        layout.addWidget(self._y_index_label)
        self._y_index_widget = _ArrayIndexWidget()
        self._y_index_widget.selection_changed.connect(lambda *_: self._emit_state_changed())
        layout.addWidget(self._y_index_widget, 2)

        self._split_label = QtWidgets.QLabel("split")
        layout.addWidget(self._split_label)
        self._split_combo = _PopupAwareComboBox()
        self._split_combo.currentIndexChanged.connect(self._emit_state_changed)
        layout.addWidget(self._split_combo, 1)

        self._repeat_label = QtWidgets.QLabel("repeats")
        layout.addWidget(self._repeat_label)
        self._repeat_combo = _PopupAwareComboBox()
        self._repeat_combo.currentIndexChanged.connect(self._emit_state_changed)
        layout.addWidget(self._repeat_combo, 1)

        self._remove_button = QtWidgets.QPushButton("−")
        self._remove_button.setMaximumWidth(24)
        self._remove_button.clicked.connect(self.remove_requested.emit)
        layout.addWidget(self._remove_button)

    def set_state(
        self,
        *,
        site: ScanSiteData,
        x_key: str | None,
        state: _SeriesUiState,
        can_remove: bool,
    ) -> None:
        y_choices = _default_y_choices(site)
        channel_keys = _channel_choice_keys(site)
        current_y = self._sync_choice_combo(
            self._y_combo,
            y_choices,
            state.y_key,
            preferred_keys=channel_keys,
            excluded_keys={x_key} if x_key is not None else None,
        )
        self._sync_array_index_widget(
            self._y_index_widget,
            site.channels.get(current_y) if current_y is not None else None,
            state.y_index_tokens,
        )
        split_choices = _group_by_choices(site, x_key)
        array_choice = self._current_array_group_choice(
            site=site,
            y_key=current_y,
            y_index_tokens=self._y_index_widget.current_tokens(),
        )
        if array_choice is not None:
            split_choices = split_choices + [array_choice]
        self._sync_choice_combo(
            self._split_combo,
            split_choices,
            state.group_key if state.group_key is not None else _NO_GROUP_KEY,
            preferred_keys=[_NO_GROUP_KEY],
        )
        self._sync_choice_combo(
            self._repeat_combo,
            _repeat_combine_choices(),
            state.repeat_combine_mode if state.repeat_combine_mode is not None else _REPEAT_COMBINE_NONE,
            preferred_keys=[_REPEAT_COMBINE_NONE],
        )
        self._remove_button.setVisible(can_remove)
        self._y_index_label.setVisible(self._y_index_widget.has_array_schema())
        self._y_index_widget.setVisible(self._y_index_widget.has_array_schema())

    def current_state(self) -> _SeriesUiState:
        group_key = self._split_combo.currentData()
        return _SeriesUiState(
            y_key=self._y_combo.currentData(),
            y_index_tokens=self._y_index_widget.current_tokens(),
            group_key=None if group_key in {None, _NO_GROUP_KEY} else group_key,
            repeat_combine_mode=self._repeat_combo.currentData() or _REPEAT_COMBINE_NONE,
        )

    def display_label(self) -> str:
        return self._y_combo.currentText()

    def _emit_state_changed(self) -> None:
        self.state_changed.emit()

    @staticmethod
    def _sync_choice_combo(
        combo: _PopupAwareComboBox,
        choices: list[tuple[Any, str]],
        selected_key: Any,
        *,
        preferred_keys: list[Any] | None = None,
        excluded_keys: set[Any] | None = None,
    ) -> Any:
        if combo.is_popup_visible():
            return combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for key, label in choices:
            combo.addItem(label, key)
        index = combo.findData(selected_key) if selected_key is not None else -1
        if index == -1:
            default_key = _select_default_choice(
                choices,
                preferred_keys=preferred_keys,
                excluded_keys=excluded_keys,
            )
            index = combo.findData(default_key) if default_key is not None else -1
        if index != -1:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)
        return combo.currentData()

    @staticmethod
    def _sync_array_index_widget(
        widget: _ArrayIndexWidget,
        schema: dict[str, Any] | None,
        selected_tokens: tuple[str, ...] | None,
    ) -> None:
        widget.blockSignals(True)
        widget.set_state(schema=schema, index_tokens=selected_tokens)
        widget.blockSignals(False)

    @staticmethod
    def _current_array_group_choice(
        *,
        site: ScanSiteData,
        y_key: str | None,
        y_index_tokens: tuple[str, ...] | None,
    ) -> tuple[str, str] | None:
        if y_key is None:
            return None
        y_schema = site.channels.get(y_key)
        y_dim_name = _ranged_dim_name_for_selection(
            y_schema if isinstance(y_schema, dict) else None,
            y_index_tokens,
        )
        return _array_group_choice(dim_name=y_dim_name)

class _VerticalColorBarWidget(QtWidgets.QWidget):
    """Small fixed-layout colorbar used by the lightweight 2D plot modes."""

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.setLayout(layout)
        self.setMinimumWidth(90)

        self._title = QtWidgets.QLabel("color")
        self._title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        self._max_label = QtWidgets.QLabel("")
        self._max_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._max_label)

        self._gradient = QtWidgets.QLabel()
        self._gradient.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._gradient.setPixmap(_colorbar_pixmap())
        layout.addWidget(self._gradient, 1)

        self._min_label = QtWidgets.QLabel("")
        self._min_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._min_label)

        self.hide()

    def set_state(
        self,
        *,
        visible: bool,
        label: str = "",
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> None:
        self.setVisible(visible)
        if not visible:
            return
        self._title.setText(label)
        self._max_label.setText(_format_colorbar_value(float(z_max)))
        self._min_label.setText(_format_colorbar_value(float(z_min)))


class _MovableViewBoxTextItem(pg.TextItem):
    """Text overlay positioned in view-box fractions rather than data coordinates."""

    def __init__(
        self,
        *,
        text: str = "",
        relative_pos: tuple[float, float] = (0.02, 0.98),
        anchor: tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(
            text=text,
            anchor=anchor,
            color=(30, 30, 30),
            fill=(255, 255, 255, 185),
            border=pg.mkPen(100, 100, 100, 150),
        )
        self._relative_pos = QtCore.QPointF(*relative_pos)
        self._view_box: pg.ViewBox | None = None
        self._syncing_position = False
        self._display_text = text
        self._drag_offset = None
        self.setZValue(1000)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
            True,
        )
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable,
            True,
        )
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            True,
        )
        # ``TextItem`` renders via an internal QGraphicsTextItem child. Make the child
        # transparent to mouse events so dragging hits this movable parent item.
        self.textItem.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        font = QtGui.QFont(self.textItem.font())
        font.setPointSize(max(8, font.pointSize() - 2))
        self.textItem.setFont(font)

    def attach_to_view_box(self, view_box: pg.ViewBox) -> None:
        self._view_box = view_box
        view_box.addItem(self, ignoreBounds=True)
        view_box.sigRangeChanged.connect(self._update_position_from_view_range)
        self._update_position_from_view_range()

    def set_display_text(self, text: str | None) -> None:
        self._display_text = "" if text is None else str(text)
        self.setText("" if not self._display_text else f" {self._display_text} ")
        self.setVisible(bool(self._display_text))
        if self.isVisible():
            self._update_position_from_view_range()

    def display_text(self) -> str:
        return self._display_text

    def itemChange(self, change, value):
        if (
            change
            == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
            and not self._syncing_position
            and self._view_box is not None
        ):
            x_range, y_range = self._view_box.viewRange()
            x0, x1 = map(float, x_range)
            y0, y1 = map(float, y_range)
            x_span = x1 - x0
            y_span = y1 - y0
            if x_span != 0.0:
                rel_x = (float(value.x()) - x0) / x_span
            else:
                rel_x = 0.0
            if y_span != 0.0:
                rel_y = (float(value.y()) - y0) / y_span
            else:
                rel_y = 1.0
            self._relative_pos = QtCore.QPointF(
                min(1.0, max(0.0, rel_x)),
                min(1.0, max(0.0, rel_y)),
            )
        return super().itemChange(change, value)

    def mouseDragEvent(self, ev) -> None:
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()
        if self._view_box is None:
            return
        if ev.isStart():
            self._drag_offset = self.pos() - self.mapToView(ev.buttonDownPos())
        if self._drag_offset is None:
            return
        self.setPos(self._drag_offset + self.mapToView(ev.pos()))
        if ev.isFinish():
            self._drag_offset = None

    def _update_position_from_view_range(self, *args) -> None:
        del args
        if self._view_box is None or not self._display_text:
            return
        x_range, y_range = self._view_box.viewRange()
        x0, x1 = map(float, x_range)
        y0, y1 = map(float, y_range)
        x = x0 + self._relative_pos.x() * (x1 - x0)
        y = y0 + self._relative_pos.y() * (y1 - y0)
        self._syncing_position = True
        try:
            self.setPos(x, y)
        finally:
            self._syncing_position = False


class _SiteColumnWidget(QtWidgets.QWidget):
    """Render one scan site and expose user selections through Qt signals.

    The widget deliberately owns both the controls and the pyqtgraph items for one
    column. That keeps the live viewer logic simple: the outer viewer only decides which
    site each column should show, while the column widget handles labels, axis choices,
    and the actual drawing.
    """

    point_selected = QtCore.pyqtSignal(object, object)
    child_site_changed = QtCore.pyqtSignal(object, object)
    plot_mode_changed = QtCore.pyqtSignal(object, str)
    x_key_changed = QtCore.pyqtSignal(object, str)
    x_index_tokens_changed = QtCore.pyqtSignal(object, object)
    y_key_changed = QtCore.pyqtSignal(object, str)
    y_index_tokens_changed = QtCore.pyqtSignal(object, object)
    z_key_changed = QtCore.pyqtSignal(object, str)
    z_index_tokens_changed = QtCore.pyqtSignal(object, object)
    group_key_changed = QtCore.pyqtSignal(object, str)
    show_lines_changed = QtCore.pyqtSignal(object, bool)
    repeat_combine_mode_changed = QtCore.pyqtSignal(object, str)
    series_states_changed = QtCore.pyqtSignal(object, object)

    def __init__(
        self,
        *,
        site: ScanSiteData,
        child_site_options: list[ScanSiteData],
        selected_child_path: tuple[str, ...] | None,
        parent_point_index: int | None,
        selected_point_index: int | None,
        selected_plot_mode: str | None,
        selected_x_key: str | None,
        selected_y_key: str | None,
        selected_z_key: str | None,
        selected_group_key: str | None,
        show_lines: bool,
        selected_repeat_combine_mode: str | None = None,
        selected_x_index_tokens: tuple[str, ...] | None = None,
        selected_y_index_tokens: tuple[str, ...] | None = None,
        selected_z_index_tokens: tuple[str, ...] | None = None,
        selected_display_point_key: _DisplayPointKey | None = None,
        selected_series_states: tuple[_SeriesUiState, ...] | None = None,
        fit_backend: FitBackend | None = None,
    ):
        super().__init__()
        self._site = site
        self._fit_backend = default_fit_backend() if fit_backend is None else fit_backend
        self._selected_point_index = selected_point_index
        self._selected_display_point_key = selected_display_point_key
        self._parent_point_index = parent_point_index
        self._plot_data = _site_plot_data(site, parent_point_index)
        self._child_site_options = child_site_options
        self._rendered_point_values = dict[_DisplayPointKey, tuple[float, float, float | None]]()
        self._rendered_group_values = dict[_DisplayPointKey, float]()
        self._rendered_group_displays = dict[_DisplayPointKey, str]()
        self._rendered_source_indices = dict[_DisplayPointKey, int]()
        self._rendered_series_indices = dict[_DisplayPointKey, int]()
        self._rendered_series_labels = dict[int, str]()
        self._rendered_group_label_map = dict[float, str]()
        self._group_colors = dict[float, QtGui.QColor]()
        self._current_x_label = "x"
        self._current_y_label = "y"
        self._current_z_label: str | None = None
        self._current_group_label: str | None = None
        self._last_cursor_plot_pos: tuple[float, float] | None = None
        self._fit_active = False
        self._fit_result: FitResult | None = None
        self._fit_target_group_value: float | None = None
        self._current_plot_mode = _PLOT_MODE_1D
        self._current_plot_arrays: tuple[np.ndarray, np.ndarray] | None = None
        self._current_display_point_keys: list[_DisplayPointKey] = []
        self._current_group_values: np.ndarray | None = None
        self._current_visible_fit_series: list[_VisibleFitSeries] = []

        layout = QtWidgets.QVBoxLayout()
        self._configure_widget_layout(layout)
        self._build_header(layout)
        self._build_details_panel(layout)
        self._build_plot_stack(layout)

        self.update_state(
            site=site,
            child_site_options=child_site_options,
            selected_child_path=selected_child_path,
            parent_point_index=parent_point_index,
            selected_point_index=selected_point_index,
            selected_display_point_key=selected_display_point_key,
            selected_plot_mode=selected_plot_mode,
            selected_x_key=selected_x_key,
            selected_x_index_tokens=selected_x_index_tokens,
            selected_y_key=selected_y_key,
            selected_y_index_tokens=selected_y_index_tokens,
            selected_z_key=selected_z_key,
            selected_z_index_tokens=selected_z_index_tokens,
            selected_group_key=selected_group_key,
            show_lines=show_lines,
            selected_repeat_combine_mode=selected_repeat_combine_mode,
            selected_series_states=selected_series_states,
        )

    def _configure_widget_layout(self, layout: QtWidgets.QVBoxLayout) -> None:
        layout.setContentsMargins(6, 6, 6, 6)
        self.setLayout(layout)
        self.setMinimumWidth(380)
        self.setStyleSheet(
            "QGroupBox { font-size: 10px; } "
            "QLabel { font-size: 10px; } "
            "QComboBox { font-size: 10px; } "
            "QPushButton { font-size: 10px; } "
            "QCheckBox { font-size: 10px; }"
        )

    def _build_header(self, layout: QtWidgets.QVBoxLayout) -> None:
        self._child_combo = _PopupAwareComboBox()
        self._child_combo.currentIndexChanged.connect(self._emit_child_site_changed)
        layout.addWidget(self._child_combo)

        self._title = QtWidgets.QLabel("/".join(self._site.path) or "root")
        self._title.setWordWrap(True)
        title_font = QtGui.QFont(self._title.font())
        title_font.setPointSize(max(9, title_font.pointSize() - 1))
        title_font.setBold(True)
        self._title.setFont(title_font)
        layout.addWidget(self._title)

        self._build_plot_controls_box()
        self._build_fit_controls_box()
        layout.addWidget(self._plot_controls_box)
        layout.addWidget(self._fit_controls_box)

    def _build_plot_controls_box(self) -> None:
        self._plot_controls_box = QtWidgets.QGroupBox("Plot")
        plot_controls_layout = QtWidgets.QVBoxLayout()
        plot_controls_layout.setContentsMargins(8, 8, 8, 8)
        plot_controls_layout.setSpacing(6)
        self._plot_controls_box.setLayout(plot_controls_layout)

        global_row = QtWidgets.QHBoxLayout()
        global_row.setContentsMargins(0, 0, 0, 0)
        global_row.setSpacing(6)

        self._mode_label = QtWidgets.QLabel("mode")
        self._plot_mode_combo = _PopupAwareComboBox()
        self._plot_mode_combo.currentIndexChanged.connect(
            lambda *_: self.plot_mode_changed.emit(
                self._site.path, self._current_combo_data(self._plot_mode_combo)
            )
        )
        global_row.addWidget(self._mode_label)
        global_row.addWidget(self._plot_mode_combo, 1)

        self._x_label = QtWidgets.QLabel("x")
        self._x_combo = _PopupAwareComboBox()
        self._x_combo.currentIndexChanged.connect(
            lambda *_: self.x_key_changed.emit(
                self._site.path, self._current_combo_data(self._x_combo)
            )
        )
        global_row.addWidget(self._x_label)
        global_row.addWidget(self._x_combo, 1)

        self._x_index_label = QtWidgets.QLabel("x idx")
        self._x_index_widget = _ArrayIndexWidget()
        self._x_index_widget.selection_changed.connect(
            lambda tokens: self.x_index_tokens_changed.emit(self._site.path, tokens)
        )
        global_row.addWidget(self._x_index_label)
        global_row.addWidget(self._x_index_widget, 1)

        self._plane_y_label = QtWidgets.QLabel("y")
        self._plane_y_combo = _PopupAwareComboBox()
        self._plane_y_combo.currentIndexChanged.connect(
            lambda *_: self.y_key_changed.emit(
                self._site.path, self._current_combo_data(self._plane_y_combo)
            )
        )
        global_row.addWidget(self._plane_y_label)
        global_row.addWidget(self._plane_y_combo, 1)

        self._plane_y_index_label = QtWidgets.QLabel("y idx")
        self._plane_y_index_widget = _ArrayIndexWidget()
        self._plane_y_index_widget.selection_changed.connect(
            lambda tokens: self.y_index_tokens_changed.emit(self._site.path, tokens)
        )
        global_row.addWidget(self._plane_y_index_label)
        global_row.addWidget(self._plane_y_index_widget, 1)
        global_row.addStretch(1)
        plot_controls_layout.addLayout(global_row)

        self._series_rows_widget = QtWidgets.QWidget()
        self._series_rows_layout = QtWidgets.QVBoxLayout()
        self._series_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._series_rows_layout.setSpacing(4)
        self._series_rows_widget.setLayout(self._series_rows_layout)
        plot_controls_layout.addWidget(self._series_rows_widget)

        series_buttons_row = QtWidgets.QHBoxLayout()
        series_buttons_row.setContentsMargins(0, 0, 0, 0)
        series_buttons_row.setSpacing(6)
        self._add_series_button = QtWidgets.QPushButton("+")
        self._add_series_button.setMaximumWidth(26)
        self._add_series_button.clicked.connect(self._add_series_row)
        series_buttons_row.addWidget(self._add_series_button)
        self._show_lines_checkbox = QtWidgets.QCheckBox("connect points")
        self._show_lines_checkbox.toggled.connect(
            lambda checked: self.show_lines_changed.emit(self._site.path, checked)
        )
        series_buttons_row.addWidget(self._show_lines_checkbox)
        series_buttons_row.addStretch(1)
        plot_controls_layout.addLayout(series_buttons_row)

        self._z_row_widget = QtWidgets.QWidget()
        z_row = QtWidgets.QHBoxLayout()
        z_row.setContentsMargins(0, 0, 0, 0)
        z_row.setSpacing(6)
        self._z_row_widget.setLayout(z_row)
        self._z_label = QtWidgets.QLabel("z")
        self._z_combo = _PopupAwareComboBox()
        self._z_combo.currentIndexChanged.connect(
            lambda *_: self.z_key_changed.emit(
                self._site.path, self._current_combo_data(self._z_combo)
            )
        )
        z_row.addWidget(self._z_label)
        z_row.addWidget(self._z_combo, 1)

        self._z_index_label = QtWidgets.QLabel("z idx")
        self._z_index_widget = _ArrayIndexWidget()
        self._z_index_widget.selection_changed.connect(
            lambda tokens: self.z_index_tokens_changed.emit(self._site.path, tokens)
        )
        z_row.addWidget(self._z_index_label)
        z_row.addWidget(self._z_index_widget, 1)
        z_row.addStretch(1)
        plot_controls_layout.addWidget(self._z_row_widget)

    def _build_fit_controls_box(self) -> None:
        self._fit_controls_box = QtWidgets.QGroupBox("Fit")
        fit_controls = QtWidgets.QHBoxLayout()
        fit_controls.setContentsMargins(8, 8, 8, 8)
        fit_controls.setSpacing(8)
        self._fit_controls_box.setLayout(fit_controls)

        self._fit_series_label = QtWidgets.QLabel("series")
        self._fit_series_combo = _PopupAwareComboBox()
        self._fit_series_combo.currentIndexChanged.connect(
            lambda *_: (self._sync_fit_target_combo(), self._sync_fit_controls())
        )
        fit_controls.addWidget(self._fit_series_label)
        fit_controls.addWidget(self._fit_series_combo, 1)

        self._fit_model_label = QtWidgets.QLabel("fit")
        self._fit_model_combo = _PopupAwareComboBox()
        self._fit_model_combo.currentIndexChanged.connect(self._sync_fit_controls)
        fit_controls.addWidget(self._fit_model_label)
        fit_controls.addWidget(self._fit_model_combo, 1)

        self._fit_target_label = QtWidgets.QLabel("target")
        self._fit_target_combo = _PopupAwareComboBox()
        self._fit_target_combo.currentIndexChanged.connect(self._sync_fit_controls)
        fit_controls.addWidget(self._fit_target_label)
        fit_controls.addWidget(self._fit_target_combo, 1)

        self._fit_button = QtWidgets.QPushButton("fit")
        self._fit_button.clicked.connect(self._fit_now)
        fit_controls.addWidget(self._fit_button)
        self._clear_fit_button = QtWidgets.QPushButton("clear")
        self._clear_fit_button.clicked.connect(self._clear_fit)
        fit_controls.addWidget(self._clear_fit_button)
        fit_controls.addStretch(1)

    def _build_details_panel(self, layout: QtWidgets.QVBoxLayout) -> None:
        self._details_box = QtWidgets.QGroupBox("Details")
        details_layout = QtWidgets.QGridLayout()
        details_layout.setContentsMargins(8, 8, 8, 8)
        details_layout.setHorizontalSpacing(10)
        details_layout.setVerticalSpacing(6)
        details_layout.setColumnStretch(0, 1)
        details_layout.setColumnStretch(1, 1)
        self._details_box.setLayout(details_layout)

        self._status = QtWidgets.QLabel(self._plot_data.mode_label)
        self._status.setWordWrap(True)
        details_layout.addWidget(self._status, 0, 0, 1, 2)

        self._selected_readout = QtWidgets.QLabel("selected: none")
        self._selected_readout.setWordWrap(True)
        details_layout.addWidget(self._selected_readout, 1, 0)

        self._cursor_readout = QtWidgets.QLabel("cursor: move over plot")
        self._cursor_readout.setWordWrap(True)
        details_layout.addWidget(self._cursor_readout, 1, 1)

        self._fit_readout = QtWidgets.QLabel("fit: none")
        self._fit_readout.setWordWrap(True)
        details_layout.addWidget(self._fit_readout, 2, 0, 1, 2)

        self._artifact_readout = QtWidgets.QLabel("analysis: none")
        self._artifact_readout.setWordWrap(True)
        artifact_font = QtGui.QFont(self._artifact_readout.font())
        artifact_font.setFamily("Monospace")
        artifact_font.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)
        artifact_font.setPointSize(max(9, artifact_font.pointSize() - 1))
        self._artifact_readout.setFont(artifact_font)
        details_layout.addWidget(self._artifact_readout, 3, 0, 1, 2)

        layout.addWidget(self._details_box)

    def _build_plot_stack(self, layout: QtWidgets.QVBoxLayout) -> None:
        self._plot_stack = QtWidgets.QStackedWidget()
        self._build_pyqtgraph_panel()
        self._bo_plot_widget = BoCornerPlotWidget()
        self._plot_stack.addWidget(self._pyqtgraph_panel)
        self._plot_stack.addWidget(self._bo_plot_widget)
        layout.addWidget(self._plot_stack, 1)

    def _build_pyqtgraph_panel(self) -> None:
        self._pyqtgraph_panel = QtWidgets.QWidget()
        plot_row = QtWidgets.QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.setSpacing(6)
        self._pyqtgraph_panel.setLayout(plot_row)

        self._plot_widget = pg.PlotWidget()
        self._plot_item = self._plot_widget.getPlotItem()
        self._plot_item.showGrid(x=True, y=True, alpha=0.25)
        self._legend = self._plot_item.addLegend(offset=(8, 8))
        self._legend.hide()
        self._image_item = pg.ImageItem(axisOrder="row-major")
        self._image_item.hide()
        self._plot_item.addItem(self._image_item)
        self._line_item = self._plot_item.plot(pen=pg.mkPen("b", width=1.5))
        self._extra_line_items: list[pg.PlotDataItem] = []
        self._error_bar_item = pg.ErrorBarItem(
            beam=0.0,
            pen=_series_error_bar_pen("#1f77b4"),
        )
        self._plot_item.addItem(self._error_bar_item)
        self._extra_error_bar_items: list[pg.ErrorBarItem] = []
        self._scatter = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush("#1f77b4"))
        self._summary_scatter = pg.ScatterPlotItem(size=10, pen=None, brush=pg.mkBrush("#1f77b4"))
        self._highlight = pg.ScatterPlotItem(
            size=16,
            pen=pg.mkPen("#ffd84d", width=2.5),
            brush=pg.mkBrush(255, 255, 255, 120),
        )
        self._crosshair_x = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(
                "#ffd84d",
                width=1.2,
                style=QtCore.Qt.PenStyle.DashLine,
            ),
        )
        self._crosshair_y = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(
                "#ffd84d",
                width=1.2,
                style=QtCore.Qt.PenStyle.DashLine,
            ),
        )
        self._crosshair_x.hide()
        self._crosshair_y.hide()
        self._plot_item.addItem(self._scatter)
        self._plot_item.addItem(self._summary_scatter)
        self._plot_item.addItem(self._highlight)
        self._plot_item.addItem(self._crosshair_x, ignoreBounds=True)
        self._plot_item.addItem(self._crosshair_y, ignoreBounds=True)
        self._fit_curve_item = self._plot_item.plot(
            pen=pg.mkPen(
                "#ff7f0e",
                width=2.2,
                style=QtCore.Qt.PenStyle.DashLine,
            )
        )
        self._fit_curve_item.hide()
        self._annotation_items: list[Any] = []
        self._scatter.sigClicked.connect(self._point_clicked)
        self._summary_scatter.sigClicked.connect(self._point_clicked)
        self._plot_item.scene().sigMouseClicked.connect(self._background_clicked)
        self._plot_item.scene().sigMouseMoved.connect(self._mouse_moved)
        plot_row.addWidget(self._plot_widget, 1)

        self._colorbar = _VerticalColorBarWidget()
        plot_row.addWidget(self._colorbar)

        self._run_id_overlay = _MovableViewBoxTextItem()
        self._run_id_overlay.attach_to_view_box(self._plot_item.vb)

    @property
    def site_path(self) -> tuple[str, ...]:
        return self._site.path

    @property
    def _primary_series_row(self) -> _SeriesControlRow | None:
        if not hasattr(self, "_series_rows") or not self._series_rows:
            return None
        return self._series_rows[0]

    @property
    def _y_combo(self) -> _PopupAwareComboBox | None:
        row = self._primary_series_row
        return None if row is None else row._y_combo

    @property
    def _y_index_widget(self) -> _ArrayIndexWidget | None:
        row = self._primary_series_row
        return None if row is None else row._y_index_widget

    @property
    def _group_combo(self) -> _PopupAwareComboBox | None:
        row = self._primary_series_row
        return None if row is None else row._split_combo

    @property
    def _repeat_combine_combo(self) -> _PopupAwareComboBox | None:
        row = self._primary_series_row
        return None if row is None else row._repeat_combo

    def update_state(
        self,
        *,
        site: ScanSiteData,
        child_site_options: list[ScanSiteData],
        selected_child_path: tuple[str, ...] | None,
        parent_point_index: int | None,
        selected_point_index: int | None,
        selected_plot_mode: str | None,
        selected_x_key: str | None,
        selected_y_key: str | None,
        selected_z_key: str | None,
        selected_group_key: str | None,
        show_lines: bool,
        selected_repeat_combine_mode: str | None = None,
        selected_x_index_tokens: tuple[str, ...] | None = None,
        selected_y_index_tokens: tuple[str, ...] | None = None,
        selected_z_index_tokens: tuple[str, ...] | None = None,
        selected_display_point_key: _DisplayPointKey | None = None,
        selected_series_states: tuple[_SeriesUiState, ...] | None = None,
    ) -> None:
        """Refresh the widget to match the current site-tree and UI selection state."""
        self._site = site
        self._selected_point_index = selected_point_index
        self._selected_display_point_key = selected_display_point_key
        self._parent_point_index = parent_point_index
        self._plot_data = _site_plot_data(site, parent_point_index)
        self._child_site_options = child_site_options

        self._title.setText("/".join(site.path) or "root")
        self._run_id_overlay.set_display_text(_run_id_text_for_site(site))
        self._sync_child_combo(selected_child_path)
        self._sync_plot_mode_combo(selected_plot_mode)
        self._sync_xyz_combos(
            selected_x_key,
            selected_x_index_tokens,
            selected_y_key,
            selected_y_index_tokens,
            selected_z_key,
            selected_z_index_tokens,
        )
        self._sync_series_rows(
            self._ensure_default_series_states(
                selected_series_states,
                selected_y_key=selected_y_key,
                selected_y_index_tokens=selected_y_index_tokens,
                selected_group_key=selected_group_key,
                selected_repeat_combine_mode=selected_repeat_combine_mode,
            )
        )
        self._sync_fit_series_combo()
        self._sync_fit_model_combo()
        self._sync_fit_target_combo()
        self._show_lines_checkbox.blockSignals(True)
        self._show_lines_checkbox.setChecked(show_lines)
        self._show_lines_checkbox.blockSignals(False)
        self._sync_control_visibility()
        self._sync_fit_controls()
        self._render_plot()

    def _sync_plot_mode_combo(self, selected_plot_mode: str | None) -> None:
        if self._plot_mode_combo.is_popup_visible():
            return
        self._plot_mode_combo.blockSignals(True)
        self._plot_mode_combo.clear()
        for key, label in _plot_mode_choices(self._site):
            self._plot_mode_combo.addItem(label, key)
        mode_index = (
            self._plot_mode_combo.findData(selected_plot_mode)
            if selected_plot_mode is not None
            else -1
        )
        if mode_index == -1:
            default_mode = (
                _PLOT_MODE_BO
                if site_supports_bo_corner_plot(self._site)
                else _PLOT_MODE_1D
            )
            mode_index = self._plot_mode_combo.findData(default_mode)
        if mode_index == -1:
            mode_index = 0
        self._plot_mode_combo.setCurrentIndex(mode_index)
        self._plot_mode_combo.blockSignals(False)

    def _sync_child_combo(self, selected_child_path: tuple[str, ...] | None) -> None:
        if self._child_combo.is_popup_visible():
            return
        was_visible = self._child_combo.isVisible()
        self._child_combo.blockSignals(True)
        self._child_combo.clear()
        for option in self._child_site_options:
            label = "/".join(option.path) or "root"
            self._child_combo.addItem(label, option.path)
        if self._child_site_options:
            index = (
                self._child_combo.findData(selected_child_path)
                if selected_child_path is not None
                else -1
            )
            if index == -1:
                index = 0
            self._child_combo.setCurrentIndex(index)
            self._child_combo.show()
        else:
            self._child_combo.hide()
        self._child_combo.blockSignals(False)
        self._child_combo.updateGeometry()
        layout = self.layout()
        if layout is not None and (was_visible != bool(self._child_site_options)):
            layout.invalidate()
            layout.activate()
        self.updateGeometry()

    def _sync_choice_combo(
        self,
        combo: _PopupAwareComboBox,
        choices: list[tuple[Any, str]],
        selected_key: Any,
        *,
        preferred_keys: list[Any] | None = None,
        excluded_keys: set[Any] | None = None,
    ) -> Any:
        """Populate one chooser without overwriting a selection mid-popup.

        The runtime viewer rebuilds often while points arrive. If a popup is currently
        open, leave it alone so the user can finish the interaction. Otherwise repopulate
        it, restore the requested selection if possible, and fall back to a sensible
        default.
        """
        if combo.is_popup_visible():
            return combo.currentData()

        combo.blockSignals(True)
        combo.clear()
        for key, label in choices:
            combo.addItem(label, key)
        index = combo.findData(selected_key) if selected_key is not None else -1
        if index == -1:
            default_key = _select_default_choice(
                choices,
                preferred_keys=preferred_keys,
                excluded_keys=excluded_keys,
            )
            index = combo.findData(default_key) if default_key is not None else -1
        if index != -1:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)
        return combo.currentData()

    def _sync_xyz_combos(
        self,
        selected_x_key: str | None,
        selected_x_index_tokens: tuple[str, ...] | None,
        selected_y_key: str | None,
        selected_y_index_tokens: tuple[str, ...] | None,
        selected_z_key: str | None,
        selected_z_index_tokens: tuple[str, ...] | None,
    ) -> None:
        x_choices = _default_x_choices(self._site)
        y_choices = _default_y_choices(self._site)
        z_choices = _default_z_choices(self._site)
        channel_keys = _channel_choice_keys(self._site)
        current_x_key = self._sync_choice_combo(
            self._x_combo,
            x_choices,
            selected_x_key,
            preferred_keys=_default_x_preferred_keys(self._site, x_choices),
        )
        current_plane_y_key = self._sync_choice_combo(
            self._plane_y_combo,
            y_choices,
            selected_y_key,
            preferred_keys=channel_keys,
            excluded_keys={current_x_key} if current_x_key is not None else None,
        )
        self._sync_choice_combo(
            self._z_combo,
            z_choices,
            selected_z_key,
            preferred_keys=channel_keys,
            excluded_keys={key for key in (current_x_key, current_plane_y_key) if key is not None},
        )
        self._sync_array_index_widget(
            self._x_index_widget,
            self._site.channels.get(current_x_key) if current_x_key is not None else None,
            selected_x_index_tokens,
        )
        self._sync_array_index_widget(
            self._plane_y_index_widget,
            self._site.channels.get(current_plane_y_key) if current_plane_y_key is not None else None,
            selected_y_index_tokens,
        )
        self._sync_array_index_widget(
            self._z_index_widget,
            self._site.channels.get(self._selected_z_key()) if self._selected_z_key() is not None else None,
            selected_z_index_tokens,
        )

    def _sync_array_index_widget(
        self,
        widget: _ArrayIndexWidget,
        schema: dict[str, Any] | None,
        selected_tokens: tuple[str, ...] | None,
    ) -> None:
        widget.blockSignals(True)
        widget.set_state(schema=schema, index_tokens=selected_tokens)
        widget.blockSignals(False)

    def _ensure_default_series_states(
        self,
        selected_series_states: tuple[_SeriesUiState, ...] | None,
        *,
        selected_y_key: str | None,
        selected_y_index_tokens: tuple[str, ...] | None,
        selected_group_key: str | None,
        selected_repeat_combine_mode: str | None,
    ) -> tuple[_SeriesUiState, ...]:
        if selected_series_states:
            return selected_series_states
        return (
            _SeriesUiState(
                y_key=selected_y_key,
                y_index_tokens=selected_y_index_tokens,
                group_key=selected_group_key,
                repeat_combine_mode=selected_repeat_combine_mode or _REPEAT_COMBINE_NONE,
            ),
        )

    def _sync_series_rows(
        self,
        selected_series_states: tuple[_SeriesUiState, ...],
    ) -> None:
        if not hasattr(self, "_series_rows"):
            self._series_rows: list[_SeriesControlRow] = []
        while len(self._series_rows) > len(selected_series_states):
            row = self._series_rows.pop()
            self._series_rows_layout.removeWidget(row)
            row.deleteLater()
        while len(self._series_rows) < len(selected_series_states):
            row = _SeriesControlRow()
            row.state_changed.connect(self._emit_series_states_changed)
            row.remove_requested.connect(self._remove_series_row)
            self._series_rows.append(row)
            self._series_rows_layout.addWidget(row)
        x_key = self._selected_x_key()
        for index, (row, state) in enumerate(zip(self._series_rows, selected_series_states, strict=True)):
            row.set_state(
                site=self._site,
                x_key=x_key,
                state=state,
                can_remove=len(selected_series_states) > 1,
            )

    def _current_series_states(self) -> tuple[_SeriesUiState, ...]:
        if not hasattr(self, "_series_rows"):
            return ()
        return tuple(row.current_state() for row in self._series_rows)

    def _emit_series_states_changed(self) -> None:
        self.series_states_changed.emit(self._site.path, self._current_series_states())

    def _add_series_row(self) -> None:
        states = list(self._current_series_states())
        states.append(_SeriesUiState())
        self._sync_series_rows(tuple(states))
        self._emit_series_states_changed()

    def _remove_series_row(self) -> None:
        sender = self.sender()
        if not isinstance(sender, _SeriesControlRow):
            return
        states = [row.current_state() for row in self._series_rows if row is not sender]
        if not states:
            states = [_SeriesUiState()]
        self._sync_series_rows(tuple(states))
        self._emit_series_states_changed()

    def _sync_fit_model_combo(self) -> None:
        self._sync_choice_combo(
            self._fit_model_combo,
            _fit_model_choices(self._fit_backend),
            self._current_combo_data(self._fit_model_combo),
            preferred_keys=[_NO_FIT_MODEL_KEY],
        )

    def _sync_fit_series_combo(self) -> None:
        current_data = self._current_combo_data(self._fit_series_combo)
        choices = [
            (index, series.label)
            for index, series in enumerate(self._current_visible_fit_series)
        ]
        if not choices:
            choices = [(0, "series 1")]
        self._sync_choice_combo(
            self._fit_series_combo,
            choices,
            current_data if current_data is not None else choices[0][0],
            preferred_keys=[choices[0][0]],
        )

    def _sync_fit_target_combo(self) -> None:
        fit_series_index = self._selected_fit_series_index()
        fit_series = (
            self._current_visible_fit_series[fit_series_index]
            if 0 <= fit_series_index < len(self._current_visible_fit_series)
            else None
        )
        self._sync_choice_combo(
            self._fit_target_combo,
            _fit_target_choices(
                None if fit_series is None or fit_series.group_values is None else "__group__"
            ),
            self._current_combo_data(self._fit_target_combo),
            preferred_keys=[_FIT_TARGET_VISIBLE],
        )

    def _selected_fit_model(self) -> str | None:
        model_id = self._current_combo_data(self._fit_model_combo)
        return None if model_id in {None, _NO_FIT_MODEL_KEY} else model_id

    def _selected_fit_series_index(self) -> int:
        return int(self._current_combo_data(self._fit_series_combo) or 0)

    def _selected_fit_target(self) -> str:
        return self._current_combo_data(self._fit_target_combo) or _FIT_TARGET_VISIBLE

    def _sync_fit_controls(self) -> None:
        fit_enabled = self._selected_fit_model() is not None
        show_fit = self._selected_plot_mode() == _PLOT_MODE_1D
        if (not show_fit or not fit_enabled) and self._fit_active:
            self._clear_fit()
        self._fit_controls_box.setVisible(show_fit)
        self._fit_series_label.setVisible(show_fit)
        self._fit_series_combo.setVisible(show_fit)
        self._fit_model_label.setVisible(show_fit)
        self._fit_model_combo.setVisible(show_fit)
        self._fit_target_label.setVisible(show_fit and fit_enabled)
        self._fit_target_combo.setVisible(show_fit and fit_enabled)
        self._fit_button.setVisible(show_fit)
        self._clear_fit_button.setVisible(show_fit)
        self._fit_readout.setVisible(show_fit)
        if not show_fit:
            self._fit_curve_item.hide()
            self._fit_readout.setText("fit: 1D only")
        elif not fit_enabled and not self._fit_active:
            self._fit_readout.setText("fit: none")

    def _emit_child_site_changed(self) -> None:
        if not self._child_site_options:
            return
        self.child_site_changed.emit(
            self._site.parent_path or (),
            self._child_combo.currentData(),
        )

    def _current_combo_data(self, combo: QtWidgets.QComboBox) -> Any:
        return combo.currentData()

    def _selected_x_key(self) -> str | None:
        return self._current_combo_data(self._x_combo)

    def _selected_x_index_tokens(self) -> tuple[str, ...] | None:
        return self._x_index_widget.current_tokens()

    def _selected_y_key(self) -> str | None:
        if self._selected_plot_mode() != _PLOT_MODE_1D:
            return self._current_combo_data(self._plane_y_combo)
        series_states = self._current_series_states()
        return series_states[0].y_key if series_states else None

    def _selected_y_index_tokens(self) -> tuple[str, ...] | None:
        if self._selected_plot_mode() != _PLOT_MODE_1D:
            return self._plane_y_index_widget.current_tokens()
        series_states = self._current_series_states()
        return series_states[0].y_index_tokens if series_states else None

    def _selected_z_key(self) -> str | None:
        return self._current_combo_data(self._z_combo)

    def _selected_z_index_tokens(self) -> tuple[str, ...] | None:
        return self._z_index_widget.current_tokens()

    def _annotation_axis_slice_matches(
        self,
        axis_key: str,
        expected_indices: Any,
        *,
        x_key: str,
        y_key: str,
    ) -> bool:
        expected = _annotation_index_tuple(expected_indices)
        if expected is None:
            return expected_indices is None

        if axis_key == x_key:
            schema = self._site.channels.get(x_key)
            return _selected_fixed_array_indices(
                schema if isinstance(schema, dict) else None,
                self._selected_x_index_tokens(),
            ) == expected

        if axis_key == y_key:
            schema = self._site.channels.get(y_key)
            return _selected_fixed_array_indices(
                schema if isinstance(schema, dict) else None,
                self._selected_y_index_tokens(),
            ) == expected

        z_key = self._selected_z_key()
        if z_key is not None and axis_key == z_key:
            schema = self._site.channels.get(z_key)
            return _selected_fixed_array_indices(
                schema if isinstance(schema, dict) else None,
                self._selected_z_index_tokens(),
            ) == expected

        return False

    def _annotation_xy_slice_matches(
        self,
        parameters: dict[str, Any],
        *,
        x_key: str,
        y_key: str,
    ) -> bool:
        return self._annotation_axis_slice_matches(
            x_key,
            parameters.get("x_indices"),
            x_key=x_key,
            y_key=y_key,
        ) and self._annotation_axis_slice_matches(
            y_key,
            parameters.get("y_indices"),
            x_key=x_key,
            y_key=y_key,
        )

    def _selected_plot_mode(self) -> str:
        return self._current_combo_data(self._plot_mode_combo) or _PLOT_MODE_1D

    def _selected_group_key(self) -> str | None:
        series_states = self._current_series_states()
        return series_states[0].group_key if series_states else None

    def _selected_repeat_combine_mode(self) -> str:
        series_states = self._current_series_states()
        if not series_states:
            return _REPEAT_COMBINE_NONE
        return series_states[0].repeat_combine_mode or _REPEAT_COMBINE_NONE

    def _sync_control_visibility(self) -> None:
        plot_mode = self._selected_plot_mode()
        is_bo = plot_mode == _PLOT_MODE_BO
        is_2d = plot_mode in {_PLOT_MODE_2D_SCATTER, _PLOT_MODE_2D_IMAGE}
        show_xy = not is_bo
        self._x_label.setVisible(show_xy)
        self._x_combo.setVisible(show_xy)
        self._x_index_label.setVisible(show_xy and self._x_index_widget.has_array_schema())
        self._x_index_widget.setVisible(show_xy and self._x_index_widget.has_array_schema())
        self._plane_y_label.setVisible(show_xy and is_2d)
        self._plane_y_combo.setVisible(show_xy and is_2d)
        self._plane_y_index_label.setVisible(
            show_xy and is_2d and self._plane_y_index_widget.has_array_schema()
        )
        self._plane_y_index_widget.setVisible(
            show_xy and is_2d and self._plane_y_index_widget.has_array_schema()
        )
        self._series_rows_widget.setVisible(show_xy and not is_2d)
        self._add_series_button.setVisible(show_xy and not is_2d)
        self._show_lines_checkbox.setVisible(show_xy and not is_2d)
        if hasattr(self, "_series_rows"):
            for row in self._series_rows:
                row.setVisible(show_xy and not is_2d)
                show_split = show_xy and not is_2d and row._split_combo.count() > 1
                row._split_label.setVisible(show_split)
                row._split_combo.setVisible(show_split)
                row._repeat_label.setVisible(show_xy and not is_2d)
                row._repeat_combo.setVisible(show_xy and not is_2d)
        self._z_row_widget.setVisible(is_2d)
        self._z_label.setVisible(is_2d)
        self._z_combo.setVisible(is_2d)
        self._z_index_label.setVisible(is_2d and self._z_index_widget.has_array_schema())
        self._z_index_widget.setVisible(is_2d and self._z_index_widget.has_array_schema())
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        self.updateGeometry()

    def _point_clicked(self, scatter_item, points, event) -> None:
        if len(points) == 0:
            return
        distances = np.array([(point.pos() - event.pos()).length() for point in points])
        point = points[int(distances.argmin())]
        selection = point.data()
        self.point_selected.emit(self._site.path, selection)

    def _background_clicked(self, event) -> None:
        if event.isAccepted():
            return
        self.point_selected.emit(self._site.path, None)

    def _mouse_moved(self, scene_pos) -> None:
        """Update the cursor readout from scene coordinates."""
        if scene_pos is None:
            self._last_cursor_plot_pos = None
            self._update_cursor_readout()
            return
        if not self._plot_widget.sceneBoundingRect().contains(scene_pos):
            self._last_cursor_plot_pos = None
            self._update_cursor_readout()
            return
        view_pos = self._plot_item.vb.mapSceneToView(scene_pos)
        self._last_cursor_plot_pos = (float(view_pos.x()), float(view_pos.y()))
        self._update_cursor_readout()

    def _clear_plot_items(self) -> None:
        """Hide all plot elements when the current selection cannot be rendered."""
        self._line_item.setData([], [])
        for line_item in self._extra_line_items:
            line_item.setData([], [])
        _clear_error_bar_item(self._error_bar_item)
        for error_bar_item in self._extra_error_bar_items:
            _clear_error_bar_item(error_bar_item)
        self._clear_annotation_items()
        self._scatter.setData([])
        self._summary_scatter.setData([])
        self._highlight.setData([])
        self._image_item.hide()
        self._crosshair_x.hide()
        self._crosshair_y.hide()
        self._colorbar.set_state(visible=False)
        self._fit_curve_item.hide()
        self._legend.clear()
        self._legend.hide()
        self._rendered_point_values.clear()
        self._rendered_group_values.clear()
        self._rendered_group_displays.clear()
        self._rendered_source_indices.clear()
        self._rendered_series_indices.clear()
        self._rendered_series_labels.clear()
        self._rendered_group_label_map.clear()
        self._group_colors.clear()
        self._current_plot_arrays = None
        self._current_display_point_keys = []
        self._current_group_values = None
        self._current_visible_fit_series = []
        self._selected_readout.setText("selected: none")
        artifact_source_kind, display_artifacts = _annotation_artifact_map_for_display(
            self._site, self._parent_point_index
        )
        self._artifact_readout.setText(
            _format_artifact_readout(artifact_source_kind, display_artifacts)
        )
        self._update_plot_overlay_text()

    def _clear_annotation_items(self) -> None:
        """Remove all persisted-analysis overlay items from the plot."""

        for item in self._annotation_items:
            self._plot_item.removeItem(item)
        self._annotation_items.clear()

    def _add_location_annotation_line(
        self,
        *,
        value: float,
        vertical: bool,
        pen: QtGui.QPen,
        error: float | None = None,
    ) -> None:
        """Add one location annotation, optionally with simple error bounds."""

        main_line = pg.InfiniteLine(
            angle=90 if vertical else 0,
            movable=False,
            pen=pen,
        )
        main_line.setPos(value)
        self._plot_item.addItem(main_line, ignoreBounds=True)
        self._annotation_items.append(main_line)

        if error is None or error <= 0.0 or not np.isfinite(error):
            return

        side_pen = QtGui.QPen(pen)
        side_pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        for offset in (-error, error):
            line = pg.InfiniteLine(
                angle=90 if vertical else 0,
                movable=False,
                pen=side_pen,
            )
            line.setPos(value + offset)
            self._plot_item.addItem(line, ignoreBounds=True)
            self._annotation_items.append(line)

    def _render_annotation_curve(
        self,
        render_spec: _AnnotationRenderSpec,
        *,
        x_key: str,
        y_key: str,
    ) -> None:
        """Render one explicit curve annotation when it matches the selected axes."""

        annotation = render_spec.annotation
        parameters = annotation.get("parameters", {})
        if not self._annotation_xy_slice_matches(parameters, x_key=x_key, y_key=y_key):
            return
        coordinates = annotation.get("coordinates", {})
        x_values = _annotation_curve_data(
            _annotation_value_from_spec(self._site, render_spec, coordinates.get(x_key))
        )
        y_values = _annotation_curve_data(
            _annotation_value_from_spec(self._site, render_spec, coordinates.get(y_key))
        )
        if x_values is None or y_values is None or len(x_values) != len(y_values):
            return

        item = self._plot_item.plot(
            x_values,
            y_values,
            pen=_annotation_pen(render_spec.source_kind),
        )
        self._annotation_items.append(item)

    def _render_annotation_computed_curve(
        self,
        render_spec: _AnnotationRenderSpec,
        *,
        x_key: str,
        y_key: str,
        x: np.ndarray,
    ) -> None:
        """Render a computed-curve annotation over the visible x range."""

        annotation = render_spec.annotation
        parameters = annotation.get("parameters", {})
        function_name = parameters.get("function_name")
        if function_name not in FIT_OBJECTS:
            return
        if not self._annotation_xy_slice_matches(parameters, x_key=x_key, y_key=y_key):
            return

        associated_channels = parameters.get("associated_channels")
        if associated_channels is not None and y_key not in associated_channels:
            return

        function_parameters = {}
        for name, spec in annotation.get("data", {}).items():
            value = _annotation_scalar_value(
                _annotation_value_from_spec(self._site, render_spec, spec)
            )
            if value is None:
                return
            function_parameters[name] = value

        if len(x) == 0:
            return
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        curve_x = np.linspace(x_min, x_max, 200)
        curve_y = FIT_OBJECTS[function_name].fitting_function(curve_x, function_parameters)
        item = self._plot_item.plot(
            curve_x,
            curve_y,
            pen=_annotation_pen(render_spec.source_kind),
        )
        self._annotation_items.append(item)

    def _render_annotation_artifact_curve(
        self,
        render_spec: _AnnotationRenderSpec,
        *,
        x_key: str,
        y_key: str,
        x: np.ndarray,
    ) -> None:
        """Render a parametric curve from a named fit artifact."""

        annotation = render_spec.annotation
        parameters = annotation.get("parameters", {})
        if parameters.get("x_axis") != x_key or parameters.get("y_axis") != y_key:
            return
        if not self._annotation_xy_slice_matches(parameters, x_key=x_key, y_key=y_key):
            return

        associated_channels = parameters.get("associated_channels")
        if associated_channels is not None and y_key not in associated_channels:
            return

        artifact_name = parameters.get("artifact")
        if not artifact_name:
            return
        artifact = render_spec.artifacts.get(artifact_name)
        if artifact is None or len(x) == 0:
            return

        try:
            curve_x, curve_y = curve_points_for_artifact(
                artifact,
                x_min=float(np.min(x)),
                x_max=float(np.max(x)),
                num_points=200,
            )
        except Exception:
            return

        item = self._plot_item.plot(
            curve_x,
            curve_y,
            pen=_annotation_pen(render_spec.source_kind),
        )
        self._annotation_items.append(item)

    def _render_annotation_artifact_location(
        self,
        render_spec: _AnnotationRenderSpec,
        *,
        x_key: str,
        y_key: str,
    ) -> None:
        """Render a marker line from one named parameter inside an artifact."""

        annotation = render_spec.annotation
        parameters = annotation.get("parameters", {})
        associated_channels = parameters.get("associated_channels")
        if associated_channels is not None and y_key not in associated_channels:
            return

        axis_key = parameters.get("axis")
        artifact_name = parameters.get("artifact")
        parameter_name = parameters.get("parameter")
        if not axis_key or not artifact_name or not parameter_name:
            return
        if not self._annotation_axis_slice_matches(
            axis_key,
            parameters.get("axis_indices"),
            x_key=x_key,
            y_key=y_key,
        ):
            return

        artifact = render_spec.artifacts.get(artifact_name)
        if not isinstance(artifact, dict):
            return
        parameter_spec = dict(artifact.get("parameters", {})).get(parameter_name, {})
        value = _annotation_scalar_value(parameter_spec.get("value"))
        if value is None:
            return

        error_value = None
        error_parameter = parameters.get("error_parameter")
        if error_parameter:
            error_spec = dict(artifact.get("parameters", {})).get(error_parameter, {})
            error_value = _annotation_scalar_value(error_spec.get("value"))
        else:
            error_value = _annotation_scalar_value(parameter_spec.get("stderr"))

        if axis_key == x_key:
            self._add_location_annotation_line(
                value=value,
                vertical=True,
                pen=_annotation_pen(render_spec.source_kind, width=1.8),
                error=error_value,
            )
        elif axis_key == y_key:
            self._add_location_annotation_line(
                value=value,
                vertical=False,
                pen=_annotation_pen(render_spec.source_kind, width=1.8),
                error=error_value,
            )

    def _render_annotation_location(
        self,
        render_spec: _AnnotationRenderSpec,
        *,
        x_key: str,
        y_key: str,
    ) -> None:
        """Render simple axis-location annotations in the current 1D view."""

        annotation = render_spec.annotation
        parameters = annotation.get("parameters", {})
        associated_channels = parameters.get("associated_channels")
        if associated_channels is not None and y_key not in associated_channels:
            return

        coordinates = annotation.get("coordinates", {})
        data = annotation.get("data", {})
        for axis_key, spec in coordinates.items():
            if not self._annotation_axis_slice_matches(
                axis_key,
                parameters.get("axis_indices"),
                x_key=x_key,
                y_key=y_key,
            ):
                continue
            value = _annotation_scalar_value(
                _annotation_value_from_spec(self._site, render_spec, spec)
            )
            if value is None:
                continue

            error_value = None
            if axis_key in {x_key, y_key}:
                error_key = f"{axis_key}_error"
                error_value = _annotation_scalar_value(
                    _annotation_value_from_spec(
                        self._site, render_spec, data.get(error_key)
                    )
                )

            if axis_key == x_key:
                self._add_location_annotation_line(
                    value=value,
                    vertical=True,
                    pen=_annotation_pen(render_spec.source_kind, width=1.8),
                    error=error_value,
                )
            elif axis_key == y_key:
                self._add_location_annotation_line(
                    value=value,
                    vertical=False,
                    pen=_annotation_pen(render_spec.source_kind, width=1.8),
                    error=error_value,
                )

    def _render_annotations(
        self,
        plot_mode: str,
        *,
        x_key: str,
        y_key: str,
        x: np.ndarray,
    ) -> None:
        """Render final or online analysis annotations for the current site slice.

        This first pass intentionally keeps runtime annotations simple:

        - 1D only,
        - static redraw on every viewer update,
        - explicit curves, computed curves, and axis markers.

        That covers the current examples cleanly while keeping the implementation
        understandable for future editing.
        """

        self._clear_annotation_items()
        if plot_mode != _PLOT_MODE_1D:
            return

        for render_spec in _annotation_specs_for_display(self._site, self._parent_point_index):
            kind = render_spec.annotation.get("kind")
            if kind == "curve":
                self._render_annotation_curve(
                    render_spec,
                    x_key=x_key,
                    y_key=y_key,
                )
            elif kind == "computed_curve":
                self._render_annotation_computed_curve(
                    render_spec,
                    x_key=x_key,
                    y_key=y_key,
                    x=x,
                )
            elif kind == "artifact_curve":
                self._render_annotation_artifact_curve(
                    render_spec,
                    x_key=x_key,
                    y_key=y_key,
                    x=x,
                )
            elif kind == "artifact_location":
                self._render_annotation_artifact_location(
                    render_spec,
                    x_key=x_key,
                    y_key=y_key,
                )
            elif kind == "location":
                self._render_annotation_location(
                    render_spec,
                    x_key=x_key,
                    y_key=y_key,
                )

    def _update_colorbar(self, z_key: str | None, z: np.ndarray) -> None:
        if z_key is None or len(z) == 0:
            self._colorbar.set_state(visible=False)
            return
        finite = np.isfinite(z)
        if not finite.any():
            self._colorbar.set_state(visible=False)
            return
        z_label_map = _choice_label_map(_default_z_choices(self._site))
        self._colorbar.set_state(
            visible=True,
            label=z_label_map.get(z_key, z_key),
            z_min=float(np.min(z[finite])),
            z_max=float(np.max(z[finite])),
        )

    def _point_stream_values(self, key: str) -> list[Any]:
        """Return one point stream for the active site slice.

        Most streams come straight from the site data. ``point_index`` is synthetic and
        is derived from the preserved source indices instead.
        """
        if key == _POINT_INDEX_KEY:
            return list(self._plot_data.source_indices)
        return list(self._plot_data.raw_points.get(key, []))

    def _clear_fit(self) -> None:
        """Remove the current viewer-side fit overlay."""
        self._fit_active = False
        self._fit_result = None
        self._fit_target_group_value = None
        self._fit_curve_item.setData([], [])
        self._fit_curve_item.hide()
        self._fit_readout.setText("fit: none")
        self._update_plot_overlay_text(
            x_key=self._selected_x_key(),
            y_key=self._selected_y_key(),
        )

    def _resolved_selected_display_point_key(self) -> _DisplayPointKey | None:
        """Return the visible marker currently selected in this plot.

        The viewer persists both the underlying logical source point and, when the user
        clicked a specific visible marker, a display-point key. The latter is needed for
        array-channel fan-out and other cases where one source point produces multiple
        plotted markers.
        """

        if (
            self._selected_display_point_key is not None
            and self._selected_display_point_key in self._rendered_point_values
        ):
            return self._selected_display_point_key
        if self._selected_point_index is None:
            return None
        for key, source_index in self._rendered_source_indices.items():
            if int(source_index) == int(self._selected_point_index):
                return key
        return None

    def _fit_request_data(self) -> tuple[FitRequest, QtGui.QColor | str] | None:
        """Return the currently selected 1D data slice for viewer-side fitting."""
        if self._current_plot_mode != _PLOT_MODE_1D or not self._current_visible_fit_series:
            self._fit_readout.setText("fit: 1D only")
            return None

        fit_series_index = self._selected_fit_series_index()
        if fit_series_index < 0 or fit_series_index >= len(self._current_visible_fit_series):
            self._fit_readout.setText("fit: choose a visible series")
            return None
        fit_series = self._current_visible_fit_series[fit_series_index]
        x, y = fit_series.x, fit_series.y
        target = self._selected_fit_target()
        if target == _FIT_TARGET_VISIBLE or fit_series.group_values is None:
            self._fit_target_group_value = None
            return FitRequest(x=x, y=y), "#ff7f0e"

        selected_display_key = self._resolved_selected_display_point_key()
        if selected_display_key is None:
            self._fit_readout.setText("fit: select a point to choose its group")
            return None
        selected_series_index = self._rendered_series_indices.get(selected_display_key)
        if selected_series_index != fit_series_index:
            self._fit_readout.setText("fit: select a point from the chosen series")
            return None
        selected_group = self._rendered_group_values.get(selected_display_key)
        if selected_group is None:
            self._fit_readout.setText("fit: selected point group is not visible")
            return None

        if fit_series.group_values is None:
            self._fit_readout.setText("fit: selected group is not available")
            return None
        group_mask = np.asarray(fit_series.group_values == float(selected_group), dtype=bool)
        if not group_mask.any():
            self._fit_readout.setText("fit: selected group has no visible points")
            return None
        self._fit_target_group_value = float(selected_group)
        color = self._group_colors.get(float(selected_group), QtGui.QColor("#ff7f0e"))
        return FitRequest(x=x[group_mask], y=y[group_mask]), color

    def _fit_now(self) -> None:
        """Fit the currently visible 1D data with the selected viewer-side model."""
        model_id = self._selected_fit_model()
        if model_id is None:
            self._fit_readout.setText("fit: choose a model")
            return

        request_data = self._fit_request_data()
        if request_data is None:
            self._fit_result = None
            self._fit_curve_item.hide()
            return

        request, color = request_data
        try:
            result = self._fit_backend.fit(model_id, request)
        except Exception as exc:
            self._fit_active = False
            self._fit_result = None
            self._fit_curve_item.hide()
            self._fit_readout.setText(f"fit: {exc}")
            self._update_plot_overlay_text()
            return

        self._fit_active = True
        self._fit_result = result
        self._fit_curve_item.setPen(
            pg.mkPen(color, width=2.2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self._fit_curve_item.setData(result.curve_x, result.curve_y)
        self._fit_curve_item.show()
        summary = result.summary()
        fit_series_index = self._selected_fit_series_index()
        fit_series_label = (
            self._current_visible_fit_series[fit_series_index].label
            if 0 <= fit_series_index < len(self._current_visible_fit_series)
            else "series"
        )
        include_series_label = len(self._current_visible_fit_series) > 1
        if self._fit_target_group_value is not None:
            selected_key = self._resolved_selected_display_point_key()
            group_display = (
                self._rendered_group_displays.get(selected_key)
                if selected_key is not None
                else None
            )
            if not group_display:
                group_display = self._rendered_group_label_map.get(
                    float(self._fit_target_group_value),
                    _format_readout_value(self._fit_target_group_value),
                )
            prefix = (
                f"fit: {model_id} on {fit_series_label}, "
                if include_series_label
                else f"fit: {model_id} on "
            )
            self._fit_readout.setText(
                prefix + f"{self._current_group_label} = {group_display}; {summary}"
            )
        else:
            if include_series_label:
                self._fit_readout.setText(f"fit: {model_id} on {fit_series_label}; {summary}")
            else:
                self._fit_readout.setText(f"fit: {model_id}; {summary}")
        self._update_plot_overlay_text()

    def _update_selected_readout(self, plot_mode: str) -> None:
        """Show the numeric value of the selected sampled point."""
        selected_key = self._resolved_selected_display_point_key()
        if selected_key is None:
            self._selected_readout.setText("selected: none")
            return

        values = self._rendered_point_values.get(selected_key)
        if values is None:
            self._selected_readout.setText("selected: current point not visible")
            return

        source_index = self._rendered_source_indices.get(selected_key)
        x_value, y_value, z_value = values
        series_index = self._rendered_series_indices.get(selected_key)
        y_label = (
            self._rendered_series_labels.get(series_index, self._current_y_label)
            if series_index is not None and len(self._rendered_series_labels) > 1
            else self._current_y_label
        )
        parts = [
            f"{self._current_x_label} = {_format_readout_value(x_value)}",
            f"{y_label} = {_format_readout_value(y_value)}",
        ]
        if series_index is not None and len(self._rendered_series_labels) > 1:
            parts.append(
                f"series = {self._rendered_series_labels.get(series_index, f'series {series_index + 1}')}"
            )
        if plot_mode in {_PLOT_MODE_2D_SCATTER, _PLOT_MODE_2D_IMAGE} and z_value is not None:
            parts.append(f"{self._current_z_label} = {_format_readout_value(z_value)}")
        if plot_mode == _PLOT_MODE_1D and self._current_group_label is not None:
            group_value = self._rendered_group_values.get(selected_key)
            if group_value is not None:
                group_display = self._rendered_group_displays.get(
                    selected_key,
                    self._rendered_group_label_map.get(
                        float(group_value),
                        _format_readout_value(group_value),
                    ),
                )
                parts.append(
                    f"{self._current_group_label} = {group_display}"
                )
        self._selected_readout.setText(
            f"selected point {int(source_index) if source_index is not None else '?'}: "
            + ", ".join(parts)
        )

    def _update_cursor_readout(self) -> None:
        """Show the current cursor position in plot coordinates."""
        if self._last_cursor_plot_pos is None:
            self._cursor_readout.setText("cursor: move over plot")
            return
        x_value, y_value = self._last_cursor_plot_pos
        self._cursor_readout.setText(
            "cursor: "
            f"{self._current_x_label} = {_format_readout_value(x_value)}, "
            f"{self._current_y_label} = {_format_readout_value(y_value)}"
        )

    def _set_error_bars_item(
        self,
        item: pg.ErrorBarItem,
        *,
        x_values: np.ndarray,
        y_values: np.ndarray,
        error_values: np.ndarray | None,
        color: QtGui.QColor | str,
    ) -> None:
        if error_values is None or len(error_values) != len(y_values):
            _clear_error_bar_item(item)
            return
        finite = np.isfinite(error_values) & (error_values > 0.0)
        if finite.any():
            item.setOpts(pen=_series_error_bar_pen(color))
            item.setData(
                x=x_values[finite],
                y=y_values[finite],
                top=error_values[finite],
                bottom=error_values[finite],
            )
        else:
            _clear_error_bar_item(item)

    def _ensure_trace_items(self, count: int) -> tuple[list[pg.PlotDataItem], list[pg.ErrorBarItem]]:
        while len(self._extra_line_items) + 1 < count:
            self._extra_line_items.append(self._plot_item.plot())
        while len(self._extra_error_bar_items) + 1 < count:
            error_bar_item = pg.ErrorBarItem(beam=0.0)
            self._extra_error_bar_items.append(error_bar_item)
            self._plot_item.addItem(error_bar_item)
        all_line_items = [self._line_item, *self._extra_line_items]
        all_error_bar_items = [self._error_bar_item, *self._extra_error_bar_items]
        for line_item in all_line_items[count:]:
            line_item.setData([], [])
        for error_bar_item in all_error_bar_items[count:]:
            _clear_error_bar_item(error_bar_item)
        return all_line_items[:count], all_error_bar_items[:count]

    def _render_multi_series_1d(
        self,
        *,
        x_key: str,
        x_label_map: dict[str, str],
        y_label_map: dict[str, str],
    ) -> bool:
        self._colorbar.set_state(visible=False)
        self._legend.clear()
        self._legend.hide()
        self._image_item.hide()
        self._crosshair_x.hide()
        self._crosshair_y.hide()
        self._clear_annotation_items()
        self._rendered_group_displays.clear()
        self._rendered_group_label_map = {}
        self._group_colors.clear()

        raw_scatter_points: list[dict[str, Any]] = []
        summary_scatter_points: list[dict[str, Any]] = []
        rendered_entries: list[tuple[_DisplayPointKey, int, float, float, float | None, float | None, int]] = []
        fit_series: list[_VisibleFitSeries] = []
        line_payloads: list[tuple[np.ndarray, np.ndarray, QtGui.QColor, str]] = []
        error_payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, QtGui.QColor]] = []
        repeated_status_bits: list[str] = []

        for series_index, series_state in enumerate(self._current_series_states()):
            if series_state.y_key is None:
                continue
            prepared = self._prepare_plot_arrays(
                plot_mode=_PLOT_MODE_1D,
                x_key=x_key,
                y_key=series_state.y_key,
                group_key=series_state.group_key,
                y_index_tokens=series_state.y_index_tokens,
            )
            if prepared is None:
                continue

            series_label = _axis_label_with_indices(
                y_label_map.get(series_state.y_key, series_state.y_key),
                site=self._site,
                key=series_state.y_key,
                index_tokens=series_state.y_index_tokens,
            )

            repeat_mode = series_state.repeat_combine_mode or _REPEAT_COMBINE_NONE
            aggregated = _aggregate_repeated_1d_points(
                prepared.x,
                prepared.y,
                prepared.source_indices,
                prepared.group_values,
            )
            use_aggregated = (
                repeat_mode != _REPEAT_COMBINE_NONE and aggregated.used_repeats
            )
            display_x = aggregated.x if use_aggregated else prepared.x
            display_y = aggregated.y if use_aggregated else prepared.y
            display_group_values = (
                aggregated.group_values if use_aggregated else prepared.group_values
            )
            display_y_error = (
                aggregated.y_std
                if use_aggregated and repeat_mode == _REPEAT_COMBINE_STD
                else aggregated.y_sem
                if use_aggregated and repeat_mode == _REPEAT_COMBINE_SEM
                else prepared.y_error
            )
            if use_aggregated:
                repeat_mode_labels = dict(_REPEAT_COMBINE_CHOICES)
                repeated_status_bits.append(
                    f"repeated x: {repeat_mode_labels.get(repeat_mode, 'combined')}"
                )
            elif prepared.array_series_label_map is not None and series_state.group_key is None:
                repeated_status_bits.append(
                    f"repeated {prepared.array_series_label_name or 'series'} values"
                )
            fit_series.append(
                _VisibleFitSeries(
                    label=series_label,
                    x=display_x.copy(),
                    y=display_y.copy(),
                    group_values=(
                        None
                        if display_group_values is None
                        else display_group_values.copy()
                    ),
                )
            )
            raw_point_keys = [
                ("series_point", series_index, point_index)
                for point_index in range(len(prepared.source_indices))
            ]
            display_point_keys = (
                [
                    ("series_summary", series_index, point_index)
                    for point_index in range(len(display_x))
                ]
                if use_aggregated
                else list(raw_point_keys)
            )

            if display_group_values is None:
                color = (
                    QtGui.QColor("#1f77b4")
                    if len(self._current_series_states()) == 1
                    else QtGui.QColor(
                        pg.intColor(len(line_payloads), hues=max(3, len(self._current_series_states()) + 1))
                    )
                )
                line_payloads.append((display_x, display_y, color, series_label))
                error_payloads.append((display_x, display_y, display_y_error, color))
                for point_index, (xi, yi, point_key, source_index) in enumerate(
                    zip(prepared.x, prepared.y, raw_point_keys, prepared.source_indices, strict=True)
                ):
                    raw_scatter_points.append(
                        {
                            "pos": (float(xi), float(yi)),
                            "data": _DisplayedPointSelection(int(source_index), point_key),
                            "brush": pg.mkBrush(_alpha_color(color, 40)) if use_aggregated else pg.mkBrush(color),
                            "pen": pg.mkPen(_alpha_color(color, 60)) if use_aggregated else pg.mkPen(30, 30, 30, 120),
                            "size": 6 if use_aggregated else 9,
                        }
                    )
                    rendered_entries.append(
                        (point_key, int(source_index), float(xi), float(yi), None, None, series_index)
                    )
                if use_aggregated:
                    for point_index, (xi, yi, point_key, source_index) in enumerate(
                        zip(display_x, display_y, display_point_keys, aggregated.source_indices, strict=True)
                    ):
                        summary_scatter_points.append(
                            {
                                "pos": (float(xi), float(yi)),
                                "data": _DisplayedPointSelection(int(source_index), point_key),
                                "brush": pg.mkBrush(color),
                                "pen": pg.mkPen(30, 30, 30, 150),
                                "size": 10,
                            }
                        )
                        rendered_entries.append(
                            (point_key, int(source_index), float(xi), float(yi), None, None, series_index)
                        )
                continue

            groups: dict[float, list[int]] = {}
            for point_index, group_value in enumerate(display_group_values):
                groups.setdefault(float(group_value), []).append(point_index)
            group_color_map: dict[float, QtGui.QColor] = {}
            for group_value, indices in groups.items():
                color = QtGui.QColor(pg.intColor(len(line_payloads), hues=max(3, len(line_payloads) + len(groups) + 1)))
                group_color_map[float(group_value)] = color
                self._group_colors[float(group_value)] = color
                group_display = (
                    prepared.group_label_map.get(float(group_value), _format_readout_value(group_value))
                    if prepared.group_label_map is not None
                    else _format_readout_value(group_value)
                )
                self._rendered_group_label_map[float(group_value)] = group_display
                trace_label = f"{series_label}; {prepared.group_label_name or 'group'} = {group_display}"
                ordered_indices = np.asarray(indices, dtype=int)[
                    np.argsort(display_x[indices], kind="stable")
                ]
                line_payloads.append(
                    (display_x[ordered_indices], display_y[ordered_indices], color, trace_label)
                )
                group_error = (
                    None
                    if display_y_error is None
                    else np.asarray(display_y_error[indices], dtype=float)
                )
                error_payloads.append(
                    (display_x[indices], display_y[indices], group_error, color)
                )
            for raw_index, (xi, yi, point_key, source_index, raw_group_value) in enumerate(
                zip(
                    prepared.x,
                    prepared.y,
                    raw_point_keys,
                    prepared.source_indices,
                    [None] * len(prepared.source_indices)
                    if prepared.group_values is None
                    else prepared.group_values,
                    strict=True,
                )
            ):
                raw_color = (
                    group_color_map.get(float(raw_group_value), QtGui.QColor("#1f77b4"))
                    if raw_group_value is not None
                    else QtGui.QColor("#1f77b4")
                )
                raw_scatter_points.append(
                    {
                        "pos": (float(xi), float(yi)),
                        "data": _DisplayedPointSelection(int(source_index), point_key),
                        "brush": pg.mkBrush(_alpha_color(raw_color, 40)) if use_aggregated else pg.mkBrush(raw_color),
                        "pen": pg.mkPen(_alpha_color(raw_color, 60)) if use_aggregated else pg.mkPen(30, 30, 30, 120),
                        "size": 6 if use_aggregated else 9,
                    }
                )
                group_display = (
                    prepared.group_label_map.get(float(raw_group_value), _format_readout_value(raw_group_value))
                    if raw_group_value is not None and prepared.group_label_map is not None
                    else _format_readout_value(raw_group_value) if raw_group_value is not None else ""
                )
                rendered_entries.append(
                    (
                        point_key,
                        int(source_index),
                        float(xi),
                        float(yi),
                        None,
                        None if raw_group_value is None else float(raw_group_value),
                        series_index,
                    )
                )
                self._rendered_group_displays[point_key] = group_display
            if use_aggregated:
                for point_index, (xi, yi, point_key, source_index, group_value) in enumerate(
                    zip(
                        display_x,
                        display_y,
                        display_point_keys,
                        aggregated.source_indices,
                        display_group_values,
                        strict=True,
                    )
                ):
                    color = group_color_map.get(float(group_value), QtGui.QColor("#1f77b4"))
                    summary_scatter_points.append(
                        {
                            "pos": (float(xi), float(yi)),
                            "data": _DisplayedPointSelection(int(source_index), point_key),
                            "brush": pg.mkBrush(color),
                            "pen": pg.mkPen(30, 30, 30, 150),
                            "size": 10,
                        }
                    )
                    rendered_entries.append(
                        (
                            point_key,
                            int(source_index),
                            float(xi),
                            float(yi),
                            None,
                            float(group_value),
                            series_index,
                        )
                    )
                    group_display = (
                        prepared.group_label_map.get(float(group_value), _format_readout_value(group_value))
                        if prepared.group_label_map is not None
                        else _format_readout_value(group_value)
                    )
                    self._rendered_group_displays[point_key] = group_display

        if not fit_series:
            return False

        all_line_items, all_error_bar_items = self._ensure_trace_items(len(line_payloads))
        for item, payload in zip(all_line_items, line_payloads, strict=True):
            x_values, y_values, color, label = payload
            item.setPen(pg.mkPen(color, width=1.8))
            if self._show_lines_checkbox.isChecked():
                item.setData(x_values, y_values)
            else:
                item.setData([], [])
            self._legend.addItem(item, label)
        for item, payload in zip(all_error_bar_items, error_payloads, strict=True):
            x_values, y_values, error_values, color = payload
            self._set_error_bars_item(
                item,
                x_values=x_values,
                y_values=y_values,
                error_values=error_values,
                color=color,
            )
        self._legend.setVisible(len(line_payloads) > 1)
        self._scatter.setData(raw_scatter_points)
        self._summary_scatter.setData(summary_scatter_points)

        self._current_visible_fit_series = fit_series
        self._sync_fit_series_combo()
        self._sync_fit_target_combo()

        first_series = fit_series[0]
        self._current_plot_arrays = (first_series.x.copy(), first_series.y.copy())
        self._current_group_values = (
            None if first_series.group_values is None else first_series.group_values.copy()
        )
        self._current_x_label = _axis_label_with_indices(
            x_label_map.get(x_key, x_key if x_key != _POINT_INDEX_KEY else "point_index"),
            site=self._site,
            key=x_key,
            index_tokens=self._selected_x_index_tokens(),
        )
        self._current_y_label = first_series.label if len(fit_series) == 1 else "value"
        self._plot_item.setLabel("bottom", self._current_x_label)
        self._plot_item.setLabel("left", self._current_y_label)
        self._current_plot_mode = _PLOT_MODE_1D
        self._current_z_label = None
        first_group_label = None
        first_group_key = self._current_series_states()[0].group_key if self._current_series_states() else None
        if first_group_key is not None:
            if first_group_key == _ARRAY_SERIES_GROUP_KEY:
                first_series_state = self._current_series_states()[0]
                y_schema = self._site.channels.get(first_series_state.y_key) if first_series_state.y_key is not None else None
                first_group_label = _ranged_dim_name_for_selection(
                    y_schema if isinstance(y_schema, dict) else None,
                    first_series_state.y_index_tokens,
                ) or "group"
            else:
                group_label_map = _choice_label_map(_group_by_choices(self._site, x_key))
                first_group_label = group_label_map.get(first_group_key, first_group_key)
        self._current_group_label = first_group_label
        self._rendered_point_values = {
            point_key: (x_value, y_value, z_value)
            for point_key, _, x_value, y_value, z_value, _, _ in rendered_entries
        }
        self._rendered_source_indices = {
            point_key: source_index
            for point_key, source_index, _, _, _, _, _ in rendered_entries
        }
        self._rendered_group_values = {
            point_key: group_value
            for point_key, _, _, _, _, group_value, _ in rendered_entries
            if group_value is not None
        }
        self._rendered_series_indices = {
            point_key: series_index
            for point_key, _, _, _, _, _, series_index in rendered_entries
        }
        self._rendered_series_labels = {
            index: series.label for index, series in enumerate(fit_series)
        }
        status = self._plot_data.mode_label
        split_labels = []
        for series_state in self._current_series_states():
            if series_state.group_key is None:
                continue
            if series_state.group_key == _ARRAY_SERIES_GROUP_KEY:
                y_schema = self._site.channels.get(series_state.y_key) if series_state.y_key is not None else None
                dim_name = _ranged_dim_name_for_selection(
                    y_schema if isinstance(y_schema, dict) else None,
                    series_state.y_index_tokens,
                )
                if dim_name:
                    split_labels.append(f"split by {dim_name}")
            else:
                group_label_map = _choice_label_map(_group_by_choices(self._site, x_key))
                split_labels.append(
                    f"grouped by {group_label_map.get(series_state.group_key, series_state.group_key)}"
                )
        if split_labels:
            status += "; " + "; ".join(split_labels)
        if repeated_status_bits:
            status += "; " + "; ".join(dict.fromkeys(repeated_status_bits))
        if len(fit_series) > 1:
            status += f"; {len(fit_series)} series"
        self._status.setText(status)
        self._highlight_selected_point(_PLOT_MODE_1D)
        self._update_selected_readout(_PLOT_MODE_1D)
        self._update_cursor_readout()
        return True

    def _render_1d_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_error: np.ndarray | None,
        point_keys: list[_DisplayPointKey],
        source_indices: list[int],
        group_key: str | None,
        group_values: np.ndarray | None,
        group_label_map: dict[float, str] | None,
    ) -> _Rendered1DView:
        """Render a 1D scatter/line view.

        When a repeat-combine mode is enabled, repeated x-values collapse into one
        foreground mean trace and the raw repeated points stay visible as a faded
        clickable backdrop.
        """
        self._colorbar.set_state(visible=False)
        self._legend.clear()
        self._legend.hide()

        aggregated = _aggregate_repeated_1d_points(x, y, source_indices, group_values)
        repeat_combine_mode = self._selected_repeat_combine_mode()
        use_aggregated = (
            repeat_combine_mode != _REPEAT_COMBINE_NONE and aggregated.used_repeats
        )
        display_x = aggregated.x if use_aggregated else x
        display_y = aggregated.y if use_aggregated else y
        display_point_keys = (
            [("summary", index) for index in range(len(aggregated.x))]
            if use_aggregated
            else list(point_keys)
        )
        display_source_indices = (
            aggregated.source_indices if use_aggregated else list(source_indices)
        )
        display_group_values = aggregated.group_values if use_aggregated else group_values
        base_series_color = QtGui.QColor("#1f77b4")

        if use_aggregated:
            display_y_error = (
                aggregated.y_std
                if repeat_combine_mode == _REPEAT_COMBINE_STD
                else aggregated.y_sem
            )
        else:
            display_y_error = y_error

        def set_error_bars(
            item: pg.ErrorBarItem,
            *,
            x_values: np.ndarray,
            y_values: np.ndarray,
            error_values: np.ndarray | None,
            color: QtGui.QColor | str,
        ) -> None:
            if error_values is None or len(error_values) != len(y_values):
                _clear_error_bar_item(item)
                return
            finite = np.isfinite(error_values) & (error_values > 0.0)
            if finite.any():
                item.setOpts(pen=_series_error_bar_pen(color))
                item.setData(
                    x=x_values[finite],
                    y=y_values[finite],
                    top=error_values[finite],
                    bottom=error_values[finite],
                )
            else:
                _clear_error_bar_item(item)

        if group_key is None or display_group_values is None:
            set_error_bars(
                self._error_bar_item,
                x_values=display_x,
                y_values=display_y,
                error_values=display_y_error,
                color=base_series_color,
            )
            for error_bar_item in self._extra_error_bar_items:
                _clear_error_bar_item(error_bar_item)
            self._group_colors.clear()
            order = np.argsort(display_x, kind="stable")
            if self._show_lines_checkbox.isChecked():
                self._line_item.setPen(pg.mkPen(base_series_color, width=1.5))
                self._line_item.setData(display_x[order], display_y[order])
            else:
                self._line_item.setData([], [])
            for line_item in self._extra_line_items:
                line_item.setData([], [])
            if use_aggregated:
                self._scatter.setData(
                    [
                        {
                            "pos": (float(xi), float(yi)),
                            "data": _DisplayedPointSelection(
                                int(source_index),
                                point_key,
                            ),
                            "brush": pg.mkBrush(_alpha_color("#1f77b4", 40)),
                            "pen": pg.mkPen(_alpha_color("#1f77b4", 60)),
                            "size": 6,
                        }
                        for xi, yi, point_key, source_index in zip(
                            x,
                            y,
                            point_keys,
                            source_indices,
                            strict=True,
                        )
                    ]
                )
                self._summary_scatter.setData(
                    [
                        {
                            "pos": (float(xi), float(yi)),
                            "data": _DisplayedPointSelection(
                                int(source_index),
                                point_key,
                            ),
                            "brush": pg.mkBrush("#1f77b4"),
                            "pen": pg.mkPen(30, 30, 30, 150),
                            "size": 10,
                        }
                        for xi, yi, point_key, source_index in zip(
                            display_x,
                            display_y,
                            display_point_keys,
                            display_source_indices,
                            strict=True,
                        )
                    ]
                )
            else:
                self._summary_scatter.setData([])
                self._scatter.setData(
                    [
                        {
                            "pos": (float(xi), float(yi)),
                            "data": _DisplayedPointSelection(
                                int(source_index),
                                point_key,
                            ),
                        }
                        for xi, yi, point_key, source_index in zip(
                            x,
                            y,
                            point_keys,
                            source_indices,
                            strict=True,
                        )
                    ]
                )
            return _Rendered1DView(
                display_x=display_x,
                display_y=display_y,
                display_point_keys=display_point_keys,
                display_source_indices=display_source_indices,
                display_group_values=display_group_values,
                raw_x=x,
                raw_y=y,
                raw_point_keys=list(point_keys),
                raw_source_indices=list(source_indices),
                raw_group_values=group_values,
                repeats_aggregated=use_aggregated,
            )

        groups = dict[float, list[int]]()
        for index, value in enumerate(display_group_values):
            groups.setdefault(float(value), []).append(index)
        group_items = list(groups.items())
        self._group_colors = {}

        while len(self._extra_line_items) + 1 < len(group_items):
            self._extra_line_items.append(self._plot_item.plot())
        while len(self._extra_error_bar_items) + 1 < len(group_items):
            error_bar_item = pg.ErrorBarItem(beam=0.0)
            self._extra_error_bar_items.append(error_bar_item)
            self._plot_item.addItem(error_bar_item)
        all_line_items = [self._line_item, *self._extra_line_items]
        all_error_bar_items = [self._error_bar_item, *self._extra_error_bar_items]
        for line_item in all_line_items[len(group_items) :]:
            line_item.setData([], [])
        for error_bar_item in all_error_bar_items[len(group_items) :]:
            _clear_error_bar_item(error_bar_item)

        raw_scatter_points = []
        summary_scatter_points = []
        for group_index, (group_value, indices) in enumerate(group_items):
            color = pg.intColor(group_index, hues=max(3, len(group_items)))
            self._group_colors[float(group_value)] = color
            pen = pg.mkPen(color, width=1.8)
            brush = pg.mkBrush(color)
            line_item = all_line_items[group_index]
            error_bar_item = all_error_bar_items[group_index]
            line_item.setPen(pen)
            ordered_indices = np.asarray(indices, dtype=int)[
                np.argsort(display_x[indices], kind="stable")
            ]
            if self._show_lines_checkbox.isChecked():
                line_item.setData(display_x[ordered_indices], display_y[ordered_indices])
            else:
                line_item.setData([], [])
            if display_y_error is not None:
                group_error = np.asarray(display_y_error[indices], dtype=float)
            else:
                group_error = None
            set_error_bars(
                error_bar_item,
                x_values=display_x[indices],
                y_values=display_y[indices],
                error_values=group_error,
                color=color,
            )
            if use_aggregated:
                for point_index in indices:
                    summary_scatter_points.append(
                        {
                            "pos": (
                                float(display_x[point_index]),
                                float(display_y[point_index]),
                            ),
                            "data": _DisplayedPointSelection(
                                int(display_source_indices[point_index]),
                                display_point_keys[point_index],
                            ),
                            "brush": brush,
                            "pen": pg.mkPen(30, 30, 30, 150),
                            "size": 10,
                        }
                    )
            for point_index in (
                indices if not use_aggregated else np.flatnonzero(group_values == group_value)
            ):
                raw_scatter_points.append(
                    {
                        "pos": (
                            float(x[point_index]) if use_aggregated else float(display_x[point_index]),
                            float(y[point_index]) if use_aggregated else float(display_y[point_index]),
                        ),
                        "data": _DisplayedPointSelection(
                            int(source_indices[point_index])
                            if use_aggregated
                            else int(display_source_indices[point_index]),
                            point_keys[point_index]
                            if use_aggregated
                            else display_point_keys[point_index],
                        ),
                        "brush": (
                            pg.mkBrush(_alpha_color(color, 40))
                            if use_aggregated
                            else brush
                        ),
                        "pen": (
                            pg.mkPen(_alpha_color(color, 60))
                            if use_aggregated
                            else pg.mkPen(30, 30, 30, 120)
                        ),
                        "size": 6 if use_aggregated else 9,
                    }
                )
            legend_label = (
                group_label_map.get(float(group_value), _format_readout_value(group_value))
                if group_label_map is not None
                else _format_readout_value(group_value)
            )
            self._legend.addItem(line_item, legend_label)
        self._legend.show()
        self._scatter.setData(raw_scatter_points)
        self._summary_scatter.setData(summary_scatter_points if use_aggregated else [])
        return _Rendered1DView(
            display_x=display_x,
            display_y=display_y,
            display_point_keys=display_point_keys,
            display_source_indices=display_source_indices,
            display_group_values=display_group_values,
            raw_x=x,
            raw_y=y,
            raw_point_keys=list(point_keys),
            raw_source_indices=list(source_indices),
            raw_group_values=group_values,
            repeats_aggregated=use_aggregated,
        )

    def _render_2d_plot(
        self,
        plot_mode: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        point_keys: list[_DisplayPointKey],
        source_indices: list[int],
        z_key: str | None,
        z_label_map: dict[str, str],
    ) -> None:
        """Render either a 2D scatter plot or a 2D image with point overlays."""
        brushes = _color_brushes(z)
        self._line_item.setData([], [])
        for line_item in self._extra_line_items:
            line_item.setData([], [])
        _clear_error_bar_item(self._error_bar_item)
        for error_bar_item in self._extra_error_bar_items:
            _clear_error_bar_item(error_bar_item)
        self._legend.clear()
        self._legend.hide()
        self._summary_scatter.setData([])
        if plot_mode == _PLOT_MODE_2D_IMAGE:
            image = _grid_resampled_image(x, y, z)
            if image is None:
                image = _nearest_resampled_image(x, y, z)
            if image is not None:
                image_data = image.image.copy()
                finite_image = np.isfinite(image_data)
                if finite_image.any():
                    z_min = float(np.min(image_data[finite_image]))
                    z_max = float(np.max(image_data[finite_image]))
                    if z_max == z_min:
                        z_max = z_min + 1.0
                    image_data[~finite_image] = z_min
                    self._image_item.setLookupTable(
                        colormaps.plasma.getLookupTable(0.0, 1.0, 256)
                    )
                    self._image_item.setLevels((z_min, z_max))
                    self._image_item.setImage(image_data, autoLevels=False)
                    self._image_item.setRect(image.rect)
                    self._image_item.show()
                    self._status.setText(
                        f"{self._plot_data.mode_label}; color = {z_label_map.get(z_key, z_key or '')}; {image.mode_label}"
                    )
            self._scatter.setData(
                [
                    {
                        "pos": (float(xi), float(yi)),
                        "data": _DisplayedPointSelection(int(source_index), point_key),
                        "brush": pg.mkBrush(255, 255, 255, 70),
                        "pen": pg.mkPen(255, 255, 255, 160),
                        "size": 6,
                    }
                    for xi, yi, point_key, source_index in zip(
                        x,
                        y,
                        point_keys,
                        source_indices,
                        strict=True,
                    )
                ]
            )
        else:
            self._scatter.setData(
                [
                    {
                        "pos": (float(xi), float(yi)),
                        "data": _DisplayedPointSelection(int(source_index), point_key),
                        "brush": brush,
                        "pen": pg.mkPen(30, 30, 30, 120),
                        "size": 9,
                    }
                    for xi, yi, point_key, source_index, brush in zip(
                        x,
                        y,
                        point_keys,
                        source_indices,
                        brushes,
                        strict=True,
                    )
                ]
            )
            self._status.setText(
                f"{self._plot_data.mode_label}; color = {z_label_map.get(z_key, z_key or '')}"
            )
        self._update_colorbar(z_key, z)

    def _highlight_selected_point(self, plot_mode: str) -> None:
        """Show the selected visible marker, and crosshairs in 2D modes."""
        highlighted = []
        selected_key = self._resolved_selected_display_point_key()
        if selected_key is not None:
            values = self._rendered_point_values.get(selected_key)
            source_index = self._rendered_source_indices.get(selected_key)
            if values is not None and source_index is not None:
                xi, yi, _ = values
                highlighted.append(
                    {"pos": (float(xi), float(yi)), "data": int(source_index)}
                )
                if plot_mode in {_PLOT_MODE_2D_SCATTER, _PLOT_MODE_2D_IMAGE}:
                    self._crosshair_x.setPos(float(xi))
                    self._crosshair_y.setPos(float(yi))
                    self._crosshair_x.show()
                    self._crosshair_y.show()
        self._highlight.setData(highlighted)

    def _render_bo_plot(self) -> None:
        """Render the embedded matplotlib BO dashboard for compatible root sites."""

        self._plot_stack.setCurrentWidget(self._bo_plot_widget)
        self._clear_plot_items()
        status = self._bo_plot_widget.render_site(self._site)
        self._status.setText(status)
        self._current_plot_mode = _PLOT_MODE_BO
        self._current_plot_arrays = None
        self._current_display_point_keys = []
        self._current_group_values = None
        self._current_x_label = "x"
        self._current_y_label = "y"
        self._current_z_label = None
        self._current_group_label = None
        self._selected_readout.setText("selected: not available in BO dashboard")
        self._cursor_readout.setText("cursor: not available in BO dashboard")
        self._fit_readout.setText("fit: BO dashboard")
        self._update_plot_overlay_text()

    def _overlay_fit_summary_from_annotations(
        self,
        *,
        x_key: str,
        y_key: str,
    ) -> str | None:
        for render_spec in _annotation_specs_for_display(
            self._site, self._parent_point_index
        ):
            if render_spec.annotation.get("kind") != "artifact_curve":
                continue
            parameters = render_spec.annotation.get("parameters", {})
            if parameters.get("x_axis") != x_key or parameters.get("y_axis") != y_key:
                continue
            if not self._annotation_xy_slice_matches(parameters, x_key=x_key, y_key=y_key):
                continue
            associated_channels = parameters.get("associated_channels")
            if associated_channels is not None and y_key not in associated_channels:
                continue
            artifact_name = parameters.get("artifact")
            if not artifact_name:
                continue
            artifact = render_spec.artifacts.get(artifact_name)
            if not isinstance(artifact, dict) or artifact.get("kind") != "model_fit":
                continue
            model_name = str(artifact.get("model_name", artifact.get("model_id", "fit")))
            summary = artifact_summary(artifact)
            return _wrap_overlay_text(f"{model_name}: {summary}")
        return None

    def _update_plot_overlay_text(
        self,
        *,
        x_key: str | None = None,
        y_key: str | None = None,
    ) -> None:
        lines = []
        run_id_text = _run_id_text_for_site(self._site)
        if run_id_text:
            lines.append(run_id_text)

        fit_line = None
        if self._fit_active and self._fit_result is not None:
            fit_line = _wrap_overlay_text(
                f"{self._fit_result.model_id}: {self._fit_result.summary()}"
            )
        elif x_key is not None and y_key is not None:
            fit_line = self._overlay_fit_summary_from_annotations(
                x_key=x_key,
                y_key=y_key,
            )
        if fit_line:
            lines.append(fit_line)

        self._run_id_overlay.set_display_text("\n".join(lines) if lines else None)

    def _prepare_plot_arrays(
        self,
        *,
        plot_mode: str,
        x_key: str,
        y_key: str,
        group_key: str | None,
        y_index_tokens: tuple[str, ...] | None = None,
    ) -> _PreparedPlotArrays | None:
        x_values = self._point_stream_values(x_key)
        y_values = self._point_stream_values(y_key)
        group_axis_values = (
            self._point_stream_values(group_key)
            if group_key not in {None, _ARRAY_SERIES_GROUP_KEY}
            else []
        )
        y_error_key = _error_bar_channel_key(self._site, y_key)
        y_error_values = (
            self._point_stream_values(y_error_key) if y_error_key is not None else []
        )
        x_schema = self._site.channels.get(x_key)
        y_schema = self._site.channels.get(y_key)
        y_error_schema = (
            self._site.channels.get(y_error_key) if y_error_key is not None else None
        )

        count = min(
            len(x_values),
            len(y_values),
            len(self._plot_data.source_indices),
        )
        if group_key not in {None, _ARRAY_SERIES_GROUP_KEY}:
            count = min(count, len(group_axis_values))
        if y_error_key is not None:
            count = min(count, len(y_error_values))
        if count == 0:
            return None

        source_indices = self._plot_data.source_indices[:count]
        x_series = _extract_axis_series_values(
            list(x_values[:count]),
            schema=x_schema if isinstance(x_schema, dict) else None,
            index_tokens=self._selected_x_index_tokens(),
        )
        y_series = _extract_axis_series_values(
            list(y_values[:count]),
            schema=y_schema if isinstance(y_schema, dict) else None,
            index_tokens=y_index_tokens if y_index_tokens is not None else self._selected_y_index_tokens(),
        )
        y_error_series = (
            _extract_axis_series_values(
                list(y_error_values[:count]),
                schema=y_error_schema if isinstance(y_error_schema, dict) else None,
                index_tokens=y_index_tokens if y_index_tokens is not None else self._selected_y_index_tokens(),
            )
            if y_error_key is not None
            else None
        )
        z_key = None
        z = None
        group_values = None
        group_label_map = None
        group_label_name = None
        array_series_label_map = None
        array_series_label_name = None

        if plot_mode != _PLOT_MODE_1D and (
            len(x_series.series_values) > 1 or len(y_series.series_values) > 1
        ):
            raise ValueError("Array ranges are currently only supported in 1D mode")

        num_series = max(len(x_series.series_values), len(y_series.series_values))
        if len(x_series.series_values) not in {1, num_series}:
            raise ValueError("X array selection produced an unsupported number of series")
        if len(y_series.series_values) not in {1, num_series}:
            raise ValueError("Y array selection produced an unsupported number of series")
        if y_error_series is not None and len(y_error_series.series_values) not in {
            1,
            num_series,
        }:
            raise ValueError("Y error selection does not match the plotted series")

        if num_series == 1:
            x = x_series.series_values[0]
            y = y_series.series_values[0]
            y_error = (
                y_error_series.series_values[0] if y_error_series is not None else None
            )
            if group_key is not None:
                group_values = np.asarray(group_axis_values[:count], dtype=float)
        else:
            active_labels = (
                x_series.series_labels
                if len(x_series.series_values) > 1
                else y_series.series_labels
            )
            active_label_name = (
                _ranged_dim_name_for_selection(
                    x_schema if isinstance(x_schema, dict) else None,
                    self._selected_x_index_tokens(),
                )
                if len(x_series.series_values) > 1
                else _ranged_dim_name_for_selection(
                    y_schema if isinstance(y_schema, dict) else None,
                    self._selected_y_index_tokens(),
                )
            ) or "series"

            flat_x = []
            flat_y = []
            flat_y_error = [] if y_error_series is not None else None
            flat_source_indices = []
            flat_group_values = [] if group_key is not None else None
            for series_index in range(num_series):
                x_part = (
                    x_series.series_values[series_index]
                    if len(x_series.series_values) > 1
                    else x_series.series_values[0]
                )
                y_part = (
                    y_series.series_values[series_index]
                    if len(y_series.series_values) > 1
                    else y_series.series_values[0]
                )
                series_count = min(len(x_part), len(y_part), len(source_indices))
                if series_count == 0:
                    continue
                flat_x.append(np.asarray(x_part[:series_count], dtype=float))
                flat_y.append(np.asarray(y_part[:series_count], dtype=float))
                if flat_y_error is not None and y_error_series is not None:
                    error_part = (
                        y_error_series.series_values[series_index]
                        if len(y_error_series.series_values) > 1
                        else y_error_series.series_values[0]
                    )
                    flat_y_error.append(np.asarray(error_part[:series_count], dtype=float))
                flat_source_indices.extend(source_indices[:series_count])
                if flat_group_values is not None:
                    if group_key == _ARRAY_SERIES_GROUP_KEY:
                        flat_group_values.extend([float(series_index)] * series_count)
                    else:
                        flat_group_values.extend(
                            np.asarray(group_axis_values[:series_count], dtype=float).tolist()
                        )

            if not flat_x or not flat_y:
                return None
            x = np.concatenate(flat_x)
            y = np.concatenate(flat_y)
            y_error = (
                np.concatenate(flat_y_error)
                if flat_y_error is not None
                else None
            )
            source_indices = flat_source_indices
            group_values = (
                np.asarray(flat_group_values, dtype=float)
                if flat_group_values is not None
                else None
            )
            array_series_label_map = {
                float(index): label
                for index, label in enumerate(active_labels or [])
            }
            array_series_label_name = active_label_name
            if group_key == _ARRAY_SERIES_GROUP_KEY:
                group_label_map = array_series_label_map
                group_label_name = array_series_label_name

        if plot_mode in {_PLOT_MODE_2D_SCATTER, _PLOT_MODE_2D_IMAGE}:
            z_key = self._selected_z_key()
            z_values = self._point_stream_values(z_key) if z_key is not None else []
            count = min(len(x), len(y), len(source_indices), len(z_values))
            if count == 0:
                return None
            x = x[:count]
            y = y[:count]
            source_indices = source_indices[:count]
            z_series = _extract_axis_series_values(
                list(z_values[:count]),
                schema=(
                    self._site.channels.get(z_key)
                    if z_key is not None and isinstance(self._site.channels.get(z_key), dict)
                    else None
                ),
                index_tokens=self._selected_z_index_tokens(),
            )
            if len(z_series.series_values) != 1:
                raise ValueError("Z array ranges are currently only supported in 1D mode")
            z = np.asarray(z_series.series_values[0], dtype=float)

            finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if not finite_mask.any():
                return None
            x = x[finite_mask]
            y = y[finite_mask]
            z = z[finite_mask]
            source_indices = [
                src
                for src, keep in zip(source_indices, finite_mask, strict=True)
                if keep
            ]
        else:
            finite_mask = np.isfinite(x) & np.isfinite(y)
            if group_values is not None:
                finite_mask &= np.isfinite(group_values)
            if not finite_mask.any():
                return None
            x = x[finite_mask]
            y = y[finite_mask]
            if y_error is not None:
                y_error = y_error[finite_mask]
            if group_values is not None:
                group_values = group_values[finite_mask]
            source_indices = [
                src
                for src, keep in zip(source_indices, finite_mask, strict=True)
                if keep
            ]
        point_keys = [("point", index) for index in range(len(source_indices))]

        return _PreparedPlotArrays(
            plot_mode=plot_mode,
            x_key=x_key,
            y_key=y_key,
            group_key=group_key,
            z_key=z_key,
            x=x,
            y=y,
            y_error=y_error,
            point_keys=point_keys,
            source_indices=source_indices,
            group_values=group_values,
            group_label_map=group_label_map,
            group_label_name=group_label_name,
            array_series_label_map=array_series_label_map,
            array_series_label_name=array_series_label_name,
            z=z,
        )

    def _render_selected_plot_mode(
        self,
        prepared: _PreparedPlotArrays,
        *,
        z_label_map: dict[str, str],
    ) -> _Rendered1DView | None:
        if prepared.plot_mode == _PLOT_MODE_1D:
            rendered_1d = self._render_1d_plot(
                prepared.x,
                prepared.y,
                prepared.y_error,
                prepared.point_keys,
                prepared.source_indices,
                prepared.group_key,
                prepared.group_values,
                prepared.group_label_map,
            )
            status = self._plot_data.mode_label
            if prepared.group_key == _ARRAY_SERIES_GROUP_KEY:
                status += (
                    "; split by "
                    f"{prepared.group_label_name or 'series'}"
                )
            elif prepared.group_key is not None:
                group_label_map = _choice_label_map(
                    _group_by_choices(self._site, prepared.x_key)
                )
                status += (
                    "; grouped by "
                    f"{group_label_map.get(prepared.group_key, prepared.group_key)}"
                )
            elif prepared.array_series_label_map is not None:
                status += (
                    "; repeated "
                    f"{prepared.array_series_label_name or 'series'} values"
                )
            if rendered_1d.repeats_aggregated:
                repeat_mode_labels = dict(_REPEAT_COMBINE_CHOICES)
                status += (
                    "; repeated x: "
                    f"{repeat_mode_labels.get(self._selected_repeat_combine_mode(), 'combined')}"
                )
            self._status.setText(status)
            return rendered_1d

        assert prepared.z is not None
        self._render_2d_plot(
            prepared.plot_mode,
            prepared.x,
            prepared.y,
            prepared.z,
            prepared.point_keys,
            prepared.source_indices,
            prepared.z_key,
            z_label_map,
        )
        return None

    def _update_rendered_plot_state(
        self,
        prepared: _PreparedPlotArrays,
        rendered_1d: _Rendered1DView | None,
        *,
        x_label_map: dict[str, str],
        y_label_map: dict[str, str],
        z_label_map: dict[str, str],
    ) -> None:
        annotation_x = rendered_1d.display_x if rendered_1d is not None else prepared.x
        self._render_annotations(
            prepared.plot_mode,
            x_key=prepared.x_key,
            y_key=prepared.y_key,
            x=annotation_x,
        )
        artifact_source_kind, display_artifacts = _annotation_artifact_map_for_display(
            self._site, self._parent_point_index
        )
        self._artifact_readout.setText(
            _format_artifact_readout(artifact_source_kind, display_artifacts)
        )

        x_label = x_label_map.get(
            prepared.x_key,
            prepared.x_key
            if prepared.x_key != _POINT_INDEX_KEY
            else "point_index",
        )
        x_label = _axis_label_with_indices(
            x_label,
            site=self._site,
            key=prepared.x_key,
            index_tokens=self._selected_x_index_tokens(),
        )
        y_label = _axis_label_with_indices(
            y_label_map.get(prepared.y_key, prepared.y_key),
            site=self._site,
            key=prepared.y_key,
            index_tokens=self._selected_y_index_tokens(),
        )
        z_label = (
            _axis_label_with_indices(
                z_label_map.get(prepared.z_key, prepared.z_key),
                site=self._site,
                key=prepared.z_key,
                index_tokens=self._selected_z_index_tokens(),
            )
            if prepared.z_key is not None
            else None
        )
        self._plot_item.setLabel("bottom", x_label)
        self._plot_item.setLabel("left", y_label)
        self._current_x_label = x_label
        self._current_y_label = y_label
        self._current_z_label = z_label
        self._current_plot_mode = prepared.plot_mode

        if rendered_1d is not None:
            self._current_plot_arrays = (
                rendered_1d.display_x.copy(),
                rendered_1d.display_y.copy(),
            )
            self._current_display_point_keys = list(rendered_1d.display_point_keys)
            self._current_group_values = (
                None
                if rendered_1d.display_group_values is None
                else np.asarray(rendered_1d.display_group_values, dtype=float).copy()
            )
            rendered_entries = [
                *zip(
                    rendered_1d.raw_point_keys,
                    rendered_1d.raw_source_indices,
                    rendered_1d.raw_x,
                    rendered_1d.raw_y,
                    [None] * len(rendered_1d.raw_point_keys),
                    [None] * len(rendered_1d.raw_point_keys)
                    if rendered_1d.raw_group_values is None
                    else rendered_1d.raw_group_values,
                    strict=True,
                ),
            ]
            if rendered_1d.repeats_aggregated:
                rendered_entries.extend(
                    zip(
                        rendered_1d.display_point_keys,
                        rendered_1d.display_source_indices,
                        rendered_1d.display_x,
                        rendered_1d.display_y,
                        [None] * len(rendered_1d.display_point_keys),
                        [None] * len(rendered_1d.display_point_keys)
                        if rendered_1d.display_group_values is None
                        else rendered_1d.display_group_values,
                        strict=True,
                    )
                )
        else:
            self._current_plot_arrays = (prepared.x.copy(), prepared.y.copy())
            self._current_display_point_keys = list(prepared.point_keys)
            self._current_group_values = (
                None
                if prepared.group_values is None
                else np.asarray(prepared.group_values, dtype=float).copy()
            )
            rendered_entries = list(
                zip(
                    prepared.point_keys,
                    prepared.source_indices,
                    prepared.x,
                    prepared.y,
                    [None] * len(prepared.point_keys)
                    if prepared.z is None
                    else prepared.z,
                    [None] * len(prepared.point_keys)
                    if prepared.group_values is None
                    else prepared.group_values,
                    strict=True,
                )
            )

        if prepared.group_label_map is not None:
            self._current_group_label = prepared.group_label_name or "series"
            self._rendered_group_label_map = dict(prepared.group_label_map)
        else:
            group_label_map = _choice_label_map(
                _group_by_choices(self._site, prepared.x_key)
            )
            self._current_group_label = (
                group_label_map.get(prepared.group_key, prepared.group_key)
                if prepared.group_key is not None
                else None
            )
            self._rendered_group_label_map = {}
        self._rendered_point_values = {
            point_key: (
                float(xi),
                float(yi),
                None if zi is None else float(zi),
            )
            for point_key, _, xi, yi, zi, _ in rendered_entries
        }
        self._rendered_source_indices = {
            point_key: int(source_index)
            for point_key, source_index, _, _, _, _ in rendered_entries
        }
        self._rendered_group_values = {
            point_key: float(group_value)
            for point_key, _, _, _, _, group_value in rendered_entries
            if group_value is not None
        }
        self._highlight_selected_point(prepared.plot_mode)
        self._update_selected_readout(prepared.plot_mode)
        self._update_cursor_readout()
        self._update_plot_overlay_text(
            x_key=prepared.x_key,
            y_key=prepared.y_key,
        )

    def _render_plot(self) -> None:
        """Render the current site slice using the selected plot mode and axes."""
        if self._fit_active:
            self._clear_fit()
        self._clear_annotation_items()
        self._status.setText(self._plot_data.mode_label)

        x_choices = _default_x_choices(self._site)
        y_choices = _default_y_choices(self._site)
        z_choices = _default_z_choices(self._site)
        x_label_map = _choice_label_map(x_choices)
        y_label_map = _choice_label_map(y_choices)
        z_label_map = _choice_label_map(z_choices)
        plot_mode = self._selected_plot_mode()
        if plot_mode == _PLOT_MODE_BO:
            self._render_bo_plot()
            return
        self._plot_stack.setCurrentWidget(self._pyqtgraph_panel)
        x_key = self._selected_x_key()
        if x_key is None:
            self._clear_plot_items()
            return

        self._image_item.hide()
        self._crosshair_x.hide()
        self._crosshair_y.hide()
        if plot_mode == _PLOT_MODE_1D:
            if not self._render_multi_series_1d(
                x_key=x_key,
                x_label_map=x_label_map,
                y_label_map=y_label_map,
            ):
                self._clear_plot_items()
                return
            first_series = next(
                (series for series in self._current_series_states() if series.y_key is not None),
                None,
            )
            if first_series is not None and self._current_plot_arrays is not None:
                self._render_annotations(
                    plot_mode,
                    x_key=x_key,
                    y_key=first_series.y_key,
                    x=self._current_plot_arrays[0],
                )
            artifact_source_kind, display_artifacts = _annotation_artifact_map_for_display(
                self._site, self._parent_point_index
            )
            self._artifact_readout.setText(
                _format_artifact_readout(artifact_source_kind, display_artifacts)
            )
            self._update_plot_overlay_text(x_key=x_key, y_key=self._selected_y_key())
            return

        y_key = self._selected_y_key()
        group_key = self._selected_group_key()
        if y_key is None:
            self._clear_plot_items()
            return
        try:
            prepared = self._prepare_plot_arrays(
                plot_mode=plot_mode,
                x_key=x_key,
                y_key=y_key,
                group_key=group_key,
            )
        except ValueError as exc:
            self._clear_plot_items()
            self._status.setText(str(exc))
            return
        if prepared is None:
            self._clear_plot_items()
            return

        rendered_1d = self._render_selected_plot_mode(
            prepared,
            z_label_map=z_label_map,
        )
        self._update_rendered_plot_state(
            prepared,
            rendered_1d,
            x_label_map=x_label_map,
            y_label_map=y_label_map,
            z_label_map=z_label_map,
        )


class RuntimePlotViewer(QtWidgets.QWidget):
    """Prepared-runtime plot viewer for live dataset updates or loaded snapshots.

    The viewer deliberately keeps very little model logic of its own. It consumes the
    common :class:`ScanSiteSnapshot` site tree, chooses which sites should appear as
    a recursive set of columns, and lets each column widget render itself from that
    site plus a small amount of UI selection state. Live applets feed it through
    :meth:`data_changed`; offline HDF5 tools feed it through :meth:`set_snapshot`.
    """

    def __init__(self, prefix: str | None = None):
        super().__init__()
        self._prefix = prefix
        self._snapshot = None
        self._snapshot_status_text = "Live prepared-runtime scan"
        self._paused = False
        self._pending_values: dict[str, Any] | None = None
        self._site_ui_state = dict[tuple[str, ...], _SiteUiState]()
        self._columns: list[_SiteColumnWidget] = []

        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(8)
        outer.addLayout(top_bar)

        self._status = QtWidgets.QLabel("Waiting for prepared-runtime scan data…")
        top_bar.addWidget(self._status, 1)

        self._pause_checkbox = QtWidgets.QCheckBox("pause live updates")
        self._pause_checkbox.toggled.connect(self._set_paused)
        self._pause_checkbox.setVisible(prefix is not None)
        top_bar.addWidget(self._pause_checkbox)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        outer.addWidget(self._scroll, 1)

        self._columns_widget = QtWidgets.QWidget()
        self._columns_layout = QtWidgets.QHBoxLayout()
        self._columns_layout.setContentsMargins(6, 6, 6, 6)
        self._columns_layout.setSpacing(8)
        self._columns_widget.setLayout(self._columns_layout)
        self._scroll.setWidget(self._columns_widget)

    def _ui_state_for(self, site_path: tuple[str, ...]) -> _SiteUiState:
        state = self._site_ui_state.get(site_path)
        if state is None:
            state = _SiteUiState()
            self._site_ui_state[site_path] = state
        return state

    def data_changed(
        self,
        values: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        persist: dict[str, bool] | None = None,
        mods=None,
    ) -> None:
        """Rebuild the visible site tree from the latest live dataset snapshot."""
        del metadata, persist, mods
        if self._prefix is None:
            raise RuntimeError(
                "RuntimePlotViewer was created for an offline snapshot; construct it "
                "with a dataset prefix to use live data_changed() updates"
            )
        if self._paused:
            self._pending_values = dict(values)
            self._update_status_text()
            return
        self._apply_live_values(values)

    def set_snapshot(
        self,
        snapshot: ScanSiteSnapshot,
        *,
        status_text: str | None = None,
    ) -> None:
        """Render one already-loaded prepared-runtime snapshot.

        This is the offline counterpart to :meth:`data_changed`: callers such as an
        HDF5 viewer can read a results file into a :class:`ScanSiteSnapshot` and use
        the normal runtime viewer UI without going through the live dataset adapter.
        """

        self._snapshot = snapshot
        self._snapshot_status_text = status_text or "Loaded prepared-runtime snapshot"
        self._pending_values = None
        self._paused = False
        self._pause_checkbox.blockSignals(True)
        self._pause_checkbox.setChecked(False)
        self._pause_checkbox.blockSignals(False)
        self._pause_checkbox.setVisible(False)
        self._rebuild_columns()

    def _apply_live_values(self, values: dict[str, Any]) -> None:
        """Apply one live dataset snapshot to the visible viewer state."""
        self.setUpdatesEnabled(False)
        try:
            self._snapshot_status_text = "Live prepared-runtime scan"
            self._pause_checkbox.setVisible(True)
            self._snapshot = snapshot_from_live_values(self._prefix, values)
            self._rebuild_columns()
        finally:
            self.setUpdatesEnabled(True)

    def _set_paused(self, paused: bool) -> None:
        """Freeze or resume the visible live snapshot.

        While paused, the visible site tree stays fixed and only the latest incoming
        dataset state is remembered. Resuming applies that latest buffered state in one
        jump.
        """
        self._paused = bool(paused)
        if not self._paused and self._pending_values is not None:
            pending_values = self._pending_values
            self._pending_values = None
            self._apply_live_values(pending_values)
            return
        self._update_status_text()

    def _update_status_text(self, base_text: str | None = None) -> None:
        """Refresh the top status label, including paused/pending state."""
        if base_text is None:
            current_text = self._status.text().strip()
            if current_text.startswith("Paused "):
                base_text = current_text.removeprefix("Paused ").split(" (", 1)[0]
            else:
                base_text = current_text or "Waiting for prepared-runtime scan data…"

        if self._paused:
            if self._pending_values is not None:
                self._status.setText(f"Paused {base_text} (updates buffered)")
            else:
                self._status.setText(f"Paused {base_text}")
            return

        self._status.setText(base_text)

    def _clear_descendant_navigation(self, site_path: tuple[str, ...]) -> None:
        """Drop only navigation state below one site path.

        Plot preferences such as x/y/z choice, plot mode, and group-by are keyed by
        site path and should survive content changes for that same site. What becomes
        stale on parent-point or child-site changes is the descendant navigation
        itself: selected descendant points and chosen descendant child sites.
        """

        for path, state in self._site_ui_state.items():
            if _descends_from(path, site_path):
                state.selected_point_index = None
                state.selected_display_point_key = None
                state.selected_child_path = None

    def _choose_child_site(
        self, parent_path: tuple[str, ...], child_sites: list[ScanSiteData]
    ) -> ScanSiteData:
        """Return the chosen child site for one parent path, defaulting to the first."""
        state = self._ui_state_for(parent_path)
        selected_path = state.selected_child_path
        for site in child_sites:
            if site.path == selected_path:
                return site
        chosen = child_sites[0]
        state.selected_child_path = chosen.path
        return chosen

    def _rebuild_columns(self) -> None:
        """Project the current snapshot into the recursive column UI."""
        snapshot = self._snapshot
        if snapshot is None or not snapshot.sites:
            self._update_status_text("Waiting for prepared-runtime scan data…")
            self._sync_column_widgets([])
            return
        if () not in snapshot.sites:
            self._update_status_text("Root prepared-runtime site has not appeared yet")
            self._sync_column_widgets([])
            return

        root_site = snapshot.get_site(())
        title = root_site.fragment_fqn
        source_id = root_site.metadata.get("site.source_id")
        if source_id:
            title += f" [{source_id}]"
        self.setWindowTitle(title)
        self._update_status_text(self._snapshot_status_text)

        column_specs = [
            _ColumnSpec(
                site=root_site,
                child_site_options=[],
                selected_child_path=None,
                parent_point_index=None,
            )
        ]

        parent_path = ()
        while True:
            child_sites = snapshot.child_sites(parent_path)
            if not child_sites:
                break
            child_site = self._choose_child_site(parent_path, child_sites)
            column_specs.append(
                _ColumnSpec(
                    site=child_site,
                    child_site_options=child_sites,
                    selected_child_path=child_site.path,
                    parent_point_index=self._ui_state_for(parent_path).selected_point_index,
                )
            )
            parent_path = child_site.path

        self._sync_column_widgets(column_specs)

    def _sync_column_widgets(self, column_specs: list[_ColumnSpec]) -> None:
        """Create, remove, or update column widgets to match the desired site list."""
        while len(self._columns) > len(column_specs):
            column = self._columns.pop()
            self._columns_layout.removeWidget(column)
            column.deleteLater()

        while len(self._columns) < len(column_specs):
            spec = column_specs[len(self._columns)]
            column = _SiteColumnWidget(
                site=spec.site,
                child_site_options=spec.child_site_options,
                selected_child_path=spec.selected_child_path,
                parent_point_index=spec.parent_point_index,
                selected_point_index=None,
                selected_display_point_key=None,
                selected_plot_mode=None,
                selected_x_key=None,
                selected_x_index_tokens=None,
                selected_y_key=None,
                selected_y_index_tokens=None,
                selected_z_key=None,
                selected_z_index_tokens=None,
                selected_group_key=None,
                show_lines=False,
                selected_repeat_combine_mode=None,
                selected_series_states=None,
            )
            column.point_selected.connect(self._on_point_selected)
            column.child_site_changed.connect(self._on_child_site_changed)
            column.plot_mode_changed.connect(self._on_plot_mode_changed)
            column.x_key_changed.connect(self._on_x_key_changed)
            column.x_index_tokens_changed.connect(self._on_x_index_tokens_changed)
            column.y_key_changed.connect(self._on_y_key_changed)
            column.y_index_tokens_changed.connect(self._on_y_index_tokens_changed)
            column.z_key_changed.connect(self._on_z_key_changed)
            column.z_index_tokens_changed.connect(self._on_z_index_tokens_changed)
            column.group_key_changed.connect(self._on_group_key_changed)
            column.show_lines_changed.connect(self._on_show_lines_changed)
            column.repeat_combine_mode_changed.connect(
                self._on_repeat_combine_mode_changed
            )
            column.series_states_changed.connect(self._on_series_states_changed)
            self._columns.append(column)
            self._columns_layout.addWidget(column)

        for column, spec in zip(self._columns, column_specs, strict=True):
            site = spec.site
            state = self._ui_state_for(site.path)
            column.update_state(
                site=site,
                child_site_options=spec.child_site_options,
                selected_child_path=spec.selected_child_path,
                parent_point_index=spec.parent_point_index,
                selected_point_index=state.selected_point_index,
                selected_display_point_key=state.selected_display_point_key,
                selected_plot_mode=state.plot_mode,
                selected_x_key=state.x_key,
                selected_x_index_tokens=state.x_index_tokens,
                selected_y_key=state.y_key,
                selected_y_index_tokens=state.y_index_tokens,
                selected_z_key=state.z_key,
                selected_z_index_tokens=state.z_index_tokens,
                selected_group_key=state.group_key,
                show_lines=state.show_lines,
                selected_repeat_combine_mode=state.repeat_combine_mode,
                selected_series_states=state.series_states,
            )
        self._columns_layout.invalidate()
        self._columns_layout.activate()
        self._columns_widget.updateGeometry()
        self._columns_widget.adjustSize()
        self._scroll.widget().updateGeometry()
        self._scroll.viewport().update()

    def _on_point_selected(
        self, site_path: tuple[str, ...], selection: _DisplayedPointSelection | int | None
    ) -> None:
        state = self._ui_state_for(site_path)
        if selection is None:
            state.selected_point_index = None
            state.selected_display_point_key = None
        elif isinstance(selection, _DisplayedPointSelection):
            state.selected_point_index = int(selection.source_index)
            state.selected_display_point_key = selection.display_key
        else:
            state.selected_point_index = int(selection)
            state.selected_display_point_key = None
        self._clear_descendant_navigation(site_path)
        self._rebuild_columns()

    def _on_child_site_changed(
        self, parent_path: tuple[str, ...], child_path: tuple[str, ...]
    ) -> None:
        self._ui_state_for(parent_path).selected_child_path = child_path
        self._clear_descendant_navigation(parent_path)
        self._rebuild_columns()

    def _on_x_key_changed(self, site_path: tuple[str, ...], key: str) -> None:
        self._ui_state_for(site_path).x_key = key
        self._rebuild_columns()

    def _on_x_index_tokens_changed(
        self, site_path: tuple[str, ...], tokens: tuple[str, ...] | None
    ) -> None:
        self._ui_state_for(site_path).x_index_tokens = tokens
        self._rebuild_columns()

    def _on_y_key_changed(self, site_path: tuple[str, ...], key: str) -> None:
        state = self._ui_state_for(site_path)
        state.y_key = key
        series_states = list(state.series_states or (_SeriesUiState(),))
        series_states[0] = _SeriesUiState(
            y_key=key,
            y_index_tokens=series_states[0].y_index_tokens,
            group_key=series_states[0].group_key,
            repeat_combine_mode=series_states[0].repeat_combine_mode,
        )
        state.series_states = tuple(series_states)
        self._rebuild_columns()

    def _on_y_index_tokens_changed(
        self, site_path: tuple[str, ...], tokens: tuple[str, ...] | None
    ) -> None:
        state = self._ui_state_for(site_path)
        state.y_index_tokens = tokens
        series_states = list(state.series_states or (_SeriesUiState(),))
        series_states[0] = _SeriesUiState(
            y_key=series_states[0].y_key,
            y_index_tokens=tokens,
            group_key=series_states[0].group_key,
            repeat_combine_mode=series_states[0].repeat_combine_mode,
        )
        state.series_states = tuple(series_states)
        self._rebuild_columns()

    def _on_plot_mode_changed(self, site_path: tuple[str, ...], mode: str) -> None:
        self._ui_state_for(site_path).plot_mode = mode
        self._rebuild_columns()

    def _on_z_key_changed(self, site_path: tuple[str, ...], key: str) -> None:
        self._ui_state_for(site_path).z_key = key
        self._rebuild_columns()

    def _on_z_index_tokens_changed(
        self, site_path: tuple[str, ...], tokens: tuple[str, ...] | None
    ) -> None:
        self._ui_state_for(site_path).z_index_tokens = tokens
        self._rebuild_columns()

    def _on_group_key_changed(self, site_path: tuple[str, ...], key: str) -> None:
        state = self._ui_state_for(site_path)
        state.group_key = key
        series_states = list(state.series_states or (_SeriesUiState(),))
        series_states[0] = _SeriesUiState(
            y_key=series_states[0].y_key,
            y_index_tokens=series_states[0].y_index_tokens,
            group_key=None if key in {None, _NO_GROUP_KEY} else key,
            repeat_combine_mode=series_states[0].repeat_combine_mode,
        )
        state.series_states = tuple(series_states)
        self._rebuild_columns()

    def _on_show_lines_changed(
        self, site_path: tuple[str, ...], show_lines: bool
    ) -> None:
        self._ui_state_for(site_path).show_lines = show_lines
        self._rebuild_columns()

    def _on_repeat_combine_mode_changed(
        self, site_path: tuple[str, ...], mode: str
    ) -> None:
        state = self._ui_state_for(site_path)
        state.repeat_combine_mode = mode
        series_states = list(state.series_states or (_SeriesUiState(),))
        series_states[0] = _SeriesUiState(
            y_key=series_states[0].y_key,
            y_index_tokens=series_states[0].y_index_tokens,
            group_key=series_states[0].group_key,
            repeat_combine_mode=mode,
        )
        state.series_states = tuple(series_states)
        self._rebuild_columns()

    def _on_series_states_changed(
        self,
        site_path: tuple[str, ...],
        states: tuple[_SeriesUiState, ...],
    ) -> None:
        ui_state = self._ui_state_for(site_path)
        ui_state.series_states = tuple(states)
        if states:
            ui_state.y_key = states[0].y_key
            ui_state.y_index_tokens = states[0].y_index_tokens
            ui_state.group_key = states[0].group_key
            ui_state.repeat_combine_mode = states[0].repeat_combine_mode
        self._rebuild_columns()

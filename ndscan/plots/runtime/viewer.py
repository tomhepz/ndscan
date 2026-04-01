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

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg

from ..._qt import QtCore, QtGui, QtWidgets
from .. import colormaps
from ...results.scan_site_reader import HostRuntimeSiteData, HostRuntimeSiteSegment
from .fitting import BuiltinFitBackend, FitBackend, FitRequest, FitResult
from .live import snapshot_from_live_values

_POINT_INDEX_KEY = "__point_index__"
_NO_GROUP_KEY = "__no_group__"
_NO_FIT_MODEL_KEY = "__no_fit_model__"
_FIT_TARGET_VISIBLE = "visible"
_FIT_TARGET_SELECTED_GROUP = "selected_group"
_PLOT_MODE_1D = "1d"
_PLOT_MODE_2D_SCATTER = "2d_scatter"
_PLOT_MODE_2D_IMAGE = "2d_image"
_PLOT_MODE_CHOICES = (
    (_PLOT_MODE_1D, "1D"),
    (_PLOT_MODE_2D_SCATTER, "2D scatter"),
    (_PLOT_MODE_2D_IMAGE, "2D image"),
)

_COLORBAR_PIXMAP_CACHE: QtGui.QPixmap | None = None


@dataclass(frozen=True)
class _SitePlotData:
    """Point data slice currently shown for one site column."""

    source_indices: list[int]
    point_data: dict[str, list[Any]]
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

    site: HostRuntimeSiteData
    child_site_options: list[HostRuntimeSiteData]
    selected_child_path: tuple[str, ...] | None
    parent_point_index: int | None


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


def _x_axis_choices(site: HostRuntimeSiteData) -> list[tuple[str, str]]:
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

    def add_choice(key: str, label: str) -> None:
        if key in seen:
            return
        values = site.point_data.get(key)
        if values is None or not _is_numeric_point_stream(values):
            return
        choices.append((key, label))
        seen.add(key)

    for key, schema in site.pseudoparams.items():
        add_choice(key, _pseudoparam_choice_label(key, schema))

    for key, schema in site.parameters.items():
        if schema.get("is_scanned", False):
            add_choice(key, _parameter_choice_label(key, schema))

    for key, schema in site.parameters.items():
        if not schema.get("is_scanned", False):
            add_choice(key, _parameter_choice_label(key, schema))

    for key, schema in site.channels.items():
        add_choice(key, _channel_choice_label(key, schema))

    for key in sorted(site.point_data.keys()):
        add_choice(key, _generic_point_stream_choice_label(key))

    choices.append((_POINT_INDEX_KEY, _machine_and_human_label(_POINT_INDEX_KEY, "point_index")))
    return _unique_choice_labels(choices)


def _default_x_choices(site: HostRuntimeSiteData) -> list[tuple[str, str]]:
    """Return the default x-axis choices for a site column."""
    return _x_axis_choices(site)


def _default_y_choices(site: HostRuntimeSiteData) -> list[tuple[str, str]]:
    """Return the default y-axis choices for a site column."""
    return _x_axis_choices(site)


def _default_z_choices(site: HostRuntimeSiteData) -> list[tuple[str, str]]:
    """Return the default z-axis choices for a site column."""
    return _x_axis_choices(site)


def _group_by_choices(
    site: HostRuntimeSiteData, selected_x_key: str | None
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
        values = site.point_data.get(key)
        if values is None or not _is_numeric_point_stream(values):
            continue
        choices.append((key, _parameter_choice_label(key, schema)))
    return _unique_choice_labels(choices)


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


def _choice_label_map(choices: list[tuple[str, str]]) -> dict[str, str]:
    return {key: label for key, label in choices}


def _choice_keys(choices: list[tuple[str, str]]) -> list[str]:
    return [key for key, _ in choices]


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


def _channel_choice_keys(site: HostRuntimeSiteData) -> list[str]:
    return [
        key
        for key in site.channels.keys()
        if key in site.point_data and _is_numeric_point_stream(site.point_data[key])
    ]


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
    site: HostRuntimeSiteData,
) -> HostRuntimeSiteSegment | None:
    """Return the currently active child segment, or the last finished one."""
    segments = site.segments()
    if not segments:
        return None
    current_index = int(site.metadata.get("state.current_segment", -1))
    if 0 <= current_index < len(segments):
        return segments[current_index]
    return segments[-1]


def _concat_slices(
    site: HostRuntimeSiteData, segments: list[HostRuntimeSiteSegment], mode_label: str
) -> _SitePlotData:
    """Merge one or more segment slices into the point data for a displayed column."""
    source_indices = []
    merged = dict[str, list[Any]]()
    for segment in segments:
        source_indices.extend(range(segment.start_index, segment.stop_index))
        sliced = site.slice_point_data(segment.start_index, segment.stop_index)
        for key, values in sliced.items():
            merged.setdefault(key, []).extend(values)
    return _SitePlotData(source_indices=source_indices, point_data=merged, mode_label=mode_label)


def _site_plot_data(
    site: HostRuntimeSiteData, parent_point_index: int | None
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
            point_data={key: list(values) for key, values in site.point_data.items()},
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
    y_key_changed = QtCore.pyqtSignal(object, str)
    z_key_changed = QtCore.pyqtSignal(object, str)
    group_key_changed = QtCore.pyqtSignal(object, str)
    show_lines_changed = QtCore.pyqtSignal(object, bool)

    def __init__(
        self,
        *,
        site: HostRuntimeSiteData,
        child_site_options: list[HostRuntimeSiteData],
        selected_child_path: tuple[str, ...] | None,
        parent_point_index: int | None,
        selected_point_index: int | None,
        selected_plot_mode: str | None,
        selected_x_key: str | None,
        selected_y_key: str | None,
        selected_z_key: str | None,
        selected_group_key: str | None,
        show_lines: bool,
        fit_backend: FitBackend | None = None,
    ):
        super().__init__()
        self._site = site
        self._fit_backend = BuiltinFitBackend() if fit_backend is None else fit_backend
        self._selected_point_index = selected_point_index
        self._plot_data = _site_plot_data(site, parent_point_index)
        self._child_site_options = child_site_options
        self._rendered_point_values = dict[int, tuple[float, float, float | None]]()
        self._rendered_group_values = dict[int, float]()
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
        self._current_source_indices: list[int] = []

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        self.setLayout(layout)
        self.setMinimumWidth(380)

        self._child_combo = _PopupAwareComboBox()
        self._child_combo.currentIndexChanged.connect(self._emit_child_site_changed)
        layout.addWidget(self._child_combo)

        self._title = QtWidgets.QLabel("/".join(site.path) or "root")
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        self._plot_controls_box = QtWidgets.QGroupBox("Plot")
        plot_controls_layout = QtWidgets.QGridLayout()
        plot_controls_layout.setContentsMargins(8, 8, 8, 8)
        plot_controls_layout.setHorizontalSpacing(8)
        plot_controls_layout.setVerticalSpacing(6)
        plot_controls_layout.setColumnStretch(1, 1)
        plot_controls_layout.setColumnStretch(3, 1)
        self._plot_controls_box.setLayout(plot_controls_layout)

        self._mode_label = QtWidgets.QLabel("mode")
        self._plot_mode_combo = _PopupAwareComboBox()
        self._plot_mode_combo.currentIndexChanged.connect(
            lambda *_: self.plot_mode_changed.emit(
                self._site.path, self._current_combo_data(self._plot_mode_combo)
            )
        )
        plot_controls_layout.addWidget(self._mode_label, 0, 0)
        plot_controls_layout.addWidget(self._plot_mode_combo, 0, 1)

        self._x_label = QtWidgets.QLabel("x")
        self._x_combo = _PopupAwareComboBox()
        self._x_combo.currentIndexChanged.connect(
            lambda *_: self.x_key_changed.emit(
                self._site.path, self._current_combo_data(self._x_combo)
            )
        )
        plot_controls_layout.addWidget(self._x_label, 0, 2)
        plot_controls_layout.addWidget(self._x_combo, 0, 3)

        self._y_label = QtWidgets.QLabel("y")
        self._y_combo = _PopupAwareComboBox()
        self._y_combo.currentIndexChanged.connect(
            lambda *_: self.y_key_changed.emit(
                self._site.path, self._current_combo_data(self._y_combo)
            )
        )
        plot_controls_layout.addWidget(self._y_label, 1, 0)
        plot_controls_layout.addWidget(self._y_combo, 1, 1)

        self._z_label = QtWidgets.QLabel("z")
        self._z_combo = _PopupAwareComboBox()
        self._z_combo.currentIndexChanged.connect(
            lambda *_: self.z_key_changed.emit(
                self._site.path, self._current_combo_data(self._z_combo)
            )
        )
        plot_controls_layout.addWidget(self._z_label, 1, 2)
        plot_controls_layout.addWidget(self._z_combo, 1, 3)

        self._group_label = QtWidgets.QLabel("group by")
        self._group_combo = _PopupAwareComboBox()
        self._group_combo.currentIndexChanged.connect(
            lambda *_: self.group_key_changed.emit(
                self._site.path, self._current_combo_data(self._group_combo)
            )
        )
        plot_controls_layout.addWidget(self._group_label, 2, 0)
        plot_controls_layout.addWidget(self._group_combo, 2, 1)

        self._show_lines_checkbox = QtWidgets.QCheckBox("connect points")
        self._show_lines_checkbox.toggled.connect(
            lambda checked: self.show_lines_changed.emit(self._site.path, checked)
        )
        plot_controls_layout.addWidget(self._show_lines_checkbox, 2, 2, 1, 2)

        self._fit_controls_box = QtWidgets.QGroupBox("Fit")
        fit_controls = QtWidgets.QGridLayout()
        fit_controls.setContentsMargins(8, 8, 8, 8)
        fit_controls.setHorizontalSpacing(8)
        fit_controls.setVerticalSpacing(6)
        fit_controls.setColumnStretch(1, 1)
        self._fit_controls_box.setLayout(fit_controls)

        self._fit_model_label = QtWidgets.QLabel("fit")
        self._fit_model_combo = _PopupAwareComboBox()
        self._fit_model_combo.currentIndexChanged.connect(self._sync_fit_controls)
        fit_controls.addWidget(self._fit_model_label, 0, 0)
        fit_controls.addWidget(self._fit_model_combo, 0, 1)
        self._fit_target_label = QtWidgets.QLabel("target")
        self._fit_target_combo = _PopupAwareComboBox()
        self._fit_target_combo.currentIndexChanged.connect(self._sync_fit_controls)
        fit_controls.addWidget(self._fit_target_label, 1, 0)
        fit_controls.addWidget(self._fit_target_combo, 1, 1)
        fit_buttons = QtWidgets.QHBoxLayout()
        fit_buttons.setContentsMargins(0, 0, 0, 0)
        fit_buttons.setSpacing(6)
        self._fit_button = QtWidgets.QPushButton("fit")
        self._fit_button.clicked.connect(self._fit_now)
        fit_buttons.addWidget(self._fit_button)
        self._clear_fit_button = QtWidgets.QPushButton("clear")
        self._clear_fit_button.clicked.connect(self._clear_fit)
        fit_buttons.addWidget(self._clear_fit_button)
        fit_buttons.addStretch(1)
        fit_buttons_widget = QtWidgets.QWidget()
        fit_buttons_widget.setLayout(fit_buttons)
        fit_controls.addWidget(fit_buttons_widget, 2, 0, 1, 2)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        header_row.addWidget(self._plot_controls_box, 3)
        header_row.addWidget(self._fit_controls_box, 2)
        layout.addLayout(header_row)

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

        layout.addWidget(self._details_box)

        plot_row = QtWidgets.QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.setSpacing(6)

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
        self._scatter = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush("#1f77b4"))
        self._highlight = pg.ScatterPlotItem(
            size=16,
            pen=pg.mkPen("#ffd84d", width=2.5),
            brush=pg.mkBrush(255, 255, 255, 120),
        )
        self._crosshair_x = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#ffd84d", width=1.2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self._crosshair_y = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("#ffd84d", width=1.2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self._crosshair_x.hide()
        self._crosshair_y.hide()
        self._plot_item.addItem(self._scatter)
        self._plot_item.addItem(self._highlight)
        self._plot_item.addItem(self._crosshair_x)
        self._plot_item.addItem(self._crosshair_y)
        self._fit_curve_item = self._plot_item.plot(pen=pg.mkPen("#ff7f0e", width=2.2, style=QtCore.Qt.PenStyle.DashLine))
        self._fit_curve_item.hide()
        self._scatter.sigClicked.connect(self._point_clicked)
        self._plot_item.scene().sigMouseClicked.connect(self._background_clicked)
        self._plot_item.scene().sigMouseMoved.connect(self._mouse_moved)
        plot_row.addWidget(self._plot_widget, 1)

        self._colorbar = _VerticalColorBarWidget()
        plot_row.addWidget(self._colorbar)

        layout.addLayout(plot_row, 1)

        self.update_state(
            site=site,
            child_site_options=child_site_options,
            selected_child_path=selected_child_path,
            parent_point_index=parent_point_index,
            selected_point_index=selected_point_index,
            selected_plot_mode=selected_plot_mode,
            selected_x_key=selected_x_key,
            selected_y_key=selected_y_key,
            selected_z_key=selected_z_key,
            selected_group_key=selected_group_key,
            show_lines=show_lines,
        )

    @property
    def site_path(self) -> tuple[str, ...]:
        return self._site.path

    def update_state(
        self,
        *,
        site: HostRuntimeSiteData,
        child_site_options: list[HostRuntimeSiteData],
        selected_child_path: tuple[str, ...] | None,
        parent_point_index: int | None,
        selected_point_index: int | None,
        selected_plot_mode: str | None,
        selected_x_key: str | None,
        selected_y_key: str | None,
        selected_z_key: str | None,
        selected_group_key: str | None,
        show_lines: bool,
    ) -> None:
        """Refresh the widget to match the current site-tree and UI selection state."""
        self._site = site
        self._selected_point_index = selected_point_index
        self._plot_data = _site_plot_data(site, parent_point_index)
        self._child_site_options = child_site_options

        self._title.setText("/".join(site.path) or "root")
        self._sync_child_combo(selected_child_path)
        self._sync_plot_mode_combo(selected_plot_mode)
        self._sync_xyz_combos(selected_x_key, selected_y_key, selected_z_key)
        self._sync_group_combo(selected_group_key)
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
        for key, label in _PLOT_MODE_CHOICES:
            self._plot_mode_combo.addItem(label, key)
        mode_index = (
            self._plot_mode_combo.findData(selected_plot_mode)
            if selected_plot_mode is not None
            else -1
        )
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
        selected_y_key: str | None,
        selected_z_key: str | None,
    ) -> None:
        x_choices = _default_x_choices(self._site)
        y_choices = _default_y_choices(self._site)
        z_choices = _default_z_choices(self._site)
        channel_keys = _channel_choice_keys(self._site)
        current_x_key = self._sync_choice_combo(
            self._x_combo,
            x_choices,
            selected_x_key,
            preferred_keys=[key for key in _choice_keys(x_choices) if key != _POINT_INDEX_KEY],
        )
        current_y_key = self._sync_choice_combo(
            self._y_combo,
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
            excluded_keys={key for key in (current_x_key, current_y_key) if key is not None},
        )

    def _sync_group_combo(self, selected_group_key: str | None) -> None:
        self._sync_choice_combo(
            self._group_combo,
            _group_by_choices(self._site, self._selected_x_key()),
            selected_group_key if selected_group_key is not None else _NO_GROUP_KEY,
            preferred_keys=[_NO_GROUP_KEY],
        )

    def _sync_fit_model_combo(self) -> None:
        self._sync_choice_combo(
            self._fit_model_combo,
            _fit_model_choices(self._fit_backend),
            self._current_combo_data(self._fit_model_combo),
            preferred_keys=[_NO_FIT_MODEL_KEY],
        )

    def _sync_fit_target_combo(self) -> None:
        self._sync_choice_combo(
            self._fit_target_combo,
            _fit_target_choices(self._selected_group_key()),
            self._current_combo_data(self._fit_target_combo),
            preferred_keys=[_FIT_TARGET_VISIBLE],
        )

    def _selected_fit_model(self) -> str | None:
        model_id = self._current_combo_data(self._fit_model_combo)
        return None if model_id in {None, _NO_FIT_MODEL_KEY} else model_id

    def _selected_fit_target(self) -> str:
        return self._current_combo_data(self._fit_target_combo) or _FIT_TARGET_VISIBLE

    def _sync_fit_controls(self) -> None:
        fit_enabled = self._selected_fit_model() is not None
        show_fit = self._selected_plot_mode() == _PLOT_MODE_1D
        if (not show_fit or not fit_enabled) and self._fit_active:
            self._clear_fit()
        self._fit_controls_box.setVisible(show_fit)
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

    def _selected_y_key(self) -> str | None:
        return self._current_combo_data(self._y_combo)

    def _selected_z_key(self) -> str | None:
        return self._current_combo_data(self._z_combo)

    def _selected_plot_mode(self) -> str:
        return self._current_combo_data(self._plot_mode_combo) or _PLOT_MODE_1D

    def _selected_group_key(self) -> str | None:
        group_key = self._current_combo_data(self._group_combo)
        return None if group_key in {None, _NO_GROUP_KEY} else group_key

    def _sync_control_visibility(self) -> None:
        is_2d = self._selected_plot_mode() in {_PLOT_MODE_2D_SCATTER, _PLOT_MODE_2D_IMAGE}
        show_group = not is_2d and self._group_combo.count() > 1
        self._z_label.setVisible(is_2d)
        self._z_combo.setVisible(is_2d)
        self._group_label.setVisible(show_group)
        self._group_combo.setVisible(show_group)
        self._show_lines_checkbox.setVisible(not is_2d)
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
        source_index = point.data()
        self.point_selected.emit(self._site.path, source_index)

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
        self._scatter.setData([])
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
        self._group_colors.clear()
        self._current_plot_arrays = None
        self._current_source_indices = []
        self._selected_readout.setText("selected: none")

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
        return list(self._plot_data.point_data.get(key, []))

    def _clear_fit(self) -> None:
        """Remove the current viewer-side fit overlay."""
        self._fit_active = False
        self._fit_result = None
        self._fit_target_group_value = None
        self._fit_curve_item.setData([], [])
        self._fit_curve_item.hide()
        self._fit_readout.setText("fit: none")

    def _fit_request_data(self) -> tuple[FitRequest, QtGui.QColor | str] | None:
        """Return the currently selected 1D data slice for viewer-side fitting."""
        if self._current_plot_mode != _PLOT_MODE_1D or self._current_plot_arrays is None:
            self._fit_readout.setText("fit: 1D only")
            return None

        x, y = self._current_plot_arrays
        target = self._selected_fit_target()
        if target == _FIT_TARGET_VISIBLE or not self._rendered_group_values:
            self._fit_target_group_value = None
            return FitRequest(x=x, y=y), "#ff7f0e"

        if self._selected_point_index is None:
            self._fit_readout.setText("fit: select a point to choose its group")
            return None
        selected_group = self._rendered_group_values.get(int(self._selected_point_index))
        if selected_group is None:
            self._fit_readout.setText("fit: selected point group is not visible")
            return None

        group_mask = np.array(
            [
                self._rendered_group_values.get(int(source_index)) == float(selected_group)
                for source_index in self._current_source_indices
            ],
            dtype=bool,
        )
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
            return

        self._fit_active = True
        self._fit_result = result
        self._fit_curve_item.setPen(
            pg.mkPen(color, width=2.2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self._fit_curve_item.setData(result.curve_x, result.curve_y)
        self._fit_curve_item.show()
        summary = result.summary()
        if self._fit_target_group_value is not None:
            self._fit_readout.setText(
                "fit: "
                f"{model_id} on {self._current_group_label} = "
                f"{_format_readout_value(self._fit_target_group_value)}; {summary}"
            )
        else:
            self._fit_readout.setText(f"fit: {model_id}; {summary}")

    def _update_selected_readout(self, plot_mode: str) -> None:
        """Show the numeric value of the selected sampled point."""
        if self._selected_point_index is None:
            self._selected_readout.setText("selected: none")
            return

        values = self._rendered_point_values.get(int(self._selected_point_index))
        if values is None:
            self._selected_readout.setText("selected: current point not visible")
            return

        x_value, y_value, z_value = values
        parts = [
            f"{self._current_x_label} = {_format_readout_value(x_value)}",
            f"{self._current_y_label} = {_format_readout_value(y_value)}",
        ]
        if plot_mode in {_PLOT_MODE_2D_SCATTER, _PLOT_MODE_2D_IMAGE} and z_value is not None:
            parts.append(f"{self._current_z_label} = {_format_readout_value(z_value)}")
        if plot_mode == _PLOT_MODE_1D and self._current_group_label is not None:
            group_value = self._rendered_group_values.get(int(self._selected_point_index))
            if group_value is not None:
                parts.append(
                    f"{self._current_group_label} = {_format_readout_value(group_value)}"
                )
        self._selected_readout.setText(
            f"selected point {int(self._selected_point_index)}: " + ", ".join(parts)
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

    def _render_1d_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        source_indices: list[int],
        group_key: str | None,
        group_values: np.ndarray | None,
    ) -> None:
        """Render a 1D scatter/line view."""
        self._colorbar.set_state(visible=False)
        self._legend.clear()
        self._legend.hide()

        if group_key is None or group_values is None:
            self._group_colors.clear()
            order = np.argsort(x, kind="stable")
            if self._show_lines_checkbox.isChecked():
                self._line_item.setPen(pg.mkPen("b", width=1.5))
                self._line_item.setData(x[order], y[order])
            else:
                self._line_item.setData([], [])
            for line_item in self._extra_line_items:
                line_item.setData([], [])
            self._scatter.setData(
                [
                    {
                        "pos": (float(xi), float(yi)),
                        "data": int(source_index),
                    }
                    for xi, yi, source_index in zip(x, y, source_indices, strict=True)
                ]
            )
            return

        groups = dict[float, list[int]]()
        for index, value in enumerate(group_values):
            groups.setdefault(float(value), []).append(index)
        group_items = list(groups.items())
        self._group_colors = {}

        while len(self._extra_line_items) + 1 < len(group_items):
            self._extra_line_items.append(self._plot_item.plot())
        all_line_items = [self._line_item, *self._extra_line_items]
        for line_item in all_line_items[len(group_items) :]:
            line_item.setData([], [])

        scatter_points = []
        for group_index, (group_value, indices) in enumerate(group_items):
            color = pg.intColor(group_index, hues=max(3, len(group_items)))
            self._group_colors[float(group_value)] = color
            pen = pg.mkPen(color, width=1.8)
            brush = pg.mkBrush(color)
            line_item = all_line_items[group_index]
            line_item.setPen(pen)
            ordered_indices = np.asarray(indices, dtype=int)[
                np.argsort(x[indices], kind="stable")
            ]
            if self._show_lines_checkbox.isChecked():
                line_item.setData(x[ordered_indices], y[ordered_indices])
            else:
                line_item.setData([], [])
            for point_index in indices:
                scatter_points.append(
                    {
                        "pos": (float(x[point_index]), float(y[point_index])),
                        "data": int(source_indices[point_index]),
                        "brush": brush,
                        "pen": pg.mkPen(30, 30, 30, 120),
                        "size": 9,
                    }
                )
            self._legend.addItem(line_item, _format_readout_value(group_value))
        self._legend.show()
        self._scatter.setData(scatter_points)

    def _render_2d_plot(
        self,
        plot_mode: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        source_indices: list[int],
        z_key: str | None,
        z_label_map: dict[str, str],
    ) -> None:
        """Render either a 2D scatter plot or a 2D image with point overlays."""
        brushes = _color_brushes(z)
        self._line_item.setData([], [])
        for line_item in self._extra_line_items:
            line_item.setData([], [])
        self._legend.clear()
        self._legend.hide()
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
                        "data": int(source_index),
                        "brush": pg.mkBrush(255, 255, 255, 70),
                        "pen": pg.mkPen(255, 255, 255, 160),
                        "size": 6,
                    }
                    for xi, yi, source_index in zip(x, y, source_indices, strict=True)
                ]
            )
        else:
            self._scatter.setData(
                [
                    {
                        "pos": (float(xi), float(yi)),
                        "data": int(source_index),
                        "brush": brush,
                        "pen": pg.mkPen(30, 30, 30, 120),
                        "size": 9,
                    }
                    for xi, yi, source_index, brush in zip(
                        x, y, source_indices, brushes, strict=True
                    )
                ]
            )
            self._status.setText(
                f"{self._plot_data.mode_label}; color = {z_label_map.get(z_key, z_key or '')}"
            )
        self._update_colorbar(z_key, z)

    def _highlight_selected_point(
        self, plot_mode: str, x: np.ndarray, y: np.ndarray, source_indices: list[int]
    ) -> None:
        """Show the selected point, and crosshairs in 2D modes."""
        highlighted = []
        if self._selected_point_index is not None:
            for xi, yi, source_index in zip(x, y, source_indices, strict=True):
                if int(source_index) == int(self._selected_point_index):
                    highlighted.append(
                        {"pos": (float(xi), float(yi)), "data": int(source_index)}
                    )
                    if plot_mode in {_PLOT_MODE_2D_SCATTER, _PLOT_MODE_2D_IMAGE}:
                        self._crosshair_x.setPos(float(xi))
                        self._crosshair_y.setPos(float(yi))
                        self._crosshair_x.show()
                        self._crosshair_y.show()
                    break
        self._highlight.setData(highlighted)

    def _render_plot(self) -> None:
        """Render the current site slice using the selected plot mode and axes."""
        if self._fit_active:
            self._clear_fit()
        self._status.setText(self._plot_data.mode_label)

        x_choices = _default_x_choices(self._site)
        y_choices = _default_y_choices(self._site)
        z_choices = _default_z_choices(self._site)
        x_label_map = _choice_label_map(x_choices)
        y_label_map = _choice_label_map(y_choices)
        z_label_map = _choice_label_map(z_choices)
        plot_mode = self._selected_plot_mode()
        x_key = self._selected_x_key()
        y_key = self._selected_y_key()
        group_key = self._selected_group_key()
        if x_key is None or y_key is None:
            self._clear_plot_items()
            return

        x_values = self._point_stream_values(x_key)
        y_values = self._point_stream_values(y_key)
        group_axis_values = self._point_stream_values(group_key) if group_key is not None else []

        count = min(
            len(x_values),
            len(y_values),
            len(self._plot_data.source_indices),
        )
        if group_key is not None:
            count = min(count, len(group_axis_values))
        if count == 0:
            self._clear_plot_items()
            return

        x = np.asarray(x_values[:count], dtype=float)
        y = np.asarray(y_values[:count], dtype=float)
        source_indices = self._plot_data.source_indices[:count]
        group = (
            np.asarray(group_axis_values[:count], dtype=float)
            if group_key is not None
            else None
        )
        z_key = None
        z = None
        if plot_mode in {_PLOT_MODE_2D_SCATTER, _PLOT_MODE_2D_IMAGE}:
            z_key = self._selected_z_key()
            z_values = self._point_stream_values(z_key) if z_key is not None else []
            count = min(count, len(z_values))
            if count == 0:
                self._clear_plot_items()
                return
            z = np.asarray(z_values[:count], dtype=float)
            x = x[:count]
            y = y[:count]
            source_indices = source_indices[:count]

            finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if not finite_mask.any():
                self._clear_plot_items()
                return
            x = x[finite_mask]
            y = y[finite_mask]
            z = z[finite_mask]
            source_indices = [src for src, keep in zip(source_indices, finite_mask, strict=True) if keep]
        else:
            finite_mask = np.isfinite(x) & np.isfinite(y)
            if group is not None:
                finite_mask &= np.isfinite(group)
            if not finite_mask.any():
                self._clear_plot_items()
                return
            x = x[finite_mask]
            y = y[finite_mask]
            if group is not None:
                group = group[finite_mask]
            source_indices = [src for src, keep in zip(source_indices, finite_mask, strict=True) if keep]

        self._image_item.hide()
        self._crosshair_x.hide()
        self._crosshair_y.hide()

        if plot_mode == _PLOT_MODE_1D:
            self._render_1d_plot(x, y, source_indices, group_key, group)
            if group_key is not None:
                group_label_map = _choice_label_map(_group_by_choices(self._site, x_key))
                self._status.setText(
                    f"{self._plot_data.mode_label}; grouped by {group_label_map.get(group_key, group_key)}"
                )
        else:
            assert z is not None
            self._render_2d_plot(plot_mode, x, y, z, source_indices, z_key, z_label_map)

        self._highlight_selected_point(plot_mode, x, y, source_indices)
        self._plot_item.setLabel(
            "bottom",
            x_label_map.get(
                x_key,
                x_key if x_key != _POINT_INDEX_KEY else "point_index",
            ),
        )
        self._plot_item.setLabel("left", y_label_map.get(y_key, y_key))
        self._current_x_label = x_label_map.get(
            x_key,
            x_key if x_key != _POINT_INDEX_KEY else "point_index",
        )
        self._current_y_label = y_label_map.get(y_key, y_key)
        self._current_z_label = z_label_map.get(z_key, z_key) if z_key is not None else None
        self._current_plot_mode = plot_mode
        self._current_plot_arrays = (x.copy(), y.copy())
        self._current_source_indices = list(source_indices)
        group_label_map = _choice_label_map(_group_by_choices(self._site, x_key))
        self._current_group_label = (
            group_label_map.get(group_key, group_key) if group_key is not None else None
        )
        self._rendered_point_values = {
            int(source_index): (
                float(xi),
                float(yi),
                None if z is None else float(zi),
            )
            for source_index, xi, yi, zi in zip(
                source_indices,
                x,
                y,
                [None] * len(source_indices) if z is None else z,
                strict=True,
            )
        }
        self._rendered_group_values = (
            {
                int(source_index): float(group_value)
                for source_index, group_value in zip(source_indices, group, strict=True)
            }
            if group is not None
            else {}
        )
        self._update_selected_readout(plot_mode)
        self._update_cursor_readout()


class RuntimePlotViewer(QtWidgets.QWidget):
    """Simple live prepared-runtime plot viewer.

    The viewer deliberately keeps very little model logic of its own. It reconstructs a
    snapshot from the current dataset subtree, chooses which sites should appear as a
    recursive set of columns, and lets each column widget render itself from that site
    plus a small amount of UI selection state.
    """

    def __init__(self, prefix: str):
        super().__init__()
        self._prefix = prefix
        self._snapshot = None
        self._paused = False
        self._pending_values: dict[str, Any] | None = None
        self._selected_points = dict[tuple[str, ...], int | None]()
        self._selected_child_paths = dict[tuple[str, ...], tuple[str, ...]]()
        self._selected_plot_modes = dict[tuple[str, ...], str]()
        self._selected_x_keys = dict[tuple[str, ...], str]()
        self._selected_y_keys = dict[tuple[str, ...], str]()
        self._selected_z_keys = dict[tuple[str, ...], str]()
        self._selected_group_keys = dict[tuple[str, ...], str]()
        self._show_lines = dict[tuple[str, ...], bool]()
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

    def data_changed(
        self,
        values: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        persist: dict[str, bool] | None = None,
        mods=None,
    ) -> None:
        """Rebuild the visible site tree from the latest live dataset snapshot."""
        del metadata, persist, mods
        if self._paused:
            self._pending_values = dict(values)
            self._update_status_text()
            return
        self._apply_live_values(values)

    def _apply_live_values(self, values: dict[str, Any]) -> None:
        """Apply one live dataset snapshot to the visible viewer state."""
        self.setUpdatesEnabled(False)
        try:
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

    def _clear_descendants(self, site_path: tuple[str, ...]) -> None:
        for store in (
            self._selected_points,
            self._selected_child_paths,
            self._selected_plot_modes,
            self._selected_x_keys,
            self._selected_y_keys,
            self._selected_z_keys,
            self._selected_group_keys,
            self._show_lines,
        ):
            for path in list(store.keys()):
                if _descends_from(path, site_path):
                    del store[path]

    def _choose_child_site(
        self, parent_path: tuple[str, ...], child_sites: list[HostRuntimeSiteData]
    ) -> HostRuntimeSiteData:
        """Return the chosen child site for one parent path, defaulting to the first."""
        selected_path = self._selected_child_paths.get(parent_path)
        for site in child_sites:
            if site.path == selected_path:
                return site
        chosen = child_sites[0]
        self._selected_child_paths[parent_path] = chosen.path
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
        self._update_status_text("Live prepared-runtime scan")

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
                    parent_point_index=self._selected_points.get(parent_path),
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
                selected_plot_mode=None,
                selected_x_key=None,
                selected_y_key=None,
                selected_z_key=None,
                selected_group_key=None,
                show_lines=False,
            )
            column.point_selected.connect(self._on_point_selected)
            column.child_site_changed.connect(self._on_child_site_changed)
            column.plot_mode_changed.connect(self._on_plot_mode_changed)
            column.x_key_changed.connect(self._on_x_key_changed)
            column.y_key_changed.connect(self._on_y_key_changed)
            column.z_key_changed.connect(self._on_z_key_changed)
            column.group_key_changed.connect(self._on_group_key_changed)
            column.show_lines_changed.connect(self._on_show_lines_changed)
            self._columns.append(column)
            self._columns_layout.addWidget(column)

        for column, spec in zip(self._columns, column_specs, strict=True):
            site = spec.site
            column.update_state(
                site=site,
                child_site_options=spec.child_site_options,
                selected_child_path=spec.selected_child_path,
                parent_point_index=spec.parent_point_index,
                selected_point_index=self._selected_points.get(site.path),
                selected_plot_mode=self._selected_plot_modes.get(site.path),
                selected_x_key=self._selected_x_keys.get(site.path),
                selected_y_key=self._selected_y_keys.get(site.path),
                selected_z_key=self._selected_z_keys.get(site.path),
                selected_group_key=self._selected_group_keys.get(site.path),
                show_lines=self._show_lines.get(site.path, False),
            )
        self._columns_layout.invalidate()
        self._columns_layout.activate()
        self._columns_widget.updateGeometry()
        self._columns_widget.adjustSize()
        self._scroll.widget().updateGeometry()
        self._scroll.viewport().update()

    def _on_point_selected(
        self, site_path: tuple[str, ...], point_index: int | None
    ) -> None:
        self._selected_points[site_path] = None if point_index is None else int(point_index)
        self._clear_descendants(site_path)
        self._rebuild_columns()

    def _on_child_site_changed(
        self, parent_path: tuple[str, ...], child_path: tuple[str, ...]
    ) -> None:
        self._selected_child_paths[parent_path] = child_path
        self._clear_descendants(parent_path)
        self._rebuild_columns()

    def _on_x_key_changed(self, site_path: tuple[str, ...], key: str) -> None:
        self._selected_x_keys[site_path] = key
        self._rebuild_columns()

    def _on_y_key_changed(self, site_path: tuple[str, ...], key: str) -> None:
        self._selected_y_keys[site_path] = key
        self._rebuild_columns()

    def _on_plot_mode_changed(self, site_path: tuple[str, ...], mode: str) -> None:
        self._selected_plot_modes[site_path] = mode
        self._rebuild_columns()

    def _on_z_key_changed(self, site_path: tuple[str, ...], key: str) -> None:
        self._selected_z_keys[site_path] = key
        self._rebuild_columns()

    def _on_group_key_changed(self, site_path: tuple[str, ...], key: str) -> None:
        self._selected_group_keys[site_path] = key
        self._rebuild_columns()

    def _on_show_lines_changed(
        self, site_path: tuple[str, ...], show_lines: bool
    ) -> None:
        self._show_lines[site_path] = show_lines
        self._rebuild_columns()

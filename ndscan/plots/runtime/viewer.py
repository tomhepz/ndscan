"""Simple prepared-runtime live plot viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg

from ..._qt import QtCore, QtWidgets
from ...results.scan_site_reader import HostRuntimeSiteData, HostRuntimeSiteSegment
from .live import snapshot_from_live_values

_POINT_INDEX_KEY = "__point_index__"


@dataclass(frozen=True)
class _SitePlotData:
    source_indices: list[int]
    point_data: dict[str, list[Any]]
    mode_label: str


def _schema_description_label(description: str | None, fallback: str) -> str:
    description = (description or "").strip()
    return description if description else fallback


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
    label = _schema_description_label(variable.get("description"), variable.get("name", key))
    return _append_unit_label(label, variable.get("spec", {}).get("unit"))


def _parameter_choice_label(key: str, schema: dict[str, Any]) -> str:
    param = schema.get("param", schema)
    label = _schema_description_label(param.get("description"), key)
    return _append_unit_label(label, param.get("spec", {}).get("unit"))


def _channel_choice_label(key: str, schema: dict[str, Any]) -> str:
    label = _schema_description_label(schema.get("description"), key)
    return _append_unit_label(label, schema.get("unit"))


def _default_x_choices(site: HostRuntimeSiteData) -> list[tuple[str, str]]:
    choices = [(_POINT_INDEX_KEY, "point_index")]
    seen = {_POINT_INDEX_KEY}
    for key in site.pseudoparams.keys():
        if key not in seen:
            choices.append((key, _pseudoparam_choice_label(key, site.pseudoparams[key])))
            seen.add(key)
    for key, schema in site.parameters.items():
        if schema.get("is_scanned", False) and key not in seen:
            choices.append((key, _parameter_choice_label(key, schema)))
            seen.add(key)
    return _unique_choice_labels(choices)


def _default_y_choices(site: HostRuntimeSiteData) -> list[tuple[str, str]]:
    return _unique_choice_labels(
        [(key, _channel_choice_label(key, schema)) for key, schema in site.channels.items()]
    )


def _choice_label_map(choices: list[tuple[str, str]]) -> dict[str, str]:
    return {key: label for key, label in choices}


def _current_or_latest_segment(
    site: HostRuntimeSiteData,
) -> HostRuntimeSiteSegment | None:
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


class _SiteColumnWidget(QtWidgets.QWidget):
    point_selected = QtCore.pyqtSignal(object, object)
    child_site_changed = QtCore.pyqtSignal(object, object)
    x_key_changed = QtCore.pyqtSignal(object, str)
    y_key_changed = QtCore.pyqtSignal(object, str)
    show_lines_changed = QtCore.pyqtSignal(object, bool)

    def __init__(
        self,
        *,
        site: HostRuntimeSiteData,
        child_site_options: list[HostRuntimeSiteData],
        selected_child_path: tuple[str, ...] | None,
        parent_point_index: int | None,
        selected_point_index: int | None,
        selected_x_key: str | None,
        selected_y_key: str | None,
        show_lines: bool,
    ):
        super().__init__()
        self._site = site
        self._selected_point_index = selected_point_index
        self._plot_data = _site_plot_data(site, parent_point_index)
        self._child_site_options = child_site_options

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        self.setLayout(layout)
        self.setMinimumWidth(380)

        self._child_combo = QtWidgets.QComboBox()
        self._child_combo.currentIndexChanged.connect(self._emit_child_site_changed)
        layout.addWidget(self._child_combo)

        self._title = QtWidgets.QLabel("/".join(site.path) or "root")
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("x"))
        self._x_combo = QtWidgets.QComboBox()
        self._x_combo.currentIndexChanged.connect(
            lambda *_: self.x_key_changed.emit(
                self._site.path, self._current_combo_data(self._x_combo)
            )
        )
        controls.addWidget(self._x_combo)

        controls.addWidget(QtWidgets.QLabel("y"))
        self._y_combo = QtWidgets.QComboBox()
        self._y_combo.currentIndexChanged.connect(
            lambda *_: self.y_key_changed.emit(
                self._site.path, self._current_combo_data(self._y_combo)
            )
        )
        controls.addWidget(self._y_combo)
        layout.addLayout(controls)

        self._show_lines_checkbox = QtWidgets.QCheckBox("connect points")
        self._show_lines_checkbox.toggled.connect(
            lambda checked: self.show_lines_changed.emit(self._site.path, checked)
        )
        layout.addWidget(self._show_lines_checkbox)

        self._status = QtWidgets.QLabel(self._plot_data.mode_label)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._plot_widget = pg.PlotWidget()
        self._plot_item = self._plot_widget.getPlotItem()
        self._plot_item.showGrid(x=True, y=True, alpha=0.25)
        self._line_item = self._plot_item.plot(pen=pg.mkPen("b", width=1.5))
        self._scatter = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush("#1f77b4"))
        self._highlight = pg.ScatterPlotItem(size=12, pen=pg.mkPen("r", width=2), brush=None)
        self._plot_item.addItem(self._scatter)
        self._plot_item.addItem(self._highlight)
        self._scatter.sigClicked.connect(self._point_clicked)
        self._plot_item.scene().sigMouseClicked.connect(self._background_clicked)
        layout.addWidget(self._plot_widget, 1)

        self.update_state(
            site=site,
            child_site_options=child_site_options,
            selected_child_path=selected_child_path,
            parent_point_index=parent_point_index,
            selected_point_index=selected_point_index,
            selected_x_key=selected_x_key,
            selected_y_key=selected_y_key,
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
        selected_x_key: str | None,
        selected_y_key: str | None,
        show_lines: bool,
    ) -> None:
        self._site = site
        self._selected_point_index = selected_point_index
        self._plot_data = _site_plot_data(site, parent_point_index)
        self._child_site_options = child_site_options

        self._title.setText("/".join(site.path) or "root")
        self._sync_child_combo(selected_child_path)
        self._sync_xy_combos(selected_x_key, selected_y_key)
        self._show_lines_checkbox.blockSignals(True)
        self._show_lines_checkbox.setChecked(show_lines)
        self._show_lines_checkbox.blockSignals(False)
        self._render_plot()

    def _sync_child_combo(self, selected_child_path: tuple[str, ...] | None) -> None:
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

    def _sync_xy_combos(
        self,
        selected_x_key: str | None,
        selected_y_key: str | None,
    ) -> None:
        x_choices = _default_x_choices(self._site)
        y_choices = _default_y_choices(self._site)

        self._x_combo.blockSignals(True)
        self._x_combo.clear()
        for key, label in x_choices:
            self._x_combo.addItem(label, key)
        x_index = self._x_combo.findData(selected_x_key) if selected_x_key is not None else -1
        if x_index == -1 and self._x_combo.count() > 1:
            x_index = 1
        if x_index == -1 and self._x_combo.count() > 0:
            x_index = 0
        if x_index != -1:
            self._x_combo.setCurrentIndex(x_index)
        self._x_combo.blockSignals(False)

        self._y_combo.blockSignals(True)
        self._y_combo.clear()
        for key, label in y_choices:
            self._y_combo.addItem(label, key)
        y_index = self._y_combo.findData(selected_y_key) if selected_y_key is not None else -1
        if y_index == -1 and self._y_combo.count() > 0:
            y_index = 0
        if y_index != -1:
            self._y_combo.setCurrentIndex(y_index)
        self._y_combo.blockSignals(False)

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

    def _point_clicked(self, scatter_item, points, event) -> None:
        if not points:
            return
        distances = np.array([(point.pos() - event.pos()).length() for point in points])
        point = points[int(distances.argmin())]
        source_index = point.data()
        self.point_selected.emit(self._site.path, source_index)

    def _background_clicked(self, event) -> None:
        if event.isAccepted():
            return
        self.point_selected.emit(self._site.path, None)

    def _render_plot(self) -> None:
        self._status.setText(self._plot_data.mode_label)

        x_choices = _default_x_choices(self._site)
        y_choices = _default_y_choices(self._site)
        x_label_map = _choice_label_map(x_choices)
        y_label_map = _choice_label_map(y_choices)
        x_key = self._selected_x_key()
        y_key = self._selected_y_key()
        if x_key is None or y_key is None:
            self._line_item.setData([], [])
            self._scatter.setData([])
            self._highlight.setData([])
            return

        y_values = list(self._plot_data.point_data.get(y_key, []))
        if x_key == _POINT_INDEX_KEY:
            x_values = list(self._plot_data.source_indices)
        else:
            x_values = list(self._plot_data.point_data.get(x_key, []))

        count = min(
            len(x_values),
            len(y_values),
            len(self._plot_data.source_indices),
        )
        if count == 0:
            self._line_item.setData([], [])
            self._scatter.setData([])
            self._highlight.setData([])
            return

        x = np.asarray(x_values[:count], dtype=float)
        y = np.asarray(y_values[:count], dtype=float)
        source_indices = self._plot_data.source_indices[:count]
        order = np.argsort(x, kind="stable")

        if self._show_lines_checkbox.isChecked():
            self._line_item.setData(x[order], y[order])
        else:
            self._line_item.setData([], [])
        self._scatter.setData(
            [
                {
                    "pos": (float(xi), float(yi)),
                    "data": int(source_index),
                }
                for xi, yi, source_index in zip(x, y, source_indices, strict=True)
            ]
        )

        highlighted = []
        if self._selected_point_index is not None:
            for xi, yi, source_index in zip(x, y, source_indices, strict=True):
                if int(source_index) == int(self._selected_point_index):
                    highlighted.append(
                        {"pos": (float(xi), float(yi)), "data": int(source_index)}
                    )
                    break
        self._highlight.setData(highlighted)
        self._plot_item.setLabel(
            "bottom",
            x_label_map.get(
                x_key,
                x_key if x_key != _POINT_INDEX_KEY else "point_index",
            ),
        )
        self._plot_item.setLabel("left", y_label_map.get(y_key, y_key))


class RuntimePlotViewer(QtWidgets.QWidget):
    """Simple live prepared-runtime plot viewer."""

    def __init__(self, prefix: str, set_dataset):
        super().__init__()
        self._prefix = prefix
        self._set_dataset = set_dataset
        self._snapshot = None
        self._selected_points = dict[tuple[str, ...], int | None]()
        self._selected_child_paths = dict[tuple[str, ...], tuple[str, ...]]()
        self._selected_x_keys = dict[tuple[str, ...], str]()
        self._selected_y_keys = dict[tuple[str, ...], str]()
        self._show_lines = dict[tuple[str, ...], bool]()
        self._columns: list[_SiteColumnWidget] = []

        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer)

        self._status = QtWidgets.QLabel("Waiting for prepared-runtime scan data…")
        outer.addWidget(self._status)

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
        del metadata, persist, mods
        self.setUpdatesEnabled(False)
        try:
            self._snapshot = snapshot_from_live_values(self._prefix, values)
            self._rebuild_columns()
        finally:
            self.setUpdatesEnabled(True)

    def _clear_descendants(self, site_path: tuple[str, ...]) -> None:
        for store in (
            self._selected_points,
            self._selected_child_paths,
            self._selected_x_keys,
            self._selected_y_keys,
            self._show_lines,
        ):
            for path in list(store.keys()):
                if _descends_from(path, site_path):
                    del store[path]

    def _choose_child_site(
        self, parent_path: tuple[str, ...], child_sites: list[HostRuntimeSiteData]
    ) -> HostRuntimeSiteData:
        selected_path = self._selected_child_paths.get(parent_path)
        for site in child_sites:
            if site.path == selected_path:
                return site
        chosen = child_sites[0]
        self._selected_child_paths[parent_path] = chosen.path
        return chosen

    def _rebuild_columns(self) -> None:
        snapshot = self._snapshot
        if snapshot is None or not snapshot.sites:
            self._status.setText("Waiting for prepared-runtime scan data…")
            self._sync_column_widgets([])
            return
        if () not in snapshot.sites:
            self._status.setText("Root prepared-runtime site has not appeared yet")
            self._sync_column_widgets([])
            return

        root_site = snapshot.get_site(())
        title = root_site.fragment_fqn
        source_id = root_site.metadata.get("site.source_id")
        if source_id:
            title += f" [{source_id}]"
        self.setWindowTitle(title)
        self._status.setText("Live prepared-runtime scan")

        column_specs = [
            dict(
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
                dict(
                    site=child_site,
                    child_site_options=child_sites,
                    selected_child_path=child_site.path,
                    parent_point_index=self._selected_points.get(parent_path),
                )
            )
            parent_path = child_site.path

        self._sync_column_widgets(column_specs)

    def _sync_column_widgets(self, column_specs: list[dict[str, Any]]) -> None:
        while len(self._columns) > len(column_specs):
            column = self._columns.pop()
            self._columns_layout.removeWidget(column)
            column.deleteLater()

        while len(self._columns) < len(column_specs):
            column = _SiteColumnWidget(
                site=column_specs[len(self._columns)]["site"],
                child_site_options=column_specs[len(self._columns)]["child_site_options"],
                selected_child_path=column_specs[len(self._columns)]["selected_child_path"],
                parent_point_index=column_specs[len(self._columns)]["parent_point_index"],
                selected_point_index=None,
                selected_x_key=None,
                selected_y_key=None,
                show_lines=False,
            )
            column.point_selected.connect(self._on_point_selected)
            column.child_site_changed.connect(self._on_child_site_changed)
            column.x_key_changed.connect(self._on_x_key_changed)
            column.y_key_changed.connect(self._on_y_key_changed)
            column.show_lines_changed.connect(self._on_show_lines_changed)
            self._columns.append(column)
            self._columns_layout.addWidget(column)

        for column, spec in zip(self._columns, column_specs, strict=True):
            site = spec["site"]
            column.update_state(
                site=site,
                child_site_options=spec["child_site_options"],
                selected_child_path=spec["selected_child_path"],
                parent_point_index=spec["parent_point_index"],
                selected_point_index=self._selected_points.get(site.path),
                selected_x_key=self._selected_x_keys.get(site.path),
                selected_y_key=self._selected_y_keys.get(site.path),
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

    def _on_show_lines_changed(
        self, site_path: tuple[str, ...], show_lines: bool
    ) -> None:
        self._show_lines[site_path] = show_lines
        self._rebuild_columns()

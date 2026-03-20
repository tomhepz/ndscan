"""Host-runtime-specific scan-row widgets.

The host dashboard path currently supports only *finite* grid generators. These
widgets reuse the shared numeric option primitives from :mod:`scan_options` while
keeping the host-specific subset out of the legacy-oriented module.
"""

from __future__ import annotations

from collections import OrderedDict

from .._qt import QtCore, QtWidgets
from .scan_options import (
    BoolFixedScanOption,
    BoolScanOption,
    EnumFixedScanOption,
    EnumScanOption,
    ExpandingScanOption,
    FixedScanOption,
    ListScanOption,
    NumericScanOption,
    StringFixedScanOption,
    SyncValue,
    make_divider,
)

__all__ = [
    "FiniteMinMaxScanOption",
    "FiniteCentreSpanScanOption",
    "GpoBoundsScanOption",
    "HostSubmissionModeSettings",
    "get_host_fixed_option_type",
    "list_host_scan_generator_option_types",
]


class FiniteMinMaxScanOption(NumericScanOption):
    """Finite-only min/max entry used by the host-runtime dashboard path."""

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box_start = self._make_spin_box()
        layout.addWidget(self.box_start)
        layout.setStretchFactor(self.box_start, 1)

        layout.addWidget(make_divider())

        self.box_points = QtWidgets.QSpinBox()
        self.box_points.setMinimum(2)
        self.box_points.setMaximum(0xFFFF)
        self.box_points.setValue(21)
        self.box_points.setSuffix(" pts")
        self.box_points.valueChanged.connect(self.value_changed)
        layout.addWidget(self.box_points)
        layout.setStretchFactor(self.box_points, 0)

        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 0)

        layout.addWidget(make_divider())

        self.box_stop = self._make_spin_box()
        layout.addWidget(self.box_stop)
        layout.setStretchFactor(self.box_stop, 1)

    def read_sync_values(self, sync_values: dict) -> None:
        if SyncValue.lower in sync_values:
            self.box_start.setValue(sync_values[SyncValue.lower])
        if SyncValue.upper in sync_values:
            self.box_stop.setValue(sync_values[SyncValue.upper])
        if SyncValue.num_points in sync_values:
            self.box_points.setValue(sync_values[SyncValue.num_points])

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.lower] = self.box_start.value()
        sync_values[SyncValue.upper] = self.box_stop.value()
        sync_values[SyncValue.num_points] = self.box_points.value()

    def write_to_submission(self, submission) -> None:
        submission.add_scan_axis(
            fqn=self.schema["fqn"],
            path=self.path,
            axis_type="linear",
            axis_range={
                "start": self.box_start.value() * self.scale,
                "stop": self.box_stop.value() * self.scale,
                "num_points": self.box_points.value(),
                "randomise_order": self.check_randomise.isChecked(),
            },
        )

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] != "linear":
            return False
        self.box_start.setValue(axis["range"].get("start", 0.0) / self.scale)
        self.box_stop.setValue(axis["range"].get("stop", 0.0) / self.scale)
        self.box_points.setValue(axis["range"].get("num_points", 21))
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True


class FiniteCentreSpanScanOption(NumericScanOption):
    """Finite-only centre/span entry used by the host-runtime dashboard path."""

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box_centre = self._make_spin_box()
        layout.addWidget(self.box_centre)
        layout.setStretchFactor(self.box_centre, 1)

        plusminus = QtWidgets.QLabel("±")
        layout.addWidget(plusminus)
        layout.setStretchFactor(plusminus, 0)

        self.box_half_span = self._make_spin_box(set_limits_from_spec=False)
        layout.addWidget(self.box_half_span)
        layout.setStretchFactor(self.box_half_span, 1)

        layout.addWidget(make_divider())

        self.box_points = QtWidgets.QSpinBox()
        self.box_points.setMinimum(1)
        self.box_points.setMaximum(0xFFFF)
        self.box_points.setValue(21)
        self.box_points.setSuffix(" pts")
        self.box_points.valueChanged.connect(self.value_changed)
        layout.addWidget(self.box_points)
        layout.setStretchFactor(self.box_points, 0)

        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 0)

    def read_sync_values(self, sync_values: dict) -> None:
        if SyncValue.centre in sync_values:
            self.box_centre.setValue(sync_values[SyncValue.centre])
        if SyncValue.num_points in sync_values:
            self.box_points.setValue(sync_values[SyncValue.num_points])

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.centre] = self.box_centre.value()
        sync_values[SyncValue.num_points] = self.box_points.value()

    def write_to_submission(self, submission) -> None:
        submission.add_scan_axis(
            fqn=self.schema["fqn"],
            path=self.path,
            axis_type="centre_span",
            axis_range={
                "centre": self.box_centre.value() * self.scale,
                "half_span": self.box_half_span.value() * self.scale,
                "limit_lower": self.min,
                "limit_upper": self.max,
                "num_points": self.box_points.value(),
                "randomise_order": self.check_randomise.isChecked(),
            },
        )

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] != "centre_span":
            return False
        self.box_half_span.setValue(axis["range"].get("half_span", 0.0) / self.scale)
        self.box_centre.setValue(axis["range"].get("centre", 0.0) / self.scale)
        self.box_points.setValue(axis["range"].get("num_points", 21))
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True


class GpoBoundsScanOption(NumericScanOption):
    """Bounded optimisation-dimension entry used in host GPO mode."""

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box_lower = self._make_spin_box()
        layout.addWidget(self.box_lower)
        layout.setStretchFactor(self.box_lower, 1)

        layout.addWidget(make_divider())

        self.box_upper = self._make_spin_box()
        layout.addWidget(self.box_upper)
        layout.setStretchFactor(self.box_upper, 1)

    def read_sync_values(self, sync_values: dict) -> None:
        if SyncValue.lower in sync_values:
            self.box_lower.setValue(sync_values[SyncValue.lower])
        if SyncValue.upper in sync_values:
            self.box_upper.setValue(sync_values[SyncValue.upper])

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.lower] = self.box_lower.value()
        sync_values[SyncValue.upper] = self.box_upper.value()

    def write_to_submission(self, submission) -> None:
        submission.add_gpo_axis(
            fqn=self.schema["fqn"],
            path=self.path,
            lower=self.box_lower.value() * self.scale,
            upper=self.box_upper.value() * self.scale,
        )

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] != "gpo_scan":
            return False
        self.box_lower.setValue(axis.get("lower", 0.0) / self.scale)
        self.box_upper.setValue(axis.get("upper", 0.0) / self.scale)
        return True


class HostSubmissionModeSettings(QtWidgets.QWidget):
    """Top-level host submission settings shown above the row editor."""

    value_changed = QtCore.pyqtSignal()
    mode_changed = QtCore.pyqtSignal(str)

    def __init__(self, *, initial_state: dict[str, object], channels, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setColumnStretch(1, 1)
        self.setLayout(layout)

        mode_label = QtWidgets.QLabel("Mode")
        self._mode_box = QtWidgets.QComboBox()
        self._mode_box.addItem("Grid", "grid")
        self._mode_box.addItem("GPO", "gpo")
        layout.addWidget(mode_label, 0, 0)
        layout.addWidget(self._mode_box, 0, 1)

        self._gpo_panel = QtWidgets.QWidget()
        gpo_layout = QtWidgets.QGridLayout()
        gpo_layout.setContentsMargins(0, 4, 0, 0)
        gpo_layout.setColumnStretch(1, 1)
        self._gpo_panel.setLayout(gpo_layout)
        layout.addWidget(self._gpo_panel, 1, 0, 1, 2)

        objective_label = QtWidgets.QLabel("Objective")
        self._objective_box = QtWidgets.QComboBox()
        self._objective_box.setEditable(True)
        for channel in channels:
            path = channel.get("path", "")
            if not path:
                continue
            description = channel.get("description", "")
            label = f"{path} - {description}" if description else path
            self._objective_box.addItem(label, path)
        gpo_layout.addWidget(objective_label, 0, 0)
        gpo_layout.addWidget(self._objective_box, 0, 1)

        acquisition_label = QtWidgets.QLabel("Acquisition")
        self._acquisition_box = QtWidgets.QComboBox()
        self._acquisition_box.addItem("UCB", "ucb")
        self._acquisition_box.addItem("EI", "ei")
        gpo_layout.addWidget(acquisition_label, 1, 0)
        gpo_layout.addWidget(self._acquisition_box, 1, 1)

        self._batch_size_box = self._make_optional_spin_box("default")
        self._initial_design_box = QtWidgets.QSpinBox()
        self._initial_design_box.setMinimum(1)
        self._initial_design_box.setMaximum(0xFFFF)
        self._initial_design_box.setValue(1)
        self._max_batches_box = self._make_optional_spin_box("unbounded")
        self._minimise_box = QtWidgets.QCheckBox("Minimise objective")
        self._minimise_box.setChecked(True)

        gpo_layout.addWidget(QtWidgets.QLabel("Batch size"), 2, 0)
        gpo_layout.addWidget(self._batch_size_box, 2, 1)
        gpo_layout.addWidget(QtWidgets.QLabel("Initial points"), 3, 0)
        gpo_layout.addWidget(self._initial_design_box, 3, 1)
        gpo_layout.addWidget(QtWidgets.QLabel("Max batches"), 4, 0)
        gpo_layout.addWidget(self._max_batches_box, 4, 1)
        gpo_layout.addWidget(self._minimise_box, 5, 1)

        self._mode_box.currentIndexChanged.connect(self._mode_box_changed)
        self._objective_box.currentTextChanged.connect(lambda *_: self.value_changed.emit())
        self._acquisition_box.currentIndexChanged.connect(lambda *_: self.value_changed.emit())
        self._batch_size_box.valueChanged.connect(lambda *_: self.value_changed.emit())
        self._initial_design_box.valueChanged.connect(lambda *_: self.value_changed.emit())
        self._max_batches_box.valueChanged.connect(lambda *_: self.value_changed.emit())
        self._minimise_box.toggled.connect(lambda *_: self.value_changed.emit())

        self._apply_initial_state(initial_state)
        self._update_mode_ui()

    def submission_mode(self) -> str:
        return self._mode_box.currentData()

    def write_to_submission(self, submission_state) -> None:
        if self.submission_mode() == "grid":
            submission_state.set_grid_mode()
            return
        submission_state.set_gpo_mode(
            objective_channel_path=self.objective_channel_path(),
            batch_size=self._optional_spin_box_value(self._batch_size_box),
            initial_design_size=self._initial_design_box.value(),
            max_batches=self._optional_spin_box_value(self._max_batches_box),
            acquisition=self._acquisition_box.currentData(),
            minimise=self._minimise_box.isChecked(),
        )

    def objective_channel_path(self) -> str:
        current_index = self._objective_box.currentIndex()
        if current_index >= 0 and self._objective_box.currentText() == self._objective_box.itemText(current_index):
            data = self._objective_box.currentData()
        else:
            data = None
        if isinstance(data, str) and data:
            return data
        return self._objective_box.currentText().strip()

    def _mode_box_changed(self, *_args) -> None:
        self._update_mode_ui()
        self.value_changed.emit()
        self.mode_changed.emit(self.submission_mode())

    def _update_mode_ui(self) -> None:
        self._gpo_panel.setVisible(self.submission_mode() == "gpo")

    def _apply_initial_state(self, state: dict[str, object]) -> None:
        mode_type = state.get("mode_type", "grid")
        index = 0 if mode_type != "gpo" else 1
        self._mode_box.setCurrentIndex(index)

        objective_path = state.get("objective_channel_path", "")
        if isinstance(objective_path, str) and objective_path:
            combo_index = self._objective_box.findData(objective_path)
            if combo_index >= 0:
                self._objective_box.setCurrentIndex(combo_index)
            else:
                self._objective_box.setEditText(objective_path)

        acquisition = state.get("acquisition", "ucb")
        if isinstance(acquisition, str):
            acq_index = self._acquisition_box.findData(acquisition)
            if acq_index >= 0:
                self._acquisition_box.setCurrentIndex(acq_index)

        initial_design_size = state.get("initial_design_size", 1)
        if isinstance(initial_design_size, int) and initial_design_size > 0:
            self._initial_design_box.setValue(initial_design_size)

        self._set_optional_spin_box(self._batch_size_box, state.get("batch_size", None))
        self._set_optional_spin_box(self._max_batches_box, state.get("max_batches", None))

        minimise = state.get("minimise", True)
        if isinstance(minimise, bool):
            self._minimise_box.setChecked(minimise)

    def _make_optional_spin_box(self, special_value_text: str) -> QtWidgets.QSpinBox:
        box = QtWidgets.QSpinBox()
        box.setMinimum(0)
        box.setMaximum(0xFFFF)
        box.setSpecialValueText(special_value_text)
        box.setValue(0)
        return box

    def _set_optional_spin_box(self, box: QtWidgets.QSpinBox, value) -> None:
        if isinstance(value, int) and value > 0:
            box.setValue(value)
        else:
            box.setValue(0)

    def _optional_spin_box_value(self, box: QtWidgets.QSpinBox) -> int | None:
        value = box.value()
        return value if value > 0 else None


def get_host_fixed_option_type(schema_type: str) -> type:
    """Return the fixed-value widget for a host dashboard parameter row."""

    if schema_type == "string":
        return StringFixedScanOption
    if schema_type == "bool":
        return BoolFixedScanOption
    if schema_type == "enum":
        return EnumFixedScanOption
    return FixedScanOption


def list_host_scan_generator_option_types(
    schema_type: str, is_scannable: bool
) -> OrderedDict[str, type]:
    """Return scan-generator widgets for the editable host-runtime dashboard subset."""

    result = OrderedDict([])
    if not is_scannable:
        return result

    if schema_type == "bool":
        result["Boolean"] = BoolScanOption
    elif schema_type == "enum":
        result["Choice"] = EnumScanOption
    elif schema_type != "string":
        result["Min./Max."] = FiniteMinMaxScanOption
        result["Centered"] = FiniteCentreSpanScanOption
        result["Expanding"] = ExpandingScanOption
        result["List"] = ListScanOption
    return result

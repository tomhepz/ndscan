"""Prepared-runtime-specific scan-row widgets.

The scan-submission dashboard path currently supports only *finite* grid generators. These
widgets reuse the shared numeric option primitives from :mod:`scan_options` while
keeping the scan-submission subset out of the legacy-oriented module.
"""

from __future__ import annotations

from collections import OrderedDict

from .._qt import QtCore, QtWidgets
from ..submission.scan_submission_schema import (
    DEFAULT_GPO_ACQUISITION_NUM_STARTS,
    DEFAULT_GPO_FIT_LR,
    DEFAULT_GPO_FIT_STEPS,
    DEFAULT_GPO_INITIAL_DESIGN_SIZE,
    DEFAULT_GPO_SURROGATE_NUM_STARTS,
)
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
    "ScanSubmissionModeSettings",
    "get_scan_submission_fixed_option_type",
    "list_scan_submission_generator_option_types",
]

_UNTIL_STOPPED_REPEAT_COUNT = 0
_MAX_REPEAT_COUNT = 2**31 - 1


class FiniteMinMaxScanOption(NumericScanOption):
    """Finite-only min/max entry used by the prepared-runtime dashboard path."""

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
    """Finite-only centre/span entry used by the prepared-runtime dashboard path."""

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
    """Bounded optimisation-dimension entry used in scan-submission GPO mode."""

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


class ScanSubmissionModeSettings(QtWidgets.QWidget):
    """Top-level scan submission settings shown above the row editor."""

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

        self._grid_panel = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout()
        grid_layout.setContentsMargins(0, 4, 0, 0)
        grid_layout.setColumnStretch(1, 1)
        self._grid_panel.setLayout(grid_layout)
        layout.addWidget(self._grid_panel, 1, 0, 1, 2)

        self._randomise_order_box = QtWidgets.QCheckBox("Randomise point order")
        self._randomise_order_box.setToolTip(
            "Randomise the order of the logical grid points before applying repeats."
        )
        grid_layout.addWidget(self._randomise_order_box, 0, 1)

        repeats_label = QtWidgets.QLabel("Repeats per point")
        self._repeat_count_box = QtWidgets.QSpinBox()
        self._repeat_count_box.setMinimum(_UNTIL_STOPPED_REPEAT_COUNT)
        self._repeat_count_box.setMaximum(_MAX_REPEAT_COUNT)
        self._repeat_count_box.setSpecialValueText("until stopped")
        self._repeat_count_box.setValue(1)
        self._repeat_count_box.setToolTip(
            "0 means repeat until the run is stopped. This can produce a very large "
            "result file if left running."
        )
        grid_layout.addWidget(repeats_label, 1, 0)
        grid_layout.addWidget(self._repeat_count_box, 1, 1)

        schedule_label = QtWidgets.QLabel("Repeat schedule")
        self._repeat_schedule_box = QtWidgets.QComboBox()
        self._repeat_schedule_box.addItem("Serial", "serial")
        self._repeat_schedule_box.addItem("Interleaved", "interleaved")
        self._repeat_schedule_box.addItem("Globally shuffled", "shuffled")
        self._repeat_schedule_box.setToolTip(
            "Serial repeats one grid point before moving to the next. Interleaved "
            "takes one repeat of each point per round. Globally shuffled makes all "
            "repeated acquisitions first, then randomises their complete order."
        )
        grid_layout.addWidget(schedule_label, 2, 0)
        grid_layout.addWidget(self._repeat_schedule_box, 2, 1)

        self._gpo_panel = QtWidgets.QWidget()
        gpo_layout = QtWidgets.QGridLayout()
        gpo_layout.setContentsMargins(0, 4, 0, 0)
        gpo_layout.setColumnStretch(1, 1)
        self._gpo_panel.setLayout(gpo_layout)
        layout.addWidget(self._gpo_panel, 2, 0, 1, 2)

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
        self._initial_design_box.setValue(DEFAULT_GPO_INITIAL_DESIGN_SIZE)
        self._max_batches_box = self._make_optional_spin_box("unbounded")
        self._fit_steps_box = QtWidgets.QSpinBox()
        self._fit_steps_box.setMinimum(1)
        self._fit_steps_box.setMaximum(1_000_000)
        self._fit_steps_box.setValue(DEFAULT_GPO_FIT_STEPS)
        self._fit_steps_box.setToolTip(
            "Number of optimizer steps used when fitting the GP model each batch."
        )
        self._fit_lr_box = QtWidgets.QDoubleSpinBox()
        self._fit_lr_box.setDecimals(6)
        self._fit_lr_box.setMinimum(1e-6)
        self._fit_lr_box.setMaximum(10.0)
        self._fit_lr_box.setSingleStep(0.01)
        self._fit_lr_box.setValue(DEFAULT_GPO_FIT_LR)
        self._fit_lr_box.setToolTip(
            "Learning rate used when fitting the GP model each batch."
        )
        self._acquisition_num_starts_box = QtWidgets.QSpinBox()
        self._acquisition_num_starts_box.setMinimum(1)
        self._acquisition_num_starts_box.setMaximum(0xFFFF)
        self._acquisition_num_starts_box.setValue(DEFAULT_GPO_ACQUISITION_NUM_STARTS)
        self._acquisition_num_starts_box.setToolTip(
            "Number of starting points used to optimise the acquisition function."
        )
        self._surrogate_num_starts_box = QtWidgets.QSpinBox()
        self._surrogate_num_starts_box.setMinimum(1)
        self._surrogate_num_starts_box.setMaximum(0xFFFF)
        self._surrogate_num_starts_box.setValue(DEFAULT_GPO_SURROGATE_NUM_STARTS)
        self._surrogate_num_starts_box.setToolTip(
            "Number of starting points used for surrogate-model argmin/argmax estimates."
        )
        self._minimise_box = QtWidgets.QCheckBox("Minimise objective")
        self._minimise_box.setChecked(True)

        gpo_layout.addWidget(QtWidgets.QLabel("Batch size"), 2, 0)
        gpo_layout.addWidget(self._batch_size_box, 2, 1)
        gpo_layout.addWidget(QtWidgets.QLabel("Initial points"), 3, 0)
        gpo_layout.addWidget(self._initial_design_box, 3, 1)
        gpo_layout.addWidget(QtWidgets.QLabel("Max batches"), 4, 0)
        gpo_layout.addWidget(self._max_batches_box, 4, 1)
        gpo_layout.addWidget(QtWidgets.QLabel("Fit steps"), 5, 0)
        gpo_layout.addWidget(self._fit_steps_box, 5, 1)
        gpo_layout.addWidget(QtWidgets.QLabel("Fit learning rate"), 6, 0)
        gpo_layout.addWidget(self._fit_lr_box, 6, 1)
        gpo_layout.addWidget(QtWidgets.QLabel("Acquisition restarts"), 7, 0)
        gpo_layout.addWidget(self._acquisition_num_starts_box, 7, 1)
        gpo_layout.addWidget(QtWidgets.QLabel("Surrogate restarts"), 8, 0)
        gpo_layout.addWidget(self._surrogate_num_starts_box, 8, 1)
        gpo_layout.addWidget(self._minimise_box, 9, 1)

        self._mode_box.currentIndexChanged.connect(self._mode_box_changed)
        self._randomise_order_box.toggled.connect(lambda *_: self.value_changed.emit())
        self._repeat_count_box.valueChanged.connect(lambda *_: self.value_changed.emit())
        self._repeat_schedule_box.currentIndexChanged.connect(
            self._repeat_schedule_changed
        )
        self._objective_box.currentTextChanged.connect(lambda *_: self.value_changed.emit())
        self._acquisition_box.currentIndexChanged.connect(lambda *_: self.value_changed.emit())
        self._batch_size_box.valueChanged.connect(lambda *_: self.value_changed.emit())
        self._initial_design_box.valueChanged.connect(lambda *_: self.value_changed.emit())
        self._max_batches_box.valueChanged.connect(lambda *_: self.value_changed.emit())
        self._fit_steps_box.valueChanged.connect(lambda *_: self.value_changed.emit())
        self._fit_lr_box.valueChanged.connect(lambda *_: self.value_changed.emit())
        self._acquisition_num_starts_box.valueChanged.connect(
            lambda *_: self.value_changed.emit()
        )
        self._surrogate_num_starts_box.valueChanged.connect(
            lambda *_: self.value_changed.emit()
        )
        self._minimise_box.toggled.connect(lambda *_: self.value_changed.emit())

        self._apply_initial_state(initial_state)
        self._update_repeat_schedule_ui()
        self._update_mode_ui()

    def submission_mode(self) -> str:
        return self._mode_box.currentData()

    def write_to_submission(self, submission_state) -> None:
        if self.submission_mode() == "grid":
            repeat_schedule = self._repeat_schedule_box.currentData()
            submission_state.set_grid_mode(
                randomise_order_globally=(
                    repeat_schedule != "shuffled"
                    and self._randomise_order_box.isChecked()
                ),
                num_repeats_per_point=self._repeat_count_value(),
                repeat_schedule=repeat_schedule,
            )
            return
        submission_state.set_gpo_mode(
            objective_channel_path=self.objective_channel_path(),
            batch_size=self._optional_spin_box_value(self._batch_size_box),
            initial_design_size=self._initial_design_box.value(),
            max_batches=self._optional_spin_box_value(self._max_batches_box),
            acquisition=self._acquisition_box.currentData(),
            minimise=self._minimise_box.isChecked(),
            fit_steps=self._fit_steps_box.value(),
            fit_lr=self._fit_lr_box.value(),
            acquisition_num_starts=self._acquisition_num_starts_box.value(),
            surrogate_num_starts=self._surrogate_num_starts_box.value(),
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

    def _repeat_schedule_changed(self, *_args) -> None:
        self._update_repeat_schedule_ui()
        self.value_changed.emit()

    def _update_repeat_schedule_ui(self) -> None:
        globally_shuffled = self._repeat_schedule_box.currentData() == "shuffled"

        # Shuffling the complete repeated list already randomises the logical points,
        # so the separate pre-repeat randomisation checkbox would have no further effect.
        self._randomise_order_box.setEnabled(not globally_shuffled)
        if globally_shuffled:
            self._randomise_order_box.setToolTip(
                "Globally shuffled repeats already randomise every acquisition."
            )
        else:
            self._randomise_order_box.setToolTip(
                "Randomise the order of the logical grid points before applying repeats."
            )

        # An unbounded sequence has no complete list that could be shuffled. Raising the
        # minimum also moves a previous 'until stopped' value to the smallest valid one.
        self._repeat_count_box.setMinimum(
            1 if globally_shuffled else _UNTIL_STOPPED_REPEAT_COUNT
        )

    def _update_mode_ui(self) -> None:
        self._grid_panel.setVisible(self.submission_mode() == "grid")
        self._gpo_panel.setVisible(self.submission_mode() == "gpo")

    def _apply_initial_state(self, state: dict[str, object]) -> None:
        mode_type = state.get("mode_type", "grid")
        index = 0 if mode_type != "gpo" else 1
        self._mode_box.setCurrentIndex(index)

        self._randomise_order_box.setChecked(
            bool(state.get("randomise_order_globally", False))
        )
        self._set_repeat_count_value(state.get("num_repeats_per_point", 1))
        repeat_schedule = state.get("repeat_schedule", "serial")
        if isinstance(repeat_schedule, str):
            schedule_index = self._repeat_schedule_box.findData(repeat_schedule)
            if schedule_index >= 0:
                self._repeat_schedule_box.setCurrentIndex(schedule_index)

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

        initial_design_size = state.get(
            "initial_design_size",
            DEFAULT_GPO_INITIAL_DESIGN_SIZE,
        )
        if isinstance(initial_design_size, int) and initial_design_size > 0:
            self._initial_design_box.setValue(initial_design_size)

        self._set_optional_spin_box(self._batch_size_box, state.get("batch_size", None))
        self._set_optional_spin_box(self._max_batches_box, state.get("max_batches", None))

        fit_steps = state.get("fit_steps", DEFAULT_GPO_FIT_STEPS)
        if isinstance(fit_steps, int) and fit_steps > 0:
            self._fit_steps_box.setValue(fit_steps)

        fit_lr = state.get("fit_lr", DEFAULT_GPO_FIT_LR)
        if isinstance(fit_lr, (int, float)) and fit_lr > 0.0:
            self._fit_lr_box.setValue(float(fit_lr))

        acquisition_num_starts = state.get(
            "acquisition_num_starts",
            DEFAULT_GPO_ACQUISITION_NUM_STARTS,
        )
        if isinstance(acquisition_num_starts, int) and acquisition_num_starts > 0:
            self._acquisition_num_starts_box.setValue(acquisition_num_starts)

        surrogate_num_starts = state.get(
            "surrogate_num_starts",
            DEFAULT_GPO_SURROGATE_NUM_STARTS,
        )
        if isinstance(surrogate_num_starts, int) and surrogate_num_starts > 0:
            self._surrogate_num_starts_box.setValue(surrogate_num_starts)

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

    def _repeat_count_value(self) -> int | None:
        value = self._repeat_count_box.value()
        return None if value == _UNTIL_STOPPED_REPEAT_COUNT else value

    def _set_repeat_count_value(self, value) -> None:
        if value is None:
            self._repeat_count_box.setValue(_UNTIL_STOPPED_REPEAT_COUNT)
        elif isinstance(value, int) and value > 0:
            self._repeat_count_box.setValue(value)
        else:
            self._repeat_count_box.setValue(1)


def get_scan_submission_fixed_option_type(schema_type: str) -> type:
    """Return the fixed-value widget for a scan-submission dashboard parameter row."""

    if schema_type == "string":
        return StringFixedScanOption
    if schema_type == "bool":
        return BoolFixedScanOption
    if schema_type == "enum":
        return EnumFixedScanOption
    return FixedScanOption


def list_scan_submission_generator_option_types(
    schema_type: str, is_scannable: bool
) -> OrderedDict[str, type]:
    """Return scan-generator widgets for the editable prepared-runtime dashboard subset."""

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

"""Parameter-row widgets for the dashboard argument editor.

Keeping these widgets separate from :mod:`argument_editor` lets the editor stay a
shell around layout and persistence orchestration rather than also being the home for
all row-level mode logic.
"""

from __future__ import annotations

import logging

from artiq.gui.tools import LayoutWidget

from .._qt import QtCore, QtGui, QtWidgets
from .scan_submission_options import (
    GpoBoundsScanOption,
    get_scan_submission_fixed_option_type,
    list_scan_submission_generator_option_types,
)
from .utils import eval_default_using_local_datasets, format_override_identity

logger = logging.getLogger(__name__)

__all__ = ["OverrideEntry", "ScanOverrideEntry", "ScanPseudoparamEntry"]


class _BaseOverrideEntry(LayoutWidget):
    value_changed = QtCore.pyqtSignal()

    def __init__(self, schema, path, *args):
        super().__init__(*args)
        self.schema = schema
        self.path = path
        self.sync_values = {}

    def _default_value(self, manager_datasets, id_for_log: str):
        try:
            return eval_default_using_local_datasets(
                self.schema["default"], manager_datasets
            )
        except Exception as e:
            logger.error(
                'Failed to evaluate defaults string "%s" for %s: %s',
                self.schema["default"],
                id_for_log,
                e,
            )
            return None


class OverrideEntry(_BaseOverrideEntry):
    def __init__(self, option_classes, schema, path, *args):
        super().__init__(schema, path, *args)

        self.scan_type = QtWidgets.QComboBox()
        self.addWidget(self.scan_type, col=0)

        self.widget_stack = QtWidgets.QStackedWidget()

        # The non-scan option should be on the top.
        assert next(iter(option_classes.keys())) == "Fixed"
        if len(option_classes) == 1:
            self.scan_type.setEnabled(False)
        self.current_option_idx = 0

        self.options = []
        for name, option_cls in option_classes.items():
            self.scan_type.addItem(name)

            option = option_cls(self.schema, self.path)
            option.value_changed.connect(self.value_changed)
            container = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout()
            # For tight spacing, let other parts of entry line dominate margins.
            layout.setContentsMargins(0, 0, 0, 0)
            option.build_ui(layout)
            container.setLayout(layout)

            self.widget_stack.addWidget(container)
            self.options.append(option)
        self.scan_type.currentIndexChanged.connect(self._current_index_changed)
        self.addWidget(self.widget_stack, col=1)

    def read_from_params(self, params: dict, manager_datasets) -> None:
        id_for_log = format_override_identity(self.schema["fqn"], self.path)

        # Check if this parameter is part of the legacy scan axes.
        for axis in params.get("scan", {}).get("axes", []):
            if axis["fqn"] == self.schema["fqn"] and axis["path"] == self.path:
                for idx, option in enumerate(self.options):
                    if option.attempt_read_from_axis(axis):
                        self.current_option_idx = idx
                        self._current_index_changed(idx)
                        self.scan_type.setCurrentIndex(idx)
                        return
                logger.warning("Failed to read scan params for %s", id_for_log)

        for override in params.get("overrides", {}).get(self.schema["fqn"], []):
            if override["path"] == self.path:
                self._set_fixed_value(override["value"])
                return

        self._set_fixed_value(self._default_value(manager_datasets, id_for_log))
        self.disable_scan()

    def write_to_submission(self, submission_state) -> None:
        self.options[self.scan_type.currentIndex()].write_to_submission(submission_state)

    def disable_scan(self) -> None:
        self.scan_type.setCurrentIndex(0)

    def _set_fixed_value(self, value) -> None:
        self.options[0].set_value(value)
        self.options[0].write_sync_values(self.sync_values)

    def _current_index_changed(self, new_idx) -> None:
        self.options[self.current_option_idx].write_sync_values(self.sync_values)
        self.options[new_idx].read_sync_values(self.sync_values)
        self.widget_stack.setCurrentIndex(new_idx)
        self.current_option_idx = new_idx


class ScanOverrideEntry(_BaseOverrideEntry):
    """Prepared-runtime row editor for the current dashboard subset.

    Scan-submission rows intentionally expose a smaller set of top-level modes than the legacy
    path.  The primary choice is "Fixed" vs "Scan"; once "Scan" is selected, a nested
    generator selector chooses between min/max, centered, expanding, and list.
    """

    def __init__(
        self,
        schema,
        path,
        *,
        is_scannable: bool,
        backend,
        submission_mode: str = "grid",
        **kwargs,
    ):
        super().__init__(schema, path, **kwargs)
        self._scan_submission_backend = backend
        self._supports_rebind = schema["type"] not in {"string", "bool", "enum"}
        self._supports_gpo_scan = schema["type"] not in {"string", "bool", "enum"} and is_scannable
        self._symbol_name = backend.symbol_name_for_target(
            fqn=schema["fqn"],
            path=path,
        )
        self._submission_mode = submission_mode
        self._fixed_option = self._build_option(get_scan_submission_fixed_option_type(schema["type"]))
        self._scan_option_names = []
        self._scan_options = []
        for name, option_cls in list_scan_submission_generator_option_types(
            schema["type"],
            is_scannable,
        ).items():
            self._scan_option_names.append(name)
            self._scan_options.append(self._build_option(option_cls))
        self._gpo_option = (
            self._build_option(GpoBoundsScanOption) if self._supports_gpo_scan else None
        )

        self._mode_box = QtWidgets.QComboBox()
        self._mode_box.currentIndexChanged.connect(self._mode_changed)
        self.addWidget(self._mode_box, col=0)

        self._mode_stack = QtWidgets.QStackedWidget()
        self._mode_stack.addWidget(self._fixed_option.container)

        self._scan_container = QtWidgets.QWidget()
        self._scan_layout = QtWidgets.QHBoxLayout()
        self._scan_layout.setContentsMargins(0, 0, 0, 0)
        self._scan_container.setLayout(self._scan_layout)

        self._scan_kind_box = QtWidgets.QComboBox()
        for name in self._scan_option_names:
            self._scan_kind_box.addItem(name)
        self._scan_kind_box.currentIndexChanged.connect(self._scan_kind_changed)
        self._scan_layout.addWidget(self._scan_kind_box)

        self._scan_stack = QtWidgets.QStackedWidget()
        for option in self._scan_options:
            self._scan_stack.addWidget(option.container)
        self._scan_layout.addWidget(self._scan_stack, stretch=1)
        self._scan_kind_box.setVisible(len(self._scan_options) > 1)

        self._mode_stack.addWidget(self._scan_container)
        if self._gpo_option is not None:
            self._mode_stack.addWidget(self._gpo_option.container)
        self._rebind_editor = _ScanRebindEditor(symbol_name=self._symbol_name)
        self._rebind_editor.box.textChanged.connect(lambda *_: self.value_changed.emit())
        self._mode_stack.addWidget(self._rebind_editor.container)
        self.addWidget(self._mode_stack, col=1)

        self._group_container = LayoutWidget()
        self._group_container.layout.setContentsMargins(6, 0, 0, 0)
        group_label = QtWidgets.QLabel("Group")
        group_label.setToolTip(
            "Scan rows in the same non-empty group are zipped together. "
            "Different groups remain Cartesian."
        )
        self._group_box = QtWidgets.QLineEdit()
        self._group_box.setPlaceholderText("optional")
        self._group_box.setMaximumWidth(110)
        self._group_box.setToolTip(group_label.toolTip())
        self._group_box.textChanged.connect(lambda *_: self.value_changed.emit())
        self._group_container.addWidget(group_label, col=0)
        self._group_container.addWidget(self._group_box, col=1)
        self.addWidget(self._group_container, col=2)
        self._current_mode = "Fixed"
        self._current_scan_option_idx = 0
        self._rebuild_mode_box()
        self._update_mode_ui()

    def read_from_params(self, params: dict, manager_datasets) -> None:
        id_for_log = format_override_identity(self.schema["fqn"], self.path)
        entry = self._scan_submission_backend.find_entry(
            params,
            fqn=self.schema["fqn"],
            path=self.path,
        )
        if entry is not None:
            if entry.mode.type == "fixed":
                self._group_box.setText("")
                self._set_fixed_value(entry.mode.value)
                self.disable_scan()
                return

            if entry.mode.type == "scan":
                axis = {
                    "type": entry.mode.generator.type,
                    "range": dict(entry.mode.generator.range),
                }
                for idx, option in enumerate(self._scan_options):
                    if option.attempt_read_from_axis(axis):
                        self._group_box.setText(entry.mode.group or "")
                        self._scan_kind_box.setCurrentIndex(idx)
                        self._mode_box.setCurrentText("Scan")
                        return
                logger.warning("Failed to read scan submission params for %s", id_for_log)

            if entry.mode.type == "rebind":
                self._group_box.setText("")
                self._rebind_editor.set_expression(entry.mode.expr)
                self._mode_box.setCurrentText("Rebind")
                return

            if entry.mode.type == "gpo_scan" and self._gpo_option is not None:
                axis = {"type": "gpo_scan", "lower": entry.mode.lower, "upper": entry.mode.upper}
                if self._gpo_option.attempt_read_from_axis(axis):
                    self._group_box.setText("")
                    self._mode_box.setCurrentText("GPO scan")
                    return

        for override in params.get("overrides", {}).get(self.schema["fqn"], []):
            if override["path"] == self.path:
                self._group_box.setText("")
                self._set_fixed_value(override["value"])
                self.disable_scan()
                return

        self._group_box.setText("")
        self._set_fixed_value(self._default_value(manager_datasets, id_for_log))
        if self._supports_rebind and not self._rebind_editor.expression():
            self._rebind_editor.set_expression("0.0")
        self.disable_scan()

    def write_to_submission(self, submission_state) -> None:
        mode = self._mode_box.currentText()
        proxy = _ScanParamSubmissionProxy(
            submission_state,
            entry_id=self._symbol_name,
            scan_group=self._normalised_scan_group(),
        )
        if mode == "Fixed":
            self._fixed_option.write_to_submission(proxy)
            return
        if mode == "Scan":
            self._scan_options[self._scan_kind_box.currentIndex()].write_to_submission(proxy)
            return
        if mode == "GPO scan" and self._gpo_option is not None:
            self._gpo_option.write_to_submission(proxy)
            return
        if mode == "Rebind":
            submission_state.add_rebind(
                fqn=self.schema["fqn"],
                path=self.path,
                expression=self._rebind_editor.expression(),
                entry_id=self._symbol_name,
            )
            return
        raise RuntimeError(f"Unsupported scan-submission row mode: {mode!r}")

    def _normalised_scan_group(self) -> str | None:
        text = self._group_box.text().strip()
        return text or None

    def symbol_name(self) -> str:
        return self._symbol_name

    def disable_scan(self) -> None:
        self._mode_box.setCurrentText("Fixed")

    def set_submission_mode(self, mode: str) -> None:
        if mode == self._submission_mode:
            return
        current = self._mode_box.currentText()
        self._submission_mode = mode
        self._rebuild_mode_box(preferred_mode=current)
        self._update_mode_ui()

    def _set_fixed_value(self, value) -> None:
        self._fixed_option.set_value(value)
        self._fixed_option.write_sync_values(self.sync_values)

    def _build_option(self, option_cls):
        option = option_cls(self.schema, self.path)
        option.value_changed.connect(self.value_changed)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        option.build_ui(layout)
        container.setLayout(layout)
        return _RowOptionWidget(option=option, container=container)

    def _active_option(self):
        if self._current_mode == "Scan" and self._scan_options:
            return self._scan_options[self._current_scan_option_idx]
        if self._current_mode == "GPO scan" and self._gpo_option is not None:
            return self._gpo_option
        if self._current_mode == "Rebind" and self._supports_rebind:
            return self._rebind_editor
        return self._fixed_option

    def _mode_changed(self, new_idx) -> None:
        del new_idx
        self._active_option().write_sync_values(self.sync_values)
        new_mode = self._mode_box.currentText()
        if new_mode == "Scan" and self._scan_options:
            self._scan_options[self._current_scan_option_idx].read_sync_values(
                self.sync_values
            )
            self._mode_stack.setCurrentWidget(self._scan_container)
        elif new_mode == "GPO scan" and self._gpo_option is not None:
            self._gpo_option.read_sync_values(self.sync_values)
            self._mode_stack.setCurrentWidget(self._gpo_option.container)
        elif new_mode == "Rebind" and self._supports_rebind:
            self._mode_stack.setCurrentWidget(self._rebind_editor.container)
        else:
            self._fixed_option.read_sync_values(self.sync_values)
            self._mode_stack.setCurrentWidget(self._fixed_option.container)
        self._current_mode = new_mode
        self._update_mode_ui()

    def _scan_kind_changed(self, new_idx) -> None:
        if not self._scan_options:
            return
        self._scan_options[self._current_scan_option_idx].write_sync_values(
            self.sync_values
        )
        self._scan_options[new_idx].read_sync_values(self.sync_values)
        self._scan_stack.setCurrentIndex(new_idx)
        self._current_scan_option_idx = new_idx

    def _update_mode_ui(self) -> None:
        is_scan = self._mode_box.currentText() == "Scan"
        self._group_container.setVisible(is_scan)
        if is_scan and self._scan_options:
            self._scan_stack.setCurrentIndex(self._current_scan_option_idx)

    def _allowed_mode_labels(self) -> list[str]:
        labels = ["Fixed"]
        if self._submission_mode == "grid" and self._scan_options:
            labels.append("Scan")
        if self._submission_mode == "gpo" and self._gpo_option is not None:
            labels.append("GPO scan")
        if self._supports_rebind:
            labels.append("Rebind")
        return labels

    def _rebuild_mode_box(self, preferred_mode: str | None = None) -> None:
        allowed = self._allowed_mode_labels()
        target = preferred_mode if preferred_mode in allowed else "Fixed"
        old_mode = self._current_mode
        with QtCore.QSignalBlocker(self._mode_box):
            self._mode_box.clear()
            self._mode_box.addItems(allowed)
            self._mode_box.setCurrentText(target)
        self._current_mode = old_mode
        self._mode_changed(0)


class ScanPseudoparamEntry(LayoutWidget):
    """Prepared-runtime pseudoparameter row editor.

    Pseudoparams are purely logical symbols, so this widget focuses on three pieces of
    state only:

    - the user-editable symbol id,
    - whether the symbol is fixed or scanned,
    - the finite scan generator and optional scan group if scanned.
    """

    value_changed = QtCore.pyqtSignal()

    def __init__(self, entry_id: str, *args, submission_mode: str = "grid"):
        super().__init__(*args)
        self.sync_values = {}
        self._last_valid_id = entry_id
        self._submission_mode = submission_mode

        self._schema = {
            "fqn": "__pseudoparam__",
            "type": "float",
            "default": "0.0",
            "spec": {},
        }
        self._fixed_option = self._build_option(get_scan_submission_fixed_option_type("float"))
        self._scan_option_names = []
        self._scan_options = []
        for name, option_cls in list_scan_submission_generator_option_types("float", True).items():
            self._scan_option_names.append(name)
            self._scan_options.append(self._build_option(option_cls))
        self._gpo_option = self._build_option(GpoBoundsScanOption)

        self._id_label = QtWidgets.QLabel("Id")
        self.addWidget(self._id_label, col=0)

        self._id_box = QtWidgets.QLineEdit(entry_id)
        self._id_box.setToolTip(
            "Stable identifier used by rebind expressions. Duplicates are suffixed "
            "deterministically on save."
        )
        self._id_box.setValidator(
            QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(r"[A-Za-z_][A-Za-z0-9_]*")
            )
        )
        self._id_box.textChanged.connect(self._id_text_changed)
        self.addWidget(self._id_box, col=1)

        self._mode_box = QtWidgets.QComboBox()
        self._mode_box.currentIndexChanged.connect(self._mode_changed)
        self.addWidget(self._mode_box, col=2)

        self._mode_stack = QtWidgets.QStackedWidget()
        self._mode_stack.addWidget(self._fixed_option.container)

        self._scan_container = QtWidgets.QWidget()
        self._scan_layout = QtWidgets.QHBoxLayout()
        self._scan_layout.setContentsMargins(0, 0, 0, 0)
        self._scan_container.setLayout(self._scan_layout)

        self._scan_kind_box = QtWidgets.QComboBox()
        for name in self._scan_option_names:
            self._scan_kind_box.addItem(name)
        self._scan_kind_box.currentIndexChanged.connect(self._scan_kind_changed)
        self._scan_layout.addWidget(self._scan_kind_box)

        self._scan_stack = QtWidgets.QStackedWidget()
        for option in self._scan_options:
            self._scan_stack.addWidget(option.container)
        self._scan_layout.addWidget(self._scan_stack, stretch=1)
        self._scan_kind_box.setVisible(len(self._scan_options) > 1)

        self._mode_stack.addWidget(self._scan_container)
        self._mode_stack.addWidget(self._gpo_option.container)
        self.addWidget(self._mode_stack, col=3)

        self._group_container = LayoutWidget()
        self._group_container.layout.setContentsMargins(6, 0, 0, 0)
        self._group_label = QtWidgets.QLabel("Group")
        self._group_label.setToolTip(
            "Scan rows in the same non-empty group are zipped together. "
            "Different groups remain Cartesian."
        )
        self._group_box = QtWidgets.QLineEdit()
        self._group_box.setPlaceholderText("optional")
        self._group_box.setMaximumWidth(110)
        self._group_box.setToolTip(self._group_label.toolTip())
        self._group_box.textChanged.connect(lambda *_: self.value_changed.emit())
        self._group_container.addWidget(self._group_label, col=0)
        self._group_container.addWidget(self._group_box, col=1)
        self.addWidget(self._group_container, col=4)

        self._current_mode = "Fixed"
        self._current_scan_option_idx = 0
        self._rebuild_mode_box()
        self._update_mode_ui()

    def read_from_entry(self, entry) -> None:
        self._set_identifier(entry.id)
        if entry.mode.type == "fixed":
            self._group_box.setText("")
            self._set_fixed_value(entry.mode.value)
            self.disable_scan()
            return

        if entry.mode.type == "scan":
            axis = {
                "type": entry.mode.generator.type,
                "range": dict(entry.mode.generator.range),
            }
            for idx, option in enumerate(self._scan_options):
                if option.attempt_read_from_axis(axis):
                    self._group_box.setText(entry.mode.group or "")
                    self._scan_kind_box.setCurrentIndex(idx)
                    self._mode_box.setCurrentText("Scan")
                    return
            logger.warning("Failed to read scan-submission pseudoparam scan params for %s", entry.id)
            return

        if entry.mode.type == "gpo_scan":
            axis = {"type": "gpo_scan", "lower": entry.mode.lower, "upper": entry.mode.upper}
            if self._gpo_option.attempt_read_from_axis(axis):
                self._group_box.setText("")
                self._mode_box.setCurrentText("GPO scan")
                return

    def write_to_submission(self, submission_state) -> None:
        proxy = _ScanPseudoparamSubmissionProxy(
            submission_state,
            entry_id=self.identifier(),
            scan_group=self._normalised_scan_group(),
        )
        mode = self._mode_box.currentText()
        if mode == "Fixed":
            self._fixed_option.write_to_submission(proxy)
            return
        if mode == "Scan":
            self._scan_options[self._scan_kind_box.currentIndex()].write_to_submission(proxy)
            return
        if mode == "GPO scan":
            self._gpo_option.write_to_submission(proxy)
            return
        raise RuntimeError(f"Unsupported scan-submission pseudoparam mode: {mode!r}")

    def disable_scan(self) -> None:
        self._mode_box.setCurrentText("Fixed")

    def set_submission_mode(self, mode: str) -> None:
        if mode == self._submission_mode:
            return
        current = self._mode_box.currentText()
        self._submission_mode = mode
        self._rebuild_mode_box(preferred_mode=current)
        self._update_mode_ui()

    def identifier(self) -> str:
        text = self._id_box.text().strip()
        if text.isidentifier():
            return text
        return self._last_valid_id

    def _set_identifier(self, text: str) -> None:
        self._last_valid_id = text
        self._id_box.setText(text)

    def _id_text_changed(self, text: str) -> None:
        if text.isidentifier():
            self._last_valid_id = text
        self.value_changed.emit()

    def _set_fixed_value(self, value) -> None:
        self._fixed_option.set_value(value)
        self._fixed_option.write_sync_values(self.sync_values)

    def _build_option(self, option_cls):
        option = option_cls(self._schema, "")
        option.value_changed.connect(self.value_changed)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        option.build_ui(layout)
        container.setLayout(layout)
        return _RowOptionWidget(option=option, container=container)

    def _mode_changed(self, new_idx) -> None:
        del new_idx
        if self._current_mode == "Scan" and self._scan_options:
            self._scan_options[self._current_scan_option_idx].write_sync_values(
                self.sync_values
            )
        elif self._current_mode == "GPO scan":
            self._gpo_option.write_sync_values(self.sync_values)
        else:
            self._fixed_option.write_sync_values(self.sync_values)

        new_mode = self._mode_box.currentText()
        if new_mode == "Scan":
            self._scan_options[self._current_scan_option_idx].read_sync_values(
                self.sync_values
            )
            self._mode_stack.setCurrentWidget(self._scan_container)
        elif new_mode == "GPO scan":
            self._gpo_option.read_sync_values(self.sync_values)
            self._mode_stack.setCurrentWidget(self._gpo_option.container)
        else:
            self._fixed_option.read_sync_values(self.sync_values)
            self._mode_stack.setCurrentWidget(self._fixed_option.container)
        self._current_mode = new_mode
        self._update_mode_ui()

    def _scan_kind_changed(self, new_idx) -> None:
        self._scan_options[self._current_scan_option_idx].write_sync_values(
            self.sync_values
        )
        self._scan_options[new_idx].read_sync_values(self.sync_values)
        self._scan_stack.setCurrentIndex(new_idx)
        self._current_scan_option_idx = new_idx

    def _update_mode_ui(self) -> None:
        is_scan = self._mode_box.currentText() == "Scan"
        self._group_container.setVisible(is_scan)
        if is_scan:
            self._scan_stack.setCurrentIndex(self._current_scan_option_idx)

    def _allowed_mode_labels(self) -> list[str]:
        labels = ["Fixed"]
        if self._submission_mode == "grid":
            labels.append("Scan")
        if self._submission_mode == "gpo":
            labels.append("GPO scan")
        return labels

    def _rebuild_mode_box(self, preferred_mode: str | None = None) -> None:
        allowed = self._allowed_mode_labels()
        target = preferred_mode if preferred_mode in allowed else "Fixed"
        old_mode = self._current_mode
        with QtCore.QSignalBlocker(self._mode_box):
            self._mode_box.clear()
            self._mode_box.addItems(allowed)
            self._mode_box.setCurrentText(target)
        self._current_mode = old_mode
        self._mode_changed(0)

    def _normalised_scan_group(self) -> str | None:
        text = self._group_box.text().strip()
        return text or None


class _RowOptionWidget:
    """Small wrapper so row code can treat fixed and scan widgets uniformly."""

    def __init__(self, *, option, container):
        self.option = option
        self.container = container

    def __getattr__(self, name):
        return getattr(self.option, name)


class _ScanRebindEditor:
    """Simple line-edit based scan-submission rebind editor.

    The actual expression parsing/semantic validation lives in the worker-side
    expression compiler. The dashboard keeps the widget intentionally lightweight and
    only ensures that a non-empty string is transported.
    """

    def __init__(self, *, symbol_name: str):
        self.container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.container.setLayout(layout)

        self.box = QtWidgets.QLineEdit()
        self.box.setText("0.0")
        self.box.setPlaceholderText("0.0")
        self.box.setToolTip(
            "Arithmetic expression compiled into a runtime parameter mapping. "
            "Refer to other rows by their stable ids, e.g. x or detuning. "
            f"This row's id is {symbol_name!r}."
        )
        layout.addWidget(self.box)

    def expression(self) -> str:
        text = self.box.text().strip()
        return text or "0.0"

    def set_expression(self, expression: str) -> None:
        self.box.setText(expression)

    def read_sync_values(self, sync_values: dict) -> None:
        del sync_values

    def write_sync_values(self, sync_values: dict) -> None:
        del sync_values


class _ScanParamSubmissionProxy:
    """Inject scan-submission row metadata into existing row-widget serialisation calls."""

    def __init__(self, submission_state, *, entry_id: str, scan_group: str | None):
        self._submission_state = submission_state
        self._entry_id = entry_id
        self._scan_group = scan_group

    def add_override(self, *, fqn: str, path: str, value):
        self._submission_state.add_override(
            fqn=fqn,
            path=path,
            value=value,
            entry_id=self._entry_id,
        )

    def add_scan_axis(
        self,
        *,
        fqn: str,
        path: str,
        axis_type: str,
        axis_range,
    ):
        self._submission_state.add_scan_axis(
            fqn=fqn,
            path=path,
            axis_type=axis_type,
            axis_range=axis_range,
            entry_id=self._entry_id,
            scan_group=self._scan_group,
        )

    def add_gpo_axis(self, *, fqn: str, path: str, lower, upper):
        self._submission_state.add_gpo_axis(
            fqn=fqn,
            path=path,
            lower=lower,
            upper=upper,
            entry_id=self._entry_id,
        )


class _ScanPseudoparamSubmissionProxy:
    """Route existing scan-option serialisation calls into pseudoparam state."""

    def __init__(self, submission_state, *, entry_id: str, scan_group: str | None):
        self._submission_state = submission_state
        self._entry_id = entry_id
        self._scan_group = scan_group

    def add_override(self, *, fqn: str, path: str, value):
        del fqn, path
        self._submission_state.add_pseudoparam_fixed(
            entry_id=self._entry_id,
            value=value,
        )

    def add_scan_axis(
        self,
        *,
        fqn: str,
        path: str,
        axis_type: str,
        axis_range,
    ):
        del fqn, path
        self._submission_state.add_pseudoparam_scan(
            entry_id=self._entry_id,
            axis_type=axis_type,
            axis_range=axis_range,
            scan_group=self._scan_group,
        )

    def add_gpo_axis(self, *, fqn: str, path: str, lower, upper):
        del fqn, path
        self._submission_state.add_pseudoparam_gpo_axis(
            entry_id=self._entry_id,
            lower=lower,
            upper=upper,
        )

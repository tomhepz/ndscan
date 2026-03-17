"""Handling of dashboard override type widgets.

Notably, this is _not_ related to the list of global "Scan options" at the bottom of
the argument editor (as mirrored by ndscan.experiment.scan_generator).

The legacy and host-runtime dashboard submission paths share most of the fixed-value
widgets, but the host runtime currently only supports *finite* grid generators from
the dashboard.  The host-specific option list therefore reuses the same visual idioms
without exposing the legacy "infinitely refining" generator types that the typed host
schema does not yet compile.
"""

import logging
from collections import OrderedDict
from enum import Enum, unique
from typing import Any

from artiq.gui.scientific_spinbox import ScientificSpinBox
from artiq.gui.tools import disable_scroll_wheel
from sipyco import pyon

from .._qt import QtCore, QtGui, QtWidgets
from .utils import format_override_identity, load_icon_cached

logger = logging.getLogger(__name__)


def parse_list_pyon(values: str) -> list[float]:
    return pyon.decode("[" + values + "]")


def make_divider():
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.Shape.VLine)
    f.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    f.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding
    )
    return f


@unique
class SyncValue(Enum):
    """Equivalent values to be synchronised between similar scan types.

    Not all values will have a meaning for all scan types; they should just be left
    alone so that arguments for like scans are synchronised between each other.
    """

    centre = "centre"
    lower = "lower"
    upper = "upper"
    num_points = "num_points"


class ScanOption(QtCore.QObject):
    """One "line" of scan options (the widgets specific to it, not including the
    selection box), and the code for serialising/deserialising it to the scan schema,
    plus synchronisation of the SyncValues between options.
    """

    value_changed = QtCore.pyqtSignal()

    def __init__(self, schema: dict[str, Any], path: str):
        super().__init__()
        self.schema = schema
        self.path = path

    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        raise NotImplementedError

    def write_to_submission(self, submission) -> None:
        raise NotImplementedError

    def read_sync_values(self, sync_values: dict) -> None:
        pass

    def write_sync_values(self, sync_values: dict) -> None:
        pass

    def attempt_read_from_axis(self, axis: dict) -> bool:
        return False

    def make_randomise_box(self):
        box = QtWidgets.QCheckBox()
        box.setToolTip("Randomise scan point order")
        box.setIcon(load_icon_cached("media-playlist-shuffle-32.svg"))
        box.setChecked(True)
        box.stateChanged.connect(self.value_changed)
        return box


class StringFixedScanOption(ScanOption):
    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box = QtWidgets.QLineEdit()
        layout.addWidget(self.box)

    def write_to_submission(self, submission) -> None:
        submission.add_override(
            fqn=self.schema["fqn"],
            path=self.path,
            value=self.box.text(),
        )

    def set_value(self, value) -> None:
        self.box.setText(value)


class BoolFixedScanOption(ScanOption):
    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box = QtWidgets.QCheckBox()
        layout.addWidget(self.box)

    def write_to_submission(self, submission) -> None:
        submission.add_override(
            fqn=self.schema["fqn"],
            path=self.path,
            value=self.box.isChecked(),
        )

    def set_value(self, value) -> None:
        self.box.setChecked(value)


class EnumFixedScanOption(ScanOption):
    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box = QtWidgets.QComboBox()
        self._members = self.schema["spec"]["members"]
        self._member_values_to_keys = {val: key for key, val in self._members.items()}
        self.box.addItems(self._members.values())
        layout.addWidget(self.box)

    def write_to_submission(self, submission) -> None:
        submission.add_override(
            fqn=self.schema["fqn"],
            path=self.path,
            value=self._member_values_to_keys[self.box.currentText()],
        )

    def set_value(self, value) -> None:
        try:
            text = self._members[value]
        except KeyError:
            text = next(iter(self._members.values()))
            identity = format_override_identity(self.schema["fqn"], self.path)
            logger.warning(
                f"Stored value '{value}' not in schema for enum parameter "
                f"'{identity}', setting to '{text}'"
            )
        self.box.setCurrentText(text)


class NumericScanOption(ScanOption):
    def __init__(self, schema: dict[str, Any], path: str):
        super().__init__(schema, path)
        spec = schema.get("spec", {})
        self.scale = spec.get("scale", 1.0)
        self.min = spec.get("min", float("-inf"))
        self.max = spec.get("max", float("inf"))

    def _make_spin_box(self, set_limits_from_spec=True):
        box = ScientificSpinBox()
        disable_scroll_wheel(box)
        box.valueChanged.connect(self.value_changed)

        spec = self.schema.get("spec", {})
        step = spec.get("step", 1.0)

        box.setDecimals(8)
        # setPrecision() was renamed in ARTIQ 8.
        if hasattr(box, "setPrecision"):
            box.setPrecision()
        else:
            box.setSigFigs()
        box.setSingleStep(step / self.scale)
        box.setRelativeStep()

        if set_limits_from_spec:
            box.setMinimum(self.min / self.scale)
            box.setMaximum(self.max / self.scale)

        unit = spec.get("unit", "")
        if unit:
            box.setSuffix(" " + unit)
        return box


class FixedScanOption(NumericScanOption):
    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box = self._make_spin_box()
        layout.addWidget(self.box)

    def write_to_submission(self, submission) -> None:
        submission.add_override(
            fqn=self.schema["fqn"],
            path=self.path,
            value=self.box.value() * self.scale,
        )

    def set_value(self, value) -> None:
        if value is None:
            # Error evaluating defaults, no better guess.
            value = 0.0
        self.box.setValue(float(value) / self.scale)

    def read_sync_values(self, sync_values: dict) -> None:
        if SyncValue.centre in sync_values:
            self.box.setValue(sync_values[SyncValue.centre])

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.centre] = self.box.value()


class RangeScanOption(NumericScanOption):
    """Base class for different ways of specifying scans across a given numerical
    range.
    """

    def _make_inf_points_box(self):
        box = QtWidgets.QCheckBox()
        box.setToolTip("Infinitely refine scan grid")
        box.setText("∞")
        box.setChecked(True)
        box.stateChanged.connect(self.value_changed)
        return box

    def _build_points_ui(self, layout):
        self.check_infinite = self._make_inf_points_box()
        layout.addWidget(self.check_infinite)
        layout.setStretchFactor(self.check_infinite, 0)

        self.box_points = QtWidgets.QSpinBox()
        self.box_points.setMinimum(2)
        self.box_points.setValue(21)

        # Somewhat gratuitously restrict the number of scan points for sizing, and to
        # avoid the user accidentally pasting in millions of points, etc.
        self.box_points.setMaximum(0xFFFF)

        self.box_points.setSuffix(" pts")
        layout.addWidget(self.box_points)
        layout.setStretchFactor(self.box_points, 0)

        self.check_infinite.setChecked(True)
        self.box_points.setEnabled(False)
        self.check_infinite.stateChanged.connect(
            lambda *_: self.box_points.setEnabled(not self.check_infinite.isChecked())
        )

        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 0)

    def write_to_submission(self, submission) -> None:
        axis_range = {
            "randomise_order": self.check_randomise.isChecked(),
        }
        axis_type = self.write_type_and_range(axis_range)
        submission.add_scan_axis(
            fqn=self.schema["fqn"],
            path=self.path,
            axis_type=axis_type,
            axis_range=axis_range,
        )


class MinMaxScanOption(RangeScanOption):
    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box_start = self._make_spin_box()
        layout.addWidget(self.box_start)
        layout.setStretchFactor(self.box_start, 1)

        layout.addWidget(make_divider())

        self._build_points_ui(layout)

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

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] == "refining":
            self.check_infinite.setChecked(True)
            self.box_start.setValue(axis["range"].get("lower", 0.0) / self.scale)
            self.box_stop.setValue(axis["range"].get("upper", 0.0) / self.scale)
            self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
            return True
        if axis["type"] == "linear":
            self.check_infinite.setChecked(False)
            self.box_start.setValue(axis["range"].get("start", 0.0) / self.scale)
            self.box_stop.setValue(axis["range"].get("stop", 0.0) / self.scale)
            self.box_points.setValue(axis["range"].get("num_points", 21))
            self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
            return True
        return False

    def write_type_and_range(self, axis_range: dict) -> str:
        start = self.box_start.value()
        stop = self.box_stop.value()
        if self.check_infinite.isChecked():
            axis_range |= {
                "lower": start * self.scale,
                "upper": stop * self.scale,
            }
            return "refining"

        axis_range |= {
            "start": start * self.scale,
            "stop": stop * self.scale,
            "num_points": self.box_points.value(),
        }
        return "linear"


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


class CentreSpanScanOption(RangeScanOption):
    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box_centre = self._make_spin_box()
        layout.addWidget(self.box_centre)
        layout.setStretchFactor(self.box_centre, 1)

        self.plusminus = QtWidgets.QLabel("±")
        layout.addWidget(self.plusminus)
        layout.setStretchFactor(self.plusminus, 0)

        self.box_half_span = self._make_spin_box(set_limits_from_spec=False)
        layout.addWidget(self.box_half_span)
        layout.setStretchFactor(self.box_half_span, 1)

        layout.addWidget(make_divider())

        self._build_points_ui(layout)

    def read_sync_values(self, sync_values: dict) -> None:
        if SyncValue.centre in sync_values:
            self.box_centre.setValue(sync_values[SyncValue.centre])
        if SyncValue.num_points in sync_values:
            self.box_points.setValue(sync_values[SyncValue.num_points])

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.centre] = self.box_centre.value()
        sync_values[SyncValue.num_points] = self.box_points.value()

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] == "centre_span_refining":
            self.check_infinite.setChecked(True)
        elif axis["type"] == "centre_span":
            self.check_infinite.setChecked(False)
        else:
            return False

        # Common to both finite/refining:
        self.box_half_span.setValue((axis["range"].get("half_span", 0.0) / self.scale))
        self.box_centre.setValue((axis["range"].get("centre", 0.0) / self.scale))
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True

    def write_type_and_range(self, axis_range: dict) -> str:
        centre = self.box_centre.value()
        half_span = self.box_half_span.value()
        axis_range |= {
            "centre": centre * self.scale,
            "half_span": half_span * self.scale,
            "limit_lower": self.min,
            "limit_upper": self.max,
        }
        if self.check_infinite.isChecked():
            return "centre_span_refining"

        axis_range["num_points"] = self.box_points.value()
        return "centre_span"


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


class ExpandingScanOption(NumericScanOption):
    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.box_centre = self._make_spin_box()
        layout.addWidget(self.box_centre)
        layout.setStretchFactor(self.box_centre, 1)

        layout.addWidget(make_divider())

        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 0)

        layout.addWidget(make_divider())

        self.box_spacing = self._make_spin_box()
        self.box_spacing.setSuffix(self.box_spacing.suffix() + " steps")
        layout.addWidget(self.box_spacing)
        layout.setStretchFactor(self.box_spacing, 1)

    def write_to_submission(self, submission) -> None:
        submission.add_scan_axis(
            fqn=self.schema["fqn"],
            path=self.path,
            axis_type="expanding",
            axis_range={
                "centre": self.box_centre.value() * self.scale,
                "spacing": self.box_spacing.value() * self.scale,
                "randomise_order": self.check_randomise.isChecked(),
                "limit_lower": self.min,
                "limit_upper": self.max,
            },
        )

    def read_sync_values(self, sync_values: dict) -> None:
        if SyncValue.centre in sync_values:
            self.box_centre.setValue(sync_values[SyncValue.centre])

    def write_sync_values(self, sync_values: dict) -> None:
        sync_values[SyncValue.centre] = self.box_centre.value()

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] != "expanding":
            return False
        self.box_centre.setValue(axis["range"].get("centre", 0.0) / self.scale)
        self.box_spacing.setValue(axis["range"].get("spacing", 0.0) / self.scale)
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True


class ListScanOption(NumericScanOption):
    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        class Validator(QtGui.QValidator):
            def validate(self, input, pos):
                try:
                    [float(f) for f in parse_list_pyon(input)]
                    return QtGui.QValidator.State.Acceptable, input, pos
                except Exception:
                    return QtGui.QValidator.State.Intermediate, input, pos

        self.box_pyon = QtWidgets.QLineEdit()
        self.box_pyon.setValidator(Validator(self))
        layout.addWidget(self.box_pyon)

        layout.addWidget(make_divider())

        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 0)

    def write_to_submission(self, submission) -> None:
        try:
            values = [v * self.scale for v in parse_list_pyon(self.box_pyon.text())]
        except Exception as e:
            logger.info(e)
            values = []
        submission.add_scan_axis(
            fqn=self.schema["fqn"],
            path=self.path,
            axis_type="list",
            axis_range={
                "values": values,
                "randomise_order": self.check_randomise.isChecked(),
            },
        )

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] != "list":
            return False
        values = axis["range"].get("values", [])
        list_str = ", ".join([str(v / self.scale) for v in values])
        self.box_pyon.setText(list_str)
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True


class BoolScanOption(ScanOption):
    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        dummy_box = QtWidgets.QCheckBox()
        dummy_box.setTristate()
        dummy_box.setEnabled(False)
        dummy_box.setChecked(True)
        layout.addWidget(dummy_box)
        layout.setStretchFactor(dummy_box, 0)
        layout.addWidget(make_divider())
        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 1)

    def write_to_submission(self, submission) -> None:
        submission.add_scan_axis(
            fqn=self.schema["fqn"],
            path=self.path,
            axis_type="list",
            axis_range={
                "values": [False, True],
                "randomise_order": self.check_randomise.isChecked(),
            },
        )

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] != "list":
            return False
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True


class EnumScanOption(ScanOption):
    def build_ui(self, layout: QtWidgets.QLayout) -> None:
        self.check_randomise = self.make_randomise_box()
        layout.addWidget(self.check_randomise)
        layout.setStretchFactor(self.check_randomise, 0)

    def write_to_submission(self, submission) -> None:
        submission.add_scan_axis(
            fqn=self.schema["fqn"],
            path=self.path,
            axis_type="list",
            axis_range={
                "values": list(self.schema["spec"]["members"].keys()),
                "randomise_order": self.check_randomise.isChecked(),
            },
        )

    def attempt_read_from_axis(self, axis: dict) -> bool:
        if axis["type"] != "list":
            return False
        self.check_randomise.setChecked(axis["range"].get("randomise_order", True))
        return True


def list_scan_option_types(
    schema_type: str, is_scannable: bool
) -> OrderedDict[str, type[ScanOption]]:
    """Return a list of scan option types appropriate for the given parameter.

    :param schema_type: The "type" field of the parameter schema.
    :param is_scannable: Whether to show non-Fixed options.
    :return: An ordered list of option labels mapping to the ScanOption subclass
        representing them.
    """
    result = OrderedDict([])
    if schema_type == "string":
        result["Fixed"] = StringFixedScanOption
    elif schema_type == "bool":
        result["Fixed"] = BoolFixedScanOption
        if is_scannable:
            result["Scanning"] = BoolScanOption
    elif schema_type == "enum":
        result["Fixed"] = EnumFixedScanOption
        if is_scannable:
            result["Scanning"] = EnumScanOption
    else:
        # TODO: Properly handle int, add errors (or default to PYON value).
        result["Fixed"] = FixedScanOption
        if is_scannable:
            result["Min./Max."] = MinMaxScanOption
            result["Centered"] = CentreSpanScanOption
            result["Expanding"] = ExpandingScanOption
            result["List"] = ListScanOption
    return result


def list_host_scan_option_types(
    schema_type: str, is_scannable: bool
) -> OrderedDict[str, type[ScanOption]]:
    """Return row option widgets for the editable host-runtime dashboard subset."""

    result = OrderedDict([])
    if schema_type == "string":
        result["Fixed"] = StringFixedScanOption
    elif schema_type == "bool":
        result["Fixed"] = BoolFixedScanOption
        if is_scannable:
            result["Scanning"] = BoolScanOption
    elif schema_type == "enum":
        result["Fixed"] = EnumFixedScanOption
        if is_scannable:
            result["Scanning"] = EnumScanOption
    else:
        result["Fixed"] = FixedScanOption
        if is_scannable:
            result["Min./Max."] = FiniteMinMaxScanOption
            result["Centered"] = FiniteCentreSpanScanOption
            result["Expanding"] = ExpandingScanOption
            result["List"] = ListScanOption
    return result

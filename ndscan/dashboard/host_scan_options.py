"""Host-runtime-specific scan-row widgets.

The host dashboard path currently supports only *finite* grid generators. These
widgets reuse the shared numeric option primitives from :mod:`scan_options` while
keeping the host-specific subset out of the legacy-oriented module.
"""

from __future__ import annotations

from collections import OrderedDict

from .._qt import QtWidgets
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

"""Prepared-runtime demo for the viewer's 1D ``group by`` control.

This example is intentionally shaped to make grouped 1D plotting obvious:

- ``time_us`` is the natural x-axis,
- ``detuning_mhz`` is a second scanned axis that is natural to group by,
- the saved ``signal`` channel forms a family of offset damped oscillations,
- points are globally randomised so all line families build up together live.

Suggested viewer setup:

1. keep the plot in ``1D`` mode,
2. choose ``time_us`` for ``x``,
3. choose ``signal`` for ``y``,
4. choose ``detuning_mhz`` for ``group by``.

That should give one coloured trace per detuning value rather than all points stacked
into one line.
"""

from __future__ import annotations

import time

import numpy as np

from artiq.experiment import *
from ndscan.define import *
from ndscan.runtime.api import *
from ndscan.scan import *


class GroupedLineFamilyFragment(ExpFragment):
    """Produce a small family of damped oscillation curves."""

    def build_fragment(self):
        self.setattr_param(
            "time_us",
            FloatParam,
            "Probe time",
            default=0.0,
            unit="us",
            min=0.0,
        )
        self.setattr_param(
            "detuning_mhz",
            FloatParam,
            "Detuning",
            default=0.0,
            unit="MHz",
        )
        self.setattr_result("signal", FloatChannel)

    def run_once(self):
        time_us = self.time_us.get()
        detuning_mhz = self.detuning_mhz.get()

        envelope = np.exp(-time_us / 8.0)
        phase = 1.55 * time_us + 0.9 * detuning_mhz
        baseline = 0.35 * detuning_mhz
        signal = baseline + envelope * np.cos(phase) + 0.12 * np.sin(0.45 * time_us)

        self.signal.push(float(signal))
        time.sleep(0.02)


PreparedScanGroupedLineFamily = make_fragment_prepared_scan_exp(
    GroupedLineFamilyFragment,
    lambda fragment: ScanRequest.cartesian(
        [
            (fragment.time_us, np.linspace(0.0, 14.0, 36).tolist()),
            (fragment.detuning_mhz, [-2.0, -1.0, 0.0, 1.0, 2.0]),
        ],
        execution_policy=ExecutionPolicy(max_points_per_batch=1),
        randomise_order_globally=True,
        random_seed=12345,
        metadata={"demo_name": "prepared_scan_grouped_line_family"},
    ),
)

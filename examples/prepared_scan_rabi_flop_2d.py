"""Prepared-runtime 2D Rabi-flop simulation.

This is the prepared-runtime counterpart to the legacy ``examples/rabi_flop.py``
example, intended for the runtime site-tree viewer:

- scan ``rabi_freq`` and ``duration`` as a 2D Cartesian grid,
- view the result as a 2D scatter plot with
  ``x = duration``, ``y = rabi_freq``, ``z = readout/p``.

The parameter values are stored directly in ``MHz`` / ``us`` units so they are easy to
read in the live plotter. The simulation converts them back to SI units internally
when evaluating the Rabi model.
"""

from __future__ import annotations

import random
import time
from enum import Enum, unique

import numpy as np
from oitg.errorbars import binom_onesided

from artiq.experiment import *
from ndscan.define import *
from ndscan.runtime.api import *
from ndscan.scan import *


class Readout(ExpFragment):
    """Simple thresholded bright/dark readout simulation."""

    def build_fragment(self):
        self.setattr_param(
            "num_shots", IntParam, "Number of shots", 100, is_scannable=False
        )
        self.setattr_param(
            "mean_0", FloatParam, "Dark counts over readout duration", 0.1
        )
        self.setattr_param(
            "mean_1", FloatParam, "Bright counts over readout duration", 20.0
        )
        self.setattr_param("threshold", IntParam, "Threshold", 5)

        self.setattr_result("counts", OpaqueChannel)
        self.setattr_result("p", FloatChannel)
        self.setattr_result(
            "p_err", FloatChannel, display_hints={"error_bar_for": self.p.path}
        )

    def simulate_shots(self, p: float) -> None:
        num_shots = self.num_shots.get()

        counts = np.empty(num_shots, dtype=np.int16)
        for i in range(num_shots):
            mean = self.mean_0.get() if random.random() > p else self.mean_1.get()
            counts[i] = np.random.poisson(mean)
        self.counts.push(counts)

        num_brights = np.sum(counts >= self.threshold.get())
        p, p_err = binom_onesided(num_brights, num_shots)
        self.p.push(p)
        self.p_err.push(p_err)


@unique
class InitialState(Enum):
    dark = "Dark"
    bright = "Bright"


class RabiFlopPreparedSim(ExpFragment):
    """Simulate a Rabi flop over frequency and pulse duration."""

    def build_fragment(self):
        self.setattr_fragment("readout", Readout)

        # Store values directly in MHz/us for more readable plotting.
        self.setattr_param(
            "rabi_freq",
            FloatParam,
            "Rabi frequency",
            1.0,
            unit="MHz",
            min=0.0,
        )
        self.setattr_param(
            "duration",
            FloatParam,
            "Pulse duration",
            0.5,
            unit="us",
            min=0.0,
        )
        self.setattr_param(
            "detuning",
            FloatParam,
            "Detuning",
            0.0,
            unit="MHz",
        )
        self.setattr_param(
            "initial_state", EnumParam, "Initial state", InitialState.bright
        )

    def run_once(self):
        omega0 = 2 * np.pi * self.rabi_freq.get() * MHz
        delta = 2 * np.pi * self.detuning.get() * MHz
        omega = np.sqrt(omega0**2 + delta**2)
        pulse_duration = self.duration.get() * us

        p = (omega0 / omega * np.sin(omega / 2 * pulse_duration)) ** 2
        if self.initial_state.get() == InitialState.bright:
            p = 1 - p

        self.readout.simulate_shots(float(p))
        time.sleep(0.001)


PreparedScanRabiFlop2D = make_fragment_prepared_scan_exp(
    RabiFlopPreparedSim,
    lambda fragment: ScanRequest.cartesian(
        [
            (fragment.rabi_freq, np.linspace(0.2, 1.8, 90).tolist()),
            (fragment.duration, np.linspace(0.0, 2.5, 130).tolist()),
        ],
        execution_policy=ExecutionPolicy(max_points_per_batch=8),
        randomise_order_globally=True,
        metadata={"demo_name": "prepared_scan_rabi_flop_2d"},
    ),
)

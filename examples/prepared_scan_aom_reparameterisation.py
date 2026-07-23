"""AMO-style reparameterisation examples for the prepared runtime.

This file shows the three intended ways to express a simple "scan laser frequency while
keeping laser intensity constant" workflow:

- ``PreparedScanAomMappedLogicalAxes`` uses ad hoc ``ScanVariable`` objects plus
  ``ParameterMapping`` in a code-first request.
- ``PreparedScanAomWrapperRebind`` uses a reusable wrapper fragment and
  ``rebind_param(...)``.
- ``PreparedScanAomDashboardRebind`` exposes only the hardware-facing parameters and is
  intended for the dashboard submission path, where pseudoparams and text rebinds can
  recreate the same logical scan interactively.
- ``LegacyAomManualWrapperScan`` shows the older pre-``rebind_param(...)`` style:
  scan wrapper parameters with the legacy runtime, manually compute the child RF
  values, and write them into overridden child parameter stores in ``run_once()``.

The hardware-facing fragment has two parameters:

- ``rf_amplitude``
- ``rf_frequency``

and publishes one result channel:

- ``scattering_rate``

The model is deliberately simple:

- the AOM diffraction efficiency has a quadratic roll-off away from a centre RF,
- the atomic transition is not at that centre RF,
- the AOM is double passed, so optical detuning is twice the RF detuning,
- the scattering rate is the realised laser intensity times a Lorentzian line shape.
"""

from __future__ import annotations

from artiq.experiment import *

from ndscan.define import *
from ndscan.runtime.api import *
from ndscan.scan import *

# --- MODEL ---

AOM_CENTRE_RF = 80.0
TRANSITION_OPTICAL_OFFSET = 12.0
MIN_DIFFRACTION_EFFICIENCY = 0.2
OPTICAL_DIFFRACTION_HALF_WIDTH = 30.0
TRANSITION_HALF_WIDTH = 8.0
DEFAULT_LASER_INTENSITY = 1.0
LASER_DETUNING_SCAN_MIN = -20.0
LASER_DETUNING_SCAN_MAX = 20.0
LASER_DETUNING_NUM_POINTS = 50
RESONANT_RF_FREQUENCY = AOM_CENTRE_RF + TRANSITION_OPTICAL_OFFSET / 2.0


def optical_shift_for_laser_detuning(laser_detuning: float) -> float:
    """Return the optical shift from the AOM centre for a requested laser detuning."""

    return TRANSITION_OPTICAL_OFFSET + laser_detuning


def diffraction_efficiency(optical_shift: float) -> float:
    """Return a simple clipped quadratic diffraction-efficiency model."""

    return max(
        MIN_DIFFRACTION_EFFICIENCY,
        1.0 - (optical_shift / OPTICAL_DIFFRACTION_HALF_WIDTH) ** 2,
    )


def rf_frequency_for_laser_detuning(laser_detuning: float) -> float:
    """Return the RF needed to realise the requested detuning from the transition."""

    return AOM_CENTRE_RF + optical_shift_for_laser_detuning(laser_detuning) / 2.0


def rf_amplitude_for_laser_settings(
    laser_intensity: float, laser_detuning: float
) -> float:
    """Compensate RF amplitude for the AOM diffraction-efficiency roll-off."""

    return laser_intensity / diffraction_efficiency(
        optical_shift_for_laser_detuning(laser_detuning)
    )


def scattering_rate_from_hardware(rf_amplitude: float, rf_frequency: float) -> float:
    """Evaluate the realised scattering rate for one hardware point."""

    optical_shift = 2.0 * (rf_frequency - AOM_CENTRE_RF)
    laser_detuning = optical_shift - TRANSITION_OPTICAL_OFFSET
    realised_laser_intensity = rf_amplitude * diffraction_efficiency(optical_shift)
    line_shape = 1.0 / (1.0 + (laser_detuning / TRANSITION_HALF_WIDTH) ** 2)
    return realised_laser_intensity * line_shape

# --- Fragments ---

class AomHardwareFragment(ExpFragment):
    """Hardware-facing fragment exposing only RF controls."""

    def build_fragment(self):
        self.setattr_param("rf_amplitude", FloatParam, "RF amplitude", 1.0)
        self.setattr_param(
            "rf_frequency", FloatParam, "RF frequency", RESONANT_RF_FREQUENCY
        )
        self.setattr_result("scattering_rate", FloatChannel)

    def get_always_shown_params(self):
        return [self.rf_amplitude, self.rf_frequency]

    def run_once(self):
        self.scattering_rate.push(
            scattering_rate_from_hardware(
                self.rf_amplitude.get(),
                self.rf_frequency.get(),
            )
        )


laser_detuning_axis = ScanVariable(
    "laser_detuning",
    description="Optical detuning from the transition after the double-passed AOM",
)

PreparedScanAomMappedLogicalAxes = make_fragment_prepared_scan_exp(
    AomHardwareFragment,
    lambda fragment: ScanRequest.linear(
        laser_detuning_axis,
        start=LASER_DETUNING_SCAN_MIN,
        stop=LASER_DETUNING_SCAN_MAX,
        num_points=LASER_DETUNING_NUM_POINTS
    ).with_parameter_mappings(
        [
            ParameterMapping.single_target(
                fragment.rf_frequency,
                [laser_detuning_axis],
                lambda values: rf_frequency_for_laser_detuning(
                    values[laser_detuning_axis]
                ),
                description="Map optical detuning to double-pass RF frequency",
            ),
            ParameterMapping.single_target(
                fragment.rf_amplitude,
                [laser_detuning_axis],
                lambda values: rf_amplitude_for_laser_settings(
                    DEFAULT_LASER_INTENSITY,
                    values[laser_detuning_axis],
                ),
                description="Keep laser intensity constant while compensating AOM efficiency",
            ),
        ]
    ),
)


class AomLogicalWrapperFragment(ExpFragment):
    """Wrapper fragment exposing logical AMO parameters above RF hardware controls."""

    def build_fragment(self):
        self.setattr_fragment("hardware", AomHardwareFragment)
        self.setattr_param("laser_intensity", FloatParam, "laser intensity", 1.0)
        self.setattr_param("laser_detuning", FloatParam, "laser detuning", 0.0)

        self.rebind_param(
            self.hardware.rf_frequency,
            [self.laser_detuning],
            lambda values: rf_frequency_for_laser_detuning(values[self.laser_detuning]),
            description="Map optical detuning to double-pass RF frequency",
        )
        self.rebind_param(
            self.hardware.rf_amplitude,
            [self.laser_intensity, self.laser_detuning],
            lambda values: rf_amplitude_for_laser_settings(
                values[self.laser_intensity],
                values[self.laser_detuning],
            ),
            description="Compensate RF amplitude for diffraction efficiency",
        )

    def get_always_shown_params(self):
        return [self.laser_intensity, self.laser_detuning]

    def run_once(self):
        self.hardware.run_once()


PreparedScanAomWrapperRebind = make_fragment_prepared_scan_exp(
    AomLogicalWrapperFragment,
    lambda fragment: ScanRequest.linear(
        fragment.laser_detuning,
        start=LASER_DETUNING_SCAN_MIN,
        stop=LASER_DETUNING_SCAN_MAX,
        num_points=LASER_DETUNING_NUM_POINTS,
        metadata={"demo_name": "prepared_scan_aom_reparameterisation_wrapper"},
    ),
)


class LegacyManualAomWrapperFragment(ExpFragment):
    """Legacy/manual wrapper equivalent from before ``rebind_param(...)`` existed.

    The old fragment API already supported simple parameter aliasing via
    ``bind_param()`` / ``setattr_param_rebind()``, but not a generic "per-point apply
    this transform to another parameter" mechanism.

    A historically accurate workaround is:

    - expose the logical parameters on the wrapper,
    - override the child's hardware-facing parameters to dedicated stores,
    - update those stores manually for each point before calling the child.

    This works with the legacy runtime, but the transformation logic is much more ad
    hoc than the prepared-runtime ``ParameterMapping`` path above.
    """

    def build_fragment(self):
        self.setattr_fragment("hardware", AomHardwareFragment)
        self.setattr_param("laser_intensity", FloatParam, "laser intensity", 1.0)
        self.setattr_param("laser_detuning", FloatParam, "laser detuning", 0.0)

        # Keep direct handles to manually controlled child stores.
        _, self._rf_amplitude_store = self.hardware.override_param("rf_amplitude", 1.0)
        _, self._rf_frequency_store = self.hardware.override_param(
            "rf_frequency", RESONANT_RF_FREQUENCY
        )

    def get_always_shown_params(self):
        return [self.laser_intensity, self.laser_detuning]

    def run_once(self):
        self._rf_frequency_store.set_value(
            rf_frequency_for_laser_detuning(self.laser_detuning.get())
        )
        self._rf_amplitude_store.set_value(
            rf_amplitude_for_laser_settings(
                self.laser_intensity.get(),
                self.laser_detuning.get(),
            )
        )
        self.hardware.run_once()


# LegacyAomManualWrapperScan = make_fragment_scan_exp(LegacyManualAomWrapperFragment)


"""
To reproduce the logical scan from the dashboard:

1. Open ``PreparedScanAomDashboardRebind`` in the ARTIQ dashboard.
2. Add two pseudoparams named ``laser_intensity`` and ``laser_detuning``.
3. Set ``laser_intensity`` to ``Fixed`` at the desired constant intensity.
4. Set ``laser_detuning`` to ``Scan`` over the desired frequency sweep.
5. Set ``rf_frequency`` to ``Rebind`` with:

   ``80.0 + (12.0 + laser_detuning) / 2.0``

6. Set ``rf_amplitude`` to ``Rebind`` with:

   ``laser_intensity / max(0.2, 1.0 - (((12.0 + laser_detuning) / 30.0) ** 2))``

Those expressions match the helper functions used by the code-first examples above.
"""
PreparedScanAomDashboardRebind = make_fragment_prepared_dashboard_scan_exp(
    AomHardwareFragment
)

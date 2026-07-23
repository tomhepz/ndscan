"""Examples of runtime parameter mappings in the new prepared runtime.

This file shows the two intended usage styles for reparameterisation:

- ad hoc code-first scans using ``ScanVariable`` plus ``ParameterMapping``,
- reusable wrapper fragments using ``rebind_param(...)``.

Both routes end up in the same prepared-runtime execution path. The runtime installs any
directly scanned parameter values first, evaluates parameter mappings second, and only
then runs the point body.
"""

from artiq.experiment import *

from ndscan.define import *
from ndscan.runtime.api import *
from ndscan.scan import *


class HardwareDriveFragment(ExpFragment):
    """Small hardware-facing fragment exposing only the physical drive parameter."""

    def build_fragment(self):
        self.setattr_param("drive", FloatParam, "drive", 0.0)
        self.setattr_result("result", FloatChannel)

    def run_once(self):
        self.result.push(2.0 * self.drive.get())


logical_drive = ScanVariable(
    "logical_drive",
    description="Logical scan axis mapped to the physical drive",
)


PreparedScanMappedLogicalAxis = make_fragment_prepared_scan_exp(
    HardwareDriveFragment,
    lambda fragment: ScanRequest.cartesian(
        [(logical_drive, [0.0, 1.0, 2.0, 3.0])]
    ).with_parameter_mappings(
        [
            ParameterMapping.single_target(
                fragment.drive,
                [logical_drive],
                lambda values: values[logical_drive] + 0.5,
                description="Offset the physical drive from the logical axis",
            )
        ]
    ),
)

class WrapperMappedDriveFragment(ExpFragment):
    """Wrapper fragment exposing a logical parameter while driving a child fragment.

    This is the natural composition pattern for a simple reparameterising wrapper:

    - keep the child attached to the normal fragment lifecycle,
    - rebind one of the child's hardware-facing parameters from a wrapper parameter,
    - let the child's own result channel appear in the scan site with its deeper path.

    The wrapper does not need to mirror the child's result into a second parent-owned
    channel unless it wants to publish an intentionally different summary value.
    """

    def build_fragment(self):
        self.setattr_fragment("hardware", HardwareDriveFragment)
        self.setattr_param("logical_drive", FloatParam, "logical drive", 0.0)
        self.rebind_param(
            self.hardware.drive,
            [self.logical_drive],
            lambda values: values[self.logical_drive] + 0.5,
            description="Offset child drive from the wrapper's logical axis",
        )

    def run_once(self):
        self.hardware.run_once()


PreparedScanWrapperRebind = make_fragment_prepared_scan_exp(
    WrapperMappedDriveFragment,
    lambda fragment: ScanRequest.cartesian(
        [(fragment.logical_drive, [0.0, 1.0, 2.0, 3.0])]
    ),
)

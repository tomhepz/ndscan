"""Kernel-streaming examples for pseudoparams and wrapper rebinds.

These are the kernel-capable counterparts to ``host_runtime_parameter_mapping.py``.

The important property is that the kernel loop still has a fixed compiled shape:

- the host chooses batches,
- the host resolves pseudoparams and parameter mappings into concrete parameter values,
- the resident kernel only receives the concrete installed values it needs.

That means logical scan features such as ``ScanVariable`` and ``rebind_param(...)`` can
work without forcing repeated kernel compilation.

This file shows three closely related cases:

- ad hoc pseudoparam -> parameter mapping,
- ad hoc parameter -> parameter mapping,
- wrapper-fragment ``rebind_param(...)``.
"""

from __future__ import annotations

from ndscan.experiment import (
    ExecutionPolicy,
    ExpFragment,
    FloatChannel,
    FloatParam,
    ParameterMapping,
    ScanRequest,
    ScanVariable,
    kernel,
    make_fragment_host_scan_exp,
)


class KernelHardwareDriveFragment(ExpFragment):
    """Small hardware-facing fragment exposing only the physical drive parameter."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("drive", FloatParam, "drive", 0.0)
        self.setattr_result("result", FloatChannel)

    @kernel
    def run_once(self):
        self.result.push(2.0 * self.drive.get())


class KernelLogicalAndPhysicalDriveFragment(ExpFragment):
    """Fragment exposing both a logical and a hardware-facing parameter."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("logical_drive", FloatParam, "logical drive", 0.0)
        self.setattr_param("drive", FloatParam, "drive", 0.0)
        self.setattr_result("result", FloatChannel)

    @kernel
    def run_once(self):
        self.result.push(2.0 * self.drive.get())


logical_drive = ScanVariable(
    "logical_drive",
    description="Logical scan axis mapped to the physical drive",
)


HostRuntimeKernelMappedLogicalAxis = make_fragment_host_scan_exp(
    KernelHardwareDriveFragment,
    lambda fragment: ScanRequest.cartesian(
        [(logical_drive, [0.0, 1.0, 2.0, 3.0])],
        execution_policy=ExecutionPolicy(max_points_per_batch=2),
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


HostRuntimeKernelMappedParamAxis = make_fragment_host_scan_exp(
    KernelLogicalAndPhysicalDriveFragment,
    lambda fragment: ScanRequest.cartesian(
        [(fragment.logical_drive, [0.0, 1.0, 2.0, 3.0])],
        execution_policy=ExecutionPolicy(max_points_per_batch=2),
    ).with_parameter_mappings(
        [
            ParameterMapping.single_target(
                fragment.drive,
                [fragment.logical_drive],
                lambda values: values[fragment.logical_drive] + 0.5,
                description="Offset the physical drive from the logical parameter",
            )
        ]
    ),
)


class KernelWrapperMappedDriveFragment(ExpFragment):
    """Wrapper fragment exposing a logical parameter while driving a kernel child."""

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_fragment("hardware", KernelHardwareDriveFragment)
        self.setattr_param("logical_drive", FloatParam, "logical drive", 0.0)
        self.rebind_param(
            self.hardware.drive,
            [self.logical_drive],
            lambda values: values[self.logical_drive] + 0.5,
            description="Offset child drive from the wrapper logical axis",
        )

    @kernel
    def run_once(self):
        self.hardware.run_once()


HostRuntimeKernelWrapperRebind = make_fragment_host_scan_exp(
    KernelWrapperMappedDriveFragment,
    lambda fragment: ScanRequest.cartesian(
        [(fragment.logical_drive, [0.0, 1.0, 2.0, 3.0])],
        execution_policy=ExecutionPolicy(max_points_per_batch=2),
    ),
)

"""Public scan request and execution-policy types.

These are scan-semantics objects, not runtime-internal program objects:

- ``ScanRequest`` describes what to scan,
- ``ExecutionPolicy`` describes batch/preview scheduling preferences,
- ``PreviewPolicy`` describes preview HDF5 cadence and output path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from artiq.language import HasEnvironment

from ..define.parameters import ParamHandle
from ..schema.scan_site import ScanSite
from .mapping import FixedPseudoparam, ParameterMapping, ScanVariable
from .point_policy import (
    CartesianPointPolicy,
    ExplicitPointPolicy,
    PointPolicy,
    RepeatPointPolicy,
    ShuffledPointPolicy,
    SinglePointPolicy,
    ZipPointPolicy,
)

__all__ = [
    "PreviewPolicy",
    "ExecutionPolicy",
    "ScanRequest",
]


@dataclass(frozen=True)
class PreviewPolicy:
    """Configuration for periodic preview HDF5 snapshots.

    Preview files are root-run state. Nested scans inherit the root preview coordinator
    so a dashboard or applet sees one coherent HDF5 snapshot rather than separate files
    for each child scan.
    """

    path: str | None = None
    min_interval_s: float = 120.0
    write_on_completion: bool = False
    remove_on_completion: bool = True

    def __post_init__(self) -> None:
        if self.min_interval_s < 0.0:
            raise ValueError("min_interval_s must be non-negative")

    def resolve_path(self, owner: HasEnvironment) -> str:
        if self.path is not None:
            return self.path
        scheduler = owner.get_device("scheduler")
        rid = getattr(scheduler, "rid", 0)
        return f"{rid:09d}-{owner.__class__.__name__}.preview.h5"


@dataclass(frozen=True)
class ExecutionPolicy:
    """Prepared-runtime scheduling and flush policy for one scan request.

    This affects how points are grouped and previewed, not which points exist. The
    point policy still owns the scan trajectory; the runtime may ask for several points
    at a time to reduce host/kernel boundary traffic.
    """

    max_points_per_batch: int | None = None
    preview_policy: PreviewPolicy | None = None

    def __post_init__(self) -> None:
        if self.max_points_per_batch is not None and self.max_points_per_batch <= 0:
            raise ValueError("max_points_per_batch must be positive when specified")
        if self.preview_policy is not None and not isinstance(
            self.preview_policy, PreviewPolicy
        ):
            raise TypeError("preview_policy must be a PreviewPolicy instance")


@dataclass(frozen=True)
class ScanRequest:
    """User-facing prepared-runtime scan request.

    A request is declarative. It names the logical axes, supplies a point policy, and
    optionally adds parameter mappings, metadata, execution hints, and a scan-site
    location. The runtime clones the request before execution because point policies can
    carry mutable progress state.
    """

    axes: tuple[ParamHandle | ScanVariable, ...]
    point_policy: PointPolicy
    site: ScanSite = field(default_factory=ScanSite)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    parameter_mappings: tuple[ParameterMapping, ...] = ()
    fixed_pseudoparams: tuple[FixedPseudoparam, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.execution_policy, ExecutionPolicy):
            raise TypeError("execution_policy must be an ExecutionPolicy instance")
        for mapping in self.parameter_mappings:
            if not isinstance(mapping, ParameterMapping):
                raise TypeError(
                    "parameter_mappings must contain ParameterMapping instances"
                )
        for pseudoparam in self.fixed_pseudoparams:
            if not isinstance(pseudoparam, FixedPseudoparam):
                raise TypeError(
                    "fixed_pseudoparams must contain FixedPseudoparam instances"
                )

    def with_site(self, site: ScanSite) -> "ScanRequest":
        return ScanRequest(
            axes=self.axes,
            point_policy=self.point_policy,
            site=site,
            metadata=self.metadata,
            execution_policy=self.execution_policy,
            parameter_mappings=self.parameter_mappings,
            fixed_pseudoparams=self.fixed_pseudoparams,
        )

    def with_parameter_mappings(
        self, parameter_mappings: Sequence[ParameterMapping]
    ) -> "ScanRequest":
        return ScanRequest(
            axes=self.axes,
            point_policy=self.point_policy,
            site=self.site,
            metadata=self.metadata,
            execution_policy=self.execution_policy,
            parameter_mappings=self.parameter_mappings + tuple(parameter_mappings),
            fixed_pseudoparams=self.fixed_pseudoparams,
        )

    def with_global_randomisation(
        self, *, random_seed: int | None = None
    ) -> "ScanRequest":
        return ScanRequest(
            axes=self.axes,
            point_policy=ShuffledPointPolicy(self.point_policy, random_seed=random_seed),
            site=self.site,
            metadata=self.metadata,
            execution_policy=self.execution_policy,
            parameter_mappings=self.parameter_mappings,
            fixed_pseudoparams=self.fixed_pseudoparams,
        )

    def with_repeats(
        self,
        *,
        repeats: int | None = None,
        stop_predicate: Callable[[Any], bool] | None = None,
        min_repeats: int = 1,
        max_repeats: int | None = None,
        predicate_description: str = "custom",
        schedule: str = "serial",
    ) -> "ScanRequest":
        """Wrap this request's point policy in a RepeatPointPolicy.

        This is convenience sugar for the common case where repetition is an execution
        concern layered on top of an otherwise normal scan request. It keeps the base
        scan shape expressed through the usual ``ScanRequest`` helpers, and then applies
        repeated execution as a modifier afterwards.
        """

        return ScanRequest(
            axes=self.axes,
            point_policy=RepeatPointPolicy(
                self.point_policy,
                repeats=repeats,
                stop_predicate=stop_predicate,
                min_repeats=min_repeats,
                max_repeats=max_repeats,
                predicate_description=predicate_description,
                schedule=schedule,
            ),
            site=self.site,
            metadata=self.metadata,
            execution_policy=self.execution_policy,
            parameter_mappings=self.parameter_mappings,
            fixed_pseudoparams=self.fixed_pseudoparams,
        )

    @classmethod
    def single(
        cls,
        *,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
        execution_policy: ExecutionPolicy | None = None,
        randomise_order_globally: bool = False,
        random_seed: int | None = None,
    ) -> "ScanRequest":
        request = cls(
            axes=(),
            point_policy=SinglePointPolicy(),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
            execution_policy=ExecutionPolicy()
            if execution_policy is None
            else execution_policy,
        )
        if randomise_order_globally:
            return request.with_global_randomisation(random_seed=random_seed)
        return request

    @classmethod
    def linear(
        cls,
        axis: ParamHandle | ScanVariable,
        *,
        start: float,
        stop: float,
        num_points: int,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
        execution_policy: ExecutionPolicy | None = None,
        randomise_order_globally: bool = False,
        random_seed: int | None = None,
    ) -> "ScanRequest":
        if num_points < 2:
            raise ValueError("linear scans require at least 2 points")
        values = np.linspace(start=float(start), stop=float(stop), num=int(num_points))
        return cls.cartesian(
            [(axis, values.tolist())],
            site=site,
            metadata=metadata,
            execution_policy=execution_policy,
            randomise_order_globally=randomise_order_globally,
            random_seed=random_seed,
        )

    @classmethod
    def cartesian(
        cls,
        axes: Sequence[tuple[ParamHandle | ScanVariable, Sequence[Any]]],
        *,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
        execution_policy: ExecutionPolicy | None = None,
        randomise_order_globally: bool = False,
        random_seed: int | None = None,
    ) -> "ScanRequest":
        request = cls(
            axes=tuple(handle for handle, _ in axes),
            point_policy=CartesianPointPolicy([values for _, values in axes]),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
            execution_policy=ExecutionPolicy()
            if execution_policy is None
            else execution_policy,
        )
        if randomise_order_globally:
            return request.with_global_randomisation(random_seed=random_seed)
        return request

    @classmethod
    def zipped(
        cls,
        axes: Sequence[tuple[ParamHandle | ScanVariable, Sequence[Any]]],
        *,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
        execution_policy: ExecutionPolicy | None = None,
        randomise_order_globally: bool = False,
        random_seed: int | None = None,
    ) -> "ScanRequest":
        request = cls(
            axes=tuple(handle for handle, _ in axes),
            point_policy=ZipPointPolicy([values for _, values in axes]),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
            execution_policy=ExecutionPolicy()
            if execution_policy is None
            else execution_policy,
        )
        if randomise_order_globally:
            return request.with_global_randomisation(random_seed=random_seed)
        return request

    @classmethod
    def explicit(
        cls,
        axes: Sequence[ParamHandle | ScanVariable],
        points: Sequence[Sequence[Any]],
        *,
        site: ScanSite | None = None,
        metadata: Mapping[str, Any] | None = None,
        execution_policy: ExecutionPolicy | None = None,
        randomise_order_globally: bool = False,
        random_seed: int | None = None,
    ) -> "ScanRequest":
        request = cls(
            axes=tuple(axes),
            point_policy=ExplicitPointPolicy(len(axes), points),
            site=ScanSite() if site is None else site,
            metadata={} if metadata is None else metadata,
            execution_policy=ExecutionPolicy()
            if execution_policy is None
            else execution_policy,
        )
        if randomise_order_globally:
            return request.with_global_randomisation(random_seed=random_seed)
        return request

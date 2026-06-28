"""Typed scan-submission model and compiler.

There are three distinct layers in the new prepared-runtime submission flow:

- ``ScanSubmissionSpec`` is the user-facing submission model,
- a plain ``dict`` is the transport format that ARTIQ can ship between processes,
- ``ScanRequest`` is the resolved executable runtime object.

This module owns the first and third of those layers.  The runtime itself should only
execute a validated ``ScanRequest``.  The dashboard or any other submission frontend
should build a ``ScanSubmissionSpec`` (or, today, a dict transport payload), and this module
turns that declarative description into runtime objects.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from .expression import ExpressionCompileError
from ..define.fragment import ExpFragment
from ..define.parameters import ParamHandle, ParamStore
from ..define.result_channels import ResultChannel
from ..define.utils import path_matches_spec
from ..legacy.scan_generator import GENERATORS
from ..scan.mapping import FixedPseudoparam, ParameterMapping, ScanVariable
from ..scan.point_policy import (
    AskTellOptimiserPointPolicy,
    ExplicitPointPolicy,
    PointPolicy,
    ProductPointPolicy,
    SinglePointPolicy,
    ZipPointPolicy,
)

if TYPE_CHECKING:
    from ndscan.scan.request import ExecutionPolicy, ScanRequest

__all__ = [
    "ScanSubmissionSchemaError",
    "ScanSubmissionSpec",
    "compile_scan_submission_spec",
    "compile_scan_submission_schema",
]


class ScanSubmissionSchemaError(ValueError):
    """Raised when a scan-submission spec cannot be validated or compiled."""


def _reject_unknown_keys(
    obj: Mapping[str, Any],
    *,
    allowed: set[str],
    name: str,
) -> None:
    unknown = set(obj.keys()) - allowed
    if unknown:
        raise ScanSubmissionSchemaError(
            f"{name} contains unsupported keys: {', '.join(sorted(unknown))}"
        )


def _schema_mapping(obj: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise ScanSubmissionSchemaError(f"{name} must be a mapping")
    return obj


def _optional_mapping(obj: Any, name: str) -> Mapping[str, Any]:
    if obj is None:
        return {}
    return _schema_mapping(obj, name)


def _schema_sequence(obj: Any, name: str) -> Sequence[Any]:
    if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes, bytearray)):
        raise ScanSubmissionSchemaError(f"{name} must be a sequence")
    return obj


def _schema_string(obj: Mapping[str, Any], key: str) -> str:
    value = obj.get(key, None)
    if not isinstance(value, str):
        raise ScanSubmissionSchemaError(f"{key} must be a string")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ScanSubmissionSchemaError(f"{name} must be a string or null")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ScanSubmissionSchemaError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ScanSubmissionSchemaError(f"{name} must be positive")
    return result


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ScanSubmissionSchemaError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ScanSubmissionSchemaError(f"{name} must be finite")
    return result


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ScanSubmissionSchemaError(f"{name} must be positive")
    return result


@dataclass(slots=True)
class ScanSubmissionExecutionSpec:
    """Submission-time execution settings independent of any fragment tree."""

    max_points_per_batch: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ScanSubmissionExecutionSpec":
        mapping = _optional_mapping(data, "execution")
        _reject_unknown_keys(
            mapping,
            allowed={"max_points_per_batch"},
            name="execution",
        )
        return cls(
            max_points_per_batch=_optional_positive_int(
                mapping.get("max_points_per_batch", None),
                "execution.max_points_per_batch",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.max_points_per_batch is not None:
            data["max_points_per_batch"] = self.max_points_per_batch
        return data

    def validate(self) -> None:
        if self.max_points_per_batch is not None and self.max_points_per_batch <= 0:
            raise ScanSubmissionSchemaError("execution.max_points_per_batch must be positive")


@dataclass(slots=True)
class ScanSubmissionParamTargetSpec:
    """Reference to one or more concrete fragment parameters."""

    fqn: str
    path: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> "ScanSubmissionParamTargetSpec":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(mapping, allowed={"fqn", "path"}, name=name)
        return cls(
            fqn=_schema_string(mapping, "fqn"),
            path=_schema_string(mapping, "path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"fqn": self.fqn, "path": self.path}

    def validate(self, *, name: str) -> None:
        if not self.fqn:
            raise ScanSubmissionSchemaError(f"{name}.fqn must not be empty")
        # ``""`` is the canonical root-fragment path throughout ndscan.  The host
        # runtime metadata, legacy submission path, and path-matching helper all use
        # it already, so the typed scan-submission schema must accept it too.


@dataclass(slots=True)
class ScanSubmissionGeneratorSpec:
    """Serialized description of a finite grid-style generator."""

    type: str
    range: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> "ScanSubmissionGeneratorSpec":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(mapping, allowed={"type", "range"}, name=name)
        return cls(
            type=_schema_string(mapping, "type"),
            range=dict(_schema_mapping(mapping.get("range", None), f"{name}.range")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "range": dict(self.range)}

    def validate(self, *, name: str) -> None:
        if not self.type:
            raise ScanSubmissionSchemaError(f"{name}.type must not be empty")


@dataclass(slots=True)
class ScanSubmissionChannelTargetSpec:
    """Reference to a saved result channel by persisted channel path."""

    path: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> "ScanSubmissionChannelTargetSpec":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(mapping, allowed={"path"}, name=name)
        return cls(path=_schema_string(mapping, "path"))

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path}

    def validate(self, *, name: str) -> None:
        if not self.path:
            raise ScanSubmissionSchemaError(f"{name}.path must not be empty")


@dataclass(slots=True)
class ScanSubmissionChannelObjectiveSpec:
    """Current objective model for GPO mode.

    The submission schema deliberately models objectives separately from the optimiser
    backend.  Today the only supported objective is a scalar saved result channel, but
    keeping it explicit makes the compiler's responsibilities obvious.
    """

    target: ScanSubmissionChannelTargetSpec
    kind: ClassVar[str] = "channel"

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        name: str,
    ) -> "ScanSubmissionChannelObjectiveSpec":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(mapping, allowed={"kind", "target"}, name=name)
        kind = _schema_string(mapping, "kind")
        if kind != cls.kind:
            raise ScanSubmissionSchemaError(
                f"{name}.kind must be {cls.kind!r}, got {kind!r}"
            )
        return cls(
            target=ScanSubmissionChannelTargetSpec.from_dict(
                mapping.get("target", None),
                name=f"{name}.target",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target.to_dict()}

    def validate(self, *, name: str) -> None:
        self.target.validate(name=f"{name}.target")


@dataclass(slots=True)
class ScanSubmissionNuboBackendSpec:
    """Declarative settings for the optional NUBO BO backend."""

    batch_size: int | None = None
    initial_design_size: int = 1
    max_batches: int | None = None
    acquisition: str = "ucb"
    minimise: bool = True
    fit_steps: int | None = None
    fit_lr: float | None = None
    kind: ClassVar[str] = "nubo"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> "ScanSubmissionNuboBackendSpec":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(
            mapping,
            allowed={
                "kind",
                "batch_size",
                "initial_design_size",
                "max_batches",
                "acquisition",
                "minimise",
                "fit_steps",
                "fit_lr",
            },
            name=name,
        )
        kind = _schema_string(mapping, "kind")
        if kind != cls.kind:
            raise ScanSubmissionSchemaError(
                f"{name}.kind must be {cls.kind!r}, got {kind!r}"
            )

        acquisition = mapping.get("acquisition", "ucb")
        if not isinstance(acquisition, str):
            raise ScanSubmissionSchemaError(f"{name}.acquisition must be a string")
        minimise = mapping.get("minimise", True)
        if not isinstance(minimise, bool):
            raise ScanSubmissionSchemaError(f"{name}.minimise must be a boolean")

        return cls(
            batch_size=_optional_positive_int(mapping.get("batch_size", None), f"{name}.batch_size"),
            initial_design_size=_positive_int(
                mapping.get("initial_design_size", 1),
                f"{name}.initial_design_size",
            ),
            max_batches=_optional_positive_int(mapping.get("max_batches", None), f"{name}.max_batches"),
            acquisition=acquisition,
            minimise=minimise,
            fit_steps=_optional_positive_int(mapping.get("fit_steps", None), f"{name}.fit_steps"),
            fit_lr=_optional_positive_float(mapping.get("fit_lr", None), f"{name}.fit_lr"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "kind": self.kind,
            "initial_design_size": self.initial_design_size,
            "acquisition": self.acquisition,
            "minimise": self.minimise,
        }
        if self.batch_size is not None:
            data["batch_size"] = self.batch_size
        if self.max_batches is not None:
            data["max_batches"] = self.max_batches
        if self.fit_steps is not None:
            data["fit_steps"] = self.fit_steps
        if self.fit_lr is not None:
            data["fit_lr"] = self.fit_lr
        return data

    def validate(self, *, name: str) -> None:
        if self.batch_size is not None and self.batch_size <= 0:
            raise ScanSubmissionSchemaError(f"{name}.batch_size must be positive")
        if self.initial_design_size <= 0:
            raise ScanSubmissionSchemaError(f"{name}.initial_design_size must be positive")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ScanSubmissionSchemaError(f"{name}.max_batches must be positive")
        if not self.acquisition:
            raise ScanSubmissionSchemaError(f"{name}.acquisition must not be empty")
        if self.fit_steps is not None and self.fit_steps <= 0:
            raise ScanSubmissionSchemaError(f"{name}.fit_steps must be positive")
        if self.fit_lr is not None and self.fit_lr <= 0.0:
            raise ScanSubmissionSchemaError(f"{name}.fit_lr must be positive")


@dataclass(slots=True)
class ScanSubmissionGridModeSpec:
    """Cartesian/zipped grid mode."""

    randomise_order_globally: bool = False
    type: ClassVar[str] = "grid"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> "ScanSubmissionGridModeSpec":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(mapping, allowed={"type", "randomise_order_globally"}, name=name)
        mode_type = _schema_string(mapping, "type")
        if mode_type != cls.type:
            raise ScanSubmissionSchemaError(f"{name}.type must be {cls.type!r}, got {mode_type!r}")
        randomise_order_globally = mapping.get("randomise_order_globally", False)
        if not isinstance(randomise_order_globally, bool):
            raise ScanSubmissionSchemaError(f"{name}.randomise_order_globally must be a boolean")
        return cls(randomise_order_globally=randomise_order_globally)

    def to_dict(self) -> dict[str, Any]:
        data = {"type": self.type}
        if self.randomise_order_globally:
            data["randomise_order_globally"] = True
        return data

    def validate(self, *, name: str) -> None:
        del name


@dataclass(slots=True)
class ScanSubmissionGpoModeSpec:
    """Gaussian-process optimisation mode."""

    objective: ScanSubmissionChannelObjectiveSpec
    backend: ScanSubmissionNuboBackendSpec
    type: ClassVar[str] = "gpo"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> "ScanSubmissionGpoModeSpec":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(mapping, allowed={"type", "objective", "backend"}, name=name)
        mode_type = _schema_string(mapping, "type")
        if mode_type != cls.type:
            raise ScanSubmissionSchemaError(f"{name}.type must be {cls.type!r}, got {mode_type!r}")
        return cls(
            objective=ScanSubmissionChannelObjectiveSpec.from_dict(
                mapping.get("objective", None),
                name=f"{name}.objective",
            ),
            backend=ScanSubmissionNuboBackendSpec.from_dict(
                mapping.get("backend", None),
                name=f"{name}.backend",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "objective": self.objective.to_dict(),
            "backend": self.backend.to_dict(),
        }

    def validate(self, *, name: str) -> None:
        self.objective.validate(name=f"{name}.objective")
        self.backend.validate(name=f"{name}.backend")


@dataclass(slots=True)
class ScanSubmissionFixedModeSpec:
    """Constant value for a row entry."""

    value: Any
    type: ClassVar[str] = "fixed"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> "ScanSubmissionFixedModeSpec":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(mapping, allowed={"type", "value"}, name=name)
        mode_type = _schema_string(mapping, "type")
        if mode_type != cls.type:
            raise ScanSubmissionSchemaError(f"{name}.type must be {cls.type!r}, got {mode_type!r}")
        if "value" not in mapping:
            raise ScanSubmissionSchemaError(f"{name}.value is required")
        return cls(value=mapping["value"])

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "value": self.value}

    def validate(self, *, name: str) -> None:
        del name


@dataclass(slots=True)
class ScanSubmissionScanModeSpec:
    """Finite grid-style scan over a generated value list."""

    generator: ScanSubmissionGeneratorSpec
    group: str | None = None
    type: ClassVar[str] = "scan"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> "ScanSubmissionScanModeSpec":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(mapping, allowed={"type", "generator", "group"}, name=name)
        mode_type = _schema_string(mapping, "type")
        if mode_type != cls.type:
            raise ScanSubmissionSchemaError(f"{name}.type must be {cls.type!r}, got {mode_type!r}")
        return cls(
            generator=ScanSubmissionGeneratorSpec.from_dict(
                mapping.get("generator", None),
                name=f"{name}.generator",
            ),
            group=_optional_string(mapping.get("group", None), f"{name}.group"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {"type": self.type, "generator": self.generator.to_dict()}
        if self.group is not None:
            data["group"] = self.group
        return data

    def validate(self, *, name: str) -> None:
        self.generator.validate(name=f"{name}.generator")
        if self.group == "":
            raise ScanSubmissionSchemaError(f"{name}.group must not be empty")


@dataclass(slots=True)
class ScanSubmissionGpoScanModeSpec:
    """Bounded optimisation dimension for GPO mode."""

    lower: float
    upper: float
    type: ClassVar[str] = "gpo_scan"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> "ScanSubmissionGpoScanModeSpec":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(mapping, allowed={"type", "lower", "upper"}, name=name)
        mode_type = _schema_string(mapping, "type")
        if mode_type != cls.type:
            raise ScanSubmissionSchemaError(f"{name}.type must be {cls.type!r}, got {mode_type!r}")
        return cls(
            lower=_finite_real(mapping.get("lower", None), f"{name}.lower"),
            upper=_finite_real(mapping.get("upper", None), f"{name}.upper"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "lower": self.lower, "upper": self.upper}

    def validate(self, *, name: str) -> None:
        if self.lower >= self.upper:
            raise ScanSubmissionSchemaError(f"{name} must satisfy lower < upper")


@dataclass(slots=True)
class ScanSubmissionRebindModeSpec:
    """Text formula compiled later into a ``ParameterMapping``."""

    expr: str
    type: ClassVar[str] = "rebind"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> "ScanSubmissionRebindModeSpec":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(mapping, allowed={"type", "expr"}, name=name)
        mode_type = _schema_string(mapping, "type")
        if mode_type != cls.type:
            raise ScanSubmissionSchemaError(f"{name}.type must be {cls.type!r}, got {mode_type!r}")
        return cls(expr=_schema_string(mapping, "expr"))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "expr": self.expr}

    def validate(self, *, name: str) -> None:
        if not self.expr:
            raise ScanSubmissionSchemaError(f"{name}.expr must not be empty")


type ScanSubmissionModeSpec = ScanSubmissionGridModeSpec | ScanSubmissionGpoModeSpec
type ScanSubmissionEntryModeSpec = (
    ScanSubmissionFixedModeSpec
    | ScanSubmissionScanModeSpec
    | ScanSubmissionGpoScanModeSpec
    | ScanSubmissionRebindModeSpec
)


def _mode_from_dict(data: Mapping[str, Any], *, name: str) -> ScanSubmissionModeSpec:
    mapping = _schema_mapping(data, name)
    mode_type = _schema_string(mapping, "type")
    if mode_type == ScanSubmissionGridModeSpec.type:
        return ScanSubmissionGridModeSpec.from_dict(mapping, name=name)
    if mode_type == ScanSubmissionGpoModeSpec.type:
        return ScanSubmissionGpoModeSpec.from_dict(mapping, name=name)
    raise ScanSubmissionSchemaError(f"Unsupported scan submission mode: {mode_type!r}")


def _entry_mode_from_dict(data: Mapping[str, Any], *, name: str) -> ScanSubmissionEntryModeSpec:
    mapping = _schema_mapping(data, name)
    mode_type = _schema_string(mapping, "type")
    if mode_type == ScanSubmissionFixedModeSpec.type:
        return ScanSubmissionFixedModeSpec.from_dict(mapping, name=name)
    if mode_type == ScanSubmissionScanModeSpec.type:
        return ScanSubmissionScanModeSpec.from_dict(mapping, name=name)
    if mode_type == ScanSubmissionGpoScanModeSpec.type:
        return ScanSubmissionGpoScanModeSpec.from_dict(mapping, name=name)
    if mode_type == ScanSubmissionRebindModeSpec.type:
        return ScanSubmissionRebindModeSpec.from_dict(mapping, name=name)
    raise ScanSubmissionSchemaError(f"Unsupported row mode: {mode_type!r}")


@dataclass(slots=True)
class ScanSubmissionEntry:
    """One submitted parameter or pseudoparameter row."""

    id: str
    kind: str
    mode: ScanSubmissionEntryModeSpec
    target: ScanSubmissionParamTargetSpec | None = None
    description: str = ""
    default: Any = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> "ScanSubmissionEntry":
        mapping = _schema_mapping(data, name)
        _reject_unknown_keys(
            mapping,
            allowed={"id", "kind", "target", "mode", "description", "default"},
            name=name,
        )
        description = mapping.get("description", "")
        if description is None:
            description = ""
        if not isinstance(description, str):
            raise ScanSubmissionSchemaError(f"{name}.description must be a string")
        target = None
        if "target" in mapping:
            target = ScanSubmissionParamTargetSpec.from_dict(
                mapping.get("target", None),
                name=f"{name}.target",
            )
        return cls(
            id=_schema_string(mapping, "id"),
            kind=_schema_string(mapping, "kind"),
            target=target,
            mode=_entry_mode_from_dict(mapping.get("mode", None), name=f"{name}.mode"),
            description=description,
            default=mapping.get("default", None),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "kind": self.kind,
            "mode": self.mode.to_dict(),
        }
        if self.target is not None:
            data["target"] = self.target.to_dict()
        if self.description:
            data["description"] = self.description
        if self.default is not None:
            data["default"] = self.default
        return data

    def validate(self, *, name: str) -> None:
        if not self.id.isidentifier():
            raise ScanSubmissionSchemaError(
                f"{name}.id {self.id!r} is not a valid Python identifier"
            )
        if self.kind not in {"param", "pseudoparam"}:
            raise ScanSubmissionSchemaError(
                f"{name}.kind must be 'param' or 'pseudoparam', got {self.kind!r}"
            )
        if self.kind == "param":
            if self.target is None:
                raise ScanSubmissionSchemaError(f"{name}.target is required for parameter entries")
            self.target.validate(name=f"{name}.target")
        else:
            if self.target is not None:
                raise ScanSubmissionSchemaError(
                    f"{name}.target is only valid for parameter entries"
                )
            if isinstance(self.mode, ScanSubmissionRebindModeSpec):
                raise ScanSubmissionSchemaError(
                    f"Pseudoparameter entry {self.id!r} cannot use row mode 'rebind'"
                )
        self.mode.validate(name=f"{name}.mode")


@dataclass(slots=True)
class ScanSubmissionSpec:
    """Typed submission model for prepared-runtime scans.

    ``ScanSubmissionSpec`` is intentionally declarative.  It does not resolve fragment
    handles, create point policies, or compile expressions by itself.  Those steps
    belong to ``compile_scan_submission_spec()``.
    """

    mode: ScanSubmissionModeSpec
    entries: tuple[ScanSubmissionEntry, ...] = ()
    execution: ScanSubmissionExecutionSpec = field(default_factory=ScanSubmissionExecutionSpec)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        self.entries = tuple(self.entries)
        self.metadata = dict(self.metadata)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScanSubmissionSpec":
        mapping = _schema_mapping(data, "scan submission schema")
        _reject_unknown_keys(
            mapping,
            allowed={"version", "mode", "entries", "execution", "metadata"},
            name="scan submission schema",
        )
        version = mapping.get("version", 1)
        if isinstance(version, bool) or not isinstance(version, (int, np.integer)):
            raise ScanSubmissionSchemaError("version must be an integer")
        spec = cls(
            version=int(version),
            mode=_mode_from_dict(mapping.get("mode", None), name="mode"),
            entries=tuple(
                ScanSubmissionEntry.from_dict(entry, name=f"entries[{index}]")
                for index, entry in enumerate(
                    _schema_sequence(mapping.get("entries", ()), "entries")
                )
            ),
            execution=ScanSubmissionExecutionSpec.from_dict(mapping.get("execution", {})),
            metadata=dict(_optional_mapping(mapping.get("metadata", {}), "metadata")),
        )
        spec.validate()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode.to_dict(),
            "entries": [entry.to_dict() for entry in self.entries],
            "execution": self.execution.to_dict(),
            "metadata": dict(self.metadata),
        }

    def validate(self) -> None:
        if self.version != 1:
            raise ScanSubmissionSchemaError(f"Unsupported scan submission schema version: {self.version!r}")

        if not isinstance(self.mode, (ScanSubmissionGridModeSpec, ScanSubmissionGpoModeSpec)):
            raise ScanSubmissionSchemaError("mode must be a supported scan-submission mode spec")
        self.mode.validate(name="mode")
        self.execution.validate()

        seen_ids = set[str]()
        num_gpo_entries = 0

        for index, entry in enumerate(self.entries):
            entry_name = f"entries[{index}]"
            entry.validate(name=entry_name)
            if entry.id in seen_ids:
                raise ScanSubmissionSchemaError(f"Duplicate scan submission entry id: {entry.id!r}")
            seen_ids.add(entry.id)

            if isinstance(self.mode, ScanSubmissionGridModeSpec) and isinstance(
                entry.mode, ScanSubmissionGpoScanModeSpec
            ):
                raise ScanSubmissionSchemaError(
                    f"{entry_name}.mode.type == 'gpo_scan' is only valid in gpo mode"
                )
            if isinstance(self.mode, ScanSubmissionGpoModeSpec):
                if isinstance(entry.mode, ScanSubmissionScanModeSpec):
                    raise ScanSubmissionSchemaError(
                        f"{entry_name}.mode.type == 'scan' is only valid in grid mode"
                    )
                if isinstance(entry.mode, ScanSubmissionGpoScanModeSpec):
                    num_gpo_entries += 1

        if isinstance(self.mode, ScanSubmissionGpoModeSpec) and num_gpo_entries == 0:
            raise ScanSubmissionSchemaError("GPO mode requires at least one gpo_scan entry")


def compile_scan_submission_spec(
    fragment: ExpFragment,
    spec: ScanSubmissionSpec,
) -> tuple["ScanRequest", dict[str, list[tuple[str, ParamStore]]]]:
    """Compile a typed ``ScanSubmissionSpec`` into ``ScanRequest`` plus fixed overrides."""

    spec.validate()
    execution_policy = _compile_execution_policy_from_spec(spec.execution)
    compiled_entries = _compile_spec_entries(fragment, spec.entries)

    if isinstance(spec.mode, ScanSubmissionGridModeSpec):
        return _compile_grid_schema_request(
            compiled_entries,
            spec.mode,
            metadata=spec.metadata,
            execution_policy=execution_policy,
        )
    if isinstance(spec.mode, ScanSubmissionGpoModeSpec):
        return _compile_gpo_schema_request(
            fragment,
            compiled_entries,
            spec.mode,
            metadata=spec.metadata,
            execution_policy=execution_policy,
        )
    raise ScanSubmissionSchemaError(f"Unsupported scan submission mode: {type(spec.mode).__name__!r}")


def compile_scan_submission_schema(
    fragment: ExpFragment,
    schema: Mapping[str, Any],
) -> tuple["ScanRequest", dict[str, list[tuple[str, ParamStore]]]]:
    """Compile transport-form dicts by round-tripping through ``ScanSubmissionSpec``."""

    return compile_scan_submission_spec(fragment, ScanSubmissionSpec.from_dict(schema))


@dataclass(frozen=True)
class _CompiledSchemaEntry:
    entry_id: str
    kind: str
    mode_type: str
    source: ParamHandle | ScanVariable | None
    pathspec: str | None
    targets: tuple[ParamHandle, ...] = ()
    values: tuple[Any, ...] = ()
    bounds: tuple[float, float] | None = None
    group: str | None = None
    override: tuple[str, tuple[str, ParamStore]] | None = None
    parameter_mappings: tuple[ParameterMapping, ...] = ()
    description: str = ""
    expression: str | None = None
    defines_symbol: bool = False
    symbol_value: Any = None
    fixed_pseudoparam: FixedPseudoparam | None = None


def _compile_execution_policy_from_spec(
    execution: ScanSubmissionExecutionSpec,
) -> "ExecutionPolicy":
    from ndscan.scan.request import ExecutionPolicy

    return ExecutionPolicy(max_points_per_batch=execution.max_points_per_batch)


def _compile_spec_entries(
    fragment: ExpFragment,
    entries: Sequence[ScanSubmissionEntry],
) -> list[_CompiledSchemaEntry]:
    compiled: list[_CompiledSchemaEntry] = []
    seen_ids = set[str]()
    claimed_targets = set[tuple[int, str]]()

    for entry in entries:
        if entry.id in seen_ids:
            raise ScanSubmissionSchemaError(f"Duplicate scan submission entry id: {entry.id!r}")
        seen_ids.add(entry.id)

        if entry.kind == "param":
            assert entry.target is not None
            handles = _resolve_schema_param_handles(fragment, entry.target.fqn, entry.target.path)
            if not handles:
                raise ScanSubmissionSchemaError(
                    f"No parameter handles matched target fqn={entry.target.fqn!r}, "
                    f"path={entry.target.path!r}"
                )
            overlap = {(_mapping_target_key(handle)) for handle in handles} & claimed_targets
            if overlap:
                raise ScanSubmissionSchemaError(
                    f"Parameter target for entry {entry.id!r} overlaps an earlier entry"
                )
            claimed_targets.update(_mapping_target_key(handle) for handle in handles)
            compiled.append(_compile_param_entry(entry, handles))
            continue

        compiled.append(_compile_pseudoparam_entry(entry))

    symbols = _collect_schema_symbols(compiled)
    return [
        _attach_rebind_mapping(entry, symbols)
        if entry.mode_type == ScanSubmissionRebindModeSpec.type
        else entry
        for entry in compiled
    ]


def _compile_param_entry(
    entry: ScanSubmissionEntry,
    handles: Sequence[ParamHandle],
) -> _CompiledSchemaEntry:
    assert entry.target is not None
    mode = entry.mode
    first_handle = handles[0]
    if isinstance(mode, ScanSubmissionFixedModeSpec):
        value = mode.value
        store = first_handle.parameter.make_store(
            (first_handle.parameter.fqn, entry.target.path), value
        )
        return _CompiledSchemaEntry(
            entry_id=entry.id,
            kind="param",
            mode_type=mode.type,
            source=None,
            pathspec=entry.target.path,
            targets=tuple(handles),
            override=(first_handle.parameter.fqn, (entry.target.path, store)),
            description=entry.description,
            defines_symbol=True,
            symbol_value=value,
        )

    if isinstance(mode, ScanSubmissionScanModeSpec):
        values = _materialise_scan_mode_values(mode, entry.id)
        source, mappings = _compile_entry_axis_source(
            entry.id,
            handles,
            values,
            description=entry.description or first_handle.parameter.description,
        )
        return _CompiledSchemaEntry(
            entry_id=entry.id,
            kind="param",
            mode_type=mode.type,
            source=source,
            pathspec=entry.target.path,
            targets=tuple(handles),
            values=values,
            group=mode.group,
            parameter_mappings=mappings,
            description=entry.description,
            defines_symbol=True,
            symbol_value=source,
        )

    if isinstance(mode, ScanSubmissionGpoScanModeSpec):
        source, mappings = _compile_entry_axis_source(
            entry.id,
            handles,
            (mode.lower, mode.upper),
            description=entry.description or first_handle.parameter.description,
        )
        return _CompiledSchemaEntry(
            entry_id=entry.id,
            kind="param",
            mode_type=mode.type,
            source=source,
            pathspec=entry.target.path,
            targets=tuple(handles),
            bounds=(mode.lower, mode.upper),
            parameter_mappings=mappings,
            description=entry.description,
            defines_symbol=True,
            symbol_value=source,
        )

    if isinstance(mode, ScanSubmissionRebindModeSpec):
        return _CompiledSchemaEntry(
            entry_id=entry.id,
            kind="param",
            mode_type=mode.type,
            source=None,
            pathspec=entry.target.path,
            targets=tuple(handles),
            description=entry.description,
            expression=mode.expr,
        )

    raise ScanSubmissionSchemaError(
        f"Unsupported row mode for parameter entry {entry.id!r}: {type(mode).__name__!r}"
    )


def _compile_pseudoparam_entry(entry: ScanSubmissionEntry) -> _CompiledSchemaEntry:
    mode = entry.mode
    if isinstance(mode, ScanSubmissionFixedModeSpec):
        value = mode.value
        variable = ScanVariable(
            name=entry.id,
            description=entry.description,
            type=_infer_scan_variable_type(value if value is not None else entry.default),
            spec={},
        )
        return _CompiledSchemaEntry(
            entry_id=entry.id,
            kind="pseudoparam",
            mode_type=mode.type,
            source=None,
            pathspec=None,
            description=entry.description,
            defines_symbol=True,
            symbol_value=value,
            fixed_pseudoparam=FixedPseudoparam(variable=variable, value=value),
        )

    if isinstance(mode, ScanSubmissionScanModeSpec):
        values = _materialise_scan_mode_values(mode, entry.id)
        variable = ScanVariable(
            name=entry.id,
            description=entry.description,
            type=_infer_scan_variable_type(values[0] if values else entry.default),
            spec=_describe_scan_variable_values(values),
        )
        return _CompiledSchemaEntry(
            entry_id=entry.id,
            kind="pseudoparam",
            mode_type=mode.type,
            source=variable,
            pathspec=None,
            values=values,
            group=mode.group,
            description=entry.description,
            defines_symbol=True,
            symbol_value=variable,
        )

    if isinstance(mode, ScanSubmissionGpoScanModeSpec):
        variable = ScanVariable(
            name=entry.id,
            description=entry.description,
            type="float",
            spec={"min": mode.lower, "max": mode.upper},
        )
        return _CompiledSchemaEntry(
            entry_id=entry.id,
            kind="pseudoparam",
            mode_type=mode.type,
            source=variable,
            pathspec=None,
            bounds=(mode.lower, mode.upper),
            description=entry.description,
            defines_symbol=True,
            symbol_value=variable,
        )

    raise ScanSubmissionSchemaError(
        f"Unsupported row mode for pseudoparam entry {entry.id!r}: {type(mode).__name__!r}"
    )


def _compile_entry_axis_source(
    entry_id: str,
    handles: Sequence[ParamHandle],
    values: Sequence[Any],
    *,
    description: str,
) -> tuple[ParamHandle | ScanVariable, tuple[ParameterMapping, ...]]:
    if len(handles) == 1:
        return handles[0], ()

    variable = ScanVariable(
        name=entry_id,
        description=description,
        type=_infer_scan_variable_type(values[0] if values else None),
        spec=_describe_scan_variable_values(values),
    )
    targets = tuple(handles)
    mapping = ParameterMapping(
        targets=targets,
        dependencies=(variable,),
        evaluate=lambda dependency_values, variable=variable, targets=targets: {
            target: dependency_values[variable] for target in targets
        },
        description=f"{entry_id} identity mapping",
    )
    return variable, (mapping,)


def _collect_schema_symbols(
    compiled_entries: Sequence[_CompiledSchemaEntry],
) -> dict[str, ParamHandle | ScanVariable | Any]:
    symbols: dict[str, ParamHandle | ScanVariable | Any] = {}
    for entry in compiled_entries:
        if entry.defines_symbol:
            symbols[entry.entry_id] = entry.symbol_value
    return symbols


def _attach_rebind_mapping(
    entry: _CompiledSchemaEntry,
    symbols: Mapping[str, ParamHandle | ScanVariable | Any],
) -> _CompiledSchemaEntry:
    assert entry.mode_type == ScanSubmissionRebindModeSpec.type
    assert entry.expression is not None
    try:
        mapping = ParameterMapping.from_text(
            targets=entry.targets,
            expression=entry.expression,
            symbols=symbols,
            description=entry.description,
        )
    except (ExpressionCompileError, TypeError, ValueError) as exc:
        raise ScanSubmissionSchemaError(
            f"Invalid rebind expression for entry {entry.entry_id!r}: {exc}"
        ) from exc

    return _CompiledSchemaEntry(
        entry_id=entry.entry_id,
        kind=entry.kind,
        mode_type=entry.mode_type,
        source=entry.source,
        pathspec=entry.pathspec,
        targets=entry.targets,
        values=entry.values,
        bounds=entry.bounds,
        group=entry.group,
        override=entry.override,
        parameter_mappings=(mapping,),
        description=entry.description,
        expression=entry.expression,
    )


def _compile_grid_schema_request(
    compiled_entries: Sequence[_CompiledSchemaEntry],
    mode: ScanSubmissionGridModeSpec,
    *,
    metadata: Mapping[str, Any],
    execution_policy: "ExecutionPolicy",
) -> tuple["ScanRequest", dict[str, list[tuple[str, ParamStore]]]]:
    from ndscan.scan.request import ScanRequest

    overrides = _collect_compiled_overrides(compiled_entries)
    fixed_pseudoparams = tuple(
        entry.fixed_pseudoparam
        for entry in compiled_entries
        if entry.fixed_pseudoparam is not None
    )
    scanned_entries = [
        entry for entry in compiled_entries if entry.mode_type == ScanSubmissionScanModeSpec.type
    ]
    parameter_mappings = tuple(
        mapping
        for entry in compiled_entries
        for mapping in entry.parameter_mappings
    )
    if not scanned_entries:
        return (
            ScanRequest(
                axes=(),
                point_policy=SinglePointPolicy(),
                metadata=metadata,
                execution_policy=execution_policy,
                parameter_mappings=parameter_mappings,
                fixed_pseudoparams=fixed_pseudoparams,
            ),
            overrides,
        )

    groups: OrderedDict[str, list[_CompiledSchemaEntry]] = OrderedDict()
    for entry in scanned_entries:
        group_name = entry.group if entry.group else f"__singleton__:{entry.entry_id}"
        groups.setdefault(group_name, []).append(entry)

    axes = tuple(
        entry.source
        for group_entries in groups.values()
        for entry in group_entries
        if entry.source is not None
    )
    child_policies = [_compile_group_point_policy(group_entries) for group_entries in groups.values()]
    point_policy: PointPolicy
    if len(child_policies) == 1:
        point_policy = child_policies[0]
    else:
        point_policy = ProductPointPolicy(child_policies)

    request = ScanRequest(
        axes=axes,
        point_policy=point_policy,
        metadata=metadata,
        execution_policy=execution_policy,
        parameter_mappings=parameter_mappings,
        fixed_pseudoparams=fixed_pseudoparams,
    )
    if mode.randomise_order_globally:
        request = request.with_global_randomisation(random_seed=random.getrandbits(32))

    return (request, overrides)


def _compile_group_point_policy(
    group_entries: Sequence[_CompiledSchemaEntry],
) -> PointPolicy:
    axis_values = [entry.values for entry in group_entries]
    lengths = {len(values) for values in axis_values}
    if len(lengths) == 1:
        return ZipPointPolicy(axis_values)
    return ExplicitPointPolicy(
        len(group_entries),
        list(zip(*axis_values)),
    )


def _compile_gpo_schema_request(
    fragment: ExpFragment,
    compiled_entries: Sequence[_CompiledSchemaEntry],
    mode: ScanSubmissionGpoModeSpec,
    *,
    metadata: Mapping[str, Any],
    execution_policy: "ExecutionPolicy",
) -> tuple["ScanRequest", dict[str, list[tuple[str, ParamStore]]]]:
    from ndscan.scan.request import ExecutionPolicy, ScanRequest

    overrides = _collect_compiled_overrides(compiled_entries)
    fixed_pseudoparams = tuple(
        entry.fixed_pseudoparam
        for entry in compiled_entries
        if entry.fixed_pseudoparam is not None
    )
    gpo_entries = [
        entry for entry in compiled_entries if entry.mode_type == ScanSubmissionGpoScanModeSpec.type
    ]
    if not gpo_entries:
        raise ScanSubmissionSchemaError("GPO mode requires at least one gpo_scan entry")

    objective_channel_key = _resolve_saved_channel_key(fragment, mode.objective.target.path)

    bounds = [[], []]
    for entry in gpo_entries:
        assert entry.bounds is not None
        bounds[0].append(entry.bounds[0])
        bounds[1].append(entry.bounds[1])

    try:
        from ..scan.optimisation import (
            NuboBatchBayesianOptimisationBackend,
            extract_scalar_channel_objective,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optional Bayesian optimisation dependencies are not installed"
        ) from exc

    batch_size = mode.backend.batch_size or execution_policy.max_points_per_batch or 1
    backend_instance = NuboBatchBayesianOptimisationBackend(
        bounds=bounds,
        batch_size=batch_size,
        initial_design_size=mode.backend.initial_design_size,
        max_batches=mode.backend.max_batches,
        acquisition_name=mode.backend.acquisition,
        minimise=mode.backend.minimise,
        **(
            {"fit_steps": mode.backend.fit_steps}
            if mode.backend.fit_steps is not None
            else {}
        ),
        **(
            {"fit_lr": mode.backend.fit_lr}
            if mode.backend.fit_lr is not None
            else {}
        ),
    )
    point_policy = AskTellOptimiserPointPolicy(
        backend_instance,
        extract_scalar_channel_objective(objective_channel_key),
    )
    parameter_mappings = tuple(
        mapping
        for entry in compiled_entries
        for mapping in entry.parameter_mappings
    )
    resolved_execution_policy = execution_policy
    if execution_policy.max_points_per_batch is None:
        resolved_execution_policy = ExecutionPolicy(max_points_per_batch=batch_size)
    elif execution_policy.max_points_per_batch == 1 and batch_size != 1:
        resolved_execution_policy = ExecutionPolicy(max_points_per_batch=batch_size)

    return (
        ScanRequest(
            axes=tuple(entry.source for entry in gpo_entries if entry.source is not None),
            point_policy=point_policy,
            metadata=metadata,
            execution_policy=resolved_execution_policy,
            parameter_mappings=parameter_mappings,
            fixed_pseudoparams=fixed_pseudoparams,
        ),
        overrides,
    )


def _collect_compiled_overrides(
    compiled_entries: Sequence[_CompiledSchemaEntry],
) -> dict[str, list[tuple[str, ParamStore]]]:
    overrides: dict[str, list[tuple[str, ParamStore]]] = {}
    for entry in compiled_entries:
        if entry.override is None:
            continue
        fqn, pair = entry.override
        overrides.setdefault(fqn, []).append(pair)
    return overrides


def _resolve_schema_param_handles(
    fragment: ExpFragment,
    fqn: str,
    pathspec: str,
) -> list[ParamHandle]:
    handles = []
    stack = [fragment]
    while stack:
        current = stack.pop(0)
        if path_matches_spec(current._fragment_path, pathspec):
            for name, param in current._free_params.items():
                if param.fqn == fqn:
                    handles.append(getattr(current, name))
        stack.extend(current._subfragments)
    return handles


def _resolve_saved_channel_key(fragment: ExpFragment, path: str) -> str:
    channel_dict = dict[str, ResultChannel]()
    fragment._collect_result_channels(channel_dict)
    saved_channels = [channel for channel in channel_dict.values() if channel.save_by_default]
    for index, channel in enumerate(saved_channels):
        if channel.path == path:
            return f"channel_{index}"
    raise ScanSubmissionSchemaError(f"No saved result channel matched path {path!r}")


def _materialise_scan_mode_values(
    mode: ScanSubmissionScanModeSpec,
    entry_id: str,
) -> tuple[Any, ...]:
    generator_class = GENERATORS.get(mode.generator.type, None)
    if generator_class is None:
        raise ScanSubmissionSchemaError(
            f"Unsupported scan generator type for entry {entry_id!r}: {mode.generator.type!r}"
        )

    if mode.generator.type in {"refining", "centre_span_refining"}:
        raise NotImplementedError(
            f"Generator type {mode.generator.type!r} is not yet supported in typed scan submissions"
        )

    generator_instance = generator_class(**dict(mode.generator.range))
    rng = np.random.RandomState(random.getrandbits(32))
    if mode.generator.type in {"linear", "centre_span", "list"}:
        return tuple(generator_instance.points_for_level(0, rng))
    if mode.generator.type == "expanding":
        values = []
        level = 0
        while generator_instance.has_level(level):
            values.extend(generator_instance.points_for_level(level, rng))
            level += 1
            if level > 1_000_000:
                raise ScanSubmissionSchemaError(
                    f"Refusing to materialise more than 1,000,000 expanding levels for {entry_id!r}"
                )
        return tuple(values)

    raise ScanSubmissionSchemaError(
        f"Generator type {mode.generator.type!r} is not yet supported in typed scan submissions"
    )


def _infer_scan_variable_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return "int"
    if isinstance(value, (float, np.floating)) or value is None:
        return "float"
    if isinstance(value, str):
        return "string"
    return "float"


def _describe_scan_variable_values(values: Sequence[Any]) -> dict[str, Any]:
    if not values:
        return {}
    first = values[0]
    if isinstance(first, bool):
        return {}
    if isinstance(first, (int, float, np.integer, np.floating)):
        return {"min": min(values), "max": max(values)}
    return {}


def _mapping_target_key(handle: ParamHandle) -> tuple[int, str]:
    return (id(handle.owner), handle.name)

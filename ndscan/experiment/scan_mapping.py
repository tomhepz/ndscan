"""Logical scan axes and parameter-mapping declarations.

The host runtime distinguishes between two concepts:

- a scan axis says which *logical coordinate* varies from point to point,
- a parameter mapping says how that logical coordinate translates into concrete
  fragment parameter values before the point body runs.

This lets low-level fragments stay hardware-facing while still supporting richer
submission-time reparameterisations. Wrapper fragments can declare reusable mappings in
Python, and future GUI formula entry can compile to the same ``ParameterMapping``
objects without changing the execution core again.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .parameters import ParamHandle

__all__ = [
    "ScanVariable",
    "ParameterMapping",
]


@dataclass(frozen=True)
class ScanVariable:
    """Logical scan coordinate used by the runtime, not directly by hardware.

    ``ScanVariable`` exists for ad hoc reparameterisations where a scan should be
    expressed in a convenient logical basis without modifying the fragment tree. The
    corresponding hardware-facing parameters are then supplied by one or more
    ``ParameterMapping`` objects.

    Wrapper fragments can often use ordinary fragment parameters instead, which keeps
    those logical coordinates visible to default analyses and other fragment-local
    machinery.
    """

    name: str
    description: str = ""
    type: str = "float"
    spec: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError("ScanVariable name must be a valid Python identifier")

    def __hash__(self) -> int:
        return hash((self.name, self.description, self.type, repr(dict(self.spec))))

    @property
    def fqn(self) -> str:
        """Return a stable synthetic identity string for metadata/debugging."""

        return "scan_variable." + self.name

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "spec": dict(self.spec),
        }


@dataclass(frozen=True)
class ParameterMapping:
    """Map logical inputs to one or more concrete fragment parameters.

    The runtime evaluates mappings once per point, after any directly scanned
    ``ParamHandle`` axes have been installed but before ``device_setup()`` or
    ``run_once()`` are entered.

    ``dependencies`` can contain ordinary fragment ``ParamHandle`` objects or runtime
    ``ScanVariable`` objects. ``targets`` must always be real fragment parameters,
    because the end result of a mapping is to install actual parameter-store values.

    ``evaluate`` receives a mapping keyed by the original dependency objects. For a
    single-target mapping it may return either the direct value for that target or an
    explicit ``{target_handle: value}`` dictionary. Multi-target mappings must return a
    dictionary keyed by the target handles.
    """

    targets: tuple[ParamHandle, ...]
    dependencies: tuple[ParamHandle | ScanVariable, ...]
    evaluate: Callable[[Mapping[ParamHandle | ScanVariable, Any]], Any]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("ParameterMapping requires at least one target")
        if len({_param_key(target) for target in self.targets}) != len(self.targets):
            raise ValueError("ParameterMapping target parameters must be unique")

    @classmethod
    def single_target(
        cls,
        target: ParamHandle,
        dependencies: Sequence[ParamHandle | ScanVariable],
        evaluate: Callable[[Mapping[ParamHandle | ScanVariable, Any]], Any],
        *,
        description: str = "",
    ) -> "ParameterMapping":
        """Convenience constructor for the common one-target case."""

        return cls(
            targets=(target,),
            dependencies=tuple(dependencies),
            evaluate=evaluate,
            description=description,
        )

    @classmethod
    def from_text(
        cls,
        *,
        targets: Sequence[ParamHandle],
        dependencies: Sequence[ParamHandle | ScanVariable],
        expression: str,
        description: str = "",
    ) -> "ParameterMapping":
        """Future GUI-oriented text entry for parameter mappings.

        The intended design is for dashboard/UI formula entry to compile to ordinary
        ``ParameterMapping`` objects. The runtime should not need a second execution
        path just because a mapping originated from text rather than Python code.
        """

        raise NotImplementedError(
            "Text-based mapping expressions are not implemented yet; compile GUI "
            "formula input to a ParameterMapping before execution."
        )

    def compute(
        self, dependency_values: Mapping[ParamHandle | ScanVariable, Any]
    ) -> dict[ParamHandle, Any]:
        """Evaluate this mapping for one concrete scan point."""

        missing = [
            dependency
            for dependency in self.dependencies
            if dependency not in dependency_values
        ]
        if missing:
            missing_desc = ", ".join(_dependency_name(dep) for dep in missing)
            raise KeyError(
                f"Missing dependency values for ParameterMapping: {missing_desc}"
            )

        result = self.evaluate(
            {dependency: dependency_values[dependency] for dependency in self.dependencies}
        )
        if len(self.targets) == 1 and not isinstance(result, Mapping):
            return {self.targets[0]: result}

        if not isinstance(result, Mapping):
            raise TypeError(
                "Multi-target ParameterMapping.evaluate() must return a mapping"
            )

        converted = {}
        for key, value in result.items():
            if not isinstance(key, ParamHandle):
                raise TypeError(
                    "ParameterMapping result keys must be ParamHandle objects"
                )
            converted[key] = value

        expected = {_param_key(target): target for target in self.targets}
        actual = {_param_key(target): target for target in converted.keys()}
        if expected != actual:
            missing_keys = expected.keys() - actual.keys()
            extra_keys = actual.keys() - expected.keys()
            problems = []
            if missing_keys:
                problems.append(
                    "missing targets: "
                    + ", ".join(_dependency_name(expected[key]) for key in missing_keys)
                )
            if extra_keys:
                problems.append(
                    "unexpected targets: "
                    + ", ".join(_dependency_name(actual[key]) for key in extra_keys)
                )
            raise ValueError(
                "ParameterMapping returned the wrong target set ("
                + "; ".join(problems)
                + ")"
            )
        return converted

    def describe(
        self,
        axis_keys: Mapping[ParamHandle | ScanVariable, str],
    ) -> dict[str, Any]:
        """Return serialisable metadata describing this mapping."""

        dependencies = []
        for dependency in self.dependencies:
            if dependency in axis_keys:
                dependencies.append(
                    {
                        "kind": "axis",
                        "axis": axis_keys[dependency],
                    }
                )
                continue

            if isinstance(dependency, ParamHandle):
                dependencies.append(
                    {
                        "kind": "param",
                        "path": dependency.owner._stringize_path(),
                        "param": dependency.parameter.describe(),
                    }
                )
                continue

            dependencies.append(
                {
                    "kind": "variable",
                    "variable": dependency.describe(),
                }
            )

        return {
            "description": self.description,
            "targets": [
                {
                    "path": target.owner._stringize_path(),
                    "param": target.parameter.describe(),
                }
                for target in self.targets
            ],
            "dependencies": dependencies,
        }


def _param_key(handle: ParamHandle) -> tuple[int, str]:
    return (id(handle.owner), handle.name)


def _dependency_name(dependency: ParamHandle | ScanVariable) -> str:
    if isinstance(dependency, ParamHandle):
        return f"{dependency.owner._stringize_path()}/{dependency.name}"
    return dependency.name

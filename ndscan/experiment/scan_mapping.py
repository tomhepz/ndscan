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

from .expression import compile_expression
from .parameters import ParamHandle

__all__ = [
    "ScanVariable",
    "FixedPseudoparam",
    "ParameterMapping",
]


@dataclass(frozen=True)
class ScanVariable:
    """Logical scan coordinate used by the runtime, not directly by hardware.

    ``ScanVariable`` exists for ad hoc reparameterisations where a scan should be
    expressed in a convenient logical basis without modifying the fragment tree. The
    corresponding hardware-facing parameters are then supplied by one or more
    ``ParameterMapping`` objects.

    In the persisted scan-site schema these logical coordinates are written under
    ``scan.pseudoparams`` and ``points.pseudoparam_*`` to distinguish them from
    actual fragment parameters whose installed values are recorded under
    ``scan.parameters`` / ``points.param_*``.

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
class FixedPseudoparam:
    """Constant logical value available to text mappings but not scanned point-by-point.

    Fixed pseudoparams exist for request-level reparameterisations where a symbolic
    name is still useful for expressions and offline inspection, even though the value
    never changes during the run. They are written into the scan-site schema under
    ``scan.fixed_pseudoparams`` rather than ``points.pseudoparam_*``.
    """

    variable: ScanVariable
    value: Any

    @property
    def name(self) -> str:
        return self.variable.name


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
    expression: str | None = None

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
        expression: str,
        symbols: Mapping[str, ParamHandle | ScanVariable | Any] | None = None,
        dependencies: Sequence[ParamHandle | ScanVariable] | None = None,
        constants: Mapping[str, Any] | None = None,
        functions: Mapping[str, Callable[..., Any]] | None = None,
        description: str = "",
    ) -> "ParameterMapping":
        """Compile a small text expression into a normal ``ParameterMapping``.

        This is the GUI-oriented counterpart to writing a mapping function in Python.
        The expression language is intentionally tiny and is implemented by the
        standalone :mod:`ndscan.experiment.expression` module.

        ``symbols`` maps expression variable names onto either dynamic dependencies
        (``ParamHandle`` / ``ScanVariable``) or fixed values. Fixed values are baked
        into the compiled expression; dynamic values become ``dependencies`` of the
        resulting mapping.

        ``dependencies`` remains as a convenience for simple cases. It auto-generates
        names from the dependency objects themselves:

        - ``ScanVariable("logical")`` -> ``logical``
        - ``fragment.drive`` -> ``drive``

        Exactly one of ``symbols`` or ``dependencies`` may be provided.

        For multi-target mappings the expression result is broadcast to every target.
        This matches the intended GUI semantics where one row defines one formula for
        one logical quantity, even if that quantity is rebound onto several concrete
        handles selected by a wildcard path.
        """
        if symbols is not None and dependencies is not None:
            raise ValueError(
                "ParameterMapping.from_text() accepts either symbols= or "
                "dependencies=, not both"
            )
        if symbols is None:
            symbols = _symbols_from_dependencies(dependencies or ())
        elif not isinstance(symbols, Mapping):
            raise TypeError("symbols must be a mapping when specified")

        symbol_map = dict(symbols)
        if constants is None:
            constant_map: dict[str, Any] = {}
        else:
            constant_map = dict(constants)
        overlap = set(symbol_map.keys()) & set(constant_map.keys())
        if overlap:
            raise ValueError(
                "Expression symbols and explicit constants overlap: "
                + ", ".join(sorted(overlap))
            )

        dynamic_symbols: dict[str, ParamHandle | ScanVariable] = {}
        fixed_symbols: dict[str, Any] = {}
        for name, value in symbol_map.items():
            if not isinstance(name, str):
                raise TypeError("Expression symbol names must be strings")
            if isinstance(value, (ParamHandle, ScanVariable)):
                dynamic_symbols[name] = value
            else:
                fixed_symbols[name] = value

        compiled = compile_expression(
            expression,
            variable_names=dynamic_symbols.keys(),
            constants={**constant_map, **fixed_symbols},
            functions=functions,
        )
        referenced_symbol_map = {
            name: dynamic_symbols[name] for name in compiled.variable_names
        }
        referenced_dependencies = tuple(
            referenced_symbol_map[name] for name in compiled.variable_names
        )

        targets = tuple(targets)
        if len(targets) == 1:
            return cls(
                targets=targets,
                dependencies=referenced_dependencies,
                evaluate=_make_text_mapping_evaluator(compiled, referenced_symbol_map),
                description=description,
                expression=expression,
            )

        return cls(
            targets=targets,
            dependencies=referenced_dependencies,
            evaluate=_make_text_broadcast_mapping_evaluator(
                compiled, referenced_symbol_map, targets
            ),
            description=description,
            expression=expression,
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
                key = axis_keys[dependency]
                kind = "pseudoparam" if key.startswith("pseudoparam_") else "parameter"
                dependencies.append(
                    {
                        "kind": kind,
                        "key": key,
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
            **({} if self.expression is None else {"expression": self.expression}),
        }


def _param_key(handle: ParamHandle) -> tuple[int, str]:
    return (id(handle.owner), handle.name)


def _dependency_name(dependency: ParamHandle | ScanVariable) -> str:
    if isinstance(dependency, ParamHandle):
        return f"{dependency.owner._stringize_path()}/{dependency.name}"
    return dependency.name


def _symbols_from_dependencies(
    dependencies: Sequence[ParamHandle | ScanVariable],
) -> dict[str, ParamHandle | ScanVariable]:
    result: dict[str, ParamHandle | ScanVariable] = {}
    for dependency in dependencies:
        name = dependency.name
        if name in result:
            raise ValueError(
                "Auto-generated dependency names are not unique; use symbols= "
                f"instead (duplicate name: {name!r})"
            )
        result[name] = dependency
    return result


def _make_text_mapping_evaluator(
    compiled_expression,
    dynamic_symbols: Mapping[str, ParamHandle | ScanVariable],
) -> Callable[[Mapping[ParamHandle | ScanVariable, Any]], Any]:
    def evaluate(values: Mapping[ParamHandle | ScanVariable, Any]) -> Any:
        return compiled_expression.evaluate(
            {name: values[dependency] for name, dependency in dynamic_symbols.items()}
        )

    return evaluate


def _make_text_broadcast_mapping_evaluator(
    compiled_expression,
    dynamic_symbols: Mapping[str, ParamHandle | ScanVariable],
    targets: Sequence[ParamHandle],
) -> Callable[[Mapping[ParamHandle | ScanVariable, Any]], dict[ParamHandle, Any]]:
    def evaluate(
        values: Mapping[ParamHandle | ScanVariable, Any],
    ) -> dict[ParamHandle, Any]:
        result = compiled_expression.evaluate(
            {name: values[dependency] for name, dependency in dynamic_symbols.items()}
        )
        return {target: result for target in targets}

    return evaluate

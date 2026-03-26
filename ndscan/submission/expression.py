"""Small safe expression compiler used by host-scan text rebinds.

This module is intentionally independent from the rest of ndscan:

- it does not import fragment, parameter, or runtime types,
- it only knows about parsing and evaluating a tiny arithmetic expression language,
- callers provide the allowed variable names, constants, and functions explicitly.

The implementation is deliberately simple rather than clever:

1. parse the user string with :mod:`ast` in ``eval`` mode,
2. walk the tree and reject any syntax outside a small whitelist,
3. compile the validated tree to a Python code object,
4. evaluate that code object in a namespace containing only the caller-supplied
   values/functions/constants.

Using Python's parser keeps the expression language familiar and makes the validator
easy to audit. The validator is the security boundary; the later ``eval()`` only sees a
tree that has already been checked to contain no attribute access, indexing, imports,
comprehensions, assignments, or arbitrary function calls.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import CodeType
from typing import Any

__all__ = [
    "DEFAULT_EXPRESSION_CONSTANTS",
    "DEFAULT_EXPRESSION_FUNCTIONS",
    "ExpressionCompileError",
    "ExpressionNameError",
    "ExpressionSyntaxError",
    "ExpressionValidationError",
    "CompiledExpression",
    "compile_expression",
]


DEFAULT_EXPRESSION_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "floor": math.floor,
    "ceil": math.ceil,
}

DEFAULT_EXPRESSION_CONSTANTS: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
}


class ExpressionCompileError(ValueError):
    """Base class for expression-compilation errors."""


class ExpressionSyntaxError(ExpressionCompileError):
    """Raised when the input is not even syntactically a Python expression."""


class ExpressionNameError(ExpressionCompileError):
    """Raised when the expression references an unknown or reserved name."""


class ExpressionValidationError(ExpressionCompileError):
    """Raised when the expression uses unsupported syntax."""


@dataclass(frozen=True)
class CompiledExpression:
    """Validated expression plus the minimal information needed to evaluate it."""

    source: str
    variable_names: tuple[str, ...]
    constant_names: tuple[str, ...]
    function_names: tuple[str, ...]
    _code: CodeType = field(repr=False)
    _constants: Mapping[str, Any] = field(repr=False)
    _functions: Mapping[str, Callable[..., Any]] = field(repr=False)

    def evaluate(self, variables: Mapping[str, Any]) -> Any:
        """Evaluate the expression with the given variable bindings.

        ``variables`` needs to provide the names listed in ``variable_names``. Extra
        keys are ignored to keep the caller-side plumbing simple.
        """

        missing = [name for name in self.variable_names if name not in variables]
        if missing:
            raise KeyError(
                "Missing values for expression variables: " + ", ".join(missing)
            )

        namespace: dict[str, Any] = {}
        namespace.update(self._functions)
        namespace.update(self._constants)
        namespace.update({name: variables[name] for name in self.variable_names})
        return eval(self._code, {"__builtins__": {}}, namespace)


def compile_expression(
    source: str,
    *,
    variable_names: Iterable[str] = (),
    constants: Mapping[str, Any] | None = None,
    functions: Mapping[str, Callable[..., Any]] | None = None,
) -> CompiledExpression:
    """Parse, validate, and compile a tiny expression language.

    ``variable_names`` supplies the identifiers that the expression is allowed to use
    as run-time inputs. ``constants`` and ``functions`` are named helpers made
    available during evaluation.
    """

    if not isinstance(source, str):
        raise TypeError("Expression source must be a string")

    variable_names = tuple(variable_names)
    # Caller-supplied constants/functions extend the default vocabulary rather than
    # replacing it outright. This keeps helpers like ``pi`` or ``sin`` available even
    # when the caller adds domain-specific names of its own.
    constants = {
        **DEFAULT_EXPRESSION_CONSTANTS,
        **({} if constants is None else dict(constants)),
    }
    functions = {
        **DEFAULT_EXPRESSION_FUNCTIONS,
        **({} if functions is None else dict(functions)),
    }

    _validate_identifier_collection(variable_names, "variable")
    _validate_identifier_collection(constants.keys(), "constant")
    _validate_identifier_collection(functions.keys(), "function")
    _validate_no_name_collisions(variable_names, constants, functions)

    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionSyntaxError(str(exc)) from None

    validator = _ExpressionValidator(
        allowed_variables=set(variable_names),
        allowed_constants=set(constants.keys()),
        allowed_functions=set(functions.keys()),
    )
    validator.visit(tree)

    return CompiledExpression(
        source=source,
        variable_names=tuple(validator.variable_names),
        constant_names=tuple(validator.constant_names),
        function_names=tuple(validator.function_names),
        _code=compile(tree, "<ndscan expression>", "eval"),
        _constants=constants,
        _functions=functions,
    )


class _ExpressionValidator(ast.NodeVisitor):
    """Whitelist validator for the supported expression subset."""

    _allowed_binary_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
    )
    _allowed_unary_ops = (
        ast.UAdd,
        ast.USub,
    )
    _allowed_constant_types = (bool, int, float, str, type(None))

    def __init__(
        self,
        *,
        allowed_variables: set[str],
        allowed_constants: set[str],
        allowed_functions: set[str],
    ) -> None:
        self._allowed_variables = allowed_variables
        self._allowed_constants = allowed_constants
        self._allowed_functions = allowed_functions
        self.variable_names: list[str] = []
        self.constant_names: list[str] = []
        self.function_names: list[str] = []
        self._seen_variables = set[str]()
        self._seen_constants = set[str]()
        self._seen_functions = set[str]()

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self._allowed_binary_ops):
            raise ExpressionValidationError(
                f"Unsupported binary operator: {type(node.op).__name__}"
            )
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self._allowed_unary_ops):
            raise ExpressionValidationError(
                f"Unsupported unary operator: {type(node.op).__name__}"
            )
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise ExpressionValidationError(
                "Only direct calls to explicitly allowed functions are supported"
            )
        func_name = node.func.id
        if func_name not in self._allowed_functions:
            raise ExpressionNameError(f"Unknown function: {func_name}")
        if node.keywords:
            raise ExpressionValidationError("Keyword arguments are not supported")
        for arg in node.args:
            self.visit(arg)
        if func_name not in self._seen_functions:
            self._seen_functions.add(func_name)
            self.function_names.append(func_name)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            raise ExpressionValidationError("Assignment is not supported")
        name = node.id
        if name in self._allowed_variables:
            if name not in self._seen_variables:
                self._seen_variables.add(name)
                self.variable_names.append(name)
            return
        if name in self._allowed_constants:
            if name not in self._seen_constants:
                self._seen_constants.add(name)
                self.constant_names.append(name)
            return
        if name in self._allowed_functions:
            raise ExpressionValidationError(
                f"Function name '{name}' must be called as {name}(...)"
            )
        raise ExpressionNameError(f"Unknown name: {name}")

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, self._allowed_constant_types):
            raise ExpressionValidationError(
                f"Unsupported literal type: {type(node.value).__name__}"
            )

    def generic_visit(self, node: ast.AST) -> None:
        raise ExpressionValidationError(
            f"Unsupported syntax: {type(node).__name__}"
        )


def _validate_identifier_collection(names: Iterable[str], kind: str) -> None:
    seen = set[str]()
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"{kind.capitalize()} names must be strings")
        if not name.isidentifier():
            raise ValueError(
                f"{kind.capitalize()} name {name!r} is not a valid Python identifier"
            )
        if name in seen:
            raise ValueError(f"Duplicate {kind} name: {name!r}")
        seen.add(name)


def _validate_no_name_collisions(
    variable_names: Iterable[str],
    constants: Mapping[str, Any],
    functions: Mapping[str, Callable[..., Any]],
) -> None:
    variable_names = set(variable_names)
    constant_names = set(constants.keys())
    function_names = set(functions.keys())

    variable_constant = variable_names & constant_names
    if variable_constant:
        raise ValueError(
            "Variables and constants share reserved names: "
            + ", ".join(sorted(variable_constant))
        )

    variable_function = variable_names & function_names
    if variable_function:
        raise ValueError(
            "Variables and functions share reserved names: "
            + ", ".join(sorted(variable_function))
        )

    constant_function = constant_names & function_names
    if constant_function:
        raise ValueError(
            "Constants and functions share reserved names: "
            + ", ".join(sorted(constant_function))
        )

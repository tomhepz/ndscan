"""Helpers for parsing and safely evaluating scan relation expressions.

Typical workflow:

1. Rewrite bracket references (e.g. ``[fragment.param]``) to temporary aliases:

   >>> rewrite_bracket_param_refs("[foo.bar]**2 + [foo.bar]")
   ('__param_0**2 + __param_0', OrderedDict([('foo.bar', '__param_0')]))

2. Compile a safe expression with explicit dependency aliases:

   >>> fn = compile_safe_relation_expr("p**2 + 4", ("p",))
   >>> fn(3.0)
   13.0

Only a restricted AST subset is accepted (arithmetic, comparisons, conditionals, and
calls to a small allowlist of math functions). Anything else raises
``RelationExpressionError``.
"""

import ast
import math
import re
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

__all__ = [
    "RelationExpressionError",
    "rewrite_bracket_param_refs",
    "compile_safe_relation_expr",
]


class RelationExpressionError(ValueError):
    """Raised when a relation expression is malformed or unsafe."""


_BRACKET_PARAM_RE = re.compile(r"\[([^\[\]]+)\]")


def rewrite_bracket_param_refs(expr: str) -> tuple[str, OrderedDict[str, str]]:
    """Rewrite ``[param.fqn]`` references to generated variable aliases.

    Returns a pair ``(rewritten_expr, token_to_alias)``, where ``token_to_alias``
    preserves first-seen order.

    Example:

        >>> rewrite_bracket_param_refs("[a.b] + [c.d] + [a.b]")
        ('__param_0 + __param_1 + __param_0', OrderedDict([('a.b', '__param_0'), ('c.d', '__param_1')]))
    """
    token_to_alias = OrderedDict[str, str]()

    def replace(match: re.Match) -> str:
        token = match.group(1).strip()
        if not token:
            raise RelationExpressionError(
                "Empty parameter reference '[]' is not allowed in relation expression"
            )
        alias = token_to_alias.get(token)
        if alias is None:
            alias = f"__param_{len(token_to_alias)}"
            token_to_alias[token] = alias
        return alias

    rewritten = _BRACKET_PARAM_RE.sub(replace, expr)
    return rewritten, token_to_alias


_ALLOWED_BINOPS = {
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
}

_ALLOWED_UNARYOPS = {ast.UAdd, ast.USub}
_ALLOWED_BOOLOPS = {ast.And, ast.Or}
_ALLOWED_CMPOPS = {
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
}

_ALLOWED_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
}

_ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


class _SafeExprValidator(ast.NodeVisitor):
    def __init__(self, allowed_names: set[str], allowed_functions: set[str]):
        self._allowed_names = allowed_names
        self._allowed_functions = allowed_functions

    def visit_Expression(self, node: ast.Expression) -> Any:
        self.visit(node.body)

    def visit_Constant(self, _node: ast.Constant) -> Any:
        return None

    # Python <3.8 compatibility shim (not expected to be used here, but harmless)
    def visit_Num(self, _node: ast.Num) -> Any:  # pragma: no cover
        return None

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self._allowed_names and node.id not in self._allowed_functions:
            raise RelationExpressionError(
                f"Unknown name '{node.id}' in relation expression"
            )

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if type(node.op) not in _ALLOWED_BINOPS:
            raise RelationExpressionError(
                f"Operator '{type(node.op).__name__}' not allowed in relation expression"
            )
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if type(node.op) not in _ALLOWED_UNARYOPS:
            raise RelationExpressionError(
                f"Unary operator '{type(node.op).__name__}' not allowed"
            )
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if type(node.op) not in _ALLOWED_BOOLOPS:
            raise RelationExpressionError(
                f"Boolean operator '{type(node.op).__name__}' not allowed"
            )
        for value in node.values:
            self.visit(value)

    def visit_Compare(self, node: ast.Compare) -> Any:
        for op in node.ops:
            if type(op) not in _ALLOWED_CMPOPS:
                raise RelationExpressionError(
                    f"Comparison operator '{type(op).__name__}' not allowed"
                )
        self.visit(node.left)
        for c in node.comparators:
            self.visit(c)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise RelationExpressionError(
                "Only direct function calls are allowed in relation expressions"
            )
        if node.func.id not in self._allowed_functions:
            raise RelationExpressionError(
                f"Function '{node.func.id}' is not allowed in relation expressions"
            )
        if node.keywords:
            raise RelationExpressionError(
                "Keyword arguments are not allowed in relation expressions"
            )
        for arg in node.args:
            self.visit(arg)

    def visit_List(self, node: ast.List) -> Any:
        for elt in node.elts:
            self.visit(elt)

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        for elt in node.elts:
            self.visit(elt)

    def visit_Dict(self, node: ast.Dict) -> Any:
        for k in node.keys:
            if k is not None:
                self.visit(k)
        for v in node.values:
            self.visit(v)

    def generic_visit(self, node: ast.AST) -> Any:
        raise RelationExpressionError(
            f"Expression element '{type(node).__name__}' is not allowed"
        )


def compile_safe_relation_expr(expr: str, dep_aliases: tuple[str, ...]) -> Callable[..., Any]:
    """Compile a restricted expression into a callable ``fn(*dep_values)``.

    Example:

        >>> fn = compile_safe_relation_expr("sqrt(p) + 1", ("p",))
        >>> fn(9.0)
        4.0
    """
    stripped = expr.strip()
    if not stripped:
        raise RelationExpressionError("Relation expression must not be empty")

    alias_set = set(dep_aliases)
    if len(alias_set) != len(dep_aliases):
        raise RelationExpressionError("Duplicate dependency aliases are not allowed")
    collisions = alias_set & (set(_ALLOWED_FUNCTIONS.keys()) | set(_ALLOWED_CONSTANTS.keys()))
    if collisions:
        names = ", ".join(sorted(collisions))
        raise RelationExpressionError(
            "Dependency aliases shadow reserved relation names: " + names
        )

    try:
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError as e:
        raise RelationExpressionError(
            f"Invalid relation expression syntax: {e.msg}"
        ) from None

    _SafeExprValidator(
        allowed_names=alias_set | set(_ALLOWED_CONSTANTS.keys()),
        allowed_functions=set(_ALLOWED_FUNCTIONS.keys()),
    ).visit(tree)

    code = compile(tree, "<ndscan relation expr>", "eval")
    globals_dict = {"__builtins__": {}}
    static_locals = dict(_ALLOWED_FUNCTIONS)
    static_locals.update(_ALLOWED_CONSTANTS)

    def evaluate(*dep_values):
        if len(dep_values) != len(dep_aliases):
            raise RelationExpressionError(
                "Relation expression called with incorrect number of dependency values"
            )
        eval_locals = dict(static_locals)
        eval_locals.update(zip(dep_aliases, dep_values, strict=False))
        return eval(code, globals_dict, eval_locals)

    return evaluate

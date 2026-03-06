import unittest

from ndscan.experiment.relation_expressions import (
    RelationExpressionError,
    compile_safe_relation_expr,
    rewrite_bracket_param_refs,
)


class RelationExpressionCase(unittest.TestCase):
    def test_rewrite_bracket_refs_deduplicates_tokens(self):
        rewritten, token_map = rewrite_bracket_param_refs("[a.b] + [c.d] + [a.b]")
        self.assertEqual(rewritten, "__param_0 + __param_1 + __param_0")
        self.assertEqual(
            list(token_map.items()),
            [("a.b", "__param_0"), ("c.d", "__param_1")],
        )

    def test_rewrite_bracket_refs_rejects_empty_token(self):
        with self.assertRaisesRegex(RelationExpressionError, "Empty parameter reference"):
            rewrite_bracket_param_refs("[   ] + 1")

    def test_compile_safe_relation_expr_basic_arithmetic(self):
        fn = compile_safe_relation_expr("p**2 + 4", ("p",))
        self.assertEqual(fn(3.0), 13.0)
        self.assertEqual(fn(6.0), 40.0)

    def test_compile_safe_relation_expr_allows_math_functions_and_constants(self):
        fn = compile_safe_relation_expr("sin(pi / 2) + sqrt(p)", ("p",))
        self.assertAlmostEqual(fn(4.0), 3.0)

    def test_compile_safe_relation_expr_allows_if_expression(self):
        fn = compile_safe_relation_expr("p if p > 0 else -p", ("p",))
        self.assertEqual(fn(3.0), 3.0)
        self.assertEqual(fn(-3.0), 3.0)

    def test_compile_safe_relation_expr_rejects_unknown_name(self):
        with self.assertRaisesRegex(RelationExpressionError, "Unknown name 'q'"):
            compile_safe_relation_expr("p + q", ("p",))

    def test_compile_safe_relation_expr_rejects_unsafe_function(self):
        with self.assertRaisesRegex(RelationExpressionError, "not allowed"):
            compile_safe_relation_expr("__import__('os')", tuple())

    def test_compile_safe_relation_expr_rejects_attribute_function_call(self):
        with self.assertRaisesRegex(RelationExpressionError, "direct function calls"):
            compile_safe_relation_expr("math.sin(p)", ("p",))

    def test_compile_safe_relation_expr_rejects_keyword_args(self):
        with self.assertRaisesRegex(RelationExpressionError, "Keyword arguments"):
            compile_safe_relation_expr("pow(p, y=2)", ("p",))

    def test_compile_safe_relation_expr_rejects_reserved_alias_collision(self):
        with self.assertRaisesRegex(RelationExpressionError, "shadow reserved"):
            compile_safe_relation_expr("sqrt(p)", ("sqrt", "p"))

    def test_compiled_relation_expr_checks_arity(self):
        fn = compile_safe_relation_expr("p + 1", ("p",))
        with self.assertRaisesRegex(RelationExpressionError, "incorrect number"):
            fn()


if __name__ == "__main__":
    unittest.main()

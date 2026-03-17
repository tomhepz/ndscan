import math
import unittest

from ndscan.experiment.expression import (
    ExpressionNameError,
    ExpressionSyntaxError,
    ExpressionValidationError,
    compile_expression,
)


class ExpressionCompilerTest(unittest.TestCase):
    def test_compile_expression_tracks_names_in_first_use_order(self):
        compiled = compile_expression(
            "sin(x) + offset * y + pi",
            variable_names=["x", "y"],
            constants={"offset": 2.0},
        )

        self.assertEqual(compiled.variable_names, ("x", "y"))
        self.assertEqual(compiled.constant_names, ("offset", "pi"))
        self.assertEqual(compiled.function_names, ("sin",))
        self.assertAlmostEqual(
            compiled.evaluate({"x": 0.5, "y": 3.0}),
            math.sin(0.5) + 6.0 + math.pi,
        )

    def test_compile_expression_allows_default_helpers_alongside_custom_constants(self):
        compiled = compile_expression(
            "cos(theta) + bias",
            variable_names=["theta"],
            constants={"bias": 1.5},
        )

        self.assertAlmostEqual(compiled.evaluate({"theta": 0.0}), 2.5)

    def test_compile_expression_rejects_unknown_name(self):
        with self.assertRaisesRegex(ExpressionNameError, "Unknown name: bias"):
            compile_expression("x + bias", variable_names=["x"])

    def test_compile_expression_rejects_attribute_access(self):
        with self.assertRaisesRegex(
            ExpressionValidationError,
            "Only direct calls to explicitly allowed functions are supported",
        ):
            compile_expression("math.sin(x)", variable_names=["x"])

    def test_compile_expression_rejects_comparisons(self):
        with self.assertRaisesRegex(
            ExpressionValidationError, "Unsupported syntax: Compare"
        ):
            compile_expression("x < 3", variable_names=["x"])

    def test_compile_expression_requires_function_names_to_be_called(self):
        with self.assertRaisesRegex(
            ExpressionValidationError,
            "must be called as sin",
        ):
            compile_expression("sin + x", variable_names=["x"])

    def test_compile_expression_rejects_invalid_syntax(self):
        with self.assertRaises(ExpressionSyntaxError):
            compile_expression("x +")

    def test_compile_expression_reports_missing_variables_at_evaluation_time(self):
        compiled = compile_expression("x + y", variable_names=["x", "y"])

        with self.assertRaisesRegex(KeyError, "Missing values for expression variables"):
            compiled.evaluate({"x": 1.0})

    def test_compile_expression_rejects_name_collisions(self):
        with self.assertRaisesRegex(ValueError, "share reserved names"):
            compile_expression("sin(x)", variable_names=["x", "sin"])


if __name__ == "__main__":
    unittest.main()

"""Host-runtime gradient-descent example.

This example shows the first adaptive point policy in the new host runtime:
finite-difference gradient descent over four continuously scanned parameters.

The fragment itself is deliberately simple. It exposes a convex quadratic loss with a
known minimum, so the resulting point stream is easy to inspect:

- each optimisation batch evaluates one centre point,
- then one positive and one negative probe per dimension,
- the next centre point is chosen from the observed gradient.

Because the policy needs all probe points from one optimisation step together, the
request uses ``ExecutionPolicy(max_points_per_batch=9)`` for the four-dimensional case.
"""

from __future__ import annotations

from ndscan.experiment import (
    ExecutionPolicy,
    ExpFragment,
    FloatChannel,
    FloatParam,
    GradientDescentPointPolicy,
    ScanRequest,
    make_fragment_host_scan_exp,
)


class QuadraticLossFragment(ExpFragment):
    """Four-dimensional convex loss surface with a known minimum."""

    def build_fragment(self):
        self.setattr_param("x0", FloatParam, "x0", default=0.0)
        self.setattr_param("x1", FloatParam, "x1", default=0.0)
        self.setattr_param("x2", FloatParam, "x2", default=0.0)
        self.setattr_param("x3", FloatParam, "x3", default=0.0)
        self.setattr_result("loss", FloatChannel)

    def run_once(self):
        optimum = (1.0, -2.0, 0.5, 3.0)
        values = (
            self.x0.get(),
            self.x1.get(),
            self.x2.get(),
            self.x3.get(),
        )
        self.loss.push(
            sum((value - target) ** 2 for value, target in zip(values, optimum))
        )


HostRuntimeGradientDescent = make_fragment_host_scan_exp(
    QuadraticLossFragment,
    lambda fragment: ScanRequest(
        axes=(fragment.x0, fragment.x1, fragment.x2, fragment.x3),
        point_policy=GradientDescentPointPolicy(
            initial_point=(0.0, 0.0, 0.0, 0.0),
            objective=lambda observation: observation.channel_values["channel_0"],
            probe_steps=(0.1, 0.1, 0.1, 0.1),
            learning_rate=0.5,
            max_iterations=3,
            gradient_tolerance=1e-9,
            objective_description="quadratic loss",
        ),
        execution_policy=ExecutionPolicy(max_points_per_batch=9),
        metadata={"demo_name": "host_runtime_gradient_descent"},
    ),
)
HostRuntimeGradientDescent.__doc__ = "Host-runtime gradient-descent optimisation"

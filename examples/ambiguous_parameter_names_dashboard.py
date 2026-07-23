"""Small dashboard example for repeated parameter names/FQNs.

This file intentionally exposes the same fragment tree through both dashboard paths:

- ``LegacyAmbiguousParameterNames`` uses the legacy ndscan submission/runtime path.
- ``PreparedAmbiguousParameterNamesDashboard`` uses the scan-submission dashboard submission path.

The fragment tree contains:

- two *different* subfragment classes with the same parameter name ``shared``,
- two instances of the *same* subfragment class with the same parameter name ``twin``,
- several parameters that are not returned by ``get_always_shown_params()``.

Open both experiments in the dashboard to compare how the editor labels:

- same name, different FQN,
- same name, same FQN, different path,
- hidden parameters that must be added manually.
"""

from __future__ import annotations

from artiq.experiment import *

from ndscan.define import *
from ndscan.runtime.api import *
from ndscan.scan import *


class AlphaSharedFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("shared", FloatParam, "shared", 1.0)
        self.setattr_param("hidden_alpha", FloatParam, "hidden alpha", 0.0)


class BetaSharedFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("shared", FloatParam, "shared", 2.0)
        self.setattr_param("hidden_beta", FloatParam, "hidden beta", 0.0)


class TwinFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("twin", FloatParam, "twin", 0.0)
        self.setattr_param("hidden_twin", FloatParam, "hidden twin", 0.0)


class AmbiguousParameterNamesFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("scale", FloatParam, "scale", 1.0)
        self.setattr_param("hidden_root", FloatParam, "hidden root", 0.0)

        self.setattr_fragment("alpha", AlphaSharedFragment)
        self.setattr_fragment("beta", BetaSharedFragment)
        self.setattr_fragment("left", TwinFragment)
        self.setattr_fragment("right", TwinFragment)

        self.setattr_result("total", FloatChannel)

    def get_always_shown_params(self):
        return [
            self.scale,
            self.alpha.shared,
            self.left.twin,
            self.right.twin,
        ]

    def run_once(self):
        total = self.scale.get() * (
            self.alpha.shared.get()
            + self.beta.shared.get()
            + self.left.twin.get()
            + self.right.twin.get()
            + self.hidden_root.get()
            + self.alpha.hidden_alpha.get()
            + self.beta.hidden_beta.get()
            + self.left.hidden_twin.get()
            + self.right.hidden_twin.get()
        )
        self.total.push(total)


LegacyAmbiguousParameterNames = make_fragment_scan_exp(
    AmbiguousParameterNamesFragment
)

PreparedAmbiguousParameterNamesDashboard = make_fragment_prepared_dashboard_scan_exp(
    AmbiguousParameterNamesFragment
)

"""
Minimal lifecycle trace for ndscan fragments + subscan.

Run ``LifecycleTraceScan`` from the ARTIQ explorer and watch the console output.

For the simplest run, leave scan axes empty (single top-level point).
To see repeated top-level points, scan ``num_inner_points`` in the argument editor
with e.g. a list scan over ``[2, 3, 4]``.

This example prints from:
- outer fragment: host_setup/device_setup/run_once
- subscan fragment: host_setup/device_setup/run_once
- scanned leaf fragment: host_setup/device_setup/run_once
"""

from ndscan.experiment import *


def _trace(name: str, where: str, extra: str = "") -> None:
    message = f"[trace] {name}.{where}"
    if extra:
        message += f" | {extra}"
    print(message)


class Leaf(ExpFragment):
    """The fragment that is actually scanned inside the subscan."""

    def build_fragment(self):
        self.setattr_param("x", FloatParam, "Leaf x", default=0.0)
        self.setattr_result("y", FloatChannel, "Leaf y")

    def host_setup(self):
        _trace("Leaf", "host_setup")
        super().host_setup()

    def device_setup(self):
        _trace("Leaf", "device_setup", f"x={self.x.get()}")
        self.device_setup_subfragments()

    def run_once(self):
        _trace("Leaf", "run_once", f"x={self.x.get()}")
        self.y.push(self.x.get() + 1.0)


class LeafSubscan(SubscanExpFragment):
    """Wraps ``Leaf`` in a subscan."""

    def build_fragment(self):
        self.setattr_fragment("leaf", Leaf)
        self.setattr_param(
            "num_inner_points",
            IntParam,
            "Number of subscan points",
            default=3,
            min=2,
        )
        super().build_fragment(self, self.leaf, [(self.leaf, "x")])

    def _configure(self):
        n = self.num_inner_points.use()
        generator = LinearGenerator(0.0, float(n - 1), n, randomise_order=False)
        options = ScanOptions(
            num_repeats=1,
            num_repeats_per_point=1,
            randomise_order_globally=False,
        )
        self.configure([(self.leaf.x, generator)], options=options)
        _trace("LeafSubscan", "configure", f"num_inner_points={n}")

    def host_setup(self):
        _trace("LeafSubscan", "host_setup(start)")
        self._configure()
        super().host_setup()
        _trace("LeafSubscan", "host_setup(end)")

    def device_setup(self):
        _trace("LeafSubscan", "device_setup")
        # The scanned fragment was detached; this only affects regular subfragments.
        self.device_setup_subfragments()

    def run_once(self):
        _trace("LeafSubscan", "run_once(start)")
        # Equivalent to SubscanExpFragment.run_once(), but with trace prints.
        self._subscan.acquire()
        _trace("LeafSubscan", "run_once(end)")


class LifecycleTrace(ExpFragment):
    """Top-level fragment.

    Scan ``num_inner_points`` at the top level to see outer-point repetition and
    changing subscan sizes.
    """

    def build_fragment(self):
        self.setattr_param(
            "num_inner_points",
            IntParam,
            "Inner points (forwarded to subscan)",
            default=3,
            min=2,
        )
        self.setattr_result("outer_value", IntChannel, "Echo of num_inner_points")

        self.setattr_fragment("subscan", LeafSubscan)
        self.subscan.bind_param("num_inner_points", self.num_inner_points)

    def host_setup(self):
        _trace("LifecycleTrace", "host_setup(start)")
        super().host_setup()
        _trace("LifecycleTrace", "host_setup(end)")

    def device_setup(self):
        _trace(
            "LifecycleTrace",
            "device_setup",
            f"num_inner_points={self.num_inner_points.get()}",
        )
        self.device_setup_subfragments()

    def run_once(self):
        _trace(
            "LifecycleTrace",
            "run_once(start)",
            f"num_inner_points={self.num_inner_points.get()}",
        )
        self.outer_value.push(self.num_inner_points.get())
        self.subscan.run_once()
        _trace("LifecycleTrace", "run_once(end)")


LifecycleTraceScan = make_fragment_scan_exp(LifecycleTrace)

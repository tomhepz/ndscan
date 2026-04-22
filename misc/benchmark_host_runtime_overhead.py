"""Small prepared-runtime host/kernel-boundary overhead benchmark.

This benchmark is intentionally simple. It runs a few fragments through
``PreparedScan`` and reports first-run and warm-run wall-clock timings. Kernel-marked
fragments use a tiny fake core driver that times host-observed calls through
``core.run(...)``; this gives the same accounting shape we want on real hardware, but
it is not a substitute for real core-device timing.

Pass ``--real-core`` to add an opt-in row that uses an ARTIQ ``DeviceDB`` and a real
``core`` device, following the same route as the kernel tests. In that mode, the
kernel columns are host-observed wall time inside the real ``core.run(...)`` boundary,
which includes compilation/transfer/execution/wait from the host's perspective.

Run from the repository root, for example:

    /home/lab/artiq-files/install/ndscan/.venv/bin/python \
        misc/benchmark_host_runtime_overhead.py --points 2000 --repeats 5

To add the hardware-backed row:

    ARTIQ_ROOT=/home/lab/artiq-files/dnamic-lab \
    /home/lab/artiq-files/install/ndscan/.venv/bin/python \
        misc/benchmark_host_runtime_overhead.py --real-core --real-core-points 200

To also benchmark the nested TTL example:

    ARTIQ_ROOT=/home/lab/artiq-files/dnamic-lab \
    /home/lab/artiq-files/install/ndscan/.venv/bin/python \
        misc/benchmark_host_runtime_overhead.py --nested-ttl

To benchmark a nested kernel that waits a known duration before returning:

    ARTIQ_ROOT=/home/lab/artiq-files/dnamic-lab \
    /home/lab/artiq-files/install/ndscan/.venv/bin/python \
        misc/benchmark_host_runtime_overhead.py --nested-wait --nested-wait-us 5000

For host call attribution:

    /home/lab/artiq-files/install/ndscan/.venv/bin/python \
        misc/benchmark_host_runtime_overhead.py --profile /tmp/ndscan_bench.prof

Then inspect with:

    /home/lab/artiq-files/install/ndscan/.venv/bin/python -m pstats \
        /tmp/ndscan_bench.prof
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import os
import pstats
import time
import unittest.mock
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from artiq.language.core import kernel
from artiq.language.environment import ProcessArgumentManager
from artiq.master.databases import DeviceDB
from artiq.master.worker_db import DatasetManager, DeviceManager
from sipyco.sync_struct import process_mod

from ndscan.define import ArrayChannel, ExpFragment, FloatChannel, FloatParam
from ndscan.runtime.api import ExecutionPolicy, PreparedScan
from ndscan.scan import ScanRequest


class _MockDatasetDB:
    def __init__(self):
        self.data = dict()

    def get(self, key):
        return self.data[key][1]

    def update(self, mod):
        # DatasetManager sends sync_struct mutations. Copy them so repeated benchmark
        # runs do not accidentally share mutable payloads through the mock DB.
        process_mod(self.data, copy.deepcopy(mod))

    def delete(self, key):
        del self.data[key]


class _MockScheduler:
    rid = 0

    def check_pause(self) -> bool:
        return False

    def pause(self):
        pass


class _TimingCore:
    """Minimal core-device stand-in that times host-observed kernel calls."""

    def __init__(self):
        self.elapsed_s = 0.0
        self.num_calls = 0
        self._depth = 0

    def reset_timing(self) -> None:
        self.elapsed_s = 0.0
        self.num_calls = 0
        self._depth = 0

    def run(self, kernel_fn, args, kwargs):
        outermost = self._depth == 0
        if outermost:
            start = time.perf_counter()
            self.num_calls += 1
        self._depth += 1
        try:
            embedded = getattr(kernel_fn, "artiq_embedded")
            function = embedded.function
            if isinstance(function, str):
                context = {"np": np}
                exec(function, context)
                function = context["kernel_from_string_fn"]
            return function(*args, **kwargs)
        finally:
            self._depth -= 1
            if outermost:
                self.elapsed_s += time.perf_counter() - start


class _RealCoreRunTimer:
    """Records host-observed calls through an ARTIQ core device.

    This patches the core instance's ``run`` method in place instead of wrapping the
    whole device object. ARTIQ's compiler checks core-device identity across nested
    kernel calls, so fragments and helper runners must still hold the real core object.
    """

    def __init__(self, core):
        self.core = core
        self._original_run = core.run
        self._execute_string_kernels_locally = (
            core.__class__.__module__ == "artiq.sim.devices"
        )
        self.elapsed_s = 0.0
        self.num_calls = 0
        self._depth = 0
        core.run = self.run

    def reset_timing(self) -> None:
        self.elapsed_s = 0.0
        self.num_calls = 0
        self._depth = 0

    def run(self, kernel_fn, args, kwargs):
        outermost = self._depth == 0
        if outermost:
            start = time.perf_counter()
            self.num_calls += 1
        self._depth += 1
        try:
            embedded = getattr(kernel_fn, "artiq_embedded", None)
            function = None if embedded is None else embedded.function
            if self._execute_string_kernels_locally and isinstance(function, str):
                context = {"np": np}
                exec(function, context)
                function = context["kernel_from_string_fn"]
                return function(*args, **kwargs)
            return self._original_run(kernel_fn, args, kwargs)
        finally:
            self._depth -= 1
            if outermost:
                self.elapsed_s += time.perf_counter() - start

    def close(self) -> None:
        self.core.run = self._original_run


class _MockDeviceDB:
    devices = {"core": {"type": "dummy"}}

    def get(self, key):
        return self.devices[key]

    def get_device_db(self):
        return self.devices


class _BenchmarkEnvironmentProtocol(Protocol):
    core_timer: _TimingCore | _RealCoreRunTimer

    def create_fragment(self, fragment_cls: type[ExpFragment]) -> ExpFragment: ...

    def close(self) -> None: ...


class _BenchmarkEnvironment:
    """Tiny ARTIQ environment for running host-only fragments outside the dashboard."""

    def __init__(self):
        self.dataset_db = _MockDatasetDB()
        self.device_db = _MockDeviceDB()
        self.ccb = unittest.mock.Mock()
        self.core_timer = _TimingCore()
        self.scheduler = _MockScheduler()
        self.device_mgr = DeviceManager(
            self.device_db,
            virtual_devices={
                "ccb": self.ccb,
                "core": self.core_timer,
                "scheduler": self.scheduler,
            },
        )

    def create_fragment(self, fragment_cls: type[ExpFragment]) -> ExpFragment:
        fragment = fragment_cls(
            (
                self.device_mgr,
                DatasetManager(self.dataset_db),
                ProcessArgumentManager({}),
                None,
            ),
            [],
        )
        fragment.init_params()
        return fragment

    def close(self) -> None:
        self.device_mgr.close_devices()


class _RealCoreEnvironment:
    """ARTIQ environment that uses a real ``core`` from a device DB.

    This intentionally only overrides ``ccb`` and ``scheduler``. The real core object
    is created from the device DB and its ``run`` method is patched in place so
    fragments still see the same core device object that ARTIQ's compiler expects.
    """

    def __init__(self, device_db_path: str):
        self.dataset_db = _MockDatasetDB()
        self.device_db = DeviceDB(device_db_path)
        self.ccb = unittest.mock.Mock()
        self.scheduler = _MockScheduler()
        self.device_mgr = DeviceManager(
            self.device_db,
            virtual_devices={
                "ccb": self.ccb,
                "scheduler": self.scheduler,
            },
        )
        real_core = self.device_mgr.get("core")
        self.core_timer = _RealCoreRunTimer(real_core)

    def create_fragment(self, fragment_cls: type[ExpFragment]) -> ExpFragment:
        fragment = fragment_cls(
            (
                self.device_mgr,
                DatasetManager(self.dataset_db),
                ProcessArgumentManager({}),
                None,
            ),
            [],
        )
        fragment.init_params()
        return fragment

    def close(self) -> None:
        self.core_timer.close()
        self.device_mgr.close_devices()


class NoResultFragment(ExpFragment):
    """Scanned parameter with no result-channel write."""

    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.last_x = 0.0

    def run_once(self):
        self.last_x = self.x.get()


class ScalarResultFragment(ExpFragment):
    """Scanned parameter plus one scalar result-channel write."""

    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        self.y.push(2.0 * self.x.get() + 1.0)


class SmallArrayResultFragment(ExpFragment):
    """Scanned parameter plus one small integer array result-channel write."""

    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result(
            "counts",
            ArrayChannel,
            element_type="int",
            shape=(8, 8),
            dim_names=("row", "col"),
        )
        self._payload = np.arange(64, dtype=np.uint16).reshape(8, 8)

    def run_once(self):
        self.counts.push(self._payload)


class KernelScalarResultFragment(ExpFragment):
    """Kernel-shaped leaf plus one scalar result-channel write.

    In this local benchmark, ``self.core`` is a timing emulator. On real ARTIQ hardware,
    the same host boundary is where compilation/transfer/execution would happen.
    """

    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    @kernel
    def run_once(self):
        self.y.push(2.0 * self.x.get() + 1.0)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    run: Callable[[], "CaseResult"]


@dataclass(frozen=True)
class CaseResult:
    name: str
    points: int
    build_s: float
    configure_s: float
    first_execute_s: float
    first_kernel_s: float
    warm_execute_s: tuple[float, ...]
    warm_kernel_s: tuple[float, ...]
    known_kernel_wait_s: float = 0.0

    @property
    def warm_mean_s(self) -> float:
        return float(np.mean(self.warm_execute_s))

    @property
    def warm_median_s(self) -> float:
        return float(np.median(self.warm_execute_s))

    @property
    def warm_std_s(self) -> float:
        if len(self.warm_execute_s) < 2:
            return 0.0
        return float(np.std(self.warm_execute_s, ddof=1))

    @property
    def warm_us_per_point(self) -> float:
        return 1e6 * self.warm_median_s / max(1, self.points)

    @property
    def warm_kernel_median_s(self) -> float:
        return float(np.median(self.warm_kernel_s))

    @property
    def warm_kernel_us_per_point(self) -> float:
        return 1e6 * self.warm_kernel_median_s / max(1, self.points)

    @property
    def warm_host_outside_kernel_s(self) -> float:
        return max(0.0, self.warm_median_s - self.warm_kernel_median_s)

    @property
    def warm_host_outside_kernel_us_per_point(self) -> float:
        return 1e6 * self.warm_host_outside_kernel_s / max(1, self.points)

    @property
    def warm_kernel_minus_known_wait_s(self) -> float:
        return self.warm_kernel_median_s - self.known_kernel_wait_s


def _time_call(callable_: Callable[[], object]) -> float:
    start = time.perf_counter()
    callable_()
    return time.perf_counter() - start


def _make_scan_result(
    fragment_cls: type[ExpFragment],
    *,
    name: str,
    points: int,
    repeats: int,
    batch_size: int,
    environment_factory: Callable[[], _BenchmarkEnvironmentProtocol] = _BenchmarkEnvironment,
    reuse_scan_for_warm_runs: bool = False,
) -> CaseResult:
    def prepare_scan() -> tuple[
        float,
        float,
        PreparedScan,
        _TimingCore | _RealCoreRunTimer,
        _BenchmarkEnvironmentProtocol,
    ]:
        env = environment_factory()

        build_start = time.perf_counter()
        fragment = env.create_fragment(fragment_cls)
        build_s = time.perf_counter() - build_start

        values = np.linspace(0.0, 1.0, points).tolist()
        request = ScanRequest.cartesian(
            [(fragment.x, values)],
            execution_policy=ExecutionPolicy(max_points_per_batch=batch_size),
            metadata={"benchmark": name},
        )

        configure_start = time.perf_counter()
        scan = PreparedScan(fragment, fragment, request)
        configure_s = time.perf_counter() - configure_start
        return build_s, configure_s, scan, env.core_timer, env

    def execute_with_kernel_timing(
        scan: PreparedScan, core_timer: _TimingCore | _RealCoreRunTimer
    ) -> tuple[float, float]:
        core_timer.reset_timing()
        execute_s = _time_call(scan.execute)
        return execute_s, core_timer.elapsed_s

    build_s, configure_s, scan, core_timer, env = prepare_scan()
    if reuse_scan_for_warm_runs:
        try:
            first_execute_s, first_kernel_s = execute_with_kernel_timing(
                scan, core_timer
            )
            warm_results = tuple(
                execute_with_kernel_timing(scan, core_timer) for _ in range(repeats)
            )
        finally:
            env.close()
    else:
        try:
            first_execute_s, first_kernel_s = execute_with_kernel_timing(
                scan, core_timer
            )
        finally:
            env.close()
        warm_results = []
        for _ in range(repeats):
            _, _, warm_scan, warm_core_timer, warm_env = prepare_scan()
            try:
                warm_results.append(
                    execute_with_kernel_timing(warm_scan, warm_core_timer)
                )
            finally:
                warm_env.close()
    warm_execute_s = tuple(result[0] for result in warm_results)
    warm_kernel_s = tuple(result[1] for result in warm_results)
    return CaseResult(
        name=name,
        points=points,
        build_s=build_s,
        configure_s=configure_s,
        first_execute_s=first_execute_s,
        first_kernel_s=first_kernel_s,
        warm_execute_s=warm_execute_s,
        warm_kernel_s=warm_kernel_s,
    )


def _python_loop_baseline(*, points: int, repeats: int) -> CaseResult:
    values = np.linspace(0.0, 1.0, points)

    def run_loop():
        total = 0.0
        for value in values:
            total += 2.0 * float(value) + 1.0
        return total

    first_execute_s = _time_call(run_loop)
    warm_execute_s = tuple(_time_call(run_loop) for _ in range(repeats))
    return CaseResult(
        name="python_loop_baseline",
        points=points,
        build_s=0.0,
        configure_s=0.0,
        first_execute_s=first_execute_s,
        first_kernel_s=0.0,
        warm_execute_s=warm_execute_s,
        warm_kernel_s=tuple(0.0 for _ in warm_execute_s),
    )


def _make_nested_ttl_result(
    *,
    device_db_path: str,
    repeats: int,
) -> CaseResult:
    from examples.host_runtime_prepared_kernel_nested_ttl import (
        PreparedKernelNestedTtlFragment,
    )

    def environment_factory() -> _RealCoreEnvironment:
        return _RealCoreEnvironment(device_db_path)

    def prepare_scan() -> tuple[
        float,
        float,
        PreparedScan,
        _TimingCore | _RealCoreRunTimer,
        _BenchmarkEnvironmentProtocol,
    ]:
        env = environment_factory()

        build_start = time.perf_counter()
        fragment = env.create_fragment(PreparedKernelNestedTtlFragment)
        build_s = time.perf_counter() - build_start

        request = ScanRequest.explicit(
            [fragment.outer],
            [[1.0]],
            metadata={"benchmark": "real_core_prepared_nested_ttl"},
        )

        configure_start = time.perf_counter()
        scan = PreparedScan(fragment, fragment, request)
        configure_s = time.perf_counter() - configure_start
        return build_s, configure_s, scan, env.core_timer, env

    def execute_with_kernel_timing(
        scan: PreparedScan, core_timer: _TimingCore | _RealCoreRunTimer
    ) -> tuple[float, float]:
        core_timer.reset_timing()
        execute_s = _time_call(scan.execute)
        return execute_s, core_timer.elapsed_s

    build_s, configure_s, scan, core_timer, env = prepare_scan()
    try:
        first_execute_s, first_kernel_s = execute_with_kernel_timing(scan, core_timer)
        warm_results = tuple(
            execute_with_kernel_timing(scan, core_timer) for _ in range(repeats)
        )
    finally:
        env.close()

    # PreparedKernelNestedTtlFragment scans 3 p-points and, for each p, 5 x-points.
    inner_leaf_points = 15
    warm_execute_s = tuple(result[0] for result in warm_results)
    warm_kernel_s = tuple(result[1] for result in warm_results)
    return CaseResult(
        name="real_core_prepared_nested_ttl",
        points=inner_leaf_points,
        build_s=build_s,
        configure_s=configure_s,
        first_execute_s=first_execute_s,
        first_kernel_s=first_kernel_s,
        warm_execute_s=warm_execute_s,
        warm_kernel_s=warm_kernel_s,
    )


def _make_nested_wait_result(
    *,
    device_db_path: str,
    repeats: int,
    wait_us: float,
) -> CaseResult:
    from examples import host_runtime_prepared_kernel_nested_wait as wait_example

    def environment_factory() -> _RealCoreEnvironment:
        return _RealCoreEnvironment(device_db_path)

    def prepare_scan() -> tuple[
        float,
        float,
        PreparedScan,
        _TimingCore | _RealCoreRunTimer,
        _BenchmarkEnvironmentProtocol,
    ]:
        env = environment_factory()

        build_start = time.perf_counter()
        fragment = env.create_fragment(wait_example.PreparedKernelNestedWaitFragment)
        build_s = time.perf_counter() - build_start

        request = ScanRequest.explicit(
            [fragment.outer],
            [[1.0]],
            metadata={"benchmark": "real_core_prepared_nested_wait"},
        )

        configure_start = time.perf_counter()
        scan = PreparedScan(fragment, fragment, request)
        configure_s = time.perf_counter() - configure_start
        return build_s, configure_s, scan, env.core_timer, env

    def execute_with_kernel_timing(
        scan: PreparedScan, core_timer: _TimingCore | _RealCoreRunTimer
    ) -> tuple[float, float]:
        core_timer.reset_timing()
        execute_s = _time_call(scan.execute)
        return execute_s, core_timer.elapsed_s

    previous_default_wait_us = wait_example.DEFAULT_WALL_WAIT_US
    wait_example.DEFAULT_WALL_WAIT_US = wait_us
    try:
        build_s, configure_s, scan, core_timer, env = prepare_scan()
        try:
            first_execute_s, first_kernel_s = execute_with_kernel_timing(
                scan, core_timer
            )
            warm_results = tuple(
                execute_with_kernel_timing(scan, core_timer) for _ in range(repeats)
            )
        finally:
            env.close()
    finally:
        wait_example.DEFAULT_WALL_WAIT_US = previous_default_wait_us

    # PreparedKernelNestedWaitFragment scans 3 p-points and, for each p, 5 x-points.
    inner_leaf_points = 15
    known_kernel_wait_s = inner_leaf_points * wait_us * 1e-6
    warm_execute_s = tuple(result[0] for result in warm_results)
    warm_kernel_s = tuple(result[1] for result in warm_results)
    return CaseResult(
        name="real_core_prepared_nested_wait",
        points=inner_leaf_points,
        build_s=build_s,
        configure_s=configure_s,
        first_execute_s=first_execute_s,
        first_kernel_s=first_kernel_s,
        warm_execute_s=warm_execute_s,
        warm_kernel_s=warm_kernel_s,
        known_kernel_wait_s=known_kernel_wait_s,
    )


def _default_device_db_path() -> str | None:
    artiq_root = os.getenv("ARTIQ_ROOT")
    if not artiq_root:
        return None
    return os.path.join(artiq_root, "device_db.py")


def run_benchmarks(args: argparse.Namespace) -> list[CaseResult]:
    results = [
        _python_loop_baseline(points=args.points, repeats=args.repeats),
        _make_scan_result(
            NoResultFragment,
            name="prepared_no_result",
            points=args.points,
            repeats=args.repeats,
            batch_size=args.batch_size,
        ),
        _make_scan_result(
            ScalarResultFragment,
            name="prepared_scalar_result",
            points=args.points,
            repeats=args.repeats,
            batch_size=args.batch_size,
        ),
        _make_scan_result(
            SmallArrayResultFragment,
            name="prepared_small_array_result",
            points=args.points,
            repeats=args.repeats,
            batch_size=args.batch_size,
        ),
        _make_scan_result(
            KernelScalarResultFragment,
            name="prepared_kernel_scalar_result",
            points=args.points,
            repeats=args.repeats,
            batch_size=args.batch_size,
        ),
    ]
    if args.real_core:
        device_db_path = args.device_db or _default_device_db_path()
        if device_db_path is None:
            raise ValueError("--real-core requires --device-db or ARTIQ_ROOT")

        def real_core_environment() -> _RealCoreEnvironment:
            return _RealCoreEnvironment(device_db_path)

        real_core_points = args.real_core_points or args.points
        results.append(
            _make_scan_result(
                KernelScalarResultFragment,
                name="real_core_prepared_kernel_scalar",
                points=real_core_points,
                repeats=args.repeats,
                batch_size=args.batch_size,
                environment_factory=real_core_environment,
                reuse_scan_for_warm_runs=True,
            )
        )
    if args.nested_ttl:
        device_db_path = args.device_db or _default_device_db_path()
        if device_db_path is None:
            raise ValueError("--nested-ttl requires --device-db or ARTIQ_ROOT")
        results.append(
            _make_nested_ttl_result(
                device_db_path=device_db_path,
                repeats=args.repeats,
            )
        )
    if args.nested_wait:
        device_db_path = args.device_db or _default_device_db_path()
        if device_db_path is None:
            raise ValueError("--nested-wait requires --device-db or ARTIQ_ROOT")
        results.append(
            _make_nested_wait_result(
                device_db_path=device_db_path,
                repeats=args.repeats,
                wait_us=args.nested_wait_us,
            )
        )
    return results


def print_results(results: list[CaseResult]) -> None:
    baseline = results[0].warm_us_per_point if results else 0.0

    print()
    print("Prepared-runtime host/kernel-boundary overhead benchmark")
    print("========================================================")
    print()
    print(
        "Default rows use a local mock ARTIQ environment. Their kernel row times a "
        "Python fake core.run(...) boundary, not real core-device execution."
    )
    if any(result.name.startswith("real_core_") for result in results):
        print(
            "Rows prefixed real_core_ use the ARTIQ DeviceDB; with a hardware core "
            "this times the host-observed real core.run(...) boundary."
        )
    print()
    print(
        f"{'case':31} {'points':>8} {'build ms':>10} {'config ms':>10} "
        f"{'first ms':>10} {'first kern':>10} {'warm med':>10} {'warm sd':>9} "
        f"{'us/pt':>10} {'kernel us/pt':>12} {'outside us/pt':>13} {'net us/pt':>11}"
    )
    print("-" * 158)
    for result in results:
        net_us = result.warm_us_per_point - baseline
        print(
            f"{result.name:31} "
            f"{result.points:8d} "
            f"{1e3 * result.build_s:10.3f} "
            f"{1e3 * result.configure_s:10.3f} "
            f"{1e3 * result.first_execute_s:10.3f} "
            f"{1e3 * result.first_kernel_s:10.3f} "
            f"{1e3 * result.warm_median_s:10.3f} "
            f"{1e3 * result.warm_std_s:9.3f} "
            f"{result.warm_us_per_point:10.3f} "
            f"{result.warm_kernel_us_per_point:12.3f} "
            f"{result.warm_host_outside_kernel_us_per_point:13.3f} "
            f"{net_us:11.3f}"
        )
    print()
    print("warm med is the median of fresh prepared-scan executions.")
    print("real_core_ warm med reuses one prepared scan to separate first vs warm kernel calls.")
    print(
        "kernel columns are host-observed time spent inside core.run(...), including "
        "host RPC service time while the call is blocking."
    )
    print("outside us/pt is only host time outside core.run(...).")
    print("build/config are outside the warm-run timer.")
    print("net us/pt subtracts the plain Python loop baseline.")
    known_wait_results = [result for result in results if result.known_kernel_wait_s > 0.0]
    if known_wait_results:
        print()
        print("Known core-side waits:")
        for result in known_wait_results:
            print(
                f"{result.name}: wait={1e3 * result.known_kernel_wait_s:.3f} ms, "
                f"warm kernel minus wait={1e3 * result.warm_kernel_minus_known_wait_s:.3f} ms"
            )


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark simple host-only prepared-runtime scan overhead"
    )
    parser.add_argument("--points", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Prepared-runtime max_points_per_batch for each benchmark scan.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional cProfile output path for host-side call attribution.",
    )
    parser.add_argument(
        "--real-core",
        action="store_true",
        help=(
            "Also run a kernel benchmark row using a real ARTIQ core from the "
            "device DB. This connects to hardware and compiles/runs a kernel."
        ),
    )
    parser.add_argument(
        "--device-db",
        type=str,
        default=None,
        help="Device DB path for --real-core. Defaults to $ARTIQ_ROOT/device_db.py.",
    )
    parser.add_argument(
        "--real-core-points",
        type=int,
        default=None,
        help="Point count for the --real-core row. Defaults to --points.",
    )
    parser.add_argument(
        "--nested-ttl",
        action="store_true",
        help=(
            "Also run the hardware-oriented prepared nested TTL example. This uses "
            "ttl0 and is therefore opt-in."
        ),
    )
    parser.add_argument(
        "--nested-wait",
        action="store_true",
        help=(
            "Also run a hardware-oriented prepared nested example whose leaf waits "
            "on the core RTIO counter before yielding."
        ),
    )
    parser.add_argument(
        "--nested-wait-us",
        type=float,
        default=5000.0,
        help="Core-side wall wait per inner leaf point for --nested-wait.",
    )
    return parser


def main() -> None:
    args = get_argparser().parse_args()

    if args.profile is None:
        results = run_benchmarks(args)
    else:
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            results = run_benchmarks(args)
        finally:
            profiler.disable()
            profiler.dump_stats(args.profile)
        stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
        print()
        print(f"Wrote cProfile data to {args.profile}")
        print("Top cumulative host functions:")
        stats.print_stats(20)

    print_results(results)


if __name__ == "__main__":
    main()

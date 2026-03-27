"""Experiment and dashboard adapters over prepared scans."""

from __future__ import annotations

from contextlib import suppress
from collections.abc import Mapping
from typing import Any

from artiq.language.core import TerminationRequested
from artiq.language import EnvExperiment, HasEnvironment, PYONValue

from .prepared import PreparedScan
from ..define.fragment import ExpFragment
from ..define.parameters import ParamStore
from ..define.result_channels import ResultChannel
from ..scan.request import ScanRequest
from ..schema.scan_site import make_scan_site_prefix
from ..submission.host_scan_schema import (
    HostScanGridModeSpec,
    HostScanSchemaError,
    HostScanSpec,
    compile_host_scan_schema,
    compile_host_scan_spec,
)
from ..utils import PARAMS_ARG_KEY

__all__ = [
    "HostArgumentInterface",
    "PreparedScanExperiment",
    "PreparedDashboardScanExperiment",
    "make_fragment_prepared_scan_exp",
    "make_fragment_prepared_dashboard_scan_exp",
]


class HostArgumentInterface(HasEnvironment):
    """Expose prepared-runtime submissions through the existing ndscan dashboard channel."""

    def build(
        self,
        fragment: ExpFragment,
        default_request_spec: HostScanSpec | Mapping[str, Any] | None = None,
    ) -> None:
        instances = dict[str, list[str]]()
        self._schemata = dict[str, dict]()
        self._sample_instances = dict[str, Any]()
        always_shown_params = []
        result_channels = dict[str, ResultChannel]()

        fragment._collect_params(instances, self._schemata, self._sample_instances)
        fragment._collect_result_channels(result_channels)
        for handle in fragment.get_always_shown_params():
            path = handle.owner._stringize_path()
            try:
                param = handle.owner._free_params[handle.name]
                always_shown_params += [(param.fqn, path)]
            except KeyError:
                pass

        desc: dict[str, Any] = {
            "instances": instances,
            "schemata": self._schemata,
            "always_shown": always_shown_params,
            "channels": {
                path: channel.describe()
                for path, channel in result_channels.items()
                if channel.save_by_default
            },
            "overrides": {},
        }
        default_transport = _host_request_transport_dict(default_request_spec)
        if default_transport is not None:
            desc["host_scan"] = default_transport

        self._params = self.get_argument(PARAMS_ARG_KEY, PYONValue(default=desc))

    def make_override_stores(self) -> dict[str, list[tuple[str, ParamStore]]]:
        stores = {}
        for fqn, specs in self._params.get("overrides", {}).items():
            try:
                store_type = self._sample_instances[fqn].StoreType
            except KeyError:
                raise KeyError(
                    "Experiment does not have parameters matching override for FQN "
                    f"{fqn!r}"
                )
            stores[fqn] = [
                (
                    spec["path"],
                    store_type(
                        (fqn, spec["path"]),
                        store_type.value_from_pyon(spec["value"]),
                    ),
                )
                for spec in specs
            ]
        return stores

    def resolve_request(
        self,
        fragment: ExpFragment,
        default_request_spec: HostScanSpec | Mapping[str, Any] | None,
    ) -> tuple[ScanRequest, dict[str, list[tuple[str, ParamStore]]]]:
        if "host_scan" in self._params:
            request, compiled_overrides = compile_host_scan_schema(
                fragment, self._params["host_scan"]
            )
        elif default_request_spec is None:
            raise ValueError(
                "No host_scan submission was provided for this dashboard-driven "
                "prepared dashboard scan experiment"
            )
        else:
            request, compiled_overrides = _resolve_dashboard_request_spec(
                fragment, default_request_spec
            )
        return request, _merge_override_store_maps(
            compiled_overrides,
            self.make_override_stores(),
        )


class PreparedScanExperiment(EnvExperiment):
    """Thin ``EnvExperiment`` adapter over a root ``PreparedScan``."""

    def build(
        self,
        fragment_init,
        request_factory,
        *,
        max_rtio_underflow_retries: int = 3,
        max_transitory_error_retries: int = 10,
    ) -> None:
        self.setattr_device("ccb")
        self.fragment = fragment_init()
        self._request_spec = (
            request_factory(self.fragment) if callable(request_factory) else request_factory
        )
        self._max_rtio_underflow_retries = max_rtio_underflow_retries
        self._max_transitory_error_retries = max_transitory_error_retries
        self._session = None
        self._plot_prefix = None

    def prepare(self) -> None:
        request, overrides = _resolve_code_request_spec(self._request_spec)
        self._plot_prefix = make_scan_site_prefix(self, request.site)
        self._session = PreparedScan(
            self,
            self.fragment,
            request=request,
            overrides=overrides,
            max_rtio_underflow_retries=self._max_rtio_underflow_retries,
            max_transitory_error_retries=self._max_transitory_error_retries,
        )

    def run(self) -> None:
        _create_runtime_applet(
            self.ccb,
            prefix=self._plot_prefix,
            title=f"{self.fragment.__class__.__name__} ({self.fragment.fqn})",
        )
        with suppress(TerminationRequested):
            self._session.execute()


class PreparedDashboardScanExperiment(EnvExperiment):
    """Dashboard-driven ``PreparedScan`` adapter."""

    argument_ui = "ndscan"

    def build(
        self,
        fragment_init,
        *,
        default_request_spec: HostScanSpec | Mapping[str, Any] | None = None,
        max_rtio_underflow_retries: int = 3,
        max_transitory_error_retries: int = 10,
    ) -> None:
        self.setattr_device("ccb")
        self.fragment = fragment_init()
        if default_request_spec is None:
            default_request_spec = HostScanSpec(mode=HostScanGridModeSpec())
        self._default_request_spec = default_request_spec
        self._max_rtio_underflow_retries = max_rtio_underflow_retries
        self._max_transitory_error_retries = max_transitory_error_retries
        self._session = None
        self._plot_prefix = None
        self.args = HostArgumentInterface(self, self.fragment, self._default_request_spec)

    def prepare(self) -> None:
        request, overrides = self.args.resolve_request(
            self.fragment,
            self._default_request_spec,
        )
        self._plot_prefix = make_scan_site_prefix(self, request.site)
        self._session = PreparedScan(
            self,
            self.fragment,
            request=request,
            overrides=overrides,
            max_rtio_underflow_retries=self._max_rtio_underflow_retries,
            max_transitory_error_retries=self._max_transitory_error_retries,
        )

    def run(self) -> None:
        _create_runtime_applet(
            self.ccb,
            prefix=self._plot_prefix,
            title=f"{self.fragment.__class__.__name__} ({self.fragment.fqn})",
        )
        with suppress(TerminationRequested):
            self._session.execute()


def make_fragment_prepared_scan_exp(
    fragment_class: type[ExpFragment],
    request_factory,
    *args,
    max_rtio_underflow_retries: int = 3,
    max_transitory_error_retries: int = 10,
) -> type[PreparedScanExperiment]:
    """Create a runnable ``EnvExperiment`` convenience wrapper over ``PreparedScan``."""

    class FragmentPreparedScanShim(PreparedScanExperiment):
        def build(self):
            super().build(
                lambda: fragment_class(self, [], *args),
                request_factory,
                max_rtio_underflow_retries=max_rtio_underflow_retries,
                max_transitory_error_retries=max_transitory_error_retries,
            )

    FragmentPreparedScanShim.__name__ = fragment_class.__name__
    FragmentPreparedScanShim.__qualname__ = fragment_class.__name__
    FragmentPreparedScanShim.__module__ = fragment_class.__module__
    FragmentPreparedScanShim.__doc__ = fragment_class.__doc__
    return FragmentPreparedScanShim


def make_fragment_prepared_dashboard_scan_exp(
    fragment_class: type[ExpFragment],
    default_request_spec: HostScanSpec | Mapping[str, Any] | None = None,
    *args,
    max_rtio_underflow_retries: int = 3,
    max_transitory_error_retries: int = 10,
) -> type[PreparedDashboardScanExperiment]:
    """Create a dashboard-driven ``PreparedScan`` convenience wrapper."""

    class FragmentPreparedDashboardScanShim(PreparedDashboardScanExperiment):
        def build(self):
            super().build(
                lambda: fragment_class(self, [], *args),
                default_request_spec=default_request_spec,
                max_rtio_underflow_retries=max_rtio_underflow_retries,
                max_transitory_error_retries=max_transitory_error_retries,
            )

    FragmentPreparedDashboardScanShim.__name__ = fragment_class.__name__
    FragmentPreparedDashboardScanShim.__qualname__ = fragment_class.__name__
    FragmentPreparedDashboardScanShim.__module__ = fragment_class.__module__
    FragmentPreparedDashboardScanShim.__doc__ = fragment_class.__doc__
    return FragmentPreparedDashboardScanShim


def _resolve_code_request_spec(
    request_spec: Any,
) -> tuple[ScanRequest, dict[str, list[tuple[str, ParamStore]]]]:
    if isinstance(request_spec, ScanRequest):
        return request_spec, {}

    if (
        isinstance(request_spec, tuple)
        and len(request_spec) == 2
        and isinstance(request_spec[0], ScanRequest)
        and isinstance(request_spec[1], Mapping)
    ):
        return request_spec[0], dict(request_spec[1])

    raise TypeError(
        "Code-defined prepared scans require a ScanRequest or a pair of "
        "(ScanRequest, overrides)"
    )


def _resolve_dashboard_request_spec(
    fragment: ExpFragment,
    request_spec: HostScanSpec | Mapping[str, Any],
) -> tuple[ScanRequest, dict[str, list[tuple[str, ParamStore]]]]:
    if isinstance(request_spec, HostScanSpec):
        return compile_host_scan_spec(fragment, request_spec)
    if isinstance(request_spec, Mapping):
        return compile_host_scan_schema(fragment, request_spec)
    raise TypeError("Dashboard prepared scans require a HostScanSpec or dict schema")


def _host_request_transport_dict(
    request_spec: HostScanSpec | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(request_spec, HostScanSpec):
        return request_spec.to_dict()
    if isinstance(request_spec, Mapping):
        return dict(request_spec)
    return None


def _merge_override_store_maps(
    *sources: Mapping[str, list[tuple[str, ParamStore]]],
) -> dict[str, list[tuple[str, ParamStore]]]:
    merged: dict[str, dict[str, ParamStore]] = {}
    for source in sources:
        for fqn, pairs in source.items():
            target = merged.setdefault(fqn, {})
            for path, store in pairs:
                target[path] = store
    return {
        fqn: list(path_map.items())
        for fqn, path_map in merged.items()
    }


def _create_runtime_applet(ccb, *, prefix: str | None, title: str, group: str = "ndscan"):
    if not prefix:
        return
    cmd = [
        "${python}",
        "-m ndscan.runtime_applet",
        "--server=${server}",
        "--port-notify=${port_notify}",
        "--port-control=${port_control}",
        f"--prefix={prefix}",
    ]
    ccb.issue("create_applet", title, " ".join(cmd), group=group)

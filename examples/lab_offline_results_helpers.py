"""Example reusable lab-side helpers for prepared-runtime offline analysis.

This file is deliberately placed in ``examples`` so it can be copied into a lab helper
repository. It should stay generic across many different experiment types:

- open a prepared-runtime HDF5 file through the ndscan site-tree reader,
- access sites by path,
- pull semantic selector arrays,
- build simple 1D/errorbar payloads,
- split array-valued series along an array axis,
- resolve fixed parameter values by FQN.

Experiment-specific blob interpretation, ROI/image semantics, and recomputation of a
particular saved statistic should live in individual analysis scripts or more specific
lab-helper modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ndscan.results.scan_site_reader import (
    HostRuntimeSite,
    HostRuntimeSnapshot,
    read_host_runtime_snapshot,
)
from ndscan.results.series import (
    build_1d_errorbar_payload,
    series_dict,
    series_for_selector,
    series_slices_along_axis,
)

__all__ = [
    "LabNdscanRun",
    "LabNdscanSite",
]


@dataclass(frozen=True)
class LabNdscanRun:
    """Small lab-facing wrapper around one prepared-runtime snapshot."""

    snapshot: HostRuntimeSnapshot

    @classmethod
    def open(cls, path: str | Path) -> "LabNdscanRun":
        return cls(read_host_runtime_snapshot(path))

    def site(self, path: tuple[str, ...] = ()) -> "LabNdscanSite":
        return LabNdscanSite(self.snapshot.get_site(path))

    def child_sites(self, path: tuple[str, ...] = ()) -> list["LabNdscanSite"]:
        return [
            LabNdscanSite(site)
            for site in self.snapshot.child_sites(path)
        ]


@dataclass(frozen=True)
class LabNdscanSite:
    """Convenience wrapper for semantic selector-based access to one site."""

    site: HostRuntimeSite

    @property
    def path(self) -> tuple[str, ...]:
        return self.site.path

    def series(self, selector: str) -> np.ndarray:
        return series_for_selector(self.site, selector)

    def arrays(self, selectors: Mapping[str, str] | list[str] | tuple[str, ...]):
        return series_dict(self.site, selectors)

    def errorbar(
        self,
        *,
        x: str | None = None,
        y: str,
        yerr: str | None = None,
        yerr_lower: str | None = None,
        yerr_upper: str | None = None,
    ) -> dict[str, Any]:
        return build_1d_errorbar_payload(
            self.site,
            x=x,
            y=y,
            yerr=yerr,
            yerr_lower=yerr_lower,
            yerr_upper=yerr_upper,
        )

    def split_array_series(
        self,
        path: str,
        *,
        axis: int = 0,
        indices=None,
        fixed_indices: Mapping[int, int] | None = None,
    ) -> dict[int, np.ndarray]:
        return series_slices_along_axis(
            self.site,
            path,
            axis=axis,
            indices=indices,
            fixed_indices=fixed_indices,
        )

    def fixed_parameter_value_by_fqn(self, fqn: str) -> Any:
        matches = [
            entry["value"]
            for entry in self.site.fixed_parameters.values()
            if isinstance(entry, Mapping)
            and isinstance(entry.get("param"), Mapping)
            and entry["param"].get("fqn") == fqn
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Could not resolve unique fixed parameter {fqn!r} on site "
                f"{'/'.join(self.path) or '<root>'}"
            )
        return matches[0]

    def metadata_blob(self, name: str) -> Any:
        return self.site.require_metadata_blob(name)

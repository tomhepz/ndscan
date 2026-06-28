"""Example reusable lab-side helpers for prepared-runtime offline analysis.

This file is deliberately placed in ``examples`` so it can be copied into a lab helper
repository. It should stay generic across many different experiment types:

- open a prepared-runtime HDF5 file through the ndscan site-tree reader,
- access sites by path,
- pull saved series arrays by semantic path,
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
    ScanSiteData,
    ScanSiteSnapshot,
    read_scan_site_snapshot,
)
from ndscan.results.series import series_slices_along_axis

__all__ = [
    "LabNdscanRun",
    "LabNdscanSite",
]


@dataclass(frozen=True)
class LabNdscanRun:
    """Small lab-facing wrapper around one prepared-runtime snapshot."""

    snapshot: ScanSiteSnapshot

    @classmethod
    def open(cls, path: str | Path) -> "LabNdscanRun":
        return cls(read_scan_site_snapshot(path))

    def site(self, path: tuple[str, ...] = ()) -> "LabNdscanSite":
        return LabNdscanSite(self.snapshot.get_site(path))

    def child_sites(self, path: tuple[str, ...] = ()) -> list["LabNdscanSite"]:
        return [
            LabNdscanSite(site)
            for site in self.snapshot.child_sites(path)
        ]


@dataclass(frozen=True)
class LabNdscanSite:
    """Small lab-facing wrapper around one offline site."""

    site: ScanSiteData

    @property
    def path(self) -> tuple[str, ...]:
        return self.site.path

    def series(self, path: str, *, dtype: Any | None = None) -> np.ndarray:
        return self.site.series(path, dtype=dtype)

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

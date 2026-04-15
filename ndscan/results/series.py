"""Pure-Python helpers for extracting plottable series from offline site data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import re
from typing import Any

import numpy as np

from .scan_site_reader import HostRuntimeSite

__all__ = [
    "average_series",
    "build_1d_errorbar_payload",
    "build_1d_series_payload",
    "format_series_selector",
    "parse_series_selector",
    "series_dict",
    "series_for_selector",
    "series_slices_along_axis",
    "slice_series",
]


_INDEX_PATTERN = re.compile(r"^(?P<path>.+?)\[(?P<indices>[0-9,\s]+)\]$")


def format_series_selector(path: str, indices: Sequence[int] | None = None) -> str:
    """Return a semantic selector string for ``path`` plus optional fixed indices."""

    stripped_path = path.strip()
    if indices is None:
        return stripped_path
    return f"{stripped_path}[{','.join(str(int(index)) for index in indices)}]"


def parse_series_selector(selector: str) -> tuple[str, tuple[int, ...] | None]:
    """Split a semantic selector into path plus optional fixed indices.

    Examples:

    - ``"shot/probe_frequency"`` -> ``("shot/probe_frequency", None)``
    - ``"occupancy_probability_image2[0,0]"`` ->
      ``("occupancy_probability_image2", (0, 0))``
    """

    match = _INDEX_PATTERN.fullmatch(selector.strip())
    if match is None:
        return selector.strip(), None
    indices = tuple(int(part.strip()) for part in match.group("indices").split(","))
    return match.group("path").strip(), indices


def slice_series(values: np.ndarray, indices: tuple[int, ...] | None) -> np.ndarray:
    """Return one fixed slice of a saved series array."""

    if indices is None:
        return values
    return values[(slice(None),) + indices]


def series_for_selector(site: HostRuntimeSite, selector: str) -> np.ndarray:
    """Return one saved series addressed by a public selector string."""

    path, indices = parse_series_selector(selector)
    values = np.asarray(site.series(path))
    return slice_series(values, indices)


def series_dict(
    site: HostRuntimeSite,
    selectors: Mapping[str, str] | Iterable[str],
) -> dict[str, np.ndarray]:
    """Return several series addressed by public selectors.

    ``selectors`` may be either:

    - a mapping from output label to selector, or
    - an iterable of selector strings, in which case the selector is also the label.
    """

    if isinstance(selectors, Mapping):
        items = selectors.items()
    else:
        items = ((selector, selector) for selector in selectors)
    return {
        str(label): series_for_selector(site, str(selector))
        for label, selector in items
    }


def series_slices_along_axis(
    site: HostRuntimeSite,
    path: str,
    *,
    axis: int = 0,
    indices: Iterable[int] | None = None,
    fixed_indices: Mapping[int, int] | None = None,
) -> dict[int, np.ndarray]:
    """Split an array-valued series along one array axis.

    ``axis`` and ``fixed_indices`` refer to the array dimensions after the leading
    point dimension. For example, a saved ``(points, group, roi)`` stream can be split
    by group with ``axis=0`` and a fixed ROI with ``fixed_indices={1: 0}``.
    """

    values = np.asarray(site.series(path))
    if values.ndim < 2:
        raise ValueError(
            f"Path {path!r} resolved to a scalar series with shape {values.shape}; "
            "expected at least one array dimension"
        )

    array_rank = values.ndim - 1
    if not 0 <= int(axis) < array_rank:
        raise ValueError(
            f"Axis {axis} is out of range for array-valued series {path!r} with "
            f"array rank {array_rank}"
        )

    fixed = {int(key): int(value) for key, value in (fixed_indices or {}).items()}
    if int(axis) in fixed:
        raise ValueError("The split axis must not also be listed in fixed_indices")
    for fixed_axis in fixed:
        if not 0 <= fixed_axis < array_rank:
            raise ValueError(
                f"Fixed axis {fixed_axis} is out of range for array-valued series "
                f"{path!r} with array rank {array_rank}"
            )

    axis = int(axis)
    selected_indices = (
        range(values.shape[axis + 1])
        if indices is None
        else [int(index) for index in indices]
    )

    result: dict[int, np.ndarray] = {}
    for index in selected_indices:
        slicer: list[object] = [slice(None)]
        for array_axis in range(array_rank):
            if array_axis == axis:
                slicer.append(index)
            elif array_axis in fixed:
                slicer.append(fixed[array_axis])
            else:
                slicer.append(slice(None))
        result[index] = values[tuple(slicer)]
    return result


def build_1d_series_payload(
    site: HostRuntimeSite,
    *,
    x: str | None = None,
    y: str,
) -> dict[str, Any]:
    """Return one simple 1D plotting payload using semantic selectors."""

    x_selector = site.choose_default_x_path() if x is None else x
    if x_selector is None:
        raise ValueError(
            f"Site {'/'.join(site.path) or '<root>'} does not have a default x path; "
            f"available paths are {site.available_series_paths()}"
        )

    x_values = series_for_selector(site, x_selector)
    y_values = series_for_selector(site, y)

    if x_values.ndim != 1:
        raise ValueError(f"x selector {x_selector!r} did not resolve to a 1D series")
    if y_values.ndim != 1:
        raise ValueError(f"y selector {y!r} did not resolve to a 1D series")
    if len(x_values) != len(y_values):
        raise ValueError(
            f"x selector {x_selector!r} and y selector {y!r} have different lengths "
            f"({len(x_values)} vs {len(y_values)})"
        )

    order = np.argsort(x_values)
    return {
        "x_selector": x_selector,
        "y_selector": y,
        "x_values": np.asarray(x_values[order], dtype=float),
        "y_values": np.asarray(y_values[order]),
    }


def build_1d_errorbar_payload(
    site: HostRuntimeSite,
    *,
    x: str | None = None,
    y: str,
    yerr: str | None = None,
    yerr_lower: str | None = None,
    yerr_upper: str | None = None,
) -> dict[str, Any]:
    """Return a simple 1D payload with optional symmetric/asymmetric error bars."""

    if yerr is not None and (yerr_lower is not None or yerr_upper is not None):
        raise ValueError("Use either yerr or yerr_lower/yerr_upper, not both")
    if (yerr_lower is None) != (yerr_upper is None):
        raise ValueError("yerr_lower and yerr_upper must be supplied together")

    payload = build_1d_series_payload(site, x=x, y=y)
    order = np.argsort(series_for_selector(site, payload["x_selector"]))
    if yerr is not None:
        error_values = np.asarray(series_for_selector(site, yerr))[order]
        payload["yerr_selector"] = yerr
        payload["yerr_values"] = error_values
    elif yerr_lower is not None and yerr_upper is not None:
        lower_values = np.asarray(series_for_selector(site, yerr_lower))[order]
        upper_values = np.asarray(series_for_selector(site, yerr_upper))[order]
        payload["yerr_lower_selector"] = yerr_lower
        payload["yerr_upper_selector"] = yerr_upper
        payload["yerr_lower_values"] = lower_values
        payload["yerr_upper_values"] = upper_values
    return payload


def average_series(site: HostRuntimeSite, path: str) -> np.ndarray:
    """Return the mean of one saved series over the site's leading point dimension."""

    values = np.asarray(site.series(path))
    if values.size == 0:
        raise ValueError(
            f"Site {'/'.join(site.path) or '<root>'} has no points for series {path!r}"
        )
    return np.asarray(np.mean(values, axis=0), dtype=float)

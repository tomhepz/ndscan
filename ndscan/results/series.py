"""Small pure-Python helpers for working with saved series arrays offline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .scan_site_reader import HostRuntimeSite

__all__ = [
    "series_slices_along_axis",
]


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

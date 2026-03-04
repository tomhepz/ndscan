from typing import Any

import numpy

from ...utils import strip_prefix
from . import ScanModel, SinglePointModel


class SelectPointFromScanModel(SinglePointModel):
    def __init__(self, source: ScanModel):
        super().__init__(source.schema_revision, source.context)
        self._source = source
        self._source_index = None
        self._point = None

        self._source.points_rewritten.connect(
            lambda: self._set_point(self._source_index, silently_fail=True)
        )
        self._source.points_appended.connect(
            lambda: self._set_point(self._source_index, silently_fail=True)
        )

        # TODO: Invalidate point data (reset index?) on channel schema change.
        self._source.channel_schemata_changed.connect(self.channel_schemata_changed)

    def set_source_index(self, idx: int | None) -> None:
        if idx == self._source_index:
            return
        self._set_point(idx, silently_fail=False)

    def get_source_index(self) -> int | None:
        return self._source_index

    def get_channel_schemata(self) -> dict[str, Any]:
        return self._source.get_channel_schemata()

    def get_point(self) -> dict[str, Any] | None:
        return self._point

    def is_completed(self) -> bool | None:
        is_completed = getattr(self._source, "is_completed", None)
        if is_completed is None:
            return None
        return is_completed()

    def _set_point(self, idx: int | None, silently_fail: bool) -> None:
        old_idx = self._source_index
        self._source_index = idx
        if idx is None:
            point = None
        else:
            points = self._source.get_point_data()
            if not points:
                point = None
            else:
                num_values = len(next(iter(points.values())))
                if idx == num_values:
                    # Support selecting the in-progress point just beyond the currently
                    # completed data for live subscan follow mode.
                    point = None
                elif idx > num_values or idx < 0:
                    if silently_fail:
                        point = None
                    else:
                        raise ValueError(
                            "Invalid source index {} for length {}".format(idx, num_values)
                        )
                else:
                    point = {}
                    for key, values in points.items():
                        name = strip_prefix(key, "channel_")
                        if name != key:
                            point[name] = values[idx]
        # The point data can include NumPy arrays, which breaks object comparison (as
        # comparing two arrays gives back a bool array of element-wise results). We thus
        # need to use array_equal() to work around this.
        # Selection changes must always be propagated, even if two points have identical
        # payloads (e.g. no scalar top-level channels besides subscan metadata).
        if idx == old_idx and _all_array_equal(point, self._point):
            return
        self._point = point
        self.point_changed.emit(point)


def _all_array_equal(left, right):
    if left is None or right is None:
        return left is None and right is None
    keys = set(left.keys())
    if keys != set(right.keys()):
        return False
    for k in keys:
        if not numpy.array_equal(left[k], right[k]):
            return False
    return True

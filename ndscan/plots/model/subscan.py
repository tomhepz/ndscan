import hashlib
import json
import re
from typing import Any

import numpy

from ...utils import strip_suffix
from . import FixedDataSource, Model, Root, ScanModel, SinglePointModel


class SubscanRoot(Root):
    def __init__(self, parent: SinglePointModel, schema_key: str):
        super().__init__()
        self._parent = parent
        self._schema_key = schema_key
        self._model = None
        self._schema = None
        self._schema_str = None
        self._schema_channel_path = ""
        self._parent.point_changed.connect(self._update)
        self._parent.context.datasets_changed.connect(self._on_datasets_changed)

        self.name = strip_suffix(self._schema_key, "_spec")
        if self.name == self._schema_key:
            raise ValueError(
                "Unexpected scan schema channel name: {}".format(self._schema_key)
            )

        self._update(parent.get_point())

    def _refresh_schema_channel_path(self) -> None:
        if self._schema_channel_path:
            return
        schemata = self._parent.get_channel_schemata() or {}
        path = schemata.get(self._schema_key, {}).get("path", "")
        if path:
            self._schema_channel_path = path

    def _update(self, data: dict[str, Any]) -> None:
        self._refresh_schema_channel_path()
        if data is None:
            schema = self._schema_from_preview_datasets()
            if schema is None:
                if self._model:
                    self._model.quit()
                    self.model_changed.emit(None)
                self._model = None
                self._schema = None
                self._schema_str = None
                return
            if self._schema == schema:
                return
            self._set_schema(schema, schema_str=None)
            return

        schema_str = data.get(self._schema_key, None)
        if schema_str is None:
            # While a parent point exists, some subscan schemas are only pushed once the
            # parent point is fully complete. Keep probing preview metadata so live
            # subscan views can still bootstrap.
            schema = self._schema_from_preview_datasets()
            if schema is None:
                return
            if self._schema == schema:
                return
            self._set_schema(schema, schema_str=None)
            return
        schema = json.loads(schema_str)
        if self._schema == schema:
            return

        self._set_schema(schema, schema_str=schema_str)

    def _set_schema(self, schema: dict[str, Any], schema_str: str | None) -> None:
        if self._model is not None:
            self._model.quit()
        self._schema = schema
        self._schema_str = schema_str
        self._model = SubscanModel(
            self._schema,
            self._parent,
            self.name + "_",
            self._schema_key,
            self._schema_channel_path,
        )
        self.model_changed.emit(self._model)

    def _on_datasets_changed(self, _values, _mods) -> None:
        self._update(self._parent.get_point())

    def _schema_from_preview_datasets(self) -> dict[str, Any] | None:
        self._refresh_schema_channel_path()
        values = self._parent.context.get_dataset_values()
        if not values:
            return None
        prefix = _find_subscan_dataset_prefix_by_channel_path(
            values, "subscan_preview", self._schema_channel_path
        )
        if prefix is None:
            return None

        axes = values.get(prefix + "axes", None)
        channels = values.get(prefix + "channels", None)
        fragment_fqn = values.get(prefix + "fragment_fqn", None)
        if axes is None or channels is None or fragment_fqn is None:
            return None

        schema = {
            "axes": json.loads(axes),
            "channels": json.loads(channels),
            "fragment_fqn": fragment_fqn,
            "analysis_results": {},
            "online_analyses": {},
            "annotations": [],
        }
        for name in ("analysis_results", "online_analyses", "annotations"):
            value = values.get(prefix + name, None)
            if value is not None:
                schema[name] = json.loads(value)
        return schema

    def get_model(self) -> Model | None:
        return self._model

    def get_title(self) -> str:
        return f"subscan '{self._display_name()}'"

    def _display_name(self) -> str:
        if self.name:
            return self.name
        path = self._schema_channel_path
        if path:
            if "/" in path:
                parent_path, leaf = path.rsplit("/", 1)
                if leaf.endswith("spec") and parent_path:
                    return parent_path.rsplit("/", 1)[-1]
            return path.replace("/", ".")
        return self._schema_key


class SubscanModel(ScanModel):
    """A scan selected out of a single point with a subscan channel.

    Point content changes are forwarded, but the schema is static; changes to the latter
    necessitate a new model instance.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        parent: SinglePointModel,
        result_prefix: str,
        schema_key: str,
        schema_channel_path: str | None = None,
    ):
        super().__init__(schema["axes"], parent.schema_revision, parent.context)

        self._schema = schema
        self._channel_schemata = schema["channels"]
        self._schema_key = schema_key

        self._result_prefix = result_prefix
        self._point_data = {}
        self._parent = parent
        self._parent.point_changed.connect(self._update)
        self._parent.context.datasets_changed.connect(self._on_datasets_changed)

        parent_schemata = self._parent.get_channel_schemata() or {}
        self._schema_channel_path = schema_channel_path or parent_schemata.get(
            self._schema_key, {}
        ).get("path", "")
        self._site_id = None
        self._site_slug = None
        self._preview_dataset_prefix = None
        self._flat_dataset_prefix = None
        self._compute_subscan_site_keys()

        # Do not require analysis results to be present for backwards-compatibility.
        self._analysis_results = {}
        self._analysis_result_mappings = []
        for result_name, path in schema.get("analysis_results", {}).items():
            for (
                channel_name,
                channel_schema,
            ) in parent_schemata.items():
                if channel_schema["path"] == path:
                    source = FixedDataSource(None)
                    self._analysis_results[result_name] = source
                    self._analysis_result_mappings.append((result_name, channel_name))

        self._set_online_analyses(schema.get("online_analyses", {}))
        self._set_annotation_schemata(schema.get("annotations", []))
        self._update(parent.get_point())

    def _refresh_schema_channel_path(self) -> None:
        if self._schema_channel_path:
            return
        parent_schemata = self._parent.get_channel_schemata() or {}
        path = parent_schemata.get(self._schema_key, {}).get("path", "")
        if path:
            self._schema_channel_path = path
            self._compute_subscan_site_keys()

    def quit(self) -> None:
        self._parent.point_changed.disconnect(self._update)
        self._parent.context.datasets_changed.disconnect(self._on_datasets_changed)

    def _update(self, parent_data: dict[str, Any] | None) -> None:
        source_index = self._get_selected_source_index()
        # Selected-point mode: prefer exact embedded point payload from the parent.
        # This avoids mixing a selected parent point with a globally indexed flat
        # subscan segment in nested subscan trees.
        point_data = None
        if source_index is not None and parent_data is not None:
            point_data = self._extract_embedded_point_data(parent_data)
        if point_data is None:
            # Live-preview mode (no selection): read preview stream from datasets.
            # Selected-point mode (if embedded data is unavailable): try flat segment.
            point_data = self._extract_dataset_point_data(source_index)
        if point_data is None and parent_data is not None:
            # No selection or no dataset-backed data yet; fall back to embedded payload.
            # For selected mode this only applies when embedded data was unavailable on
            # the first pass and source index is None.
            point_data = self._extract_embedded_point_data(parent_data)
        if point_data is not None:
            self._set_point_data(point_data)

        if parent_data is None:
            return
        for r, c in self._analysis_result_mappings:
            if c in parent_data:
                self._analysis_results[r].set(parent_data[c])

    def _set_point_data(self, point_data: dict[str, Any]) -> None:
        if not self._point_data:
            self._point_data = point_data
            self.points_rewritten.emit(self._point_data)
            return

        rewritten = False
        appended = False
        for name in self._required_names():
            old = self._point_data.get(name, [])
            new = point_data.get(name, [])

            old_len = _safe_len(old)
            new_len = _safe_len(new)
            if old_len is None or new_len is None:
                rewritten = True
                break

            if new_len < old_len:
                rewritten = True
                break
            if not _prefix_equal(old, new, old_len):
                rewritten = True
                break
            if new_len > old_len:
                appended = True

        self._point_data = point_data
        if rewritten:
            self.points_rewritten.emit(self._point_data)
        elif appended:
            self.points_appended.emit(self._point_data)

    def _on_datasets_changed(self, _values, _mods) -> None:
        self._update(self._parent.get_point())

    def _extract_embedded_point_data(
        self, parent_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        point_data = {}
        for name in self._required_names():
            key = self._result_prefix + name
            if key not in parent_data:
                return None
            point_data[name] = parent_data[key]
        return point_data

    def _extract_dataset_point_data(
        self, source_index: int | None = None
    ) -> dict[str, Any] | None:
        values = self._parent.context.get_dataset_values()
        if not values:
            return None
        self._resolve_dataset_prefixes(values)

        if source_index is not None:
            flat_data = self._extract_flat_segment(values, source_index)
            if flat_data is not None:
                return flat_data
            # Explicit point selection should never silently drift to live preview data.
            return None

        return self._extract_preview_data(values)

    def _get_selected_source_index(self) -> int | None:
        getter = getattr(self._parent, "get_source_index", None)
        if getter is None:
            return None
        return getter()

    def _extract_flat_segment(
        self, values: dict[str, Any], source_index: int
    ) -> dict[str, Any] | None:
        if self._flat_dataset_prefix is None:
            return None

        starts = values.get(self._flat_dataset_prefix + "starts", None)
        if starts is None or source_index < 0 or source_index >= len(starts):
            return None

        # Starts are currently only pushed after a subscan completes. Before the scan is
        # complete, the final segment boundary is not known in the flat stream yet.
        if (
            source_index == len(starts) - 1
            and not values.get(self._flat_dataset_prefix + "completed", False)
        ):
            return None

        start = int(starts[source_index])
        if source_index + 1 < len(starts):
            end = int(starts[source_index + 1])
        else:
            end = self._point_count(values, self._flat_dataset_prefix)
            if end is None:
                return None

        max_points = self._point_count(values, self._flat_dataset_prefix)
        if max_points is None:
            return None
        start = max(0, min(start, max_points))
        end = max(start, min(end, max_points))
        return self._slice_data(values, self._flat_dataset_prefix, start, end)

    def _extract_preview_data(self, values: dict[str, Any]) -> dict[str, Any] | None:
        if self._preview_dataset_prefix is None:
            return None
        num_points = self._point_count(
            values, self._preview_dataset_prefix, allow_partial=True
        )
        if num_points is None:
            return None
        return self._slice_data(
            values,
            self._preview_dataset_prefix,
            0,
            num_points,
            allow_missing=True,
        )

    def _point_count(
        self, values: dict[str, Any], dataset_prefix: str, allow_partial: bool = False
    ) -> int | None:
        lengths = []
        strict_names = self._strict_required_names()
        for name in strict_names:
            key = dataset_prefix + "points." + name
            if key not in values:
                return None
            try:
                lengths.append(len(values[key]))
            except TypeError:
                return None
        if allow_partial:
            for name in self._required_names():
                if name in strict_names:
                    continue
                key = dataset_prefix + "points." + name
                if key not in values:
                    continue
                try:
                    lengths.append(len(values[key]))
                except TypeError:
                    continue
        else:
            for name in self._required_names():
                if name in strict_names:
                    continue
                key = dataset_prefix + "points." + name
                if key not in values:
                    return None
                try:
                    lengths.append(len(values[key]))
                except TypeError:
                    return None
        if not lengths:
            return 0
        return min(lengths)

    def _slice_data(
        self,
        values: dict[str, Any],
        dataset_prefix: str,
        start: int,
        end: int,
        allow_missing: bool = False,
    ) -> dict[str, Any]:
        point_data = {}
        target_len = max(0, end - start)
        for name in self._required_names():
            key = dataset_prefix + "points." + name
            if key not in values:
                if allow_missing:
                    point_data[name] = [None] * target_len
                    continue
                raise KeyError(key)

            sliced = values[key][start:end]
            if allow_missing and _safe_len(sliced) is not None and len(sliced) < target_len:
                # Keep channel arrays aligned with axes while lagging optional fields
                # (e.g. nested subscan schemas) catch up.
                sliced = list(sliced) + [None] * (target_len - len(sliced))
            point_data[name] = sliced
        return point_data

    def _required_names(self) -> list[str]:
        return [f"axis_{i}" for i in range(len(self.axes))] + [
            "channel_" + c for c in self._channel_schemata.keys()
        ]

    def _strict_required_names(self) -> list[str]:
        required = [f"axis_{i}" for i in range(len(self.axes))]
        for channel_name, schema in self._channel_schemata.items():
            if schema["type"] in ("opaque", "subscan"):
                continue
            required.append("channel_" + channel_name)
        return required

    def _compute_subscan_site_keys(self) -> None:
        if not self._schema_channel_path:
            return
        if "/" in self._schema_channel_path:
            target_path, channel_leaf = self._schema_channel_path.rsplit("/", 1)
        else:
            target_path, channel_leaf = "", self._schema_channel_path
        if not channel_leaf.endswith("spec"):
            return

        fragment_fqn = self._schema.get("fragment_fqn", "")
        if not fragment_fqn:
            return

        name_prefix = channel_leaf[:-4]
        site_id = "|".join([target_path, name_prefix, fragment_fqn])
        self._site_id = hashlib.sha1(site_id.encode("utf-8")).hexdigest()[:12]

        target_path_slug = target_path.replace("/", "_") if target_path else "root"
        scan_name = name_prefix.strip("_") or "subscan"
        fragment_name = fragment_fqn.split(".")[-1]
        raw_slug = f"{target_path_slug}_{scan_name}_{fragment_name}"
        self._site_slug = re.sub(r"[^A-Za-z0-9]+", "_", raw_slug).strip("_").lower()
        if not self._site_slug:
            self._site_slug = "subscan"

    def _resolve_dataset_prefixes(self, values: dict[str, Any]) -> None:
        self._refresh_schema_channel_path()
        if self._site_id is None or self._site_slug is None:
            return
        if self._preview_dataset_prefix is None:
            self._preview_dataset_prefix = _find_subscan_dataset_prefix(
                values, "subscan_preview", self._site_slug, self._site_id
            )
        if self._flat_dataset_prefix is None:
            self._flat_dataset_prefix = _find_subscan_dataset_prefix(
                values, "subscan_flat", self._site_slug, self._site_id
            )

    def get_channel_schemata(self) -> dict[str, Any]:
        return self._channel_schemata

    def get_point_data(self) -> dict[str, Any]:
        return self._point_data

    def is_completed(self) -> bool | None:
        values = self._parent.context.get_dataset_values()
        if not values:
            parent_is_completed = getattr(self._parent, "is_completed", None)
            if parent_is_completed is None:
                return None
            return parent_is_completed()

        self._resolve_dataset_prefixes(values)
        source_index = self._get_selected_source_index()
        if self._flat_dataset_prefix is not None and source_index is not None:
            starts = values.get(self._flat_dataset_prefix + "starts", None)
            if starts is not None and 0 <= source_index < len(starts):
                # Completed segments before the currently running one are stable.
                if source_index < len(starts) - 1:
                    return True
                flat_completed = values.get(self._flat_dataset_prefix + "completed", None)
                if flat_completed is not None:
                    return bool(flat_completed)

        if self._preview_dataset_prefix is not None:
            preview_completed = values.get(
                self._preview_dataset_prefix + "completed", None
            )
            if preview_completed is not None:
                return bool(preview_completed)

        if self._flat_dataset_prefix is not None:
            flat_completed = values.get(self._flat_dataset_prefix + "completed", None)
            if flat_completed is not None:
                return bool(flat_completed)

        parent_is_completed = getattr(self._parent, "is_completed", None)
        if parent_is_completed is not None:
            completed = parent_is_completed()
            if completed is False:
                return False

        return None

    def get_analysis_result_source(self, name: str) -> FixedDataSource | None:
        return self._analysis_results.get(name, None)


def create_subscan_roots(model: SinglePointModel) -> dict[str, SubscanRoot]:
    schemata = model.get_channel_schemata()
    if schemata is None:
        return {}
    result = {}
    for key, schema in schemata.items():
        if schema["type"] != "subscan":
            continue
        root = SubscanRoot(model, key)
        result[root.name] = root
    return result


def _find_subscan_dataset_prefix(
    values: dict[str, Any], kind: str, slug: str, site_id: str
) -> str | None:
    token = f".{kind}.{slug}__{site_id}."
    for key in values.keys():
        idx = key.find(token)
        if idx >= 0:
            return key[: idx + len(token)]
    return None


def _find_subscan_dataset_prefix_by_channel_path(
    values: dict[str, Any], kind: str, schema_channel_path: str
) -> str | None:
    if not schema_channel_path:
        return None
    if "/" in schema_channel_path:
        target_path, channel_leaf = schema_channel_path.rsplit("/", 1)
    else:
        target_path, channel_leaf = "", schema_channel_path
    if not channel_leaf.endswith("spec"):
        return None

    name_prefix = channel_leaf[:-4]
    target_path_slug = target_path.replace("/", "_") if target_path else "root"
    scan_name = name_prefix.strip("_") or "subscan"
    slug_prefix = re.sub(
        r"[^A-Za-z0-9]+", "_", f"{target_path_slug}_{scan_name}_"
    ).strip("_").lower()

    token = f".{kind}."
    candidates = {}
    for key in values.keys():
        idx = key.find(token)
        if idx < 0:
            continue
        suffix = key[idx + len(token) :]
        first_component = suffix.split(".", 1)[0]
        if "__" not in first_component:
            continue
        slug, _site_id = first_component.rsplit("__", 1)
        if slug.startswith(slug_prefix):
            prefix = key[: idx + len(token)] + first_component + "."
            candidates[prefix] = slug
    if len(candidates) == 1:
        return next(iter(candidates.keys()))
    if candidates:
        # If there are multiple candidates (e.g. nested subscan trees), prefer the
        # most local root by picking the shortest slug.
        return min(candidates.items(), key=lambda item: len(item[1]))[0]
    return None


def _safe_len(value) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _prefix_equal(left, right, n: int) -> bool:
    if n == 0:
        return True
    try:
        return _value_equal(left[:n], right[:n])
    except Exception:
        return False


def _value_equal(left, right) -> bool:
    if isinstance(left, numpy.ndarray) or isinstance(right, numpy.ndarray):
        return numpy.array_equal(left, right)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(
            _value_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            return False
        return all(
            _value_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_value_equal(left[k], right[k]) for k in left.keys())
    return left == right

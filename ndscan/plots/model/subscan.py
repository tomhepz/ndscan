import hashlib
import json
import logging
import re
from typing import Any

from ...utils import strip_suffix
from . import FixedDataSource, Model, Root, ScanModel, SinglePointModel

logger = logging.getLogger(__name__)


class SubscanRoot(Root):
    def __init__(self, parent: SinglePointModel, schema_key: str):
        super().__init__()
        self._parent = parent
        self._schema_key = schema_key
        self._model = None
        self._schema = None
        self._schema_str = None
        self._parent.point_changed.connect(self._update)

        self.name = strip_suffix(self._schema_key, "_spec")
        if self.name == self._schema_key:
            raise ValueError(
                "Unexpected scan schema channel name: {}".format(self._schema_key)
            )

        self._update(parent.get_point())

    def _update(self, data: dict[str, Any]) -> None:
        if data is None:
            if self._model:
                self._model.quit()
                self.model_changed.emit(None)
            self._model = None
            self._schema = None
            self._schema_str = None
            return

        schema_str = data[self._schema_key]
        if schema_str == self._schema_str:
            return
        self._schema_str = schema_str
        self._schema = json.loads(schema_str)

        self._model = SubscanModel(
            self._schema, self._parent, self.name + "_", self._schema_key
        )
        self.model_changed.emit(self._model)

    def get_model(self) -> Model | None:
        return self._model

    def get_title(self) -> str:
        return f"subscan '{self.name}'"


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
        self._schema_channel_path = parent_schemata.get(self._schema_key, {}).get(
            "path", ""
        )
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

    def quit(self) -> None:
        self._parent.point_changed.disconnect(self._update)
        self._parent.context.datasets_changed.disconnect(self._on_datasets_changed)

    def _update(self, parent_data: dict[str, Any] | None) -> None:
        if parent_data is None:
            logger.debug("Ignoring update")
            return

        point_data = self._extract_dataset_point_data()
        if point_data is None:
            point_data = self._extract_embedded_point_data(parent_data)
        if point_data is not None:
            self._point_data = point_data
            self.points_rewritten.emit(self._point_data)

        for r, c in self._analysis_result_mappings:
            if c in parent_data:
                self._analysis_results[r].set(parent_data[c])

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

    def _extract_dataset_point_data(self) -> dict[str, Any] | None:
        values = self._parent.context.get_dataset_values()
        if not values:
            return None
        self._resolve_dataset_prefixes(values)

        source_index = self._get_selected_source_index()
        if source_index is not None:
            flat_data = self._extract_flat_segment(values, source_index)
            if flat_data is not None:
                return flat_data

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
        num_points = self._point_count(values, self._preview_dataset_prefix)
        if num_points is None:
            return None
        return self._slice_data(values, self._preview_dataset_prefix, 0, num_points)

    def _point_count(self, values: dict[str, Any], dataset_prefix: str) -> int | None:
        lengths = []
        for name in self._required_names():
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
        self, values: dict[str, Any], dataset_prefix: str, start: int, end: int
    ) -> dict[str, Any]:
        point_data = {}
        for name in self._required_names():
            point_data[name] = values[dataset_prefix + "points." + name][start:end]
        return point_data

    def _required_names(self) -> list[str]:
        return [f"axis_{i}" for i in range(len(self.axes))] + [
            "channel_" + c for c in self._channel_schemata.keys()
        ]

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

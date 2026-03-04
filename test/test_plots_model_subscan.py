import hashlib
import json
import unittest

from ndscan.plots.model import Context, SinglePointModel
from ndscan.plots.model.subscan import (
    SubscanModel,
    SubscanRoot,
    _find_subscan_dataset_prefix,
    _find_subscan_dataset_prefix_by_channel_path,
    _prefix_equal,
)


class _StubParent(SinglePointModel):
    def __init__(self, point, channel_schemata, source_index):
        super().__init__(schema_revision=2, context=Context())
        self._point = point
        self._channel_schemata = channel_schemata
        self._source_index = source_index

    def get_channel_schemata(self):
        return self._channel_schemata

    def get_point(self):
        return self._point

    def get_source_index(self):
        return self._source_index

    def set_source_index(self, idx):
        self._source_index = idx
        self.point_changed.emit(self._point)

    def set_point(self, point):
        self._point = point
        self.point_changed.emit(point)


class SubscanDatasetModelTest(unittest.TestCase):
    def _make_schema(self):
        return {
            "fragment_fqn": "pkg.inner.Fragment",
            "axes": [{"param": {"fqn": "x", "type": "float", "spec": {}}, "path": "*"}],
            "channels": {
                "y": {
                    "description": "Y",
                    "path": "root/inner/y",
                    "type": "float",
                    "unit": "",
                }
            },
            "analysis_results": {},
        }

    def _make_parent(self, schema, source_index):
        return _StubParent(
            point={
                "scan_spec": json.dumps(schema),
                "scan_axis_0": [0.0],
                "scan_channel_y": [0.0],
            },
            channel_schemata={
                "scan_spec": {
                    "description": "Subscan spec",
                    "path": "root/scan_spec",
                    "type": "subscan",
                }
            },
            source_index=source_index,
        )

    def test_uses_flat_segment_for_selected_point_when_embedded_absent(self):
        schema = self._make_schema()
        parent = self._make_parent(schema, source_index=1)
        parent.set_point({"scan_spec": json.dumps(schema)})
        model = SubscanModel(schema, parent, "scan_", "scan_spec")

        flat_prefix = f"ndscan.rid_123.subscan_flat.{model._site_slug}__{model._site_id}."
        preview_prefix = (
            f"ndscan.rid_123.subscan_preview.{model._site_slug}__{model._site_id}."
        )
        parent.context.update_datasets(
            {
                flat_prefix + "completed": True,
                flat_prefix + "starts": [0, 2],
                flat_prefix + "points.axis_0": [1.0, 2.0, 5.0, 6.0, 7.0],
                flat_prefix + "points.channel_y": [10.0, 20.0, 50.0, 60.0, 70.0],
                preview_prefix + "points.axis_0": [99.0],
                preview_prefix + "points.channel_y": [199.0],
            },
            [],
        )

        self.assertEqual(list(model.get_point_data()["axis_0"]), [5.0, 6.0, 7.0])
        self.assertEqual(list(model.get_point_data()["channel_y"]), [50.0, 60.0, 70.0])

    def test_falls_back_to_embedded_if_selected_segment_is_unavailable(self):
        schema = self._make_schema()
        parent = self._make_parent(schema, source_index=5)
        model = SubscanModel(schema, parent, "scan_", "scan_spec")

        flat_prefix = f"ndscan.rid_123.subscan_flat.{model._site_slug}__{model._site_id}."
        preview_prefix = (
            f"ndscan.rid_123.subscan_preview.{model._site_slug}__{model._site_id}."
        )
        parent.context.update_datasets(
            {
                flat_prefix + "completed": False,
                flat_prefix + "starts": [0, 2],
                flat_prefix + "points.axis_0": [1.0, 2.0, 5.0],
                flat_prefix + "points.channel_y": [10.0, 20.0, 50.0],
                preview_prefix + "points.axis_0": [9.0, 10.0],
                preview_prefix + "points.channel_y": [90.0, 100.0],
            },
            [],
        )

        self.assertEqual(list(model.get_point_data()["axis_0"]), [0.0])
        self.assertEqual(list(model.get_point_data()["channel_y"]), [0.0])

    def test_selected_point_prefers_embedded_data_over_preview(self):
        schema = self._make_schema()
        parent = self._make_parent(schema, source_index=1)
        parent.set_point(
            {
                "scan_spec": json.dumps(schema),
                "scan_axis_0": [11.0, 12.0, 13.0],
                "scan_channel_y": [101.0, 102.0, 103.0],
            }
        )
        model = SubscanModel(schema, parent, "scan_", "scan_spec")

        flat_prefix = f"ndscan.rid_123.subscan_flat.{model._site_slug}__{model._site_id}."
        preview_prefix = (
            f"ndscan.rid_123.subscan_preview.{model._site_slug}__{model._site_id}."
        )
        parent.context.update_datasets(
            {
                flat_prefix + "completed": True,
                flat_prefix + "starts": [0, 2],
                flat_prefix + "points.axis_0": [1.0, 2.0, 5.0, 6.0, 7.0],
                flat_prefix + "points.channel_y": [10.0, 20.0, 50.0, 60.0, 70.0],
                preview_prefix + "points.axis_0": [9.0, 10.0],
                preview_prefix + "points.channel_y": [90.0, 100.0],
            },
            [],
        )

        self.assertEqual(list(model.get_point_data()["axis_0"]), [11.0, 12.0, 13.0])
        self.assertEqual(
            list(model.get_point_data()["channel_y"]), [101.0, 102.0, 103.0]
        )

    def test_subscan_dataset_prefix_lookup(self):
        values = {
            "ndscan.rid_88.subscan_flat.root_scan_fragment__abc123.points.axis_0": [1]
        }
        self.assertEqual(
            _find_subscan_dataset_prefix(values, "subscan_flat", "root_scan_fragment", "abc123"),
            "ndscan.rid_88.subscan_flat.root_scan_fragment__abc123.",
        )

    def test_subscan_dataset_prefix_lookup_by_channel_path(self):
        values = {
            "ndscan.rid_88.subscan_preview.root_scan_fragment__abc123.points.axis_0": [
                1
            ]
        }
        self.assertEqual(
            _find_subscan_dataset_prefix_by_channel_path(
                values, "subscan_preview", "root/scan_spec"
            ),
            "ndscan.rid_88.subscan_preview.root_scan_fragment__abc123.",
        )

    def test_can_update_from_preview_without_parent_point(self):
        schema = self._make_schema()
        parent = self._make_parent(schema, source_index=None)
        parent.set_point(None)

        site_id = hashlib.sha1("root|scan_|pkg.inner.Fragment".encode("utf-8")).hexdigest()[
            :12
        ]
        prefix = f"ndscan.rid_123.subscan_preview.root_scan_fragment__{site_id}."
        parent.context.update_datasets(
            {
                prefix + "fragment_fqn": "pkg.inner.Fragment",
                prefix + "axes": json.dumps(schema["axes"]),
                prefix + "channels": json.dumps(schema["channels"]),
                prefix + "points.axis_0": [1.0, 2.0, 3.0],
                prefix + "points.channel_y": [10.0, 20.0, 30.0],
            },
            [],
        )

        root = SubscanRoot(parent, "scan_spec")
        self.assertIsNotNone(root.get_model())
        self.assertEqual(list(root.get_model().get_point_data()["axis_0"]), [1.0, 2.0, 3.0])

    def test_preview_growth_emits_points_appended(self):
        schema = self._make_schema()
        parent = self._make_parent(schema, source_index=None)
        parent.set_point(None)
        model = SubscanModel(schema, parent, "scan_", "scan_spec", "root/scan_spec")

        rewritten_count = 0
        appended_count = 0

        def on_rewritten(*_args):
            nonlocal rewritten_count
            rewritten_count += 1

        def on_appended(*_args):
            nonlocal appended_count
            appended_count += 1

        model.points_rewritten.connect(on_rewritten)
        model.points_appended.connect(on_appended)

        site_id = hashlib.sha1("root|scan_|pkg.inner.Fragment".encode("utf-8")).hexdigest()[
            :12
        ]
        prefix = f"ndscan.rid_123.subscan_preview.root_scan_fragment__{site_id}."

        parent.context.update_datasets(
            {
                prefix + "fragment_fqn": "pkg.inner.Fragment",
                prefix + "axes": json.dumps(schema["axes"]),
                prefix + "channels": json.dumps(schema["channels"]),
                prefix + "points.axis_0": [1.0],
                prefix + "points.channel_y": [10.0],
            },
            [],
        )
        parent.context.update_datasets(
            {
                prefix + "fragment_fqn": "pkg.inner.Fragment",
                prefix + "axes": json.dumps(schema["axes"]),
                prefix + "channels": json.dumps(schema["channels"]),
                prefix + "points.axis_0": [1.0, 2.0],
                prefix + "points.channel_y": [10.0, 20.0],
            },
            [],
        )

        self.assertEqual(rewritten_count, 1)
        self.assertEqual(appended_count, 1)

    def test_can_bootstrap_from_preview_when_parent_point_lacks_spec(self):
        schema = self._make_schema()
        parent = self._make_parent(schema, source_index=None)
        parent.set_point({"scan_axis_0": [0.0], "scan_channel_y": [1.0]})

        site_id = hashlib.sha1("root|scan_|pkg.inner.Fragment".encode("utf-8")).hexdigest()[
            :12
        ]
        prefix = f"ndscan.rid_123.subscan_preview.root_scan_fragment__{site_id}."
        parent.context.update_datasets(
            {
                prefix + "fragment_fqn": "pkg.inner.Fragment",
                prefix + "axes": json.dumps(schema["axes"]),
                prefix + "channels": json.dumps(schema["channels"]),
                prefix + "points.axis_0": [1.0, 2.0],
                prefix + "points.channel_y": [10.0, 20.0],
            },
            [],
        )

        root = SubscanRoot(parent, "scan_spec")
        self.assertIsNotNone(root.get_model())
        self.assertEqual(list(root.get_model().get_point_data()["axis_0"]), [1.0, 2.0])

    def test_prefix_equal_handles_ragged_nested_lists(self):
        left = [[0.0, 1.0], [0.0, 1.0, 2.0]]
        right = [[0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 1.0]]
        self.assertTrue(_prefix_equal(left, right, 2))

    def test_preview_partial_optional_channels_do_not_block_live_points(self):
        schema = self._make_schema()
        schema["channels"]["nested_spec"] = {
            "description": "nested",
            "path": "root/inner/spec",
            "type": "subscan",
            "unit": "",
        }
        parent = self._make_parent(schema, source_index=None)
        parent.set_point(None)
        model = SubscanModel(schema, parent, "scan_", "scan_spec", "root/scan_spec")

        prefix = f"ndscan.rid_123.subscan_preview.{model._site_slug}__{model._site_id}."
        parent.context.update_datasets(
            {
                prefix + "points.axis_0": [1.0, 2.0],
                prefix + "points.channel_y": [10.0, 20.0],
            },
            [],
        )

        self.assertEqual(list(model.get_point_data()["axis_0"]), [1.0, 2.0])
        self.assertEqual(list(model.get_point_data()["channel_nested_spec"]), [None, None])


if __name__ == "__main__":
    unittest.main()

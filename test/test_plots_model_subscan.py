import json
import unittest

from ndscan.plots.model import Context, SinglePointModel
from ndscan.plots.model.subscan import SubscanModel, _find_subscan_dataset_prefix


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

    def test_uses_flat_segment_for_selected_point(self):
        schema = self._make_schema()
        parent = self._make_parent(schema, source_index=1)
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

    def test_falls_back_to_preview_if_selected_segment_is_unavailable(self):
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

        self.assertEqual(list(model.get_point_data()["axis_0"]), [9.0, 10.0])
        self.assertEqual(list(model.get_point_data()["channel_y"]), [90.0, 100.0])

    def test_subscan_dataset_prefix_lookup(self):
        values = {
            "ndscan.rid_88.subscan_flat.root_scan_fragment__abc123.points.axis_0": [1]
        }
        self.assertEqual(
            _find_subscan_dataset_prefix(values, "subscan_flat", "root_scan_fragment", "abc123"),
            "ndscan.rid_88.subscan_flat.root_scan_fragment__abc123.",
        )


if __name__ == "__main__":
    unittest.main()

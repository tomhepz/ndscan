import unittest

from ndscan.plots.model import Context, ScanModel
from ndscan.plots.model.select_point import SelectPointFromScanModel


class _StubScanModel(ScanModel):
    def __init__(self):
        super().__init__(
            axes=[{"param": {"fqn": "x", "type": "float", "spec": {}}, "path": "*"}],
            schema_revision=2,
            context=Context(),
        )
        self._channel_schemata = {
            "scan_spec": {
                "description": "subscan",
                "path": "root/scan_spec",
                "type": "subscan",
                "unit": "",
            }
        }
        # Channel payload is intentionally identical for both points.
        self._point_data = {
            "axis_0": [0.0, 1.0],
            "channel_scan_spec": ["same", "same"],
        }

    def get_channel_schemata(self):
        return self._channel_schemata

    def get_point_data(self):
        return self._point_data

    def get_analysis_result_source(self, name):
        return None


class SelectPointModelTest(unittest.TestCase):
    def test_index_change_emits_even_for_identical_point_payloads(self):
        source = _StubScanModel()
        selected = SelectPointFromScanModel(source)

        seen = []
        selected.point_changed.connect(lambda point: seen.append(point))

        selected.set_source_index(0)
        selected.set_source_index(1)

        self.assertEqual(len(seen), 2)
        self.assertEqual(selected.get_source_index(), 1)

    def test_reselecting_same_index_is_noop(self):
        source = _StubScanModel()
        selected = SelectPointFromScanModel(source)

        seen = []
        selected.point_changed.connect(lambda point: seen.append(point))

        selected.set_source_index(0)
        selected.set_source_index(0)

        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()

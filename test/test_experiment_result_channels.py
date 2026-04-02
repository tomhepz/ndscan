import unittest

import numpy as np

from mock_environment import HasEnvironmentCase

from ndscan.define.result_channels import (
    ArrayChannel,
    ArraySink,
    LastValueSink,
    ResettableAppendingDatasetSink,
    TeeSink,
)


class TeeSinkTest(unittest.TestCase):
    def test_push_forwards_to_both_sinks(self):
        primary = ArraySink()
        secondary = ArraySink()
        sink = TeeSink(primary, secondary)

        sink.push(1)
        sink.push(2)

        assert primary.get_all() == [1, 2]
        assert secondary.get_all() == [1, 2]

    def test_clear_tolerates_secondary_without_clear_method(self):
        primary = ArraySink()
        secondary = LastValueSink()
        sink = TeeSink(primary, secondary)

        sink.push(5)
        sink.clear()

        assert primary.get_all() == []
        assert secondary.get_last() == 5


class ResettableAppendingDatasetSinkTest(HasEnvironmentCase):
    def test_clear_resets_dataset_and_last_value(self):
        sink = self.create(ResettableAppendingDatasetSink, "sink.values")

        sink.push(1)
        sink.push(2)
        sink.clear()

        assert self.dataset_db.get("sink.values") == []
        assert sink.get_last() is None
        assert sink.get_all() == []


class ArrayChannelTest(unittest.TestCase):
    def test_describe_includes_shape_and_dim_names(self):
        channel = ArrayChannel(
            "counts",
            "ROI counts",
            element_type="int",
            shape=(8, 1),
            dim_names=("group", "roi"),
            unit="cts",
            scale=1.0,
        )

        self.assertEqual(
            channel.describe(),
            {
                "path": "counts",
                "description": "ROI counts",
                "type": "array",
                "element_type": "int",
                "shape": [8, 1],
                "dim_names": ["group", "roi"],
                "scale": 1.0,
                "unit": "cts",
            },
        )

    def test_push_coerces_and_validates_shape(self):
        channel = ArrayChannel("counts", element_type="int", shape=(2, 2))
        sink = ArraySink()
        channel.set_sink(sink)

        channel.push([[1.2, 2.9], [3.4, 4.5]])

        np.testing.assert_array_equal(
            sink.get_last(),
            np.asarray([[1, 2], [3, 4]], dtype=int),
        )

        with self.assertRaises(ValueError):
            channel.push([1, 2, 3])

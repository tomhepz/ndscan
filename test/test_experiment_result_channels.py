import unittest

from mock_environment import HasEnvironmentCase

from ndscan.experiment.result_channels import (
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

        self.assertEqual(primary.get_all(), [1, 2])
        self.assertEqual(secondary.get_all(), [1, 2])

    def test_get_last_uses_primary_sink(self):
        primary = ArraySink()
        secondary = LastValueSink()
        sink = TeeSink(primary, secondary)

        sink.push("a")
        sink.push("b")

        self.assertEqual(sink.get_last(), "b")

    def test_get_all_uses_primary_sink(self):
        primary = ArraySink()
        secondary = LastValueSink()
        sink = TeeSink(primary, secondary)

        sink.push(10)
        sink.push(20)

        self.assertEqual(sink.get_all(), [10, 20])

    def test_clear_clears_sinks_with_clear_method(self):
        primary = ArraySink()
        secondary = ArraySink()
        sink = TeeSink(primary, secondary)

        sink.push(1)
        sink.push(2)
        sink.clear()

        self.assertEqual(primary.get_all(), [])
        self.assertEqual(secondary.get_all(), [])

    def test_clear_tolerates_secondary_without_clear_method(self):
        primary = ArraySink()
        secondary = LastValueSink()
        sink = TeeSink(primary, secondary)

        sink.push(5)
        sink.clear()

        self.assertEqual(primary.get_all(), [])
        self.assertEqual(secondary.get_last(), 5)


class ResettableAppendingDatasetSinkTest(HasEnvironmentCase):
    def test_clear_resets_dataset_and_last_value(self):
        sink = self.create(ResettableAppendingDatasetSink, "sink.values")

        sink.push(1)
        sink.push(2)

        self.assertEqual(self.dataset_db.get("sink.values"), [1, 2])
        self.assertEqual(sink.get_last(), 2)
        self.assertEqual(sink.get_all(), [1, 2])

        sink.clear()

        self.assertEqual(self.dataset_db.get("sink.values"), [])
        self.assertIsNone(sink.get_last())
        self.assertEqual(sink.get_all(), [])

    def test_push_after_clear_starts_new_series(self):
        sink = self.create(ResettableAppendingDatasetSink, "sink.values")

        sink.push(10)
        sink.push(20)
        sink.clear()
        sink.push(30)

        self.assertEqual(self.dataset_db.get("sink.values"), [30])
        self.assertEqual(sink.get_last(), 30)
        self.assertEqual(sink.get_all(), [30])

    def test_clear_before_first_push_is_allowed(self):
        sink = self.create(ResettableAppendingDatasetSink, "sink.values")

        sink.clear()
        self.assertEqual(self.dataset_db.get("sink.values"), [])
        self.assertIsNone(sink.get_last())
        self.assertEqual(sink.get_all(), [])

        sink.push(99)
        self.assertEqual(self.dataset_db.get("sink.values"), [99])

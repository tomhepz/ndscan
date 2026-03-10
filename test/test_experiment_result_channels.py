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

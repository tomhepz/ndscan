import unittest

from ndscan.experiment.result_channels import ArraySink, LastValueSink, TeeSink


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


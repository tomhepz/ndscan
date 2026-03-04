import json
import unittest

from sipyco.sync_struct import Notifier

from ndscan.plots.model import Context
from ndscan.plots.model.subscriber import SubscriberRoot, SubscriberScanModel
from ndscan.utils import SCHEMA_REVISION, SCHEMA_REVISION_KEY


class SinglePointTest(unittest.TestCase):
    def setUp(self):
        self.context = Context()
        self.root = SubscriberRoot("ndscan.", self.context)
        self.datasets = Notifier(
            {
                "ndscan.axes": (False, "[]", {}),
                "ndscan.channels": (
                    False,
                    json.dumps(
                        {
                            "foo": {
                                "description": "Foo",
                                "path": "foo",
                                "type": "int",
                                "unit": "",
                            },
                            "bar": {
                                "description": "Bar",
                                "path": "foo",
                                "type": "int",
                                "unit": "",
                            },
                        }
                    ),
                    {},
                ),
                ("ndscan." + SCHEMA_REVISION_KEY): (False, SCHEMA_REVISION, {}),
            }
        )
        self.pending_mods = []
        self.datasets.publish = lambda a: self.pending_mods.append(a)

    def init(self):
        self.pending_mods = [
            {"action": "init", "struct": self.datasets.raw_view.copy()}
        ]
        self.sync()

    def sync(self):
        values = {k: v[1] for k, v in self.datasets.raw_view.items()}
        self.root.data_changed(values, self.pending_mods)
        self.pending_mods.clear()

    def test_new_point(self):
        self.init()
        self.datasets["ndscan.point.foo"] = (False, 42, {})
        self.datasets["ndscan.point.bar"] = (False, 23, {})
        self.datasets["ndscan.point_phase"] = (False, True, {})
        self.sync()
        self.assertEqual(self.root.get_model().get_point(), {"foo": 42, "bar": 23})

    def test_halfway(self):
        self.datasets["ndscan.point.foo"] = (False, 42, {})
        self.init()

        # No complete point yet.
        self.assertIsNone(self.root.get_model().get_point())

        self.datasets["ndscan.point.bar"] = (False, 23, {})
        self.datasets["ndscan.point_phase"] = (False, True, {})
        self.sync()
        self.assertEqual(self.root.get_model().get_point(), {"foo": 42, "bar": 23})

    def test_one_and_a_half(self):
        self.datasets["ndscan.point.foo"] = (False, 42, {})
        self.init()

        # No complete point yet.
        self.assertIsNone(self.root.get_model().get_point())

        self.datasets["ndscan.point.bar"] = (False, 23, {})
        self.datasets["ndscan.point_phase"] = (False, True, {})

        # Already write foo value of next point.
        self.datasets["ndscan.point.foo"] = (False, 0, {})
        self.sync()

        # Foo should still be the old value.
        self.assertEqual(self.root.get_model().get_point(), {"foo": 42, "bar": 23})

    def test_preexisting(self):
        self.datasets["ndscan.point.foo"] = (False, 42, {})
        self.datasets["ndscan.point.bar"] = (False, 42, {})
        self.datasets["ndscan.point_phase"] = (False, True, {})
        self.datasets["ndscan.point.foo"] = (False, 0, {})
        self.init()

        # Can't know whether point is complete (it indeed isn't).
        self.assertIsNone(self.root.get_model().get_point())

        self.datasets["ndscan.point.bar"] = (False, 1, {})
        self.datasets["ndscan.point_phase"] = (False, False, {})
        self.sync()

        self.assertEqual(self.root.get_model().get_point(), {"foo": 0, "bar": 1})

    def test_already_completed(self):
        self.datasets["ndscan.point.foo"] = (False, 42, {})
        self.datasets["ndscan.point.bar"] = (False, 23, {})
        self.datasets["ndscan.point_phase"] = (False, True, {})
        self.datasets["ndscan.completed"] = (False, True, {})
        self.init()
        self.assertEqual(self.root.get_model().get_point(), {"foo": 42, "bar": 23})


class ScanModelTest(unittest.TestCase):
    def test_in_place_list_growth_emits_appended(self):
        context = Context()
        model = SubscriberScanModel(
            axes=[
                {
                    "param": {
                        "fqn": "test.x",
                        "description": "x",
                        "type": "float",
                        "spec": {},
                    },
                    "path": "",
                }
            ],
            prefix="ndscan.",
            schema_revision=SCHEMA_REVISION,
            context=context,
        )

        appended_count = 0

        def on_appended(*_args):
            nonlocal appended_count
            appended_count += 1

        model.points_appended.connect(on_appended)

        axis_values = []
        channel_values = []
        values = {
            "ndscan.channels": json.dumps(
                {
                    "y": {
                        "description": "y",
                        "path": "root/y",
                        "type": "float",
                        "unit": "",
                    }
                }
            ),
            "ndscan.online_analyses": "{}",
            "ndscan.points.axis_0": axis_values,
            "ndscan.points.channel_y": channel_values,
            "ndscan.completed": False,
        }

        model.data_changed(values, [])
        self.assertEqual(appended_count, 0)

        axis_values.append(1.0)
        channel_values.append(2.0)
        model.data_changed(values, [])
        self.assertEqual(appended_count, 1)

        axis_values.append(2.0)
        channel_values.append(4.0)
        model.data_changed(values, [])
        self.assertEqual(appended_count, 2)

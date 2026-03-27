import json
import unittest
from unittest.mock import patch

from examples.host_runtime_live_viewer_demo import HostRuntimeLiveViewerDemo
from mock_environment import HasEnvironmentCase

from ndscan.define.fragment import ExpFragment
from ndscan.define.result_channels import FloatChannel
from ndscan.plots.runtime.live import snapshot_from_live_values
from ndscan.plots.runtime.viewer import _default_x_choices, _default_y_choices
from ndscan.runtime.api import make_fragment_prepared_scan_exp
from ndscan.scan.request import ScanRequest


class AppletLaunchFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_result("value", FloatChannel)

    def run_once(self):
        self.value.push(1.0)


class RuntimeLiveSnapshotTest(unittest.TestCase):
    def test_snapshot_from_live_values_reconstructs_site_tree(self):
        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.parameters": json.dumps({"x": {"is_scanned": True}}),
            prefix + "scan.channels": json.dumps({"channel_value": {"type": "float"}}),
            prefix + "points.x": [0.0, 1.0],
            prefix + "points.channel_value": [10.0, 11.0],
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
            prefix + "scan_x.site.path": json.dumps(["scan_x"]),
            prefix + "scan_x.site.parent_path": json.dumps([]),
            prefix + "scan_x.site.fragment_fqn": "ChildFragment",
            prefix + "scan_x.scan.parameters": json.dumps({"child_x": {"is_scanned": True}}),
            prefix + "scan_x.scan.channels": json.dumps({"child_value": {"type": "float"}}),
            prefix + "scan_x.points.child_x": [2.0, 3.0, 4.0, 5.0],
            prefix + "scan_x.points.child_value": [20.0, 30.0, 40.0, 50.0],
            prefix + "scan_x.segments.start_index": [0, 2],
            prefix + "scan_x.segments.parent_point_index": [0, 1],
            prefix + "scan_x.state.current_segment": 1,
            prefix + "scan_x.state.num_points": 4,
            prefix + "scan_x.state.completed": False,
        }

        snapshot = snapshot_from_live_values(prefix, values)

        root = snapshot.get_site(())
        child = snapshot.get_site(("scan_x",))

        self.assertEqual(root.fragment_fqn, "RootFragment")
        self.assertEqual(list(root.point_data["x"]), [0.0, 1.0])
        self.assertEqual([site.path for site in snapshot.child_sites(())], [("scan_x",)])
        self.assertEqual(len(child.segments_for_parent_point(1)), 1)
        self.assertEqual(
            child.slice_point_data(2, 4)["child_value"],
            [40.0, 50.0],
        )

    def test_runtime_choice_labels_use_schema_descriptions(self):
        prefix = "ndscan.rid_0.site.root."
        values = {
            prefix + "site.path": json.dumps([]),
            prefix + "site.fragment_fqn": "RootFragment",
            prefix + "scan.pseudoparams": json.dumps(
                {
                    "pseudoparam_0": {
                        "variable": {
                            "name": "logical_x",
                            "description": "Logical X",
                            "spec": {"unit": "MHz"},
                        }
                    }
                }
            ),
            prefix + "scan.parameters": json.dumps(
                {
                    "param_0": {
                        "path": "",
                        "is_scanned": True,
                        "param": {
                            "fqn": "demo.root.drive_frequency",
                            "description": "Drive Frequency",
                            "spec": {"unit": "kHz"},
                        },
                    }
                    ,
                    "param_1": {
                        "path": "child",
                        "is_scanned": False,
                        "param": {
                            "fqn": "demo.child.detuning",
                            "description": "Mapped Detuning",
                            "spec": {"unit": "MHz"},
                        },
                    },
                }
            ),
            prefix + "scan.channels": json.dumps(
                {
                    "channel_value": {
                        "path": "detector/counts",
                        "description": "Detected Counts",
                        "type": "float",
                        "unit": "cts",
                    }
                }
            ),
            prefix + "points.pseudoparam_0": [0.0, 1.0],
            prefix + "points.param_0": [10.0, 11.0],
            prefix + "points.param_1": [2.0, 3.0],
            prefix + "points.channel_value": [20.0, 21.0],
            prefix + "state.num_points": 2,
            prefix + "state.completed": False,
        }

        snapshot = snapshot_from_live_values(prefix, values)
        root = snapshot.get_site(())

        self.assertEqual(
            _default_x_choices(root),
            [
                ("pseudoparam_0", "logical_x (Logical X / MHz)"),
                ("param_0", "drive_frequency (Drive Frequency / kHz)"),
                ("param_1", "child/detuning (Mapped Detuning / MHz)"),
                ("channel_value", "detector/counts (Detected Counts / cts)"),
                ("__point_index__", "point_index"),
            ],
        )
        self.assertEqual(
            _default_y_choices(root),
            [
                ("pseudoparam_0", "logical_x (Logical X / MHz)"),
                ("param_0", "drive_frequency (Drive Frequency / kHz)"),
                ("param_1", "child/detuning (Mapped Detuning / MHz)"),
                ("channel_value", "detector/counts (Detected Counts / cts)"),
                ("__point_index__", "point_index"),
            ],
        )


class RuntimeAppletLaunchTest(HasEnvironmentCase):
    def test_prepared_scan_experiment_launches_runtime_applet(self):
        RuntimeAppletExperiment = make_fragment_prepared_scan_exp(
            AppletLaunchFragment,
            lambda fragment: ScanRequest.single(metadata={"demo_name": "runtime_applet"}),
        )

        exp = self.create(RuntimeAppletExperiment)
        exp.prepare()
        exp.run()

        self.ccb.issue.assert_called_once()
        args, kwargs = self.ccb.issue.call_args
        self.assertEqual(args[0], "create_applet")
        self.assertIn("-m ndscan.runtime_applet", args[2])
        self.assertIn("--prefix=ndscan.rid_0.site.root.", args[2])
        self.assertEqual(kwargs["group"], "ndscan")

    @patch("examples.host_runtime_live_viewer_demo.time.sleep", return_value=None)
    def test_live_viewer_demo_example_runs(self, _sleep):
        exp = self.create(HostRuntimeLiveViewerDemo)
        exp.prepare()
        exp.run()

        dataset_keys = set(self.dataset_db.data.keys())
        self.assertIn("ndscan.rid_0.site.root.scan_outer.site.path", dataset_keys)
        self.assertIn(
            "ndscan.rid_0.site.root.scan_outer.coarse_scan.site.path",
            dataset_keys,
        )
        self.assertIn(
            "ndscan.rid_0.site.root.scan_outer.fine_scan.site.path",
            dataset_keys,
        )
        for actual, expected in zip(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.points.channel_0"),
            [-1.8, -0.8, 0.2, 1.1, 2.0],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.points.channel_1"),
            [-1.8, -0.8, 0.2, 1.1, 2.0],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.coarse_scan.points.param_1"),
            [-1.8] * 9 + [-0.8] * 9 + [0.2] * 9 + [1.1] * 9 + [2.0] * 9,
        )
        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.coarse_scan.points.param_2"),
            [0.10] * 9 + [0.16] * 9 + [0.24] * 9 + [0.33] * 9 + [0.45] * 9,
        )
        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.fine_scan.points.param_1"),
            [-1.8] * 7 + [-0.8] * 7 + [0.2] * 7 + [1.1] * 7 + [2.0] * 7,
        )
        self.assertEqual(
            self.dataset_db.get("ndscan.rid_0.site.root.scan_outer.fine_scan.points.param_2"),
            [0.10] * 7 + [0.16] * 7 + [0.24] * 7 + [0.33] * 7 + [0.45] * 7,
        )

"""Small end-to-end checks for disk-backed prepared scan channels."""

import os
import shutil
import tempfile
from unittest.mock import patch

import h5py
import numpy as np
from mock_environment import HasEnvironmentCase

from ndscan.define.fragment import ExpFragment
from ndscan.define.parameters import FloatParam
from ndscan.define.result_channels import ArrayChannel, FloatChannel
from ndscan.results.scan_site_reader import read_scan_site_snapshot
from ndscan.runtime.api import (
    CheckpointPolicy, ExecutionPolicy, PreparedScan, ScanRequest,
    make_fragment_prepared_dashboard_scan_exp,
    setattr_prepared_child_scan,
)


class CameraFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_param("value", FloatParam, "value", 0.0)
        self.setattr_result("image", ArrayChannel, shape=(2, 3),
                            element_type="int", stream=True)
        self.setattr_result("counts", FloatChannel)

    def run_once(self):
        self.image.push(np.full((2, 3), int(self.value.get()), dtype=np.uint16))
        self.counts.push(self.value.get())


class RepeatedCameraFragment(ExpFragment):
    def build_fragment(self):
        self.child_scan = setattr_prepared_child_scan(
            self, "camera", CameraFragment, scan_name="shots"
        )
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.setattr_result("total", FloatChannel)

    def run_once(self):
        self.child_scan.configure(ScanRequest.explicit(
            [self.camera.value], [[self.outer.get()], [self.outer.get() + 1]],
            execution_policy=ExecutionPolicy(max_points_per_batch=1),
        ))
        self.child_scan.execute()
        result = self.child_scan.inspect()
        assert self.camera.image not in result.values
        self.total.push(sum(result.values[self.camera.counts]))


class StreamRuntimeTest(HasEnvironmentCase):
    def setUp(self):
        super().setUp()
        # Checkpoints check free space; keep these tests independent of the host disk
        # (test_runtime_storage_guard covers the low-space behaviour).
        roomy = shutil.disk_usage(tempfile.gettempdir())._replace(
            total=10**15, free=10**15
        )
        patcher = patch("ndscan.runtime.context.shutil.disk_usage", return_value=roomy)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dashboard_factory_passes_checkpoint_policy(self):
        experiment_class = make_fragment_prepared_dashboard_scan_exp(
            CameraFragment, checkpoint_policy=CheckpointPolicy(min_interval_s=999999)
        )
        experiment = self.create(experiment_class)
        manager = experiment._HasEnvironment__dataset_mgr
        with tempfile.TemporaryDirectory() as directory:
            manager.results_filename = os.path.join(directory, "run.h5")
            experiment.prepare()
            experiment.run()
            # The completion checkpoint writes the results file itself.
            with h5py.File(manager.results_filename, "r") as f:
                self.assertTrue(f.attrs["artiq_in_progress"])

    def test_repeated_children_resume_stream_and_share_checkpoints(self):
        fragment = self.create(RepeatedCameraFragment, [])
        manager = fragment._HasEnvironment__dataset_mgr
        with tempfile.TemporaryDirectory() as directory:
            manager.results_filename = os.path.join(directory, "run.h5")
            request = ScanRequest.explicit(
                [fragment.outer], [[1.0], [3.0]],
                execution_policy=ExecutionPolicy(
                    max_points_per_batch=1,
                    checkpoint_policy=CheckpointPolicy(min_interval_s=999999),
                ),
            )
            scan = PreparedScan(fragment, fragment, request)
            scan.execute()
            self.assertEqual(scan.inspect().values[fragment.total], [3, 7])
            self.assertEqual(sorted(os.listdir(directory)), ["run.h5", "run.streams.h5"])
            with read_scan_site_snapshot(manager.results_filename) as snapshot:
                self.assertTrue(snapshot.in_progress)
                child = snapshot.get_site(("shots",))
                np.testing.assert_array_equal(child.series("camera/image")[:, 0, 0],
                                              [1, 2, 3, 4])
                self.assertEqual(child.num_points, 4)

    def test_archive_checkpoint_and_selective_read(self):
        fragment = self.create(CameraFragment, [])
        manager = fragment._HasEnvironment__dataset_mgr
        with tempfile.TemporaryDirectory() as directory:
            manager.results_filename = os.path.join(directory, "run.h5")
            request = ScanRequest.explicit(
                [fragment.value], [[1.0], [2.0], [3.0]],
                execution_policy=ExecutionPolicy(
                    max_points_per_batch=1,
                    checkpoint_policy=CheckpointPolicy(min_interval_s=0),
                ),
            )
            scan = PreparedScan(fragment, fragment, request)
            scan.execute()
            result = scan.inspect()
            self.assertNotIn(fragment.image, result.values)
            self.assertEqual(result.values[fragment.counts], [1.0, 2.0, 3.0])
            prefix = result.site_prefix
            image_key = prefix + "points.channel_0"
            self.assertNotIn(image_key, self.dataset_db.data)
            self.assertNotIn(image_key, manager.local)

            # The results file is the checkpoint; it links to the stream file.
            checkpoint = manager.results_filename
            with h5py.File(checkpoint, "r") as f:
                image = f["datasets"][image_key]
                self.assertEqual(image.dtype, np.dtype("uint16"))
                self.assertEqual(image.shape, (3, 2, 3))
                self.assertEqual(image[2, 0, 0], 3)
                self.assertTrue(f["datasets"][prefix + "state.completed"][()])

            # Prove the reader skips payload I/O, rather than loading then filtering.
            original_getitem = h5py.Dataset.__getitem__
            def checked_read(dataset, selection):
                self.assertNotEqual(dataset.name, "/datasets/" + image_key)
                return original_getitem(dataset, selection)
            with patch.object(h5py.Dataset, "__getitem__", checked_read):
                snapshot = read_scan_site_snapshot(
                    checkpoint, channel_filter=lambda path: path != "image"
                )
            site = snapshot.get_site(())
            self.assertIn("channel_0", site.channels)
            self.assertNotIn("channel_0", site.raw_points)
            np.testing.assert_array_equal(site.series("counts"), [1, 2, 3])
            snapshot.close()

    def test_checkpoint_failure_propagates_and_previous_data_recovers(self):
        fragment = self.create(CameraFragment, [])
        manager = fragment._HasEnvironment__dataset_mgr
        with tempfile.TemporaryDirectory() as directory:
            manager.results_filename = os.path.join(directory, "run.h5")
            original_checkpoint = fragment.checkpoint_datasets
            calls = 0
            def fail_second_checkpoint():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("disk unavailable")
                original_checkpoint()
            request = ScanRequest.explicit(
                [fragment.value], [[1.0], [2.0]],
                execution_policy=ExecutionPolicy(
                    max_points_per_batch=1,
                    checkpoint_policy=CheckpointPolicy(min_interval_s=0),
                ),
            )
            with patch.object(fragment, "checkpoint_datasets", fail_second_checkpoint):
                with self.assertRaisesRegex(OSError, "disk unavailable"):
                    PreparedScan(fragment, fragment, request).execute()
            # The results file holds the last successful checkpoint.
            with h5py.File(manager.results_filename, "r") as f:
                self.assertEqual(f["datasets/ndscan.rid_0.site.root.state.num_points"][()], 1)
                self.assertEqual(f["datasets"][
                    "ndscan.rid_0.site.root.points.channel_0"][0, 0, 0], 1)

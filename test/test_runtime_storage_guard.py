"""A checkpoint that finds too little disk space ends the run cleanly."""

from collections import namedtuple
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from artiq.language.core import TerminationRequested

from ndscan.runtime import context
from ndscan.runtime.context import (
    PreviewCoordinator,
    raise_storage_stop,
    storage_stop_requested,
)
from ndscan.scan.request import CheckpointPolicy

Usage = namedtuple("Usage", "total used free")
GB = 1e9


class _DatasetManager:
    def __init__(self, results_filename, stream_buffer_bytes=0):
        self.results_filename = results_filename
        self.stream_buffer_bytes = stream_buffer_bytes


class _Owner:
    def __init__(self, results_filename, stream_buffer_bytes=0):
        self._HasEnvironment__dataset_mgr = _DatasetManager(
            results_filename, stream_buffer_bytes)
        self.checkpoints = 0

    def checkpoint_datasets(self):
        self.checkpoints += 1


class StorageGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = str(Path(self.tmp.name) / "run.h5")

    def tearDown(self):
        self.tmp.cleanup()
        context._storage_stop_reason = None

    def test_buffered_stream_data_counts(self):
        coordinator = PreviewCoordinator(
            _Owner(self.results, stream_buffer_bytes=5 * GB),
            CheckpointPolicy(min_interval_s=0), run_start_unix_time=0.0,
        )
        usage = Usage(total=1000 * GB, used=0, free=24 * GB)
        with patch("ndscan.runtime.context.shutil.disk_usage", return_value=usage):
            with self.assertLogs("ndscan.runtime.context", "ERROR"):
                coordinator.maybe_write_preview()
        self.assertTrue(storage_stop_requested())

    def _checkpoint(self, free, policy=None):
        policy = policy or CheckpointPolicy(min_interval_s=0)
        coordinator = PreviewCoordinator(
            _Owner(self.results), policy, run_start_unix_time=0.0
        )
        usage = Usage(total=1000 * GB, used=0, free=free)
        with patch("ndscan.runtime.context.shutil.disk_usage", return_value=usage):
            coordinator.maybe_write_preview()
        return coordinator

    def test_enough_space_keeps_running(self):
        self._checkpoint(free=21 * GB)
        self.assertFalse(storage_stop_requested())

    def test_fraction_sets_reserve_on_large_disks(self):
        # 2% of 5000 GB = 100 GB exceeds the 20 GB floor.
        policy = CheckpointPolicy(min_interval_s=0)
        coordinator = PreviewCoordinator(
            _Owner(self.results), policy, run_start_unix_time=0.0
        )
        usage = Usage(total=5000 * GB, used=0, free=99 * GB)
        with patch("ndscan.runtime.context.shutil.disk_usage", return_value=usage):
            with self.assertLogs("ndscan.runtime.context", "ERROR"):
                coordinator.maybe_write_preview()
        self.assertTrue(storage_stop_requested())

    def test_low_space_requests_clean_stop(self):
        # Reserve is max(20 GB, 2% of 1000 GB) = 20 GB.
        with self.assertLogs("ndscan.runtime.context", "ERROR"):
            self._checkpoint(free=19 * GB)
        self.assertTrue(storage_stop_requested())
        with self.assertRaises(TerminationRequested):
            raise_storage_stop()

    def test_reserve_is_configurable(self):
        policy = CheckpointPolicy(
            min_interval_s=0, min_free_bytes=1 * GB, min_free_fraction=0.0
        )
        self._checkpoint(free=2 * GB, policy=policy)
        self.assertFalse(storage_stop_requested())

    def test_new_run_clears_flag(self):
        with self.assertLogs("ndscan.runtime.context", "ERROR"):
            self._checkpoint(free=1 * GB)
        self.assertTrue(storage_stop_requested())
        self._checkpoint(free=500 * GB)
        self.assertFalse(storage_stop_requested())

    def test_policy_validation(self):
        with self.assertRaises(ValueError):
            CheckpointPolicy(min_free_fraction=1.5)
        with self.assertRaises(ValueError):
            CheckpointPolicy(min_free_bytes=-1)


if __name__ == "__main__":
    unittest.main()

"""Results whose streams ARTIQ finalised into ``<results>.<group>.h5`` files."""

import json
from pathlib import Path
import tempfile
import unittest
import warnings

import h5py
import numpy as np

from ndscan.results import open_scan_site_snapshot, read_scan_site_snapshot

IMAGE_KEY = "scan.points.channel_0"
TRACE_KEY = "scan.points.channel_2"
DATA = "000000007-Camera.h5"


def _write_site(group, num_points):
    group["scan.site.path"] = json.dumps([])
    group["scan.site.fragment_fqn"] = "test.Camera"
    group["scan.state.num_points"] = num_points
    group["scan.scan.channels"] = json.dumps({
        "channel_0": {
            "path": "camera/image", "type": "array", "element_type": "int",
            "shape": [5, 6], "dim_names": ["y", "x"],
        },
        "channel_1": {"path": "camera/counts", "type": "float"},
        "channel_2": {
            "path": "scope/trace", "type": "array", "element_type": "float",
            "shape": [3], "dim_names": ["t"],
        },
    })
    group["scan.points.channel_1"] = np.arange(num_points, dtype=float)


def _write_split(directory, images, traces, *, group_rid=7, num_points=None):
    num_points = len(images) if num_points is None else num_points
    with h5py.File(directory / DATA, "w") as file:
        file["rid"] = 7
        _write_site(file.create_group("datasets"), num_points)
        file.attrs["artiq_streams"] = json.dumps({
            "images": {"file": "000000007-Camera.images.h5",
                       "streams": {IMAGE_KEY: len(images)}},
            "streams": {"file": "000000007-Camera.streams.h5",
                        "streams": {TRACE_KEY: len(traces)}},
        })
    for name, key, values in (("images", IMAGE_KEY, images),
                              ("streams", TRACE_KEY, traces)):
        with h5py.File(directory / f"000000007-Camera.{name}.h5", "w") as file:
            file["rid"] = group_rid
            file.create_group("datasets")[key] = values
    return directory / DATA


class StreamGroupResultsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.images = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
        self.traces = np.arange(12, dtype=float).reshape(4, 3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_eager_and_lazy_read_group_files(self):
        data = _write_split(self.directory, self.images, self.traces)
        for reader in (read_scan_site_snapshot, open_scan_site_snapshot):
            with self.subTest(reader=reader.__name__), reader(data) as snapshot:
                site = snapshot.get_site()
                np.testing.assert_array_equal(
                    site.series("camera/counts"), np.arange(4.0))
                stream = site.series("camera/image")
                # Stream channels stay on disk even for an eager snapshot.
                self.assertNotIsInstance(stream, np.ndarray)
                np.testing.assert_array_equal(stream[1], self.images[1])
                np.testing.assert_array_equal(
                    site.series("scope/trace")[:], self.traces)
                self.assertEqual(set(snapshot.stream_files), {"images", "streams"})

    def test_missing_group_file_keeps_data_usable(self):
        data = _write_split(self.directory, self.images, self.traces)
        (self.directory / "000000007-Camera.images.h5").unlink()
        snapshot = read_scan_site_snapshot(data)
        site = snapshot.get_site()
        np.testing.assert_array_equal(site.series("camera/counts"), np.arange(4.0))
        np.testing.assert_array_equal(site.series("scope/trace")[:], self.traces)
        stream = site.series("camera/image")
        self.assertEqual(stream.shape, (4, 5, 6))
        with self.assertRaisesRegex(FileNotFoundError, "images.h5"):
            stream[0]
        snapshot.close()

    def test_channel_filter_skips_streams(self):
        data = _write_split(self.directory, self.images, self.traces)
        snapshot = read_scan_site_snapshot(
            data, channel_filter=lambda path: path != "camera/image")
        with self.assertRaises(KeyError):
            snapshot.get_site().series("camera/image")
        snapshot.close()

    def test_mismatched_run_is_rejected(self):
        data = _write_split(self.directory, self.images, self.traces, group_rid=8)
        with self.assertRaisesRegex(ValueError, "different run"):
            read_scan_site_snapshot(data)

    def test_row_count_mismatch_warns(self):
        data = _write_split(self.directory, self.images, self.traces, num_points=5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            read_scan_site_snapshot(data).close()
        self.assertTrue(any("may not align" in str(w.message) for w in caught))

    def test_checkpoint_with_deleted_segments_fails_loudly(self):
        segment = self.directory / "000000000.h5"
        with h5py.File(segment, "w") as file:
            file.create_group("datasets")[IMAGE_KEY] = self.images
        checkpoints = self.directory / "checkpoints"
        checkpoints.mkdir()
        checkpoint = checkpoints / "000000000.h5"
        with h5py.File(checkpoint, "w", libver="latest") as file:
            group = file.create_group("datasets")
            _write_site(group, 4)
            layout = h5py.VirtualLayout(self.images.shape, dtype=np.uint16)
            layout[:] = h5py.VirtualSource(
                "../000000000.h5", "datasets/" + IMAGE_KEY, shape=self.images.shape)
            group.create_virtual_dataset(IMAGE_KEY, layout)
            file.attrs["artiq_checkpoint"] = json.dumps({
                "version": 1, "streams": [IMAGE_KEY],
                "segments": [{"path": "../000000000.h5",
                              "size": segment.stat().st_size}],
            })
        snapshot = open_scan_site_snapshot(checkpoint)
        stream = snapshot.get_site().series("camera/image")
        np.testing.assert_array_equal(stream[0], self.images[0])
        segment.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "finalised"):
            stream[0]
        snapshot.close()
        with self.assertRaisesRegex(FileNotFoundError, "finalised"):
            read_scan_site_snapshot(checkpoint)


if __name__ == "__main__":
    unittest.main()

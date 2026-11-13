"""Lazy access must not read an acquisition just to discover its channels."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from ndscan.results import open_scan_site_snapshot, read_scan_site_snapshot


class LazySnapshotTest(unittest.TestCase):
    def test_metadata_slices_and_file_lifetime(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            images = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
            with h5py.File(directory / 'segment.h5', 'w') as file:
                file['images'] = images
            checkpoint = directory / 'checkpoint.h5'
            with h5py.File(checkpoint, 'w', libver='latest') as file:
                group = file.create_group('datasets')
                group['scan.site.path'] = json.dumps([])
                group['scan.site.fragment_fqn'] = 'test.Camera'
                group['scan.state.num_points'] = 4
                group['scan.scan.channels'] = json.dumps({'channel_0': {
                    'path': 'camera/image', 'type': 'array', 'element_type': 'int',
                    'shape': [5, 6], 'dim_names': ['y', 'x'],
                }})
                layout = h5py.VirtualLayout(images.shape, dtype=images.dtype)
                layout[:] = h5py.VirtualSource('segment.h5', 'images', shape=images.shape)
                group.create_virtual_dataset('scan.points.channel_0', layout)

            # Opening and inspecting shape/dtype/descriptions must read no points.
            original_read = h5py.Dataset.__getitem__
            def metadata_only(dataset, selection):
                if '.points.' in dataset.name:
                    raise AssertionError('Unexpected point payload read')
                return original_read(dataset, selection)
            with patch.object(h5py.Dataset, '__getitem__', metadata_only):
                snapshot = open_scan_site_snapshot(checkpoint)
                site = snapshot.get_site(())
                self.assertEqual(snapshot.format_site_tree(), '<root>')
                self.assertIn('camera/image', site.available_series_paths())
                stream = site.series('camera/image')
                self.assertEqual(stream.shape, images.shape)
                self.assertEqual(stream.dtype, images.dtype)
                self.assertEqual(stream.nbytes, images.nbytes)
                self.assertEqual(site.describe_series()[0].shape, images.shape)

            # Selections are passed to HDF5, including across image dimensions.
            with snapshot:
                one = stream[2]
                roi = stream[:, 1:3, 2:5]
                np.testing.assert_array_equal(one, images[2])
                np.testing.assert_array_equal(roi, images[:, 1:3, 2:5])
            snapshot.close()  # Explicit closure is also safe and idempotent.
            np.testing.assert_array_equal(one, images[2])
            with self.assertRaisesRegex(ValueError, 'Snapshot is closed'):
                stream[0]

            eager = read_scan_site_snapshot(checkpoint)
            self.assertIsInstance(eager.get_site().series('camera/image'), np.ndarray)
            np.testing.assert_array_equal(eager.get_site().series('camera/image'), images)

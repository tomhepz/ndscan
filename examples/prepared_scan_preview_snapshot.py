"""Prepared-runtime preview snapshot example.

This example is deliberately slow so the preview HDF5 path is easy to observe while
the experiment is still running.

It uses:

- one ordinary scanned parameter,
- one result channel,
- a point body that sleeps on the host,
- `PreviewPolicy` to rewrite a separate preview HDF5 file every two seconds, but only
  on completed batch boundaries.

The preview file is written relative to ARTIQ's worker result directory, so it will
appear alongside the final experiment HDF5 file. With the default naming policy it
matches ARTIQ's RID/class-name pattern and inserts `.preview` before `.h5`.

Notes:

- the stable preview file will look like `000002484-SlowPreviewFragment.preview.h5`
- the temporary file `000002484-SlowPreviewFragment.preview.h5.tmp` is expected to be
  short-lived because the write is atomic via `os.replace(...)`
- if you want to watch it live, a fast `watch` or `inotifywait` loop is more reliable
  than visually refreshing a file browser
- for example:
  `inotifywait -m -e create -e close_write -e moved_to -e delete results/*/*`

For this demo specifically, the preview file is kept after completion so it is easy to
inspect. The default prepared-runtime policy removes successful preview files at the end
of a run to avoid doubling disk usage.
"""

from __future__ import annotations

import time

from artiq.experiment import *
from ndscan.define import *
from ndscan.define import annotations
from ndscan.scan import *
from ndscan.runtime.api import *


class SlowPreviewFragment(ExpFragment):
    """Small fragment with an intentionally slow host-side point body."""

    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", default=0.0)
        self.setattr_result("y", FloatChannel)

    def run_once(self):
        # The sleep is here only so batch boundaries are easy to observe from outside
        # the worker process while the preview file is being rewritten.
        time.sleep(0.5)
        self.y.push(self.x.get() ** 2)


PreparedScanPreviewSnapshot = make_fragment_prepared_scan_exp(
    SlowPreviewFragment,
    lambda fragment: ScanRequest.cartesian(
        [(fragment.x, [0.5 * i for i in range(24)])],
        execution_policy=ExecutionPolicy(
            max_points_per_batch=2,
            preview_policy=PreviewPolicy(
                min_interval_s=2.0,
                write_on_completion=True,
                remove_on_completion=False,
            ),
        ),
        metadata={"demo_name": "prepared_scan_preview_snapshot"},
    ),
)

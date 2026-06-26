"""Host-runtime example showing multiple child segments for one parent point.

The important behaviour in this example is:

- the root scan picks one outer point,
- that outer point executes the same child scan site twice,
- each child execution becomes its own segment,
- both child segments point back to the same parent point index.

This is the simplest example of why ``segments_for_parent_point(...)`` returns a list
instead of a single segment.

Important caveat:

- reusing one child site this way is only safe when each child execution has the same
  logical schema,
- for example, the same scanned parameter(s) and the same result channels,
- because a site path has one scalar metadata/schema record and one append-only point
  stream,
- so later child executions overwrite the site's scalar metadata while still appending
  into the same flat point datasets.

If two child executions really scan different things, they should usually be modelled
as different child scan sites instead.
"""

from __future__ import annotations

import numpy as np
from artiq.experiment import *

from ndscan.define import *
from ndscan.runtime.api import *
from ndscan.scan import *


class SegmentLeafFragment(ExpFragment):
    """Leaf fragment with one scanned input and one simple output."""

    def build_fragment(self):
        self.setattr_param("x", FloatParam, "x", 0.0)
        self.setattr_result("y", FloatChannel, description="y = x + 1")

    def run_once(self):
        self.y.push(self.x.get() + 1.0)

    def get_default_analyses(self):
        # Save one tiny per-segment summary so the offline script can inspect it.
        return [
            CustomAnalysis(
                [self.x],
                self._analyse_segment,
                [
                    FloatChannel("segment_total", "Sum of y over this segment"),
                    IntChannel("num_points", "Number of points in this segment"),
                ],
            )
        ]

    def _analyse_segment(self, axis_values, result_values, analysis_results):
        del axis_values, analysis_results

        ys = np.asarray(result_values[self.y], dtype=float)
        return AnalysisFeedback(
            outputs={
                "segment_total": float(np.sum(ys)),
                "num_points": int(len(ys)),
            }
        )


class ReusedChildSegmentsParentFragment(ExpFragment):
    """Root fragment that launches the same child site twice per outer point."""

    def build_fragment(self):
        self.setattr_param("outer", FloatParam, "outer", 0.0)
        self.child_scan = setattr_prepared_child_scan(
            self,
            "child",
            SegmentLeafFragment,
            scan_name="child_scan",
            extra_metadata={
                "viewer_note": "same child site is executed twice for each parent point"
            },
            expose_outputs=["segment_total", "num_points"],
        )
        self.setattr_result("first_total", FloatChannel)
        self.setattr_result("second_total", FloatChannel)
        self.setattr_result("combined_total", FloatChannel)

    def run_once(self):
        # First child execution for this outer point.
        first_request = ScanRequest.explicit(
            [self.child.x],
            [[self.outer.get()], [self.outer.get() + 1.0]],
            metadata={"phase": "first"},
        )
        self.child_scan.configure(first_request)
        first_outputs = self.child_scan.execute()

        # Second child execution for the same outer point.
        second_request = ScanRequest.explicit(
            [self.child.x],
            [[self.outer.get() + 10.0], [self.outer.get() + 11.0]],
            metadata={"phase": "second"},
        )
        self.child_scan.configure(second_request)
        second_outputs = self.child_scan.execute()

        first_total = float(first_outputs["segment_total"])
        second_total = float(second_outputs["segment_total"])

        self.first_total.push(first_total)
        self.second_total.push(second_total)
        self.combined_total.push(first_total + second_total)


HostRuntimeReusedChildSegments = make_fragment_prepared_scan_exp(
    ReusedChildSegmentsParentFragment,
    lambda fragment: ScanRequest.explicit(
        [fragment.outer],
        [[10.0], [20.0], [30.0]],
        metadata={"demo_name": "host_runtime_reused_child_segments"},
    ),
)

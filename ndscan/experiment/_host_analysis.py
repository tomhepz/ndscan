"""Internal analysis helpers for the host runtime.

The new host runtime already had a clean execution loop, but the analysis-related
logic had grown into a sizeable cluster inside ``host_runtime.py``:

- selecting the applicable default analyses,
- building metadata for online/final analysis outputs,
- binding temporary result sinks for analysis execution,
- publishing online analysis snapshots after each completed batch,
- publishing final analysis outputs at the end of the run.

This module keeps that logic together without changing the fragment-facing analysis
API. The runtime still owns batch boundaries and dataset publication; this helper
only answers:

- which analyses apply to this concrete scan,
- what metadata they imply,
- and what outputs/annotations they produce from accumulated run data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Any

from ..utils import merge_no_duplicates
from .annotations import AnnotationContext
from .default_analysis import AnalysisFeedback
from .fragment import ExpFragment
from .result_channels import LastValueSink, ResultChannel
from .scan_runner import describe_analyses, filter_default_analyses


class _TemporaryAnalysisResultSinks:
    """Temporarily bind analysis result channels to in-memory last-value sinks.

    Default analyses are declared in terms of ordinary ``ResultChannel`` instances.
    Running them through temporary ``LastValueSink`` objects keeps analysis execution
    separate from dataset publication: analyses push to channels exactly as they would
    in the legacy runtime, while the host runtime decides afterwards which values
    become final results or online feedback.
    """

    def __init__(self, channels: Mapping[str, ResultChannel]):
        self._channels = dict(channels)
        self._original_sinks = dict[ResultChannel, Any]()
        self._temporary_sinks = dict[str, LastValueSink]()

    def __enter__(self) -> dict[str, LastValueSink]:
        for name, channel in self._channels.items():
            self._original_sinks[channel] = channel.sink
            sink = LastValueSink()
            channel.set_sink(sink)
            self._temporary_sinks[name] = sink
        return self._temporary_sinks

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for channel, original_sink in self._original_sinks.items():
            channel.set_sink(original_sink)
        self._original_sinks.clear()
        self._temporary_sinks.clear()


class HostScanAnalysisEngine:
    """Selected default analyses for one concrete host-runtime scan program.

    The fragment-side analysis API naturally splits into two phases:

    - declaration/description, which determines scan-site metadata before the scan
      starts,
    - execution, which consumes accumulated run data either after each completed batch
      (online) or after the final point (final analysis).

    This helper keeps those phases together without mixing them into the main point
    execution loop in ``host_runtime.py``.
    """

    def __init__(
        self,
        analyses,
        analysis_results: Mapping[str, ResultChannel],
        annotation_context: AnnotationContext,
    ):
        self._analyses = tuple(analyses)
        self._analysis_results = dict(analysis_results)
        self._annotation_context = annotation_context
        self._metadata = describe_analyses(self._analyses, self._annotation_context)
        self._metadata["analysis_results"] = {
            name: channel.describe() for name, channel in self._analysis_results.items()
        }

    @classmethod
    def build(
        cls,
        fragment: ExpFragment,
        axes: Sequence[Any],
        channels: Sequence[Any],
    ) -> "HostScanAnalysisEngine":
        """Build an analysis engine for one concrete fragment/request pair.

        ``axes`` and ``channels`` are the host-runtime bound axis/channel objects. This
        module keeps the dependency surface narrow by only relying on the attributes it
        needs rather than importing those runtime-only dataclasses directly.
        """

        analysable_axes = [axis for axis in axes if axis.param_store is not None]
        analyses = filter_default_analyses(fragment, analysable_axes)

        axis_keys = {
            axis.param_store.identity: axis.point_key
            for axis in axes
            if axis.param_store is not None
        }
        # AnnotationContext expects bare channel names and adds the "channel_" prefix
        # itself when serialising coordinate references.
        channel_names = {
            binding.channel: binding.key.removeprefix("channel_") for binding in channels
        }
        analysis_results = reduce(
            lambda x, y: merge_no_duplicates(x, y, kind="analysis result"),
            (analysis.get_analysis_results() for analysis in analyses),
            {},
        )
        exported_analysis_channels = set(analysis_results.values())

        context = AnnotationContext(
            lambda handle: axis_keys[handle._store.identity],
            lambda channel: channel_names[channel],
            lambda channel: channel in exported_analysis_channels,
        )
        return cls(analyses, analysis_results, context)

    def metadata(self) -> dict[str, Any]:
        metadata = {}
        if self._metadata["annotations"]:
            metadata["analysis.annotations"] = list(self._metadata["annotations"])
        if self._metadata["online_analyses"]:
            metadata["analysis.online"] = dict(self._metadata["online_analyses"])
        if self._metadata["analysis_results"]:
            metadata["analysis.outputs"] = dict(self._metadata["analysis_results"])
        return metadata

    def initial_annotations(self) -> list[dict[str, Any]]:
        return list(self._metadata["annotations"])

    def execute_final(self, run_result, site_writer) -> None:
        """Execute final analyses against the completed run result."""

        if not self._analyses:
            return

        axis_data = dict(run_result.coordinates)
        result_data = dict(run_result.values)

        with _TemporaryAnalysisResultSinks(self._analysis_results) as sinks:
            annotations = []
            for analysis in self._analyses:
                annotations.extend(
                    analysis.execute(axis_data, result_data, self._annotation_context)
                )
            analysis_results = {name: sink.get_last() for name, sink in sinks.items()}

        feedback = AnalysisFeedback(
            outputs=analysis_results,
            annotations=annotations,
        )

        for name, value in feedback.outputs.items():
            site_writer.set_analysis_result(name, value)
        run_result.analysis_results = dict(feedback.outputs)

        if feedback.annotations:
            site_writer.set_annotations(feedback.annotations)
            run_result.annotations = list(feedback.annotations)

    def observe_batch(
        self,
        observations,
        run_result,
        site_writer,
    ) -> dict[str, AnalysisFeedback]:
        """Run online analyses on the accumulated data after one completed batch."""

        del observations

        if not self._analyses:
            return {}

        axis_data = dict(run_result.coordinates)
        result_data = dict(run_result.values)
        raw_online_results = reduce(
            lambda x, y: merge_no_duplicates(x, y, kind="online analysis result"),
            (
                analysis.execute_online(axis_data, result_data, self._annotation_context)
                for analysis in self._analyses
            ),
            {},
        )
        online_results = {
            name: self._normalise_online_feedback(value)
            for name, value in raw_online_results.items()
        }
        for name, feedback in online_results.items():
            site_writer.set_online_analysis_result(name, feedback.outputs)
            site_writer.set_online_analysis_annotations(name, feedback.annotations)
        run_result.online_analysis_results = {
            name: feedback.outputs for name, feedback in online_results.items()
        }
        run_result.online_analysis_annotations = {
            name: list(feedback.annotations) for name, feedback in online_results.items()
        }
        return online_results

    def _normalise_online_feedback(
        self, value: AnalysisFeedback | dict[str, Any]
    ) -> AnalysisFeedback:
        """Return a structured online-analysis payload.

        Older online analyses returned only a dict of outputs. Newer code can return an
        ``AnalysisFeedback`` directly so outputs and annotations travel through the
        runtime together.
        """

        if isinstance(value, AnalysisFeedback):
            return value
        return AnalysisFeedback(outputs=dict(value))

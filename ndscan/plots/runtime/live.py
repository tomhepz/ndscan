"""Build a runtime site-tree snapshot from a live applet dataset view."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...results.scan_site_reader import (
    HostRuntimeSegmentFinalAnalysis,
    HostRuntimeSiteData,
    HostRuntimeSnapshot,
)

_STRUCTURED_KEYS = {
    "site.path",
    "site.parent_path",
    "scan.point_policy",
    "scan.pseudoparams",
    "scan.fixed_pseudoparams",
    "scan.parameters",
    "scan.fixed_parameters",
    "scan.channels",
    "scan.parameter_mappings",
    "analysis.online",
    "analysis.outputs",
    "analysis.annotations",
}
_STRUCTURED_ARRAY_KEYS = {
    "segments.analysis.final_feedback",
}


def _decode_json_string_list(values: list[Any]) -> list[Any]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            decoded.append(json.loads(value))
        else:
            decoded.append(value)
    return decoded


def _decode_live_value(key: str, raw: Any) -> Any:
    """Convert raw applet dataset values into plain Python objects.

    Live applet updates can arrive as NumPy scalars/arrays, byte strings, or JSON-encoded
    structured payloads. The snapshot builder works with normal Python containers, so
    this function normalises the common ARTIQ transport forms in one place.
    """
    if isinstance(raw, np.ndarray):
        if raw.dtype.kind == "S":
            decoded = [item.decode("utf-8") for item in raw.tolist()]
            if any(key.endswith(suffix) for suffix in _STRUCTURED_ARRAY_KEYS):
                return _decode_json_string_list(decoded)
            return decoded
        if raw.dtype.kind == "U":
            decoded = raw.tolist()
            if any(key.endswith(suffix) for suffix in _STRUCTURED_ARRAY_KEYS):
                return _decode_json_string_list(decoded)
            return decoded
        if raw.dtype.kind == "O":
            decoded = []
            changed = False
            for item in raw.tolist():
                if isinstance(item, bytes):
                    decoded.append(item.decode("utf-8"))
                    changed = True
                elif isinstance(item, np.generic):
                    decoded.append(item.item())
                    changed = True
                else:
                    decoded.append(item)
            if any(key.endswith(suffix) for suffix in _STRUCTURED_ARRAY_KEYS):
                return _decode_json_string_list(decoded if changed else raw.tolist())
            return decoded if changed else raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, tuple):
        raw = list(raw)
    if isinstance(raw, str):
        if any(key.endswith(suffix) for suffix in _STRUCTURED_KEYS):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        if (
            key.startswith("analysis.online_result.")
            or key.startswith("analysis.artifact.")
            or key.startswith("analysis.online_artifact.")
            or key.startswith("analysis.online_annotation.")
            or ".analysis.online_result." in key
            or ".analysis.artifact." in key
            or ".analysis.online_artifact." in key
            or ".analysis.online_annotation." in key
        ):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
    return raw


def _find_site_prefixes(dataset_values: dict[str, Any]) -> list[str]:
    """Return site prefixes in parent-before-child order."""
    suffix = "site.path"
    prefixes = [
        key[: -len(suffix)]
        for key in dataset_values.keys()
        if key.endswith(suffix)
    ]
    return sorted(prefixes, key=lambda prefix: (prefix.count("."), prefix))


def _point_keys_for_prefix(dataset_values: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Extract one site's point streams from the live dataset mapping."""
    point_prefix = prefix + "points."
    return {
        key[len(point_prefix) :]: value
        for key, value in dataset_values.items()
        if key.startswith(point_prefix)
    }


def _analysis_outputs_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> dict[str, Any]:
    """Extract final analysis outputs for one site prefix."""
    analysis_prefix = prefix + "analysis.output."
    return {
        key[len(analysis_prefix) :]: value
        for key, value in dataset_values.items()
        if key.startswith(analysis_prefix)
    }


def _online_results_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> dict[str, Any]:
    """Extract live online-analysis scalar outputs for one site prefix."""
    analysis_prefix = prefix + "analysis.online_result."
    return {
        key[len(analysis_prefix) :]: value
        for key, value in dataset_values.items()
        if key.startswith(analysis_prefix)
    }


def _analysis_artifacts_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> dict[str, Any]:
    """Extract final analysis artifacts for one site prefix."""
    analysis_prefix = prefix + "analysis.artifact."
    return {
        key[len(analysis_prefix) :]: value
        for key, value in dataset_values.items()
        if key.startswith(analysis_prefix)
    }


def _online_artifacts_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> dict[str, Any]:
    """Extract live online-analysis artifacts for one site prefix."""
    analysis_prefix = prefix + "analysis.online_artifact."
    return {
        key[len(analysis_prefix) :]: value
        for key, value in dataset_values.items()
        if key.startswith(analysis_prefix)
    }


def _online_annotations_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> dict[str, list[dict[str, Any]]]:
    """Extract live online-analysis annotations for one site prefix."""
    analysis_prefix = prefix + "analysis.online_annotation."
    return {
        key[len(analysis_prefix) :]: value
        for key, value in dataset_values.items()
        if key.startswith(analysis_prefix)
    }


def _segment_final_analysis_for_prefix(
    dataset_values: dict[str, Any], prefix: str
) -> list[HostRuntimeSegmentFinalAnalysis]:
    raw_feedback = dataset_values.get(prefix + "segments.analysis.final_feedback", [])
    return [HostRuntimeSegmentFinalAnalysis.from_dict(item) for item in raw_feedback]


def snapshot_from_live_values(
    prefix: str, values: dict[str, Any]
) -> HostRuntimeSnapshot:
    """Reconstruct a runtime site-tree snapshot from a live applet dataset view.

    The applet receives a flat dataset mapping keyed by full dataset name. This function
    filters that mapping to one prepared-runtime subtree, decodes the structured payloads,
    and rebuilds the same site-tree shape used by the offline results reader.
    """
    datasets = {
        key[len(prefix) :]: _decode_live_value(key[len(prefix) :], value)
        for key, value in values.items()
        if key.startswith(prefix)
    }

    sites = {}
    for site_prefix in _find_site_prefixes(datasets):
        path_value = tuple(datasets[site_prefix + "site.path"])
        parent_path = datasets.get(site_prefix + "site.parent_path")
        sites[path_value] = HostRuntimeSiteData(
            prefix=prefix + site_prefix,
            path=path_value,
            parent_path=None if parent_path is None else tuple(parent_path),
            fragment_fqn=datasets.get(site_prefix + "site.fragment_fqn", "<unknown>"),
            pseudoparams=datasets.get(site_prefix + "scan.pseudoparams", {}),
            fixed_pseudoparams=datasets.get(site_prefix + "scan.fixed_pseudoparams", {}),
            parameters=datasets.get(site_prefix + "scan.parameters", {}),
            fixed_parameters=datasets.get(site_prefix + "scan.fixed_parameters", {}),
            channels=datasets.get(site_prefix + "scan.channels", {}),
            point_data=_point_keys_for_prefix(datasets, site_prefix),
            analysis_outputs_schema=datasets.get(site_prefix + "analysis.outputs", {}),
            analysis_outputs=_analysis_outputs_for_prefix(datasets, site_prefix),
            analysis_artifacts=_analysis_artifacts_for_prefix(datasets, site_prefix),
            online_analysis_schema=datasets.get(site_prefix + "analysis.online", {}),
            online_analysis_results=_online_results_for_prefix(datasets, site_prefix),
            online_analysis_artifacts=_online_artifacts_for_prefix(
                datasets, site_prefix
            ),
            online_analysis_annotations=_online_annotations_for_prefix(
                datasets, site_prefix
            ),
            annotations=datasets.get(site_prefix + "analysis.annotations", []),
            segment_final_analysis=_segment_final_analysis_for_prefix(
                datasets, site_prefix
            ),
            segmented=(site_prefix + "segments.start_index") in datasets,
            metadata={
                key[len(site_prefix) :]: value
                for key, value in datasets.items()
                if key.startswith(site_prefix)
                and not key.startswith(site_prefix + "points.")
                and not key.startswith(site_prefix + "analysis.output.")
                and not key.startswith(site_prefix + "analysis.artifact.")
                and not key.startswith(site_prefix + "analysis.online_result.")
                and not key.startswith(site_prefix + "analysis.online_artifact.")
                and not key.startswith(site_prefix + "analysis.online_annotation.")
                and not key.startswith(site_prefix + "segments.analysis.final_feedback")
            },
        )

    return HostRuntimeSnapshot(
        path=Path("<live>"),
        top_level_metadata={},
        sites=sites,
    )

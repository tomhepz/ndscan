"""Persisted scan-site schema shared by runtime and offline consumers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from artiq.language import HasEnvironment

__all__ = [
    "SCAN_SITE_SCHEMA_REVISION",
    "ScanSite",
    "make_scan_site_prefix",
]


# The legacy runtime still writes schema revision 2. The prepared-runtime scan-site
# schema has diverged enough that readers should be able to distinguish it explicitly.
SCAN_SITE_SCHEMA_REVISION = 4


@dataclass(frozen=True)
class ScanSite:
    """Description of a scan site's place in the dataset tree.

    ``path`` is structural, not cosmetic. A top-level scan lives at ``()`` and nested
    scan sites later use child paths like ``("cooling",)`` or ``("cooling", "probe")``.

    ``parent_path`` is optional metadata describing which other scan site this site is
    nested under. Child scan sites still have their own full ``path``; the parent path
    is kept separately so dashboard tools, matplotlib helpers, tests, and readers do not
    need to reverse-engineer hierarchy from dataset prefixes.
    """

    path: tuple[str, ...] = ()
    parent_path: tuple[str, ...] | None = None
    dataset_prefix: str | None = None
    segmented: bool = False
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


def _normalise_site_component(name: str) -> str:
    """Return a dataset-safe site path component."""

    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return cleaned or "unnamed"


def make_scan_site_prefix(owner: HasEnvironment, site: ScanSite) -> str:
    """Return the canonical dataset prefix for one scan site."""

    if site.dataset_prefix is not None:
        return (
            site.dataset_prefix
            if site.dataset_prefix.endswith(".")
            else site.dataset_prefix + "."
        )

    scheduler = owner.get_device("scheduler")
    rid = getattr(scheduler, "rid", 0)
    parts = ["ndscan", f"rid_{rid}", "site", "root"]
    parts.extend(_normalise_site_component(part) for part in site.path)
    return ".".join(parts) + "."

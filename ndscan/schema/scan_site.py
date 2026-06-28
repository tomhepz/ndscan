"""Persisted scan-site schema shared by runtime and offline consumers.

Runtime code, live plotting, and offline result readers all depend on the physical
dataset layout described here. Keep this module small and conservative: it should only
encode facts needed to locate a scan site, not higher-level reader conveniences.
"""

from __future__ import annotations

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
SCAN_SITE_SCHEMA_REVISION = 8


@dataclass(frozen=True)
class ScanSite:
    """Description of a scan site's place in the dataset tree.

    ``path`` is structural, not cosmetic. A top-level scan lives at ``()`` and nested
    scan sites later use child paths like ``("cooling",)`` or ``("cooling", "probe")``.

    ``parent_path`` is optional metadata describing which other scan site this site is
    nested under. Child scan sites still have their own full ``path``; the parent path
    is kept separately so dashboard tools, matplotlib helpers, tests, and readers do not
    need to reverse-engineer hierarchy from dataset prefixes.

    ``dataset_prefix`` is an explicit override for specialised callers. Normal runtime
    paths should leave it as ``None`` so all sites use the canonical ``site.root`` /
    ``subscans`` tree.
    """

    path: tuple[str, ...] = ()
    parent_path: tuple[str, ...] | None = None
    dataset_prefix: str | None = None
    segmented: bool = False
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


def _quote_site_path_component(component: str) -> str:
    """Return a readable dataset-key fragment for one logical site path component.

    ARTIQ dataset keys are dot-separated, so arbitrary child names cannot be inserted
    literally. ASCII letters, digits, and underscores remain readable; every other
    UTF-8 byte is escaped as ``~xx``.
    """

    result = []
    for byte in component.encode("utf-8"):
        char = chr(byte)
        if (
            "a" <= char <= "z"
            or "A" <= char <= "Z"
            or "0" <= char <= "9"
            or char == "_"
        ):
            result.append(char)
        else:
            result.append(f"~{byte:02x}")
    return "".join(result) or "~empty"


def _site_path_storage_parts(path: tuple[str, ...]) -> list[str]:
    """Return readable dataset-key components for a logical site path.

    Every child name sits below a reserved ``subscans`` component. This lets a child
    called ``points`` or ``segments`` remain human-readable without colliding with the
    parent site's own ``points.*`` or ``segments.*`` datasets.
    """

    if not path:
        return ["root"]

    parts = ["root"]
    for component in path:
        parts.extend(("subscans", _quote_site_path_component(component)))
    return parts


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
    parts = ["ndscan", f"rid_{rid}", "site"]
    parts.extend(_site_path_storage_parts(site.path))
    return ".".join(parts) + "."

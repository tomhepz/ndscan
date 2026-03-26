"""Submission-side request compilation helpers.

Import the concrete submodules directly:

- ``ndscan.submission.expression`` for the tiny safe expression language,
- ``ndscan.submission.host_scan_schema`` for typed host-scan schema compilation.

Keeping the package ``__init__`` intentionally light avoids pulling fragment/runtime
types into low-level imports such as ``ndscan.scan.mapping``.
"""

__all__: list[str] = []

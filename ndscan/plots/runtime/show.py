"""Standalone viewer for prepared-runtime HDF5 snapshots."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

from oitg import results
from qasync import QEventLoop

from ..._qt import QtWidgets
from ...results import ScanSiteSnapshot, read_scan_site_snapshot
from .viewer import RuntimePlotViewer


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display a prepared-runtime ndscan HDF5 file in the runtime viewer",
        epilog=(
            "For legacy ndscan HDF5 files that use axes/channels/points.channel_* "
            "datasets, use ndscan_show instead."
        ),
    )
    parser.add_argument(
        "path",
        metavar="FILE",
        type=str,
        help=(
            "Path to an ARTIQ HDF5 results file. oitg-style magic run ids are also "
            "accepted when available, matching ndscan_show."
        ),
    )
    return parser


def resolve_snapshot_path(path_or_magic: str) -> Path:
    """Resolve either a direct path or an oitg magic result identifier."""

    magic_spec = results.parse_magic(path_or_magic)
    if magic_spec is None:
        return Path(path_or_magic)

    paths = results.find_results(day="auto", **magic_spec)
    if len(paths) != 1:
        raise RuntimeError(
            f"Could not resolve {path_or_magic!r} to exactly one result file: {paths}"
        )
    return Path(next(iter(paths.values())).path)


def load_runtime_snapshot(path_or_magic: str) -> ScanSiteSnapshot:
    """Load and validate a prepared-runtime HDF5 snapshot for display."""

    path = resolve_snapshot_path(path_or_magic)
    snapshot = read_scan_site_snapshot(path)
    if not snapshot.sites:
        raise RuntimeError(
            f"No prepared-runtime scan sites found in {path}. If this is a legacy "
            "ndscan file, use ndscan_show instead."
        )
    if () not in snapshot.sites:
        raise RuntimeError(f"Prepared-runtime snapshot {path} has no root scan site")
    return snapshot


def _show_error(title: str, message: str) -> None:
    QtWidgets.QMessageBox.critical(None, title, message)


def main() -> None:
    args = get_argparser().parse_args()

    app = QtWidgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    try:
        snapshot = load_runtime_snapshot(args.path)
    except Exception as exc:
        _show_error("Unable to load prepared-runtime snapshot", str(exc))
        sys.exit(1)

    viewer = RuntimePlotViewer()
    viewer.set_snapshot(
        snapshot,
        status_text=f"Loaded prepared-runtime snapshot from {snapshot.path}",
    )
    if viewer.windowTitle():
        viewer.setWindowTitle(f"{viewer.windowTitle()} - {snapshot.path.name}")
    else:
        viewer.setWindowTitle(f"{snapshot.path.name} - ndscan runtime viewer")
    viewer.resize(1200, 700)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

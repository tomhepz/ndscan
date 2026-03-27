"""ARTIQ applet for prepared-runtime site-tree plots."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Any

import pyqtgraph
from artiq.applets.simple import SimpleApplet
from sipyco import common_args

from ._qt import QtWidgets
from .plots.runtime import RuntimePlotViewer


class _MainWidget(RuntimePlotViewer):
    def __init__(self, args, ctl):
        common_args.init_logger_from_args(args)
        super().__init__(args.prefix, ctl.set_dataset)
        self.resize(1600, 900)
        self.setWindowTitle("ndscan prepared-runtime plot")

    def close(self):
        QtWidgets.QWidget.close(self)

    def data_changed(
        self,
        values: dict[str, Any],
        metadata: dict[str, Any],
        persist: dict[str, bool],
        mods: Iterable[dict[str, Any]],
    ):
        super().data_changed(values, metadata, persist, mods)


class NdscanRuntimeApplet(SimpleApplet):
    def __init__(self):
        super().__init__(
            _MainWidget,
            default_update_delay=20e-3,
            cmd_description="Prepared-runtime ARTIQ applet for ndscan experiments",
        )
        self.argparser.add_argument(
            "--prefix", default=None, type=str, help="Root of the prepared-runtime dataset tree"
        )
        self.argparser.add_argument("--rid", default=None, help=argparse.SUPPRESS)
        common_args.verbosity_args(self.argparser)

    def args_init(self):
        super().args_init()
        if self.args.prefix is None:
            raise ValueError("prepared-runtime ndscan applets require an explicit --prefix")
        if self.args.rid is not None:
            raise ValueError("prepared-runtime ndscan applets use --prefix instead of --rid")
        if not hasattr(self, "dataset_prefixes"):
            raise RuntimeError(
                "Client-side ARTIQ version out of date; update dashboard to a "
                "version with dataset_prefixes support."
            )
        self.dataset_prefixes.append(self.args.prefix)


def main():
    pyqtgraph.setConfigOptions(antialias=True)
    NdscanRuntimeApplet().run()


if __name__ == "__main__":
    main()

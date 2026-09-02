"""Lightweight console-script entry points for optional ndscan capabilities."""


def dataset_janitor():
    """Run the ARTIQ dataset janitor from the runtime installation."""

    try:
        from .dataset_janitor import main
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing == "sipyco" or missing.startswith("artiq"):
            raise SystemExit(
                "ndscan_dataset_janitor requires ndscan[runtime]"
            ) from error
        raise
    return main()


def show():
    """Run the prepared-runtime result viewer from the GUI installation."""

    try:
        from .plots.runtime.show import main
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing in {
            "oitg",
            "pyqtgraph",
            "qasync",
            "matplotlib",
        } or missing.startswith(("PyQt", "PySide")):
            raise SystemExit("ndscan_show requires ndscan[gui]") from error
        raise
    return main()

"""Import-surface regression tests for ARTIQ experiment discovery."""

import subprocess
import sys
import textwrap
import unittest


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        text=True,
    )


class ExperimentImportTest(unittest.TestCase):
    def test_results_import_has_only_base_dependencies(self):
        result = _run_python(
            """
            from ndscan.results import read_scan_site_snapshot
            import sys
            print(any(name in sys.modules for name in (
                "artiq", "oitg", "pyqtgraph", "qasync", "torch"
            )))
            """
        )

        self.assertEqual(result.stdout.strip(), "False")

    def test_scan_wildcard_import_does_not_import_torch(self):
        result = _run_python(
            """
            from ndscan.scan import *
            import sys
            print("torch" in sys.modules)
            """
        )

        self.assertEqual(result.stdout.strip(), "False")

    def test_experiment_wildcard_import_does_not_import_torch(self):
        result = _run_python(
            """
            from ndscan.experiment import *
            import sys
            print("torch" in sys.modules)
            """
        )

        self.assertEqual(result.stdout.strip(), "False")

    def test_artiq_file_import_does_not_import_torch(self):
        result = _run_python(
            """
            import pathlib
            import sys
            import tempfile

            from artiq import tools
            from artiq.compiler.import_cache import install_hook

            with tempfile.TemporaryDirectory() as tmp_dir:
                path = pathlib.Path(tmp_dir) / "probe_experiment.py"
                path.write_text("from ndscan.experiment import *\\n", encoding="utf-8")

                install_hook()
                tools.file_import(str(path))

            print("torch" in sys.modules)
            """
        )

        self.assertEqual(result.stdout.strip(), "False")

    def test_runtime_viewer_import_does_not_import_torch(self):
        result = _run_python(
            """
            from ndscan.plots.runtime import RuntimePlotViewer
            import sys
            print("torch" in sys.modules)
            """
        )

        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()

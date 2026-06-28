"""Schema-compiled prepared-runtime example using pseudoparams plus a text rebind.

This is the schema-driven counterpart to ``prepared_scan_parameter_mapping.py``:

- ``logical_drive`` is a runtime-only pseudoparameter scanned directly,
- ``offset`` is a fixed pseudoparameter constant,
- the physical ``drive`` parameter is rebound from a small text expression.
"""

from ndscan.runtime.api import make_fragment_prepared_scan_exp
from ndscan.submission.scan_submission_schema import compile_scan_submission_schema

from prepared_scan_parameter_mapping import HardwareDriveFragment


PreparedScanMappedLogicalAxisSchema = make_fragment_prepared_scan_exp(
    HardwareDriveFragment,
    lambda fragment: compile_scan_submission_schema(
        fragment,
        {
            "version": 1,
            "mode": {"type": "grid"},
            "entries": [
                {
                    "id": "logical_drive",
                    "kind": "pseudoparam",
                    "description": "Logical scan axis mapped to the physical drive",
                    "mode": {
                        "type": "scan",
                        "generator": {
                            "type": "list",
                            "range": {
                                "values": [0.0, 1.0, 2.0, 3.0],
                                "randomise_order": False,
                            },
                        },
                    },
                },
                {
                    "id": "offset",
                    "kind": "pseudoparam",
                    "mode": {
                        "type": "fixed",
                        "value": 0.5,
                    },
                },
                {
                    "id": "drive",
                    "kind": "param",
                    "target": {
                        "fqn": fragment.drive.parameter.fqn,
                        "path": "*",
                    },
                    "mode": {
                        "type": "rebind",
                        "expr": "logical_drive + offset",
                    },
                },
            ],
            "metadata": {"demo_name": "prepared_scan_parameter_mapping_schema"},
        },
    ),
)

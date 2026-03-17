"""Dict-schema host-runtime example using pseudoparams plus a text rebind.

This is the schema-driven counterpart to ``host_runtime_parameter_mapping.py``:

- ``logical_drive`` is a runtime-only pseudoparameter scanned directly,
- ``offset`` is a fixed pseudoparameter constant,
- the physical ``drive`` parameter is rebound from a small text expression.
"""

from ndscan.experiment import make_fragment_host_scan_exp

from host_runtime_parameter_mapping import HardwareDriveFragment


HostRuntimeMappedLogicalAxisSchema = make_fragment_host_scan_exp(
    HardwareDriveFragment,
    lambda fragment: {
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
        "metadata": {"demo_name": "host_runtime_parameter_mapping_schema"},
    },
)

import unittest

from ndscan.dashboard.submission import (
    LegacyScanOptionsState,
    LegacySubmissionBackend,
    ScanSubmissionBackend,
    select_submission_backend,
)


class DashboardSubmissionBackendTest(unittest.TestCase):
    def test_select_submission_backend_prefers_legacy_scan_payload(self):
        backend = select_submission_backend({"scan": {}, "overrides": {}})
        self.assertIsInstance(backend, LegacySubmissionBackend)
        self.assertTrue(backend.supports_editing)

    def test_select_submission_backend_detects_scan_submission_payload(self):
        backend = select_submission_backend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        self.assertIsInstance(backend, ScanSubmissionBackend)
        self.assertTrue(backend.supports_editing)

    def test_scan_submission_backend_rejects_unsupported_scan_submission_payload(self):
        backend = select_submission_backend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [
                        {
                            "id": "x",
                            "kind": "param",
                            "target": {"fqn": "frag.x", "path": "*"},
                            "mode": {
                                "type": "scan",
                                "generator": {
                                    "type": "refining",
                                    "range": {"lower": 0.0, "upper": 1.0},
                                },
                            },
                        }
                    ],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        self.assertIsInstance(backend, ScanSubmissionBackend)
        self.assertFalse(backend.supports_editing)

    def test_scan_submission_backend_falls_back_for_malformed_cached_scan_submission_payload(self):
        backend = select_submission_backend({"scan_submission": {"version": "old"}})
        self.assertIsInstance(backend, ScanSubmissionBackend)
        self.assertTrue(backend.supports_editing)

    def test_scan_submission_row_exposes_scan_mode_for_scannable_float_param(self):
        try:
            from ndscan._qt import QtWidgets
            from ndscan.dashboard.override_entry import ScanOverrideEntry
        except ModuleNotFoundError as exc:
            if exc.name == "PyQt5":
                self.skipTest("ARTIQ GUI PyQt5 dependency is not installed")
            raise

        qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.addCleanup(lambda: qt_app.processEvents())
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        row = ScanOverrideEntry(
            {
                "fqn": "frag.preload_time",
                "type": "float",
                "description": "preload time",
                "default": "3.0",
                "spec": {"is_scannable": True},
            },
            "*",
            is_scannable=True,
            backend=backend,
            submission_mode="grid",
        )

        mode_labels = [
            row._mode_box.itemText(index) for index in range(row._mode_box.count())
        ]

        self.assertIn("Fixed", mode_labels)
        self.assertIn("Scan", mode_labels)
        self.assertIn("Rebind", mode_labels)
        row.deleteLater()

    def test_legacy_backend_iterates_axes_then_overrides(self):
        backend = LegacySubmissionBackend()
        entries = list(
            backend.iter_configured_entries(
                {
                    "scan": {
                        "axes": [
                            {"fqn": "frag.x", "path": "*", "type": "linear", "range": {}},
                            {"fqn": "frag.y", "path": "child", "type": "list", "range": {}},
                        ]
                    },
                    "overrides": {
                        "frag.z": [{"path": "*", "value": 3.0}],
                    },
                }
            )
        )

        self.assertEqual(
            entries,
            [
                ("frag.x", "*"),
                ("frag.y", "child"),
                ("frag.z", "*"),
            ],
        )

    def test_legacy_backend_applies_submission_state(self):
        backend = LegacySubmissionBackend()
        state = backend.new_submission_state()
        state.add_override(fqn="frag.z", path="*", value=3.0)
        state.add_scan_axis(
            fqn="frag.x",
            path="*",
            axis_type="linear",
            axis_range={"start": 0.0, "stop": 1.0, "num_points": 11},
        )
        state.set_scan_options(
            LegacyScanOptionsState(
                num_repeats=2,
                num_repeats_per_point=3,
                no_axes_mode="single",
                randomise_order_globally=True,
                skip_on_persistent_transitory_error=True,
            )
        )

        params = {"scan": {"old": True}, "overrides": {"old": []}}
        backend.apply_submission_state(params, state)

        self.assertEqual(
            params["overrides"],
            {"frag.z": [{"path": "*", "value": 3.0}]},
        )
        self.assertEqual(
            params["scan"],
            {
                "num_repeats": 2,
                "num_repeats_per_point": 3,
                "no_axes_mode": "single",
                "randomise_order_globally": True,
                "skip_on_persistent_transitory_error": True,
                "axes": [
                    {
                        "fqn": "frag.x",
                        "path": "*",
                        "type": "linear",
                        "range": {"start": 0.0, "stop": 1.0, "num_points": 11},
                    }
                ],
            },
        )

    def test_legacy_backend_removes_scan_key_when_submission_has_no_scan_state(self):
        backend = LegacySubmissionBackend()
        params = {"scan": {"axes": [{"fqn": "frag.x", "path": "*"}]}, "overrides": {}}
        backend.apply_submission_state(params, backend.new_submission_state())
        self.assertNotIn("scan", params)

    def test_scan_submission_backend_serialises_fixed_and_scanned_entries(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {"max_points_per_batch": 8},
                    "metadata": {"demo_name": "scan_submission_dashboard"},
                },
                "overrides": {},
            }
        )
        state = backend.new_submission_state()
        state.add_override(fqn="frag.y", path="*", value=3.0)
        state.add_scan_axis(
            fqn="frag.x",
            path="*",
            axis_type="linear",
            axis_range={"start": 0.0, "stop": 1.0, "num_points": 11},
        )

        params = {"scan_submission": {"old": True}, "overrides": {"old": []}}
        backend.apply_submission_state(params, state)

        self.assertEqual(params["overrides"], {})
        self.assertEqual(params["scan_submission"]["mode"], {"type": "grid"})
        self.assertEqual(
            params["scan_submission"]["execution"],
            {"max_points_per_batch": 8},
        )
        self.assertEqual(
            params["scan_submission"]["metadata"],
            {"demo_name": "scan_submission_dashboard"},
        )
        self.assertEqual(len(params["scan_submission"]["entries"]), 2)
        by_fqn = {
            entry["target"]["fqn"]: entry for entry in params["scan_submission"]["entries"]
        }
        self.assertEqual(by_fqn["frag.y"]["mode"], {"type": "fixed", "value": 3.0})
        self.assertEqual(
            by_fqn["frag.x"]["mode"],
            {
                "type": "scan",
                "generator": {
                    "type": "linear",
                    "range": {"start": 0.0, "stop": 1.0, "num_points": 11},
                },
            },
        )

    def test_scan_submission_backend_serialises_grid_repeat_settings(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {"max_points_per_batch": 8},
                    "metadata": {"demo_name": "scan_submission_dashboard"},
                }
            }
        )
        state = backend.new_submission_state()
        state.set_grid_mode(
            randomise_order_globally=True,
            num_repeats_per_point=None,
            repeat_schedule="interleaved",
        )

        params = {"scan_submission": {"old": True}, "overrides": {"old": []}}
        backend.apply_submission_state(params, state)

        self.assertEqual(
            params["scan_submission"]["mode"],
            {
                "type": "grid",
                "randomise_order_globally": True,
                "num_repeats_per_point": None,
                "repeat_schedule": "interleaved",
            },
        )
        self.assertEqual(
            backend.initial_mode_state(),
            {
                "mode_type": "grid",
                "randomise_order_globally": False,
                "num_repeats_per_point": 1,
                "repeat_schedule": "serial",
            },
        )

    def test_scan_submission_backend_accepts_root_parameter_path(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [
                        {
                            "id": "x",
                            "kind": "param",
                            "target": {"fqn": "frag.x", "path": ""},
                            "mode": {"type": "fixed", "value": 1.0},
                        }
                    ],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        self.assertTrue(backend.supports_editing)

    def test_scan_submission_backend_accepts_grouped_scan_entries(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [
                        {
                            "id": "x",
                            "kind": "param",
                            "target": {"fqn": "frag.x", "path": "*"},
                            "mode": {
                                "type": "scan",
                                "group": "pair",
                                "generator": {
                                    "type": "list",
                                    "range": {"values": [1.0, 2.0]},
                                },
                            },
                        }
                    ],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        self.assertTrue(backend.supports_editing)

    def test_scan_submission_backend_accepts_rebind_entries(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [
                        {
                            "id": "detuning",
                            "kind": "param",
                            "target": {"fqn": "frag.detuning", "path": "*"},
                            "mode": {"type": "rebind", "expr": "x + offset"},
                        }
                    ],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        self.assertTrue(backend.supports_editing)

    def test_scan_submission_backend_accepts_pseudoparam_entries(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [
                        {
                            "id": "logical_drive",
                            "kind": "pseudoparam",
                            "mode": {
                                "type": "scan",
                                "generator": {
                                    "type": "list",
                                    "range": {"values": [0.0, 1.0, 2.0]},
                                },
                            },
                        }
                    ],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        self.assertTrue(backend.supports_editing)

    def test_scan_submission_backend_accepts_simple_gpo_entries(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {
                        "type": "gpo",
                        "objective": {
                            "kind": "channel",
                            "target": {"path": "scattering_rate"},
                        },
                        "backend": {
                            "kind": "nubo",
                            "initial_design_size": 3,
                            "acquisition": "ucb",
                            "minimise": True,
                        },
                    },
                    "entries": [
                        {
                            "id": "detuning",
                            "kind": "param",
                            "target": {"fqn": "frag.detuning", "path": ""},
                            "mode": {"type": "gpo_scan", "lower": -1.0, "upper": 1.0},
                        },
                        {
                            "id": "drive",
                            "kind": "param",
                            "target": {"fqn": "frag.drive", "path": ""},
                            "mode": {"type": "rebind", "expr": "detuning + 0.5"},
                        },
                    ],
                    "execution": {},
                    "metadata": {},
                },
                "channels": {
                    "scattering_rate": {
                        "path": "scattering_rate",
                        "description": "Scattering rate",
                        "type": "float",
                    }
                },
            }
        )
        self.assertTrue(backend.supports_editing)
        self.assertEqual(backend.initial_mode_state()["mode_type"], "gpo")
        self.assertEqual(
            backend.available_result_channels(),
            (
                {
                    "path": "scattering_rate",
                    "description": "Scattering rate",
                    "type": "float",
                },
            ),
        )

    def test_scan_submission_backend_iterates_existing_param_entries(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [
                        {
                            "id": "x",
                            "kind": "param",
                            "target": {"fqn": "frag.x", "path": "*"},
                            "mode": {
                                "type": "scan",
                                "generator": {
                                    "type": "list",
                                    "range": {"values": [1.0, 2.0]},
                                },
                            },
                        }
                    ],
                    "execution": {},
                    "metadata": {},
                },
                "overrides": {"frag.y": [{"path": "*", "value": 5.0}]},
            }
        )
        entries = list(
            backend.iter_configured_entries(
                {
                    "scan_submission": {
                        "version": 1,
                        "mode": {"type": "grid"},
                        "entries": [
                            {
                                "id": "x",
                                "kind": "param",
                                "target": {"fqn": "frag.x", "path": "*"},
                                "mode": {
                                    "type": "scan",
                                    "generator": {
                                        "type": "list",
                                        "range": {"values": [1.0, 2.0]},
                                    },
                                },
                            }
                        ],
                        "execution": {},
                        "metadata": {},
                    },
                    "overrides": {"frag.y": [{"path": "*", "value": 5.0}]},
                }
            )
        )
        self.assertEqual(entries, [("frag.x", "*"), ("frag.y", "*")])

    def test_scan_submission_backend_iterates_existing_pseudoparam_entries(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [
                        {
                            "id": "logical_drive",
                            "kind": "pseudoparam",
                            "mode": {
                                "type": "scan",
                                "generator": {
                                    "type": "list",
                                    "range": {"values": [0.0, 1.0, 2.0]},
                                },
                            },
                        }
                    ],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        entries = tuple(
            backend.iter_configured_pseudoparams(
                {
                    "scan_submission": {
                        "version": 1,
                        "mode": {"type": "grid"},
                        "entries": [
                            {
                                "id": "logical_drive",
                                "kind": "pseudoparam",
                                "mode": {
                                    "type": "scan",
                                    "generator": {
                                        "type": "list",
                                        "range": {"values": [0.0, 1.0, 2.0]},
                                    },
                                },
                            }
                        ],
                        "execution": {},
                        "metadata": {},
                    }
                }
            )
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].id, "logical_drive")

    def test_scan_submission_backend_serialises_scan_group(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        state = backend.new_submission_state()
        state.add_scan_axis(
            fqn="frag.left",
            path="*",
            axis_type="list",
            axis_range={"values": [1.0, 2.0]},
            scan_group="pair",
        )
        state.add_scan_axis(
            fqn="frag.right",
            path="*",
            axis_type="list",
            axis_range={"values": [10.0, 20.0]},
            scan_group="pair",
        )

        params = {"scan_submission": {"old": True}, "overrides": {}}
        backend.apply_submission_state(params, state)

        groups = [
            entry["mode"].get("group", None)
            for entry in params["scan_submission"]["entries"]
        ]
        self.assertEqual(groups, ["pair", "pair"])

    def test_scan_submission_backend_serialises_rebind_entry(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        state = backend.new_submission_state()
        state.add_scan_axis(
            fqn="frag.x",
            path="",
            axis_type="linear",
            axis_range={"start": 0.0, "stop": 1.0, "num_points": 11},
        )
        state.add_rebind(
            fqn="frag.y",
            path="",
            expression="x + 0.5",
        )

        params = {"scan_submission": {"old": True}, "overrides": {}}
        backend.apply_submission_state(params, state)

        by_fqn = {
            entry["target"]["fqn"]: entry for entry in params["scan_submission"]["entries"]
        }
        self.assertEqual(
            by_fqn["frag.y"]["mode"],
            {"type": "rebind", "expr": "x + 0.5"},
        )

    def test_scan_submission_backend_serialises_pseudoparam_entries(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        state = backend.new_submission_state()
        state.add_pseudoparam_fixed(entry_id="offset", value=0.5)
        state.add_pseudoparam_scan(
            entry_id="logical_drive",
            axis_type="list",
            axis_range={"values": [0.0, 1.0, 2.0]},
            scan_group="pair",
        )

        params = {"scan_submission": {"old": True}, "overrides": {}}
        backend.apply_submission_state(params, state)

        by_id = {entry["id"]: entry for entry in params["scan_submission"]["entries"]}
        self.assertEqual(by_id["offset"]["kind"], "pseudoparam")
        self.assertEqual(by_id["offset"]["mode"], {"type": "fixed", "value": 0.5})
        self.assertEqual(
            by_id["logical_drive"]["mode"],
            {
                "type": "scan",
                "group": "pair",
                "generator": {
                    "type": "list",
                    "range": {"values": [0.0, 1.0, 2.0]},
                },
            },
        )

    def test_scan_submission_backend_preserves_existing_param_entry_id(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [
                        {
                            "id": "detuning_symbol",
                            "kind": "param",
                            "target": {"fqn": "frag.detuning", "path": ""},
                            "mode": {"type": "fixed", "value": 1.0},
                        }
                    ],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        self.assertEqual(
            backend.symbol_name_for_target(fqn="frag.detuning", path=""),
            "detuning_symbol",
        )

    def test_scan_submission_backend_derives_unique_default_param_entry_ids(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {},
                    "metadata": {},
                },
                "schemata": {
                    "pkg.root.detuning": {},
                    "pkg.child.detuning": {},
                },
                "instances": {
                    "": ["pkg.root.detuning"],
                    "child": ["pkg.child.detuning"],
                },
            }
        )
        root_id = backend.symbol_name_for_target(fqn="pkg.root.detuning", path="")
        child_id = backend.symbol_name_for_target(fqn="pkg.child.detuning", path="child")
        self.assertEqual(root_id, "root_detuning")
        self.assertEqual(child_id, "child_detuning")
        self.assertNotEqual(root_id, child_id)

    def test_scan_submission_backend_prefers_short_name_for_unambiguous_parameter(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {},
                    "metadata": {},
                },
                "schemata": {
                    "pkg.demo.DashboardMappedDriveFragment.logical_drive": {},
                },
                "instances": {
                    "": ["pkg.demo.DashboardMappedDriveFragment.logical_drive"],
                },
            }
        )
        self.assertEqual(
            backend.symbol_name_for_target(
                fqn="pkg.demo.DashboardMappedDriveFragment.logical_drive",
                path="",
            ),
            "logical_drive",
        )

    def test_scan_submission_backend_serialises_gpo_mode_and_dimensions(self):
        backend = ScanSubmissionBackend(
            {
                "scan_submission": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {"max_points_per_batch": 8},
                    "metadata": {"demo_name": "scan_submission_dashboard"},
                }
            }
        )
        state = backend.new_submission_state()
        state.set_gpo_mode(
            objective_channel_path="hardware/scattering_rate",
            batch_size=4,
            initial_design_size=7,
            max_batches=12,
            acquisition="ei",
            minimise=False,
        )
        state.add_gpo_axis(
            fqn="frag.detuning",
            path="",
            lower=-20.0,
            upper=20.0,
        )
        state.add_rebind(
            fqn="frag.frequency",
            path="",
            expression="80.0 + detuning / 2.0",
        )
        state.add_pseudoparam_gpo_axis(
            entry_id="logical_intensity",
            lower=0.5,
            upper=2.0,
        )

        params = {"scan_submission": {"old": True}, "overrides": {"old": []}}
        backend.apply_submission_state(params, state)

        self.assertEqual(params["overrides"], {})
        self.assertEqual(
            params["scan_submission"]["mode"],
            {
                "type": "gpo",
                "objective": {
                    "kind": "channel",
                    "target": {"path": "hardware/scattering_rate"},
                },
                "backend": {
                    "kind": "nubo",
                    "batch_size": 4,
                    "initial_design_size": 7,
                    "max_batches": 12,
                    "acquisition": "ei",
                    "minimise": False,
                    "fit_steps": 800,
                    "fit_lr": 0.05,
                    "acquisition_num_starts": 10,
                    "surrogate_num_starts": 32,
                },
            },
        )
        by_target = {
            entry.get("target", {}).get("fqn", entry["id"]): entry
            for entry in params["scan_submission"]["entries"]
        }
        self.assertEqual(
            by_target["frag.detuning"]["mode"],
            {"type": "gpo_scan", "lower": -20.0, "upper": 20.0},
        )
        self.assertEqual(
            by_target["logical_intensity"]["mode"],
            {"type": "gpo_scan", "lower": 0.5, "upper": 2.0},
        )
        self.assertEqual(
            by_target["frag.frequency"]["mode"],
            {"type": "rebind", "expr": "80.0 + detuning / 2.0"},
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from ndscan.dashboard.submission import (
    HostSubmissionBackend,
    LegacyScanOptionsState,
    LegacySubmissionBackend,
    select_submission_backend,
)


class DashboardSubmissionBackendTest(unittest.TestCase):
    def test_select_submission_backend_prefers_legacy_scan_payload(self):
        backend = select_submission_backend({"scan": {}, "overrides": {}})
        self.assertIsInstance(backend, LegacySubmissionBackend)
        self.assertTrue(backend.supports_editing)

    def test_select_submission_backend_detects_host_scan_payload(self):
        backend = select_submission_backend(
            {
                "host_scan": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        self.assertIsInstance(backend, HostSubmissionBackend)
        self.assertTrue(backend.supports_editing)

    def test_host_backend_rejects_unsupported_host_scan_payload(self):
        backend = select_submission_backend(
            {
                "host_scan": {
                    "version": 1,
                    "mode": {
                        "type": "gpo",
                        "objective": {"kind": "channel", "target": {"path": "y"}},
                        "backend": {"kind": "nubo"},
                    },
                    "entries": [
                        {
                            "id": "x",
                            "kind": "param",
                            "target": {"fqn": "frag.x", "path": "*"},
                            "mode": {"type": "gpo_scan", "lower": 0.0, "upper": 1.0},
                        }
                    ],
                    "execution": {},
                    "metadata": {},
                }
            }
        )
        self.assertIsInstance(backend, HostSubmissionBackend)
        self.assertFalse(backend.supports_editing)

    def test_host_backend_falls_back_for_malformed_cached_host_scan_payload(self):
        backend = select_submission_backend({"host_scan": {"version": "old"}})
        self.assertIsInstance(backend, HostSubmissionBackend)
        self.assertTrue(backend.supports_editing)

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

    def test_host_backend_serialises_fixed_and_scanned_entries(self):
        backend = HostSubmissionBackend(
            {
                "host_scan": {
                    "version": 1,
                    "mode": {"type": "grid"},
                    "entries": [],
                    "execution": {"max_points_per_batch": 8},
                    "metadata": {"demo_name": "host_dashboard"},
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

        params = {"host_scan": {"old": True}, "overrides": {"old": []}}
        backend.apply_submission_state(params, state)

        self.assertEqual(params["overrides"], {})
        self.assertEqual(params["host_scan"]["mode"], {"type": "grid"})
        self.assertEqual(
            params["host_scan"]["execution"],
            {"max_points_per_batch": 8},
        )
        self.assertEqual(
            params["host_scan"]["metadata"],
            {"demo_name": "host_dashboard"},
        )
        self.assertEqual(len(params["host_scan"]["entries"]), 2)
        by_fqn = {
            entry["target"]["fqn"]: entry for entry in params["host_scan"]["entries"]
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

    def test_host_backend_accepts_root_parameter_path(self):
        backend = HostSubmissionBackend(
            {
                "host_scan": {
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

    def test_host_backend_iterates_existing_param_entries(self):
        backend = HostSubmissionBackend(
            {
                "host_scan": {
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
                    "host_scan": {
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


if __name__ == "__main__":
    unittest.main()

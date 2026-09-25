from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.reputation_aggregation import rffl_reputation_step
from flower_prototype.analyze_q3_v43 import build_runs
from flower_prototype.access_control import (
    AccessLedger, public_key_b64, sign_payload,
)


class ReputationAggregationTests(unittest.TestCase):
    def test_opposed_update_loses_reputation_and_is_removed(self):
        reference = [np.asarray([0.0, 0.0], dtype=np.float32)]
        models = [
            [np.asarray([1.0, 1.0], dtype=np.float32)],
            [np.asarray([1.0, 1.0], dtype=np.float32)],
            [np.asarray([-4.0, -4.0], dtype=np.float32)],
        ]
        prior = {"honest-1": 0.45, "honest-2": 0.45, "attacker-1": 0.10}
        model, reputations, contributions, removed, used_weights = rffl_reputation_step(
            reference, models, list(prior), prior, fade=0.0,
        )
        self.assertEqual(removed, [False, False, True])
        self.assertEqual(contributions[-1], 0.0)
        np.testing.assert_allclose(model[0], np.asarray([0.5, 0.5]))
        self.assertAlmostEqual(sum(reputations), 1.0)
        self.assertEqual(used_weights, [0.45, 0.45, 0.10])

    def test_uniform_aligned_updates_are_retained(self):
        reference = [np.asarray([0.0], dtype=np.float32)]
        models = [[np.asarray([1.0], dtype=np.float32)] for _ in range(4)]
        _, reputations, _, removed, _ = rffl_reputation_step(
            reference, models, [f"n{i}" for i in range(4)], {},
        )
        self.assertFalse(any(removed))
        self.assertTrue(all(abs(value - 0.25) < 1e-8 for value in reputations))

    def test_rffl_removal_does_not_create_hard_failure_or_penalty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = AccessLedger(
                root / "ledger.sqlite", root / "server.pem",
                mechanism_variant="rffl_reputation",
            )
            key = Ed25519PrivateKey.generate()
            public_key = public_key_b64(key)
            registration = {
                "action": "REGISTER", "node_id": "node", "profile": "honest",
                "public_key": public_key,
            }
            ledger.register_node(
                "node", "honest", public_key, sign_payload(key, registration),
            )
            removed = ledger.remove_baseline_participant(
                "node", 7, "RFFL_LOW_REPUTATION",
            )
            self.assertTrue(removed["removed"])
            self.assertEqual(removed["access_state"], "QUARANTINE")
            self.assertEqual(removed["penalty_debt"], 0.0)
            self.assertEqual(removed["hard_failures"], 0)
            self.assertEqual(ledger.connection.execute(
                "SELECT COUNT(*) FROM training_evidence"
            ).fetchone()[0], 0)
            self.assertEqual(ledger.connection.execute(
                "SELECT COUNT(*) FROM node_penalties"
            ).fetchone()[0], 0)
            event = ledger.connection.execute(
                "SELECT event_type FROM audit_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()[0]
            self.assertEqual(event, "BASELINE_PARTICIPANT_REMOVED")
            with self.assertRaisesRegex(ValueError, "invalid revocation"):
                ledger.revoke_training_access("node", 7, "RFFL_LOW_REPUTATION")
            ledger.connection.close()


class V43RunnerTests(unittest.TestCase):
    def test_analysis_aggregates_two_attackers_as_one_cohort(self):
        base = {
            "dataset": "fashion_mnist", "client_count": "20",
            "attacker_count": "2", "cohort_mode": "mixed_maturity",
            "variant": "access_full", "trust_estimator": "dirichlet_lcb",
            "attack_scenario": "diverse_then_repeat_backdoor",
            "noniid_alpha": "0.5", "repeat": "0", "attack_start_round": "2",
        }
        nodes = [
            {**base, "node_id": "attacker-1", "profile": "diverse_then_repeat_backdoor", "initial_access_state": "LIMITED", "onboarding_duration_seconds": "1"},
            {**base, "node_id": "attacker-2", "profile": "diverse_then_repeat_backdoor", "initial_access_state": "LIMITED", "onboarding_duration_seconds": "1"},
            {**base, "node_id": "honest-1", "profile": "honest", "initial_access_state": "ADMITTED", "onboarding_duration_seconds": "1"},
        ]
        events = [
            {**base, "round": "2", "node_id": "attacker-1", "returned": "1", "aggregated": "1", "aggregation_effective_weight": "0.025", "update_payload_bytes": "100", "server_round_processing_seconds": "0.2"},
            {**base, "round": "2", "node_id": "attacker-2", "returned": "1", "aggregated": "1", "aggregation_effective_weight": "0.025", "update_payload_bytes": "100", "server_round_processing_seconds": "0.2"},
            {**base, "round": "2", "node_id": "honest-1", "returned": "1", "aggregated": "1", "aggregation_effective_weight": "0.95", "update_payload_bytes": "100", "server_round_processing_seconds": "0.2"},
        ]
        metrics = [{**base, "round": "2", "accuracy": "0.8", "backdoor_asr": "0.1"}]
        configs = [{**base, "configuration_runtime_seconds": "10", "controller_state_bytes": "4096"}]
        run = build_runs(nodes, events, metrics, configs, "fixed_attacker_ratio")[0]
        self.assertEqual(run["target_aggregated_updates_after_start"], 2)
        self.assertAlmostEqual(run["target_effective_aggregation_share_after_start"], 0.05)
        self.assertEqual(run["update_payload_bytes"], 300)
        self.assertAlmostEqual(run["server_processing_seconds"], 0.2)

    def test_runner_has_external_baseline_ratio_and_overhead_stages(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (root / "flower_prototype" / "run_q3_evidence_v43.ps1").read_text()
        runner = (root / "flower_prototype" / "run_real_fl_matrix.ps1").read_text()
        compose = (root / "flower_prototype" / "docker-compose.evidence-farming.yml").read_text()
        server = (root / "flower_prototype" / "server.py").read_text()
        self.assertIn('"external_baselines"', wrapper)
        self.assertIn('"fixed_attacker_ratio"', wrapper)
        self.assertIn('Name = "rffl_reputation"', wrapper)
        self.assertIn('Clients=20; Attackers=2', wrapper)
        self.assertIn('[switch]$Resume', wrapper)
        self.assertIn('reusing complete batch', wrapper)
        self.assertIn('python -m flower_prototype.analyze_q3_v43', wrapper)
        self.assertIn('node-access-flower-prototype:0.23', wrapper)
        self.assertIn("real_fl_configuration_metrics.csv", runner)
        self.assertIn("server_round_processing_seconds", runner)
        self.assertIn("attacker-2:", compose)
        self.assertIn('self._post("/internal/fit/remove"', server)


if __name__ == "__main__":
    unittest.main()

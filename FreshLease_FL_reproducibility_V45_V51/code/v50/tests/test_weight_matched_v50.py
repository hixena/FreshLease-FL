from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import AccessLedger, V50_MATCHED_WEIGHT
from flower_prototype.analyze_weight_matched_v50 import paired_v50, target_weight_mass
from tests.test_progressive_access import accept_round, graduate_to_limited


class WeightMatchedV50Tests(unittest.TestCase):
    def test_promoted_newcomer_receives_time_invariant_matched_weight(self):
        self.assertAlmostEqual(V50_MATCHED_WEIGHT * 45, 34.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = AccessLedger(
                root / "db.sqlite", root / "server.pem",
                mechanism_variant="access_weight_matched",
                min_completed_tasks=9, min_limited_observation_updates=5,
                limited_aggregation_weight=0.5,
            )
            key = Ed25519PrivateKey.generate()
            graduate_to_limited(ledger, key)
            observed = []
            for rnd in range(1, 11):
                update_hash, decision = accept_round(ledger, key, rnd)
                observed.append((decision["access_state"], decision["aggregation_weight"],
                                 decision["access_transition"]))
                ledger.record_fit_aggregation("node", rnd, update_hash)
            self.assertEqual([x[1] for x in observed[:5]], [0.5] * 4 + [1.0])
            self.assertEqual(observed[4][0], "ADMITTED")
            self.assertEqual(observed[4][2], "PROMOTED")
            self.assertTrue(all(x == ("ADMITTED", V50_MATCHED_WEIGHT, "")
                                for x in observed[5:]))
            status = ledger.node_status("node", expire=False)
            self.assertFalse(status["freshness_lease_enabled"])
            self.assertTrue(ledger.verify_integrity()["valid"])
            ledger.connection.close()

    def test_mature_honest_node_keeps_full_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = AccessLedger(
                root / "db.sqlite", root / "server.pem",
                mechanism_variant="access_weight_matched",
                min_completed_tasks=9, min_limited_observation_updates=5,
                incumbent_node_ids={"node"},
            )
            key = Ed25519PrivateKey.generate()
            graduate_to_limited(ledger, key)
            for rnd in range(1, 8):
                update_hash, decision = accept_round(ledger, key, rnd)
                self.assertEqual(decision["aggregation_weight"], 1.0)
                ledger.record_fit_aggregation("node", rnd, update_hash)
            ledger.connection.close()

    def test_paired_analysis_audits_actual_mass_and_effect_direction(self):
        shared = dict(dataset="cifar10", client_count="20", attacker_count="2",
                      cohort_mode="mixed_maturity", trust_estimator="dirichlet_lcb",
                      attack_scenario="diverse_then_repeat_backdoor",
                      noniid_alpha="0.5", repeat="0")
        lease_run = dict(shared, variant="access_freshness_lease",
                         backdoor_asr_auc_after_start="0.1",
                         final_backdoor_asr="0.05", final_accuracy="0.30",
                         target_effective_aggregation_share_sustained="0.06")
        control_run = dict(shared, variant="access_weight_matched",
                           backdoor_asr_auc_after_start="0.2",
                           final_backdoor_asr="0.07", final_accuracy="0.31",
                           target_effective_aggregation_share_sustained="0.07")
        def guard(row):
            return dict(row, target_full_access_before_attack_rate="1.0",
                        target_first_full_access_round_max="5",
                        attack_start_round="6")
        def mass(row):
            return target_weight_mass(
                [dict(row, node_id="attacker-1", profile="attacker")],
                [dict(row, node_id="attacker-1", round="6", aggregated="1",
                      aggregation_weight=str(V50_MATCHED_WEIGHT))],
            )
        pairs, summary = paired_v50(
            [control_run], [lease_run], mass(control_run), mass(lease_run),
            [guard(control_run)], [guard(lease_run)],
        )
        self.assertAlmostEqual(pairs[0]["asr_auc_reduction"], 0.1)
        self.assertEqual(summary[0]["independent_pairs"], 1)
        self.assertEqual(summary[0]["empirical_weight_mass_matched_within_1pct_rate"], 1)


if __name__ == "__main__":
    unittest.main()

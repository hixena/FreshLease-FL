from __future__ import annotations

import unittest
from pathlib import Path

from flower_prototype.analyze_freshness_lease_v45 import (
    enrich_round_runs,
    paired_effects_v45,
)


class V45FreshnessAnalysisTests(unittest.TestCase):
    def test_paired_effects_compare_lease_with_both_controls(self):
        common = {
            "dataset": "fashion_mnist", "client_count": "20",
            "attacker_count": "2", "attack_scenario": "diverse_then_repeat_backdoor",
            "noniid_alpha": "0.5", "repeat": "0",
            "target_effective_aggregation_share_sustained": 0.20,
            "backdoor_asr_auc_after_start": 0.40,
            "backdoor_asr_at_round_8": 0.30,
            "final_backdoor_asr": 0.60, "final_accuracy": 0.80,
            "configuration_runtime_seconds": 10.0,
            "first_round_asr_ge_10pct": 4.0,
            "first_round_asr_ge_20pct": 6.0,
            "first_round_asr_ge_30pct": 8.0,
        }
        lease = {
            **common, "variant": "access_freshness_lease",
            "target_effective_aggregation_share_sustained": 0.12,
            "backdoor_asr_auc_after_start": 0.30,
            "backdoor_asr_at_round_8": 0.20,
            "final_backdoor_asr": 0.45,
            "first_round_asr_ge_10pct": 6.0,
            "first_round_asr_ge_20pct": 8.0,
            "first_round_asr_ge_30pct": 10.0,
        }
        progressive = {**common, "variant": "access_full"}
        direct = {
            **common, "variant": "access_no_limited",
            "target_effective_aggregation_share_sustained": 0.22,
            "backdoor_asr_auc_after_start": 0.45,
            "backdoor_asr_at_round_8": 0.35,
            "final_backdoor_asr": 0.65,
        }
        effects = paired_effects_v45([lease, progressive, direct])
        lease_vs_progressive = next(
            row for row in effects
            if row["method_variant"] == "access_freshness_lease"
            and row["control_variant"] == "access_full"
        )
        self.assertAlmostEqual(
            lease_vs_progressive["sustained_target_share_reduction"], 0.08
        )
        self.assertAlmostEqual(
            lease_vs_progressive["final_backdoor_asr_reduction"], 0.15
        )
        self.assertEqual(lease_vs_progressive["asr_ge_20pct_delay"], 2.0)

    def test_round_enrichment_uses_node_membership_not_profile_names(self):
        base = {
            "dataset": "fashion_mnist", "client_count": "20",
            "attacker_count": "2", "cohort_mode": "mixed_maturity",
            "variant": "access_freshness_lease",
            "trust_estimator": "dirichlet_lcb",
            "attack_scenario": "benign_concept_drift",
            "noniid_alpha": "0.5", "repeat": "0",
        }
        nodes = [
            {**base, "node_id": "attacker-1", "profile": "benign_concept_drift"},
            {**base, "node_id": "honest-1", "profile": "honest"},
        ]
        events = [
            {**base, "round": "10", "node_id": "attacker-1",
             "access_transition": "LEASE_EXPIRED"},
            {**base, "round": "10", "node_id": "honest-1",
             "access_transition": ""},
        ]
        round_runs = [{**base, "round": 10}]
        enriched = enrich_round_runs(round_runs, nodes, events)[0]
        self.assertEqual(enriched["target_lease_expirations"], 1)
        self.assertEqual(enriched["target_lease_renewals"], 0)


class V45RunnerTests(unittest.TestCase):
    def test_runner_freezes_preregistered_matrix_and_outputs(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (
            root / "flower_prototype" / "run_freshness_lease_v45.ps1"
        ).read_text()
        compose = (
            root / "flower_prototype" / "docker-compose.evidence-farming.yml"
        ).read_text()
        self.assertIn('[int]$Rounds = 50', wrapper)
        self.assertIn(') { 16 } else { $Rounds }', wrapper)
        self.assertIn('AccessLeaseFullUpdates=5', wrapper)
        self.assertIn('LimitedObservationUpdates=5', wrapper)
        self.assertIn('Name = "access_freshness_lease"', wrapper)
        self.assertIn('"benign_concept_drift"', wrapper)
        self.assertIn('"diverse_then_repeat_backdoor"', wrapper)
        self.assertIn('ClientCount=20; TargetCount=2', wrapper)
        self.assertIn(
            'python -m flower_prototype.analyze_freshness_lease_v45', wrapper
        )
        self.assertIn(
            'ACCESS_LEASE_FULL_UPDATES: ${ACCESS_LEASE_FULL_UPDATES:-5}', compose
        )


if __name__ == "__main__":
    unittest.main()

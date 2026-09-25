from __future__ import annotations

import unittest
from pathlib import Path

from flower_prototype.analyze_q3_access_v51 import (
    access_runs, audit_integrity_rows, summarize_access,
)
from flower_prototype.audit_probe_v51 import run_all


class Q3AccessV51Tests(unittest.TestCase):
    def test_protocol_challenges_include_unanchored_suffix_negative_control(self):
        cases = {item["case"]: item for item in run_all()}
        self.assertEqual(len(cases), 7)
        self.assertTrue(all(item["detected_or_valid"] == item["expected"]
                            for item in cases.values()))
        self.assertEqual(cases["delete_latest_audit_event"]["internal_chain_valid"], 1)
        self.assertEqual(cases["delete_latest_audit_event"]["head_matches_snapshot"], 0)

    def test_benign_newcomer_count_not_confused_with_incumbents(self):
        common = dict(dataset="cifar10", client_count="10", attacker_count="1",
                      cohort_mode="mixed_maturity", variant="access_freshness_lease",
                      trust_estimator="dirichlet_lcb", attack_scenario="benign_concept_drift",
                      noniid_alpha="0.5", repeat="0")
        target = {**common, "node_id": "target-1", "profile": "benign_concept_drift",
                  "initial_access_state": "LIMITED", "completed_tasks": "9",
                  "onboarding_duration_seconds": "4.2"}
        incumbents = [{**common, "node_id": f"honest-{i}", "profile": "honest",
                       "initial_access_state": "ADMITTED", "completed_tasks": "0",
                       "onboarding_duration_seconds": "0.1"} for i in range(9)]
        guard = [{**common, "target_full_access_before_attack_rate": "1.0",
                  "target_first_full_access_round_max": "5"}]
        row = access_runs([target] + incumbents, [common], guard)[0]
        self.assertEqual(row["legitimate_newcomer_case"], 1)
        self.assertEqual(row["newcomer_initial_training_eligible_rate"], 1)
        self.assertEqual(row["newcomer_initial_limited_rate"], 1)
        self.assertEqual(row["incumbent_count"], 9)
        self.assertEqual(row["newcomer_completed_tasks_mean"], 9)
        self.assertEqual(summarize_access([row])[0]["independent_runs"], 1)

    def test_runner_keeps_formal_conditions_separate(self):
        root = Path(__file__).resolve().parents[1]
        runner = (root / "flower_prototype" / "run_q3_access_v51.ps1").read_text()
        self.assertIn('Alpha = 0.1; Clients = 20; Attackers = 2', runner)
        self.assertIn('Alpha = 10.0; Clients = 20; Attackers = 2', runner)
        self.assertIn('Alpha = 0.5; Clients = 10; Attackers = 1', runner)
        self.assertIn('AttackStartRound=6', runner)
        self.assertIn('AccessLeaseFullUpdates=5', runner)
        self.assertIn('analyze_q3_access_v51', runner)

    def test_missing_controller_audit_rejected(self):
        config = {"dataset": "cifar10", "variant": "access_freshness_lease",
                  "client_count": "10", "attacker_count": "1", "cohort_mode": "mixed_maturity",
                  "trust_estimator": "dirichlet_lcb", "attack_scenario": "benign_concept_drift",
                  "noniid_alpha": "0.5", "repeat": "0"}
        with self.assertRaisesRegex(ValueError, "missing or duplicated"):
            audit_integrity_rows([], [config])
        audit = {**config, "audit_valid": "1", "chain_valid": "1",
                 "signatures_valid": "1", "event_count": "10", "checked_signatures": "5"}
        audit_integrity_rows([audit], [config])
        audit["signatures_valid"] = "0"
        with self.assertRaisesRegex(ValueError, "failed"):
            audit_integrity_rows([audit], [config])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path

from flower_prototype.analyze_post_promotion_v49 import (
    adjust_windows_to_attack_start, audit_post_promotion,
)
from flower_prototype.attack_schedule import effective_training_attack


def fixture():
    common = {
        "dataset": "cifar10", "client_count": "20", "attacker_count": "2",
        "cohort_mode": "mixed_maturity", "variant": "access_freshness_lease",
        "trust_estimator": "dirichlet_lcb",
        "attack_scenario": "diverse_then_repeat_backdoor",
        "noniid_alpha": "0.5", "repeat": "0", "attack_start_round": "6",
    }
    nodes = [
        {**common, "node_id": name, "profile": "diverse_then_repeat_backdoor",
         "initial_access_state": "LIMITED"}
        for name in ("target-1", "target-2")
    ]
    nodes += [{**common, "node_id": "honest-1", "profile": "honest",
               "initial_access_state": "ADMITTED"}]
    events = []
    for name in ("target-1", "target-2"):
        for round_number, state, weight in ((4, "LIMITED", 0.5),
                                            (5, "ADMITTED", 1.0),
                                            (6, "ADMITTED", 0.0),
                                            (10, "LIMITED", 0.5)):
            events.append({
                **common, "round": str(round_number), "node_id": name,
                "access_state_after": state,
                "aggregated": "1" if weight else "0",
                "aggregation_weight": str(weight),
                "aggregation_effective_weight": str(weight),
            })
    return nodes, events, [common]


class V49PostPromotionTests(unittest.TestCase):
    def test_attack_schedule_is_clean_before_full_access(self):
        self.assertEqual(effective_training_attack("backdoor", 5, 6), "none")
        self.assertEqual(effective_training_attack("backdoor", 6, 6), "backdoor")

    def test_actual_full_weight_clean_round_is_required(self):
        nodes, events, configs = fixture()
        # A rejected update on the first attacking round is a legitimate
        # outcome: preattack authorization, rather than attack success, matters.
        guard = audit_post_promotion(nodes, events, configs)
        self.assertEqual(guard[0]["target_first_full_access_round_max"], 5)
        self.assertEqual(guard[0]["target_full_access_before_attack_rate"], 1)
        for row in events:
            if row["node_id"] == "target-2" and row["round"] == "5":
                row["aggregation_weight"] = "0.5"
        with self.assertRaisesRegex(ValueError, "no full-weight clean update"):
            audit_post_promotion(nodes, events, configs)

    def test_late_promotion_or_old_attack_start_fails(self):
        nodes, events, configs = fixture()
        for row in events:
            if row["node_id"] == "target-1" and row["round"] == "5":
                row["access_state_after"] = "LIMITED"
        with self.assertRaisesRegex(ValueError, "not ADMITTED before attack"):
            audit_post_promotion(nodes, events, configs)
        nodes, events, configs = fixture()
        configs[0]["attack_start_round"] = "2"
        with self.assertRaisesRegex(ValueError, "attack start differs"):
            audit_post_promotion(nodes, events, configs)

    def test_early_window_starts_with_attack_not_empty_rounds_two_to_four(self):
        nodes, events, configs = fixture()
        for row in events:
            if row["round"] == "6":
                row["aggregation_effective_weight"] = "1"
        for round_number in (6, 10):
            events.append({**configs[0], "round": str(round_number),
                           "node_id": "honest-1", "aggregation_effective_weight": "9"})
        run = {**configs[0]}
        adjust_windows_to_attack_start([run], events, nodes)
        self.assertGreater(run["target_effective_aggregation_share_early"], 0)
        self.assertGreater(run["target_effective_aggregation_share_late"], 0)

    def test_resume_cannot_reuse_sixteen_round_smoke_as_formal(self):
        nodes, events, configs = fixture()
        metrics = [{**configs[0], "round": str(i)} for i in range(1, 17)]
        with self.assertRaisesRegex(ValueError, "missing required rounds"):
            audit_post_promotion(nodes, events, configs, metrics=metrics,
                                 expected_rounds=50)

    def test_runner_uses_separate_directory_and_ten_formal_seeds(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (root / "flower_prototype" / "run_post_promotion_v49.ps1").read_text()
        self.assertIn('[int]$Repeats = 10', wrapper)
        self.assertIn('AttackStartRound=6', wrapper)
        self.assertIn('post_promotion_v49_results-$Stage-', wrapper)
        self.assertIn('analyze_post_promotion_v49 validate', wrapper)


if __name__ == "__main__":
    unittest.main()

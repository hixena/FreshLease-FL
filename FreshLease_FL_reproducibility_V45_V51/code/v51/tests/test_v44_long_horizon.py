from __future__ import annotations

import math
import unittest
from pathlib import Path

from flower_prototype.analyze_long_horizon_v44 import (
    build_long_runs,
    build_round_runs,
    normalized_auc,
    paired_effects_long,
    summarize_long,
)


class V44LongHorizonAnalysisTests(unittest.TestCase):
    def synthetic_rows(self):
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
        events = []
        for round_number in range(1, 11):
            early = round_number <= 4
            target_weight = 0.05 if early else 0.10
            target_state = "LIMITED" if early else "ADMITTED"
            for node_id in ("attacker-1", "attacker-2"):
                events.append({
                    **base, "round": str(round_number), "node_id": node_id,
                    "returned": "1", "aggregated": "1",
                    "access_state_after": target_state,
                    "aggregation_effective_weight": str(target_weight),
                    "aggregation_reputation_removed": "0",
                    "update_payload_bytes": "100",
                    "server_round_processing_seconds": "0.2",
                })
            events.append({
                **base, "round": str(round_number), "node_id": "honest-1",
                "returned": "1", "aggregated": "1",
                "access_state_after": "ADMITTED",
                "aggregation_effective_weight": str(1.0 - 2 * target_weight),
                "aggregation_reputation_removed": "0",
                "update_payload_bytes": "100",
                "server_round_processing_seconds": "0.2",
            })
        metrics = [
            {**base, "round": str(round_number), "accuracy": "0.8", "backdoor_asr": str(round_number / 100)}
            for round_number in range(1, 11)
        ]
        configs = [{
            **base, "configuration_runtime_seconds": "10",
            "controller_state_bytes": "4096",
        }]
        return nodes, events, metrics, configs

    def test_long_horizon_metrics_split_pre_and_post_promotion(self):
        nodes, events, metrics, configs = self.synthetic_rows()
        run = build_long_runs(nodes, events, metrics, configs, "formal")[0]
        self.assertEqual(run["total_rounds"], 10)
        self.assertEqual(run["target_full_access_round_mean"], 5.0)
        self.assertEqual(run["target_promoted_rate"], 1.0)
        self.assertAlmostEqual(run["target_effective_aggregation_share_early"], 0.10)
        self.assertAlmostEqual(run["target_effective_aggregation_share_late"], 0.20)
        self.assertAlmostEqual(run["backdoor_asr_auc_after_start"], 0.06)
        self.assertAlmostEqual(run["backdoor_asr_at_round_8"], 0.08)
        self.assertAlmostEqual(run["backdoor_asr_growth_round_8_to_final"], 0.02)
        self.assertEqual(run["first_round_asr_ge_10pct"], 10.0)
        self.assertTrue(math.isnan(run["first_round_asr_ge_20pct"]))

        round_runs = build_round_runs(nodes, events, metrics)
        round_four = next(row for row in round_runs if row["round"] == 4)
        round_five = next(row for row in round_runs if row["round"] == 5)
        self.assertEqual(round_four["target_limited_rate"], 1.0)
        self.assertEqual(round_five["target_admitted_rate"], 1.0)
        self.assertAlmostEqual(round_four["target_effective_aggregation_share"], 0.10)
        self.assertAlmostEqual(round_five["target_effective_aggregation_share"], 0.20)

    def test_auc_is_time_normalized(self):
        self.assertAlmostEqual(normalized_auc([(2, 0.1), (4, 0.3)]), 0.2)

    def test_summary_reports_threshold_reach_rate(self):
        nodes, events, metrics, configs = self.synthetic_rows()
        run = build_long_runs(nodes, events, metrics, configs, "formal")[0]
        summary = summarize_long([run])[0]
        self.assertEqual(summary["asr_ge_10pct_rate"], 1.0)
        self.assertEqual(summary["asr_ge_20pct_rate"], 0.0)

    def test_pairing_uses_direct_admission_control(self):
        common = {
            "dataset": "fashion_mnist", "client_count": "20",
            "attacker_count": "2", "attack_scenario": "diverse_then_repeat_backdoor",
            "noniid_alpha": "0.5", "repeat": "0",
            "target_effective_aggregation_share_early": 0.10,
            "target_effective_aggregation_share_late": 0.20,
            "backdoor_asr_auc_after_start": 0.20,
            "backdoor_asr_at_round_8": 0.20,
            "backdoor_asr_growth_round_8_to_final": 0.10,
            "final_backdoor_asr": 0.30, "final_accuracy": 0.80,
            "configuration_runtime_seconds": 10.0,
            "first_round_asr_ge_10pct": 5.0,
            "first_round_asr_ge_20pct": 8.0,
            "first_round_asr_ge_30pct": 10.0,
        }
        method = {**common, "variant": "access_full"}
        control = {
            **common, "variant": "access_no_limited",
            "target_effective_aggregation_share_early": 0.20,
            "backdoor_asr_auc_after_start": 0.30,
            "backdoor_asr_at_round_8": 0.30,
            "backdoor_asr_growth_round_8_to_final": 0.20,
            "final_backdoor_asr": 0.50,
            "first_round_asr_ge_10pct": 3.0,
            "first_round_asr_ge_20pct": 6.0,
            "first_round_asr_ge_30pct": 8.0,
        }
        effect = paired_effects_long([method, control])[0]
        self.assertAlmostEqual(effect["early_target_share_reduction"], 0.10)
        self.assertAlmostEqual(effect["late_target_share_reduction"], 0.0)
        self.assertAlmostEqual(effect["final_backdoor_asr_reduction"], 0.20)
        self.assertEqual(effect["asr_ge_20pct_delay"], 2.0)


class V44RunnerTests(unittest.TestCase):
    def test_runner_freezes_focused_fifty_round_matrix(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (root / "flower_prototype" / "run_long_horizon_v44.ps1").read_text()
        runner = (root / "flower_prototype" / "run_real_fl_matrix.ps1").read_text()
        requirements = (root / "flower_prototype" / "requirements.txt").read_text()
        self.assertIn('[ValidateSet("smoke", "formal")]', wrapper)
        self.assertIn('[int]$Rounds = 50', wrapper)
        self.assertIn('ClientCount=20; TargetCount=2', wrapper)
        self.assertIn('$scenario = "diverse_then_repeat_backdoor"', wrapper)
        self.assertIn('Name = "rffl_reputation"', wrapper)
        self.assertIn('ProgressActivity="V44 long-horizon matrix"', wrapper)
        self.assertIn('python -m flower_prototype.analyze_long_horizon_v44', wrapper)
        self.assertNotIn('diverse_then_repeat_farming', wrapper)
        self.assertIn('[string]$ProgressActivity', runner)
        self.assertIn(r'online_access round=(\d+)', runner)
        self.assertIn('[IO.Path]::GetFullPath($resultDir)', runner)
        self.assertIn('scipy>=1.10,<2', requirements)


if __name__ == "__main__":
    unittest.main()

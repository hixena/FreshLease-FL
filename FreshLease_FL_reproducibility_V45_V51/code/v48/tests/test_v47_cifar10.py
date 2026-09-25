from __future__ import annotations

import unittest
from pathlib import Path

from flower_prototype.analyze_cifar10_v47 import paired_effects_v47


def run(variant: str) -> dict:
    return {
        "dataset": "cifar10", "client_count": "20", "attacker_count": "2",
        "cohort_mode": "mixed_maturity", "trust_estimator": "dirichlet_lcb",
        "attack_scenario": "diverse_then_repeat_backdoor",
        "noniid_alpha": "0.5", "repeat": "0", "variant": variant,
        "target_effective_aggregation_share_sustained": 0.10,
        "target_limited_returned_update_rate_after_start": 0.0,
        "target_full_weight_aggregated_update_rate_after_start": 1.0,
        "backdoor_asr_auc_after_start": 0.60,
        "backdoor_asr_at_round_8": 0.30,
        "final_backdoor_asr": 0.70,
        "final_accuracy": 0.45,
        "configuration_runtime_seconds": 100.0,
        "first_round_asr_ge_10pct": 3.0,
        "first_round_asr_ge_20pct": 5.0,
        "first_round_asr_ge_30pct": 8.0,
    }


class V47AnalysisTests(unittest.TestCase):
    def test_lease_is_paired_with_both_external_controls(self):
        lease = run("access_freshness_lease")
        lease["target_effective_aggregation_share_sustained"] = 0.07
        lease["backdoor_asr_auc_after_start"] = 0.48
        lease["final_backdoor_asr"] = 0.58
        lease["target_limited_returned_update_rate_after_start"] = 0.5
        lease["target_full_weight_aggregated_update_rate_after_start"] = 0.5
        full = run("access_full")
        rffl = run("rffl_reputation")
        rffl["backdoor_asr_auc_after_start"] = 0.54
        effects = paired_effects_v47([lease, full, rffl])
        self.assertEqual(len(effects), 2)
        controls = {row["control_variant"] for row in effects}
        self.assertEqual(controls, {"access_full", "rffl_reputation"})
        against_full = next(
            row for row in effects if row["control_variant"] == "access_full"
        )
        self.assertAlmostEqual(against_full["asr_auc_reduction"], 0.12)
        self.assertAlmostEqual(
            against_full["sustained_target_share_reduction"], 0.03
        )


class V47RunnerTests(unittest.TestCase):
    def test_runner_freezes_minimal_cifar_matrix(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (root / "flower_prototype" / "run_cifar10_v47.ps1").read_text()
        matrix = (root / "flower_prototype" / "run_real_fl_matrix.ps1").read_text()
        compose = (
            root / "flower_prototype" / "docker-compose.evidence-farming.yml"
        ).read_text()
        self.assertIn('Datasets=@("cifar10")', wrapper)
        self.assertIn('ClientCount=20; TargetCount=2', wrapper)
        self.assertIn('AccessLeaseFullUpdates=5', wrapper)
        self.assertIn('MaxLocalTrainSamples=512', wrapper)
        self.assertIn('Name = "rffl_reputation"', wrapper)
        self.assertIn('"cifar10"', matrix)
        self.assertIn('MAX_LOCAL_TRAIN_SAMPLES:', compose)
        self.assertIn('python -m flower_prototype.analyze_cifar10_v47', wrapper)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest
from pathlib import Path

from flower_prototype.analyze_v46_evidence import (
    annotate_lease_setting,
    normalize_sensitivity,
    paired_effects,
    paired_summary,
)


def run(variant: str, repeat: str = "0") -> dict:
    return {
        "dataset": "fashion_mnist", "client_count": "20",
        "attacker_count": "2", "cohort_mode": "mixed_maturity",
        "trust_estimator": "dirichlet_lcb",
        "attack_scenario": "diverse_then_repeat_backdoor",
        "noniid_alpha": "0.5", "repeat": repeat, "variant": variant,
        "target_effective_aggregation_share_sustained": 0.10,
        "target_limited_returned_update_rate_after_start": 0.10,
        "target_full_weight_aggregated_update_rate_after_start": 0.90,
        "backdoor_asr_auc_after_start": 0.50,
        "backdoor_asr_at_round_8": 0.20,
        "final_backdoor_asr": 0.70, "final_accuracy": 0.83,
        "configuration_runtime_seconds": 100.0,
        "first_round_asr_ge_10pct": 5.0,
        "first_round_asr_ge_20pct": 10.0,
        "first_round_asr_ge_30pct": 15.0,
    }


class V46AnalysisTests(unittest.TestCase):
    def test_generalization_pair_is_lease_against_progressive_control(self):
        method = run("access_freshness_lease")
        method.update({
            "target_effective_aggregation_share_sustained": 0.08,
            "target_limited_returned_update_rate_after_start": 0.50,
            "target_full_weight_aggregated_update_rate_after_start": 0.50,
            "backdoor_asr_auc_after_start": 0.42,
            "final_backdoor_asr": 0.61,
            "first_round_asr_ge_20pct": 12.0,
        })
        effects = paired_effects(
            [method, run("access_full")], "generalization"
        )
        self.assertEqual(len(effects), 1)
        self.assertAlmostEqual(
            effects[0]["sustained_target_share_reduction"], 0.02
        )
        self.assertAlmostEqual(effects[0]["asr_auc_reduction"], 0.08)
        self.assertAlmostEqual(
            effects[0]["final_backdoor_asr_reduction"], 0.09
        )
        self.assertEqual(effects[0]["asr_ge_20pct_delay"], 2.0)

    def test_sensitivity_normalization_preserves_lease_length(self):
        rows = normalize_sensitivity([
            {"variant": "access_freshness_lease",
             "configured_access_lease_full_updates": "3"},
            {"variant": "access_freshness_lease",
             "access_lease_full_updates": "10"},
            {"variant": "access_full", "access_lease_full_updates": "5"},
        ])
        self.assertEqual(
            [row["variant"] for row in rows],
            ["access_freshness_lease_L3",
             "access_freshness_lease_L10", "access_full"],
        )

    def test_unsuffixed_lease_annotation_uses_configured_length(self):
        rows = annotate_lease_setting([{
            "variant": "access_freshness_lease",
            "configured_access_lease_full_updates": "5",
        }])
        self.assertEqual(rows[0]["lease_full_updates"], 5)

    def test_single_pair_has_no_pvalue(self):
        method = run("access_freshness_lease")
        method["backdoor_asr_auc_after_start"] = 0.42
        summary = paired_summary(
            paired_effects([method, run("access_full")], "generalization")
        )[0]
        self.assertTrue(math.isnan(
            summary["asr_auc_reduction_paired_t_pvalue"]
        ))

    def test_sensitivity_builds_control_and_l5_reference_pairs(self):
        rows = [
            run("access_freshness_lease_L3"),
            run("access_freshness_lease_L5"),
            run("access_freshness_lease_L10"),
            run("access_full"),
        ]
        effects = paired_effects(rows, "sensitivity")
        comparisons = {
            (row["method_variant"], row["control_variant"])
            for row in effects
        }
        self.assertEqual(len(effects), 5)
        self.assertEqual(comparisons, {
            ("access_freshness_lease_L3", "access_full"),
            ("access_freshness_lease_L5", "access_full"),
            ("access_freshness_lease_L10", "access_full"),
            ("access_freshness_lease_L3", "access_freshness_lease_L5"),
            ("access_freshness_lease_L10", "access_freshness_lease_L5"),
        })

    def test_pair_summary_reports_direction_and_paired_t_interval(self):
        pair_rows = []
        for repeat, reduction in enumerate((0.06, 0.07, 0.08, 0.09, 0.10)):
            method = run("access_freshness_lease", str(repeat))
            control = run("access_full", str(repeat))
            method["backdoor_asr_auc_after_start"] = 0.50 - reduction
            pair_rows.extend(paired_effects(
                [method, control], "generalization"
            ))
        summary = paired_summary(pair_rows)[0]
        self.assertEqual(summary["independent_pairs"], 5)
        self.assertAlmostEqual(summary["asr_auc_reduction_mean"], 0.08)
        self.assertEqual(summary["asr_auc_reduction_positive_pairs"], 5)
        self.assertLess(summary["asr_auc_reduction_paired_t_pvalue"], 0.01)


class V46RunnerTests(unittest.TestCase):
    def test_runner_separates_generalization_and_sensitivity(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (
            root / "flower_prototype" / "run_v46_evidence.ps1"
        ).read_text()
        analyzer = (
            root / "flower_prototype" / "analyze_v46_evidence.py"
        ).read_text()
        self.assertIn('[ValidateSet("generalization", "sensitivity")]', wrapper)
        self.assertIn('Dataset = "fashion_mnist"; Alpha = 0.1', wrapper)
        self.assertIn('Dataset = "fashion_mnist"; Alpha = 10.0', wrapper)
        self.assertIn('Dataset = "digits"; Alpha = 0.5', wrapper)
        self.assertIn('foreach ($lease in @(3, 5, 10))', wrapper)
        self.assertIn('python -m flower_prototype.analyze_v46_evidence', wrapper)
        self.assertIn('paired_summary.csv', wrapper)
        self.assertIn('ttest_1samp', analyzer)


if __name__ == "__main__":
    unittest.main()

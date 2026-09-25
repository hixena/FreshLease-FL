from __future__ import annotations

import unittest

from flower_prototype.analyze_parameter_tradeoff import summarize, wilson_interval


def row(config: str, repeat: int, exposure: float, benign_accuracy: float,
        revoked: int = 1) -> dict:
    return {
        "config_id": config,
        "variant": "full" if config == "baseline_full" else "progressive_full",
        "noniid_alpha": "0.5",
        "repeat": str(repeat),
        "limited_clean_updates": "4",
        "limited_aggregation_weight": "0.25",
        "cumulative_risk_decay": "0.5",
        "cumulative_risk_threshold": "0.9",
        "self_reversal_gate": "0.2",
        "attack_revoked": revoked,
        "attack_detection_delay_rounds": "1" if revoked else "",
        "attack_weight_after_start": exposure,
        "attack_normal_false_revocations": 0,
        "benign_target_false_revoked": 0,
        "benign_normal_false_revocations": 0,
        "benign_final_accuracy": benign_accuracy,
    }


class ParameterTradeoffTests(unittest.TestCase):
    def test_paired_reduction_accuracy_gap_and_pareto(self):
        rows = []
        for repeat in range(3):
            rows.extend([
                row("baseline_full", repeat, 9.0, 0.86, revoked=0),
                row("safer", repeat, 0.25, 0.82),
                row("dominated", repeat, 0.50, 0.80),
            ])
        summary = summarize(rows)
        safer = next(item for item in summary
                     if item["config_id"] == "safer" and item["noniid_alpha"] == "0.5")
        dominated = next(item for item in summary
                         if item["config_id"] == "dominated" and item["noniid_alpha"] == "0.5")
        self.assertAlmostEqual(safer["attack_weight_reduction_vs_full_mean"], 1 - 0.25 / 9.0)
        self.assertAlmostEqual(safer["benign_accuracy_gap_vs_full_mean"], -0.04)
        self.assertEqual(safer["safety_feasible"], 1)
        self.assertEqual(safer["pareto_optimal"], 1)
        self.assertEqual(dominated["pareto_optimal"], 0)

    def test_wilson_interval_reports_bounds_not_half_width(self):
        low, high = wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.27)
        self.assertLess(high, 0.29)


if __name__ == "__main__":
    unittest.main()

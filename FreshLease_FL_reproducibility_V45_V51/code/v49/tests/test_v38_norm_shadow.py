from __future__ import annotations

import statistics
import unittest
from pathlib import Path

from flower_prototype.analyze_hybrid_v38 import t_ci95_or_blank
from flower_prototype.analyze_norm_shadow_v38 import analyze_nodes, summarize
from flower_prototype.online_checks import (
    NORM_OUTCOME_EXTREME, NORM_OUTCOME_MODERATE,
    norm_escalation_decision,
)


ROOT = Path(__file__).resolve().parents[1]


class V38NormShadowTests(unittest.TestCase):
    def test_repeated_moderate_excess_is_shadow_rejected(self):
        self.assertEqual(
            norm_escalation_decision(
                NORM_OUTCOME_MODERATE, 1, moderate_shadow=True,
            ),
            ("REPEATED_SHADOW_REJECTED", 2, False, True),
        )
        self.assertEqual(
            norm_escalation_decision(
                NORM_OUTCOME_EXTREME, 1, moderate_shadow=True,
            ),
            ("EXTREME_REVOKED", 2, True, False),
        )

    def test_student_t_interval_is_used_for_five_runs(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected = 2.7764451051977987 * statistics.stdev(values) / len(values) ** 0.5
        self.assertAlmostEqual(t_ci95_or_blank(values), expected, places=10)
        self.assertEqual(t_ci95_or_blank([1.0]), "")

    def test_shadow_analyzer_separates_target_and_normal_persistence(self):
        common = {
            "dataset": "fashion_mnist",
            "variant": "progressive_trimmed_mean",
            "attack_scenario": "diverse_then_repeat_backdoor",
            "noniid_alpha": "0.1",
            "cohort_mode": "mixed_maturity",
            "repeat": "0",
            "norm_escalation_mode": "moderate_shadow",
        }
        nodes = [
            {
                **common, "node_id": "diverse-repeat-backdoor",
                "profile": "diverse_then_repeat_backdoor", "cohort_role": "",
                "initial_access_state": "LIMITED", "access_state": "ADMITTED",
            },
            {
                **common, "node_id": "honest-1", "profile": "honest",
                "cohort_role": "", "initial_access_state": "ADMITTED",
                "access_state": "ADMITTED",
            },
        ]
        events = []
        for round_number, outcome, strikes, ratio in (
            (1, "MODERATE_REJECTED", 1, 1.02),
            (2, "REPEATED_SHADOW_REJECTED", 2, 1.03),
            (3, "REPEATED_SHADOW_REJECTED", 3, 1.04),
        ):
            events.append({
                **common, "round": str(round_number),
                "node_id": "diverse-repeat-backdoor", "returned": "1",
                "norm_screening_outcome": outcome,
                "norm_strike_count": str(strikes), "norm_ratio": str(ratio),
            })
        events.append({
            **common, "round": "1", "node_id": "honest-1", "returned": "1",
            "norm_screening_outcome": "NORMAL", "norm_strike_count": "0",
            "norm_ratio": "0.4",
        })
        rows = analyze_nodes(nodes, events)
        target = next(row for row in rows if row["evaluation_role"] == "target")
        normal = next(row for row in rows if row["evaluation_role"] == "normal")
        self.assertEqual(target["cohort_role"], "newcomer")
        self.assertEqual(normal["cohort_role"], "incumbent")
        self.assertEqual(target["max_consecutive_moderate_excess"], 3)
        self.assertEqual(target["repeated_shadow_rejections"], 2)
        self.assertEqual(normal["moderate_excess_updates"], 0)
        summary = summarize(rows)
        target_summary = next(
            row for row in summary if row["evaluation_role"] == "target"
        )
        self.assertEqual(target_summary["nodes_with_repeat_shadow"], 1)
        self.assertEqual(target_summary["repeat_shadow_node_rate"], 1.0)

    def test_v38_wrapper_is_narrow_and_uses_module_analyzers(self):
        wrapper = (
            ROOT / "flower_prototype" / "run_norm_shadow_v38.ps1"
        ).read_text(encoding="utf-8")
        for token in (
            'NormEscalationMode = "moderate_shadow"',
            'NonIIDAlphas = @(0.1)',
            '"progressive_full", "progressive_trimmed_mean"',
            'python -m flower_prototype.analyze_hybrid_v38',
            'python -m flower_prototype.analyze_norm_shadow_v38',
            'norm_shadow_v38_summary.csv',
        ):
            self.assertIn(token, wrapper)
        v37_wrapper = (
            ROOT / "flower_prototype" / "run_hybrid_aggregation_v37.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "python -m flower_prototype.analyze_hybrid_v37", v37_wrapper,
        )


if __name__ == "__main__":
    unittest.main()

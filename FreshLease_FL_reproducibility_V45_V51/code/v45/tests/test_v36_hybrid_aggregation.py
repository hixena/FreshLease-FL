from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flower_prototype.access_control import AccessLedger
from flower_prototype.variant_policy import (
    default_aggregation_rule, is_progressive_variant, uses_cumulative_risk,
)


ROOT = Path(__file__).resolve().parents[1]


class V36HybridAggregationTests(unittest.TestCase):
    def test_hybrid_variants_compose_progressive_access_and_robust_aggregation(self):
        expectations = {
            "progressive_fltrust": "fltrust",
            "progressive_trimmed_mean": "trimmed_mean",
        }
        for variant, rule in expectations.items():
            with self.subTest(variant=variant):
                self.assertTrue(is_progressive_variant(variant))
                self.assertTrue(uses_cumulative_risk(variant))
                self.assertEqual(default_aggregation_rule(variant), rule)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    ledger = AccessLedger(
                        root / "db.sqlite", root / "server.pem",
                        mechanism_variant=variant,
                    )
                    ledger.connection.close()

    def test_v36_wrapper_freezes_policy_and_names_outputs(self):
        wrapper = (ROOT / "flower_prototype" / "run_hybrid_aggregation_v36.ps1").read_text(
            encoding="utf-8",
        )
        for token in (
            'LimitedCleanUpdates = 3',
            'LimitedAggregationWeight = 0.75',
            'CumulativeRiskDecay = 0.50',
            'CumulativeRiskThreshold = 0.72',
            'SelfReversalGate = 0.20',
            '"progressive_fltrust"',
            '"progressive_trimmed_mean"',
            'hybrid_aggregation_v36_summary.csv',
        ):
            self.assertIn(token, wrapper)

    def test_matrix_maps_hybrid_variants_to_robust_rules(self):
        runner = (ROOT / "flower_prototype" / "run_real_fl_matrix.ps1").read_text(
            encoding="utf-8",
        )
        self.assertIn('"progressive_fltrust" { "fltrust" }', runner)
        self.assertIn('"progressive_trimmed_mean" { "trimmed_mean" }', runner)


if __name__ == "__main__":
    unittest.main()

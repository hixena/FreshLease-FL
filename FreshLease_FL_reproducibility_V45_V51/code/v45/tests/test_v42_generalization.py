from __future__ import annotations

import unittest
from pathlib import Path

from flower_prototype.analyze_generalization_v42 import pairs


def result(variant: str, client_count: int, accuracy: float, asr: float) -> dict:
    return {
        "dataset": "fashion_mnist",
        "client_count": str(client_count),
        "attack_scenario": "diverse_then_repeat_backdoor",
        "noniid_alpha": "0.5",
        "repeat": "0",
        "variant": variant,
        "limited_observation_updates": "5",
        "limited_aggregation_weight": "0.5" if variant == "access_full" else "1.0",
        "target_aggregation_weight_after_start": "5.5" if variant == "access_full" else "7.0",
        "target_effective_aggregation_share_after_start": (
            "0.0802919708" if client_count == 10 and variant == "access_full"
            else "0.1" if client_count == 10
            else "0.0398550725" if variant == "access_full"
            else "0.05"
        ),
        "target_limited_state_rounds_after_start": "4" if variant == "access_full" else "0",
        "target_limited_weighted_updates_after_start": "3" if variant == "access_full" else "0",
        "target_promotion_round": "5" if variant == "access_full" else "",
        "final_accuracy": str(accuracy),
        "final_backdoor_asr": str(asr),
    }


class V42AnalysisTests(unittest.TestCase):
    def test_pairs_are_isolated_by_client_count(self):
        rows = [
            result("access_full", 10, 0.80, 0.20),
            result("access_no_limited", 10, 0.81, 0.30),
            result("access_full", 20, 0.79, 0.10),
            result("access_no_limited", 20, 0.80, 0.15),
        ]
        paired = pairs(rows)
        self.assertEqual({row["client_count"] for row in paired}, {"10", "20"})
        reductions = {row["client_count"]: row["backdoor_asr_reduction"] for row in paired}
        self.assertAlmostEqual(reductions["10"], 0.10)
        self.assertAlmostEqual(reductions["20"], 0.05)


class V42RunnerTests(unittest.TestCase):
    def test_runner_freezes_v41_workpoint_and_declares_two_stages(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (root / "flower_prototype" / "run_generalization_v42.ps1").read_text()
        runner = (root / "flower_prototype" / "run_real_fl_matrix.ps1").read_text()
        compose = (root / "flower_prototype" / "docker-compose.evidence-farming.yml").read_text()
        self.assertIn('"dataset_noniid"', wrapper)
        self.assertIn('"client_scale"', wrapper)
        self.assertIn("LimitedObservationUpdates = 5", wrapper)
        self.assertIn("Variant = \"access_full\"; Weight = 0.50", wrapper)
        self.assertIn("Variant = \"access_no_limited\"; Weight = 1.0", wrapper)
        self.assertIn("ClientCount must be between 4 and 20", runner)
        # V43 additionally writes one configuration-level overhead ledger row.
        self.assertEqual(runner.count("client_count = $ClientCount"), 4)
        self.assertIn("honest-19:", compose)
        self.assertIn("node-access-flower-prototype:0.23", compose)


if __name__ == "__main__":
    unittest.main()

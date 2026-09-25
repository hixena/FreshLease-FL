from __future__ import annotations

import unittest

from flower_prototype.analyze_attack_robustness import summarize


def row(variant: str, repeat: int, weight: float, blocked: int, revoked: int) -> dict:
    return {
        "variant": variant,
        "noniid_alpha": "0.5",
        "repeat": str(repeat),
        "attack_start_round": "2",
        "gradual_sign_flip_scale": "0.35",
        "first_attack_returned": 1,
        "first_attack_blocked": blocked,
        "attack_revoked": revoked,
        "detection_delay_rounds": "0" if revoked else "",
        "attack_updates_after_start": 0 if blocked else 9,
        "attack_weight_after_start": weight,
        "normal_false_revocations": 0,
        "final_accuracy": 0.8,
        "final_backdoor_asr": 0.01,
    }


class AttackRobustnessTests(unittest.TestCase):
    def test_summary_uses_matched_full_exposure(self):
        rows = []
        for repeat in range(3):
            rows.extend([
                row("full", repeat, 9.0, 0, 0),
                row("progressive_full", repeat, 0.25, 1, 1),
                row("progressive_no_cumulative", repeat, 8.75, 0, 0),
            ])
        summary = summarize(rows)
        protected = next(
            item for item in summary if item["variant"] == "progressive_full"
        )
        self.assertEqual(protected["first_attack_block_rate"], 1.0)
        self.assertEqual(protected["attack_revocation_rate"], 1.0)
        self.assertAlmostEqual(
            protected["attack_weight_reduction_vs_full_mean"], 1 - 0.25 / 9.0
        )
        self.assertEqual(protected["normal_false_revocation_rate"], 0.0)

    def test_missing_matched_full_baseline_is_rejected(self):
        rows = [row("full", 0, 9.0, 0, 0), row("progressive_full", 1, 0.0, 1, 1)]
        with self.assertRaisesRegex(ValueError, "missing matched full baseline"):
            summarize(rows)


if __name__ == "__main__":
    unittest.main()

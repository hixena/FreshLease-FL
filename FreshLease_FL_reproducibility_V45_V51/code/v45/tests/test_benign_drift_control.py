from __future__ import annotations

import unittest

from flower_prototype.analyze_benign_drift_control import analyze_runs, summarize


def fixture(revoked: bool = False):
    nodes, events, metrics = [], [], []
    variants = (
        "progressive_full", "progressive_no_cumulative",
        "full", "no_online_revalidation",
    )
    for variant in variants:
        base = {
            "variant": variant, "attack_scenario": "benign_concept_drift",
            "noniid_alpha": "0.5", "repeat": "0",
        }
        target = {
            **base, "node_id": "benign-concept-drift",
            "profile": "benign_concept_drift",
            "initial_access_state": "LIMITED" if variant.startswith("progressive") else "ADMITTED",
            "revoked_round": "3" if revoked and variant == "progressive_full" else "",
            "revocation_reason": "CUMULATIVE_DIRECTIONAL_RISK" if revoked and variant == "progressive_full" else "",
        }
        nodes.append(target)
        for index in range(3):
            nodes.append({
                **base, "node_id": f"honest-{index}", "profile": "honest",
                "initial_access_state": target["initial_access_state"],
                "revoked_round": "", "revocation_reason": "",
            })
        for round_number in range(1, 5):
            metrics.append({
                **base, "round": str(round_number), "accuracy": "0.7",
                "backdoor_asr": "0.0",
            })
            for node in nodes[-4:]:
                is_revoked_target = (
                    node["node_id"] == "benign-concept-drift"
                    and revoked and variant == "progressive_full"
                    and round_number >= 3
                )
                events.append({
                    **base, "round": str(round_number), "node_id": node["node_id"],
                    "returned": "1", "aggregated": "0" if is_revoked_target else "1",
                    "aggregation_weight": "0.25" if variant.startswith("progressive") else "1.0",
                })
    return nodes, events, metrics


class BenignDriftControlTests(unittest.TestCase):
    def test_clean_drift_run_reports_no_false_revocation(self):
        runs = analyze_runs(*fixture(), 2)
        self.assertEqual(len(runs), 4)
        progressive = next(row for row in runs if row["variant"] == "progressive_full")
        self.assertEqual(progressive["target_false_revoked"], 0)
        self.assertEqual(progressive["normal_false_revoked"], 0)
        self.assertEqual(progressive["target_updates_after_drift"], 3)
        self.assertEqual(progressive["target_weight_after_drift"], 0.75)
        summary = summarize(runs)
        self.assertEqual(len(summary), 4)
        self.assertEqual(summary[0]["independent_runs"], 1)

    def test_target_revocation_is_explicitly_counted_as_false_positive(self):
        runs = analyze_runs(*fixture(revoked=True), 2)
        progressive = next(row for row in runs if row["variant"] == "progressive_full")
        self.assertEqual(progressive["target_false_revoked"], 1)
        self.assertEqual(progressive["revoked_round"], "3")


if __name__ == "__main__":
    unittest.main()

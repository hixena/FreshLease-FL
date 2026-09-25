from __future__ import annotations

import unittest

from flower_prototype.analyze_paper_baselines import analyze_runs, summarize


class MultiDatasetBaselineTests(unittest.TestCase):
    @staticmethod
    def _rows(dataset: str):
        common = {
            "dataset": dataset,
            "variant": "progressive_full",
            "attack_scenario": "gradual_drift_betrayal",
            "noniid_alpha": "0.5",
            "repeat": "0",
        }
        nodes = [
            {
                **common, "node_id": "gradual-drift-betrayal",
                "profile": "gradual_drift_betrayal",
                "initial_access_state": "LIMITED", "revoked_round": "3",
                "revocation_reason": "CUMULATIVE_DIRECTIONAL_RISK",
            },
            *[
                {
                    **common, "node_id": f"honest-{index}", "profile": "honest",
                    "initial_access_state": "LIMITED", "revoked_round": "",
                    "revocation_reason": "",
                }
                for index in range(1, 4)
            ],
        ]
        events = [
            {
                **common, "round": "2", "node_id": "gradual-drift-betrayal",
                "returned": "1", "aggregated": "1", "aggregation_weight": "0.75",
                "aggregation_effective_weight": "0.15",
            },
            {
                **common, "round": "3", "node_id": "gradual-drift-betrayal",
                "returned": "1", "aggregated": "0", "aggregation_weight": "",
            },
            {
                **common, "round": "2", "node_id": "honest-1",
                "returned": "1", "aggregated": "0", "aggregation_weight": "",
                "reason": "EXCESS_UPDATE_NORM_REJECTED", "validation_loss_flag": "1",
                "norm_screening_outcome": "MODERATE_REJECTED",
            },
            {
                **common, "round": "2", "node_id": "honest-2",
                "returned": "1", "aggregated": "1", "aggregation_weight": "0.75",
                "reason": "LIMITED_WEIGHTED",
            },
        ]
        metrics = [{**common, "round": "3", "accuracy": "0.8", "backdoor_asr": "0.1"}]
        return nodes, events, metrics

    def test_analyzer_keeps_dataset_and_counts_weight(self):
        nodes, events, metrics = self._rows("fashion_mnist")
        runs = analyze_runs(nodes, events, metrics, 2)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["dataset"], "fashion_mnist")
        self.assertEqual(runs[0]["target_updates_after_start"], 1)
        self.assertEqual(runs[0]["target_weight_after_start"], 0.75)
        self.assertEqual(
            runs[0]["target_effective_aggregation_share_after_start"], 0.15,
        )
        self.assertEqual(runs[0]["detection_delay_rounds"], 1)
        summary = summarize(runs)
        self.assertEqual(summary[0]["target_revocation_rate"], 1.0)
        self.assertEqual(summary[0]["normal_false_revocation_rate"], 0.0)
        self.assertEqual(summary[0]["normal_validation_loss_flagged_node_rate"], 1 / 3)
        self.assertEqual(summary[0]["normal_validation_loss_flagged_update_rate"], 0.5)
        self.assertEqual(runs[0]["normal_norm_flagged_nodes"], 1)
        self.assertEqual(runs[0]["normal_norm_rejected_updates"], 1)
        self.assertEqual(runs[0]["normal_norm_rejected_update_rate"], 0.5)
        self.assertEqual(runs[0]["normal_norm_hard_revocations"], 0)
        self.assertEqual(runs[0]["normal_norm_hard_revocation_rate"], 0.0)
        self.assertEqual(summary[0]["normal_norm_flagged_node_rate"], 1 / 3)
        self.assertEqual(summary[0]["normal_norm_rejected_update_rate"], 0.5)
        self.assertEqual(summary[0]["normal_norm_hard_revocation_rate"], 0.0)
        self.assertEqual(
            summary[0]["target_effective_aggregation_share_after_start_mean"], 0.15,
        )

    def test_analyzer_recovers_legacy_fltrust_effective_share(self):
        nodes, events, metrics = self._rows("fashion_mnist")
        for row in nodes + events + metrics:
            row["variant"] = "fltrust"
        events[0].pop("aggregation_effective_weight")
        events[0].update({
            "aggregation_rule": "fltrust",
            "aggregation_trust_score": "0.0",
        })
        events[3].update({
            "aggregation_rule": "fltrust",
            "aggregation_trust_score": "0.8",
        })
        runs = analyze_runs(nodes, events, metrics, 2)
        self.assertEqual(runs[0]["target_effective_aggregation_share_after_start"], 0.0)
        self.assertEqual(runs[0]["target_fltrust_trust_score_mean"], 0.0)
        self.assertEqual(runs[0]["target_fltrust_zero_trust_updates"], 1)
        summary = summarize(runs)
        self.assertEqual(summary[0]["target_fltrust_zero_trust_update_rate"], 1.0)

    def test_analyzer_reports_trimmed_coordinate_retention(self):
        nodes, events, metrics = self._rows("fashion_mnist")
        for row in nodes + events + metrics:
            row["variant"] = "trimmed_mean"
        events[0].update({
            "aggregation_rule": "trimmed_mean",
            "aggregation_coordinate_retention_rate": "0.25",
        })
        runs = analyze_runs(nodes, events, metrics, 2)
        self.assertEqual(runs[0]["target_trimmed_coordinate_retention_mean"], 0.25)

    def test_analyzer_rejects_mismatched_dataset_keys(self):
        nodes, events, metrics = self._rows("digits")
        metrics[0]["dataset"] = "fashion_mnist"
        with self.assertRaisesRegex(ValueError, "keys differ"):
            analyze_runs(nodes, events, metrics, 2)

    def test_analyzer_supports_nine_honest_nodes(self):
        nodes, events, metrics = self._rows("digits")
        common = {
            key: nodes[0][key]
            for key in ("dataset", "variant", "attack_scenario", "noniid_alpha", "repeat")
        }
        for index in range(4, 10):
            nodes.append({
                **common, "node_id": f"honest-{index}", "profile": "honest",
                "initial_access_state": "LIMITED", "revoked_round": "",
                "revocation_reason": "",
            })
        runs = analyze_runs(nodes, events, metrics, 2)
        self.assertEqual(runs[0]["normal_node_records"], 9)
        summary = summarize(runs)
        self.assertEqual(summary[0]["normal_node_records"], 9)

    def test_analyzer_supports_delayed_backdoor_target(self):
        nodes, events, metrics = self._rows("fashion_mnist")
        for row in nodes + events + metrics:
            row["attack_scenario"] = "diverse_then_repeat_backdoor"
        nodes[0]["node_id"] = "diverse-repeat-backdoor"
        nodes[0]["profile"] = "diverse_then_repeat_backdoor"
        events[0]["node_id"] = "diverse-repeat-backdoor"
        events[1]["node_id"] = "diverse-repeat-backdoor"
        runs = analyze_runs(nodes, events, metrics, 2)
        self.assertEqual(runs[0]["target_profile"], "diverse_then_repeat_backdoor")
        self.assertEqual(runs[0]["final_backdoor_asr"], "0.1")


if __name__ == "__main__":
    unittest.main()

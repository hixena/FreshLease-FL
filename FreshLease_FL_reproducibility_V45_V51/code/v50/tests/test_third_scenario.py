from __future__ import annotations

import unittest

from flower_prototype.analyze_third_scenario import analyze_runs, pair_runs
from flower_prototype.analyze_progressive_third_scenario import comparison_rows
from flower_prototype.attack_schedule import effective_training_attack


def fixture(attack_aggregated: int = 0, target_returned: bool = True):
    nodes, events, metrics = [], [], []
    for variant in ("full", "no_online_revalidation"):
        base = dict(variant=variant, attack_scenario="diverse_then_repeat_farming",
                    noniid_alpha="0.5", repeat="0")
        target = dict(base, node_id="diverse-repeat-farming",
                      profile="diverse_then_repeat_farming",
                      initial_access_state="ADMITTED",
                      completed_tasks="9", credited_independent_tasks="9",
                      revoked_round="2" if variant == "full" else "",
                      aggregated_fit_events=str(1 + (attack_aggregated if variant == "full" else 2)))
        nodes.append(target)
        for idx in range(3):
            nodes.append(dict(base, node_id=f"honest-{idx}", profile="honest",
                              initial_access_state="ADMITTED", revoked_round="",
                              aggregated_fit_events="3"))
        for round_number in (1, 2, 3):
            metrics.append(dict(base, round=str(round_number), accuracy="0.5",
                                backdoor_asr="0.1"))
            for node in nodes[-4:]:
                is_target = node["node_id"] == "diverse-repeat-farming"
                if is_target and variant == "full" and round_number == 3:
                    continue
                returned = not (is_target and round_number == 2 and not target_returned)
                events.append(dict(base, round=str(round_number), node_id=node["node_id"],
                                   returned=str(int(returned)),
                                   signature_verified=str(int(returned)),
                                   aggregated=str(int(returned and (
                                       not is_target or round_number == 1 or
                                       (variant != "full" or attack_aggregated)))),
                                   reason="VALIDATION_LOSS" if (is_target and round_number == 2
                                                                and variant == "full" and returned) else ""))
    return nodes, events, metrics


class ThirdScenarioTests(unittest.TestCase):
    def test_delayed_attack_starts_after_honest_round(self):
        self.assertEqual(effective_training_attack("sign_flip", 1, 2), "none")
        self.assertEqual(effective_training_attack("sign_flip", 2, 2), "sign_flip")
        self.assertEqual(effective_training_attack("backdoor", 3, 2), "backdoor")
        self.assertEqual(effective_training_attack("none", 2, 2), "none")
        with self.assertRaises(ValueError):
            effective_training_attack("backdoor", 0, 2)

    def test_stage_metrics_and_paired_comparison(self):
        runs = analyze_runs(*fixture(), 2)
        pairs = pair_runs(runs)
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertEqual(pair["full_initial_admitted"], 1)
        self.assertEqual(pair["control_initial_admitted"], 1)
        self.assertEqual(pair["full_first_attack_returned"], 1)
        self.assertEqual(pair["full_first_attack_aggregated"], 0)
        self.assertEqual(pair["control_first_attack_aggregated"], 1)
        self.assertEqual(pair["full_revoked_round"], "2")
        self.assertEqual(pair["full_normal_revoked"], 0)
        self.assertEqual(pair["control_normal_initial_admitted"], 3)

    def test_missing_attack_update_is_not_reported_as_blocked(self):
        runs = analyze_runs(*fixture(target_returned=False), 2)
        full = next(r for r in runs if r["variant"] == "full")
        self.assertEqual(full["first_attack_returned"], 0)
        self.assertEqual(full["first_attack_aggregated"], "")

    def test_aggregation_failures_and_unpaired_runs_are_rejected(self):
        nodes, events, metrics = fixture()
        events[0]["returned"] = "0"
        with self.assertRaisesRegex(ValueError, "aggregation recorded without returned"):
            analyze_runs(nodes, events, metrics, 2)
        with self.assertRaisesRegex(ValueError, "unpaired variants"):
            pair_runs(analyze_runs(*fixture(), 2)[:1])

    def test_progressive_comparison_requires_all_four_variants(self):
        runs = analyze_runs(*fixture(), 2)
        progressive = dict(runs[0])
        progressive.update({
            "variant": "progressive_full",
            "initial_access_state": "LIMITED",
            "initial_admitted": 0,
            "initial_limited": 1,
            "initial_training_eligible": 1,
            "normal_initial_admitted": 0,
            "normal_initial_limited": 3,
            "normal_initial_training_eligible": 3,
            "attack_aggregated_updates": 0,
        })
        no_cumulative = dict(progressive, variant="progressive_no_cumulative")
        compared = comparison_rows([*runs, progressive, no_cumulative])
        self.assertEqual([row["variant"] for row in compared], [
            "progressive_full", "progressive_no_cumulative",
            "full", "no_online_revalidation",
        ])
        self.assertEqual(compared[0]["initial_access_state"], "LIMITED")
        self.assertEqual(compared[0]["initial_training_eligible"], 1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            comparison_rows(runs)


if __name__ == "__main__":
    unittest.main()

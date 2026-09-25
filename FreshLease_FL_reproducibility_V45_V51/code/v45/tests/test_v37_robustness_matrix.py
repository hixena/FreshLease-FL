from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import (
    AccessLedger, challenge_answer, public_key_b64, sign_payload,
)
from flower_prototype.analyze_hybrid_v37 import (
    RUN_FIELDS, SUMMARY_FIELDS, analyze_runs, ci95_or_blank, summarize,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]


def complete_onboarding(
    ledger: AccessLedger, key: Ed25519PrivateKey, node_id: str,
) -> None:
    public_key = public_key_b64(key)
    registration = {
        "action": "REGISTER", "node_id": node_id, "profile": "honest",
        "public_key": public_key,
    }
    ledger.register_node(
        node_id, "honest", public_key,
        sign_payload(key, registration), now=0.0,
    )
    for index in range(9):
        at = index * 2.0 + 1.0
        assignment = ledger.issue_task(node_id, now=at)
        task = assignment["task"]
        receipt = {
            "action": "ACK", "task_id": task["task_id"],
            "assignment_hash": assignment["assignment_hash"],
        }
        ledger.acknowledge(
            node_id, task["task_id"], assignment["assignment_hash"],
            sign_payload(key, receipt), now=at + 0.1,
        )
        result = {
            "action": "RESULT", "task_id": task["task_id"],
            "quality": 0.95, "result_hash": f"result-{node_id}-{index}",
            "work_product": challenge_answer(task),
        }
        ledger.submit_result(
            node_id, task["task_id"], 0.95, result["result_hash"],
            sign_payload(key, result), result["work_product"], now=at + 0.2,
        )


class V37RobustnessMatrixTests(unittest.TestCase):
    def test_explicit_incumbent_is_admitted_without_profile_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = AccessLedger(
                root / "db.sqlite", root / "server.pem",
                mechanism_variant="progressive_trimmed_mean",
                min_completed_tasks=9, min_limited_clean_updates=3,
                limited_aggregation_weight=0.75,
                incumbent_node_ids={"incumbent"},
            )
            incumbent_key = Ed25519PrivateKey.generate()
            newcomer_key = Ed25519PrivateKey.generate()
            complete_onboarding(ledger, incumbent_key, "incumbent")
            complete_onboarding(ledger, newcomer_key, "newcomer")
            incumbent = ledger.node_status("incumbent", expire=False)
            newcomer = ledger.node_status("newcomer", expire=False)
            self.assertEqual(incumbent["access_state"], "ADMITTED")
            self.assertEqual(incumbent["cohort_role"], "incumbent")
            self.assertEqual(newcomer["access_state"], "LIMITED")
            self.assertEqual(newcomer["cohort_role"], "newcomer")
            ledger.connection.close()

    def test_wilson_bounds_are_asymmetric_at_boundary(self):
        lower, upper = wilson_interval(3, 3)
        self.assertAlmostEqual(lower, 0.4385, places=3)
        self.assertAlmostEqual(upper, 1.0, places=7)
        self.assertEqual(ci95_or_blank([1.0]), "")

    def test_v37_fields_disambiguate_returned_aggregated_and_rejected(self):
        for field in (
            "target_returned_updates_after_start",
            "target_aggregated_updates_after_start",
            "target_rejected_updates_after_start",
        ):
            self.assertIn(field, RUN_FIELDS)
        self.assertNotIn("target_updates_after_start", RUN_FIELDS)
        self.assertIn("target_revocation_wilson_ci95_lower", SUMMARY_FIELDS)
        self.assertIn("target_revocation_wilson_ci95_upper", SUMMARY_FIELDS)

    def test_v37_analyzer_counts_returned_aggregated_and_rejected_updates(self):
        common = {
            "dataset": "fashion_mnist",
            "variant": "progressive_trimmed_mean",
            "attack_scenario": "diverse_then_repeat_backdoor",
            "noniid_alpha": "0.5",
            "cohort_mode": "mixed_maturity",
            "repeat": "0",
        }
        nodes = []
        for node_id, profile, cohort_role in (
            ("honest-1", "honest", "incumbent"),
            ("honest-2", "honest", "incumbent"),
            ("honest-3", "honest", "incumbent"),
            ("diverse-repeat-backdoor", "diverse_then_repeat_backdoor", "newcomer"),
        ):
            nodes.append({
                **common, "node_id": node_id, "profile": profile,
                "cohort_role": cohort_role,
                "initial_access_state": "ADMITTED" if profile == "honest" else "LIMITED",
                "revoked_round": "3" if profile != "honest" else "",
                "revocation_reason": "EXCESS_UPDATE_NORM" if profile != "honest" else "",
            })
        events = []
        for round_number in (2, 3):
            for node in nodes:
                target = node["profile"] != "honest"
                aggregated = not (target and round_number == 3)
                events.append({
                    **common, "round": str(round_number),
                    "node_id": node["node_id"], "returned": "1",
                    "aggregated": "1" if aggregated else "0",
                    "aggregation_weight": "0.75" if target else "1.0",
                    "aggregation_rule": "trimmed_mean",
                    "aggregation_effective_weight": "0.08" if aggregated else "",
                    "aggregation_coordinate_retention_rate": "0.75" if aggregated else "",
                    "validation_loss_flag": "0",
                    "norm_screening_outcome": (
                        "REPEATED_REVOKED" if target and round_number == 3 else "NORMAL"
                    ),
                })
        metrics = [
            {**common, "round": "2", "accuracy": "0.80", "backdoor_asr": "0.03"},
            {**common, "round": "3", "accuracy": "0.82", "backdoor_asr": "0.02"},
        ]
        runs = analyze_runs(nodes, events, metrics, attack_start_round=2)
        self.assertEqual(runs[0]["target_returned_updates_after_start"], 2)
        self.assertEqual(runs[0]["target_aggregated_updates_after_start"], 1)
        self.assertEqual(runs[0]["target_rejected_updates_after_start"], 1)
        summary = summarize(runs)[0]
        self.assertEqual(summary["detection_delay_observed_runs"], 1)
        self.assertEqual(summary["final_accuracy_ci95"], "")
        self.assertAlmostEqual(summary["target_revocation_wilson_ci95_upper"], 1.0)

    def test_v37_wrapper_freezes_working_point_and_uses_separate_outputs(self):
        wrapper = (
            ROOT / "flower_prototype" / "run_hybrid_aggregation_v37.ps1"
        ).read_text(encoding="utf-8")
        for token in (
            'LimitedCleanUpdates = 3',
            'LimitedAggregationWeight = 0.75',
            'CumulativeRiskDecay = 0.50',
            'CumulativeRiskThreshold = 0.72',
            'SelfReversalGate = 0.20',
            '"mixed_maturity"',
            'hybrid_aggregation_v37_summary.csv',
        ):
            self.assertIn(token, wrapper)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import (
    AccessLedger, challenge_answer, public_key_b64, sign_payload,
)
from flower_prototype.analyze_mixed_maturity_v41 import analyze, pairs, summarize


def onboard(ledger: AccessLedger, node_id: str, key: Ed25519PrivateKey) -> None:
    public = public_key_b64(key)
    registration = {"action": "REGISTER", "node_id": node_id, "profile": "honest",
                    "public_key": public}
    ledger.register_node(node_id, "honest", public, sign_payload(key, registration), now=0)
    for index in range(9):
        now = 2 * index + 1.0
        assignment = ledger.issue_task(node_id, now=now); task = assignment["task"]
        receipt = {"action": "ACK", "task_id": task["task_id"],
                   "assignment_hash": assignment["assignment_hash"]}
        ledger.acknowledge(node_id, task["task_id"], assignment["assignment_hash"],
                           sign_payload(key, receipt), now=now + 0.1)
        result = {"action": "RESULT", "task_id": task["task_id"], "quality": 0.95,
                  "result_hash": f"{node_id}-{index}", "work_product": challenge_answer(task)}
        ledger.submit_result(node_id, task["task_id"], 0.95, result["result_hash"],
                             sign_payload(key, result), result["work_product"], now=now + 0.2)


def accept(ledger: AccessLedger, node_id: str, key: Ed25519PrivateKey, round_number: int):
    update_hash = f"{round_number:064x}"; parent_hash = f"{round_number + 100:064x}"
    payload = {"action": "FIT_COMMITMENT", "node_id": node_id,
               "server_round": round_number, "update_hash": update_hash,
               "parent_hash": parent_hash}
    ledger.verify_fit_commitment(node_id, round_number, update_hash, parent_hash,
                                 sign_payload(key, payload))
    return ledger.accept_fit_update(node_id, round_number, update_hash)


class V41MechanismTests(unittest.TestCase):
    def test_incumbent_is_admitted_while_newcomer_remains_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = AccessLedger(
                root / "db.sqlite", root / "server.pem",
                mechanism_variant="access_full", min_completed_tasks=9,
                min_limited_observation_updates=5,
                incumbent_node_ids={"honest-1"},
            )
            incumbent_key, newcomer_key = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
            onboard(ledger, "honest-1", incumbent_key)
            onboard(ledger, "newcomer", newcomer_key)
            self.assertEqual(ledger.node_status("honest-1", expire=False)["access_state"], "ADMITTED")
            self.assertEqual(ledger.node_status("newcomer", expire=False)["access_state"], "LIMITED")
            weights = [accept(ledger, "newcomer", newcomer_key, round_number)["aggregation_weight"]
                       for round_number in range(1, 6)]
            self.assertEqual(weights, [0.25, 0.25, 0.25, 0.25, 1.0])
            status = ledger.node_status("newcomer", expire=False)
            self.assertEqual(status["access_state"], "ADMITTED")
            self.assertEqual(status["structurally_accepted_observation_updates"], 5)
            self.assertEqual(status["limited_observation_updates_required"], 5)
            outcomes = [row[0] for row in ledger.connection.execute(
                "SELECT outcome FROM training_evidence WHERE node_id='newcomer' ORDER BY server_round"
            )]
            self.assertEqual(outcomes, ["STRUCTURALLY_ACCEPTED"] * 5)
            ledger.connection.close()


def base(variant: str, repeat: int) -> dict[str, str]:
    return {"dataset": "fashion_mnist", "cohort_mode": "mixed_maturity",
            "variant": variant, "trust_estimator": "dirichlet_lcb",
            "attack_scenario": "diverse_then_repeat_backdoor", "noniid_alpha": "0.5",
            "repeat": str(repeat), "attack_start_round": "2",
            "limited_observation_updates": "5",
            "limited_aggregation_weight": "0.5" if variant == "access_full" else "1",
            "max_probation_tasks": "20", "min_probation_completed_tasks": "9",
            "min_evidence_mass": "1.5"}


class V41AnalysisTests(unittest.TestCase):
    def test_relative_share_and_promotion_round_are_reported(self):
        nodes, events, metrics = [], [], []
        for variant in ("access_full", "access_no_limited"):
            for repeat in range(2):
                common = base(variant, repeat)
                nodes += [common | {"node_id": "diverse-repeat-backdoor",
                                    "profile": "diverse_then_repeat_backdoor", "cohort_role": "newcomer",
                                    "initial_access_state": "LIMITED" if variant == "access_full" else "ADMITTED",
                                    "access_state": "ADMITTED", "completed_tasks": "9", "evidence_types": "4",
                                    "evidence_mass": "4", "history_evidence_maturity": "0.2"},
                          common | {"node_id": "honest-1", "profile": "honest", "cohort_role": "incumbent",
                                    "initial_access_state": "ADMITTED", "access_state": "ADMITTED",
                                    "completed_tasks": "9", "evidence_types": "4", "evidence_mass": "4",
                                    "history_evidence_maturity": "0.2"}]
                for round_number in (2, 3):
                    limited = variant == "access_full" and round_number == 2
                    events += [common | {"round": str(round_number), "node_id": "diverse-repeat-backdoor",
                                         "returned": "1", "aggregated": "1",
                                         "aggregation_weight": "0.5" if limited else "1",
                                         "access_state_before": "LIMITED" if limited else "ADMITTED",
                                         "access_state_after": "ADMITTED"},
                               common | {"round": str(round_number), "node_id": "honest-1",
                                         "returned": "1", "aggregated": "1", "aggregation_weight": "1",
                                         "access_state_before": "ADMITTED", "access_state_after": "ADMITTED"}]
                metrics.append(common | {"round": "3", "accuracy": "0.8",
                                         "backdoor_asr": "0.1" if variant == "access_full" else "0.4"})
        runs = analyze("primary", nodes, events, metrics)
        full = next(row for row in runs if row["variant"] == "access_full")
        self.assertAlmostEqual(full["target_effective_aggregation_share_after_start"], 1.5 / 3.5)
        self.assertEqual(full["target_vs_normal_mean_aggregation_weight_ratio"], 0.75)
        self.assertEqual(full["target_limited_state_rounds_after_start"], 1)
        self.assertEqual(full["target_limited_weighted_updates_after_start"], 1)
        self.assertEqual(full["target_promotion_round"], 2)
        self.assertEqual(len(summarize(runs)), 2)
        paired = pairs(runs)
        self.assertEqual(len(paired), 2)
        self.assertGreater(paired[0]["target_share_reduction"], 0)

    def test_runner_declares_mixed_maturity_and_progress(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (root / "flower_prototype" / "run_mixed_maturity_v41.ps1").read_text()
        runner = (root / "flower_prototype" / "run_real_fl_matrix.ps1").read_text()
        self.assertIn('CohortMode = "mixed_maturity"', wrapper)
        self.assertIn("LimitedObservationUpdates = $setting.ObservationUpdates", wrapper)
        self.assertIn("Write-Progress", runner)
        self.assertIn("ProgressStartedAt", runner)
        self.assertNotIn("$ProgressTotal:", runner)


if __name__ == "__main__":
    unittest.main()

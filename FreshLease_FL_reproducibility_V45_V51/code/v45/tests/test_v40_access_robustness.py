from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import (
    AccessLedger, challenge_answer, public_key_b64, sign_payload,
)
from flower_prototype.analyze_access_v40 import analyze, pairs, summarize
from flower_prototype.variant_policy import is_access_core_variant


def onboard(root: Path, variant: str, profile: str, skip_shadow: bool = False) -> dict:
    ledger = AccessLedger(
        root / f"{variant}-{profile}.sqlite", root / f"{variant}-{profile}.pem",
        1.0, mechanism_variant=variant, max_probation_tasks=20,
        min_completed_tasks=6, min_evidence_mass=2.70,
        min_limited_clean_updates=3,
    )
    key = Ed25519PrivateKey.generate(); node_id = "candidate"
    registration = {"action": "REGISTER", "node_id": node_id, "profile": profile,
                    "public_key": public_key_b64(key)}
    ledger.register_node(node_id, profile, registration["public_key"],
                         sign_payload(key, registration), now=0.0)
    for index in range(20):
        now = index * 2.0 + 1.0
        assignment = ledger.issue_task(node_id, now=now)
        task = assignment.get("task")
        if task is None: break
        if skip_shadow and task["task_type"] == "shadow_update":
            ledger.expire_due_tasks(now=now + 1.1); continue
        receipt = {"action": "ACK", "task_id": task["task_id"],
                   "assignment_hash": assignment["assignment_hash"]}
        ledger.acknowledge(node_id, task["task_id"], assignment["assignment_hash"],
                           sign_payload(key, receipt), now=now + 0.1)
        result = {"action": "RESULT", "task_id": task["task_id"], "quality": 0.95,
                  "result_hash": f"result-{index}", "work_product": challenge_answer(task)}
        ledger.submit_result(node_id, task["task_id"], 0.95, result["result_hash"],
                             sign_payload(key, result), result["work_product"], now=now + 0.2)
    status = ledger.node_status(node_id, expire=False); ledger.connection.close()
    return status


class V40MechanismTests(unittest.TestCase):
    def test_repeat_decay_changes_the_access_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = onboard(root, "access_full", "three_type_repeat_farming", True)
            ablated = onboard(root, "access_no_repeat_decay", "three_type_repeat_farming", True)
            honest = onboard(root, "access_full", "honest")
        self.assertEqual(full["access_state"], "QUARANTINE")
        self.assertEqual(ablated["access_state"], "LIMITED")
        self.assertEqual(honest["access_state"], "LIMITED")
        self.assertLess(full["evidence_mass"], 2.70)
        self.assertGreaterEqual(ablated["evidence_mass"], 2.70)

    def test_simple_baselines_are_explicit_and_distinct(self):
        self.assertTrue(is_access_core_variant("access_attestation_only"))
        self.assertTrue(is_access_core_variant("access_static_multisource"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attestation = onboard(root, "access_attestation_only", "attestation_camouflage")
            static = onboard(root, "access_static_multisource", "honest")
        self.assertEqual(attestation["access_state"], "ADMITTED")
        self.assertEqual(attestation["completed_tasks"], 1)
        self.assertEqual(static["access_state"], "ADMITTED")
        self.assertEqual(static["evidence_types"], 4)

    def test_invalid_evidence_mass_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "min_evidence_mass"):
                AccessLedger(root / "bad.sqlite", root / "bad.pem", min_evidence_mass=0)


def row_base(variant: str, repeat: int) -> dict[str, str]:
    return {"dataset": "fashion_mnist", "cohort_mode": "synchronous_cold_start",
            "variant": variant, "trust_estimator": "dirichlet_lcb",
            "attack_scenario": "diverse_then_repeat_backdoor", "noniid_alpha": "0.5",
            "repeat": str(repeat), "attack_start_round": "2", "limited_clean_updates": "3",
            "limited_aggregation_weight": "0.5" if variant == "access_full" else "1.0",
            "max_probation_tasks": "20", "min_probation_completed_tasks": "9",
            "min_evidence_mass": "1.5"}


class V40AnalysisTests(unittest.TestCase):
    def test_progressive_pair_reports_exposure_and_asr_reduction(self):
        nodes, events, metrics = [], [], []
        for variant in ("access_full", "access_no_limited"):
            for repeat in range(2):
                base = row_base(variant, repeat)
                nodes += [base | {"node_id": "diverse-repeat-backdoor", "profile": "diverse_then_repeat_backdoor",
                                  "initial_access_state": "LIMITED" if variant == "access_full" else "ADMITTED",
                                  "access_state": "ADMITTED", "completed_tasks": "9", "evidence_types": "4",
                                  "evidence_mass": "4", "history_evidence_maturity": "0.2"},
                          base | {"node_id": "honest-1", "profile": "honest", "initial_access_state": "LIMITED",
                                  "access_state": "ADMITTED", "completed_tasks": "9", "evidence_types": "4",
                                  "evidence_mass": "4", "history_evidence_maturity": "0.2"}]
                weight = 0.5 if variant == "access_full" else 1.0
                for round_number in (2, 3):
                    events += [base | {"round": str(round_number), "node_id": "diverse-repeat-backdoor",
                                       "returned": "1", "aggregated": "1", "aggregation_weight": str(weight)},
                               base | {"round": str(round_number), "node_id": "honest-1", "returned": "1",
                                       "aggregated": "1", "aggregation_weight": "1"}]
                metrics.append(base | {"round": "3", "accuracy": "0.8",
                                       "backdoor_asr": "0.1" if variant == "access_full" else "0.4"})
        runs = analyze("progressive", nodes, events, metrics)
        summary = summarize(runs)
        paired = pairs(runs)
        self.assertEqual(len(summary), 2)
        self.assertEqual(len(paired), 2)
        self.assertEqual(paired[0]["target_exposure_reduction"], 1.0)
        self.assertAlmostEqual(paired[0]["backdoor_asr_reduction"], 0.3)


if __name__ == "__main__":
    unittest.main()

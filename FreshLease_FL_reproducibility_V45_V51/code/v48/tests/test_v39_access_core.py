from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import (
    AccessLedger,
    challenge_answer,
    public_key_b64,
    sign_payload,
)
from flower_prototype.analyze_access_v39 import analyze_runs, summarize
from flower_prototype.variant_policy import (
    is_access_core_variant,
    is_progressive_variant,
    uses_evidence_diversity,
    uses_repeat_decay,
    uses_result_verification,
)


def complete_onboarding(root: Path, variant: str, profile: str) -> dict:
    ledger = AccessLedger(
        root / f"{variant}-{profile}.sqlite",
        root / f"{variant}-{profile}.pem",
        1.0,
        mechanism_variant=variant,
        max_probation_tasks=12,
        min_completed_tasks=5,
        min_limited_clean_updates=3,
    )
    key = Ed25519PrivateKey.generate()
    node_id = "candidate"
    registration = {
        "action": "REGISTER", "node_id": node_id, "profile": profile,
        "public_key": public_key_b64(key),
    }
    ledger.register_node(
        node_id, profile, registration["public_key"],
        sign_payload(key, registration), now=0.0,
    )
    for index in range(12):
        now = 2.0 * index + 1.0
        assignment = ledger.issue_task(node_id, now=now)
        task = assignment.get("task")
        if task is None:
            break
        if profile == "single_type_farming" and task["task_type"] != "shadow_update":
            ledger.expire_due_tasks(now=now + 1.1)
            continue
        receipt = {
            "action": "ACK", "task_id": task["task_id"],
            "assignment_hash": assignment["assignment_hash"],
        }
        ledger.acknowledge(
            node_id, task["task_id"], assignment["assignment_hash"],
            sign_payload(key, receipt), now=now + 0.1,
        )
        work_product = challenge_answer(task)
        if (
            profile == "false_quality_reporting"
            or (
                profile == "attestation_camouflage"
                and task["task_type"] != "attestation"
            )
        ):
            work_product = "invalid:" + work_product
        result = {
            "action": "RESULT", "task_id": task["task_id"],
            "quality": 0.99, "result_hash": f"result-{index}",
            "work_product": work_product,
        }
        ledger.submit_result(
            node_id, task["task_id"], 0.99, result["result_hash"],
            sign_payload(key, result), work_product, now=now + 0.2,
        )
    status = ledger.node_status(node_id, expire=False)
    ledger.connection.close()
    return status


class AccessCoreMechanismTests(unittest.TestCase):
    def test_policy_composition_is_explicit(self):
        self.assertTrue(is_access_core_variant("access_full"))
        self.assertTrue(is_progressive_variant("access_full"))
        self.assertFalse(is_progressive_variant("access_no_limited"))
        self.assertFalse(uses_evidence_diversity("access_no_diversity"))
        self.assertFalse(uses_repeat_decay("access_no_repeat_decay"))
        self.assertFalse(uses_result_verification("access_no_result_verification"))
        self.assertFalse(uses_result_verification("access_naive"))

    def test_type_coverage_blocks_single_type_farming(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = complete_onboarding(root, "access_full", "single_type_farming")
            ablated = complete_onboarding(
                root, "access_no_diversity", "single_type_farming"
            )
        self.assertEqual(full["access_state"], "QUARANTINE")
        self.assertEqual(ablated["access_state"], "LIMITED")

    def test_result_verification_blocks_false_quality_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = complete_onboarding(
                root, "access_full", "false_quality_reporting"
            )
            ablated = complete_onboarding(
                root, "access_no_result_verification", "false_quality_reporting"
            )
            naive = complete_onboarding(
                root, "access_naive", "false_quality_reporting"
            )
        self.assertEqual(full["access_state"], "QUARANTINE")
        self.assertEqual(full["completed_tasks"], 0)
        self.assertEqual(ablated["access_state"], "LIMITED")
        self.assertEqual(naive["access_state"], "ADMITTED")

    def test_limited_is_a_distinct_access_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = complete_onboarding(root, "access_full", "honest")
            direct = complete_onboarding(root, "access_no_limited", "honest")
        self.assertEqual(staged["access_state"], "LIMITED")
        self.assertEqual(staged["structurally_accepted_training_updates"], 0)
        self.assertEqual(direct["access_state"], "ADMITTED")

    def test_camouflage_is_rejected_but_future_betrayal_is_a_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camouflage = complete_onboarding(
                root, "access_full", "attestation_camouflage"
            )
            future_betrayal = complete_onboarding(
                root, "access_full", "diverse_then_repeat_farming"
            )
        self.assertEqual(camouflage["access_state"], "QUARANTINE")
        self.assertEqual(future_betrayal["access_state"], "LIMITED")


def node_row(
    repeat: int, node_id: str, profile: str, state: str, completed: float
) -> dict[str, str]:
    return {
        "dataset": "fashion_mnist", "cohort_mode": "synchronous_cold_start",
        "variant": "access_full", "trust_estimator": "dirichlet_lcb",
        "attack_scenario": "single_type_farming", "noniid_alpha": "0.5",
        "repeat": str(repeat), "node_id": node_id, "profile": profile,
        "initial_access_state": state, "access_state": state,
        "completed_tasks": str(completed), "credited_independent_tasks": str(completed),
        "evidence_types": "1" if profile != "honest" else "4",
        "evidence_mass": "1.0", "trust_score": "0.7",
        "history_decision_score": "0.6", "history_std": "0.1",
        "history_evidence_maturity": "0.2",
        "onboarding_duration_seconds": str(10 + repeat),
        "structurally_accepted_training_updates": "0",
        "flower_fit_events": "0", "aggregated_fit_events": "0",
        "aggregated_weight_mass": "0",
    }


class AccessCoreAnalysisTests(unittest.TestCase):
    def test_runs_are_the_inference_unit_for_normal_node_rates(self):
        nodes = []
        for repeat in range(2):
            nodes.append(node_row(
                repeat, "single-type-farming", "single_type_farming",
                "QUARANTINE", 4,
            ))
            nodes.append(node_row(repeat, "honest-1", "honest", "LIMITED", 5))
            nodes.append(node_row(
                repeat, "honest-2", "honest",
                "LIMITED" if repeat == 0 else "PROBATION", 5,
            ))
        runs = analyze_runs(nodes)
        summary = summarize(runs)[0]
        self.assertEqual(len(runs), 2)
        self.assertEqual(summary["independent_runs"], 2)
        self.assertEqual(summary["normal_node_records"], 4)
        self.assertEqual(summary["target_correct_access_decision_rate"], 1.0)
        self.assertEqual(
            summary["normal_initial_training_eligible_rate_mean"], 0.75
        )
        self.assertNotEqual(
            summary["normal_initial_training_eligible_rate_ci95"], ""
        )


if __name__ == "__main__":
    unittest.main()

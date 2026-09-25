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


class AccessLedgerTests(unittest.TestCase):
    def test_diverse_repeat_candidate_can_graduate_without_profile_oracle(self):
        root = Path(self.temp.name)
        ledger = AccessLedger(
            root / "repeat.sqlite", root / "repeat.pem", 1.0,
            mechanism_variant="full", min_completed_tasks=9,
        )
        key = Ed25519PrivateKey.generate()
        registration = {
            "action": "REGISTER", "node_id": "repeat",
            "profile": "diverse_then_repeat_farming",
            "public_key": public_key_b64(key),
        }
        ledger.register_node(
            "repeat", registration["profile"], registration["public_key"],
            sign_payload(key, registration), now=0.0,
        )
        issued = []
        for i in range(12):
            at = i * 2.0 + 1.0
            assignment = ledger.issue_task("repeat", now=at)
            task = assignment["task"]
            if task is None:
                break
            issued.append(task["task_type"])
            receipt = {
                "action": "ACK", "task_id": task["task_id"],
                "assignment_hash": assignment["assignment_hash"],
            }
            ledger.acknowledge(
                "repeat", task["task_id"], assignment["assignment_hash"],
                sign_payload(key, receipt), now=at + 0.1,
            )
            result = {
                "action": "RESULT", "task_id": task["task_id"],
                "quality": 0.95, "result_hash": f"result-{i}",
                "work_product": challenge_answer(task),
            }
            ledger.submit_result(
                "repeat", task["task_id"], 0.95, result["result_hash"],
                sign_payload(key, result), result["work_product"], now=at + 0.2,
            )
        status = ledger.node_status("repeat", expire=False)
        self.assertEqual(status["access_state"], "ADMITTED")
        self.assertGreaterEqual(status["completed_tasks"], 9)
        self.assertEqual(len(set(issued)), 4)
        self.assertGreaterEqual(issued.count("shadow_update"), 6)
        ledger.connection.close()

    def test_longer_common_schedule_keeps_single_type_ablation_informative(self):
        root = Path(self.temp.name)
        decisions = {}
        for variant in ("full", "no_diversity"):
            ledger = AccessLedger(
                root / f"nine-{variant}.sqlite", root / f"nine-{variant}.pem",
                1.0, mechanism_variant=variant, min_completed_tasks=9,
                max_probation_tasks=20,
            )
            key = Ed25519PrivateKey.generate()
            public_key = public_key_b64(key)
            registration = {
                "action": "REGISTER", "node_id": "single",
                "profile": "single_type_farming", "public_key": public_key,
            }
            ledger.register_node(
                "single", "single_type_farming", public_key,
                sign_payload(key, registration), now=0.0,
            )
            for i in range(20):
                at = i * 2.0 + 1.0
                assignment = ledger.issue_task("single", now=at)
                task = assignment["task"]
                if task is None:
                    break
                if task["task_type"] != "shadow_update":
                    ledger.expire_due_tasks(now=at + 1.1)
                    continue
                receipt = {
                    "action": "ACK", "task_id": task["task_id"],
                    "assignment_hash": assignment["assignment_hash"],
                }
                ledger.acknowledge(
                    "single", task["task_id"], assignment["assignment_hash"],
                    sign_payload(key, receipt), now=at + 0.1,
                )
                result = {
                    "action": "RESULT", "task_id": task["task_id"],
                    "quality": 0.95, "result_hash": f"result-{i}",
                    "work_product": challenge_answer(task),
                }
                ledger.submit_result(
                    "single", task["task_id"], 0.95, result["result_hash"],
                    sign_payload(key, result), result["work_product"], now=at + 0.2,
                )
            decisions[variant] = ledger.node_status("single", expire=False)
            ledger.connection.close()
        self.assertEqual(decisions["full"]["access_state"], "QUARANTINE")
        self.assertEqual(decisions["no_diversity"]["access_state"], "ADMITTED")

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.ledger = AccessLedger(root / "ledger.sqlite", root / "server.pem", 1.0)
        self.key = Ed25519PrivateKey.generate()
        public_key = public_key_b64(self.key)
        registration = {
            "action": "REGISTER",
            "node_id": "node-1",
            "profile": "honest",
            "public_key": public_key,
        }
        self.ledger.register_node(
            "node-1", "honest", public_key,
            sign_payload(self.key, registration), now=0.0,
        )

    def tearDown(self) -> None:
        self.ledger.connection.close()
        self.temp.cleanup()

    def _ack(self, assignment: dict, now: float) -> None:
        task = assignment["task"]
        receipt = {
            "action": "ACK",
            "task_id": task["task_id"],
            "assignment_hash": assignment["assignment_hash"],
        }
        self.ledger.acknowledge(
            "node-1", task["task_id"], assignment["assignment_hash"],
            sign_payload(self.key, receipt), now=now,
        )

    def test_signed_assignment_receipt_result_and_hash_chain_verify(self):
        assignment = self.ledger.issue_task("node-1", now=1.0)
        self._ack(assignment, now=1.1)
        task_id = assignment["task"]["task_id"]
        result = {
            "action": "RESULT",
            "task_id": task_id,
            "quality": 0.9,
            "result_hash": "abc123",
            "work_product": challenge_answer(assignment["task"]),
        }
        self.ledger.submit_result(
            "node-1", task_id, 0.9, "abc123",
            sign_payload(self.key, result), result["work_product"], now=1.2,
        )
        verification = self.ledger.verify_integrity()
        self.assertTrue(verification["valid"])
        self.assertEqual(verification["signatures"]["checked_signatures"], 4)

    def test_forged_receipt_is_rejected(self):
        assignment = self.ledger.issue_task("node-1", now=1.0)
        forged = Ed25519PrivateKey.generate()
        receipt = {
            "action": "ACK",
            "task_id": assignment["task"]["task_id"],
            "assignment_hash": assignment["assignment_hash"],
        }
        with self.assertRaises(Exception):
            self.ledger.acknowledge(
                "node-1", receipt["task_id"], receipt["assignment_hash"],
                sign_payload(forged, receipt), now=1.1,
            )

    def test_only_acknowledged_timeout_counts_as_selective_missing(self):
        acknowledged = self.ledger.issue_task("node-1", now=1.0)
        self._ack(acknowledged, now=1.1)
        self.ledger.expire_due_tasks(now=2.1)

        unconfirmed = self.ledger.issue_task("node-1", now=3.0)
        self.assertIsNotNone(unconfirmed["task"])
        self.ledger.expire_due_tasks(now=4.1)

        status = self.ledger.node_status("node-1", expire=False)
        self.assertEqual(status["acknowledged_tasks"], 1)
        self.assertEqual(status["acknowledged_missing_results"], 1)
        self.assertEqual(status["acknowledged_missing_rate"], 1.0)

    def test_late_result_becomes_acknowledged_timeout(self):
        assignment = self.ledger.issue_task("node-1", now=1.0)
        self._ack(assignment, now=1.1)
        task_id = assignment["task"]["task_id"]
        result = {
            "action": "RESULT", "task_id": task_id,
            "quality": 0.9, "result_hash": "late",
            "work_product": challenge_answer(assignment["task"]),
        }
        with self.assertRaisesRegex(ValueError, "after its deadline"):
            self.ledger.submit_result(
                "node-1", task_id, 0.9, "late",
                sign_payload(self.key, result), result["work_product"], now=2.2,
            )
        status = self.ledger.node_status("node-1", expire=False)
        self.assertEqual(status["acknowledged_missing_results"], 1)

    def test_assignment_does_not_depend_on_self_reported_profile(self):
        root = Path(self.temp.name)
        ledger = AccessLedger(
            root / "profiles.sqlite", root / "profiles.pem", 1.0,
        )
        for node_id, profile in (("good", "honest"), ("bad", "false_quality_reporting")):
            key = Ed25519PrivateKey.generate()
            public_key = public_key_b64(key)
            registration = {
                "action": "REGISTER", "node_id": node_id,
                "profile": profile, "public_key": public_key,
            }
            ledger.register_node(
                node_id, profile, public_key,
                sign_payload(key, registration), now=0.0,
            )
        observed = []
        for index in range(4):
            observed.append(tuple(
                ledger.issue_task(node, now=2.0 * index + 1.0)["task"]["task_type"]
                for node in ("good", "bad")
            ))
        self.assertTrue(all(left == right for left, right in observed))
        self.assertEqual([left for left, _ in observed], [
            "attestation", "protocol_check", "canary_training", "shadow_update",
        ])
        ledger.connection.close()

    def test_signed_high_score_cannot_replace_failed_challenge(self):
        root = Path(self.temp.name)
        results = {}
        for variant in ("full", "no_result_verification"):
            ledger = AccessLedger(
                root / f"{variant}.sqlite", root / f"{variant}.pem",
                1.0, mechanism_variant=variant, max_probation_tasks=20,
            )
            key = Ed25519PrivateKey.generate()
            public_key = public_key_b64(key)
            registration = {
                "action": "REGISTER", "node_id": "forger",
                "profile": "false_quality_reporting", "public_key": public_key,
            }
            ledger.register_node(
                "forger", "false_quality_reporting", public_key,
                sign_payload(key, registration), now=0.0,
            )
            for index in range(20):
                assignment = ledger.issue_task("forger", now=index * 2.0 + 1.0)
                if assignment["task"] is None:
                    break
                task = assignment["task"]
                receipt = {
                    "action": "ACK", "task_id": task["task_id"],
                    "assignment_hash": assignment["assignment_hash"],
                }
                ledger.acknowledge(
                    "forger", task["task_id"], assignment["assignment_hash"],
                    sign_payload(key, receipt), now=index * 2.0 + 1.1,
                )
                result = {
                    "action": "RESULT", "task_id": task["task_id"],
                    "quality": 0.99, "result_hash": str(index),
                    "work_product": "wrong-challenge-answer",
                }
                ledger.submit_result(
                    "forger", task["task_id"], 0.99, str(index),
                    sign_payload(key, result), result["work_product"],
                    now=index * 2.0 + 1.2,
                )
            results[variant] = ledger.node_status("forger", expire=False)
            self.assertTrue(ledger.verify_integrity()["valid"])
            ledger.connection.close()
        self.assertEqual(results["full"]["access_state"], "QUARANTINE")
        self.assertEqual(results["full"]["completed_tasks"], 0)
        self.assertEqual(results["no_result_verification"]["access_state"], "ADMITTED")

    def test_budget_quarantines_mature_node_below_admission_threshold(self):
        root = Path(self.temp.name)
        ledger = AccessLedger(
            root / "budget.sqlite", root / "budget.pem", 1.0,
            mechanism_variant="full", max_probation_tasks=5,
        )
        key = Ed25519PrivateKey.generate()
        public_key = public_key_b64(key)
        registration = {
            "action": "REGISTER", "node_id": "camouflage",
            "profile": "attestation_camouflage", "public_key": public_key,
        }
        ledger.register_node(
            "camouflage", "attestation_camouflage", public_key,
            sign_payload(key, registration), now=0.0,
        )
        for index in range(5):
            assignment = ledger.issue_task("camouflage", now=2.0 * index + 1.0)
            task = assignment["task"]
            receipt = {
                "action": "ACK", "task_id": task["task_id"],
                "assignment_hash": assignment["assignment_hash"],
            }
            ledger.acknowledge(
                "camouflage", task["task_id"], assignment["assignment_hash"],
                sign_payload(key, receipt), now=2.0 * index + 1.1,
            )
            quality = 0.97 if task["task_type"] == "attestation" else 0.15
            work_product = (
                challenge_answer(task) if task["task_type"] == "attestation"
                else "invalid"
            )
            result = {
                "action": "RESULT", "task_id": task["task_id"],
                "quality": quality, "result_hash": str(index),
                "work_product": work_product,
            }
            ledger.submit_result(
                "camouflage", task["task_id"], quality, str(index),
                sign_payload(key, result), work_product, now=2.0 * index + 1.2,
            )
        status = ledger.node_status("camouflage", expire=False)
        self.assertFalse(status["graduation_ready"])
        self.assertLess(status["trust_score"], 0.60)
        self.assertTrue(status["budget_exhausted"])
        self.assertEqual(status["access_state"], "QUARANTINE")
        self.assertIsNone(ledger.issue_task("camouflage", now=20.0)["task"])
        ledger.connection.close()

    def test_signed_onboarding_completion_releases_barrier(self):
        root = Path(self.temp.name)
        ledger = AccessLedger(
            root / "barrier.sqlite", root / "barrier.pem", 1.0,
            expected_experiment_nodes=1,
        )
        key = Ed25519PrivateKey.generate()
        public_key = public_key_b64(key)
        registration = {
            "action": "REGISTER", "node_id": "barrier-node",
            "profile": "honest", "public_key": public_key,
        }
        ledger.register_node(
            "barrier-node", "honest", public_key,
            sign_payload(key, registration), now=0.0,
        )
        completion = {
            "action": "ONBOARDING_FINISHED",
            "node_id": "barrier-node",
            "final_access_state": "PROBATION",
        }
        ledger.mark_onboarding_finished(
            "barrier-node", "PROBATION",
            sign_payload(key, completion), now=1.0,
        )
        barrier = ledger.barrier_status()
        self.assertTrue(barrier["released"])
        self.assertEqual(barrier["finished_nodes"], 1)
        self.assertEqual(barrier["admitted_nodes"], 0)
        self.assertTrue(ledger.verify_integrity()["valid"])
        ledger.connection.close()

    def test_oracle_benign_admits_every_profile_without_probation(self):
        root = Path(self.temp.name)
        ledger = AccessLedger(
            root / "oracle-benign.sqlite", root / "oracle-benign.pem", 1.0,
            mechanism_variant="oracle_benign",
        )
        key = Ed25519PrivateKey.generate()
        public_key = public_key_b64(key)
        registration = {
            "action": "REGISTER", "node_id": "oracle-node",
            "profile": "attestation_camouflage", "public_key": public_key,
        }
        ledger.register_node(
            "oracle-node", "attestation_camouflage", public_key,
            sign_payload(key, registration), now=0.0,
        )
        status = ledger.node_status("oracle-node", expire=False)
        self.assertEqual(status["access_state"], "ADMITTED")
        self.assertTrue(status["oracle_decision"])
        self.assertEqual(status["completed_tasks"], 0)
        ledger.connection.close()

    def test_oracle_filter_admits_honest_and_quarantines_attack_profile(self):
        root = Path(self.temp.name)
        ledger = AccessLedger(
            root / "oracle-filter.sqlite", root / "oracle-filter.pem", 1.0,
            mechanism_variant="oracle_filter",
        )
        for node_id, profile in (("good", "honest"), ("bad", "single_type_farming")):
            key = Ed25519PrivateKey.generate()
            public_key = public_key_b64(key)
            registration = {
                "action": "REGISTER", "node_id": node_id,
                "profile": profile, "public_key": public_key,
            }
            ledger.register_node(
                node_id, profile, public_key,
                sign_payload(key, registration), now=0.0,
            )
        self.assertEqual(
            ledger.node_status("good", expire=False)["access_state"], "ADMITTED"
        )
        self.assertEqual(
            ledger.node_status("bad", expire=False)["access_state"], "QUARANTINE"
        )
        ledger.connection.close()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import (
    AccessLedger, challenge_answer, public_key_b64, sign_payload,
)


def graduate_to_limited(ledger: AccessLedger, key: Ed25519PrivateKey) -> None:
    public = public_key_b64(key)
    registration = {
        "action": "REGISTER", "node_id": "node", "profile": "honest",
        "public_key": public,
    }
    ledger.register_node("node", "honest", public, sign_payload(key, registration), now=0)
    for index in range(9):
        at = index * 2 + 1.0
        assignment = ledger.issue_task("node", now=at)
        task = assignment["task"]
        receipt = {"action": "ACK", "task_id": task["task_id"],
                   "assignment_hash": assignment["assignment_hash"]}
        ledger.acknowledge("node", task["task_id"], assignment["assignment_hash"],
                           sign_payload(key, receipt), now=at + 0.1)
        result = {"action": "RESULT", "task_id": task["task_id"],
                  "quality": 0.95, "result_hash": f"result-{index}",
                  "work_product": challenge_answer(task)}
        ledger.submit_result("node", task["task_id"], 0.95, result["result_hash"],
                             sign_payload(key, result), result["work_product"], now=at + 0.2)


def accept_round(ledger: AccessLedger, key: Ed25519PrivateKey, round_number: int):
    update_hash = f"{round_number:064x}"
    parent_hash = f"{round_number + 100:064x}"
    payload = {"action": "FIT_COMMITMENT", "node_id": "node",
               "server_round": round_number, "update_hash": update_hash,
               "parent_hash": parent_hash}
    ledger.verify_fit_commitment("node", round_number, update_hash, parent_hash,
                                 sign_payload(key, payload))
    return update_hash, ledger.accept_fit_update("node", round_number, update_hash)


class ProgressiveAccessTests(unittest.TestCase):
    def test_freshness_lease_cycles_newcomer_between_full_and_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ledger = AccessLedger(
                path / "db.sqlite", path / "server.pem",
                mechanism_variant="access_freshness_lease",
                min_completed_tasks=9,
                min_limited_observation_updates=2,
                limited_aggregation_weight=0.25,
                access_lease_full_updates=2,
            )
            key = Ed25519PrivateKey.generate()
            graduate_to_limited(ledger, key)

            observed = []
            for server_round in range(1, 7):
                update_hash, decision = accept_round(ledger, key, server_round)
                observed.append((
                    decision["access_state"],
                    decision["aggregation_weight"],
                    decision["access_transition"],
                ))
                ledger.record_fit_aggregation("node", server_round, update_hash)

            self.assertEqual(observed, [
                ("LIMITED", 0.25, ""),
                ("ADMITTED", 1.0, "PROMOTED"),
                ("ADMITTED", 1.0, ""),
                ("LIMITED", 0.25, "LEASE_EXPIRED"),
                ("LIMITED", 0.25, ""),
                ("ADMITTED", 1.0, "RENEWED"),
            ])
            status = ledger.node_status("node", expire=False)
            self.assertTrue(status["freshness_lease_enabled"])
            self.assertEqual(status["access_lease_full_updates"], 2)
            events = [row["event_type"] for row in ledger.connection.execute(
                "SELECT event_type FROM audit_events ORDER BY sequence"
            )]
            self.assertEqual(events.count("ACCESS_PROMOTED"), 1)
            self.assertEqual(events.count("ACCESS_LEASE_EXPIRED"), 1)
            self.assertEqual(events.count("ACCESS_RENEWED"), 1)
            self.assertTrue(ledger.verify_integrity()["valid"])
            ledger.connection.close()

    def test_freshness_lease_does_not_cycle_incumbent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ledger = AccessLedger(
                path / "db.sqlite", path / "server.pem",
                mechanism_variant="access_freshness_lease",
                min_completed_tasks=9,
                min_limited_observation_updates=2,
                access_lease_full_updates=2,
                incumbent_node_ids={"node"},
            )
            key = Ed25519PrivateKey.generate()
            graduate_to_limited(ledger, key)
            self.assertEqual(
                ledger.node_status("node", expire=False)["access_state"],
                "ADMITTED",
            )
            for server_round in range(1, 7):
                update_hash, decision = accept_round(ledger, key, server_round)
                self.assertEqual(decision["access_state"], "ADMITTED")
                self.assertEqual(decision["aggregation_weight"], 1.0)
                self.assertEqual(decision["access_transition"], "")
                ledger.record_fit_aggregation("node", server_round, update_hash)
            ledger.connection.close()

    def test_limited_updates_are_weighted_before_full_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ledger = AccessLedger(path / "db.sqlite", path / "server.pem",
                                  mechanism_variant="progressive_full",
                                  min_completed_tasks=9,
                                  min_limited_clean_updates=2)
            key = Ed25519PrivateKey.generate()
            graduate_to_limited(ledger, key)
            self.assertEqual(ledger.node_status("node", expire=False)["access_state"], "LIMITED")

            first_hash, first = accept_round(ledger, key, 1)
            self.assertTrue(first["aggregation_authorized"])
            self.assertEqual(first["aggregation_weight"], 0.25)
            self.assertEqual(first["access_state"], "LIMITED")
            ledger.record_fit_aggregation("node", 1, first_hash)

            second_hash, second = accept_round(ledger, key, 2)
            self.assertTrue(second["aggregation_authorized"])
            self.assertEqual(second["aggregation_weight"], 1.0)
            self.assertEqual(second["access_state"], "ADMITTED")
            ledger.record_fit_aggregation("node", 2, second_hash)
            status = ledger.node_status("node", expire=False)
            self.assertEqual(status["training_clean_evidence"], 2)
            self.assertEqual(ledger.experiment_summary()["nodes"][0]["aggregated_fit_events"], 2)
            events = [row["event_type"] for row in ledger.connection.execute(
                "SELECT event_type FROM audit_events ORDER BY sequence"
            )]
            self.assertIn("ACCESS_PROMOTED", events)
            self.assertTrue(ledger.verify_integrity()["valid"])
            ledger.connection.close()

    def test_hard_failure_creates_non_compensable_penalty_and_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ledger = AccessLedger(path / "db.sqlite", path / "server.pem",
                                  mechanism_variant="progressive_full",
                                  min_completed_tasks=9,
                                  min_limited_clean_updates=2)
            key = Ed25519PrivateKey.generate()
            graduate_to_limited(ledger, key)
            accept_round(ledger, key, 1)
            status = ledger.revoke_training_access("node", 2, "VALIDATION_LOSS")
            self.assertEqual(status["access_state"], "QUARANTINE")
            self.assertEqual(status["penalty_debt"], 1.0)
            after = ledger.node_status("node", expire=False)
            self.assertEqual(after["training_clean_evidence"], 1)
            self.assertEqual(after["penalty_debt"], 1.0)
            self.assertEqual(after["hard_failures"], 1)
            self.assertEqual(after["access_state"], "QUARANTINE")
            with self.assertRaisesRegex(ValueError, "eligible"):
                accept_round(ledger, key, 3)
            ledger.connection.close()


if __name__ == "__main__":
    unittest.main()

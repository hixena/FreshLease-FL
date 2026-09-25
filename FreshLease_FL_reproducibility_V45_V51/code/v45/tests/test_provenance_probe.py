from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import (
    AccessLedger, challenge_answer, public_key_b64, sign_payload,
)
from flower_prototype.run_provenance_probe import run_case
from flower_prototype.source_verifier import ExactArtifactIssuer


class ProvenanceProbeTests(unittest.TestCase):
    def test_reference_issuer_detects_exact_reuse_and_rejects_unknown_bytes(self):
        key = Ed25519PrivateKey.generate()
        issuer = ExactArtifactIssuer(key, [b"reference-A", b"reference-B"])
        task1 = {"node_id": "node", "task_id": "one", "nonce": "first", "task_type": "shadow_update"}
        task2 = {**task1, "task_id": "two", "nonce": "second"}
        result1 = {"task_id": "one", "work_product": challenge_answer(task1), "result_hash": "hash-one"}
        result2 = {"task_id": "two", "work_product": challenge_answer(task2), "result_hash": "hash-two"}
        source1, signature1 = issuer.verify_and_sign("node", task1, result1, b"reference-A")
        source2, signature2 = issuer.verify_and_sign("node", task2, result2, b"reference-A")
        self.assertEqual(source1, source2)
        self.assertNotEqual(signature1, signature2)
        with self.assertRaisesRegex(ValueError, "not in independent reference"):
            issuer.verify_and_sign("node", task2, result2, b"fake-claim-of-new-source")
        with self.assertRaisesRegex(ValueError, "challenge failed"):
            issuer.verify_and_sign("node", task2, {**result2, "work_product": "bad"}, b"reference-B")

    def test_provenance_mode_fails_closed_without_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "requires an independent source verifier"):
                AccessLedger(root / "db.sqlite", root / "server.pem", mechanism_variant="full_provenance")

    def test_paired_genuine_reuse_and_future_betrayal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcomes = {
                (v, b): run_case(root, v, b)
                for v in ("full", "full_provenance")
                for b in ("independent_honest", "reused_source", "genuine_then_betray")
            }
        self.assertTrue(all(row["integrity_valid"] for row in outcomes.values()))
        self.assertEqual(outcomes["full", "reused_source"]["access_state"], "ADMITTED")
        self.assertEqual(outcomes["full_provenance", "reused_source"]["access_state"], "QUARANTINE")
        self.assertEqual(outcomes["full_provenance", "reused_source"]["credited_independent_tasks"], 4)
        for behavior in ("independent_honest", "genuine_then_betray"):
            self.assertEqual(outcomes["full_provenance", behavior]["access_state"], "ADMITTED")
            self.assertEqual(outcomes["full_provenance", behavior]["completed_tasks"], 9)

    def test_no_provenance_is_not_credited_and_source_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verifier_key = Ed25519PrivateKey.generate()
            ledger = AccessLedger(
                root / "db.sqlite", root / "server.pem", mechanism_variant="full_provenance",
                source_verifier_public_key=public_key_b64(verifier_key),
            )
            try:
                key = Ed25519PrivateKey.generate()
                registration = {
                    "action": "REGISTER", "node_id": "node", "profile": "honest",
                    "public_key": public_key_b64(key),
                }
                ledger.register_node("node", "honest", registration["public_key"],
                                     sign_payload(key, registration), now=0)
                assignment = ledger.issue_task("node", now=1)
                task = assignment["task"]
                initial_claim = {"action": "VERIFY_SOURCE", "node_id": "node",
                                 "task_id": task["task_id"], "source_id": "source",
                                 "result_hash": None, "work_product": None}
                with self.assertRaisesRegex(ValueError, "only completed"):
                    ledger.record_verified_source("node", task["task_id"], "source",
                                                  sign_payload(verifier_key, initial_claim), now=1.01)
                ack = {"action": "ACK", "task_id": task["task_id"],
                       "assignment_hash": assignment["assignment_hash"]}
                ledger.acknowledge("node", task["task_id"], assignment["assignment_hash"],
                                   sign_payload(key, ack), now=1.1)
                result = {"action": "RESULT", "task_id": task["task_id"],
                          "quality": 0.9, "result_hash": "result",
                          "work_product": challenge_answer(task)}
                ledger.submit_result("node", task["task_id"], 0.9, "result",
                                     sign_payload(key, result), result["work_product"], now=1.2)
                self.assertEqual(ledger.node_status("node", expire=False)["credited_independent_tasks"], 0)
                claim = {"action": "VERIFY_SOURCE", "node_id": "node",
                         "task_id": task["task_id"], "source_id": "source",
                         "result_hash": result["result_hash"],
                         "work_product": result["work_product"]}
                signature = sign_payload(verifier_key, claim)
                with self.assertRaises(Exception):
                    ledger.record_verified_source("node", task["task_id"], "source",
                                                  sign_payload(key, claim), now=1.21)
                self.assertEqual(ledger.node_status("node", expire=False)["credited_independent_tasks"], 0)
                ledger.record_verified_source("node", task["task_id"], "source",
                                              signature, now=1.3)
                self.assertEqual(ledger.node_status("node", expire=False)["credited_independent_tasks"], 1)
                with self.assertRaisesRegex(ValueError, "cannot be changed"):
                    ledger.record_verified_source("node", task["task_id"], "different",
                                                  sign_payload(verifier_key, {**claim, "source_id": "different"}), now=1.4)
                with self.assertRaises(Exception):
                    ledger.record_verified_source("node", task["task_id"], "source",
                                                  sign_payload(verifier_key, {**claim, "result_hash": "fake"}), now=1.5)
                self.assertTrue(ledger.verify_integrity()["valid"])
            finally:
                ledger.connection.close()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import (
    AccessLedger, challenge_answer, public_key_b64, sign_payload,
)
from flower_prototype.real_data import client_partition, initial_parameters, local_train
from flower_prototype.shadow_observation import inspect_shadow_artifact


def actual_update(index: int, alpha: float = .5, partition_id: int = 4) -> dict:
    x, y = client_partition(partition_id, 6, alpha, 0)
    values = local_train(initial_parameters(), x, y, 2, 0.15, 32,
                         partition_id * 100 + index)
    return {"weights": values[0].tolist(), "bias": values[1].tolist()}


def trial(root: Path, variant: str, replay: bool,
          alpha: float = .5, partition_id: int = 4) -> dict:
    ledger = AccessLedger(
        root / f"{variant}-{replay}-{alpha}-{partition_id}.sqlite",
        root / f"{variant}-{replay}-{alpha}-{partition_id}.pem",
        task_deadline_seconds=1.0, mechanism_variant=variant,
        min_completed_tasks=9, max_probation_tasks=12,
    )
    key = Ed25519PrivateKey.generate()
    node = "node"
    registration = {"action": "REGISTER", "node_id": node,
                    "profile": "unclassified", "public_key": public_key_b64(key)}
    ledger.register_node(node, registration["profile"], registration["public_key"],
                         sign_payload(key, registration), now=0.0)
    first_shadow = None
    try:
        for i in range(12):
            at = 2 * i + 1.0
            assignment = ledger.issue_task(node, now=at)
            task = assignment["task"]
            if task is None:
                break
            receipt = {"action": "ACK", "task_id": task["task_id"],
                       "assignment_hash": assignment["assignment_hash"]}
            ledger.acknowledge(node, task["task_id"], receipt["assignment_hash"],
                               sign_payload(key, receipt), now=at + 0.1)
            result = {"action": "RESULT", "task_id": task["task_id"],
                      "quality": 0.9, "result_hash": f"result-{i}",
                      "work_product": challenge_answer(task)}
            if task["task_type"] == "shadow_update":
                update = actual_update(i, alpha, partition_id)
                if first_shadow is None:
                    first_shadow = update
                result["shadow_update"] = first_shadow if replay else update
            ledger.submit_result(
                node, task["task_id"], 0.9, result["result_hash"],
                sign_payload(key, result), result["work_product"],
                now=at + 0.2, shadow_update=result.get("shadow_update"),
            )
        status = ledger.node_status(node, expire=False)
        assert ledger.verify_integrity()["valid"]
        return status
    finally:
        ledger.connection.close()


class ShadowObservationTests(unittest.TestCase):
    def test_held_out_normal_partitions_across_noniid_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for alpha in (.1, .5, 10.):
                for partition_id in (0, 1, 2):
                    status = trial(root, "full_shadow_provenance", replay=False,
                                   alpha=alpha, partition_id=partition_id)
                    self.assertEqual(status["access_state"], "ADMITTED",
                                     (alpha, partition_id, status))

    def test_paired_exact_replay_and_independent_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_base = trial(root, "full_shadow_no_dedup", replay=True)
            replay_dedup = trial(root, "full_shadow_provenance", replay=True)
            honest_base = trial(root, "full_shadow_no_dedup", replay=False)
            honest_dedup = trial(root, "full_shadow_provenance", replay=False)
        self.assertEqual((replay_base["access_state"], replay_base["completed_tasks"]),
                         ("ADMITTED", 9))
        self.assertEqual((replay_dedup["access_state"], replay_dedup["completed_tasks"]),
                         ("QUARANTINE", 12))
        self.assertEqual(replay_dedup["credited_independent_tasks"], 4)
        for status in (honest_base, honest_dedup):
            self.assertEqual(status["access_state"], "ADMITTED")
            self.assertEqual(status["completed_tasks"], 9)

    def test_reject_malformed_and_nonfinite_artifacts(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            inspect_shadow_artifact({"weights": [], "bias": []})
        artifact = actual_update(3)
        artifact["bias"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            inspect_shadow_artifact(artifact)

    def test_small_value_perturbation_bypasses_exact_fingerprint(self):
        original = actual_update(3)
        altered = {"weights": original["weights"], "bias": original["bias"].copy()}
        altered["bias"][0] += 0.0001
        self.assertNotEqual(inspect_shadow_artifact(original),
                            inspect_shadow_artifact(altered))

    def test_requires_signed_actual_update_for_shadow_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = AccessLedger(root / "db.sqlite", root / "server.pem",
                                  mechanism_variant="full_shadow_provenance")
            key = Ed25519PrivateKey.generate()
            registration = {"action": "REGISTER", "node_id": "node",
                            "profile": "unclassified", "public_key": public_key_b64(key)}
            ledger.register_node("node", "unclassified", registration["public_key"],
                                 sign_payload(key, registration), now=0)
            try:
                for i in range(4):
                    at = 2 * i + 1
                    assignment = ledger.issue_task("node", now=at)
                    task = assignment["task"]
                    receipt = {"action": "ACK", "task_id": task["task_id"],
                               "assignment_hash": assignment["assignment_hash"]}
                    ledger.acknowledge("node", task["task_id"], receipt["assignment_hash"],
                                       sign_payload(key, receipt), now=at + .1)
                    result = {"action": "RESULT", "task_id": task["task_id"],
                              "quality": .9, "result_hash": str(i),
                              "work_product": challenge_answer(task)}
                    if i == 3:
                        with self.assertRaisesRegex(ValueError, "required"):
                            ledger.submit_result("node", task["task_id"], .9, str(i),
                                                 sign_payload(key, result), result["work_product"],
                                                 now=at + .2)
                        artifact = actual_update(3)
                        with self.assertRaises(Exception):
                            ledger.submit_result("node", task["task_id"], .9, str(i),
                                                 sign_payload(key, result), result["work_product"],
                                                 now=at + .2, shadow_update=artifact)
                    else:
                        ledger.submit_result("node", task["task_id"], .9, str(i),
                                             sign_payload(key, result), result["work_product"],
                                             now=at + .2)
            finally:
                ledger.connection.close()


if __name__ == "__main__":
    unittest.main()

"""Deterministic protocol and audit tampering probe; NOT an FL training run.

Includes a deliberate negative control: deleting the newest audit row is
undetectable by an unanchored in-database hash chain. Record this limitation.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import (
    AccessLedger, challenge_answer, public_key_b64, sign_payload,
)


def setup(root: Path):
    ledger = AccessLedger(root / "ledger.sqlite", root / "controller.pem",
                          task_deadline_seconds=60, mechanism_variant="access_freshness_lease")
    key = Ed25519PrivateKey.generate()
    registration = {"action": "REGISTER", "node_id": "honest-newcomer",
                    "profile": "honest", "public_key": public_key_b64(key)}
    ledger.register_node(registration["node_id"], "honest", registration["public_key"],
                         sign_payload(key, registration), now=0)
    return ledger, key


def start(ledger, key, t=1):
    issued = ledger.issue_task("honest-newcomer", now=t)
    task = issued["task"]
    receipt = {"action": "ACK", "task_id": task["task_id"],
               "assignment_hash": issued["assignment_hash"]}
    signature = sign_payload(key, receipt)
    ledger.acknowledge("honest-newcomer", task["task_id"], issued["assignment_hash"],
                       signature, now=t + 0.1)
    return issued, receipt, signature


def submit(ledger, key, task, work_product, t):
    payload = {"action": "RESULT", "task_id": task["task_id"],
               "quality": 0.95, "result_hash": f"result-{t}",
               "work_product": work_product}
    return ledger.submit_result("honest-newcomer", task["task_id"], 0.95,
                                payload["result_hash"], sign_payload(key, payload),
                                work_product, now=t)


def run_case(name: str, root: Path) -> dict:
    ledger, key = setup(root)
    try:
        issued, receipt, signature = start(ledger, key)
        original_head = ledger.verify_chain()["head"]
        detected, interpretation = False, ""
        if name == "clean_control":
            detected = ledger.verify_integrity()["valid"]
            interpretation = "clean baseline passes integrity check"
        elif name == "mutate_audit_payload":
            ledger.connection.execute("UPDATE audit_events SET payload_json=? WHERE sequence=1",
                                      ('{"forged":true}',))
            detected = not ledger.verify_integrity()["valid"]
            interpretation = "in-place payload mutation changes linked digest"
        elif name == "delete_middle_audit_event":
            ledger.connection.execute("DELETE FROM audit_events WHERE sequence=2")
            detected = not ledger.verify_integrity()["valid"]
            interpretation = "middle deletion breaks previous-hash linkage"
        elif name == "delete_latest_audit_event":
            ledger.connection.execute("DELETE FROM audit_events WHERE sequence=(SELECT MAX(sequence) FROM audit_events)")
            detected = not ledger.verify_integrity()["valid"]
            interpretation = "no external head anchor; suffix deletion may pass internal check"
        elif name == "forged_ack_signature":
            new_task = ledger.issue_task("honest-newcomer", now=2)["task"]
            forged = {"action": "ACK", "task_id": new_task["task_id"],
                      "assignment_hash": ledger.connection.execute(
                          "SELECT assignment_hash FROM tasks WHERE task_id=?",
                          (new_task["task_id"],)).fetchone()[0]}
            try:
                ledger.acknowledge("honest-newcomer", forged["task_id"],
                                   forged["assignment_hash"],
                                   sign_payload(Ed25519PrivateKey.generate(), forged), now=2.1)
            except Exception:
                detected = True
            interpretation = "signature does not match registered node public key"
        elif name == "replay_signed_ack":
            try:
                ledger.acknowledge("honest-newcomer", receipt["task_id"],
                                   receipt["assignment_hash"], signature, now=1.2)
            except ValueError:
                detected = True
            interpretation = "task is no longer in ISSUED state"
        elif name == "reuse_old_nonce_answer":
            old_answer = challenge_answer(issued["task"])
            submit(ledger, key, issued["task"], old_answer, 1.2)
            next_task = start(ledger, key, 3)[0]["task"]
            submit(ledger, key, next_task, old_answer, 3.2)
            state = ledger.connection.execute("SELECT status FROM tasks WHERE task_id=?",
                                              (next_task["task_id"],)).fetchone()[0]
            detected = state == "INVALID_RESULT"
            interpretation = "old work product fails next task's nonce-bound challenge"
        else:
            raise ValueError(name)
        chain = ledger.verify_chain()
        return {"case": name, "detected_or_valid": int(detected),
                "expected": int(name != "delete_latest_audit_event"),
                "internal_chain_valid": int(chain["valid"]),
                "head_matches_snapshot": int(chain.get("head") == original_head),
                "interpretation": interpretation}
    finally:
        ledger.connection.close()


CASES = ("clean_control", "mutate_audit_payload", "delete_middle_audit_event",
         "delete_latest_audit_event", "forged_ack_signature", "replay_signed_ack",
         "reuse_old_nonce_answer")


def run_all() -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="v51-audit-probe-") as temporary:
        return [run_case(case, Path(temporary) / case) for case in CASES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    results = run_all()
    if any(row["detected_or_valid"] != row["expected"] for row in results):
        raise RuntimeError("protocol audit regression: expected result differs")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} deterministic audit probes: {arguments.output}")
    print("Unanchored hash chain does NOT detect suffix deletion; see negative control")


if __name__ == "__main__":
    main()

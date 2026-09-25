from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import (
    AccessLedger,
    challenge_answer,
    public_key_b64,
    sign_payload,
)


def register(ledger: AccessLedger, node_id: str, profile: str, key) -> None:
    public_key = public_key_b64(key)
    payload = {
        "action": "REGISTER", "node_id": node_id,
        "profile": profile, "public_key": public_key,
    }
    ledger.register_node(
        node_id, profile, public_key, sign_payload(key, payload), now=0.0
    )


def execute(
    ledger: AccessLedger, node_id: str, key, now: float,
    action: str, quality: float = 0.88,
) -> None:
    assignment = ledger.issue_task(node_id, now=now)
    if assignment.get("task") is None:
        return
    task = assignment["task"]
    if action == "unconfirmed":
        ledger.expire_due_tasks(now=now + 2.0)
        return
    receipt = {
        "action": "ACK", "task_id": task["task_id"],
        "assignment_hash": assignment["assignment_hash"],
    }
    ledger.acknowledge(
        node_id, task["task_id"], assignment["assignment_hash"],
        sign_payload(key, receipt), now=now + 0.1,
    )
    if action == "withhold":
        ledger.expire_due_tasks(now=now + 2.0)
        return
    result_hash = hashlib.sha256(f"{node_id}:{task['task_id']}".encode()).hexdigest()
    result = {
        "action": "RESULT", "task_id": task["task_id"],
        "quality": round(quality, 6), "result_hash": result_hash,
        "work_product": challenge_answer(task),
    }
    ledger.submit_result(
        node_id, task["task_id"], quality, result_hash,
        sign_payload(key, result), result["work_product"], now=now + 0.2,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ledger = AccessLedger(root / "ledger.sqlite", root / "server.pem", 1.0)
        keys = {
            name: Ed25519PrivateKey.generate()
            for name in ("honest", "benign-unreliable", "selective")
        }
        for name, key in keys.items():
            register(ledger, name, name, key)

        for index in range(12):
            execute(ledger, "honest", keys["honest"], index * 3.0 + 1.0, "submit")
            unreliable_action = "unconfirmed" if index in {1, 5} else "submit"
            execute(
                ledger, "benign-unreliable", keys["benign-unreliable"],
                index * 3.0 + 1.0, unreliable_action,
            )
            selective_action = "submit" if index in {0, 3, 7} else "withhold"
            execute(
                ledger, "selective", keys["selective"],
                index * 3.0 + 1.0, selective_action, 0.92,
            )

        report = {
            "nodes": {
                name: ledger.node_status(name, expire=False) for name in keys
            },
            "integrity": ledger.verify_integrity(),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        ledger.connection.close()


if __name__ == "__main__":
    main()

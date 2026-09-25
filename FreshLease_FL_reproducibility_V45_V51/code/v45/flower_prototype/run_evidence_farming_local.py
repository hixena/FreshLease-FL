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


VARIANTS = (
    "full", "no_diversity", "no_repeat_decay",
    "no_semantic_separation", "naive", "no_budget",
)
PROFILES = (
    "honest", "single_type_farming",
    "diverse_then_repeat_farming", "attestation_camouflage",
)


def quality(profile: str, task_type: str) -> float:
    if profile == "attestation_camouflage":
        return 0.97 if task_type == "attestation" else 0.25
    if profile == "diverse_then_repeat_farming":
        return 0.95 if task_type in {"attestation", "shadow_update"} else 0.15
    if profile == "single_type_farming":
        return 0.95
    return 0.88


def main() -> None:
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for variant_index, variant in enumerate(VARIANTS):
            ledger = AccessLedger(
                root / f"{variant}.sqlite", root / f"{variant}.pem",
                task_deadline_seconds=1.0,
                mechanism_variant=variant,
                max_probation_tasks=20,
            )
            keys = {}
            for profile in PROFILES:
                node_id = f"{variant}-{profile}"
                key = Ed25519PrivateKey.generate()
                keys[node_id] = key
                public_key = public_key_b64(key)
                registration = {
                    "action": "REGISTER", "node_id": node_id,
                    "profile": profile, "public_key": public_key,
                }
                ledger.register_node(
                    node_id, profile, public_key,
                    sign_payload(key, registration), now=0.0,
                )

            for attempt in range(24):
                for node_offset, (node_id, key) in enumerate(keys.items()):
                    now = 10_000 * variant_index + 10 * attempt + node_offset + 1.0
                    assignment = ledger.issue_task(node_id, now=now)
                    task = assignment.get("task")
                    if task is None:
                        continue
                    profile = ledger.node_status(node_id, expire=False)["profile"]
                    if profile == "single_type_farming" and task["task_type"] != "shadow_update":
                        # The attacker only supplies one kind of evidence.
                        ledger.expire_due_tasks(now=now + 2.0)
                        continue
                    receipt = {
                        "action": "ACK", "task_id": task["task_id"],
                        "assignment_hash": assignment["assignment_hash"],
                    }
                    ledger.acknowledge(
                        node_id, task["task_id"], assignment["assignment_hash"],
                        sign_payload(key, receipt), now=now + 0.1,
                    )
                    observed = quality(profile, task["task_type"])
                    result_hash = hashlib.sha256(
                        f"{node_id}:{task['task_id']}:{observed}".encode()
                    ).hexdigest()
                    result = {
                        "action": "RESULT", "task_id": task["task_id"],
                        "quality": round(observed, 6), "result_hash": result_hash,
                        "work_product": (
                            challenge_answer(task) if observed >= 0.60 else "invalid"
                        ),
                    }
                    ledger.submit_result(
                        node_id, task["task_id"], observed, result_hash,
                        sign_payload(key, result), result["work_product"], now=now + 0.2,
                    )

            for node_id in keys:
                status = ledger.node_status(node_id, expire=False)
                rows.append({
                    key: status[key] for key in (
                        "mechanism_variant", "profile", "access_state",
                        "trust_score", "completed_tasks", "evidence_types",
                        "graduation_ready", "budget_exhausted",
                    )
                })
            assert ledger.verify_integrity()["valid"]
            ledger.connection.close()
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

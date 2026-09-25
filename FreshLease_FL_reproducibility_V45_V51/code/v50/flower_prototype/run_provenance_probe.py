"""Offline, paired V14 evidence-provenance probe; no Docker or Flower required.

The trusted verifier is simulated with an explicit source-id oracle here. This
does NOT establish that a real verifier can obtain such identities from FL work.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import (
    AccessLedger, challenge_answer, public_key_b64, sign_payload,
)
from flower_prototype.source_verifier import ExactArtifactIssuer


def run_case(root: Path, variant: str, behavior: str) -> dict:
    verifier_key = Ed25519PrivateKey.generate()
    # The verifier's reference corpus is fixed across all variants/behaviors.
    # These are toy authenticated artifacts, NOT independently verified FL updates.
    corpus = [hashlib.sha256(f"reference-work-{i}".encode()).digest() for i in range(20)]
    issuer = ExactArtifactIssuer(verifier_key, corpus)
    ledger = AccessLedger(
        root / f"{variant}-{behavior}.sqlite",
        root / f"{variant}-{behavior}.pem",
        mechanism_variant=variant, task_deadline_seconds=1.0,
        min_completed_tasks=9, max_probation_tasks=20,
        source_verifier_public_key=public_key_b64(verifier_key),
    )
    key = Ed25519PrivateKey.generate()
    identity = f"case-{behavior}"
    public_key = public_key_b64(key)
    registration = {
        "action": "REGISTER", "node_id": identity,
        # Same self-declared profile, never used to steer tasks/decision.
        "profile": "unclassified", "public_key": public_key,
    }
    ledger.register_node(
        identity, "unclassified", public_key,
        sign_payload(key, registration), now=0.0,
    )
    try:
        for i in range(20):
            at = 2.0 * i + 1.0
            assignment = ledger.issue_task(identity, now=at)
            task = assignment["task"]
            if task is None:
                break
            receipt = {
                "action": "ACK", "task_id": task["task_id"],
                "assignment_hash": assignment["assignment_hash"],
            }
            ledger.acknowledge(
                identity, task["task_id"], assignment["assignment_hash"],
                sign_payload(key, receipt), now=at + 0.1,
            )
            result = {
                "action": "RESULT", "task_id": task["task_id"],
                "quality": 0.9, "result_hash": f"response-{i}",
                "work_product": challenge_answer(task),
            }
            ledger.submit_result(
                identity, task["task_id"], 0.9, result["result_hash"],
                sign_payload(key, result), result["work_product"], now=at + 0.2,
            )
            # Client may submit the same actual artifact with fresh signed
            # nonce-bound responses; the trusted issuer independently hashes
            # the artifact bytes and only accepts known reference artifacts.
            artifact = corpus[3 if behavior == "reused_source" and i >= 3 else i]
            source_id, proof = issuer.verify_and_sign(identity, task, result, artifact)
            ledger.record_verified_source(
                identity, task["task_id"], source_id,
                proof, now=at + 0.3,
            )
        status = ledger.node_status(identity, expire=False)
        return {
            "variant": variant,
            "behavior": behavior,
            "access_state": status["access_state"],
            "completed_tasks": status["completed_tasks"],
            "credited_independent_tasks": status["credited_independent_tasks"],
            "evidence_mass": round(status["evidence_mass"], 4),
            "integrity_valid": ledger.verify_integrity()["valid"],
        }
    finally:
        ledger.connection.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        rows = [
            run_case(root, variant, behavior)
            for behavior in ("independent_honest", "reused_source", "genuine_then_betray")
            for variant in ("full", "full_provenance")
        ]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("NOTE: genuine_then_betray is observationally identical to honest before admission;")
    print("a post-admission attack requires online update validation/revocation, not provenance deduction.")


if __name__ == "__main__":
    main()

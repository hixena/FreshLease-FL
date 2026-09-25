"""Restricted exact-artifact issuer for the *offline* provenance experiment.

Trust model: the issuer privately holds a finite reference corpus and an Ed25519
key. It sees actual submitted artifact bytes and signs their canonical digest,
never a client-chosen source ID. This cannot prove semantic independence or
actual Flower model-training quality.
"""
from __future__ import annotations

import hashlib
from typing import Any, Collection

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import challenge_answer, sign_payload


class ExactArtifactIssuer:
    def __init__(self, private_key: Ed25519PrivateKey, reference_corpus: Collection[bytes]):
        self.private_key = private_key
        self.known_hashes = {hashlib.sha256(item).hexdigest() for item in reference_corpus}

    def verify_and_sign(
        self, node_id: str, task: dict[str, Any],
        result: dict[str, Any], artifact_bytes: bytes,
    ) -> tuple[str, str]:
        if task["node_id"] != node_id or task["task_id"] != result["task_id"]:
            raise ValueError("task identity mismatch")
        if result["work_product"] != challenge_answer(task):
            raise ValueError("probation challenge failed")
        source_id = hashlib.sha256(artifact_bytes).hexdigest()
        if source_id not in self.known_hashes:
            raise ValueError("artifact is not in independent reference corpus")
        claim = {
            "action": "VERIFY_SOURCE", "node_id": node_id,
            "task_id": task["task_id"], "source_id": source_id,
            "result_hash": result["result_hash"],
            "work_product": result["work_product"],
        }
        return source_id, sign_payload(self.private_key, claim)

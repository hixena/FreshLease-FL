from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from flower_prototype.shadow_observation import inspect_shadow_artifact
from flower_prototype.variant_policy import (
    is_access_core_variant, is_progressive_variant, uses_evidence_diversity,
    uses_freshness_lease, uses_probation_budget, uses_repeat_decay,
    uses_result_verification,
    uses_semantic_separation,
)


TASK_WEIGHTS = {
    "attestation": 0.25,
    "protocol_check": 0.50,
    "canary_training": 0.75,
    "shadow_update": 1.00,
}
# Identical, predeclared schedule for every identity: first obtain all types,
# then revisit a low-risk shadow task.  This also creates an informative
# single-type-farming ablation even when nine completed results are required.
TASK_SEQUENCE = tuple(TASK_WEIGHTS) + ("shadow_update",) * 8
UTILITY = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)
DIRICHLET_BINS = np.asarray([0.20, 0.40, 0.60, 0.80], dtype=float)


def challenge_answer(task: dict[str, Any]) -> str:
    """Reference answer for a toy, nonce-bound probation challenge.

    This checks a protocol work product, not TPM attestation or training quality.
    """
    return digest({
        "node_id": task["node_id"], "task_id": task["task_id"],
        "nonce": task["nonce"], "task_type": task["task_type"],
        "challenge_version": 1,
    })


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def public_key_b64(key: Ed25519PrivateKey | Ed25519PublicKey) -> str:
    public = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return b64(public.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ))


def sign_payload(key: Ed25519PrivateKey, payload: dict[str, Any]) -> str:
    return b64(key.sign(canonical_bytes(payload)))


def verify_signature(public_key: str, payload: dict[str, Any], signature: str) -> None:
    key = Ed25519PublicKey.from_public_bytes(unb64(public_key))
    key.verify(unb64(signature), canonical_bytes(payload))


def load_or_create_private_key(path: str | Path) -> Ed25519PrivateKey:
    key_path = Path(path)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key_path.exists():
        return serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    key = Ed25519PrivateKey.generate()
    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    os.chmod(key_path, 0o600)
    return key


class AccessLedger:
    """SQLite-backed task ledger with Ed25519 signatures and a hash chain."""

    def __init__(
        self,
        database_path: str | Path,
        server_key_path: str | Path,
        task_deadline_seconds: float = 1.0,
        mechanism_variant: str = "full",
        max_probation_tasks: int = 20,
        expected_experiment_nodes: int = 0,
        min_completed_tasks: int = 5,
        min_evidence_mass: float = 1.50,
        min_limited_clean_updates: int = 2,
        min_limited_observation_updates: int | None = None,
        limited_aggregation_weight: float = 0.25,
        access_lease_full_updates: int = 5,
        source_verifier_public_key: str | None = None,
        trust_estimator: str = "dirichlet_lcb",
        incumbent_node_ids: set[str] | frozenset[str] | None = None,
    ):
        self.database_path = str(database_path)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.server_key = load_or_create_private_key(server_key_path)
        self.shadow_verifier_key = None
        self.task_deadline_seconds = float(task_deadline_seconds)
        allowed_variants = {
            "full", "full_provenance", "full_shadow_no_dedup",
            "full_shadow_provenance", "no_diversity", "no_repeat_decay",
            "no_semantic_separation", "naive", "no_budget",
            "oracle_benign", "oracle_filter", "no_result_verification",
            "no_online_revalidation",
            "progressive_full",
            "progressive_no_cumulative",
            "progressive_fltrust", "progressive_trimmed_mean",
            "fltrust", "trimmed_mean", "coordinate_median",
            "access_full", "access_no_diversity", "access_no_repeat_decay",
            "access_no_result_verification", "access_no_limited",
            "access_naive", "access_attestation_only", "access_static_multisource",
            "rffl_reputation", "access_freshness_lease",
        }
        if mechanism_variant not in allowed_variants:
            raise ValueError(f"unknown mechanism variant: {mechanism_variant}")
        self.mechanism_variant = mechanism_variant
        allowed_estimators = {"dirichlet_lcb", "dirichlet_mean", "beta_mean"}
        if trust_estimator not in allowed_estimators:
            raise ValueError(f"unknown trust estimator: {trust_estimator}")
        self.trust_estimator = trust_estimator
        if mechanism_variant in {"full_shadow_no_dedup", "full_shadow_provenance"}:
            self.shadow_verifier_key = load_or_create_private_key(
                Path(server_key_path).with_name("shadow_verifier_ed25519.pem")
            )
            source_verifier_public_key = public_key_b64(self.shadow_verifier_key)
        self.source_verifier_public_key = source_verifier_public_key
        if mechanism_variant == "full_provenance" and not source_verifier_public_key:
            raise ValueError("full_provenance requires an independent source verifier public key")
        self.max_probation_tasks = int(max_probation_tasks)
        self.expected_experiment_nodes = int(expected_experiment_nodes)
        if min_completed_tasks < 1:
            raise ValueError("min_completed_tasks must be positive")
        self.min_completed_tasks = int(min_completed_tasks)
        if min_evidence_mass <= 0.0:
            raise ValueError("min_evidence_mass must be positive")
        self.min_evidence_mass = float(min_evidence_mass)
        observation_updates = (
            min_limited_clean_updates
            if min_limited_observation_updates is None
            else min_limited_observation_updates
        )
        if observation_updates < 1:
            raise ValueError("min_limited_observation_updates must be positive")
        self.min_limited_observation_updates = int(observation_updates)
        # Compatibility alias for frozen V40 callers and historical tests.
        self.min_limited_clean_updates = self.min_limited_observation_updates
        if not 0.0 < limited_aggregation_weight <= 1.0:
            raise ValueError("limited_aggregation_weight must be in (0, 1]")
        self.limited_aggregation_weight = float(limited_aggregation_weight)
        if access_lease_full_updates < 1:
            raise ValueError("access_lease_full_updates must be positive")
        self.access_lease_full_updates = int(access_lease_full_updates)
        # V37 experimental initial condition.  Incumbency is assigned by an
        # explicit, predeclared node-id list and never inferred from profile.
        self.incumbent_node_ids = frozenset(
            node_id.strip() for node_id in (incumbent_node_ids or set())
            if node_id.strip()
        )
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(
            self.database_path, check_same_thread=False, isolation_level=None
        )
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    @property
    def server_public_key(self) -> str:
        return public_key_b64(self.server_key)

    def _create_schema(self) -> None:
        self.connection.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            public_key TEXT NOT NULL,
            profile TEXT NOT NULL,
            registered_at REAL NOT NULL,
            registration_signature TEXT
        );
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            nonce TEXT NOT NULL,
            issued_at REAL NOT NULL,
            deadline_at REAL NOT NULL,
            assignment_hash TEXT NOT NULL,
            server_signature TEXT NOT NULL,
            status TEXT NOT NULL,
            acknowledged_at REAL,
            completed_at REAL,
            quality REAL,
            reported_quality REAL,
            result_hash TEXT,
            work_product TEXT,
            shadow_update_json TEXT,
            receipt_signature TEXT,
            result_signature TEXT,
            FOREIGN KEY(node_id) REFERENCES nodes(node_id)
        );
        CREATE TABLE IF NOT EXISTS audit_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            node_id TEXT NOT NULL,
            task_id TEXT,
            payload_json TEXT NOT NULL,
            previous_hash TEXT NOT NULL,
            event_hash TEXT NOT NULL UNIQUE,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS flower_events (
            event_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            metrics_hash TEXT NOT NULL,
            client_signature TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY(node_id) REFERENCES nodes(node_id)
        );
        CREATE TABLE IF NOT EXISTS onboarding_completions (
            node_id TEXT PRIMARY KEY,
            final_access_state TEXT NOT NULL,
            client_signature TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY(node_id) REFERENCES nodes(node_id)
        );
        CREATE TABLE IF NOT EXISTS training_revocations (
            node_id TEXT PRIMARY KEY,
            revoked_round INTEGER NOT NULL,
            reason TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY(node_id) REFERENCES nodes(node_id)
        );
        CREATE TABLE IF NOT EXISTS fit_commitments (
            node_id TEXT NOT NULL,
            server_round INTEGER NOT NULL,
            update_hash TEXT NOT NULL,
            parent_hash TEXT NOT NULL,
            signature TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(node_id, server_round),
            FOREIGN KEY(node_id) REFERENCES nodes(node_id)
        );
        CREATE TABLE IF NOT EXISTS accepted_fit_updates (
            node_id TEXT NOT NULL,
            server_round INTEGER NOT NULL,
            update_hash TEXT NOT NULL,
            created_at REAL NOT NULL,
            aggregation_authorized INTEGER NOT NULL DEFAULT 1,
            aggregation_weight REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY(node_id, server_round),
            FOREIGN KEY(node_id, server_round) REFERENCES fit_commitments(node_id, server_round)
        );
        CREATE TABLE IF NOT EXISTS aggregated_fit_updates (
            node_id TEXT NOT NULL,
            server_round INTEGER NOT NULL,
            update_hash TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(node_id, server_round),
            FOREIGN KEY(node_id, server_round) REFERENCES accepted_fit_updates(node_id, server_round)
        );
        CREATE TABLE IF NOT EXISTS training_evidence (
            node_id TEXT NOT NULL,
            server_round INTEGER NOT NULL,
            outcome TEXT NOT NULL,
            evidence_weight REAL NOT NULL,
            reason TEXT,
            created_at REAL NOT NULL,
            PRIMARY KEY(node_id, server_round),
            FOREIGN KEY(node_id) REFERENCES nodes(node_id)
        );
        CREATE TABLE IF NOT EXISTS node_penalties (
            node_id TEXT PRIMARY KEY,
            penalty_debt REAL NOT NULL,
            hard_failures INTEGER NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY(node_id) REFERENCES nodes(node_id)
        );
        CREATE TABLE IF NOT EXISTS verified_evidence_sources (
            task_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            verifier_signature TEXT,
            verified_at REAL NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(task_id)
        );
        CREATE INDEX IF NOT EXISTS evidence_sources_node_source
            ON verified_evidence_sources(node_id,source_id);
        """)
        # Forward-compatible migration for databases created by an earlier prototype.
        node_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(nodes)")
        }
        if "registration_signature" not in node_columns:
            self.connection.execute(
                "ALTER TABLE nodes ADD COLUMN registration_signature TEXT"
            )
        columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(tasks)")
        }
        for column in ("receipt_signature", "result_signature", "work_product", "reported_quality", "shadow_update_json"):
            if column not in columns:
                self.connection.execute(
                    f"ALTER TABLE tasks ADD COLUMN {column} TEXT"
                )
        source_columns = {
            row["name"] for row in self.connection.execute(
                "PRAGMA table_info(verified_evidence_sources)"
            )
        }
        if "verifier_signature" not in source_columns:
            self.connection.execute(
                "ALTER TABLE verified_evidence_sources ADD COLUMN verifier_signature TEXT"
            )
        accepted_columns = {
            row["name"] for row in self.connection.execute(
                "PRAGMA table_info(accepted_fit_updates)"
            )
        }
        if "aggregation_authorized" not in accepted_columns:
            self.connection.execute(
                """ALTER TABLE accepted_fit_updates
                   ADD COLUMN aggregation_authorized INTEGER NOT NULL DEFAULT 1"""
            )
        if "aggregation_weight" not in accepted_columns:
            self.connection.execute(
                """ALTER TABLE accepted_fit_updates
                   ADD COLUMN aggregation_weight REAL NOT NULL DEFAULT 1.0"""
            )

    def verify_fit_commitment(
        self, node_id: str, server_round: int, update_hash: str,
        parent_hash: str, signature: str,
    ) -> dict[str, Any]:
        """Check a signed update against an admitted identity exactly once per round."""
        if server_round < 1 or len(update_hash) != 64 or len(parent_hash) != 64:
            raise ValueError("invalid fit commitment")
        with self.lock:
            node = self.connection.execute(
                "SELECT public_key FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            state = self.node_status(node_id, expire=False)["access_state"] if node else None
            if not node or state not in {"LIMITED", "ADMITTED"}:
                raise ValueError("node is not currently eligible for training")
            payload = {
                "action": "FIT_COMMITMENT", "node_id": node_id,
                "server_round": server_round, "update_hash": update_hash,
                "parent_hash": parent_hash,
            }
            verify_signature(node["public_key"], payload, signature)
            now = time.time()
            try:
                self.connection.execute(
                    "INSERT INTO fit_commitments VALUES (?,?,?,?,?,?)",
                    (node_id, server_round, update_hash, parent_hash, signature, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate fit commitment for node and round") from exc
            self._append_event("FIT_COMMITMENT", node_id, None, payload, now)
        return {"node_id": node_id, "verified": True, "access_state": state}

    def revoke_training_access(
        self, node_id: str, server_round: int, reason: str,
    ) -> dict[str, Any]:
        if server_round < 1 or reason not in {
            "EXCESS_UPDATE_NORM", "VALIDATION_LOSS", "INVALID_UPDATE",
            "CUMULATIVE_DIRECTIONAL_RISK",
        }:
            raise ValueError("invalid revocation request")
        with self.lock:
            if not self.connection.execute(
                "SELECT 1 FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone():
                raise ValueError("unknown node")
            existing = self.connection.execute(
                "SELECT 1 FROM training_revocations WHERE node_id=?", (node_id,)
            ).fetchone()
            if existing:
                status = self.node_status(node_id, expire=False)
                return {
                    "node_id": node_id, "revoked": True,
                    "access_state": status["access_state"],
                    "penalty_debt": status["penalty_debt"],
                }
            now = time.time()
            self.connection.execute(
                "INSERT INTO training_revocations VALUES (?,?,?,?)",
                (node_id, server_round, reason, now),
            )
            self.connection.execute(
                """INSERT OR REPLACE INTO training_evidence
                   (node_id,server_round,outcome,evidence_weight,reason,created_at)
                   VALUES (?,?,?,?,?,?)""",
                (node_id, server_round, "HARD_FAILURE", -1.0, reason, now),
            )
            self.connection.execute(
                """INSERT INTO node_penalties
                   (node_id,penalty_debt,hard_failures,updated_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(node_id) DO UPDATE SET
                     penalty_debt=node_penalties.penalty_debt+excluded.penalty_debt,
                     hard_failures=node_penalties.hard_failures+1,
                     updated_at=excluded.updated_at""",
                (node_id, 1.0, 1, now),
            )
            self._append_event(
                "TRAINING_ACCESS_REVOKED", node_id, None,
                {"server_round": server_round, "reason": reason}, now,
            )
        status = self.node_status(node_id, expire=False)
        return {
            "node_id": node_id, "revoked": True,
            "access_state": status["access_state"],
            "penalty_debt": status["penalty_debt"],
        }

    def remove_baseline_participant(
        self, node_id: str, server_round: int, reason: str,
    ) -> dict[str, Any]:
        """Remove an RFFL baseline participant without proposed-method penalties.

        RFFL's low-reputation rule governs future participation.  It is not a
        structural hard failure in the proposed access mechanism, so it must
        not create HARD_FAILURE evidence, penalty debt, or a hard-failure count.
        """
        if (self.mechanism_variant != "rffl_reputation"
                or server_round < 1 or reason != "RFFL_LOW_REPUTATION"):
            raise ValueError("invalid baseline removal request")
        with self.lock:
            if not self.connection.execute(
                "SELECT 1 FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone():
                raise ValueError("unknown node")
            existing = self.connection.execute(
                "SELECT 1 FROM training_revocations WHERE node_id=?", (node_id,)
            ).fetchone()
            if not existing:
                now = time.time()
                self.connection.execute(
                    "INSERT INTO training_revocations VALUES (?,?,?,?)",
                    (node_id, server_round, reason, now),
                )
                self._append_event(
                    "BASELINE_PARTICIPANT_REMOVED", node_id, None,
                    {"server_round": server_round, "reason": reason}, now,
                )
        status = self.node_status(node_id, expire=False)
        return {
            "node_id": node_id, "removed": True,
            "access_state": status["access_state"],
            "penalty_debt": status["penalty_debt"],
            "hard_failures": status["hard_failures"],
        }

    def accept_fit_update(self, node_id: str, server_round: int, update_hash: str) -> dict[str, Any]:
        with self.lock:
            commitment = self.connection.execute(
                "SELECT update_hash FROM fit_commitments WHERE node_id=? AND server_round=?",
                (node_id, server_round),
            ).fetchone()
            state = self.node_status(node_id, expire=False)["access_state"]
            if (not commitment or commitment["update_hash"] != update_hash
                    or state not in {"LIMITED", "ADMITTED"}):
                raise ValueError("no currently eligible signed fit for validation")
            now = time.time()
            try:
                self.connection.execute(
                    """INSERT INTO accepted_fit_updates
                       (node_id,server_round,update_hash,created_at,
                        aggregation_authorized,aggregation_weight)
                       VALUES (?,?,?,?,0,1.0)""",
                    (node_id, server_round, update_hash, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate accepted fit update") from exc
            self._append_event(
                "FIT_ACCEPTED", node_id, None,
                {"server_round": server_round, "update_hash": update_hash}, now,
            )
            evidence_outcome = (
                "STRUCTURALLY_ACCEPTED"
                if is_access_core_variant(self.mechanism_variant)
                else "VERIFIED_CLEAN"
            )
            self.connection.execute(
                """INSERT INTO training_evidence
                   (node_id,server_round,outcome,evidence_weight,reason,created_at)
                   VALUES (?,?,?,?,?,?)""",
                (node_id, server_round, evidence_outcome, 1.0, None, now),
            )
            updated_status = self.node_status(node_id, expire=False)
            updated_state = updated_status["access_state"]
            # A structurally accepted LIMITED update advances the observation
            # window. It is not called behaviorally clean when online screening
            # is disabled. The update completing promotion receives full weight.
            aggregation_authorized = updated_state in {"LIMITED", "ADMITTED"}
            aggregation_weight = (
                self.limited_aggregation_weight
                if (
                    state == "LIMITED" and updated_state == "LIMITED"
                ) or (
                    uses_freshness_lease(self.mechanism_variant)
                    and updated_state == "LIMITED"
                ) else 1.0
            )
            self.connection.execute(
                """UPDATE accepted_fit_updates
                   SET aggregation_authorized=?,aggregation_weight=?
                   WHERE node_id=? AND server_round=?""",
                (int(aggregation_authorized), aggregation_weight,
                 node_id, server_round),
            )
            access_transition = ""
            if state == "LIMITED" and updated_state == "ADMITTED":
                renewed = (
                    uses_freshness_lease(self.mechanism_variant)
                    and int(updated_status["structurally_accepted_training_updates"])
                    > self.min_limited_observation_updates
                )
                event_type = "ACCESS_RENEWED" if renewed else "ACCESS_PROMOTED"
                access_transition = "RENEWED" if renewed else "PROMOTED"
                self._append_event(
                    event_type, node_id, None,
                    {
                        "server_round": server_round,
                        "from_state": "LIMITED",
                        "to_state": "ADMITTED",
                        "structurally_accepted_observation_updates": (
                            self.min_limited_observation_updates
                        ),
                    }, now,
                )
            elif state == "ADMITTED" and updated_state == "LIMITED":
                access_transition = "LEASE_EXPIRED"
                self._append_event(
                    "ACCESS_LEASE_EXPIRED", node_id, None,
                    {
                        "server_round": server_round,
                        "from_state": "ADMITTED",
                        "to_state": "LIMITED",
                        "full_updates_per_lease": self.access_lease_full_updates,
                    }, now,
                )
        return {
            "node_id": node_id,
            "accepted": True,
            "access_state": updated_state,
            "aggregation_authorized": aggregation_authorized,
            "aggregation_weight": aggregation_weight,
            "training_evidence_outcome": evidence_outcome,
            "access_transition": access_transition,
        }

    def record_fit_aggregation(
        self, node_id: str, server_round: int, update_hash: str,
    ) -> dict[str, Any]:
        """Record actual aggregation separately from pre-aggregation authorization."""
        with self.lock:
            accepted = self.connection.execute(
                """SELECT update_hash,aggregation_authorized,aggregation_weight
                   FROM accepted_fit_updates
                   WHERE node_id=? AND server_round=?""",
                (node_id, server_round),
            ).fetchone()
            if (not accepted or accepted["update_hash"] != update_hash
                    or not accepted["aggregation_authorized"]):
                raise ValueError("fit update was not authorized for aggregation")
            now = time.time()
            try:
                self.connection.execute(
                    "INSERT INTO aggregated_fit_updates VALUES (?,?,?,?)",
                    (node_id, server_round, update_hash, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate aggregated fit update") from exc
            self._append_event(
                "FIT_AGGREGATED", node_id, None,
                {
                    "server_round": server_round, "update_hash": update_hash,
                    "aggregation_weight": float(accepted["aggregation_weight"]),
                }, now,
            )
        return {"node_id": node_id, "aggregated": True}

    def _append_event(
        self, event_type: str, node_id: str, task_id: str | None,
        payload: dict[str, Any], now: float,
    ) -> None:
        previous = self.connection.execute(
            "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous["event_hash"] if previous else "GENESIS"
        body = {
            "event_type": event_type,
            "node_id": node_id,
            "task_id": task_id,
            "payload": payload,
            "previous_hash": previous_hash,
            "created_at": round(float(now), 6),
        }
        event_hash = digest(body)
        self.connection.execute(
            """INSERT INTO audit_events
               (event_type,node_id,task_id,payload_json,previous_hash,event_hash,created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (event_type, node_id, task_id, json.dumps(payload, sort_keys=True),
             previous_hash, event_hash, round(float(now), 6)),
        )

    def register_node(
        self, node_id: str, profile: str, public_key: str,
        registration_signature: str, now: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        payload = {
            "action": "REGISTER",
            "node_id": node_id,
            "profile": profile,
            "public_key": public_key,
        }
        verify_signature(public_key, payload, registration_signature)
        with self.lock:
            existing = self.connection.execute(
                "SELECT public_key,profile FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if existing and existing["public_key"] != public_key:
                raise ValueError("node_id is already bound to another public key")
            if not existing:
                self.connection.execute(
                    """INSERT INTO nodes
                       (node_id,public_key,profile,registered_at,registration_signature)
                       VALUES (?,?,?,?,?)""",
                    (node_id, public_key, profile, now, registration_signature),
                )
                self._append_event(
                    "NODE_REGISTERED", node_id, None,
                    {**payload, "client_signature": registration_signature}, now,
                )
        return {"node_id": node_id, "registered": True}

    def expire_due_tasks(self, now: float | None = None) -> int:
        now = time.time() if now is None else float(now)
        changed = 0
        with self.lock:
            due = self.connection.execute(
                """SELECT * FROM tasks
                   WHERE deadline_at < ? AND status IN ('ISSUED','ACKNOWLEDGED')""",
                (now,),
            ).fetchall()
            for task in due:
                status = (
                    "ACKNOWLEDGED_TIMEOUT"
                    if task["status"] == "ACKNOWLEDGED"
                    else "DELIVERY_UNCONFIRMED"
                )
                self.connection.execute(
                    "UPDATE tasks SET status=? WHERE task_id=?",
                    (status, task["task_id"]),
                )
                self._append_event(
                    status, task["node_id"], task["task_id"],
                    {"deadline_at": task["deadline_at"]}, now,
                )
                changed += 1
        return changed

    def issue_task(self, node_id: str, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        self.expire_due_tasks(now)
        state = self.node_status(node_id, expire=False)
        if state["access_state"] != "PROBATION":
            return {"access_state": state["access_state"], "task": None}
        with self.lock:
            node = self.connection.execute(
                "SELECT 1 FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if not node:
                raise ValueError("unknown node")
            task_index = self.connection.execute(
                "SELECT COUNT(*) AS count FROM tasks WHERE node_id=?", (node_id,)
            ).fetchone()["count"]
            task_id = str(uuid.uuid4())
            # Every participant gets the same assignment policy.  The stored
            # profile is experimental ground truth and must not steer access.
            task_type = TASK_SEQUENCE[int(task_index) % len(TASK_SEQUENCE)]
            payload = {
                "action": "TASK",
                "task_id": task_id,
                "node_id": node_id,
                "task_type": task_type,
                "nonce": uuid.uuid4().hex,
                "issued_at": round(now, 6),
                "deadline_at": round(now + self.task_deadline_seconds, 6),
            }
            assignment_hash = digest(payload)
            signature = sign_payload(self.server_key, payload)
            self.connection.execute(
                """INSERT INTO tasks
                   (task_id,node_id,task_type,nonce,issued_at,deadline_at,
                    assignment_hash,server_signature,status)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (task_id, node_id, task_type, payload["nonce"], now,
                 payload["deadline_at"], assignment_hash, signature, "ISSUED"),
            )
            self._append_event(
                "TASK_ISSUED", node_id, task_id,
                {"assignment_hash": assignment_hash, "task_type": task_type}, now,
            )
        return {
            "access_state": "PROBATION",
            "task": payload,
            "assignment_hash": assignment_hash,
            "server_signature": signature,
            "server_public_key": self.server_public_key,
        }

    def acknowledge(
        self, node_id: str, task_id: str, assignment_hash: str,
        signature: str, now: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        with self.lock:
            task = self.connection.execute(
                "SELECT * FROM tasks WHERE task_id=? AND node_id=?",
                (task_id, node_id),
            ).fetchone()
            if not task or task["status"] != "ISSUED":
                raise ValueError("task is missing or no longer acknowledgeable")
            if now > float(task["deadline_at"]):
                self.connection.execute(
                    "UPDATE tasks SET status='DELIVERY_UNCONFIRMED' WHERE task_id=?",
                    (task_id,),
                )
                self._append_event(
                    "DELIVERY_UNCONFIRMED", node_id, task_id,
                    {"deadline_at": task["deadline_at"]}, now,
                )
                raise ValueError("task acknowledgement arrived after its deadline")
            if task["assignment_hash"] != assignment_hash:
                raise ValueError("assignment hash mismatch")
            public_key = self.connection.execute(
                "SELECT public_key FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()["public_key"]
            payload = {
                "action": "ACK",
                "task_id": task_id,
                "assignment_hash": assignment_hash,
            }
            verify_signature(public_key, payload, signature)
            self.connection.execute(
                """UPDATE tasks
                   SET status='ACKNOWLEDGED',acknowledged_at=?,receipt_signature=?
                   WHERE task_id=?""",
                (now, signature, task_id),
            )
            self._append_event(
                "TASK_ACKNOWLEDGED", node_id, task_id,
                {**payload, "client_signature": signature}, now,
            )
        return {"task_id": task_id, "acknowledged": True}

    def submit_result(
        self, node_id: str, task_id: str, quality: float, result_hash: str,
        signature: str, work_product: str, now: float | None = None,
        shadow_update: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        quality = float(np.clip(quality, 0.0, 1.0))
        with self.lock:
            task = self.connection.execute(
                "SELECT * FROM tasks WHERE task_id=? AND node_id=?",
                (task_id, node_id),
            ).fetchone()
            if not task or task["status"] != "ACKNOWLEDGED":
                raise ValueError("task is missing or has no valid receipt")
            if now > float(task["deadline_at"]):
                self.connection.execute(
                    "UPDATE tasks SET status='ACKNOWLEDGED_TIMEOUT' WHERE task_id=?",
                    (task_id,),
                )
                self._append_event(
                    "ACKNOWLEDGED_TIMEOUT", node_id, task_id,
                    {"deadline_at": task["deadline_at"]}, now,
                )
                raise ValueError("task result arrived after its deadline")
            public_key = self.connection.execute(
                "SELECT public_key FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()["public_key"]
            payload = {
                "action": "RESULT",
                "task_id": task_id,
                "quality": round(quality, 6),
                "result_hash": result_hash,
                "work_product": work_product,
            }
            if shadow_update is not None:
                payload["shadow_update"] = shadow_update
            verify_signature(public_key, payload, signature)
            expected = challenge_answer(dict(task))
            verified = work_product == expected
            shadow_source_id = None
            if self.shadow_verifier_key:
                if task["task_type"] == "shadow_update":
                    if shadow_update is None:
                        raise ValueError("verified shadow update required")
                    shadow_source_id = inspect_shadow_artifact(shadow_update)
                elif shadow_update is not None:
                    raise ValueError("shadow update on non-shadow task")
            if not verified and uses_result_verification(self.mechanism_variant):
                self.connection.execute(
                    """UPDATE tasks SET status='INVALID_RESULT',completed_at=?,
                       quality=0,reported_quality=?,result_hash=?,work_product=?,result_signature=?,shadow_update_json=?
                       WHERE task_id=?""",
                    (now, quality, result_hash, work_product, signature,
                     json.dumps(shadow_update, separators=(",", ":"))
                     if shadow_update is not None else None, task_id),
                )
                self._append_event(
                    "RESULT_REJECTED", node_id, task_id,
                    {**payload, "client_signature": signature}, now,
                )
                return {"task_id": task_id, "completed": False, "verified": False}
            # Client quality is a self-report.  Only verified protocol work
            # receives a positive reputation observation in this prototype.
            # Same score mapping in the ablation: only validation differs.
            observed_quality = 1.0
            self.connection.execute(
                """UPDATE tasks SET status='COMPLETED',completed_at=?,quality=?,reported_quality=?,result_hash=?,
                                      work_product=?,result_signature=?,shadow_update_json=?
                   WHERE task_id=?""",
                (now, observed_quality, quality, result_hash, work_product, signature,
                 json.dumps(shadow_update, separators=(",", ":")) if shadow_update is not None else None,
                 task_id),
            )
            self._append_event(
                "RESULT_COMPLETED", node_id, task_id,
                {**payload, "client_signature": signature}, now,
            )
            if self.shadow_verifier_key:
                source_id = shadow_source_id or digest({
                    "task_id": task_id, "assignment_hash": task["assignment_hash"],
                })
                source_claim = {
                    "action": "VERIFY_SOURCE", "node_id": node_id,
                    "task_id": task_id, "source_id": source_id,
                    "result_hash": result_hash, "work_product": work_product,
                }
                self.record_verified_source(
                    node_id, task_id, source_id,
                    sign_payload(self.shadow_verifier_key, source_claim), now=now,
                )
        return {"task_id": task_id, "completed": True, "verified": verified}

    def record_verified_source(
        self, node_id: str, task_id: str, source_id: str,
        verifier_signature: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Accept a source identity ONLY from an independent trusted verifier.

        The verifier key must be configured separately from node/control keys.
        This checks origin and task binding, NOT whether the issuer actually
        inspected underlying work or used a sound source-identity definition.
        """
        if not isinstance(source_id, str) or not (1 <= len(source_id) <= 128):
            raise ValueError("invalid verified source identity")
        if not self.source_verifier_public_key:
            raise ValueError("no trusted source verifier configured")
        now = time.time() if now is None else float(now)
        with self.lock:
            task = self.connection.execute(
                "SELECT status,result_hash,work_product FROM tasks WHERE task_id=? AND node_id=?",
                (task_id, node_id),
            ).fetchone()
            if not task or task["status"] != "COMPLETED":
                raise ValueError("only completed tasks can receive verified provenance")
            source_claim = {
                "action": "VERIFY_SOURCE", "node_id": node_id,
                "task_id": task_id, "source_id": source_id,
                "result_hash": task["result_hash"],
                "work_product": task["work_product"],
            }
            verify_signature(
                self.source_verifier_public_key, source_claim,
                verifier_signature,
            )
            existing = self.connection.execute(
                "SELECT source_id FROM verified_evidence_sources WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if existing:
                if existing["source_id"] != source_id:
                    raise ValueError("verified source cannot be changed")
                return {"task_id": task_id, "recorded": True}
            self.connection.execute(
                """INSERT INTO verified_evidence_sources
                   (task_id,node_id,source_id,verifier_signature,verified_at)
                   VALUES (?,?,?,?,?)""",
                (task_id, node_id, source_id, verifier_signature, now),
            )
            self._append_event(
                "SOURCE_VERIFIED", node_id, task_id,
                {"source_id": source_id, "verifier_signature": verifier_signature}, now,
            )
        return {"task_id": task_id, "recorded": True}

    def node_status(self, node_id: str, expire: bool = True) -> dict[str, Any]:
        if expire:
            self.expire_due_tasks()
        with self.lock:
            node = self.connection.execute(
                "SELECT * FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if not node:
                raise ValueError("unknown node")
            tasks = self.connection.execute(
                "SELECT * FROM tasks WHERE node_id=? ORDER BY issued_at,task_id", (node_id,)
            ).fetchall()
            verified_sources = {
                row["task_id"]: row["source_id"]
                for row in self.connection.execute(
                    """SELECT task_id,source_id FROM verified_evidence_sources
                       WHERE node_id=? AND verifier_signature IS NOT NULL""",
                    (node_id,),
                )
            }
            revocation = self.connection.execute(
                "SELECT revoked_round,reason FROM training_revocations WHERE node_id=?",
                (node_id,),
            ).fetchone()
            accepted_training_evidence = self.connection.execute(
                """SELECT COUNT(*) AS accepted_count FROM training_evidence
                   WHERE node_id=? AND outcome IN
                       ('VERIFIED_CLEAN','STRUCTURALLY_ACCEPTED')""",
                (node_id,),
            ).fetchone()["accepted_count"]
            penalty = self.connection.execute(
                """SELECT penalty_debt,hard_failures FROM node_penalties
                   WHERE node_id=?""",
                (node_id,),
            ).fetchone()
        completed = [task for task in tasks if task["status"] == "COMPLETED"]
        credited = completed
        if self.mechanism_variant == "full_shadow_no_dedup":
            credited = [task for task in completed if task["task_id"] in verified_sources]
        if self.mechanism_variant in {"full_provenance", "full_shadow_provenance"}:
            seen_sources: set[str] = set()
            credited = []
            for task in completed:
                source = verified_sources.get(task["task_id"])
                if source is not None and source not in seen_sources:
                    seen_sources.add(source)
                    credited.append(task)
        acknowledged_missing = [
            task for task in tasks if task["status"] in {"ACKNOWLEDGED_TIMEOUT", "INVALID_RESULT"}
        ]
        acknowledged_total = len(completed) + len(acknowledged_missing)
        missing_rate = (
            len(acknowledged_missing) / acknowledged_total
            if acknowledged_total else 0.0
        )

        dirichlet_prior = np.full(5, 0.4, dtype=float)
        dirichlet_alpha = dirichlet_prior.copy()
        # The plain Beta Reputation System baseline uses the same total prior
        # strength (2.0), temporal decay, evidence stream and repeat discount as
        # the Dirichlet estimator.  Only its binary observation model differs.
        beta_prior = np.asarray([1.0, 1.0], dtype=float)
        beta_alpha = beta_prior.copy()
        type_counts = {task_type: 0 for task_type in TASK_SEQUENCE}
        evidence_mass = 0.0
        for task in credited:
            task_type = task["task_type"]
            previous = type_counts[task_type]
            repeat_discount = (
                (1.0 + previous) ** -0.5
                if uses_repeat_decay(self.mechanism_variant) else 1.0
            )
            contribution = 0.95 * TASK_WEIGHTS[task_type] * repeat_discount
            type_counts[task_type] += 1
            evidence_mass += contribution
            semantic_separation = uses_semantic_separation(
                self.mechanism_variant
            )
            if task_type != "attestation" or not semantic_separation:
                if self.trust_estimator == "beta_mean":
                    beta_alpha = beta_prior + 0.97 * (beta_alpha - beta_prior)
                    quality = float(np.clip(task["quality"], 0.0, 1.0))
                    # A graded satisfaction score contributes fractional
                    # positive/negative evidence.  This avoids weakening the
                    # Beta baseline with an arbitrary binarization threshold.
                    beta_alpha[0] += contribution * quality
                    beta_alpha[1] += contribution * (1.0 - quality)
                else:
                    dirichlet_alpha = (
                        dirichlet_prior
                        + 0.97 * (dirichlet_alpha - dirichlet_prior)
                    )
                    level = int(np.digitize(float(task["quality"]), DIRICHLET_BINS))
                    dirichlet_alpha[level] += contribution

        if self.trust_estimator == "beta_mean":
            positive, negative = map(float, beta_alpha)
            total = positive + negative
            history_mean = positive / total
            variance = positive * negative / (total ** 2 * (total + 1.0))
            effective_evidence = max(0.0, float((beta_alpha - beta_prior).sum()))
        else:
            total = float(dirichlet_alpha.sum())
            weighted = float(UTILITY @ dirichlet_alpha)
            history_mean = weighted / total
            variance = (
                total * float((UTILITY ** 2) @ dirichlet_alpha) - weighted ** 2
            ) / (total ** 2 * (total + 1.0))
            effective_evidence = max(
                0.0, float((dirichlet_alpha - dirichlet_prior).sum())
            )
        history_std = math.sqrt(max(variance, 0.0))
        history_lcb = max(0.0, history_mean - 1.644854 * history_std)
        history_decision_score = (
            history_lcb
            if self.trust_estimator == "dirichlet_lcb"
            else history_mean
        )
        maturity = (
            effective_evidence / (effective_evidence + 10.0)
            if effective_evidence else 0.0
        )
        evidence_types = sum(count > 0 for count in type_counts.values())
        trust_score = float(np.clip(
            0.65 * 0.80 + 0.35 * history_decision_score - 0.20 * missing_rate,
            0.0, 1.0,
        ))
        diversity_ready = (
            evidence_types >= 3
            if uses_evidence_diversity(self.mechanism_variant) else True
        )
        if self.mechanism_variant == "access_attestation_only":
            # Conventional one-shot gate: a valid signed attestation is enough.
            # It intentionally ignores longitudinal behavioral evidence.
            graduation_ready = any(
                task["task_type"] == "attestation" for task in credited
            )
        elif self.mechanism_variant == "access_static_multisource":
            # Static multi-source baseline: all four evidence labels must appear,
            # but there is no posterior uncertainty, repeat decay, or maturity.
            graduation_ready = evidence_types >= len(TASK_WEIGHTS)
        elif self.mechanism_variant == "access_naive":
            # A deliberately permissive onboarding baseline: count accepted
            # task results, but do not use diversity, posterior uncertainty or
            # maturity.  It still retains signed task receipts so transport
            # integrity is not confused with the proposed trust mechanism.
            graduation_ready = len(completed) >= self.min_completed_tasks
        else:
            graduation_ready = (
                len(credited) >= self.min_completed_tasks
                and evidence_mass >= self.min_evidence_mass
                and diversity_ready
                and history_std <= 0.16
                and maturity >= 0.15
                and acknowledged_total >= 4
                and missing_rate <= 0.25
            )
        if self.mechanism_variant in {
            "access_attestation_only", "access_static_multisource"
        }:
            access_ready = graduation_ready
        else:
            access_ready = graduation_ready and trust_score >= 0.60
        budget_enabled = uses_probation_budget(self.mechanism_variant)
        budget_exhausted = (
            budget_enabled and len(tasks) >= self.max_probation_tasks
            and not access_ready
        )
        oracle_decision = self.mechanism_variant in {
            "oracle_benign", "oracle_filter"
        }
        progressive = is_progressive_variant(self.mechanism_variant)
        penalty_debt = float(penalty["penalty_debt"]) if penalty else 0.0
        hard_failures = int(penalty["hard_failures"]) if penalty else 0
        if revocation is not None or penalty_debt > 0.0:
            graduation_ready = False
            access_state = "QUARANTINE"
        elif self.mechanism_variant == "oracle_benign":
            # Upper utility bound: every participant is known benign and admitted.
            graduation_ready = True
            budget_exhausted = False
            access_state = "ADMITTED"
        elif self.mechanism_variant == "oracle_filter":
            # Ideal security bound: ground-truth identity is available to the controller.
            graduation_ready = node["profile"] == "honest"
            budget_exhausted = False
            access_state = "ADMITTED" if graduation_ready else "QUARANTINE"
        elif budget_exhausted or (acknowledged_total >= 10 and missing_rate > 0.50):
            access_state = "QUARANTINE"
        elif access_ready:
            if not progressive or node_id in self.incumbent_node_ids:
                access_state = "ADMITTED"
            elif uses_freshness_lease(self.mechanism_variant):
                if accepted_training_evidence < self.min_limited_observation_updates:
                    access_state = "LIMITED"
                else:
                    cycle_length = (
                        self.access_lease_full_updates
                        + self.min_limited_observation_updates
                    )
                    cycle_position = (
                        accepted_training_evidence
                        - self.min_limited_observation_updates
                    ) % cycle_length
                    access_state = (
                        "ADMITTED"
                        if cycle_position < self.access_lease_full_updates
                        else "LIMITED"
                    )
            else:
                access_state = (
                    "ADMITTED"
                    if accepted_training_evidence >= self.min_limited_observation_updates
                    else "LIMITED"
                )
        else:
            access_state = "PROBATION"
        return {
            "node_id": node_id,
            "profile": node["profile"],
            "cohort_role": (
                "incumbent" if node_id in self.incumbent_node_ids else "newcomer"
            ),
            "mechanism_variant": self.mechanism_variant,
            "trust_estimator": self.trust_estimator,
            "access_state": access_state,
            "trust_score": trust_score,
            "history_mean": history_mean,
            "history_decision_score": history_decision_score,
            "history_lcb": history_lcb,
            "history_std": history_std,
            "history_evidence_maturity": maturity,
            "completed_tasks": len(completed),
            "credited_independent_tasks": len(credited),
            "verified_source_records": sum(
                task["task_id"] in verified_sources for task in completed
            ),
            "evidence_types": evidence_types,
            "diversity_ready": diversity_ready,
            "evidence_mass": evidence_mass,
            "minimum_evidence_mass_required": self.min_evidence_mass,
            "acknowledged_tasks": acknowledged_total,
            "acknowledged_missing_results": len(acknowledged_missing),
            "acknowledged_missing_rate": missing_rate,
            "graduation_ready": graduation_ready,
            "training_clean_evidence": int(accepted_training_evidence),
            # V39 uses the precise name below in new reports.  The legacy key
            # remains for compatibility; passing structural screening is not
            # evidence that an update is behaviorally benign.
            "structurally_accepted_training_updates": int(accepted_training_evidence),
            "structurally_accepted_observation_updates": int(
                accepted_training_evidence
            ),
            "limited_clean_updates_required": (
                self.min_limited_observation_updates if progressive else 0
            ),
            "limited_observation_updates_required": (
                self.min_limited_observation_updates if progressive else 0
            ),
            "freshness_lease_enabled": uses_freshness_lease(
                self.mechanism_variant
            ),
            "access_lease_full_updates": (
                self.access_lease_full_updates
                if uses_freshness_lease(self.mechanism_variant) else 0
            ),
            "penalty_debt": penalty_debt,
            "hard_failures": hard_failures,
            "budget_exhausted": budget_exhausted,
            "oracle_decision": oracle_decision,
            "revoked_round": revocation["revoked_round"] if revocation else None,
            "revocation_reason": revocation["reason"] if revocation else None,
        }

    def record_flower_event(
        self, node_id: str, event_id: str, event_type: str,
        metrics_hash: str, signature: str, now: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        if event_type not in {"FIT", "EVALUATE"}:
            raise ValueError("unsupported Flower event type")
        payload = {
            "action": "FLOWER_EVENT",
            "event_id": event_id,
            "node_id": node_id,
            "event_type": event_type,
            "metrics_hash": metrics_hash,
        }
        with self.lock:
            node = self.connection.execute(
                "SELECT public_key FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if not node:
                raise ValueError("unknown node")
            if self.node_status(node_id, expire=False)["access_state"] not in {"LIMITED", "ADMITTED"}:
                raise ValueError("node is not eligible for Flower participation")
            verify_signature(node["public_key"], payload, signature)
            self.connection.execute(
                "INSERT INTO flower_events VALUES (?,?,?,?,?,?)",
                (event_id, node_id, event_type, metrics_hash, signature, now),
            )
            self._append_event(
                "FLOWER_PARTICIPATION", node_id, None,
                {**payload, "client_signature": signature}, now,
            )
        return {"event_id": event_id, "recorded": True}

    def mark_onboarding_finished(
        self, node_id: str, final_access_state: str,
        signature: str, now: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        payload = {
            "action": "ONBOARDING_FINISHED",
            "node_id": node_id,
            "final_access_state": final_access_state,
        }
        with self.lock:
            node = self.connection.execute(
                "SELECT public_key FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if not node:
                raise ValueError("unknown node")
            actual_state = self.node_status(node_id, expire=False)["access_state"]
            if actual_state != final_access_state:
                raise ValueError("claimed onboarding state does not match controller state")
            verify_signature(node["public_key"], payload, signature)
            self.connection.execute(
                """INSERT INTO onboarding_completions
                   (node_id,final_access_state,client_signature,created_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(node_id) DO UPDATE SET
                       final_access_state=excluded.final_access_state,
                       client_signature=excluded.client_signature,
                       created_at=excluded.created_at""",
                (node_id, final_access_state, signature, now),
            )
            self._append_event(
                "ONBOARDING_FINISHED", node_id, None,
                {**payload, "client_signature": signature}, now,
            )
        return {"node_id": node_id, "recorded": True}

    def barrier_status(self) -> dict[str, Any]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT node_id,final_access_state FROM onboarding_completions"
            ).fetchall()
        finished = len(rows)
        expected = self.expected_experiment_nodes
        released = expected > 0 and finished >= expected
        eligible_nodes = sum(
            row["final_access_state"] in {"LIMITED", "ADMITTED"} for row in rows
        )
        return {
            "released": released,
            "expected_nodes": expected,
            "finished_nodes": finished,
            # Kept as a compatibility alias for the Flower startup barrier.
            "admitted_nodes": eligible_nodes,
            "training_eligible_nodes": eligible_nodes,
            "limited_nodes": sum(row["final_access_state"] == "LIMITED" for row in rows),
            "full_nodes": sum(row["final_access_state"] == "ADMITTED" for row in rows),
            "states": {row["node_id"]: row["final_access_state"] for row in rows},
        }

    def verify_chain(self) -> dict[str, Any]:
        with self.lock:
            events = self.connection.execute(
                "SELECT * FROM audit_events ORDER BY sequence"
            ).fetchall()
        previous_hash = "GENESIS"
        for event in events:
            payload = json.loads(event["payload_json"])
            body = {
                "event_type": event["event_type"],
                "node_id": event["node_id"],
                "task_id": event["task_id"],
                "payload": payload,
                "previous_hash": previous_hash,
                "created_at": event["created_at"],
            }
            expected = digest(body)
            if event["previous_hash"] != previous_hash or event["event_hash"] != expected:
                return {"valid": False, "failed_sequence": event["sequence"]}
            previous_hash = event["event_hash"]
        return {"valid": True, "events": len(events), "head": previous_hash}

    def verify_signed_records(self) -> dict[str, Any]:
        """Verify server assignments and every stored client receipt/result."""
        with self.lock:
            nodes = self.connection.execute(
                "SELECT * FROM nodes ORDER BY registered_at,node_id"
            ).fetchall()
            tasks = self.connection.execute(
                """SELECT tasks.*,nodes.public_key FROM tasks
                   JOIN nodes ON tasks.node_id=nodes.node_id ORDER BY issued_at,task_id"""
            ).fetchall()
            flower_events = self.connection.execute(
                """SELECT flower_events.*,nodes.public_key FROM flower_events
                   JOIN nodes ON flower_events.node_id=nodes.node_id
                   ORDER BY created_at,event_id"""
            ).fetchall()
            fit_commitments = self.connection.execute(
                """SELECT fits.*,nodes.public_key FROM fit_commitments AS fits
                   JOIN nodes ON fits.node_id=nodes.node_id"""
            ).fetchall()
            completions = self.connection.execute(
                """SELECT onboarding_completions.*,nodes.public_key
                   FROM onboarding_completions
                   JOIN nodes ON onboarding_completions.node_id=nodes.node_id
                   ORDER BY created_at,onboarding_completions.node_id"""
            ).fetchall()
            sources = self.connection.execute(
                """SELECT sources.*,tasks.result_hash,tasks.work_product
                   FROM verified_evidence_sources AS sources
                   JOIN tasks ON tasks.task_id=sources.task_id
                   ORDER BY sources.verified_at,sources.task_id"""
            ).fetchall()
        checked = 0
        try:
            for node in nodes:
                if not node["registration_signature"]:
                    raise ValueError(f"registration signature missing: {node['node_id']}")
                registration = {
                    "action": "REGISTER",
                    "node_id": node["node_id"],
                    "profile": node["profile"],
                    "public_key": node["public_key"],
                }
                verify_signature(
                    node["public_key"], registration, node["registration_signature"]
                )
                checked += 1
            for task in tasks:
                assignment = {
                    "action": "TASK",
                    "task_id": task["task_id"],
                    "node_id": task["node_id"],
                    "task_type": task["task_type"],
                    "nonce": task["nonce"],
                    "issued_at": round(float(task["issued_at"]), 6),
                    "deadline_at": round(float(task["deadline_at"]), 6),
                }
                if digest(assignment) != task["assignment_hash"]:
                    raise ValueError(f"assignment hash mismatch: {task['task_id']}")
                verify_signature(
                    self.server_public_key, assignment, task["server_signature"]
                )
                checked += 1
                if task["receipt_signature"]:
                    receipt = {
                        "action": "ACK",
                        "task_id": task["task_id"],
                        "assignment_hash": task["assignment_hash"],
                    }
                    verify_signature(
                        task["public_key"], receipt, task["receipt_signature"]
                    )
                    checked += 1
                if task["result_signature"]:
                    result = {
                        "action": "RESULT", "task_id": task["task_id"],
                        "quality": round(float(
                            task["reported_quality"]
                            if task["reported_quality"] is not None else task["quality"]
                        ), 6),
                        "result_hash": task["result_hash"],
                    }
                    if task["reported_quality"] is not None:
                        result["work_product"] = task["work_product"]
                    if task["shadow_update_json"] is not None:
                        result["shadow_update"] = json.loads(task["shadow_update_json"])
                    verify_signature(
                        task["public_key"], result, task["result_signature"]
                    )
                    checked += 1
            for event in flower_events:
                payload = {
                    "action": "FLOWER_EVENT",
                    "event_id": event["event_id"],
                    "node_id": event["node_id"],
                    "event_type": event["event_type"],
                    "metrics_hash": event["metrics_hash"],
                }
                verify_signature(
                    event["public_key"], payload, event["client_signature"]
                )
                checked += 1
            for fit in fit_commitments:
                payload = {
                    "action": "FIT_COMMITMENT", "node_id": fit["node_id"],
                    "server_round": fit["server_round"],
                    "update_hash": fit["update_hash"],
                    "parent_hash": fit["parent_hash"],
                }
                verify_signature(fit["public_key"], payload, fit["signature"])
                checked += 1
            for completion in completions:
                payload = {
                    "action": "ONBOARDING_FINISHED",
                    "node_id": completion["node_id"],
                    "final_access_state": completion["final_access_state"],
                }
                verify_signature(
                    completion["public_key"], payload,
                    completion["client_signature"],
                )
                checked += 1
            for source in sources:
                if not self.source_verifier_public_key or not source["verifier_signature"]:
                    raise ValueError("source verifier signature missing")
                proof = {
                    "action": "VERIFY_SOURCE", "node_id": source["node_id"],
                    "task_id": source["task_id"], "source_id": source["source_id"],
                    "result_hash": source["result_hash"],
                    "work_product": source["work_product"],
                }
                verify_signature(
                    self.source_verifier_public_key, proof,
                    source["verifier_signature"],
                )
                if self.shadow_verifier_key:
                    task = next(
                        row for row in tasks if row["task_id"] == source["task_id"]
                    )
                    expected_source = (
                        inspect_shadow_artifact(json.loads(task["shadow_update_json"]))
                        if task["task_type"] == "shadow_update"
                        else digest({"task_id": task["task_id"],
                                     "assignment_hash": task["assignment_hash"]})
                    )
                    if expected_source != source["source_id"]:
                        raise ValueError("source does not match submitted work")
                checked += 1
        except Exception as exc:
            return {"valid": False, "checked_signatures": checked, "error": str(exc)}
        return {
            "valid": True,
            "checked_signatures": checked,
            "tasks": len(tasks),
            "flower_events": len(flower_events),
            "onboarding_completions": len(completions),
        }

    def verify_integrity(self) -> dict[str, Any]:
        chain = self.verify_chain()
        signatures = self.verify_signed_records()
        return {
            "valid": bool(chain["valid"] and signatures["valid"]),
            "hash_chain": chain,
            "signatures": signatures,
        }

    def task_records(self, node_id: str) -> list[dict[str, Any]]:
        with self.lock:
            exists = self.connection.execute(
                "SELECT 1 FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if not exists:
                raise ValueError("unknown node")
            tasks = self.connection.execute(
                """SELECT task_id,task_type,nonce,issued_at,deadline_at,
                          assignment_hash,server_signature,status,acknowledged_at,
                          completed_at,quality,reported_quality,result_hash,work_product,receipt_signature,
                          result_signature
                   FROM tasks WHERE node_id=? ORDER BY issued_at,task_id""",
                (node_id,),
            ).fetchall()
        return [dict(task) for task in tasks]

    def experiment_summary(self) -> dict[str, Any]:
        self.expire_due_tasks()
        with self.lock:
            node_ids = [
                row["node_id"] for row in self.connection.execute(
                    "SELECT node_id FROM nodes ORDER BY node_id"
                ).fetchall()
            ]
            participation = {
                row["node_id"]: {
                    "flower_fit_events": int(row["fit_events"]),
                    "flower_evaluate_events": int(row["evaluate_events"]),
                }
                for row in self.connection.execute(
                    """SELECT node_id,
                              SUM(CASE WHEN event_type='FIT' THEN 1 ELSE 0 END) fit_events,
                              SUM(CASE WHEN event_type='EVALUATE' THEN 1 ELSE 0 END) evaluate_events
                       FROM flower_events GROUP BY node_id"""
                ).fetchall()
            }
            aggregated_counts = {
                row["node_id"]: row["accepted_count"]
                for row in self.connection.execute(
                    """SELECT node_id,COUNT(*) AS accepted_count
                       FROM aggregated_fit_updates GROUP BY node_id"""
                ).fetchall()
            }
            aggregated_weight_mass = {
                row["node_id"]: float(row["weight_mass"])
                for row in self.connection.execute(
                    """SELECT aggregated.node_id,
                              SUM(accepted.aggregation_weight) AS weight_mass
                       FROM aggregated_fit_updates AS aggregated
                       JOIN accepted_fit_updates AS accepted
                         ON accepted.node_id=aggregated.node_id
                        AND accepted.server_round=aggregated.server_round
                       GROUP BY aggregated.node_id"""
                ).fetchall()
            }
            completion_info = {
                row["node_id"]: (float(row["duration"]), row["final_access_state"])
                for row in self.connection.execute(
                    """SELECT nodes.node_id,
                              onboarding_completions.created_at-nodes.registered_at duration,
                              onboarding_completions.final_access_state
                       FROM nodes JOIN onboarding_completions
                       ON nodes.node_id=onboarding_completions.node_id"""
                ).fetchall()
            }
        nodes = []
        for node_id in node_ids:
            status = self.node_status(node_id, expire=False)
            status.update(participation.get(node_id, {
                "flower_fit_events": 0, "flower_evaluate_events": 0,
            }))
            status["aggregated_fit_events"] = aggregated_counts.get(node_id, 0)
            status["aggregated_weight_mass"] = aggregated_weight_mass.get(node_id, 0.0)
            completion = completion_info.get(node_id)
            status["onboarding_duration_seconds"] = completion[0] if completion else None
            status["initial_access_state"] = completion[1] if completion else None
            nodes.append(status)
        return {
            "mechanism_variant": self.mechanism_variant,
            "trust_estimator": self.trust_estimator,
            "nodes": nodes,
            "barrier": self.barrier_status(),
            "integrity": self.verify_integrity(),
        }

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import requests

from flower_prototype.access_control import (
    challenge_answer,
    digest,
    load_or_create_private_key,
    public_key_b64,
    sign_payload,
    verify_signature,
)
from flower_prototype.attack_schedule import effective_training_attack
from flower_prototype.real_data import (
    apply_training_attack,
    clean_distribution_drift,
    client_partition,
    dataset_dimensions,
    initial_parameters,
    local_train,
    metrics,
    parameter_digest,
    sign_flip_update,
)


NODE_ID = os.getenv("NODE_ID", "honest-1")
PROFILE = os.getenv("PROFILE", "honest")
ACCESS_URL = os.getenv("ACCESS_URL", "http://access-controller:8000")
FLOWER_SERVER = os.getenv("FLOWER_SERVER", "flower-server:8080")
STATE_DIR = Path(os.getenv("CLIENT_STATE_DIR", "/state"))
MAX_ONBOARDING_TASKS = int(os.getenv("MAX_ONBOARDING_TASKS", "24"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "0.4"))
FLOWER_JOIN_DELAY = float(os.getenv("FLOWER_JOIN_DELAY", "0.0"))
EXPERIMENT_SEED = os.getenv("EXPERIMENT_SEED", "0")
EXPECTED_EXPERIMENT_NODES = int(os.getenv("EXPECTED_EXPERIMENT_NODES", "0"))
BARRIER_TIMEOUT_SECONDS = float(os.getenv("BARRIER_TIMEOUT_SECONDS", "600"))
NETWORK_JITTER_SECONDS = float(os.getenv("NETWORK_JITTER_SECONDS", "0.0"))
BENIGN_UNCONFIRMED_RATE = float(os.getenv("BENIGN_UNCONFIRMED_RATE", "0.0"))
DATA_MODE = os.getenv("DATA_MODE", "synthetic")
DATASET_NAME = os.getenv("DATASET_NAME", "digits")
DATASET_PATH = os.getenv("DATASET_PATH", "") or None
PARTITION_ID = int(os.getenv("PARTITION_ID", "0"))
NUM_DATA_PARTITIONS = int(os.getenv("NUM_DATA_PARTITIONS", "6"))
NONIID_ALPHA = float(os.getenv("NONIID_ALPHA", "0.5"))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "2"))
LOCAL_LEARNING_RATE = float(os.getenv("LOCAL_LEARNING_RATE", "0.15"))
LOCAL_BATCH_SIZE = int(os.getenv("LOCAL_BATCH_SIZE", "32"))
MAX_LOCAL_TRAIN_SAMPLES = int(os.getenv("MAX_LOCAL_TRAIN_SAMPLES", "0"))
POISON_FRACTION = float(os.getenv("POISON_FRACTION", "0.4"))
SIGN_FLIP_SCALE = float(os.getenv("SIGN_FLIP_SCALE", "3.0"))
GRADUAL_SIGN_FLIP_SCALE = float(os.getenv("GRADUAL_SIGN_FLIP_SCALE", "0.35"))
TRAINING_ATTACK = os.getenv("TRAINING_ATTACK", "auto")
ATTACK_START_ROUND = int(os.getenv("ATTACK_START_ROUND", "1"))
ATTACK_SCENARIO = os.getenv("ATTACK_SCENARIO", "all")
EXPERIMENT_VARIANT = os.getenv("EXPERIMENT_VARIANT", "full")
BENIGN_DRIFT_TRANSITION_ROUNDS = int(os.getenv("BENIGN_DRIFT_TRANSITION_ROUNDS", "4"))
BENIGN_DRIFT_TARGET_PARTITION = int(os.getenv("BENIGN_DRIFT_TARGET_PARTITION", "5"))


def post(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.post(
        f"{ACCESS_URL}{path}", json=payload, timeout=10.0
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"POST {path} failed with HTTP {response.status_code}: "
            f"{response.text[:500]}"
        ) from exc
    return response.json()


def get(path: str) -> dict[str, Any]:
    response = requests.get(f"{ACCESS_URL}{path}", timeout=10.0)
    response.raise_for_status()
    return response.json()


def wait_for_controller() -> None:
    for _ in range(120):
        try:
            if get("/health").get("status") == "ok":
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("access controller did not become ready")


def register(private_key) -> None:
    public_key = public_key_b64(private_key)
    payload = {
        "action": "REGISTER",
        "node_id": NODE_ID,
        "profile": PROFILE,
        "public_key": public_key,
    }
    post("/nodes/register", {
        "node_id": NODE_ID,
        "profile": PROFILE,
        "public_key": public_key,
        "registration_signature": sign_payload(private_key, payload),
    })


def outcome(task: dict[str, Any], rng: random.Random) -> tuple[str, float]:
    """Return delivery/result action and a synthetic task-quality observation."""
    if PROFILE == "selective":
        # The node accepts every task, but only reveals favorable outcomes.
        quality = rng.uniform(0.20, 0.95)
        return ("submit", quality) if quality >= 0.72 else ("withhold", quality)
    if PROFILE == "benign_unreliable" and rng.random() < 0.18:
        # No signed receipt: delivery/execution responsibility remains uncertain.
        return "unconfirmed", rng.uniform(0.78, 0.94)
    if PROFILE == "honest" and rng.random() < BENIGN_UNCONFIRMED_RATE:
        return "unconfirmed", min(0.99, max(0.0, rng.gauss(0.85, 0.08)))
    if PROFILE == "malicious":
        return "submit", rng.uniform(0.15, 0.48)
    if PROFILE == "single_type_farming":
        if task["task_type"] != "shadow_update":
            return "unconfirmed", 0.0
        return "submit", min(0.99, max(0.75, rng.gauss(0.93, 0.04)))
    if PROFILE == "three_type_repeat_farming":
        # Complete three inexpensive evidence types, but repeatedly avoid the
        # strongest behavioral shadow-update evidence.
        if task["task_type"] == "shadow_update":
            return "unconfirmed", 0.0
        return "submit", min(0.99, max(0.75, rng.gauss(0.93, 0.04)))
    if PROFILE == "false_quality_reporting":
        return "submit", 0.99
    if PROFILE in {"diverse_then_repeat_farming", "diverse_then_repeat_backdoor"}:
        # Behave correctly across multiple evidence types and repeated tasks
        # before switching to a malicious update after admission.
        return "submit", min(0.99, max(0.70, rng.gauss(0.90, 0.04)))
    if PROFILE == "attestation_camouflage":
        if task["task_type"] == "attestation":
            return "submit", min(0.99, max(0.80, rng.gauss(0.96, 0.025)))
        return "submit", min(0.55, max(0.01, rng.gauss(0.27, 0.10)))
    return "submit", min(0.99, max(0.50, rng.gauss(0.85, 0.08)))


def observed_shadow_update(task_index: int) -> dict[str, Any]:
    """Submit actual model parameters, never a caller-selected source ID."""
    x, y = client_partition(
        PARTITION_ID, NUM_DATA_PARTITIONS, NONIID_ALPHA, int(EXPERIMENT_SEED),
        dataset_name=DATASET_NAME, path=DATASET_PATH,
    )
    num_features, num_classes = dataset_dimensions(DATASET_NAME, DATASET_PATH)
    values = local_train(
        initial_parameters(num_features, num_classes), x, y,
        epochs=LOCAL_EPOCHS, learning_rate=LOCAL_LEARNING_RATE,
        batch_size=LOCAL_BATCH_SIZE,
        seed=int(EXPERIMENT_SEED) * 10000 + PARTITION_ID * 100 + task_index,
    )
    return {"weights": values[0].tolist(), "bias": values[1].tolist()}


def run_onboarding(private_key) -> dict[str, Any]:
    rng = random.Random(f"{EXPERIMENT_SEED}:{NODE_ID}:{PROFILE}")
    last_status: dict[str, Any] = {}
    first_shadow_update: dict[str, Any] | None = None
    shadow_variants = {"full_shadow_no_dedup", "full_shadow_provenance"}
    if EXPERIMENT_VARIANT in shadow_variants and DATA_MODE != "digits":
        raise RuntimeError("shadow observation requires digits data mode")
    for _ in range(MAX_ONBOARDING_TASKS):
        assignment = post(f"/nodes/{NODE_ID}/tasks")
        state = assignment["access_state"]
        if state != "PROBATION":
            return get(f"/nodes/{NODE_ID}/status")

        task = assignment["task"]
        if task is None:
            time.sleep(POLL_SECONDS)
            continue
        if digest(task) != assignment["assignment_hash"]:
            raise RuntimeError("server assignment hash mismatch")
        verify_signature(
            assignment["server_public_key"], task, assignment["server_signature"]
        )

        action, quality = outcome(task, rng)
        if NETWORK_JITTER_SECONDS > 0.0:
            time.sleep(rng.uniform(0.0, NETWORK_JITTER_SECONDS))
        if action != "unconfirmed":
            receipt = {
                "action": "ACK",
                "task_id": task["task_id"],
                "assignment_hash": assignment["assignment_hash"],
            }
            post("/tasks/receipt", {
                "node_id": NODE_ID,
                "task_id": task["task_id"],
                "assignment_hash": assignment["assignment_hash"],
                "signature": sign_payload(private_key, receipt),
            })

        if action == "submit":
            if NETWORK_JITTER_SECONDS > 0.0:
                time.sleep(rng.uniform(0.0, NETWORK_JITTER_SECONDS))
            result_bytes = json.dumps({
                "node_id": NODE_ID,
                "task_id": task["task_id"],
                "quality": round(quality, 6),
            }, sort_keys=True).encode("utf-8")
            result_hash = hashlib.sha256(result_bytes).hexdigest()
            work_product = challenge_answer(task)
            if PROFILE == "false_quality_reporting" or quality < 0.60:
                # Simulation: the claimed high quality does not match the
                # objectively checked challenge response.
                work_product = "invalid:" + work_product
            result = {
                "action": "RESULT",
                "task_id": task["task_id"],
                "quality": round(quality, 6),
                "result_hash": result_hash,
                "work_product": work_product,
            }
            if task["task_type"] == "shadow_update" and (
                EXPERIMENT_VARIANT in shadow_variants or PROFILE == "shadow_exact_replay"
            ):
                if DATA_MODE != "digits":
                    raise RuntimeError("actual shadow updates require digits data mode")
                update = observed_shadow_update(last_status.get("completed_tasks", 0))
                if PROFILE == "shadow_exact_replay":
                    if first_shadow_update is None:
                        first_shadow_update = update
                    update = first_shadow_update
                result["shadow_update"] = update
            post("/tasks/result", {
                "node_id": NODE_ID,
                "task_id": task["task_id"],
                "quality": round(quality, 6),
                "result_hash": result_hash,
                "work_product": work_product,
                "signature": sign_payload(private_key, result),
                **({"shadow_update": result["shadow_update"]}
                   if "shadow_update" in result else {}),
            })
        else:
            deadline = float(task["deadline_at"])
            time.sleep(max(0.0, deadline - time.time()) + 0.05)

        last_status = get(f"/nodes/{NODE_ID}/status")
        print(json.dumps(last_status, sort_keys=True), flush=True)
        if last_status["access_state"] != "PROBATION":
            return last_status
        time.sleep(POLL_SECONDS)
    return last_status or get(f"/nodes/{NODE_ID}/status")


def mark_finished_and_wait(private_key, status: dict[str, Any]) -> None:
    if EXPECTED_EXPERIMENT_NODES <= 0:
        return
    payload = {
        "action": "ONBOARDING_FINISHED",
        "node_id": NODE_ID,
        "final_access_state": status["access_state"],
    }
    post("/experiment/onboarding-finished", {
        "node_id": NODE_ID,
        "final_access_state": status["access_state"],
        "signature": sign_payload(private_key, payload),
    })
    deadline = time.time() + BARRIER_TIMEOUT_SECONDS
    while time.time() < deadline:
        barrier = get("/experiment/barrier")
        if barrier["released"]:
            print(
                f"barrier released: admitted={barrier['admitted_nodes']} "
                f"finished={barrier['finished_nodes']}", flush=True,
            )
            return
        time.sleep(0.1)
    raise RuntimeError("timed out waiting for onboarding barrier")


class SyntheticClient(fl.client.NumPyClient):
    def __init__(self, private_key) -> None:
        seed = int(hashlib.sha256(NODE_ID.encode()).hexdigest()[:8], 16)
        self.rng = np.random.default_rng(seed)
        self.parameters = [np.zeros((4,), dtype=np.float32)]
        malicious_profiles = {
            "malicious", "single_type_farming",
            "three_type_repeat_farming", "diverse_then_repeat_farming", "attestation_camouflage",
        }
        target_center = -2.5 if PROFILE in malicious_profiles else 0.8
        self.local_target = self.rng.normal(
            target_center, 0.08, size=(4,)
        ).astype(np.float32)
        self.examples = 64
        self.private_key = private_key

    def record_event(self, event_type: str, parameters) -> None:
        event_id = str(uuid.uuid4())
        digest_input = b"".join(
            np.asarray(value, dtype=np.float32).tobytes() for value in parameters
        )
        metrics_hash = hashlib.sha256(digest_input).hexdigest()
        payload = {
            "action": "FLOWER_EVENT",
            "event_id": event_id,
            "node_id": NODE_ID,
            "event_type": event_type,
            "metrics_hash": metrics_hash,
        }
        post("/flower/events", {
            **payload,
            "signature": sign_payload(self.private_key, payload),
        })

    def get_parameters(self, config):
        return self.parameters

    def fit(self, parameters, config):
        weights = np.asarray(parameters[0], dtype=np.float32)
        weights = weights + 0.35 * (self.local_target - weights)
        self.parameters = [weights]
        loss = float(np.square(weights - self.local_target).mean())
        self.record_event("FIT", self.parameters)
        return self.parameters, self.examples, {"local_loss": loss}

    def evaluate(self, parameters, config):
        weights = np.asarray(parameters[0], dtype=np.float32)
        loss = float(np.square(weights - self.local_target).mean())
        self.record_event("EVALUATE", [weights])
        return loss, self.examples, {"accuracy": float(max(0.0, 1.0 - loss))}


class DigitsClient(fl.client.NumPyClient):
    """Access-controlled client training a flattened public image dataset."""

    PROFILE_ATTACKS = {
        "single_type_farming": "label_flip",
        "three_type_repeat_farming": "sign_flip",
        "diverse_then_repeat_farming": "sign_flip",
        "gradual_drift_betrayal": "gradual_sign_flip",
        "diverse_then_repeat_backdoor": "backdoor",
        "attestation_camouflage": "backdoor",
        "false_quality_reporting": "backdoor",
    }

    def __init__(self, private_key) -> None:
        self.private_key = private_key
        self.num_features, self.num_classes = dataset_dimensions(
            DATASET_NAME, DATASET_PATH
        )
        self.x, self.y = client_partition(
            PARTITION_ID, NUM_DATA_PARTITIONS, NONIID_ALPHA, int(EXPERIMENT_SEED),
            dataset_name=DATASET_NAME, path=DATASET_PATH,
        )
        self.parameters = initial_parameters(self.num_features, self.num_classes)
        self.fit_round = 0
        self.benign_drift_enabled = (
            PROFILE == "benign_concept_drift"
            and ATTACK_SCENARIO in {"all", "benign_concept_drift"}
        )
        if self.benign_drift_enabled:
            if BENIGN_DRIFT_TRANSITION_ROUNDS < 1:
                raise ValueError("BENIGN_DRIFT_TRANSITION_ROUNDS must be positive")
            if BENIGN_DRIFT_TARGET_PARTITION == PARTITION_ID:
                raise ValueError("benign drift target must differ from the base partition")
            self.drift_x, self.drift_y = client_partition(
                BENIGN_DRIFT_TARGET_PARTITION,
                NUM_DATA_PARTITIONS,
                NONIID_ALPHA,
                int(EXPERIMENT_SEED),
                dataset_name=DATASET_NAME,
                path=DATASET_PATH,
            )
        configured_attack = (
            self.PROFILE_ATTACKS.get(PROFILE, "none")
            if TRAINING_ATTACK == "auto"
            else TRAINING_ATTACK
        )
        self.attack = (
            configured_attack
            if ATTACK_SCENARIO in {"all", PROFILE}
            else "none"
        )
        if EXPERIMENT_VARIANT == "oracle_benign":
            self.attack = "none"

    def record_event(self, event_type: str, parameters) -> None:
        event_id = str(uuid.uuid4())
        digest_input = b"".join(
            np.asarray(value, dtype=np.float32).tobytes() for value in parameters
        )
        metrics_hash = hashlib.sha256(digest_input).hexdigest()
        payload = {
            "action": "FLOWER_EVENT",
            "event_id": event_id,
            "node_id": NODE_ID,
            "event_type": event_type,
            "metrics_hash": metrics_hash,
        }
        post("/flower/events", {
            **payload,
            "signature": sign_payload(self.private_key, payload),
        })

    def get_parameters(self, config):
        return self.parameters

    def fit(self, parameters, config):
        self.fit_round += 1
        global_parameters = [
            np.asarray(value, dtype=np.float32).copy() for value in parameters
        ]
        server_round = int(config["server_round"])
        active_attack = effective_training_attack(
            self.attack, server_round, ATTACK_START_ROUND,
        )
        parent_hash = parameter_digest(global_parameters)
        if parent_hash != str(config["parent_hash"]):
            raise ValueError("server round model hash mismatch")
        training_seed = (
            int(EXPERIMENT_SEED) * 10000 + PARTITION_ID * 100 + self.fit_round
        )
        drift_fraction = 0.0
        if self.benign_drift_enabled and server_round >= ATTACK_START_ROUND:
            drift_fraction = min(
                1.0,
                (server_round - ATTACK_START_ROUND + 1)
                / BENIGN_DRIFT_TRANSITION_ROUNDS,
            )
            train_x, train_y = clean_distribution_drift(
                self.x, self.y, self.drift_x, self.drift_y,
                drift_fraction, training_seed,
            )
        else:
            train_x, train_y = apply_training_attack(
                active_attack,
                self.x,
                self.y,
                poison_fraction=POISON_FRACTION,
                seed=training_seed,
                num_classes=self.num_classes,
            )
        if MAX_LOCAL_TRAIN_SAMPLES < 0:
            raise ValueError("MAX_LOCAL_TRAIN_SAMPLES must be nonnegative")
        if MAX_LOCAL_TRAIN_SAMPLES and len(train_y) > MAX_LOCAL_TRAIN_SAMPLES:
            sample_rng = np.random.default_rng(training_seed + 73)
            sample_indices = sample_rng.choice(
                len(train_y), size=MAX_LOCAL_TRAIN_SAMPLES, replace=False,
            )
            train_x = train_x[sample_indices]
            train_y = train_y[sample_indices]
        local_parameters = local_train(
            global_parameters,
            train_x,
            train_y,
            epochs=LOCAL_EPOCHS,
            learning_rate=LOCAL_LEARNING_RATE,
            batch_size=LOCAL_BATCH_SIZE,
            seed=training_seed,
        )
        if active_attack in {"sign_flip", "gradual_sign_flip"}:
            local_parameters = sign_flip_update(
                global_parameters, local_parameters,
                (SIGN_FLIP_SCALE if active_attack == "sign_flip"
                 else GRADUAL_SIGN_FLIP_SCALE),
            )
        self.parameters = local_parameters
        clean_loss, clean_accuracy = metrics(self.parameters, self.x, self.y)
        self.record_event("FIT", self.parameters)
        commitment = {
            "action": "FIT_COMMITMENT", "node_id": NODE_ID,
            "server_round": server_round,
            "update_hash": parameter_digest(self.parameters),
            "parent_hash": parent_hash,
        }
        return self.parameters, len(self.y), {
            "clean_local_loss": clean_loss,
            "clean_local_accuracy": clean_accuracy,
            "training_attack": active_attack,
            "benign_drift_fraction": drift_fraction,
            "node_id": NODE_ID,
            "server_round": server_round,
            "update_hash": commitment["update_hash"],
            "parent_hash": parent_hash,
            "fit_signature": sign_payload(self.private_key, commitment),
        }

    def evaluate(self, parameters, config):
        loss, accuracy = metrics(parameters, self.x, self.y)
        self.record_event("EVALUATE", parameters)
        return loss, len(self.y), {"accuracy": accuracy}


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    private_key = load_or_create_private_key(STATE_DIR / "node_ed25519.pem")
    wait_for_controller()
    register(private_key)
    status = run_onboarding(private_key)
    print(f"node={NODE_ID} final_access={status['access_state']}", flush=True)
    mark_finished_and_wait(private_key, status)
    if status["access_state"] not in {"LIMITED", "ADMITTED"}:
        return
    time.sleep(FLOWER_JOIN_DELAY)
    flower_client = (
        DigitsClient(private_key) if DATA_MODE == "digits" else SyntheticClient(private_key)
    )
    print(
        f"node={NODE_ID} data_mode={DATA_MODE} "
        f"dataset={DATASET_NAME} "
        f"partition={PARTITION_ID}/{NUM_DATA_PARTITIONS}",
        flush=True,
    )
    fl.client.start_numpy_client(
        server_address=FLOWER_SERVER,
        client=flower_client,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import flwr as fl
import numpy as np
import requests

from flower_prototype.online_checks import (
    NORM_OUTCOME_EXTREME, NORM_OUTCOME_MODERATE,
    classify_update_norm, delta_opposition, directional_opposition,
    norm_escalation_decision, screen_update, shrink_update, update_norm,
)
from flower_prototype.real_data import (
    append_server_metrics,
    backdoor_success_rate,
    dataset_dimensions,
    initial_parameters,
    is_trained_round,
    load_validation_test_sets,
    metrics,
    parameter_digest,
    local_train,
)
from flower_prototype.robust_aggregation import (
    coordinate_median, coordinate_trimmed_mean_with_retention,
    fedavg_effective_delta_weights, fltrust, normalized_trust_weights,
)
from flower_prototype.reputation_aggregation import rffl_reputation_step
from flower_prototype.variant_policy import (
    default_aggregation_rule, is_access_core_variant, uses_cumulative_risk,
)


ROUND_NODE_FIELDS = (
    "round", "cid", "node_id", "selected", "returned", "signature_verified",
    "passed_screening", "accepted_for_aggregation", "aggregated", "reason",
    "access_state_before", "access_state_after", "training_evidence_outcome",
    "access_transition",
    "penalty_debt", "aggregation_weight", "instantaneous_risk",
    "cumulative_risk", "cohort_opposition", "self_reversal",
    "candidate_validation_loss", "global_validation_loss",
    "validation_loss_delta", "validation_loss_relative_delta",
    "validation_loss_flag",
    "update_norm", "median_update_norm", "norm_threshold",
    "extreme_norm_threshold", "norm_ratio", "norm_strike_count",
    "norm_screening_outcome", "norm_escalation_mode",
    "benign_drift_fraction",
    "aggregation_rule", "aggregation_trust_score", "aggregation_norm_scale",
    "aggregation_effective_weight", "aggregation_coordinate_retention_rate",
    "aggregation_reputation_score", "aggregation_reputation_contribution",
    "aggregation_reputation_removed", "update_payload_bytes",
    "server_round_processing_seconds",
)


def append_round_node_events(path: Path, records: list[dict]) -> None:
    """Persist node-level participation separately from the global model metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=ROUND_NODE_FIELDS)
        if output.tell() == 0:
            writer.writeheader()
        writer.writerows(records)


def wait_for_initial_clients(
    client_manager, expected: int, timeout: float, server_round: int = 1,
) -> None:
    """Wait for the expected connected nodes before any training round."""
    if expected < 1 or timeout <= 0:
        raise ValueError("initial client barrier must have a positive count and timeout")
    deadline = time.monotonic() + timeout
    while client_manager.num_available() < expected:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"round {server_round} client join timed out: "
                f"available={client_manager.num_available()}, required={expected}"
            )
        time.sleep(0.2)
    print(f"client join complete round={server_round} "
          f"available={client_manager.num_available()} required={expected}", flush=True)


class RevalidatingFedAvg(fl.server.strategy.FedAvg):
    """Verify signed identity and filter outliers before (not after) aggregation."""

    def __init__(
        self, x_test, y_test, access_url: str, token: str, enabled: bool,
        expected_initial_clients: int = 0, initial_join_timeout: float = 120.0,
        round_node_events_path: str | Path | None = None,
        cumulative_risk_enabled: bool = False,
        cumulative_risk_decay: float = 0.50,
        cumulative_risk_threshold: float = 0.90,
        self_reversal_gate: float = 0.20,
        aggregation_rule: str = "fedavg",
        trimmed_mean_count: int = 1,
        root_epochs: int = 1,
        root_learning_rate: float = 0.05,
        root_batch_size: int = 32,
        norm_escalation_mode: str = "enforce",
        reputation_fade: float = 0.95,
        reputation_threshold_scale: float = 1.0 / 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x_test, self.y_test = x_test, y_test
        self.access_url, self.token, self.enabled = access_url, token, enabled
        self.round_base = None
        self.rejected_cids: set[str] = set()
        self.cid_to_node: dict[str, str] = {}
        self.expected_initial_clients = expected_initial_clients
        self.initial_join_timeout = initial_join_timeout
        self.selected_by_round: dict[int, dict[str, dict]] = {}
        self.round_node_events_path = (Path(round_node_events_path)
                                       if round_node_events_path else None)
        self.cumulative_risk_enabled = cumulative_risk_enabled
        self.cumulative_risk_decay = float(cumulative_risk_decay)
        self.cumulative_risk_threshold = float(cumulative_risk_threshold)
        self.self_reversal_gate = float(self_reversal_gate)
        self.cumulative_risk: dict[str, float] = {}
        self.trusted_baseline_delta: dict[str, list[np.ndarray]] = {}
        self.norm_strikes: dict[str, int] = {}
        if norm_escalation_mode not in {"enforce", "moderate_shadow"}:
            raise ValueError(
                f"unknown norm escalation mode: {norm_escalation_mode}"
            )
        self.norm_escalation_mode = norm_escalation_mode
        allowed_rules = {
            "fedavg", "fltrust", "trimmed_mean", "coordinate_median",
            "rffl_reputation",
        }
        if aggregation_rule not in allowed_rules:
            raise ValueError(f"unknown aggregation rule: {aggregation_rule}")
        self.aggregation_rule = aggregation_rule
        self.trimmed_mean_count = int(trimmed_mean_count)
        self.root_epochs = int(root_epochs)
        self.root_learning_rate = float(root_learning_rate)
        self.root_batch_size = int(root_batch_size)
        self.reputation_fade = float(reputation_fade)
        self.reputation_threshold_scale = float(reputation_threshold_scale)
        self.reputations: dict[str, float] = {}

    def _post(self, path: str, payload: dict) -> dict:
        response = requests.post(
            f"{self.access_url}{path}", json=payload,
            headers={"X-Internal-Token": self.token}, timeout=10.0,
        )
        response.raise_for_status()
        return response.json()

    def configure_fit(self, server_round, parameters, client_manager):
        required = self.expected_initial_clients - len(self.rejected_cids)
        if self.expected_initial_clients and required <= 0:
            self.selected_by_round[server_round] = {}
            print(
                f"online_access round={server_round} no_eligible_clients=1",
                flush=True,
            )
            return []
        if self.expected_initial_clients:
            wait_for_initial_clients(
                client_manager, required, self.initial_join_timeout, server_round,
            )
        self.round_base = fl.common.parameters_to_ndarrays(parameters)
        selected = super().configure_fit(server_round, parameters, client_manager)
        live = [(proxy, instruction) for proxy, instruction in selected
                if proxy.cid not in self.rejected_cids]
        if self.expected_initial_clients and len(live) != required:
            raise RuntimeError(
                f"round {server_round} sampled {len(live)} clients; expected "
                f"{required} active clients"
            )
        self.selected_by_round[server_round] = {
            proxy.cid: {
                "round": server_round, "cid": proxy.cid,
                "node_id": self.cid_to_node.get(proxy.cid, ""), "selected": 1,
                "returned": 0, "signature_verified": 0,
                "passed_screening": 0, "accepted_for_aggregation": 0,
                "aggregated": 0, "reason": "NO_RESPONSE",
                "access_state_before": "", "access_state_after": "",
                "training_evidence_outcome": "", "penalty_debt": "",
                "access_transition": "",
                "aggregation_weight": "", "instantaneous_risk": "",
                "cumulative_risk": "",
                "cohort_opposition": "", "self_reversal": "",
                "candidate_validation_loss": "", "global_validation_loss": "",
                "validation_loss_delta": "", "validation_loss_relative_delta": "",
                "validation_loss_flag": "",
                "update_norm": "", "median_update_norm": "",
                "norm_threshold": "", "extreme_norm_threshold": "",
                "norm_ratio": "", "norm_strike_count": "",
                "norm_screening_outcome": "",
                "norm_escalation_mode": self.norm_escalation_mode,
                "benign_drift_fraction": "",
                "aggregation_rule": self.aggregation_rule,
                "aggregation_trust_score": "",
                "aggregation_norm_scale": "",
                "aggregation_effective_weight": "",
                "aggregation_coordinate_retention_rate": "",
                "aggregation_reputation_score": "",
                "aggregation_reputation_contribution": "",
                "aggregation_reputation_removed": 0,
                "update_payload_bytes": "",
                "server_round_processing_seconds": "",
            }
            for proxy, _ in live
        }
        print(f"online_access round={server_round} selected={len(live)} "
              f"removed={len(selected)-len(live)}", flush=True)
        return live

    def configure_evaluate(self, server_round, parameters, client_manager):
        selected = super().configure_evaluate(server_round, parameters, client_manager)
        return [(proxy, instruction) for proxy, instruction in selected
                if proxy.cid not in self.rejected_cids]

    def aggregate_fit(self, server_round, results, failures):
        processing_started = time.perf_counter()
        if self.round_base is None:
            raise RuntimeError("fit round has no server model reference")
        reference = self.round_base
        parent_hash = parameter_digest(reference)
        global_loss, _ = metrics(reference, self.x_test, self.y_test)
        checked = []
        seen_nodes = set()
        round_records = self.selected_by_round.pop(server_round, {})
        for proxy, result in results:
            record = round_records.get(proxy.cid)
            if record is None:
                # A result for an unsampled or already revoked connection must not enter FedAvg.
                self.rejected_cids.add(proxy.cid)
                continue
            record["returned"] = 1
            record["reason"] = ""
            if proxy.cid in self.rejected_cids:
                record["reason"] = "PREVIOUSLY_REVOKED"
                continue
            try:
                update = fl.common.parameters_to_ndarrays(result.parameters)
                record["update_payload_bytes"] = sum(
                    np.asarray(layer).nbytes for layer in update
                )
                supplied = result.metrics
                record["benign_drift_fraction"] = supplied.get(
                    "benign_drift_fraction", ""
                )
                node_id = str(supplied["node_id"])
                if not node_id or node_id in seen_nodes:
                    raise ValueError("duplicate or missing signed identity")
                if supplied["parent_hash"] != parent_hash:
                    raise ValueError("wrong parent model")
                if int(supplied["server_round"]) != server_round:
                    raise ValueError("wrong training round")
                update_hash = parameter_digest(update)
                if supplied["update_hash"] != update_hash:
                    raise ValueError("returned update hash mismatch")
                if proxy.cid in self.cid_to_node and self.cid_to_node[proxy.cid] != node_id:
                    raise ValueError("Flower connection changed signed identity")
                verification = self._post("/internal/fit/verify", {
                    "node_id": node_id, "server_round": server_round,
                    "update_hash": update_hash, "parent_hash": parent_hash,
                    "signature": str(supplied["fit_signature"]),
                })
                record["access_state_before"] = verification.get("access_state", "")
            except (KeyError, TypeError, ValueError, requests.RequestException) as exc:
                print(f"online_access round={server_round} "
                      f"discard_unverifiable_client={proxy.cid} detail={exc}", flush=True)
                self.rejected_cids.add(proxy.cid)
                record["reason"] = "UNVERIFIABLE_UPDATE"
                continue
            seen_nodes.add(node_id)
            self.cid_to_node[proxy.cid] = node_id
            record["node_id"] = node_id
            record["signature_verified"] = 1
            norm = update_norm(reference, update)
            candidate_loss, _ = metrics(update, self.x_test, self.y_test) if np.isfinite(norm) else (float("inf"), 0)
            checked.append((proxy, result, node_id, norm, candidate_loss, update))

        norms = [item[3] for item in checked if np.isfinite(item[3])]
        median_norm = float(np.median(norms)) if norms else float("inf")
        safe = []
        for checked_index, (proxy, result, node_id, norm, candidate_loss, update) in enumerate(checked):
            record = round_records[proxy.cid]
            norm_class, norm_threshold, extreme_norm_threshold, norm_ratio = (
                classify_update_norm(norm, median_norm)
            )
            prior_norm_strikes = self.norm_strikes.get(node_id, 0)
            norm_screening_outcome, norm_strike_count, norm_hard_failure, soft_norm_rejection = (
                norm_escalation_decision(
                    norm_class, prior_norm_strikes,
                    moderate_shadow=self.norm_escalation_mode == "moderate_shadow",
                )
            )
            reason = None

            # Model shape and finite values are mandatory even for ablation controls.
            screening_reason = (
                "INVALID_UPDATE"
                if not np.isfinite(norm) or not np.isfinite(candidate_loss) else
                screen_update(norm, median_norm, candidate_loss, global_loss)
                if self.enabled else None
            )
            # Evaluating one client's locally trained model in isolation on a
            # global validation set is not a valid exclusion rule under
            # Non-IID data.  Retain the finite loss regression as telemetry;
            # structural failures, excess norms and cumulative node-conditioned
            # direction reversal remain enforceable safety signals.
            validation_loss_flag = screening_reason == "VALIDATION_LOSS"
            if not np.isfinite(norm) or not np.isfinite(candidate_loss):
                reason = "INVALID_UPDATE"
                norm_screening_outcome = "NONFINITE_REVOKED"
            elif self.enabled and norm_hard_failure:
                reason = "EXCESS_UPDATE_NORM"
            if self.enabled and norm_class in {
                NORM_OUTCOME_EXTREME, NORM_OUTCOME_MODERATE,
            }:
                self.norm_strikes[node_id] = norm_strike_count
            if not self.enabled:
                norm_screening_outcome = "NORMAL"
                norm_strike_count = prior_norm_strikes
                soft_norm_rejection = False
                reason = "INVALID_UPDATE" if (
                    not np.isfinite(norm) or not np.isfinite(candidate_loss)
                ) else None
            # Leave the evaluated node out of its own robust reference; otherwise
            # an attacker can pull the cohort median toward its submitted update.
            peers = [item[5] for index, item in enumerate(checked)
                     if index != checked_index]
            cohort_median = [
                np.median(np.stack([peer[layer] for peer in peers]), axis=0)
                for layer in range(len(reference))
            ] if peers else reference
            cohort_opposition = directional_opposition(
                reference, update, cohort_median
            )
            current_delta = [
                np.asarray(new, dtype=np.float32) - np.asarray(old, dtype=np.float32)
                for old, new in zip(reference, update)
            ]
            baseline_delta = self.trusted_baseline_delta.get(node_id)
            self_reversal = (
                delta_opposition(current_delta, baseline_delta)
                if baseline_delta is not None else 0.0
            )
            # Under strong Non-IID, cohort opposition alone is not evidence of
            # betrayal.  Require a node-conditioned behavior reversal, then use
            # the cohort signal only as a bounded supporting factor.
            instant_risk = (
                self_reversal * (0.75 + 0.25 * cohort_opposition)
                if self_reversal >= self.self_reversal_gate else 0.0
            )
            cumulative_risk = self.cumulative_risk.get(node_id, 0.0)
            if (self.cumulative_risk_enabled and reason is None
                    and not soft_norm_rejection):
                cumulative_risk = (
                    self.cumulative_risk_decay * cumulative_risk + instant_risk
                )
                self.cumulative_risk[node_id] = cumulative_risk
                if cumulative_risk >= self.cumulative_risk_threshold:
                    reason = "CUMULATIVE_DIRECTIONAL_RISK"
            record["instantaneous_risk"] = f"{instant_risk:.8f}"
            record["cumulative_risk"] = f"{cumulative_risk:.8f}"
            record["cohort_opposition"] = f"{cohort_opposition:.8f}"
            record["self_reversal"] = f"{self_reversal:.8f}"
            loss_delta = candidate_loss - global_loss
            relative_delta = loss_delta / max(abs(global_loss), 1e-12)
            record["candidate_validation_loss"] = f"{candidate_loss:.8f}"
            record["global_validation_loss"] = f"{global_loss:.8f}"
            record["validation_loss_delta"] = f"{loss_delta:.8f}"
            record["validation_loss_relative_delta"] = f"{relative_delta:.8f}"
            record["validation_loss_flag"] = int(validation_loss_flag)
            record["update_norm"] = f"{norm:.8f}"
            record["median_update_norm"] = f"{median_norm:.8f}"
            record["norm_threshold"] = f"{norm_threshold:.8f}"
            record["extreme_norm_threshold"] = f"{extreme_norm_threshold:.8f}"
            record["norm_ratio"] = f"{norm_ratio:.8f}"
            record["norm_strike_count"] = norm_strike_count
            record["norm_screening_outcome"] = norm_screening_outcome
            if reason:
                # Fail closed: do not aggregate the suspect update if revocation fails.
                self.rejected_cids.add(proxy.cid)
                revocation = self._post("/internal/fit/revoke", {
                    "node_id": node_id, "server_round": server_round,
                    "reason": reason,
                })
                record["access_state_after"] = revocation.get("access_state", "QUARANTINE")
                record["training_evidence_outcome"] = "HARD_FAILURE"
                record["penalty_debt"] = revocation.get("penalty_debt", "")
                record["reason"] = reason
                print(f"online_access round={server_round} revoked={node_id} "
                      f"reason={reason} norm={norm:.5f} "
                      f"validation_loss={candidate_loss:.5f}", flush=True)
                continue
            if soft_norm_rejection:
                record["access_state_after"] = record["access_state_before"]
                record["training_evidence_outcome"] = "SOFT_REJECTED"
                record["reason"] = "EXCESS_UPDATE_NORM_REJECTED"
                print(
                    f"online_access round={server_round} norm_rejected={node_id} "
                    f"strike={norm_strike_count} norm={norm:.5f} "
                    f"threshold={norm_threshold:.5f}", flush=True,
                )
                continue
            if validation_loss_flag:
                print(
                    f"online_access round={server_round} validation_observed={node_id} "
                    f"reason=VALIDATION_LOSS_TELEMETRY norm={norm:.5f} "
                    f"validation_loss={candidate_loss:.5f} "
                    f"global_loss={global_loss:.5f}", flush=True,
                )
            record["passed_screening"] = 1
            authorization = self._post("/internal/fit/accept", {
                "node_id": node_id, "server_round": server_round,
                "update_hash": parameter_digest(fl.common.parameters_to_ndarrays(result.parameters)),
            })
            # A verified clean norm observation breaks a consecutive-excess run.
            self.norm_strikes[node_id] = 0
            record["norm_strike_count"] = 0
            if not authorization.get("aggregation_authorized", True):
                record["access_state_after"] = authorization.get("access_state", "LIMITED")
                record["training_evidence_outcome"] = "VERIFIED_CLEAN"
                record["reason"] = "LIMITED_SHADOW_ONLY"
                print(
                    f"online_access round={server_round} limited_shadow={node_id} "
                    f"state={authorization.get('access_state', 'LIMITED')}", flush=True,
                )
                continue
            record["accepted_for_aggregation"] = 1
            record["access_state_after"] = authorization.get("access_state", "ADMITTED")
            record["training_evidence_outcome"] = authorization.get(
                "training_evidence_outcome", "VERIFIED_CLEAN"
            )
            record["access_transition"] = authorization.get(
                "access_transition", ""
            )
            if baseline_delta is None:
                self.trusted_baseline_delta[node_id] = [
                    value.copy() for value in current_delta
                ]
            elif (authorization.get("access_state") == "ADMITTED"
                  and self_reversal < self.self_reversal_gate):
                self.trusted_baseline_delta[node_id] = [
                    (0.90 * old + 0.10 * new).astype(np.float32)
                    for old, new in zip(baseline_delta, current_delta)
                ]
            aggregation_weight = float(authorization.get("aggregation_weight", 1.0))
            record["aggregation_weight"] = aggregation_weight
            original_hash = parameter_digest(update)
            if aggregation_weight < 1.0:
                result.parameters = fl.common.ndarrays_to_parameters(
                    shrink_update(reference, update, aggregation_weight)
                )
                record["reason"] = (
                    "LEASE_EXPIRED_LIMITED"
                    if record["access_transition"] == "LEASE_EXPIRED"
                    else "LIMITED_WEIGHTED"
                )
            safe.append((proxy, result, node_id, original_hash))
        print(f"online_access round={server_round} accepted={len(safe)} "
              f"rejected={len(results)-len(safe)}", flush=True)
        rffl_removed_nodes: set[str] = set()
        if self.aggregation_rule == "fedavg":
            aggregated = super().aggregate_fit(
                server_round, [(proxy, result) for proxy, result, _, _ in safe], failures
            )
            effective_weights = fedavg_effective_delta_weights(
                [result.num_examples for _, result, _, _ in safe],
                [
                    float(round_records[proxy.cid].get("aggregation_weight") or 1.0)
                    for proxy, _, _, _ in safe
                ],
            )
            for (proxy, _, _, _), effective in zip(safe, effective_weights):
                round_records[proxy.cid]["aggregation_effective_weight"] = (
                    f"{effective:.8f}"
                )
        elif not safe or (failures and not self.accept_failures):
            aggregated = (None, {})
        else:
            client_models = [
                fl.common.parameters_to_ndarrays(result.parameters)
                for _, result, _, _ in safe
            ]
            if self.aggregation_rule == "trimmed_mean":
                model, retention_rates = coordinate_trimmed_mean_with_retention(
                    client_models, self.trimmed_mean_count,
                )
                contributors = len(client_models) - 2 * self.trimmed_mean_count
                for (proxy, _, _, _), retention in zip(safe, retention_rates):
                    round_records[proxy.cid][
                        "aggregation_coordinate_retention_rate"
                    ] = f"{retention:.8f}"
                    round_records[proxy.cid]["aggregation_effective_weight"] = (
                        f"{retention / contributors:.8f}"
                    )
            elif self.aggregation_rule == "coordinate_median":
                model = coordinate_median(client_models)
            elif self.aggregation_rule == "rffl_reputation":
                node_ids = [node_id for _, _, node_id, _ in safe]
                model, reputations, contributions, removed, used_weights = rffl_reputation_step(
                    reference, client_models, node_ids, self.reputations,
                    fade=self.reputation_fade,
                    threshold_scale=self.reputation_threshold_scale,
                )
                for (proxy, _, node_id, _), reputation, contribution, is_removed, used_weight in zip(
                    safe, reputations, contributions, removed, used_weights,
                ):
                    self.reputations[node_id] = reputation
                    record = round_records[proxy.cid]
                    record["aggregation_reputation_score"] = f"{reputation:.8f}"
                    record["aggregation_reputation_contribution"] = f"{contribution:.8f}"
                    record["aggregation_reputation_removed"] = int(is_removed)
                    record["aggregation_effective_weight"] = f"{used_weight:.8f}"
                    if is_removed:
                        rffl_removed_nodes.add(node_id)
            else:
                access_weights = [
                    float(round_records[proxy.cid].get("aggregation_weight") or 1.0)
                    for proxy, _, _, _ in safe
                ]
                root_model = local_train(
                    reference, self.x_test, self.y_test,
                    epochs=self.root_epochs,
                    learning_rate=self.root_learning_rate,
                    batch_size=self.root_batch_size,
                    seed=910000 + server_round,
                )
                model, trust_scores, norm_scales = fltrust(
                    reference, client_models, root_model, access_weights,
                )
                for (proxy, _, _, _), trust, scale in zip(
                    safe, trust_scores, norm_scales,
                ):
                    round_records[proxy.cid]["aggregation_trust_score"] = f"{trust:.8f}"
                    round_records[proxy.cid]["aggregation_norm_scale"] = f"{scale:.8f}"
                effective_weights = normalized_trust_weights(
                    trust_scores, access_weights,
                )
                for (proxy, _, _, _), effective in zip(safe, effective_weights):
                    round_records[proxy.cid]["aggregation_effective_weight"] = (
                        f"{effective:.8f}"
                    )
            aggregated = (fl.common.ndarrays_to_parameters(model), {})
        if aggregated[0] is not None:
            for proxy, result, node_id, update_hash in safe:
                self._post("/internal/fit/aggregated", {
                    "node_id": node_id, "server_round": server_round,
                    "update_hash": update_hash,
                })
                round_records[proxy.cid]["aggregated"] = 1
                if node_id in rffl_removed_nodes:
                    self.rejected_cids.add(proxy.cid)
                    revocation = self._post("/internal/fit/remove", {
                        "node_id": node_id, "server_round": server_round,
                        "reason": "RFFL_LOW_REPUTATION",
                    })
                    round_records[proxy.cid]["access_state_after"] = revocation.get(
                        "access_state", "QUARANTINE"
                    )
                    round_records[proxy.cid]["reason"] = (
                        "RFFL_REMOVED_AFTER_AGGREGATION"
                    )
        else:
            for proxy, _, _, _ in safe:
                round_records[proxy.cid]["reason"] = "AGGREGATION_UNAVAILABLE"
        processing_seconds = time.perf_counter() - processing_started
        for record in round_records.values():
            record["server_round_processing_seconds"] = f"{processing_seconds:.8f}"
        if self.round_node_events_path is not None:
            append_round_node_events(
                self.round_node_events_path, list(round_records.values()),
            )
        return aggregated


def wait_for_onboarding_barrier(default_minimum: int) -> int:
    expected = int(os.getenv("EXPECTED_EXPERIMENT_NODES", "0"))
    if expected <= 0:
        return default_minimum
    access_url = os.getenv("ACCESS_URL", "http://access-controller:8000")
    timeout = float(os.getenv("BARRIER_TIMEOUT_SECONDS", "600"))
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(
                f"{access_url}/experiment/barrier", timeout=5.0
            )
            response.raise_for_status()
            barrier = response.json()
            if barrier["released"]:
                admitted = int(barrier["admitted_nodes"])
                if admitted < 1:
                    raise RuntimeError("onboarding completed with no admitted clients")
                print(
                    f"onboarding barrier released: admitted={admitted}, "
                    f"finished={barrier['finished_nodes']}", flush=True,
                )
                return admitted
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise RuntimeError("timed out waiting for onboarding barrier")


def main() -> None:
    rounds = int(os.getenv("FLOWER_ROUNDS", "3"))
    data_mode = os.getenv("DATA_MODE", "synthetic")
    dataset_name = os.getenv("DATASET_NAME", "digits")
    dataset_path = os.getenv("DATASET_PATH", "") or None
    configured_minimum = int(os.getenv("FLOWER_MIN_CLIENTS", "2"))
    minimum = wait_for_onboarding_barrier(configured_minimum)
    strategy_options = {}
    if data_mode == "digits":
        x_validation, y_validation, x_test, y_test = load_validation_test_sets(
            path=dataset_path, dataset_name=dataset_name,
        )
        num_features, num_classes = dataset_dimensions(dataset_name, dataset_path)
        metrics_path = Path(os.getenv("GLOBAL_METRICS_PATH", "/results/global_metrics.csv"))
        metrics_path.unlink(missing_ok=True)
        round_node_events_path = Path(os.getenv(
            "ROUND_NODE_EVENTS_PATH", "/results/round_node_events.csv",
        ))
        round_node_events_path.unlink(missing_ok=True)

        def evaluate(server_round, parameters, config):
            loss, accuracy = metrics(parameters, x_test, y_test)
            attack_success_rate = backdoor_success_rate(parameters, x_test, y_test)
            if is_trained_round(server_round):
                append_server_metrics(
                    metrics_path, server_round, loss, accuracy, attack_success_rate
                )
            print(
                f"central_evaluate round={server_round} "
                f"formal_metric={is_trained_round(server_round)} loss={loss:.6f} "
                f"accuracy={accuracy:.6f} backdoor_asr={attack_success_rate:.6f}",
                flush=True,
            )
            return loss, {
                "central_accuracy": accuracy,
                "backdoor_asr": attack_success_rate,
            }

        strategy_options = {
            "initial_parameters": fl.common.ndarrays_to_parameters(
                initial_parameters(num_features, num_classes)
            ),
            "evaluate_fn": evaluate,
        }

    if data_mode == "digits":
        token = os.getenv("FL_CONTROL_TOKEN", "")
        if not token:
            raise RuntimeError("FL_CONTROL_TOKEN is required for signed fit verification")
        strategy_options["on_fit_config_fn"] = lambda server_round: {
            "server_round": server_round,
            "parent_hash": parameter_digest(strategy.round_base) if strategy.round_base is not None else "",
        }
    strategy_type = RevalidatingFedAvg if data_mode == "digits" else fl.server.strategy.FedAvg
    online_options = ({
        "x_test": x_validation, "y_test": y_validation,
        "access_url": os.getenv("ACCESS_URL", "http://access-controller:8000"),
        "token": token,
        "expected_initial_clients": minimum,
        "initial_join_timeout": float(os.getenv("CLIENT_JOIN_TIMEOUT_SECONDS", "300")),
        "round_node_events_path": round_node_events_path,
        "enabled": (
            os.getenv("MECHANISM_VARIANT", "full") not in {
                "no_online_revalidation", "naive", "oracle_benign", "oracle_filter",
                "fltrust", "trimmed_mean", "coordinate_median", "rffl_reputation",
            }
            and not is_access_core_variant(
                os.getenv("MECHANISM_VARIANT", "full")
            )
        ),
        "cumulative_risk_enabled": uses_cumulative_risk(
            os.getenv("MECHANISM_VARIANT", "full")
        ),
        "cumulative_risk_decay": float(os.getenv("CUMULATIVE_RISK_DECAY", "0.50")),
        "cumulative_risk_threshold": float(os.getenv("CUMULATIVE_RISK_THRESHOLD", "0.90")),
        "self_reversal_gate": float(os.getenv("SELF_REVERSAL_GATE", "0.20")),
        "aggregation_rule": os.getenv(
            "AGGREGATION_RULE",
            default_aggregation_rule(os.getenv("MECHANISM_VARIANT", "full")),
        ),
        "trimmed_mean_count": int(os.getenv("TRIMMED_MEAN_COUNT", "1")),
        "root_epochs": int(os.getenv("FLTRUST_ROOT_EPOCHS", "1")),
        "root_learning_rate": float(os.getenv("FLTRUST_ROOT_LEARNING_RATE", "0.05")),
        "root_batch_size": int(os.getenv("FLTRUST_ROOT_BATCH_SIZE", "32")),
        "reputation_fade": float(os.getenv("RFFL_REPUTATION_FADE", "0.95")),
        "reputation_threshold_scale": float(os.getenv(
            "RFFL_REPUTATION_THRESHOLD_SCALE", "0.3333333333333333"
        )),
        "norm_escalation_mode": os.getenv("NORM_ESCALATION_MODE", "enforce"),
    } if data_mode == "digits" else {})
    strategy = strategy_type(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        # The strategy enforces the full active count with a bounded timeout;
        # Flower's internal minimum must not block indefinitely after revocation.
        min_fit_clients=1 if data_mode == "digits" else min(3, minimum),
        min_evaluate_clients=1 if data_mode == "digits" else min(3, minimum),
        min_available_clients=1 if data_mode == "digits" else min(3, minimum),
        **online_options,
        **strategy_options,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

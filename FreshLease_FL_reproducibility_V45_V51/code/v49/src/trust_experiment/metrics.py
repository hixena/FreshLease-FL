from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, risk: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(risk, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = ids == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - risk[mask].mean())
    return float(ece)


def _transition_delays(group: pd.DataFrame, threshold: float) -> tuple[list[int], list[int]]:
    g = group.sort_values("round")
    states = g["malicious"].to_numpy(dtype=int)
    scores = (
        g["decision_score"].to_numpy(dtype=float)
        if "decision_score" in g.columns
        else g["trust_score"].to_numpy(dtype=float)
    )
    rounds = g["round"].to_numpy(dtype=int)
    detect: list[int] = []
    recover: list[int] = []
    for idx in range(1, len(g)):
        if states[idx - 1] == 0 and states[idx] == 1:
            end = idx + 1
            while end < len(g) and states[end] == 1:
                end += 1
            local = np.flatnonzero(scores[idx:end] < threshold)
            detect.append(int(rounds[idx + local[0]] - rounds[idx]) if len(local) else int(end - idx))
        elif states[idx - 1] == 1 and states[idx] == 0:
            end = idx + 1
            while end < len(g) and states[end] == 0:
                end += 1
            local = np.flatnonzero(scores[idx:end] >= threshold)
            recover.append(int(rounds[idx + local[0]] - rounds[idx]) if len(local) else int(end - idx))
    return detect, recover


def calculate_metrics(df: pd.DataFrame, threshold: float, calibration_bins: int) -> dict:
    y = df["malicious"].to_numpy(dtype=int)
    score = df["trust_score"].to_numpy(dtype=float)
    risk = 1.0 - score
    benign = y == 0
    malicious = y == 1
    decision_score = (
        df["decision_score"].to_numpy(dtype=float)
        if "decision_score" in df.columns else score
    )
    accepted = decision_score >= threshold

    result = {
        "far": float(accepted[malicious].mean()) if malicious.any() else np.nan,
        "frr": float((~accepted[benign]).mean()) if benign.any() else np.nan,
        "brier": float(brier_score_loss(y, risk)),
        "ece": expected_calibration_error(y, risk, calibration_bins),
        "auroc": float(roc_auc_score(y, risk)) if len(np.unique(y)) == 2 else np.nan,
        "auprc": float(average_precision_score(y, risk)) if len(np.unique(y)) == 2 else np.nan,
    }

    if "access_state" in df.columns:
        state = df["access_state"].astype(str).to_numpy()
        applicable = state != "not_applicable"
        if applicable.any():
            full = state == "full_access"
            denied = state == "deny"
            restricted = np.isin(state, ["limited_access", "probation", "quarantine"])
            result["full_access_far"] = (
                float(full[malicious].mean()) if malicious.any() else np.nan
            )
            result["deny_frr"] = (
                float(denied[benign].mean()) if benign.any() else np.nan
            )
            result["restricted_benign_rate"] = (
                float(restricted[benign].mean()) if benign.any() else np.nan
            )
            result["restricted_malicious_rate"] = (
                float(restricted[malicious].mean()) if malicious.any() else np.nan
            )
            probation = state == "probation"
            quarantine = state == "quarantine"
            promoted = np.isin(state, ["limited_access", "full_access"])
            result["probation_benign_rate"] = (
                float(probation[benign].mean()) if benign.any() else np.nan
            )
            result["probation_malicious_rate"] = (
                float(probation[malicious].mean()) if malicious.any() else np.nan
            )
            result["quarantine_benign_rate"] = (
                float(quarantine[benign].mean()) if benign.any() else np.nan
            )
            result["quarantine_malicious_rate"] = (
                float(quarantine[malicious].mean()) if malicious.any() else np.nan
            )
            result["benign_promotion_event_rate"] = (
                float(promoted[benign].mean()) if benign.any() else np.nan
            )
            result["malicious_promotion_event_rate"] = (
                float(promoted[malicious].mean()) if malicious.any() else np.nan
            )

            benign_node_promoted: list[float] = []
            malicious_node_promoted: list[float] = []
            benign_time_to_promotion: list[float] = []
            for _, node in df.groupby(["repeat", "node_id"], sort=False):
                node = node.sort_values("round")
                node_state = node["access_state"].astype(str).to_numpy()
                node_malicious = node["malicious"].to_numpy(dtype=int) == 1
                node_promoted = np.isin(node_state, ["limited_access", "full_access"])
                if (~node_malicious).any():
                    benign_promoted = bool((node_promoted & ~node_malicious).any())
                    benign_node_promoted.append(float(benign_promoted))
                    if benign_promoted:
                        first = int(np.flatnonzero(node_promoted & ~node_malicious)[0])
                        benign_time_to_promotion.append(float(
                            node.iloc[first]["round"] - node.iloc[0]["round"]
                        ))
                if node_malicious.any():
                    malicious_node_promoted.append(float(
                        (node_promoted & node_malicious).any()
                    ))
            result["benign_node_promotion_rate"] = (
                float(np.mean(benign_node_promoted)) if benign_node_promoted else np.nan
            )
            result["malicious_node_promotion_rate"] = (
                float(np.mean(malicious_node_promoted)) if malicious_node_promoted else np.nan
            )
            result["benign_time_to_promotion"] = (
                float(np.mean(benign_time_to_promotion))
                if benign_time_to_promotion else np.nan
            )

    if "probation_task_cost" in df.columns:
        costs = df["probation_task_cost"].to_numpy(dtype=float)
        result["probation_cost_total"] = float(np.nansum(costs))
        executed = df.get(
            "probation_task_executed", pd.Series(np.zeros(len(df)), index=df.index)
        ).to_numpy(dtype=int) == 1
        result["probation_cost_per_executed_task"] = (
            float(np.nanmean(costs[executed])) if executed.any() else 0.0
        )

    detect_all: list[int] = []
    recover_all: list[int] = []
    for _, group in df.groupby(["repeat", "node_id"], sort=False):
        detect, recover = _transition_delays(group, threshold)
        detect_all.extend(detect)
        recover_all.extend(recover)
    result["detection_delay"] = float(np.mean(detect_all)) if detect_all else np.nan
    result["recovery_delay"] = float(np.mean(recover_all)) if recover_all else np.nan
    return result

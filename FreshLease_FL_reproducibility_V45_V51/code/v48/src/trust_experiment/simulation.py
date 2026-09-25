from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    rounds: int
    profiles: list[str]
    observation_noise_std: float
    current_evidence_noise_std: float
    current_evidence_signal: float
    low_quality_source_rate: float
    reliable_source_quality: float
    low_source_quality: float
    max_feedback_delay: int
    freshness_tau: float
    beta_binary_threshold: float
    dirichlet_bins: list[float]
    current_evidence_missing_rate: float = 0.05
    current_evidence_age_max: int = 5
    current_evidence_freshness_tau: float = 5.0
    current_consistency_scale: float = 4.0
    current_predictive_quality: float = 0.75
    probation_completion_benign: float = 0.95
    probation_completion_malicious: float = 0.85


def _state_for(profile: str, t: int, rounds: int, rng: np.random.Generator) -> tuple[bool, bool, float]:
    """返回(active, malicious, latent_quality)。"""
    if profile == "stable_benign":
        return True, False, float(rng.beta(18, 2))
    if profile == "benign_jitter":
        # 偶发的非恶意性能波动，不改变真实恶意标签。
        quality = rng.beta(7, 3) if rng.random() < 0.12 else rng.beta(16, 3)
        return True, False, float(quality)
    if profile == "persistent_malicious":
        return True, True, float(rng.beta(2, 10))
    if profile == "betrayal":
        malicious = t >= int(rounds * 0.40)
        quality = rng.beta(2, 10) if malicious else rng.beta(18, 2)
        return True, malicious, float(quality)
    if profile == "on_off":
        malicious = (t % 10) >= 8
        quality = rng.beta(2, 9) if malicious else rng.beta(16, 3)
        return True, malicious, float(quality)
    if profile == "recovery":
        malicious = t < int(rounds * 0.30)
        quality = rng.beta(2, 9) if malicious else rng.beta(16, 3)
        return True, malicious, float(quality)
    if profile == "cold_start_malicious":
        start = int(rounds * 0.20)
        if t < start:
            return False, False, float("nan")
        return True, True, float(rng.beta(2, 10))
    raise ValueError(f"未知节点行为类型: {profile}")


def simulate_events(cfg: SimulationConfig, seed: int, repeat: int) -> pd.DataFrame:
    """生成一组可同时用于Beta和Dirichlet的共同观测事件。"""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    bins = np.asarray(cfg.dirichlet_bins, dtype=float)

    for node_idx, profile in enumerate(cfg.profiles):
        node_id = f"node_{node_idx:02d}"
        for t in range(cfg.rounds):
            active, malicious, latent_quality = _state_for(profile, t, cfg.rounds, rng)
            if not active:
                continue

            # 当前接入证据是接入前代理量，不能直接看到接入后的真实行为。
            # signal越小，隐蔽恶意节点越容易在本轮表现得“看起来正常”。
            apparent_quality = float(np.clip(
                cfg.current_evidence_signal * latent_quality
                + (1.0 - cfg.current_evidence_signal) * 0.75,
                0.0,
                1.0,
            ))
            modality_weights = np.asarray([0.20, 0.35, 0.20, 0.25], dtype=float)
            modality_values = np.clip(
                apparent_quality
                + rng.normal(0.0, cfg.current_evidence_noise_std, len(modality_weights)),
                0.0,
                1.0,
            )
            present = rng.random(len(modality_weights)) >= cfg.current_evidence_missing_rate
            completeness = float(present.mean())
            if present.any():
                observed_weights = modality_weights[present]
                observed_weights = observed_weights / observed_weights.sum()
                current_evidence = float(np.exp(np.sum(
                    observed_weights * np.log(np.maximum(1e-6, modality_values[present]))
                )))
                dispersion = float(np.std(modality_values[present]))
            else:
                current_evidence = 0.50
                dispersion = 1.0

            evidence_age = int(rng.integers(0, cfg.current_evidence_age_max + 1))
            evidence_freshness = float(np.exp(
                -evidence_age / cfg.current_evidence_freshness_tau
            ))
            evidence_consistency = float(np.exp(
                -cfg.current_consistency_scale * dispersion
            ))
            # In deployment this term must come from validation calibration of
            # the current-evidence detector, never from the unknown true label.
            predictive_quality = float(np.clip(cfg.current_predictive_quality, 0.0, 1.0))
            confidence_factors = np.asarray([
                completeness,
                evidence_freshness,
                evidence_consistency,
                predictive_quality,
            ], dtype=float)
            # Normalized geometric mean remains non-compensatory without
            # shrinking merely because four confidence dimensions are used.
            current_confidence = float(
                np.prod(np.clip(confidence_factors, 0.0, 1.0)) ** (1.0 / 4.0)
            )
            raw_observation = float(np.clip(
                latent_quality + rng.normal(0.0, cfg.observation_noise_std), 0.0, 1.0
            ))

            low_quality_source = rng.random() < cfg.low_quality_source_rate
            source_quality = (
                cfg.low_source_quality if low_quality_source else cfg.reliable_source_quality
            )
            corrupted = rng.random() > source_quality
            observed_quality = 1.0 - raw_observation if corrupted else raw_observation

            delay = int(rng.integers(0, cfg.max_feedback_delay + 1))
            freshness = float(np.exp(-delay / cfg.freshness_tau))
            detector_confidence = float(max(0.10, 2.0 * abs(raw_observation - 0.50)))
            reliability = float(np.clip(
                source_quality * detector_confidence * freshness, 0.0, 1.0
            ))

            beta_feedback = int(observed_quality >= cfg.beta_binary_threshold)
            dirichlet_level = int(np.digitize(observed_quality, bins, right=False))

            rows.append({
                "repeat": repeat,
                "round": t,
                "node_id": node_id,
                "profile": profile,
                "malicious": int(malicious),
                "latent_quality": latent_quality,
                "current_evidence": current_evidence,
                "current_confidence": current_confidence,
                "current_completeness": completeness,
                "current_freshness": evidence_freshness,
                "current_consistency": evidence_consistency,
                "current_predictive_quality": predictive_quality,
                "current_identity": float(modality_values[0]) if present[0] else np.nan,
                "current_device": float(modality_values[1]) if present[1] else np.nan,
                "current_context": float(modality_values[2]) if present[2] else np.nan,
                "current_protocol": float(modality_values[3]) if present[3] else np.nan,
                "post_access_observation": raw_observation,
                "observed_quality": observed_quality,
                "beta_feedback": beta_feedback,
                "dirichlet_level": dirichlet_level,
                "source_quality": source_quality,
                "detector_confidence": detector_confidence,
                "feedback_delay": delay,
                "freshness": freshness,
                "reliability": reliability,
                "feedback_corrupted": int(corrupted),
            })

    events = pd.DataFrame(rows).sort_values(["node_id", "round"]).reset_index(drop=True)

    # Keep probation randomness independent from the original event stream so
    # enabling this mechanism does not silently alter baseline observations.
    probation_rng = np.random.default_rng(seed + 104729)
    task_types = np.asarray([
        "attestation", "protocol_check", "canary_training", "shadow_update"
    ])
    task_cost = {
        "attestation": 0.20,
        "protocol_check": 0.40,
        "canary_training": 0.80,
        "shadow_update": 1.00,
    }
    scheduled = []
    completed = []
    costs = []
    for row in events.itertuples(index=False):
        node_number = int(str(row.node_id).split("_")[-1])
        task = str(task_types[(node_number + int(row.round)) % len(task_types)])
        completion_probability = (
            cfg.probation_completion_malicious
            if int(row.malicious) else cfg.probation_completion_benign
        )
        scheduled.append(task)
        completed.append(int(probation_rng.random() < completion_probability))
        costs.append(task_cost[task])
    events["probation_task_type"] = scheduled
    events["probation_task_completed"] = completed
    events["probation_task_acknowledged"] = 1
    events["probation_task_cost"] = costs
    events["probation_affects_aggregation"] = 0
    # Dedicated probation observations allow validation scenarios to separate
    # a probe/attestation result from the later formal-training behavior.
    events["probation_observed_quality"] = events["observed_quality"]
    events["probation_beta_feedback"] = events["beta_feedback"]
    events["probation_dirichlet_level"] = events["dirichlet_level"]
    events["probation_reliability"] = events["reliability"]
    return events

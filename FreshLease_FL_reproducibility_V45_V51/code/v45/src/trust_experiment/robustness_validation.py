from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from .models import ModelConfig
from .novelty_validation import _make_scenario, _metrics, _summary
from .runner import run_models
from .simulation import SimulationConfig


FULL_MODEL = "proposed_dirichlet_v1_controlled_probation"
MODELS = [
    FULL_MODEL,
    "probation_naive",
    "score_v1_access",
    "score_dirichlet_decay_mean_access",
    "score_ema_access",
    "score_current_only_access",
]

DISPLAY_NAMES = {
    FULL_MODEL: "full controlled probation",
    "probation_naive": "naive probation",
    "score_v1_access": "V1 score-only",
    "score_dirichlet_decay_mean_access": "decayed Dirichlet",
    "score_ema_access": "EMA",
    "score_current_only_access": "current evidence only",
}


def _build_events(
    sim_cfg: SimulationConfig,
    scenarios: list[str],
    seed: int,
    repeats: int,
    nodes: int,
) -> pd.DataFrame:
    frames = []
    for repeat_id in range(repeats):
        for scenario_index, scenario in enumerate(scenarios):
            frames.append(_make_scenario(
                sim_cfg,
                scenario,
                seed + repeat_id * 1009 + scenario_index * 100003,
                repeat_id,
                nodes,
            ))
    return pd.concat(frames, ignore_index=True)


def _calibrate_thresholds(
    sim_cfg: SimulationConfig,
    base_cfg: ModelConfig,
    seed: int,
    repeats: int,
    nodes: int,
    thresholds: list[float],
    target_benign_block_rate: float,
    warmup_round: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    events = _build_events(
        sim_cfg, ["honest_newcomer"], seed, repeats, nodes
    )
    rows = []
    chosen: dict[str, float] = {}
    for model in MODELS:
        model_rows = []
        for threshold in thresholds:
            cfg = replace(
                base_cfg,
                limited_access_threshold=float(threshold),
                full_access_threshold=float(min(0.99, threshold + 0.10)),
            )
            scores = run_models(events, cfg, [model])
            evaluation = scores[scores["round"] >= warmup_round]
            blocked = ~evaluation["aggregation_eligible"].fillna(False).to_numpy(bool)
            row = {
                "model": model,
                "threshold": float(threshold),
                "benign_block_rate": float(blocked.mean()),
                "benign_access_rate": float(1.0 - blocked.mean()),
            }
            rows.append(row)
            model_rows.append(row)
        feasible = [
            row for row in model_rows
            if row["benign_block_rate"] <= target_benign_block_rate
        ]
        chosen[model] = float(
            max(feasible, key=lambda row: row["threshold"])["threshold"]
            if feasible else min(model_rows, key=lambda row: row["benign_block_rate"])["threshold"]
        )
    raw = pd.DataFrame(rows)
    raw["selected"] = raw.apply(
        lambda row: bool(np.isclose(row["threshold"], chosen[row["model"]])), axis=1
    )
    return raw, chosen


def _temporal_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, model, repeat_id), group in scores.groupby(
        ["scenario", "model", "repeat"], sort=False
    ):
        exposure_rounds = []
        first_access = []
        betrayal_delays = []
        for _, node in group.groupby("node_id", sort=False):
            node = node.sort_values("round")
            eligible = node["aggregation_eligible"].fillna(False).to_numpy(bool)
            malicious = node["malicious"].to_numpy(int) == 1
            exposure_rounds.append(float(np.logical_and(eligible, malicious).sum()))
            if eligible.any():
                first_access.append(float(node.iloc[np.flatnonzero(eligible)[0]]["round"]))
            transitions = np.flatnonzero(
                np.logical_and(malicious, np.r_[True, ~malicious[:-1]])
            )
            if scenario == "betrayal_after_onboarding" and len(transitions):
                start = int(transitions[0])
                after = eligible[start:]
                blocked = np.flatnonzero(~after)
                betrayal_delays.append(float(blocked[0]) if len(blocked) else float(len(after)))
        rows.append({
            "scenario": scenario,
            "model": model,
            "repeat": int(repeat_id),
            "malicious_exposure_rounds_per_node": float(np.mean(exposure_rounds)),
            "mean_round_to_first_access": float(np.mean(first_access)) if first_access else np.nan,
            "betrayal_detection_delay": (
                float(np.mean(betrayal_delays)) if betrayal_delays else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _paired_tests(per_repeat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "malicious_aggregation_rate",
        "malicious_exposure_rounds_per_node",
        "betrayal_detection_delay",
        "benign_deny_rate",
        "probation_cost_per_node",
    ]
    for scenario in per_repeat["scenario"].unique():
        subset = per_repeat[per_repeat["scenario"] == scenario]
        full = subset[subset["model"] == FULL_MODEL].set_index("repeat")
        for baseline in MODELS[1:]:
            other = subset[subset["model"] == baseline].set_index("repeat")
            for metric in metrics:
                paired = pd.concat([full[metric], other[metric]], axis=1).dropna()
                paired.columns = ["full", "baseline"]
                delta = paired["full"] - paired["baseline"]
                p_value = (
                    float(wilcoxon(delta).pvalue)
                    if len(delta) and not np.allclose(delta, 0.0) else 1.0
                )
                rows.append({
                    "scenario": scenario,
                    "baseline": baseline,
                    "metric": metric,
                    "full_minus_baseline_mean": (
                        float(delta.mean()) if len(delta) else np.nan
                    ),
                    "wilcoxon_p": p_value,
                    "pairs": int(len(delta)),
                })
    return pd.DataFrame(rows)


def _figures(
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    paths = []
    plot_summary = summary.copy()
    plot_summary["method"] = plot_summary["model"].map(DISPLAY_NAMES)
    attacks = plot_summary[plot_summary["scenario"] != "honest_newcomer"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=attacks,
        x="scenario",
        y="malicious_aggregation_rate_mean",
        hue="method",
        ax=ax,
    )
    ax.set_title("Matched-usability baselines under complex attacks")
    ax.set_xlabel("")
    ax.set_ylabel("malicious formal-aggregation rate")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    path = output_dir / "robustness_attack_comparison.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    betrayal = plot_summary[
        plot_summary["scenario"] == "betrayal_after_onboarding"
    ]
    selective = plot_summary[plot_summary["scenario"] == "selective_feedback"]
    sns.barplot(
        data=betrayal, y="method", x="betrayal_detection_delay_mean", ax=axes[0]
    )
    sns.barplot(
        data=selective, y="method", x="malicious_exposure_rounds_per_node_mean",
        ax=axes[1],
    )
    axes[0].set_title("Betrayal detection delay")
    axes[1].set_title("Selective-feedback exposure")
    axes[0].set_xlabel("rounds until first block")
    axes[1].set_xlabel("accepted malicious rounds per node")
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    fig.tight_layout()
    path = output_dir / "robustness_temporal_metrics.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    selected = calibration[calibration["selected"]].copy()
    selected["method"] = selected["model"].map(DISPLAY_NAMES)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=selected, y="method", x="threshold", ax=ax)
    ax.set_title("Per-model thresholds calibrated on benign validation data")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    path = output_dir / "robustness_calibrated_thresholds.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run_robustness_validation(config: dict, project_dir: Path) -> dict[str, Path]:
    suite = config["robustness_validation"]
    sim_cfg = SimulationConfig(rounds=int(suite["rounds"]), **config["simulation"])
    model_cfg = ModelConfig(**config["models"])
    calibration, thresholds = _calibrate_thresholds(
        sim_cfg,
        model_cfg,
        int(suite["validation_seed"]),
        int(suite["validation_repeats"]),
        int(suite["nodes_per_scenario"]),
        [float(x) for x in suite["threshold_grid"]],
        float(suite["target_benign_block_rate"]),
        int(suite["calibration_warmup_round"]),
    )
    events = _build_events(
        sim_cfg,
        list(suite["scenarios"]),
        int(suite["test_seed"]),
        int(suite["test_repeats"]),
        int(suite["nodes_per_scenario"]),
    )
    frames = []
    for model in MODELS:
        threshold = thresholds[model]
        cfg = replace(
            model_cfg,
            limited_access_threshold=threshold,
            full_access_threshold=min(0.99, threshold + 0.10),
        )
        frames.append(run_models(events, cfg, [model]))
    scores = pd.concat(frames, ignore_index=True)
    scores = scores.merge(
        events[["repeat", "round", "node_id", "scenario"]],
        on=["repeat", "round", "node_id"],
        how="left",
        validate="many_to_one",
    )
    base_metrics = _metrics(scores)
    temporal = _temporal_metrics(scores)
    per_repeat = base_metrics.merge(
        temporal, on=["scenario", "model", "repeat"], validate="one_to_one"
    )
    summary = _summary(per_repeat)
    paired = _paired_tests(per_repeat)

    output_dir = project_dir / str(suite["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "calibration": output_dir / "robustness_threshold_calibration.csv",
        "per_repeat": output_dir / "robustness_metrics_per_repeat.csv",
        "summary": output_dir / "robustness_summary.csv",
        "paired_tests": output_dir / "robustness_paired_tests.csv",
    }
    calibration.to_csv(paths["calibration"], index=False)
    per_repeat.to_csv(paths["per_repeat"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["paired_tests"], index=False)
    for path in _figures(
        summary, calibration, output_dir, int(config["reporting"]["figure_dpi"])
    ):
        paths[path.stem] = path
    return paths

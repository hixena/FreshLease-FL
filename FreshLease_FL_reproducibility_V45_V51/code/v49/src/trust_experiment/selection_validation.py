from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .models import ModelConfig
from .novelty_validation import _make_scenario, _set_probe_quality
from .runner import run_models
from .simulation import SimulationConfig


MODELS = [
    "proposed_dirichlet_v1_selection_aware",
    "proposed_dirichlet_v1_controlled_probation",
    "probation_naive",
]


def _events_for_rate(
    sim_cfg: SimulationConfig,
    kind: str,
    rate: float,
    seed: int,
    repeat_id: int,
    nodes: int,
) -> pd.DataFrame:
    scenario = "honest_newcomer" if kind == "benign_loss" else "selective_feedback"
    events = _make_scenario(sim_cfg, scenario, seed, repeat_id, nodes)
    rng = np.random.default_rng(seed + 99173)
    if kind == "selective_submit":
        # Every task has a signed receipt, but only a fraction of favorable
        # results is revealed. This avoids treating non-delivery as misconduct.
        completed = rng.random(len(events)) < rate
        _set_probe_quality(events, rng.uniform(0.82, 0.98, len(events)), 0.90)
        events["probation_task_acknowledged"] = 1
        events["probation_task_completed"] = completed.astype(int)
    else:
        missing = rng.random(len(events)) < rate
        events["probation_task_acknowledged"] = 1
        events["probation_task_completed"] = (~missing).astype(int)
    events["experiment_kind"] = kind
    events["rate"] = float(rate)
    events["scenario"] = f"{kind}_{rate:.2f}"
    events["node_id"] = events["scenario"] + "__" + events["node_id"].astype(str)
    return events


def _metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["experiment_kind", "rate", "model", "repeat"]
    for key, group in scores.groupby(keys, sort=False):
        kind, rate, model, repeat_id = key
        node_graduated = group.groupby("node_id")["onboarding_complete"].max()
        eligible = group["aggregation_eligible"].fillna(False).astype(bool)
        malicious = group["malicious"].astype(bool)
        final = group.sort_values("round").groupby("node_id").tail(1)
        rows.append({
            "experiment_kind": kind,
            "rate": float(rate),
            "model": model,
            "repeat": int(repeat_id),
            "node_graduation_rate": float(node_graduated.mean()),
            "malicious_aggregation_rate": (
                float(eligible[malicious].mean()) if malicious.any() else np.nan
            ),
            "benign_block_rate": (
                float((~eligible[~malicious]).mean()) if (~malicious).any() else np.nan
            ),
            "mean_final_missing_rate": float(
                final["acknowledged_missing_rate"].mean()
            ),
            "mean_probation_cost": float(
                final["probation_total_cost"].mean()
            ),
        })
    return pd.DataFrame(rows)


def _summarize(per_repeat: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "node_graduation_rate", "malicious_aggregation_rate",
        "benign_block_rate", "mean_final_missing_rate", "mean_probation_cost",
    ]
    rows = []
    for key, group in per_repeat.groupby(
        ["experiment_kind", "rate", "model"], sort=False
    ):
        kind, rate, model = key
        row = {"experiment_kind": kind, "rate": rate, "model": model}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_ci95"] = (
                float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    attacks = summary[summary["experiment_kind"] == "selective_submit"]
    benign = summary[summary["experiment_kind"] == "benign_loss"]
    sns.lineplot(
        data=attacks, x="rate", y="node_graduation_rate_mean",
        hue="model", marker="o", ax=axes[0],
    )
    sns.lineplot(
        data=benign, x="rate", y="node_graduation_rate_mean",
        hue="model", marker="o", ax=axes[1],
    )
    axes[0].set_title("Selective reporter graduation")
    axes[0].set_xlabel("fraction of acknowledged tasks with returned result")
    axes[1].set_title("Benign-node graduation under result loss")
    axes[1].set_xlabel("acknowledged-result loss rate")
    for axis in axes:
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("node graduation rate")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_selection_validation(config: dict, project_dir: Path) -> dict[str, Path]:
    suite = config["selection_validation"]
    sim_cfg = SimulationConfig(rounds=int(suite["rounds"]), **config["simulation"])
    model_cfg = replace(
        ModelConfig(**config["models"]),
        # Keep consecutive-failure quarantine from masking the cumulative
        # selection-aware missingness effect in this dedicated experiment.
        probation_max_consecutive_failures=int(suite["max_consecutive_failures"]),
    )
    frames = []
    for repeat_id in range(int(suite["repeats"])):
        for index, rate in enumerate(suite["selective_submit_rates"]):
            frames.append(_events_for_rate(
                sim_cfg, "selective_submit", float(rate),
                int(suite["seed"]) + repeat_id * 1009 + index * 100003,
                repeat_id, int(suite["nodes"]),
            ))
        for index, rate in enumerate(suite["benign_loss_rates"]):
            frames.append(_events_for_rate(
                sim_cfg, "benign_loss", float(rate),
                int(suite["seed"]) + 5000003 + repeat_id * 1009 + index * 100003,
                repeat_id, int(suite["nodes"]),
            ))
    events = pd.concat(frames, ignore_index=True)
    scores = run_models(events, model_cfg, MODELS)
    scores = scores.merge(
        events[["repeat", "round", "node_id", "experiment_kind", "rate"]],
        on=["repeat", "round", "node_id"], how="left", validate="many_to_one",
    )
    per_repeat = _metrics(scores)
    summary = _summarize(per_repeat)
    output_dir = project_dir / str(suite["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_repeat": output_dir / "selection_metrics_per_repeat.csv",
        "summary": output_dir / "selection_summary.csv",
        "figure": output_dir / "selection_graduation_tradeoff.png",
    }
    per_repeat.to_csv(paths["per_repeat"], index=False)
    summary.to_csv(paths["summary"], index=False)
    _plot(summary, paths["figure"], int(config["reporting"]["figure_dpi"]))
    return paths

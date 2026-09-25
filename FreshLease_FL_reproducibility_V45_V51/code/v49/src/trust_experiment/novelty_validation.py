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
from .runner import run_models
from .simulation import SimulationConfig, simulate_events


FULL_MODEL = "proposed_dirichlet_v1_controlled_probation"
MODELS = [
    FULL_MODEL,
    "probation_no_diversity",
    "probation_no_repeat_decay",
    "probation_no_semantic_separation",
    "probation_naive",
    "probation_unbounded",
]


def _set_probe_quality(events: pd.DataFrame, quality: np.ndarray | float,
                       reliability: float = 0.95) -> None:
    values = np.broadcast_to(np.asarray(quality, dtype=float), len(events))
    events["probation_observed_quality"] = values
    events["probation_beta_feedback"] = (values >= 0.60).astype(int)
    events["probation_dirichlet_level"] = np.digitize(
        values, np.asarray([0.20, 0.40, 0.60, 0.80]), right=False
    ).astype(int)
    events["probation_reliability"] = reliability


def _schedule_per_node(events: pd.DataFrame, pattern: list[str]) -> None:
    for _, indices in events.groupby(["repeat", "node_id"], sort=False).groups.items():
        ordered = list(indices)
        events.loc[ordered, "probation_task_type"] = [
            pattern[i % len(pattern)] for i in range(len(ordered))
        ]
    events["probation_task_completed"] = 1
    cost = {"attestation": 0.20, "protocol_check": 0.40,
            "canary_training": 0.80, "shadow_update": 1.00}
    events["probation_task_cost"] = events["probation_task_type"].map(cost)


def _make_scenario(base: SimulationConfig, scenario: str, seed: int,
                   repeat_id: int, nodes: int) -> pd.DataFrame:
    if scenario.startswith("honest_"):
        profile = "stable_benign"
    elif scenario == "betrayal_after_onboarding":
        profile = "betrayal"
    elif scenario == "intermittent_on_off":
        profile = "on_off"
    else:
        profile = "persistent_malicious"
    cfg = replace(base, profiles=[profile] * nodes)
    events = simulate_events(cfg, seed, repeat_id)
    # Scenario-qualified IDs prevent independent scenarios from sharing model
    # state when they are concatenated for one vectorized experiment run.
    events["node_id"] = scenario + "__" + events["node_id"].astype(str)

    if scenario == "single_type_farming":
        _schedule_per_node(events, ["shadow_update"])
        _set_probe_quality(events, 0.95)
    elif scenario == "two_type_farming":
        _schedule_per_node(events, ["canary_training", "shadow_update"])
        _set_probe_quality(events, 0.95)
    elif scenario == "low_reliability_farming":
        _schedule_per_node(events, [
            "attestation", "protocol_check", "canary_training", "shadow_update"
        ])
        _set_probe_quality(events, 0.95, reliability=0.35)
    elif scenario == "diverse_then_repeat_farming":
        # Satisfy the type-count requirement once with weak behavior, then try
        # to overwrite it by farming one apparently excellent task type.
        for _, indices in events.groupby(["repeat", "node_id"], sort=False).groups.items():
            ordered = list(indices)
            tasks = ["attestation", "protocol_check", "canary_training"] \
                + ["shadow_update"] * max(0, len(ordered) - 3)
            events.loc[ordered, "probation_task_type"] = tasks[:len(ordered)]
        events["probation_task_completed"] = 1
        cost = {"attestation": 0.20, "protocol_check": 0.40,
                "canary_training": 0.80, "shadow_update": 1.00}
        events["probation_task_cost"] = events["probation_task_type"].map(cost)
        quality = np.where(
            events["probation_task_type"].eq("shadow_update"), 0.95,
            np.where(events["probation_task_type"].eq("attestation"), 0.98, 0.10),
        )
        _set_probe_quality(events, quality)
    elif scenario == "attestation_camouflage":
        # Establish type coverage once, then flood clean device attestations.
        _schedule_per_node(events, [
            "protocol_check", "canary_training", "shadow_update",
            "attestation", "attestation", "attestation", "attestation",
        ])
        quality = np.where(
            events["probation_task_type"].eq("attestation"),
            0.98,
            events["observed_quality"].to_numpy(dtype=float),
        )
        _set_probe_quality(events, quality)
    elif scenario == "honest_newcomer":
        _schedule_per_node(events, [
            "attestation", "protocol_check", "canary_training", "shadow_update"
        ])
        # Keep naturally noisy, source-reliability-weighted benign feedback.
    elif scenario == "honest_partial_coverage":
        # A heterogeneous benign node for which one optional probe modality is
        # unavailable; used to reveal the utility cost of requiring all types.
        _schedule_per_node(events, [
            "attestation", "canary_training", "shadow_update"
        ])
    elif scenario == "all_probe_evidence_spoofed":
        _schedule_per_node(events, [
            "attestation", "protocol_check", "canary_training", "shadow_update"
        ])
        _set_probe_quality(events, 0.95)
    elif scenario == "colluding_probe_sources":
        # Several apparently independent probe types return mutually
        # consistent but collusively forged positive evidence.
        _schedule_per_node(events, [
            "attestation", "protocol_check", "canary_training", "shadow_update"
        ])
        _set_probe_quality(events, 0.95, reliability=0.95)
    elif scenario == "selective_feedback":
        # The node only returns probe outcomes that look favorable and drops
        # unfavorable attempts. Missing outcomes still consume attempt/cost
        # budget and raise the consecutive-failure counter.
        _schedule_per_node(events, [
            "attestation", "protocol_check", "canary_training", "shadow_update"
        ])
        rng = np.random.default_rng(seed + 700001)
        favorable = rng.random(len(events)) < 0.35
        quality = np.where(
            favorable,
            rng.uniform(0.82, 0.98, len(events)),
            rng.uniform(0.05, 0.35, len(events)),
        )
        _set_probe_quality(events, quality, reliability=0.90)
        events["probation_task_completed"] = favorable.astype(int)
    elif scenario in {"betrayal_after_onboarding", "intermittent_on_off"}:
        _schedule_per_node(events, [
            "attestation", "protocol_check", "canary_training", "shadow_update"
        ])
        # Natural observations from the time-varying profile are retained.
    else:
        raise ValueError(f"Unknown novelty-validation scenario: {scenario}")
    events["scenario"] = scenario
    return events


def _metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (scenario, model, repeat_id), group in scores.groupby(
        ["scenario", "model", "repeat"], sort=False
    ):
        malicious = group["malicious"].to_numpy(dtype=int) == 1
        benign = ~malicious
        eligible = group["aggregation_eligible"].fillna(False).to_numpy(dtype=bool)
        full = group["access_state"].astype(str).eq("full_access").to_numpy()
        probation = group["access_state"].astype(str).eq("probation").to_numpy()
        quarantine = group["access_state"].astype(str).eq("quarantine").to_numpy()
        completed = group["onboarding_complete"].fillna(False).to_numpy(dtype=bool)
        node_completed = []
        times = []
        for _, node in group.groupby("node_id", sort=False):
            values = node["onboarding_complete"].fillna(False).to_numpy(dtype=bool)
            node_completed.append(float(values.any()))
            if values.any():
                first = int(np.flatnonzero(values)[0])
                times.append(float(node.iloc[first]["round"] - node.iloc[0]["round"]))
        rows.append({
            "scenario": scenario,
            "model": model,
            "repeat": int(repeat_id),
            "malicious_aggregation_rate": float(eligible[malicious].mean()) if malicious.any() else np.nan,
            "malicious_full_access_rate": float(full[malicious].mean()) if malicious.any() else np.nan,
            "benign_deny_rate": float(group.loc[benign, "access_state"].eq("deny").mean()) if benign.any() else np.nan,
            "benign_probation_rate": float(probation[benign].mean()) if benign.any() else np.nan,
            "quarantine_event_rate": float(quarantine.mean()),
            "terminal_denial_event_rate": float(
                group["permanently_denied"].astype("boolean").fillna(False).mean()
            ),
            "node_graduation_rate": float(np.mean(node_completed)),
            "mean_time_to_graduation": float(np.mean(times)) if times else np.nan,
            "probation_cost_per_node": float(
                group["probation_task_cost"].sum() / group["node_id"].nunique()
            ),
            "budget_exhaustions_per_node": float(
                group.sort_values("round").groupby("node_id")[
                    "probation_budget_exhaustions"
                ].last().mean()
            ),
            "onboarding_event_rate": float(completed.mean()),
        })
    return pd.DataFrame(rows)


def _summary(per_repeat: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in per_repeat.columns if c not in {"scenario", "model", "repeat"}]
    rows = []
    for (scenario, model), group in per_repeat.groupby(["scenario", "model"], sort=False):
        row = {"scenario": scenario, "model": model}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_ci95"] = (
                float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_tests(per_repeat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "malicious_aggregation_rate", "malicious_full_access_rate",
        "node_graduation_rate", "mean_time_to_graduation", "probation_cost_per_node",
        "budget_exhaustions_per_node",
    ]
    for scenario in per_repeat["scenario"].unique():
        subset = per_repeat[per_repeat["scenario"] == scenario]
        full = subset[subset["model"] == FULL_MODEL].set_index("repeat")
        for model in MODELS[1:]:
            other = subset[subset["model"] == model].set_index("repeat")
            for metric in metrics:
                paired = pd.concat([full[metric], other[metric]], axis=1).dropna()
                paired.columns = ["full", "ablation"]
                delta = paired["full"] - paired["ablation"]
                if len(delta) and not np.allclose(delta, 0.0):
                    p_value = float(wilcoxon(delta).pvalue)
                else:
                    p_value = 1.0
                rows.append({
                    "scenario": scenario, "ablation": model, "metric": metric,
                    "full_minus_ablation_mean": float(delta.mean()) if len(delta) else np.nan,
                    "wilcoxon_p": p_value, "pairs": len(delta),
                })
    return pd.DataFrame(rows)


def _figures(summary: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    paths: list[Path] = []
    attack = summary[summary["scenario"].isin([
        "single_type_farming", "diverse_then_repeat_farming",
        "attestation_camouflage", "all_probe_evidence_spoofed"
    ])]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=attack, x="scenario", y="malicious_aggregation_rate_mean",
                hue="model", ax=ax)
    ax.set_ylabel("malicious formal-aggregation rate")
    ax.set_xlabel("")
    ax.set_title("Attack exposure after controlled probation")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    path = output_dir / "novelty_attack_exposure.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    farming = summary[summary["scenario"] == "single_type_farming"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=farming, y="model", x="probation_cost_per_node_mean", ax=axes[0])
    sns.barplot(data=farming, y="model", x="terminal_denial_event_rate_mean", ax=axes[1])
    axes[0].set_title("Single-type farming: verification cost")
    axes[1].set_title("Terminal-denial event rate")
    fig.tight_layout()
    path = output_dir / "novelty_budget_guard.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    honest = summary[summary["scenario"] == "honest_newcomer"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=honest, y="model", x="node_graduation_rate_mean", ax=axes[0])
    sns.barplot(data=honest, y="model", x="mean_time_to_graduation_mean", ax=axes[1])
    axes[0].set_title("Honest-node graduation rate")
    axes[1].set_title("Mean rounds to graduation")
    fig.tight_layout()
    path = output_dir / "novelty_honest_utility.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run_novelty_validation(config: dict, project_dir: Path) -> dict[str, Path]:
    suite = config["novelty_validation"]
    sim_cfg = SimulationConfig(rounds=int(suite["rounds"]), **config["simulation"])
    model_cfg = ModelConfig(**config["models"])
    scenarios = list(suite["scenarios"])
    frames = []
    for repeat_id in range(int(suite["repeats"])):
        for scenario_index, scenario in enumerate(scenarios):
            frames.append(_make_scenario(
                sim_cfg, scenario,
                int(suite["seed"]) + repeat_id * 1009 + scenario_index * 100003,
                repeat_id, int(suite["nodes_per_scenario"]),
            ))
    events = pd.concat(frames, ignore_index=True)
    scores = run_models(events, model_cfg, MODELS)
    scores = scores.merge(
        events[["repeat", "round", "node_id", "scenario"]],
        on=["repeat", "round", "node_id"], how="left", validate="many_to_one",
    )
    per_repeat = _metrics(scores)
    summary = _summary(per_repeat)
    paired = _paired_tests(per_repeat)

    output_dir = project_dir / str(suite["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_repeat": output_dir / "novelty_metrics_per_repeat.csv",
        "summary": output_dir / "novelty_summary.csv",
        "paired_tests": output_dir / "novelty_paired_tests.csv",
    }
    if bool(suite.get("save_events", False)):
        paths["events"] = output_dir / "novelty_events.csv"
        events.to_csv(paths["events"], index=False)
    if bool(suite.get("save_round_scores", False)):
        paths["scores"] = output_dir / "novelty_round_scores.csv"
        scores.to_csv(paths["scores"], index=False)
    per_repeat.to_csv(paths["per_repeat"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["paired_tests"], index=False)
    for path in _figures(summary, output_dir, int(config["reporting"]["figure_dpi"])):
        paths[path.stem] = path
    return paths

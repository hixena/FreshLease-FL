from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import calculate_metrics
from .models import ModelConfig, build_models
from .simulation import SimulationConfig, simulate_events


def run_models(
    events: pd.DataFrame,
    model_cfg: ModelConfig,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    outputs: list[dict] = []
    for (repeat, node_id), node_events in events.groupby(["repeat", "node_id"], sort=False):
        models = build_models(model_cfg, model_names)
        for row in node_events.itertuples(index=False):
            for model in models:
                # Causal order: the access decision at t only sees history
                # through t-1. The signed post-access result updates t+1.
                result = model.decide(
                    current=float(row.current_evidence),
                    current_confidence=float(getattr(row, "current_confidence", 1.0)),
                )
                result.setdefault("decision_score", result["trust_score"])
                result.setdefault("access_state", "not_applicable")
                requires_probation = bool(result.get("requires_probation_task", False))
                probation_task_type = str(result.get(
                    "required_probation_evidence_type",
                    getattr(row, "probation_task_type", "none"),
                ))
                task_completed = bool(
                    requires_probation
                    and int(getattr(row, "probation_task_completed", 0))
                )
                task_acknowledged = bool(
                    requires_probation
                    and int(getattr(row, "probation_task_acknowledged", 1))
                )
                task_completed = task_completed and task_acknowledged
                access_state = str(result["access_state"])
                result.setdefault(
                    "aggregation_eligible",
                    access_state in {"limited_access", "full_access", "not_applicable"},
                )
                outputs.append({
                    "repeat": repeat,
                    "round": int(row.round),
                    "node_id": node_id,
                    "profile": row.profile,
                    "malicious": int(row.malicious),
                    "current_evidence": float(row.current_evidence),
                    "current_confidence": float(getattr(row, "current_confidence", 1.0)),
                    "reliability": float(row.reliability),
                    "feedback_corrupted": int(row.feedback_corrupted),
                    "probation_task_executed": int(requires_probation),
                    "probation_task_type": (
                        probation_task_type
                        if requires_probation else "none"
                    ),
                    "probation_task_completed_this_round": int(task_completed),
                    "probation_task_acknowledged_this_round": int(task_acknowledged),
                    "probation_task_cost": (
                        float(result.get(
                            "required_probation_task_cost",
                            getattr(row, "probation_task_cost", 0.0),
                        ))
                        if requires_probation else 0.0
                    ),
                    "probation_affects_aggregation": 0 if requires_probation else np.nan,
                    "model": model.name,
                    **result,
                })
                if requires_probation:
                    model.record_probation_attempt(
                        beta_feedback=int(getattr(
                            row, "probation_beta_feedback", row.beta_feedback
                        )),
                        level=int(getattr(
                            row, "probation_dirichlet_level", row.dirichlet_level
                        )),
                        reliability=float(getattr(
                            row, "probation_reliability", row.reliability
                        )),
                        evidence_type=probation_task_type,
                        completed=task_completed,
                        cost=float(result.get(
                            "required_probation_task_cost",
                            getattr(row, "probation_task_cost", 0.0),
                        )),
                        acknowledged=task_acknowledged,
                    )
                elif (
                    bool(result["aggregation_eligible"])
                    or not bool(result.get(
                        "blocks_formal_observation_when_ineligible", False
                    ))
                ):
                    model.observe(
                        beta_feedback=int(row.beta_feedback),
                        level=int(row.dirichlet_level),
                        reliability=float(row.reliability),
                        evidence_type="formal_training",
                    )
    return pd.DataFrame(outputs)


def summarize(round_df: pd.DataFrame, threshold: float, bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_repeat: list[dict] = []
    for (model, repeat), group in round_df.groupby(["model", "repeat"], sort=False):
        per_repeat.append({"model": model, "repeat": repeat, **calculate_metrics(group, threshold, bins)})
    per_repeat_df = pd.DataFrame(per_repeat)
    metric_cols = [c for c in per_repeat_df.columns if c not in {"model", "repeat"}]
    rows: list[dict] = []
    for model, group in per_repeat_df.groupby("model", sort=False):
        row = {"model": model}
        for metric in metric_cols:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_ci95"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
        rows.append(row)
    return per_repeat_df, pd.DataFrame(rows)


def summarize_by_profile(round_df: pd.DataFrame, threshold: float, bins: int) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, profile, repeat), group in round_df.groupby(["model", "profile", "repeat"], sort=False):
        rows.append({
            "model": model,
            "profile": profile,
            "repeat": repeat,
            **calculate_metrics(group, threshold, bins),
        })
    return pd.DataFrame(rows)


def threshold_analysis(round_df: pd.DataFrame, target_frr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for threshold in np.linspace(0.20, 0.90, 71):
        for (model, repeat), group in round_df.groupby(["model", "repeat"], sort=False):
            y = group["malicious"].to_numpy(dtype=int)
            decision = (
                group["decision_score"].to_numpy(dtype=float)
                if "decision_score" in group.columns
                else group["trust_score"].to_numpy(dtype=float)
            )
            accepted = decision >= threshold
            malicious = y == 1
            benign = y == 0
            rows.append({
                "model": model,
                "repeat": repeat,
                "threshold": float(threshold),
                "far": float(accepted[malicious].mean()),
                "frr": float((~accepted[benign]).mean()),
            })
    raw = pd.DataFrame(rows)
    curve = raw.groupby(["model", "threshold"], as_index=False).agg(
        far_mean=("far", "mean"),
        far_std=("far", "std"),
        frr_mean=("frr", "mean"),
        frr_std=("frr", "std"),
    )
    operating: list[pd.Series] = []
    for _, group in curve.groupby("model", sort=False):
        eligible = group[group["frr_mean"] <= target_frr]
        chosen = (
            eligible.sort_values(["far_mean", "threshold"], ascending=[True, False]).iloc[0]
            if not eligible.empty
            else group.sort_values("frr_mean").iloc[0]
        )
        operating.append(chosen)
    operating_df = pd.DataFrame(operating).reset_index(drop=True)
    operating_df["target_frr"] = target_frr
    return curve, operating_df


def run_experiment(config: dict, project_dir: Path) -> dict[str, Path]:
    exp = config["experiment"]
    sim_raw = config["simulation"]
    model_raw = config["models"]
    reporting = config["reporting"]

    sim_cfg = SimulationConfig(rounds=int(exp["rounds"]), **sim_raw)
    model_cfg = ModelConfig(**model_raw)
    all_events = []
    for repeat in range(int(exp["repeats"])):
        seed = int(exp["seed"]) + repeat * 1009
        all_events.append(simulate_events(sim_cfg, seed, repeat))
    events = pd.concat(all_events, ignore_index=True)
    round_df = run_models(events, model_cfg)
    per_repeat, summary = summarize(
        round_df,
        float(exp["decision_threshold"]),
        int(reporting["calibration_bins"]),
    )
    by_profile = summarize_by_profile(
        round_df,
        float(exp["decision_threshold"]),
        int(reporting["calibration_bins"]),
    )
    threshold_curve, operating_points = threshold_analysis(
        round_df, float(reporting["target_frr"])
    )

    output_dir = project_dir / str(exp["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "events": output_dir / "simulated_events.csv",
        "rounds": output_dir / "round_level_scores.csv",
        "per_repeat": output_dir / "metrics_per_repeat.csv",
        "summary": output_dir / "metrics_summary.csv",
        "by_profile": output_dir / "metrics_by_profile.csv",
        "threshold_curve": output_dir / "threshold_sweep.csv",
        "operating_points": output_dir / "operating_points.csv",
    }
    events.to_csv(paths["events"], index=False)
    if bool(reporting["save_round_level_data"]):
        round_df.to_csv(paths["rounds"], index=False)
    per_repeat.to_csv(paths["per_repeat"], index=False)
    summary.to_csv(paths["summary"], index=False)
    by_profile.to_csv(paths["by_profile"], index=False)
    threshold_curve.to_csv(paths["threshold_curve"], index=False)
    operating_points.to_csv(paths["operating_points"], index=False)
    return paths

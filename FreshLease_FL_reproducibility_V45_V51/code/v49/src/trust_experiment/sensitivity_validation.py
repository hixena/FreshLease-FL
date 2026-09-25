from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import ModelConfig
from .novelty_validation import FULL_MODEL, _make_scenario, _metrics
from .runner import run_models
from .simulation import SimulationConfig


def _setting_rows(config: dict) -> list[dict]:
    sweep = config["sensitivity"]["parameters"]
    rows: list[dict] = []
    for value in sweep["probation_min_evidence_types"]:
        rows.append({
            "parameter": "min_evidence_types", "value": str(int(value)),
            "value_numeric": float(value),
            "overrides": {"probation_min_evidence_types": int(value)},
            "attack_scenario": "two_type_farming",
            "honest_scenario": "honest_partial_coverage",
        })
    for value in sweep["probation_repeat_decay_power"]:
        rows.append({
            "parameter": "repeat_decay_power", "value": f"{float(value):g}",
            "value_numeric": float(value),
            "overrides": {"probation_repeat_decay_power": float(value)},
            "attack_scenario": "diverse_then_repeat_farming",
            "honest_scenario": "honest_newcomer",
        })
    for value in sweep["probation_min_evidence_mass"]:
        rows.append({
            "parameter": "min_evidence_mass", "value": f"{float(value):g}",
            "value_numeric": float(value),
            "overrides": {"probation_min_evidence_mass": float(value)},
            "attack_scenario": "low_reliability_farming",
            "honest_scenario": "honest_newcomer",
        })
    for item in sweep["probation_budget_pairs"]:
        attempts, cost = int(item["attempts"]), float(item["cost"])
        rows.append({
            "parameter": "probation_budget", "value": f"{attempts}/{cost:g}",
            "value_numeric": float(cost),
            "overrides": {
                "probation_max_attempts": attempts,
                "probation_max_cost": cost,
            },
            "attack_scenario": "single_type_farming",
            "honest_scenario": "honest_newcomer",
        })
    return rows


def _summarize(per_repeat: pd.DataFrame) -> pd.DataFrame:
    id_cols = {
        "parameter", "value", "value_numeric", "split", "scenario",
        "model", "repeat",
    }
    metrics = [c for c in per_repeat.columns if c not in id_cols]
    rows: list[dict] = []
    group_cols = ["parameter", "value", "value_numeric", "split", "scenario"]
    for keys, group in per_repeat.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_ci95"] = (
                float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _tradeoff_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (parameter, value, value_numeric, split), group in summary.groupby(
        ["parameter", "value", "value_numeric", "split"], sort=False
    ):
        honest_mask = group["scenario"].astype(str).str.startswith("honest_")
        attack = group[~honest_mask].iloc[0]
        honest = group[honest_mask].iloc[0]
        rows.append({
            "parameter": parameter,
            "value": value,
            "value_numeric": value_numeric,
            "split": split,
            "attack_scenario": attack["scenario"],
            "malicious_aggregation_rate": attack["malicious_aggregation_rate_mean"],
            "attack_probation_cost": attack["probation_cost_per_node_mean"],
            "honest_graduation_rate": honest["node_graduation_rate_mean"],
            "honest_time_to_graduation": honest["mean_time_to_graduation_mean"],
            "honest_deny_rate": honest["benign_deny_rate_mean"],
            "honest_probation_cost": honest["probation_cost_per_node_mean"],
            "utility_feasible": bool(
                honest["node_graduation_rate_mean"] >= 0.99
                and honest["benign_deny_rate_mean"] <= 0.01
            ),
        })
    return pd.DataFrame(rows)


def _plot(tradeoff: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    parameters = [
        "min_evidence_types", "repeat_decay_power",
        "min_evidence_mass", "probation_budget",
    ]
    test = tradeoff[tradeoff["split"] == "test"]
    fig, axes = plt.subplots(len(parameters), 3, figsize=(14, 14))
    for row_index, parameter in enumerate(parameters):
        data = test[test["parameter"] == parameter].sort_values("value_numeric")
        x = np.arange(len(data))
        labels = data["value"].astype(str).tolist()
        panels = [
            ("malicious_aggregation_rate", "Malicious aggregation rate"),
            ("honest_graduation_rate", "Honest graduation rate"),
            ("honest_time_to_graduation", "Honest graduation rounds"),
        ]
        for col_index, (metric, title) in enumerate(panels):
            ax = axes[row_index, col_index]
            ax.plot(x, data[metric], marker="o")
            ax.set_xticks(x, labels)
            ax.set_title(f"{parameter}: {title}")
            ax.grid(alpha=0.3)
            if parameter == "probation_budget":
                ax.set_xlabel("attempts/cost")
            else:
                ax.set_xlabel("parameter value")
    fig.tight_layout()
    path = output_dir / "sensitivity_tradeoffs.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def run_sensitivity_validation(config: dict, project_dir: Path) -> dict[str, Path]:
    suite = config["sensitivity"]
    base_sim = SimulationConfig(rounds=int(suite["rounds"]), **config["simulation"])
    base_model = ModelConfig(**config["models"])
    settings = _setting_rows(config)
    records: list[pd.DataFrame] = []

    split_specs = [
        ("validation", int(suite["validation_repeats"]), int(suite["validation_seed"])),
        ("test", int(suite["test_repeats"]), int(suite["test_seed"])),
    ]
    parameter_order = [
        "min_evidence_types", "repeat_decay_power",
        "min_evidence_mass", "probation_budget",
    ]
    for setting in settings:
        model_cfg = replace(base_model, **setting["overrides"])
        parameter_index = parameter_order.index(setting["parameter"])
        for split, repeats, seed in split_specs:
            frames = []
            for repeat_id in range(repeats):
                for scenario_index, scenario in enumerate([
                    setting["attack_scenario"], setting["honest_scenario"]
                ]):
                    frames.append(_make_scenario(
                        base_sim, scenario,
                        seed + repeat_id * 1009 + parameter_index * 100003
                        + scenario_index * 1000003,
                        repeat_id, int(suite["nodes_per_scenario"]),
                    ))
            events = pd.concat(frames, ignore_index=True)
            scores = run_models(events, model_cfg, [FULL_MODEL])
            scores = scores.merge(
                events[["repeat", "round", "node_id", "scenario"]],
                on=["repeat", "round", "node_id"],
                how="left", validate="many_to_one",
            )
            metrics = _metrics(scores)
            metrics["parameter"] = setting["parameter"]
            metrics["value"] = setting["value"]
            metrics["value_numeric"] = setting["value_numeric"]
            metrics["split"] = split
            records.append(metrics)

    per_repeat = pd.concat(records, ignore_index=True)
    summary = _summarize(per_repeat)
    tradeoff = _tradeoff_table(summary)
    output_dir = project_dir / str(suite["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_repeat": output_dir / "sensitivity_metrics_per_repeat.csv",
        "summary": output_dir / "sensitivity_summary.csv",
        "tradeoffs": output_dir / "sensitivity_tradeoffs.csv",
    }
    per_repeat.to_csv(paths["per_repeat"], index=False)
    summary.to_csv(paths["summary"], index=False)
    tradeoff.to_csv(paths["tradeoffs"], index=False)
    figure = _plot(tradeoff, output_dir, int(config["reporting"]["figure_dpi"]))
    paths["figure"] = figure
    return paths

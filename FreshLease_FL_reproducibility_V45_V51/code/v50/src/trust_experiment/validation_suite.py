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

from .metrics import calculate_metrics
from .models import ModelConfig
from .runner import run_models
from .simulation import SimulationConfig, simulate_events


CORE_MODELS = [
    "ema",
    "beta_decay_mean",
    "dirichlet_decay_mean",
    "proposed_beta",
    "proposed_dirichlet",
    "proposed_dirichlet_v1_min_evidence",
    "proposed_dirichlet_v1_joint_gate",
    "proposed_dirichlet_v1_controlled_probation",
    "proposed_dirichlet_v2",
]

ABLATION_MODELS = [
    "dirichlet_decay_mean",
    "ablation_no_reliability",
    "ablation_no_lcb",
    "ablation_no_adaptive",
    "proposed_dirichlet",
    "v2_no_reliability",
    "v2_no_current_confidence",
    "v2_lcb_as_score",
    "v2_no_probation",
    "proposed_dirichlet_v2",
]


def _simulate_repeats(
    cfg: SimulationConfig,
    seed: int,
    repeats: int,
) -> pd.DataFrame:
    return pd.concat(
        [simulate_events(cfg, seed + repeat * 1009, repeat) for repeat in range(repeats)],
        ignore_index=True,
    )


def _metric_rows(
    scores: pd.DataFrame,
    threshold: float,
    bins: int,
    extra: dict | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, repeat), group in scores.groupby(["model", "repeat"], sort=False):
        row = {"model": model, "repeat": int(repeat)}
        if extra:
            row.update(extra)
        row.update(calculate_metrics(group, threshold, bins))
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in set(group_cols + ["repeat"])
    ]
    rows: list[dict] = []
    grouped = df.groupby(group_cols, sort=False, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in numeric:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{metric}_ci95"] = (
                float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _rates_at_threshold(scores: pd.DataFrame, threshold: float) -> tuple[float, float]:
    malicious = scores["malicious"].to_numpy(dtype=int) == 1
    decision = (
        scores["decision_score"].to_numpy(dtype=float)
        if "decision_score" in scores.columns
        else scores["trust_score"].to_numpy(dtype=float)
    )
    accepted = decision >= threshold
    far = float(accepted[malicious].mean()) if malicious.any() else np.nan
    frr = float((~accepted[~malicious]).mean()) if (~malicious).any() else np.nan
    return far, frr


def _select_threshold(
    validation: pd.DataFrame,
    constraint: str,
    target: float,
    thresholds: np.ndarray,
) -> tuple[float, float, float, bool]:
    candidates = []
    for threshold in thresholds:
        far, frr = _rates_at_threshold(validation, float(threshold))
        candidates.append((float(threshold), far, frr))
    table = pd.DataFrame(candidates, columns=["threshold", "far", "frr"])
    if constraint == "frr":
        eligible = table[table["frr"] <= target]
        chosen = (
            eligible.sort_values(["far", "frr", "threshold"], ascending=[True, True, False]).iloc[0]
            if not eligible.empty else table.sort_values(["frr", "far"]).iloc[0]
        )
    elif constraint == "far":
        eligible = table[table["far"] <= target]
        chosen = (
            eligible.sort_values(["frr", "far", "threshold"], ascending=[True, True, True]).iloc[0]
            if not eligible.empty else table.sort_values(["far", "frr"]).iloc[0]
        )
    else:
        raise ValueError("constraint must be 'far' or 'frr'")
    return (
        float(chosen["threshold"]),
        float(chosen["far"]),
        float(chosen["frr"]),
        not eligible.empty,
    )


def matched_operating_experiment(
    scores: pd.DataFrame,
    output_dir: Path,
    validation_repeats: int,
    target_far: float,
    target_frr: float,
    bins: int,
) -> dict[str, Path]:
    """Select thresholds on validation repeats, then evaluate untouched repeats."""
    repeat_ids = sorted(scores["repeat"].unique())
    if validation_repeats <= 0 or validation_repeats >= len(repeat_ids):
        raise ValueError("validation_repeats must leave at least one test repeat")
    validation_ids = set(repeat_ids[:validation_repeats])
    validation = scores[scores["repeat"].isin(validation_ids)]
    test = scores[~scores["repeat"].isin(validation_ids)]
    grid = np.linspace(0.0, 1.0, 1001)

    thresholds_rows: list[dict] = []
    metric_rows: list[dict] = []
    for model, validation_model in validation.groupby("model", sort=False):
        test_model = test[test["model"] == model]
        for constraint, target in [("frr", target_frr), ("far", target_far)]:
            threshold, val_far, val_frr, feasible = _select_threshold(
                validation_model, constraint, target, grid
            )
            thresholds_rows.append({
                "model": model,
                "constraint": constraint,
                "target": target,
                "threshold": threshold,
                "validation_far": val_far,
                "validation_frr": val_frr,
                "constraint_feasible": feasible,
                "validation_repeats": validation_repeats,
                "test_repeats": len(repeat_ids) - validation_repeats,
            })
            for repeat, group in test_model.groupby("repeat", sort=False):
                metric_rows.append({
                    "model": model,
                    "constraint": constraint,
                    "target": target,
                    "selected_threshold": threshold,
                    "repeat": int(repeat),
                    **calculate_metrics(group, threshold, bins),
                })

    thresholds_df = pd.DataFrame(thresholds_rows)
    metrics_df = pd.DataFrame(metric_rows)
    summary_df = _summary(metrics_df, ["model", "constraint", "target"])
    paths = {
        "matched_thresholds": output_dir / "matched_thresholds.csv",
        "matched_metrics": output_dir / "matched_metrics_per_repeat.csv",
        "matched_summary": output_dir / "matched_metrics_summary.csv",
    }
    thresholds_df.to_csv(paths["matched_thresholds"], index=False)
    metrics_df.to_csv(paths["matched_metrics"], index=False)
    summary_df.to_csv(paths["matched_summary"], index=False)
    return paths


def _inject_pollution(
    events: pd.DataFrame,
    target_rate: float,
    seed: int,
    low_source_multiplier: float,
    bins: list[float],
    beta_threshold: float,
) -> pd.DataFrame:
    """Inject rank-informative noise while keeping expected total rate controlled."""
    if not 0.0 <= target_rate <= 1.0:
        raise ValueError("pollution rate must be in [0, 1]")
    out = events.copy()
    low = out["source_quality"] < out["source_quality"].median()
    low_fraction = float(low.mean())
    p_low = min(1.0, target_rate * low_source_multiplier)
    if low_fraction < 1.0:
        p_high = (target_rate - low_fraction * p_low) / (1.0 - low_fraction)
    else:
        p_high = target_rate
    if p_high < 0.0:
        p_high = 0.0
        p_low = target_rate / max(low_fraction, 1e-12)
    p_high = float(np.clip(p_high, 0.0, 1.0))
    p_low = float(np.clip(p_low, 0.0, 1.0))

    # Common random numbers: the same seed makes higher rates supersets of
    # lower-rate corruption events, reducing Monte Carlo comparison noise.
    draws = np.random.default_rng(seed).random(len(out))
    probability = np.where(low.to_numpy(), p_low, p_high)
    corrupted = draws < probability
    raw = out["post_access_observation"].to_numpy(dtype=float)
    observed = np.where(corrupted, 1.0 - raw, raw)
    out["observed_quality"] = observed
    out["feedback_corrupted"] = corrupted.astype(int)
    out["beta_feedback"] = (observed >= beta_threshold).astype(int)
    out["dirichlet_level"] = np.digitize(
        observed, np.asarray(bins, dtype=float), right=False
    ).astype(int)
    out["pollution_target"] = target_rate
    out["pollution_realized"] = float(corrupted.mean())
    return out


def pollution_experiment(
    base_events: pd.DataFrame,
    model_cfg: ModelConfig,
    sim_cfg: SimulationConfig,
    output_dir: Path,
    rates: list[float],
    threshold: float,
    bins: int,
    seed: int,
    low_source_multiplier: float,
) -> dict[str, Path]:
    frames = []
    selected = [
        "dirichlet_decay_mean",
        "dirichlet_plus_reliability",
        "proposed_dirichlet",
        "proposed_dirichlet_v1_min_evidence",
        "proposed_dirichlet_v1_joint_gate",
        "proposed_dirichlet_v1_controlled_probation",
        "proposed_dirichlet_v2",
    ]
    for rate in rates:
        polluted = _inject_pollution(
            base_events, float(rate), seed, low_source_multiplier,
            sim_cfg.dirichlet_bins, sim_cfg.beta_binary_threshold,
        )
        scores = run_models(polluted, model_cfg, selected)
        rate_rows = []
        actual_by_repeat = polluted.groupby("repeat")["feedback_corrupted"].mean()
        for (model, repeat), group in scores.groupby(["model", "repeat"], sort=False):
            rate_rows.append({
                "model": model,
                "repeat": int(repeat),
                "pollution_target": float(rate),
                "pollution_realized": float(actual_by_repeat.loc[repeat]),
                **calculate_metrics(group, threshold, bins),
            })
        frames.append(pd.DataFrame(rate_rows))
    metrics = pd.concat(frames, ignore_index=True)
    summary = _summary(metrics, ["model", "pollution_target"])
    paths = {
        "pollution_metrics": output_dir / "pollution_metrics_per_repeat.csv",
        "pollution_summary": output_dir / "pollution_summary.csv",
    }
    metrics.to_csv(paths["pollution_metrics"], index=False)
    summary.to_csv(paths["pollution_summary"], index=False)
    return paths


def cold_start_experiment(
    base_cfg: SimulationConfig,
    model_cfg: ModelConfig,
    output_dir: Path,
    counts: list[int],
    repeats: int,
    seed: int,
    threshold: float,
    bins: int,
    current_signal: float,
    current_predictive_quality: float,
    nodes_per_class: int,
) -> dict[str, Path]:
    max_count = max(counts)
    cfg = replace(
        base_cfg,
        rounds=max_count + 1,
        profiles=(
            ["stable_benign"] * nodes_per_class
            + ["persistent_malicious"] * nodes_per_class
        ),
        current_evidence_signal=current_signal,
        current_predictive_quality=current_predictive_quality,
    )
    events = _simulate_repeats(cfg, seed, repeats)
    selected = [
        "dirichlet_decay_mean",
        "dirichlet_plus_lcb",
        "proposed_dirichlet",
        "proposed_dirichlet_v1_min_evidence",
        "proposed_dirichlet_v1_joint_gate",
        "proposed_dirichlet_v1_controlled_probation",
        "proposed_dirichlet_v2",
    ]
    scores = run_models(events, model_cfg, selected)
    rows: list[dict] = []
    for count in counts:
        # At round count, exactly `count` completed feedback items are
        # available because the current round outcome is not leaked.
        snapshot = scores[scores["round"] == count]
        for (model, repeat), group in snapshot.groupby(["model", "repeat"], sort=False):
            row = {
                "model": model,
                "repeat": int(repeat),
                "interaction_index": int(count + 1),
                "history_evidence_available": int(count),
                **calculate_metrics(group, threshold, bins),
            }
            benign = group[group["malicious"] == 0]["trust_score"]
            malicious = group[group["malicious"] == 1]["trust_score"]
            row["benign_score"] = float(benign.mean())
            row["malicious_score"] = float(malicious.mean())
            rows.append(row)
    metrics = pd.DataFrame(rows)
    summary = _summary(
        metrics, ["model", "interaction_index", "history_evidence_available"]
    )
    paths = {
        "cold_start_metrics": output_dir / "cold_start_metrics_per_repeat.csv",
        "cold_start_summary": output_dir / "cold_start_summary.csv",
    }
    metrics.to_csv(paths["cold_start_metrics"], index=False)
    summary.to_csv(paths["cold_start_summary"], index=False)
    return paths


def ablation_experiment(
    events: pd.DataFrame,
    model_cfg: ModelConfig,
    output_dir: Path,
    threshold: float,
    bins: int,
) -> dict[str, Path]:
    scores = run_models(events, model_cfg, ABLATION_MODELS)
    metrics = _metric_rows(scores, threshold, bins)
    summary = _summary(metrics, ["model"])

    full_name = "proposed_dirichlet_v2"
    full = metrics[metrics["model"] == full_name].set_index("repeat")
    deltas: list[dict] = []
    metric_names = [
        "far", "frr", "brier", "ece", "auroc", "auprc",
        "detection_delay", "recovery_delay", "full_access_far", "deny_frr",
        "restricted_benign_rate", "restricted_malicious_rate",
    ]
    for model, group in metrics.groupby("model", sort=False):
        if model == full_name:
            continue
        candidate = group.set_index("repeat")
        common = candidate.index.intersection(full.index)
        for metric in metric_names:
            if metric not in candidate.columns or metric not in full.columns:
                continue
            values = (candidate.loc[common, metric] - full.loc[common, metric]).dropna()
            if len(values) and not np.allclose(values.to_numpy(dtype=float), 0.0):
                p_value = float(wilcoxon(values.to_numpy(dtype=float)).pvalue)
            else:
                p_value = 1.0 if len(values) else np.nan
            deltas.append({
                "model": model,
                "metric": metric,
                "definition": "ablation_or_baseline_minus_full",
                "mean_delta": float(values.mean()) if len(values) else np.nan,
                "ci95": (
                    float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else np.nan
                ),
                "n_pairs": int(len(values)),
                "wilcoxon_p_two_sided": p_value,
            })
    delta_df = pd.DataFrame(deltas)
    paths = {
        "ablation_metrics": output_dir / "ablation_metrics_per_repeat.csv",
        "ablation_summary": output_dir / "ablation_summary.csv",
        "ablation_paired_deltas": output_dir / "ablation_paired_deltas.csv",
    }
    metrics.to_csv(paths["ablation_metrics"], index=False)
    summary.to_csv(paths["ablation_summary"], index=False)
    delta_df.to_csv(paths["ablation_paired_deltas"], index=False)
    return paths


def controlled_probation_experiment(
    scores: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    bins: int,
) -> dict[str, Path]:
    """Report probation as low-risk access rather than conflating it with denial."""
    selected = scores[scores["model"].isin([
        "proposed_dirichlet_v1_joint_gate",
        "proposed_dirichlet_v1_controlled_probation",
    ])].copy()
    metrics = _metric_rows(selected, threshold, bins)
    summary = _summary(metrics, ["model"])

    controlled = selected[
        selected["model"] == "proposed_dirichlet_v1_controlled_probation"
    ].copy()
    curve_rows: list[dict] = []
    for repeat, repeat_df in controlled.groupby("repeat", sort=False):
        node_first = repeat_df.groupby("node_id")["round"].min()
        for round_number, snapshot in repeat_df.groupby("round", sort=True):
            observed = repeat_df[repeat_df["round"] <= round_number]
            latest = observed.sort_values("round").groupby("node_id").tail(1)
            first_benign = (
                observed.sort_values("round").groupby("node_id").head(1)
                .query("malicious == 0")["node_id"]
            )
            benign_latest = latest[latest["node_id"].isin(first_benign)]
            graduated = benign_latest["onboarding_complete"].eq(True)
            malicious_now = snapshot["malicious"] == 1
            aggregation_now = snapshot["aggregation_eligible"].eq(True)
            curve_rows.append({
                "repeat": int(repeat),
                "round": int(round_number),
                "benign_graduation_rate": (
                    float(graduated.mean()) if len(graduated) else np.nan
                ),
                "malicious_aggregation_rate": (
                    float(aggregation_now[malicious_now].mean())
                    if malicious_now.any() else np.nan
                ),
                "cumulative_probation_cost": float(
                    observed["probation_task_cost"].sum()
                ),
                "active_nodes": int((node_first <= round_number).sum()),
            })
    curve = pd.DataFrame(curve_rows)
    curve_summary = _summary(curve, ["round"])

    paths = {
        "controlled_probation_metrics": output_dir / "controlled_probation_metrics_per_repeat.csv",
        "controlled_probation_summary": output_dir / "controlled_probation_summary.csv",
        "controlled_probation_curve": output_dir / "controlled_probation_round_curve.csv",
        "controlled_probation_curve_summary": output_dir / "controlled_probation_round_curve_summary.csv",
    }
    metrics.to_csv(paths["controlled_probation_metrics"], index=False)
    summary.to_csv(paths["controlled_probation_summary"], index=False)
    curve.to_csv(paths["controlled_probation_curve"], index=False)
    curve_summary.to_csv(paths["controlled_probation_curve_summary"], index=False)
    return paths


def _create_suite_figures(output_dir: Path, dpi: int) -> list[Path]:
    sns.set_theme(style="whitegrid")
    paths: list[Path] = []

    matched = pd.read_csv(output_dir / "matched_metrics_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, constraint in zip(axes, ["frr", "far"]):
        part = matched[matched["constraint"] == constraint]
        metric = "far_mean" if constraint == "frr" else "frr_mean"
        sns.barplot(data=part, x=metric, y="model", ax=ax)
        ax.set_title(f"Validation-matched {constraint.upper()}")
    fig.tight_layout()
    path = output_dir / "matched_operating_comparison.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    probation_curve = pd.read_csv(output_dir / "controlled_probation_round_curve_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    axes[0].plot(probation_curve["round"], probation_curve["benign_graduation_rate_mean"])
    axes[1].plot(probation_curve["round"], probation_curve["malicious_aggregation_rate_mean"])
    axes[2].plot(probation_curve["round"], probation_curve["cumulative_probation_cost_mean"])
    axes[0].set_title("Benign graduation rate")
    axes[1].set_title("Malicious formal aggregation rate")
    axes[2].set_title("Cumulative probation cost")
    for ax in axes:
        ax.set_xlabel("round")
    fig.tight_layout()
    path = output_dir / "controlled_probation_tradeoff.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    pollution = pd.read_csv(output_dir / "pollution_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    sns.lineplot(data=pollution, x="pollution_target", y="far_mean", hue="model", marker="o", ax=axes[0])
    sns.lineplot(data=pollution, x="pollution_target", y="auroc_mean", hue="model", marker="o", ax=axes[1])
    axes[0].set_title("Feedback pollution: FAR")
    axes[1].set_title("Feedback pollution: AUROC")
    fig.tight_layout()
    path = output_dir / "pollution_robustness.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    cold = pd.read_csv(output_dir / "cold_start_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    sns.lineplot(data=cold, x="history_evidence_available", y="far_mean", hue="model", marker="o", ax=axes[0])
    sns.lineplot(data=cold, x="history_evidence_available", y="frr_mean", hue="model", marker="o", ax=axes[1])
    axes[0].set_title("Cold start: FAR")
    axes[1].set_title("Cold start: FRR")
    fig.tight_layout()
    path = output_dir / "cold_start_evidence.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    ablation = pd.read_csv(output_dir / "ablation_summary.csv")
    long = ablation.melt(
        id_vars="model", value_vars=["far_mean", "frr_mean", "auroc_mean"],
        var_name="metric", value_name="value",
    )
    g = sns.catplot(data=long, x="value", y="model", col="metric", kind="bar", sharex=False, height=3.5, aspect=1.0)
    g.figure.suptitle("Ablation study", y=1.03)
    path = output_dir / "ablation_comparison.png"
    g.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(g.figure)
    paths.append(path)

    state_metrics = [
        "full_access_far_mean", "deny_frr_mean",
        "restricted_benign_rate_mean", "restricted_malicious_rate_mean",
    ]
    available = [metric for metric in state_metrics if metric in ablation.columns]
    if available:
        state = ablation[ablation["model"].str.contains("v2")].melt(
            id_vars="model", value_vars=available,
            var_name="metric", value_name="value",
        ).dropna(subset=["value"])
        if not state.empty:
            g = sns.catplot(
                data=state, x="value", y="model", col="metric", col_wrap=2,
                kind="bar", sharex=False, height=3.3, aspect=1.2,
            )
            g.figure.suptitle("V2 multi-state access outcomes", y=1.03)
            path = output_dir / "v2_access_state_comparison.png"
            g.figure.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(g.figure)
            paths.append(path)
    return paths


def run_validation_suite(config: dict, project_dir: Path) -> dict[str, Path]:
    suite = config["suite"]
    sim_cfg = SimulationConfig(rounds=int(suite["rounds"]), **config["simulation"])
    model_cfg = ModelConfig(**config["models"])
    bins = int(config["reporting"]["calibration_bins"])
    threshold = float(suite["decision_threshold"])
    output_dir = project_dir / str(suite["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    base_events = _simulate_repeats(
        sim_cfg, int(suite["seed"]), int(suite["repeats"])
    )
    core_scores = run_models(base_events, model_cfg, CORE_MODELS)

    paths: dict[str, Path] = {}
    paths.update(matched_operating_experiment(
        core_scores,
        output_dir,
        int(suite["validation_repeats"]),
        float(suite["target_far"]),
        float(suite["target_frr"]),
        bins,
    ))
    paths.update(pollution_experiment(
        base_events,
        model_cfg,
        sim_cfg,
        output_dir,
        [float(v) for v in suite["pollution_rates"]],
        threshold,
        bins,
        int(suite["seed"]) + 7919,
        float(suite["pollution_low_source_multiplier"]),
    ))
    paths.update(cold_start_experiment(
        sim_cfg,
        model_cfg,
        output_dir,
        [int(v) for v in suite["cold_start_evidence_counts"]],
        int(suite["repeats"]),
        int(suite["seed"]) + 15401,
        threshold,
        bins,
        float(suite["cold_start_current_evidence_signal"]),
        float(suite["cold_start_current_predictive_quality"]),
        int(suite["cold_start_nodes_per_class"]),
    ))
    paths.update(ablation_experiment(
        base_events, model_cfg, output_dir, threshold, bins
    ))
    paths.update(controlled_probation_experiment(
        core_scores, output_dir, threshold, bins
    ))
    for figure in _create_suite_figures(
        output_dir, int(config["reporting"]["figure_dpi"])
    ):
        paths[figure.stem] = figure
    return paths

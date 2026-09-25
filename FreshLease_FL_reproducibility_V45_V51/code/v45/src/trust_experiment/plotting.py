from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_ORDER = [
    "current_only",
    "ema",
    "beta_no_decay_mean",
    "beta_decay_mean",
    "dirichlet_decay_mean",
    "dirichlet_plus_reliability",
    "dirichlet_plus_lcb",
    "ablation_no_reliability",
    "ablation_no_lcb",
    "ablation_no_adaptive",
    "proposed_beta",
    "proposed_dirichlet",
    "v2_no_reliability",
    "v2_no_current_confidence",
    "v2_lcb_as_score",
    "v2_no_probation",
    "proposed_dirichlet_v2",
]


def create_figures(
    rounds_path: Path,
    summary_path: Path,
    output_dir: Path,
    dpi: int = 180,
    threshold_curve_path: Path | None = None,
) -> list[Path]:
    sns.set_theme(style="whitegrid")
    rounds = pd.read_csv(rounds_path)
    summary = pd.read_csv(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    metrics = ["far_mean", "frr_mean", "brier_mean", "ece_mean", "detection_delay_mean"]
    long = summary.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="value")
    g = sns.catplot(
        data=long,
        x="value",
        y="model",
        col="metric",
        col_wrap=2,
        kind="bar",
        sharex=False,
        order=[m for m in MODEL_ORDER if m in summary["model"].unique()],
        height=3.3,
        aspect=1.25,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("value", "model")
    g.figure.suptitle("Phase-1 metric comparison", y=1.02)
    metric_path = output_dir / "metric_comparison.png"
    g.figure.savefig(metric_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.figure)
    paths.append(metric_path)

    selected = rounds[
        (rounds["repeat"] == 0)
        & (rounds["profile"].isin(["betrayal", "on_off", "recovery"]))
        & (rounds["model"].isin([
            "ema", "dirichlet_decay_mean", "proposed_dirichlet",
            "proposed_dirichlet_v2",
        ]))
    ].copy()
    # 每种profile选一个节点，避免同类型重复曲线。
    chosen_nodes = selected.groupby("profile")["node_id"].first().to_dict()
    selected = selected[selected.apply(lambda r: chosen_nodes.get(r["profile"]) == r["node_id"], axis=1)]
    g = sns.relplot(
        data=selected,
        x="round",
        y="trust_score",
        hue="model",
        row="profile",
        kind="line",
        height=2.6,
        aspect=2.5,
        facet_kws={"sharex": True, "sharey": True},
    )
    for ax in g.axes.flat:
        ax.axhline(0.60, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
    g.figure.suptitle("Trust trajectories (repeat 0)", y=1.01)
    trajectory_path = output_dir / "trust_trajectories.png"
    g.figure.savefig(trajectory_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.figure)
    paths.append(trajectory_path)

    if threshold_curve_path is not None and threshold_curve_path.exists():
        curve = pd.read_csv(threshold_curve_path)
        keep = [
            "current_only", "ema", "beta_decay_mean", "dirichlet_decay_mean",
            "proposed_beta", "proposed_dirichlet",
            "proposed_dirichlet_v2",
        ]
        curve = curve[curve["model"].isin(keep)]
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        sns.lineplot(data=curve, x="frr_mean", y="far_mean", hue="model", ax=ax)
        ax.set_xlim(0, 0.20)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("FRR (benign rejection rate)")
        ax.set_ylabel("FAR (malicious acceptance rate)")
        ax.set_title("FAR-FRR trade-off across thresholds")
        tradeoff_path = output_dir / "far_frr_tradeoff.png"
        fig.savefig(tradeoff_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(tradeoff_path)
    return paths

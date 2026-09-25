"""Combine paired attack/control runs for progressive-access parameter screening."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from flower_prototype.analyze_benign_drift_control import analyze_runs as analyze_benign
from flower_prototype.analyze_third_scenario import analyze_runs as analyze_attack


RUN_FIELDS = (
    "config_id", "variant", "noniid_alpha", "repeat",
    "limited_clean_updates", "limited_aggregation_weight",
    "cumulative_risk_decay", "cumulative_risk_threshold", "self_reversal_gate",
    "attack_revoked", "attack_revoked_round", "attack_revocation_reason",
    "attack_detection_delay_rounds", "attack_weight_after_start",
    "attack_updates_after_start", "attack_normal_false_revocations",
    "attack_final_accuracy", "attack_final_backdoor_asr",
    "benign_target_false_revoked", "benign_normal_false_revocations",
    "benign_target_updates_after_drift", "benign_target_weight_after_drift",
    "benign_final_accuracy", "benign_final_backdoor_asr",
)

SUMMARY_FIELDS = (
    "config_id", "variant", "noniid_alpha", "run_records", "independent_seeds",
    "limited_clean_updates", "limited_aggregation_weight",
    "cumulative_risk_decay", "cumulative_risk_threshold", "self_reversal_gate",
    "attack_revocation_rate", "attack_revocation_wilson_ci95_low",
    "attack_revocation_wilson_ci95_high", "attack_detection_delay_rounds_mean",
    "attack_detection_delay_rounds_ci95", "attack_weight_after_start_mean",
    "attack_weight_after_start_ci95", "attack_weight_reduction_vs_full_mean",
    "attack_weight_reduction_vs_full_ci95", "attack_normal_false_revocation_rate",
    "benign_target_false_revocation_rate", "benign_normal_false_revocation_rate",
    "benign_accuracy_mean", "benign_accuracy_ci95",
    "benign_accuracy_gap_vs_full_mean", "benign_accuracy_gap_vs_full_ci95",
    "safety_feasible", "pareto_optimal",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    if not rows:
        raise ValueError("no parameter-tradeoff rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1:
        return 0.0, 0.0
    z = 1.96
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _run_key(row: dict) -> tuple[str, str]:
    return str(row["noniid_alpha"]), str(row["repeat"])


def load_runs(root: Path, manifest: list[dict[str, str]], attack_start: int) -> list[dict]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    metadata: dict[str, dict[str, str]] = {}
    for entry in manifest:
        config_id = entry["config_id"]
        scenario = entry["scenario"]
        child = root / entry["child_dir"]
        nodes = read_csv(child / "real_fl_node_results.csv")
        events = read_csv(child / "real_fl_round_node_events.csv")
        metrics = read_csv(child / "real_fl_round_metrics.csv")
        if scenario == "gradual_drift_betrayal":
            parsed = analyze_attack(nodes, events, metrics, attack_start)
        elif scenario == "benign_concept_drift":
            parsed = analyze_benign(nodes, events, metrics, attack_start)
        else:
            raise ValueError(f"unexpected scenario in manifest: {scenario}")
        grouped[config_id][scenario] = parsed
        metadata[config_id] = entry

    output: list[dict] = []
    for config_id, scenarios in sorted(grouped.items()):
        expected = {"gradual_drift_betrayal", "benign_concept_drift"}
        if set(scenarios) != expected:
            raise ValueError(f"incomplete attack/control pair for {config_id}")
        attack = {_run_key(row): row for row in scenarios["gradual_drift_betrayal"]}
        benign = {_run_key(row): row for row in scenarios["benign_concept_drift"]}
        if set(attack) != set(benign):
            raise ValueError(f"attack/control seeds differ for {config_id}")
        meta = metadata[config_id]
        for key in sorted(attack, key=lambda item: (float(item[0]), int(item[1]))):
            a, b = attack[key], benign[key]
            revoked_round = str(a["revoked_round"])
            output.append({
                "config_id": config_id,
                "variant": meta["variant"],
                "noniid_alpha": key[0],
                "repeat": key[1],
                "limited_clean_updates": meta["limited_clean_updates"],
                "limited_aggregation_weight": meta["limited_aggregation_weight"],
                "cumulative_risk_decay": meta["cumulative_risk_decay"],
                "cumulative_risk_threshold": meta["cumulative_risk_threshold"],
                "self_reversal_gate": meta["self_reversal_gate"],
                "attack_revoked": int(bool(revoked_round)),
                "attack_revoked_round": revoked_round,
                "attack_revocation_reason": a["revocation_reason"],
                "attack_detection_delay_rounds": a["detection_delay_rounds"],
                "attack_weight_after_start": a["attack_aggregated_weight_after_start"],
                "attack_updates_after_start": a["attack_aggregated_after_start"],
                "attack_normal_false_revocations": a["normal_revoked"],
                "attack_final_accuracy": a["final_accuracy"],
                "attack_final_backdoor_asr": a["final_backdoor_asr"],
                "benign_target_false_revoked": b["target_false_revoked"],
                "benign_normal_false_revocations": b["normal_false_revoked"],
                "benign_target_updates_after_drift": b["target_updates_after_drift"],
                "benign_target_weight_after_drift": b["target_weight_after_drift"],
                "benign_final_accuracy": b["final_accuracy"],
                "benign_final_backdoor_asr": b["final_backdoor_asr"],
            })
    return output


def _summary_row(config_id: str, alpha: str, group: list[dict],
                 baseline: dict[tuple[str, str], dict]) -> dict:
    n = len(group)
    revoked = sum(int(row["attack_revoked"]) for row in group)
    low, high = wilson_interval(revoked, n)
    delays = [float(row["attack_detection_delay_rounds"]) for row in group
              if str(row["attack_detection_delay_rounds"]) != ""]
    exposure = [float(row["attack_weight_after_start"]) for row in group]
    benign_accuracy = [float(row["benign_final_accuracy"]) for row in group]
    normal_records = 3 * n
    attack_normal_fp = sum(int(row["attack_normal_false_revocations"]) for row in group)
    benign_target_fp = sum(int(row["benign_target_false_revoked"]) for row in group)
    benign_normal_fp = sum(int(row["benign_normal_false_revocations"]) for row in group)
    exposure_reduction: list[float] = []
    accuracy_gap: list[float] = []
    for row in group:
        base = baseline.get(_run_key(row))
        if base is not None:
            base_exposure = float(base["attack_weight_after_start"])
            exposure_reduction.append(
                1.0 - float(row["attack_weight_after_start"]) / base_exposure
                if base_exposure else 0.0
            )
            accuracy_gap.append(
                float(row["benign_final_accuracy"]) - float(base["benign_final_accuracy"])
            )
    first = group[0]
    feasible = (
        config_id != "baseline_full" and revoked / n >= 0.90
        and attack_normal_fp / normal_records <= 0.05
        and benign_target_fp / n <= 0.05
        and benign_normal_fp / normal_records <= 0.05
    )
    return {
        "config_id": config_id,
        "variant": first["variant"],
        "noniid_alpha": alpha,
        "run_records": n,
        "independent_seeds": len({str(row["repeat"]) for row in group}),
        "limited_clean_updates": first["limited_clean_updates"],
        "limited_aggregation_weight": first["limited_aggregation_weight"],
        "cumulative_risk_decay": first["cumulative_risk_decay"],
        "cumulative_risk_threshold": first["cumulative_risk_threshold"],
        "self_reversal_gate": first["self_reversal_gate"],
        "attack_revocation_rate": revoked / n,
        "attack_revocation_wilson_ci95_low": low,
        "attack_revocation_wilson_ci95_high": high,
        "attack_detection_delay_rounds_mean": statistics.mean(delays) if delays else "",
        "attack_detection_delay_rounds_ci95": ci95(delays) if delays else "",
        "attack_weight_after_start_mean": statistics.mean(exposure),
        "attack_weight_after_start_ci95": ci95(exposure),
        "attack_weight_reduction_vs_full_mean": statistics.mean(exposure_reduction) if exposure_reduction else "",
        "attack_weight_reduction_vs_full_ci95": ci95(exposure_reduction) if exposure_reduction else "",
        "attack_normal_false_revocation_rate": attack_normal_fp / normal_records,
        "benign_target_false_revocation_rate": benign_target_fp / n,
        "benign_normal_false_revocation_rate": benign_normal_fp / normal_records,
        "benign_accuracy_mean": statistics.mean(benign_accuracy),
        "benign_accuracy_ci95": ci95(benign_accuracy),
        "benign_accuracy_gap_vs_full_mean": statistics.mean(accuracy_gap) if accuracy_gap else "",
        "benign_accuracy_gap_vs_full_ci95": ci95(accuracy_gap) if accuracy_gap else "",
        "safety_feasible": int(feasible),
        "pareto_optimal": 0,
    }


def _mark_pareto(rows: list[dict]) -> None:
    by_alpha: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["safety_feasible"]:
            by_alpha[str(row["noniid_alpha"])].append(row)
    for candidates in by_alpha.values():
        for candidate in candidates:
            exposure = float(candidate["attack_weight_after_start_mean"])
            utility = float(candidate["benign_accuracy_gap_vs_full_mean"])
            dominated = any(
                other is not candidate
                and float(other["attack_weight_after_start_mean"]) <= exposure
                and float(other["benign_accuracy_gap_vs_full_mean"]) >= utility
                and (
                    float(other["attack_weight_after_start_mean"]) < exposure
                    or float(other["benign_accuracy_gap_vs_full_mean"]) > utility
                )
                for other in candidates
            )
            candidate["pareto_optimal"] = int(not dominated)


def summarize(rows: list[dict]) -> list[dict]:
    baseline_rows = [row for row in rows if row["config_id"] == "baseline_full"]
    baseline = {_run_key(row): row for row in baseline_rows}
    if not baseline:
        raise ValueError("baseline_full runs are required for paired comparisons")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    grouped_all: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["config_id"]), str(row["noniid_alpha"]))].append(row)
        grouped_all[str(row["config_id"])].append(row)
    per_alpha = [
        _summary_row(config_id, alpha, group, baseline)
        for (config_id, alpha), group in sorted(
            grouped.items(), key=lambda item: (item[0][0], float(item[0][1]))
        )
    ]
    overall = [
        _summary_row(config_id, "ALL", group, baseline)
        for config_id, group in sorted(grouped_all.items())
    ]
    feasibility_by_config: dict[str, list[int]] = defaultdict(list)
    for row in per_alpha:
        feasibility_by_config[str(row["config_id"])].append(int(row["safety_feasible"]))
    for row in overall:
        config_id = str(row["config_id"])
        row["safety_feasible"] = int(
            config_id != "baseline_full"
            and bool(feasibility_by_config[config_id])
            and all(feasibility_by_config[config_id])
        )
    output = [*per_alpha, *overall]
    _mark_pareto(output)
    return output


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: analyze_parameter_tradeoff ROOT MANIFEST RUN_OUT SUMMARY_OUT"
        )
    root = Path(sys.argv[1])
    manifest = read_csv(Path(sys.argv[2]))
    if not manifest:
        raise ValueError("empty parameter-tradeoff manifest")
    attack_start = int(manifest[0]["attack_start_round"])
    rows = load_runs(root, manifest, attack_start)
    write_csv(Path(sys.argv[3]), rows, RUN_FIELDS)
    write_csv(Path(sys.argv[4]), summarize(rows), SUMMARY_FIELDS)


if __name__ == "__main__":
    main()

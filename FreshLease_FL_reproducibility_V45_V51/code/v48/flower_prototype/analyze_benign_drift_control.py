"""Analyze false revocation under a clean, time-varying client distribution."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


KEYS = ("variant", "attack_scenario", "noniid_alpha", "repeat")
TARGET_NODE = "benign-concept-drift"
EXPECTED_VARIANTS = {
    "progressive_full", "progressive_no_cumulative",
    "full", "no_online_revalidation",
}
RUN_FIELDS = (
    *KEYS, "drift_start_round", "initial_access_state",
    "initial_training_eligible", "target_false_revoked", "revoked_round",
    "revocation_reason", "normal_false_revoked", "target_updates_after_drift",
    "target_weight_after_drift", "final_accuracy", "final_backdoor_asr",
)
SUMMARY_FIELDS = (
    "variant", "noniid_alpha", "independent_runs",
    "target_false_revocations", "target_false_revocation_rate",
    "target_false_revocation_wilson_ci95_half_width",
    "normal_node_records", "normal_false_revocations",
    "normal_false_revocation_rate",
    "normal_false_revocation_wilson_ci95_half_width",
    "target_updates_after_drift_mean", "target_updates_after_drift_ci95",
    "target_weight_after_drift_mean", "target_weight_after_drift_ci95",
    "final_accuracy_mean", "final_accuracy_ci95",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    if not rows:
        raise ValueError("no benign-drift runs to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in KEYS)


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def wilson_half_width(successes: int, trials: int) -> float:
    if trials < 1:
        return 0.0
    z = 1.96
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    return z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    ) / denominator


def analyze_runs(nodes: list[dict[str, str]], events: list[dict[str, str]],
                 metrics: list[dict[str, str]], drift_start_round: int) -> list[dict]:
    grouped_nodes: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    grouped_events: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    grouped_metrics: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in nodes:
        grouped_nodes[key(row)].append(row)
    for row in events:
        if row["aggregated"] == "1" and row["returned"] != "1":
            raise ValueError(f"aggregation without returned update: {key(row)}")
        grouped_events[key(row)].append(row)
    for row in metrics:
        grouped_metrics[key(row)].append(row)
    if set(grouped_nodes) != set(grouped_events) or set(grouped_nodes) != set(grouped_metrics):
        raise ValueError("node, event and metric run keys differ")

    output = []
    for run_key, group in sorted(grouped_nodes.items()):
        variant, scenario, _, _ = run_key
        if variant not in EXPECTED_VARIANTS or scenario != "benign_concept_drift":
            raise ValueError(f"unexpected benign-drift configuration: {run_key}")
        target = [row for row in group if row["node_id"] == TARGET_NODE]
        normal = [row for row in group if row["profile"] == "honest"]
        if len(group) != 4 or len(target) != 1 or len(normal) != 3:
            raise ValueError(f"expected one drift target and three honest nodes: {run_key}")
        target = target[0]
        run_events = grouped_events[run_key]
        target_events = [row for row in run_events if row["node_id"] == TARGET_NODE]
        post_drift = [
            row for row in target_events
            if int(row["round"]) >= drift_start_round and row["aggregated"] == "1"
        ]
        final = max(grouped_metrics[run_key], key=lambda row: int(row["round"]))
        initial_state = target["initial_access_state"]
        output.append(dict(zip(KEYS, run_key), **{
            "drift_start_round": drift_start_round,
            "initial_access_state": initial_state,
            "initial_training_eligible": int(initial_state in {"LIMITED", "ADMITTED"}),
            "target_false_revoked": int(bool(target.get("revoked_round", "").strip())),
            "revoked_round": target.get("revoked_round", ""),
            "revocation_reason": target.get("revocation_reason", ""),
            "normal_false_revoked": sum(
                bool(row.get("revoked_round", "").strip()) for row in normal
            ),
            "target_updates_after_drift": len(post_drift),
            "target_weight_after_drift": sum(
                float(row.get("aggregation_weight") or 1.0) for row in post_drift
            ),
            "final_accuracy": final["accuracy"],
            "final_backdoor_asr": final["backdoor_asr"],
        }))
    return output


def summarize(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in runs:
        grouped[(row["variant"], row["noniid_alpha"])].append(row)
    output = []
    for (variant, alpha), group in sorted(grouped.items()):
        target_revocations = sum(int(row["target_false_revoked"]) for row in group)
        normal_records = 3 * len(group)
        normal_revocations = sum(int(row["normal_false_revoked"]) for row in group)
        updates = [float(row["target_updates_after_drift"]) for row in group]
        weights = [float(row["target_weight_after_drift"]) for row in group]
        accuracies = [float(row["final_accuracy"]) for row in group]
        output.append({
            "variant": variant,
            "noniid_alpha": alpha,
            "independent_runs": len(group),
            "target_false_revocations": target_revocations,
            "target_false_revocation_rate": target_revocations / len(group),
            "target_false_revocation_wilson_ci95_half_width": wilson_half_width(
                target_revocations, len(group)
            ),
            "normal_node_records": normal_records,
            "normal_false_revocations": normal_revocations,
            "normal_false_revocation_rate": normal_revocations / normal_records,
            "normal_false_revocation_wilson_ci95_half_width": wilson_half_width(
                normal_revocations, normal_records
            ),
            "target_updates_after_drift_mean": statistics.mean(updates),
            "target_updates_after_drift_ci95": ci95(updates),
            "target_weight_after_drift_mean": statistics.mean(weights),
            "target_weight_after_drift_ci95": ci95(weights),
            "final_accuracy_mean": statistics.mean(accuracies),
            "final_accuracy_ci95": ci95(accuracies),
        })
    return output


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit(
            "usage: analyze_benign_drift_control NODES EVENTS METRICS "
            "RUN_OUT SUMMARY_OUT DRIFT_START_ROUND"
        )
    runs = analyze_runs(
        *(read_csv(Path(arg)) for arg in sys.argv[1:4]), int(sys.argv[6])
    )
    write_csv(Path(sys.argv[4]), runs, RUN_FIELDS)
    write_csv(Path(sys.argv[5]), summarize(runs), SUMMARY_FIELDS)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import t as student_t


IDENTITY = (
    "dataset", "client_count", "attacker_count", "cohort_mode", "variant",
    "trust_estimator", "attack_scenario", "noniid_alpha", "repeat",
)
SUMMARY_KEY = IDENTITY[:-1]


def read(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def number(row: dict, name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    return default if value in {None, ""} else float(value)


def mean_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, math.nan
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    ci = float(student_t.ppf(0.975, len(values) - 1)) * math.sqrt(variance / len(values))
    return mean, ci


def write(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_runs(nodes: list[dict], events: list[dict], metrics: list[dict], configs: list[dict], stage: str) -> list[dict]:
    node_groups: dict[tuple, list[dict]] = defaultdict(list)
    event_groups: dict[tuple, list[dict]] = defaultdict(list)
    metric_groups: dict[tuple, list[dict]] = defaultdict(list)
    config_by_key = {}
    for row in nodes:
        node_groups[tuple(row.get(field, "") for field in IDENTITY)].append(row)
    for row in events:
        event_groups[tuple(row.get(field, "") for field in IDENTITY)].append(row)
    for row in metrics:
        metric_groups[tuple(row.get(field, "") for field in IDENTITY)].append(row)
    for row in configs:
        config_by_key[tuple(row.get(field, "") for field in IDENTITY)] = row

    results = []
    for key, group in sorted(node_groups.items()):
        targets = [row for row in group if row.get("profile") != "honest"]
        normal = [row for row in group if row.get("profile") == "honest"]
        expected_targets = int(float(key[2]))
        if len(targets) != expected_targets or not normal:
            raise ValueError(f"invalid target/normal membership for {key}: {len(targets)}/{len(normal)}")
        target_ids = {row["node_id"] for row in targets}
        run_events = event_groups[key]
        start = int(float(group[0].get("attack_start_round", 1)))
        post = [row for row in run_events if int(float(row.get("round", 0))) >= start]
        target_events = [row for row in post if row.get("node_id") in target_ids]
        normal_events = [row for row in post if row.get("node_id") not in target_ids]
        target_effective = sum(number(row, "aggregation_effective_weight") for row in target_events)
        total_effective = target_effective + sum(
            number(row, "aggregation_effective_weight") for row in normal_events
        )
        per_round_processing: dict[int, float] = {}
        for row in run_events:
            round_number = int(float(row.get("round", 0)))
            per_round_processing[round_number] = max(
                per_round_processing.get(round_number, 0.0),
                number(row, "server_round_processing_seconds"),
            )
        final = max(metric_groups[key], key=lambda row: int(float(row["round"])))
        configuration = config_by_key.get(key)
        if configuration is None:
            raise ValueError(f"missing configuration metrics for {key}")
        result = {field: value for field, value in zip(IDENTITY, key)}
        result.update({
            "stage": stage,
            "attacker_fraction": expected_targets / int(float(key[1])),
            "normal_nodes": len(normal),
            "target_initial_limited_rate": sum(
                row.get("initial_access_state") == "LIMITED" for row in targets
            ) / len(targets),
            "target_initial_admitted_rate": sum(
                row.get("initial_access_state") == "ADMITTED" for row in targets
            ) / len(targets),
            "normal_initial_admitted_rate": sum(
                row.get("initial_access_state") == "ADMITTED" for row in normal
            ) / len(normal),
            "target_onboarding_seconds_mean": sum(
                number(row, "onboarding_duration_seconds") for row in targets
            ) / len(targets),
            "normal_onboarding_seconds_mean": sum(
                number(row, "onboarding_duration_seconds") for row in normal
            ) / len(normal),
            "target_returned_updates_after_start": sum(
                int(number(row, "returned")) for row in target_events
            ),
            "target_aggregated_updates_after_start": sum(
                int(number(row, "aggregated")) for row in target_events
            ),
            "target_effective_aggregation_share_after_start": (
                target_effective / total_effective if total_effective else 0.0
            ),
            "target_rffl_removals": sum(
                int(number(row, "aggregation_reputation_removed")) for row in target_events
            ),
            "normal_rffl_removals": sum(
                int(number(row, "aggregation_reputation_removed")) for row in normal_events
            ),
            "update_payload_bytes": sum(
                int(number(row, "update_payload_bytes")) for row in run_events
            ),
            "server_processing_seconds": sum(per_round_processing.values()),
            "configuration_runtime_seconds": number(configuration, "configuration_runtime_seconds"),
            "controller_state_bytes": int(number(configuration, "controller_state_bytes")),
            "final_accuracy": number(final, "accuracy"),
            "final_backdoor_asr": number(final, "backdoor_asr"),
        })
        results.append(result)
    return results


def summarize(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in runs:
        groups[tuple(row[field] for field in SUMMARY_KEY)].append(row)
    measures = [
        "target_initial_limited_rate", "target_initial_admitted_rate",
        "normal_initial_admitted_rate", "target_onboarding_seconds_mean",
        "normal_onboarding_seconds_mean", "target_returned_updates_after_start",
        "target_aggregated_updates_after_start",
        "target_effective_aggregation_share_after_start", "target_rffl_removals",
        "normal_rffl_removals", "update_payload_bytes", "server_processing_seconds",
        "configuration_runtime_seconds", "controller_state_bytes", "final_accuracy",
        "final_backdoor_asr",
    ]
    output = []
    for key, group in sorted(groups.items()):
        row = {field: value for field, value in zip(SUMMARY_KEY, key)}
        row["independent_runs"] = len(group)
        row["attacker_fraction"] = group[0]["attacker_fraction"]
        for measure in measures:
            mean, ci = mean_ci([float(item[measure]) for item in group])
            row[f"{measure}_mean"] = mean
            row[f"{measure}_ci95"] = ci
        output.append(row)
    return output


def paired_effects(runs: list[dict]) -> list[dict]:
    pair_key = (
        "dataset", "client_count", "attacker_count", "attack_scenario",
        "noniid_alpha", "repeat",
    )
    controls = {
        tuple(row[field] for field in pair_key): row
        for row in runs if row["variant"] == "access_no_limited"
    }
    output = []
    for row in runs:
        if row["variant"] == "access_no_limited":
            continue
        key = tuple(row[field] for field in pair_key)
        control = controls.get(key)
        if control is None:
            raise ValueError(f"missing direct-admission control for {key}")
        output.append({
            **{field: row[field] for field in pair_key},
            "method_variant": row["variant"],
            "method_target_share": row["target_effective_aggregation_share_after_start"],
            "control_target_share": control["target_effective_aggregation_share_after_start"],
            "target_share_reduction": (
                float(control["target_effective_aggregation_share_after_start"])
                - float(row["target_effective_aggregation_share_after_start"])
            ),
            "accuracy_difference": float(row["final_accuracy"]) - float(control["final_accuracy"]),
            "backdoor_asr_reduction": (
                float(control["final_backdoor_asr"]) - float(row["final_backdoor_asr"])
            ),
            "runtime_overhead_seconds": (
                float(row["configuration_runtime_seconds"])
                - float(control["configuration_runtime_seconds"])
            ),
            "server_processing_overhead_seconds": (
                float(row["server_processing_seconds"])
                - float(control["server_processing_seconds"])
            ),
            "controller_state_overhead_bytes": (
                int(row["controller_state_bytes"]) - int(control["controller_state_bytes"])
            ),
            "method_target_rffl_removals": row["target_rffl_removals"],
            "method_normal_rffl_removals": row["normal_rffl_removals"],
        })
    return output


def main() -> None:
    if len(sys.argv) != 9:
        raise SystemExit(
            "usage: analyze_q3_v43 STAGE nodes.csv events.csv metrics.csv "
            "configuration.csv runs.csv summary.csv paired.csv"
        )
    _, stage, node_path, event_path, metric_path, config_path, run_path, summary_path, pair_path = sys.argv
    runs = build_runs(read(node_path), read(event_path), read(metric_path), read(config_path), stage)
    write(run_path, runs)
    write(summary_path, summarize(runs))
    write(pair_path, paired_effects(runs))


if __name__ == "__main__":
    main()

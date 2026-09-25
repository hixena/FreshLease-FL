from __future__ import annotations

import math
import sys
from collections import defaultdict

from flower_prototype.analyze_q3_v43 import (
    IDENTITY,
    SUMMARY_KEY,
    build_runs,
    mean_ci,
    number,
    read,
    write,
)


EARLY_END_ROUND = 4
LATE_START_ROUND = 5
ASR_THRESHOLDS = (0.10, 0.20, 0.30)


def key_for(row: dict) -> tuple:
    return tuple(row.get(field, "") for field in IDENTITY)


def finite_values(rows: list[dict], field: str) -> list[float]:
    values = [float(row[field]) for row in rows]
    return [value for value in values if math.isfinite(value)]


def effective_share(events: list[dict], target_ids: set[str]) -> float:
    target = sum(
        number(row, "aggregation_effective_weight")
        for row in events if row.get("node_id") in target_ids
    )
    total = sum(number(row, "aggregation_effective_weight") for row in events)
    return target / total if total else 0.0


def normalized_auc(points: list[tuple[int, float]]) -> float:
    if not points:
        return math.nan
    points = sorted(points)
    if len(points) == 1:
        return points[0][1]
    area = sum(
        (right_round - left_round) * (left_value + right_value) / 2.0
        for (left_round, left_value), (right_round, right_value)
        in zip(points, points[1:])
    )
    duration = points[-1][0] - points[0][0]
    return area / duration if duration else points[-1][1]


def build_round_runs(nodes: list[dict], events: list[dict], metrics: list[dict]) -> list[dict]:
    node_groups: dict[tuple, list[dict]] = defaultdict(list)
    event_groups: dict[tuple, list[dict]] = defaultdict(list)
    metric_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in nodes:
        node_groups[key_for(row)].append(row)
    for row in events:
        event_groups[key_for(row)].append(row)
    for row in metrics:
        metric_groups[key_for(row)].append(row)

    output = []
    for key, group in sorted(node_groups.items()):
        target_ids = {
            row["node_id"] for row in group if row.get("profile") != "honest"
        }
        expected_targets = int(float(key[2]))
        events_by_round: dict[int, list[dict]] = defaultdict(list)
        for row in event_groups[key]:
            events_by_round[int(float(row["round"]))].append(row)
        for metric in sorted(metric_groups[key], key=lambda row: int(float(row["round"]))):
            round_number = int(float(metric["round"]))
            round_events = events_by_round.get(round_number, [])
            target_events = [
                row for row in round_events if row.get("node_id") in target_ids
            ]
            result = {field: value for field, value in zip(IDENTITY, key)}
            result.update({
                "round": round_number,
                "accuracy": number(metric, "accuracy", math.nan),
                "backdoor_asr": number(metric, "backdoor_asr", math.nan),
                "target_effective_aggregation_share": effective_share(
                    round_events, target_ids
                ),
                "target_returned_updates": sum(
                    int(number(row, "returned")) for row in target_events
                ),
                "target_aggregated_updates": sum(
                    int(number(row, "aggregated")) for row in target_events
                ),
                "target_observed_node_rate": (
                    len({row.get("node_id") for row in target_events}) / expected_targets
                ),
                "target_limited_rate": sum(
                    row.get("access_state_after") == "LIMITED"
                    for row in target_events
                ) / expected_targets,
                "target_admitted_rate": sum(
                    row.get("access_state_after") == "ADMITTED"
                    for row in target_events
                ) / expected_targets,
                "target_rffl_removals": sum(
                    int(number(row, "aggregation_reputation_removed"))
                    for row in target_events
                ),
            })
            output.append(result)
    return output


def build_long_runs(
    nodes: list[dict], events: list[dict], metrics: list[dict],
    configs: list[dict], stage: str,
) -> list[dict]:
    base_runs = build_runs(nodes, events, metrics, configs, stage)
    nodes_by_key: dict[tuple, list[dict]] = defaultdict(list)
    events_by_key: dict[tuple, list[dict]] = defaultdict(list)
    metrics_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for row in nodes:
        nodes_by_key[key_for(row)].append(row)
    for row in events:
        events_by_key[key_for(row)].append(row)
    for row in metrics:
        metrics_by_key[key_for(row)].append(row)

    output = []
    for run in base_runs:
        key = tuple(run[field] for field in IDENTITY)
        target_rows = [
            row for row in nodes_by_key[key] if row.get("profile") != "honest"
        ]
        target_ids = {row["node_id"] for row in target_rows}
        start = int(float(target_rows[0].get("attack_start_round", 1)))
        run_events = events_by_key[key]
        early_events = [
            row for row in run_events
            if start <= int(float(row["round"])) <= EARLY_END_ROUND
        ]
        late_events = [
            row for row in run_events
            if int(float(row["round"])) >= LATE_START_ROUND
        ]
        promotion_rounds = []
        for target in target_rows:
            if target.get("initial_access_state") == "ADMITTED":
                promotion_rounds.append(1.0)
                continue
            admitted = [
                int(float(row["round"])) for row in run_events
                if row.get("node_id") == target["node_id"]
                and row.get("access_state_after") == "ADMITTED"
            ]
            if admitted:
                promotion_rounds.append(float(min(admitted)))

        metric_points = sorted(
            (
                int(float(row["round"])),
                number(row, "backdoor_asr", math.nan),
            )
            for row in metrics_by_key[key]
            if int(float(row["round"])) >= start
        )
        metric_by_round = dict(metric_points)
        final_round = metric_points[-1][0]
        asr_at_round_8 = metric_by_round.get(8, math.nan)
        final_asr = metric_points[-1][1]
        run.update({
            "total_rounds": final_round,
            "target_full_access_round_mean": (
                sum(promotion_rounds) / len(promotion_rounds)
                if promotion_rounds else math.nan
            ),
            "target_promoted_rate": len(promotion_rounds) / len(target_rows),
            "target_effective_aggregation_share_early": effective_share(
                early_events, target_ids
            ),
            "target_effective_aggregation_share_late": effective_share(
                late_events, target_ids
            ),
            "backdoor_asr_auc_after_start": normalized_auc(metric_points),
            "backdoor_asr_at_round_8": asr_at_round_8,
            "backdoor_asr_growth_round_8_to_final": (
                final_asr - asr_at_round_8
                if math.isfinite(asr_at_round_8) else math.nan
            ),
        })
        for threshold in ASR_THRESHOLDS:
            label = int(round(threshold * 100))
            reached = [round_number for round_number, value in metric_points if value >= threshold]
            run[f"first_round_asr_ge_{label}pct"] = (
                float(min(reached)) if reached else math.nan
            )
        output.append(run)
    return output


def summarize_long(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in runs:
        groups[tuple(row[field] for field in SUMMARY_KEY)].append(row)
    measures = [
        "target_full_access_round_mean", "target_promoted_rate",
        "target_effective_aggregation_share_early",
        "target_effective_aggregation_share_late",
        "backdoor_asr_auc_after_start", "backdoor_asr_at_round_8",
        "backdoor_asr_growth_round_8_to_final", "final_accuracy",
        "final_backdoor_asr", "configuration_runtime_seconds",
        "server_processing_seconds", "controller_state_bytes",
    ]
    output = []
    for key, group in sorted(groups.items()):
        row = {field: value for field, value in zip(SUMMARY_KEY, key)}
        row["independent_runs"] = len(group)
        row["total_rounds"] = max(int(item["total_rounds"]) for item in group)
        for measure in measures:
            mean, ci = mean_ci(finite_values(group, measure))
            row[f"{measure}_mean"] = mean
            row[f"{measure}_ci95"] = ci
        for threshold in ASR_THRESHOLDS:
            label = int(round(threshold * 100))
            field = f"first_round_asr_ge_{label}pct"
            observed = finite_values(group, field)
            mean, ci = mean_ci(observed)
            row[f"asr_ge_{label}pct_rate"] = len(observed) / len(group)
            row[f"{field}_conditional_mean"] = mean
            row[f"{field}_conditional_ci95"] = ci
        output.append(row)
    return output


def paired_effects_long(runs: list[dict]) -> list[dict]:
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
        result = {
            **{field: row[field] for field in pair_key},
            "method_variant": row["variant"],
            "early_target_share_reduction": (
                float(control["target_effective_aggregation_share_early"])
                - float(row["target_effective_aggregation_share_early"])
            ),
            "late_target_share_reduction": (
                float(control["target_effective_aggregation_share_late"])
                - float(row["target_effective_aggregation_share_late"])
            ),
            "asr_auc_reduction": (
                float(control["backdoor_asr_auc_after_start"])
                - float(row["backdoor_asr_auc_after_start"])
            ),
            "round_8_asr_reduction": (
                float(control["backdoor_asr_at_round_8"])
                - float(row["backdoor_asr_at_round_8"])
            ),
            "final_backdoor_asr_reduction": (
                float(control["final_backdoor_asr"])
                - float(row["final_backdoor_asr"])
            ),
            "asr_growth_reduction": (
                float(control["backdoor_asr_growth_round_8_to_final"])
                - float(row["backdoor_asr_growth_round_8_to_final"])
            ),
            "final_accuracy_difference": (
                float(row["final_accuracy"]) - float(control["final_accuracy"])
            ),
            "runtime_overhead_seconds": (
                float(row["configuration_runtime_seconds"])
                - float(control["configuration_runtime_seconds"])
            ),
        }
        for threshold in ASR_THRESHOLDS:
            label = int(round(threshold * 100))
            field = f"first_round_asr_ge_{label}pct"
            method_round = float(row[field])
            control_round = float(control[field])
            result[f"asr_ge_{label}pct_delay"] = (
                method_round - control_round
                if math.isfinite(method_round) and math.isfinite(control_round)
                else math.nan
            )
        output.append(result)
    return output


def summarize_rounds(round_runs: list[dict]) -> list[dict]:
    key_fields = SUMMARY_KEY + ("round",)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in round_runs:
        groups[tuple(row[field] for field in key_fields)].append(row)
    measures = [
        "accuracy", "backdoor_asr", "target_effective_aggregation_share",
        "target_returned_updates", "target_aggregated_updates",
        "target_observed_node_rate", "target_limited_rate",
        "target_admitted_rate", "target_rffl_removals",
    ]
    output = []
    for key, group in sorted(groups.items()):
        row = {field: value for field, value in zip(key_fields, key)}
        row["independent_runs"] = len(group)
        for measure in measures:
            mean, ci = mean_ci(finite_values(group, measure))
            row[f"{measure}_mean"] = mean
            row[f"{measure}_ci95"] = ci
        output.append(row)
    return output


def main() -> None:
    if len(sys.argv) != 11:
        raise SystemExit(
            "usage: analyze_long_horizon_v44 STAGE nodes.csv events.csv metrics.csv "
            "configuration.csv runs.csv summary.csv paired.csv round_runs.csv "
            "round_summary.csv"
        )
    (
        _, stage, node_path, event_path, metric_path, config_path,
        run_path, summary_path, pair_path, round_run_path, round_summary_path,
    ) = sys.argv
    nodes = read(node_path)
    events = read(event_path)
    metrics = read(metric_path)
    configs = read(config_path)
    runs = build_long_runs(nodes, events, metrics, configs, stage)
    round_runs = build_round_runs(nodes, events, metrics)
    write(run_path, runs)
    write(summary_path, summarize_long(runs))
    write(pair_path, paired_effects_long(runs))
    write(round_run_path, round_runs)
    write(round_summary_path, summarize_rounds(round_runs))


if __name__ == "__main__":
    main()

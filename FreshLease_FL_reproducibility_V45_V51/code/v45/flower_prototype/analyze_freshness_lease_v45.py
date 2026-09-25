from __future__ import annotations

import math
import sys
from collections import defaultdict

from flower_prototype.analyze_long_horizon_v44 import (
    ASR_THRESHOLDS,
    build_long_runs,
    build_round_runs,
    effective_share,
    finite_values,
    key_for,
    summarize_long,
    summarize_rounds,
)
from flower_prototype.analyze_q3_v43 import (
    IDENTITY,
    SUMMARY_KEY,
    mean_ci,
    number,
    read,
    write,
)


SUSTAINED_START_ROUND = 10


def enrich_runs(
    runs: list[dict], nodes: list[dict], events: list[dict],
) -> list[dict]:
    nodes_by_key: dict[tuple, list[dict]] = defaultdict(list)
    events_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for row in nodes:
        nodes_by_key[key_for(row)].append(row)
    for row in events:
        events_by_key[key_for(row)].append(row)

    for run in runs:
        key = tuple(run[field] for field in IDENTITY)
        target_ids = {
            row["node_id"] for row in nodes_by_key[key]
            if row.get("profile") != "honest"
        }
        run_events = events_by_key[key]
        start = int(float(nodes_by_key[key][0].get("attack_start_round", 1)))
        post = [row for row in run_events if int(float(row["round"])) >= start]
        sustained = [
            row for row in run_events
            if int(float(row["round"])) >= SUSTAINED_START_ROUND
        ]
        target_post = [row for row in post if row.get("node_id") in target_ids]
        target_returned = [row for row in target_post if int(number(row, "returned"))]
        target_aggregated = [
            row for row in target_post if int(number(row, "aggregated"))
        ]
        normal_post = [row for row in post if row.get("node_id") not in target_ids]
        run.update({
            "target_effective_aggregation_share_sustained": effective_share(
                sustained, target_ids
            ),
            "target_limited_returned_update_rate_after_start": (
                sum(row.get("access_state_after") == "LIMITED" for row in target_returned)
                / len(target_returned) if target_returned else 0.0
            ),
            "target_full_weight_aggregated_update_rate_after_start": (
                sum(number(row, "aggregation_weight") >= 0.999999
                    for row in target_aggregated) / len(target_aggregated)
                if target_aggregated else 0.0
            ),
            "target_lease_expirations": sum(
                row.get("access_transition") == "LEASE_EXPIRED"
                for row in target_post
            ),
            "target_lease_renewals": sum(
                row.get("access_transition") == "RENEWED" for row in target_post
            ),
            "normal_lease_expirations": sum(
                row.get("access_transition") == "LEASE_EXPIRED"
                for row in normal_post
            ),
        })
    return runs


def enrich_round_runs(
    round_runs: list[dict], nodes: list[dict], events: list[dict],
) -> list[dict]:
    events_by_key_round: dict[tuple, list[dict]] = defaultdict(list)
    target_ids_by_key: dict[tuple, set[str]] = defaultdict(set)
    for row in nodes:
        key = key_for(row)
        if row.get("profile") != "honest":
            target_ids_by_key[key].add(row.get("node_id", ""))
    for row in events:
        key = key_for(row)
        events_by_key_round[(key, int(float(row["round"])))].append(row)
    for row in round_runs:
        key = tuple(row[field] for field in IDENTITY)
        group = events_by_key_round[(key, int(row["round"]))]
        targets = target_ids_by_key[key]
        target_rows = [item for item in group if item.get("node_id") in targets]
        row["target_lease_expirations"] = sum(
            item.get("access_transition") == "LEASE_EXPIRED"
            for item in target_rows
        )
        row["target_lease_renewals"] = sum(
            item.get("access_transition") == "RENEWED" for item in target_rows
        )
    return round_runs


def summarize_v45(runs: list[dict]) -> list[dict]:
    summary = summarize_long(runs)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in runs:
        groups[tuple(row[field] for field in SUMMARY_KEY)].append(row)
    extra = (
        "target_effective_aggregation_share_sustained",
        "target_limited_returned_update_rate_after_start",
        "target_full_weight_aggregated_update_rate_after_start",
        "target_lease_expirations", "target_lease_renewals",
        "normal_lease_expirations",
    )
    summary_by_key = {
        tuple(row[field] for field in SUMMARY_KEY): row for row in summary
    }
    for key, group in groups.items():
        row = summary_by_key[key]
        for measure in extra:
            mean, ci = mean_ci(finite_values(group, measure))
            row[f"{measure}_mean"] = mean
            row[f"{measure}_ci95"] = ci
    return summary


def paired_effects_v45(runs: list[dict]) -> list[dict]:
    pair_fields = (
        "dataset", "client_count", "attacker_count", "attack_scenario",
        "noniid_alpha", "repeat",
    )
    by_variant_key = {
        (row["variant"],) + tuple(row[field] for field in pair_fields): row
        for row in runs
    }
    comparisons = (
        ("access_freshness_lease", "access_full"),
        ("access_freshness_lease", "access_no_limited"),
        ("access_full", "access_no_limited"),
    )
    output = []
    all_keys = sorted({tuple(row[field] for field in pair_fields) for row in runs})
    for method_variant, control_variant in comparisons:
        for key in all_keys:
            method = by_variant_key.get((method_variant,) + key)
            control = by_variant_key.get((control_variant,) + key)
            if method is None or control is None:
                raise ValueError(
                    f"missing paired run {method_variant}/{control_variant} for {key}"
                )
            result = {field: value for field, value in zip(pair_fields, key)}
            result.update({
                "method_variant": method_variant,
                "control_variant": control_variant,
                "sustained_target_share_reduction": (
                    float(control["target_effective_aggregation_share_sustained"])
                    - float(method["target_effective_aggregation_share_sustained"])
                ),
                "asr_auc_reduction": (
                    float(control["backdoor_asr_auc_after_start"])
                    - float(method["backdoor_asr_auc_after_start"])
                ),
                "round_8_asr_reduction": (
                    float(control["backdoor_asr_at_round_8"])
                    - float(method["backdoor_asr_at_round_8"])
                ),
                "final_backdoor_asr_reduction": (
                    float(control["final_backdoor_asr"])
                    - float(method["final_backdoor_asr"])
                ),
                "final_accuracy_difference": (
                    float(method["final_accuracy"])
                    - float(control["final_accuracy"])
                ),
                "runtime_overhead_seconds": (
                    float(method["configuration_runtime_seconds"])
                    - float(control["configuration_runtime_seconds"])
                ),
            })
            for threshold in ASR_THRESHOLDS:
                label = int(round(threshold * 100))
                field = f"first_round_asr_ge_{label}pct"
                method_round = float(method[field])
                control_round = float(control[field])
                result[f"asr_ge_{label}pct_delay"] = (
                    method_round - control_round
                    if math.isfinite(method_round) and math.isfinite(control_round)
                    else math.nan
                )
            output.append(result)
    return output


def summarize_rounds_v45(round_runs: list[dict]) -> list[dict]:
    summary = summarize_rounds(round_runs)
    key_fields = SUMMARY_KEY + ("round",)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in round_runs:
        groups[tuple(row[field] for field in key_fields)].append(row)
    summary_by_key = {
        tuple(row[field] for field in key_fields): row for row in summary
    }
    for key, group in groups.items():
        row = summary_by_key[key]
        for measure in ("target_lease_expirations", "target_lease_renewals"):
            mean, ci = mean_ci(finite_values(group, measure))
            row[f"{measure}_mean"] = mean
            row[f"{measure}_ci95"] = ci
    return summary


def main() -> None:
    if len(sys.argv) != 11:
        raise SystemExit(
            "usage: analyze_freshness_lease_v45 STAGE nodes.csv events.csv "
            "metrics.csv configuration.csv runs.csv summary.csv paired.csv "
            "round_runs.csv round_summary.csv"
        )
    (
        _, stage, node_path, event_path, metric_path, config_path,
        run_path, summary_path, pair_path, round_run_path, round_summary_path,
    ) = sys.argv
    nodes = read(node_path)
    events = read(event_path)
    metrics = read(metric_path)
    configs = read(config_path)
    runs = enrich_runs(
        build_long_runs(nodes, events, metrics, configs, stage), nodes, events
    )
    round_runs = enrich_round_runs(
        build_round_runs(nodes, events, metrics), nodes, events
    )
    write(run_path, runs)
    write(summary_path, summarize_v45(runs))
    write(pair_path, paired_effects_v45(runs))
    write(round_run_path, round_runs)
    write(round_summary_path, summarize_rounds_v45(round_runs))


if __name__ == "__main__":
    main()

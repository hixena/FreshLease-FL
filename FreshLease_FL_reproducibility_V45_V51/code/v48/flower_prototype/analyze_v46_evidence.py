from __future__ import annotations

import math
import sys
from collections import defaultdict

from scipy.stats import ttest_1samp

from flower_prototype.analyze_freshness_lease_v45 import (
    enrich_round_runs,
    enrich_runs,
    summarize_rounds_v45,
    summarize_v45,
)
from flower_prototype.analyze_long_horizon_v44 import (
    ASR_THRESHOLDS,
    build_long_runs,
    build_round_runs,
)
from flower_prototype.analyze_q3_v43 import mean_ci, read, write


PAIR_FIELDS = (
    "dataset", "client_count", "attacker_count", "cohort_mode",
    "trust_estimator", "attack_scenario", "noniid_alpha", "repeat",
)
PAIR_SUMMARY_FIELDS = PAIR_FIELDS[:-1] + (
    "method_variant", "control_variant",
)
EFFECT_FIELDS = (
    "sustained_target_share_reduction",
    "limited_returned_update_rate_increase",
    "full_weight_aggregated_update_rate_reduction",
    "asr_auc_reduction", "round_8_asr_reduction",
    "final_backdoor_asr_reduction", "final_accuracy_difference",
    "runtime_overhead_seconds", "asr_ge_10pct_delay",
    "asr_ge_20pct_delay", "asr_ge_30pct_delay",
)


def lease_value(row: dict) -> int:
    value = row.get("configured_access_lease_full_updates", "")
    if value in {None, ""}:
        value = row.get("access_lease_full_updates", 0)
    return int(float(value or 0))


def sensitivity_variant(row: dict) -> str:
    variant = row.get("variant", "")
    if variant == "access_freshness_lease":
        return f"access_freshness_lease_L{lease_value(row)}"
    return variant


def normalize_sensitivity(rows: list[dict]) -> list[dict]:
    output = []
    for source in rows:
        row = dict(source)
        row["variant"] = sensitivity_variant(row)
        output.append(row)
    return output


def annotate_lease_setting(rows: list[dict]) -> list[dict]:
    for row in rows:
        variant = row.get("variant", "")
        if variant.startswith("access_freshness_lease_L"):
            lease_full_updates = int(variant.rsplit("_L", 1)[1])
        elif variant == "access_freshness_lease":
            lease_full_updates = lease_value(row)
        else:
            lease_full_updates = 0
        row["lease_full_updates"] = lease_full_updates
    return rows


def comparisons(experiment: str) -> tuple[tuple[str, str], ...]:
    if experiment == "generalization":
        return (("access_freshness_lease", "access_full"),)
    return (
        ("access_freshness_lease_L3", "access_full"),
        ("access_freshness_lease_L5", "access_full"),
        ("access_freshness_lease_L10", "access_full"),
        ("access_freshness_lease_L3", "access_freshness_lease_L5"),
        ("access_freshness_lease_L10", "access_freshness_lease_L5"),
    )


def paired_effects(
    runs: list[dict], experiment: str,
) -> list[dict]:
    by_variant_key = {
        (row["variant"],) + tuple(row[field] for field in PAIR_FIELDS): row
        for row in runs
    }
    all_keys = sorted({tuple(row[field] for field in PAIR_FIELDS) for row in runs})
    output = []
    for method_variant, control_variant in comparisons(experiment):
        for key in all_keys:
            method = by_variant_key.get((method_variant,) + key)
            control = by_variant_key.get((control_variant,) + key)
            if method is None or control is None:
                raise ValueError(
                    f"missing paired run {method_variant}/{control_variant} for {key}"
                )
            result = {field: value for field, value in zip(PAIR_FIELDS, key)}
            result.update({
                "method_variant": method_variant,
                "control_variant": control_variant,
                "sustained_target_share_reduction": (
                    float(control["target_effective_aggregation_share_sustained"])
                    - float(method["target_effective_aggregation_share_sustained"])
                ),
                "limited_returned_update_rate_increase": (
                    float(method["target_limited_returned_update_rate_after_start"])
                    - float(control["target_limited_returned_update_rate_after_start"])
                ),
                "full_weight_aggregated_update_rate_reduction": (
                    float(control["target_full_weight_aggregated_update_rate_after_start"])
                    - float(method["target_full_weight_aggregated_update_rate_after_start"])
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


def paired_summary(pair_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in pair_rows:
        groups[tuple(row[field] for field in PAIR_SUMMARY_FIELDS)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        row = {field: value for field, value in zip(PAIR_SUMMARY_FIELDS, key)}
        row["independent_pairs"] = len(group)
        for field in EFFECT_FIELDS:
            values = [
                float(item[field]) for item in group
                if math.isfinite(float(item[field]))
            ]
            mean, ci = mean_ci(values)
            row[f"{field}_mean"] = mean
            row[f"{field}_ci95"] = ci
            row[f"{field}_positive_pairs"] = sum(value > 0 for value in values)
            row[f"{field}_negative_pairs"] = sum(value < 0 for value in values)
            row[f"{field}_zero_pairs"] = sum(value == 0 for value in values)
            if len(values) < 2:
                pvalue = math.nan
            elif all(value == values[0] for value in values):
                pvalue = 0.0 if values[0] != 0.0 else 1.0
            else:
                pvalue = float(ttest_1samp(values, 0.0).pvalue)
            row[f"{field}_paired_t_pvalue"] = pvalue
        output.append(row)
    return output


def main() -> None:
    if len(sys.argv) != 13:
        raise SystemExit(
            "usage: analyze_v46_evidence EXPERIMENT STAGE nodes.csv events.csv "
            "metrics.csv configuration.csv runs.csv summary.csv pair_runs.csv "
            "pair_summary.csv round_runs.csv round_summary.csv"
        )
    (
        _, experiment, stage, node_path, event_path, metric_path, config_path,
        run_path, summary_path, pair_path, pair_summary_path,
        round_run_path, round_summary_path,
    ) = sys.argv
    if experiment not in {"generalization", "sensitivity"}:
        raise ValueError(f"unknown V46 experiment: {experiment}")
    nodes = read(node_path)
    events = read(event_path)
    metrics = read(metric_path)
    configs = read(config_path)
    if experiment == "sensitivity":
        nodes = normalize_sensitivity(nodes)
        events = normalize_sensitivity(events)
        metrics = normalize_sensitivity(metrics)
        configs = normalize_sensitivity(configs)
    runs = annotate_lease_setting(enrich_runs(
        build_long_runs(nodes, events, metrics, configs, stage), nodes, events
    ))
    round_runs = annotate_lease_setting(enrich_round_runs(
        build_round_runs(nodes, events, metrics), nodes, events
    ))
    summary = annotate_lease_setting(summarize_v45(runs))
    round_summary = annotate_lease_setting(summarize_rounds_v45(round_runs))
    pair_rows = paired_effects(runs, experiment)
    write(run_path, runs)
    write(summary_path, summary)
    write(pair_path, pair_rows)
    write(pair_summary_path, paired_summary(pair_rows))
    write(round_run_path, round_runs)
    write(round_summary_path, round_summary)


if __name__ == "__main__":
    main()

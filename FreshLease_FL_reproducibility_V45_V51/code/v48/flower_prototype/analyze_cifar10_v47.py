from __future__ import annotations

import math
import sys

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
from flower_prototype.analyze_q3_v43 import read, write
from flower_prototype.analyze_v46_evidence import (
    PAIR_FIELDS,
    annotate_lease_setting,
    paired_summary,
)


COMPARISONS = (
    ("access_freshness_lease", "access_full"),
    ("access_freshness_lease", "rffl_reputation"),
)


def paired_effects_v47(runs: list[dict]) -> list[dict]:
    by_variant_key = {
        (row["variant"],) + tuple(row[field] for field in PAIR_FIELDS): row
        for row in runs
    }
    all_keys = sorted({tuple(row[field] for field in PAIR_FIELDS) for row in runs})
    output = []
    for method_variant, control_variant in COMPARISONS:
        for key in all_keys:
            method = by_variant_key.get((method_variant,) + key)
            control = by_variant_key.get((control_variant,) + key)
            if method is None or control is None:
                raise ValueError(
                    f"missing V47 paired run {method_variant}/{control_variant} for {key}"
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


def main() -> None:
    if len(sys.argv) != 12:
        raise SystemExit(
            "usage: analyze_cifar10_v47 STAGE nodes.csv events.csv metrics.csv "
            "configuration.csv runs.csv summary.csv pair_runs.csv "
            "pair_summary.csv round_runs.csv round_summary.csv"
        )
    (
        _, stage, node_path, event_path, metric_path, config_path,
        run_path, summary_path, pair_path, pair_summary_path,
        round_run_path, round_summary_path,
    ) = sys.argv
    nodes = read(node_path)
    events = read(event_path)
    metrics = read(metric_path)
    configs = read(config_path)
    runs = annotate_lease_setting(enrich_runs(
        build_long_runs(nodes, events, metrics, configs, stage), nodes, events
    ))
    round_runs = annotate_lease_setting(enrich_round_runs(
        build_round_runs(nodes, events, metrics), nodes, events
    ))
    pairs = paired_effects_v47(runs)
    write(run_path, runs)
    write(summary_path, annotate_lease_setting(summarize_v45(runs)))
    write(pair_path, pairs)
    write(pair_summary_path, paired_summary(pairs))
    write(round_run_path, round_runs)
    write(round_summary_path, annotate_lease_setting(summarize_rounds_v45(round_runs)))


if __name__ == "__main__":
    main()

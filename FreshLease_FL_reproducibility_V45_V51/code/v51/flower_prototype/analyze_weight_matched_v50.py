"""Independent weight-matched control for the fixed V49 post-promotion study."""

from __future__ import annotations

import math
import sys
from collections import defaultdict

from scipy.stats import t as student_t

from flower_prototype.access_control import V50_MATCHED_WEIGHT
from flower_prototype.analyze_freshness_lease_v45 import (
    enrich_runs, summarize_v45,
)
from flower_prototype.analyze_long_horizon_v44 import build_long_runs
from flower_prototype.analyze_post_promotion_v49 import (
    adjust_windows_to_attack_start, audit_post_promotion, identity,
)
from flower_prototype.analyze_q3_v43 import mean_ci, read, write
from flower_prototype.analyze_v46_evidence import annotate_lease_setting


FIELDS = (
    "asr_auc_reduction", "final_asr_reduction", "final_accuracy_difference",
    "sustained_target_share_reduction", "target_aggregation_mass_difference",
)


def target_weight_mass(nodes: list[dict], events: list[dict]) -> dict[tuple, dict]:
    """Use actual aggregated target updates, not claimed fit submissions."""
    targets: dict[tuple, set[str]] = defaultdict(set)
    for row in nodes:
        if row["profile"] != "honest":
            targets[identity(row)].add(row["node_id"])
    mass: dict[tuple, dict] = defaultdict(lambda: {"mass": 0.0, "updates": 0})
    for row in events:
        key = identity(row)
        if (row["node_id"] in targets[key] and int(float(row["round"])) >= 6
                and int(float(row.get("aggregated") or 0)) == 1):
            mass[key]["mass"] += float(row["aggregation_weight"])
            mass[key]["updates"] += 1
    if set(targets) != set(mass):
        raise ValueError("missing aggregated target updates for weight audit")
    return mass


def paired_v50(control_runs: list[dict], lease_runs: list[dict],
               control_mass: dict[tuple, dict],
               lease_mass: dict[tuple, dict],
               guard: list[dict], lease_guard: list[dict]) -> tuple[list[dict], list[dict]]:
    fields = ("dataset", "client_count", "attacker_count", "cohort_mode",
              "trust_estimator", "attack_scenario", "noniid_alpha", "repeat")
    key = lambda r: tuple(r[f] for f in fields)
    matched = {key(r): r for r in control_runs}
    lease = {key(r): r for r in lease_runs
             if r["variant"] == "access_freshness_lease"}
    if (not matched or len(matched) != len(control_runs) or len(lease) != len(matched)
            or set(lease) != set(matched)):
        raise ValueError("V49 lease and V50 matched-weight configurations do not pair exactly")
    for name, rows, wanted in (
        ("V49", lease_guard, "access_freshness_lease"),
        ("V50", guard, "access_weight_matched"),
    ):
        selected = {key(r): r for r in rows if r["variant"] == wanted}
        if set(selected) != set(matched):
            raise ValueError(f"{name} promotion guard does not match paired runs")
        if any(float(r["target_full_access_before_attack_rate"]) != 1.0
               or int(r["target_first_full_access_round_max"]) != 5
               or int(r["attack_start_round"]) != 6 for r in selected.values()):
            raise ValueError(f"{name} includes a target attacking before promotion")

    result = []
    for pair_key in sorted(matched):
        control, method = matched[pair_key], lease[pair_key]
        c_key, m_key = identity(control), identity(method)
        if c_key not in control_mass or m_key not in lease_mass:
            raise ValueError(f"missing weight ledger for {pair_key}")
        c, m = control_mass[c_key], lease_mass[m_key]
        result.append({
            **dict(zip(fields, pair_key)), "method_variant": "access_freshness_lease",
            "control_variant": "access_weight_matched",
            "nominal_matched_weight": V50_MATCHED_WEIGHT,
            "lease_aggregated_target_updates": m["updates"],
            "matched_aggregated_target_updates": c["updates"],
            "lease_actual_target_weight_mass": m["mass"],
            "matched_actual_target_weight_mass": c["mass"],
            "target_aggregation_mass_difference": m["mass"] - c["mass"],
            "asr_auc_reduction": float(control["backdoor_asr_auc_after_start"])
                                 - float(method["backdoor_asr_auc_after_start"]),
            "final_asr_reduction": float(control["final_backdoor_asr"])
                                   - float(method["final_backdoor_asr"]),
            "final_accuracy_difference": float(method["final_accuracy"])
                                          - float(control["final_accuracy"]),
            "sustained_target_share_reduction":
                float(control["target_effective_aggregation_share_sustained"])
                - float(method["target_effective_aggregation_share_sustained"]),
        })
    summary = []
    for scenario in sorted({r["attack_scenario"] for r in result}):
        group = [r for r in result if r["attack_scenario"] == scenario]
        row = {"attack_scenario": scenario, "independent_pairs": len(group),
               "nominal_matched_weight": V50_MATCHED_WEIGHT}
        for field in FIELDS:
            values = [float(r[field]) for r in group]
            mean, ci = mean_ci(values)
            row[field + "_mean"] = mean
            row[field + "_ci95_half_width"] = ci
            row[field + "_positive_pairs"] = sum(x > 0 for x in values)
            row[field + "_negative_pairs"] = sum(x < 0 for x in values)
            if len(values) < 2:
                row[field + "_paired_t_pvalue"] = math.nan
            elif max(values) == min(values):
                row[field + "_paired_t_pvalue"] = 1.0 if values[0] == 0 else 0.0
            else:
                mean_value = sum(values) / len(values)
                variance = sum((x - mean_value)**2 for x in values) / (len(values)-1)
                stat = mean_value / math.sqrt(variance / len(values))
                row[field + "_paired_t_pvalue"] = float(
                    2 * student_t.sf(abs(stat), df=len(values)-1)
                )
        row["empirical_weight_mass_matched_within_1pct_rate"] = sum(
            abs(float(r["target_aggregation_mass_difference"])) <=
            0.01 * max(1.0, float(r["lease_actual_target_weight_mass"]))
            for r in group
        ) / len(group)
        summary.append(row)
    return result, summary


def main() -> None:
    if len(sys.argv) not in (12, 16):
        raise SystemExit("usage: analyze_weight_matched_v50 STAGE nodes events metrics "
                         "config runs summary guard pair_runs pair_summary ROUNDS "
                         "[v49_runs v49_guard v49_nodes v49_events]")
    # The reference is supplied only for a formal (50-round) comparison.
    _, stage, node_path, event_path, metric_path, config_path, run_path, \
        summary_path, guard_path, pair_path, pair_summary_path, rounds, *reference = sys.argv
    if (stage == "formal") != bool(reference):
        raise ValueError("formal comparison requires a V49 reference directory")
    if reference and len(reference) != 4:
        raise ValueError("V49 reference must contain runs, guard, nodes and events")
    nodes, events = read(node_path), read(event_path)
    configs, metrics = read(config_path), read(metric_path)
    guard = audit_post_promotion(nodes, events, configs,
                                 metrics=metrics, expected_rounds=int(rounds))
    if any(r["variant"] != "access_weight_matched" for r in guard):
        raise ValueError("V50 results must contain only the new control")
    runs = annotate_lease_setting(enrich_runs(
        build_long_runs(nodes, events, metrics, configs, stage), nodes, events,
    ))
    adjust_windows_to_attack_start(runs, events, nodes)
    write(run_path, runs)
    write(summary_path, annotate_lease_setting(summarize_v45(runs)))
    write(guard_path, guard)
    if reference:
        lease_runs, lease_guard, lease_nodes, lease_events = map(read, reference)
        paired, paired_summary = paired_v50(
            runs, lease_runs, target_weight_mass(nodes, events),
            target_weight_mass(lease_nodes, lease_events), guard, lease_guard,
        )
        write(pair_path, paired)
        write(pair_summary_path, paired_summary)


if __name__ == "__main__":
    main()

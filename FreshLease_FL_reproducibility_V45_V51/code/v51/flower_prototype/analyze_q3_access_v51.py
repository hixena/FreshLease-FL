"""V51 paired outcome and honest-newcomer lifecycle audit.

Only the benign_concept_drift TARGETS are honest newcomers. Predeclared
honest incumbents are kept separate and never used to estimate newcomer FRR.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from flower_prototype.analyze_cifar10_v47 import paired_effects_v47
from flower_prototype.analyze_freshness_lease_v45 import enrich_runs, summarize_v45
from flower_prototype.analyze_long_horizon_v44 import build_long_runs
from flower_prototype.analyze_post_promotion_v49 import (
    adjust_windows_to_attack_start, audit_post_promotion, identity,
)
from flower_prototype.analyze_q3_v43 import mean_ci, read, write
from flower_prototype.analyze_v46_evidence import annotate_lease_setting, paired_summary


KEY = (
    "dataset", "client_count", "attacker_count", "cohort_mode", "variant",
    "trust_estimator", "attack_scenario", "noniid_alpha", "repeat",
)
GROUP = KEY[:-1]


def access_runs(nodes: list[dict], configs: list[dict], guard: list[dict]) -> list[dict]:
    by_node: dict[tuple, list[dict]] = defaultdict(list)
    guard_by_key = {identity(row): row for row in guard}
    for row in nodes:
        by_node[identity(row)].append(row)
    result = []
    if len({identity(row) for row in configs}) != len(configs):
        raise ValueError("duplicate V51 configuration key")
    if set(by_node) != {identity(row) for row in configs} or set(guard_by_key) != set(by_node):
        raise ValueError("missing V51 nodes or promotion guards")
    for config in configs:
        key = identity(config)
        cohort = by_node[key]
        newcomer = [row for row in cohort if row["profile"] != "honest"]
        incumbent = [row for row in cohort if row["profile"] == "honest"]
        if len(cohort) != int(config["client_count"]) or len(newcomer) != int(config["attacker_count"]):
            raise ValueError("V51 unexpected incumbent/newcomer cohort size")
        # Raw real_fl_node_results has initial_access_state but not always an
        # initial_training_eligible column; only LIMITED/ADMITTED can train.
        eligible = lambda row: int(row["initial_access_state"] in {"LIMITED", "ADMITTED"})
        g = guard_by_key[key]
        result.append({
            **{field: config[field] for field in KEY},
            "legitimate_newcomer_case": int(config["attack_scenario"] == "benign_concept_drift"),
            "newcomer_count": len(newcomer), "incumbent_count": len(incumbent),
            "newcomer_initial_training_eligible_rate": sum(map(eligible, newcomer)) / len(newcomer),
            "newcomer_initial_limited_rate": sum(
                row["initial_access_state"] == "LIMITED" for row in newcomer
            ) / len(newcomer),
            "newcomer_completed_tasks_mean": sum(float(row["completed_tasks"]) for row in newcomer) / len(newcomer),
            "newcomer_onboarding_seconds_mean": sum(float(row["onboarding_duration_seconds"]) for row in newcomer) / len(newcomer),
            "incumbent_initial_training_eligible_rate": sum(map(eligible, incumbent)) / len(incumbent),
            "newcomer_full_access_before_attack_rate": float(g["target_full_access_before_attack_rate"]),
            "newcomer_first_full_access_round_max": g["target_first_full_access_round_max"],
        })
    return result


def summarize_access(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[f] for f in GROUP)].append(row)
    summary = []
    values = (
        "newcomer_initial_training_eligible_rate", "newcomer_initial_limited_rate",
        "newcomer_completed_tasks_mean", "newcomer_onboarding_seconds_mean",
        "incumbent_initial_training_eligible_rate", "newcomer_full_access_before_attack_rate",
    )
    for key, subset in sorted(grouped.items()):
        out = dict(zip(GROUP, key))
        out["independent_runs"] = len(subset)
        out["legitimate_newcomer_case"] = subset[0]["legitimate_newcomer_case"]
        for field in values:
            mean, half_width = mean_ci([float(row[field]) for row in subset])
            out[field + "_mean"] = mean
            out[field + "_ci95_half_width"] = half_width
        summary.append(out)
    return summary


def audit_integrity_rows(audits: list[dict], configs: list[dict]) -> None:
    if (len(audits) != len(configs)
            or {identity(row) for row in audits} != {identity(row) for row in configs}):
        raise ValueError("V51 per-configuration controller audit missing or duplicated")
    if any(int(row["audit_valid"]) != 1 or int(row["chain_valid"]) != 1
           or int(row["signatures_valid"]) != 1
           or int(row["event_count"]) < 1
           or int(row["checked_signatures"]) < 1 for row in audits):
        raise ValueError("V51 controller audit integrity failed")


def main() -> None:
    if len(sys.argv) != 15:
        raise SystemExit("usage: analyze_q3_access_v51 STAGE nodes events metrics configs audits "
                         "runs summary pair_runs pair_summary guard access_runs access_summary rounds")
    (_, stage, node_file, event_file, metric_file, config_file, audit_file, run_file,
     summary_file, paired_file, paired_summary_file, guard_file, access_file,
     access_summary_file, rounds) = sys.argv
    nodes, events = read(node_file), read(event_file)
    metrics, configs = read(metric_file), read(config_file)
    audit_integrity_rows(read(audit_file), configs)
    guard = audit_post_promotion(nodes, events, configs, metrics=metrics,
                                 expected_rounds=int(rounds))
    observed = {row["variant"] for row in configs}
    if observed != {"access_freshness_lease", "access_full", "rffl_reputation"}:
        raise ValueError("V51 missing a paired variant")
    runs = annotate_lease_setting(enrich_runs(
        build_long_runs(nodes, events, metrics, configs, stage), nodes, events,
    ))
    adjust_windows_to_attack_start(runs, events, nodes)
    pairs = paired_effects_v47(runs)
    access = access_runs(nodes, configs, guard)
    # Write only after all completeness and pairing checks have passed.
    write(run_file, runs)
    write(summary_file, annotate_lease_setting(summarize_v45(runs)))
    write(paired_file, pairs)
    write(paired_summary_file, paired_summary(pairs))
    write(guard_file, guard)
    write(access_file, access)
    write(access_summary_file, summarize_access(access))


if __name__ == "__main__":
    main()

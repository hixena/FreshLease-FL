"""Run-level and cluster-aware summaries for the V39 access-core matrix."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


KEYS = (
    "dataset", "cohort_mode", "variant", "trust_estimator",
    "attack_scenario", "noniid_alpha", "repeat",
)

TARGETS = {
    "single_type_farming": "single-type-farming",
    "false_quality_reporting": "false-quality-reporting",
    "attestation_camouflage": "attestation-camouflage",
    "diverse_then_repeat_farming": "diverse-repeat-farming",
}

SCENARIO_CLAIMS = {
    "single_type_farming": "pre_access_detectable",
    "false_quality_reporting": "pre_access_detectable",
    "attestation_camouflage": "pre_access_detectable",
    "diverse_then_repeat_farming": "post_admission_boundary",
}

RUN_FIELDS = (
    *KEYS, "scenario_claim", "client_count", "normal_nodes", "target_node_id",
    "target_profile", "target_initial_access_state", "target_initial_admitted",
    "target_initial_limited", "target_initial_training_eligible",
    "target_correct_access_decision", "target_final_access_state",
    "target_completed_tasks", "target_credited_independent_tasks",
    "target_evidence_types", "target_evidence_mass", "target_trust_score",
    "target_history_decision_score", "target_history_std",
    "target_history_evidence_maturity", "target_onboarding_duration_seconds",
    "target_structurally_accepted_training_updates", "target_flower_fit_events",
    "target_aggregated_fit_events", "target_aggregated_weight_mass",
    "normal_initial_admitted_rate", "normal_initial_limited_rate",
    "normal_initial_training_eligible_rate", "normal_final_admitted_rate",
    "normal_completed_tasks_mean", "normal_evidence_types_mean",
    "normal_evidence_mass_mean", "normal_onboarding_duration_seconds_mean",
    "normal_aggregated_fit_events_mean", "normal_aggregated_weight_mass_mean",
)

SUMMARY_FIELDS = (
    "dataset", "cohort_mode", "variant", "trust_estimator",
    "attack_scenario", "noniid_alpha", "scenario_claim", "independent_runs",
    "client_count", "target_initial_training_eligible_rate",
    "target_initial_training_eligible_wilson_ci95_lower",
    "target_initial_training_eligible_wilson_ci95_upper",
    "target_initial_admitted_rate", "target_initial_limited_rate",
    "target_correct_access_decision_rate",
    "target_correct_access_decision_wilson_ci95_lower",
    "target_correct_access_decision_wilson_ci95_upper",
    "normal_node_records", "normal_initial_training_eligible_rate_mean",
    "normal_initial_training_eligible_rate_ci95",
    "normal_initial_admitted_rate_mean", "normal_initial_admitted_rate_ci95",
    "normal_initial_limited_rate_mean", "normal_initial_limited_rate_ci95",
    "target_completed_tasks_mean", "target_completed_tasks_ci95",
    "normal_completed_tasks_mean", "normal_completed_tasks_ci95",
    "target_evidence_types_mean", "target_evidence_types_ci95",
    "target_evidence_mass_mean", "target_evidence_mass_ci95",
    "target_history_evidence_maturity_mean",
    "target_history_evidence_maturity_ci95",
    "target_onboarding_duration_seconds_mean",
    "target_onboarding_duration_seconds_ci95",
    "normal_onboarding_duration_seconds_mean",
    "normal_onboarding_duration_seconds_ci95",
    "target_aggregated_fit_events_mean", "target_aggregated_fit_events_ci95",
    "target_aggregated_weight_mass_mean", "target_aggregated_weight_mass_ci95",
    "normal_aggregated_fit_events_mean", "normal_aggregated_fit_events_ci95",
    "normal_aggregated_weight_mass_mean", "normal_aggregated_weight_mass_ci95",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.setdefault("dataset", "digits")
        row.setdefault("cohort_mode", "synchronous_cold_start")
        row.setdefault("trust_estimator", "dirichlet_lcb")
    return rows


def _number(row: dict[str, str], field: str, fallback: str | None = None) -> float:
    value = row.get(field, "")
    if value in {"", None} and fallback is not None:
        value = row.get(fallback, "")
    return float(value or 0.0)


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(_number(row, field) for row in rows)


def _run_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in KEYS)


def analyze_runs(nodes: list[dict[str, str]]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in nodes:
        grouped[_run_key(row)].append(row)
    output: list[dict] = []
    for key, group in sorted(grouped.items()):
        scenario = key[4]
        if scenario not in TARGETS:
            raise ValueError(f"unsupported V39 access scenario: {scenario}")
        targets = [row for row in group if row.get("node_id") == TARGETS[scenario]]
        if len(targets) != 1:
            raise ValueError(f"expected one target for V39 run {key}, got {len(targets)}")
        target = targets[0]
        normal = [row for row in group if row.get("profile") == "honest"]
        if not normal:
            raise ValueError(f"V39 run has no honest nodes: {key}")
        initial_state = target.get("initial_access_state") or target.get("access_state", "")
        target_eligible = initial_state in {"LIMITED", "ADMITTED"}
        claim = SCENARIO_CLAIMS[scenario]
        correct = target_eligible if claim == "post_admission_boundary" else not target_eligible
        normal_states = [
            row.get("initial_access_state") or row.get("access_state", "")
            for row in normal
        ]
        structural_field = (
            "structurally_accepted_training_updates"
            if "structurally_accepted_training_updates" in target
            else "training_clean_evidence"
        )
        result = dict(zip(KEYS, key))
        result.update({
            "scenario_claim": claim,
            "client_count": len(group),
            "normal_nodes": len(normal),
            "target_node_id": target["node_id"],
            "target_profile": target.get("profile", ""),
            "target_initial_access_state": initial_state,
            "target_initial_admitted": int(initial_state == "ADMITTED"),
            "target_initial_limited": int(initial_state == "LIMITED"),
            "target_initial_training_eligible": int(target_eligible),
            "target_correct_access_decision": int(correct),
            "target_final_access_state": target.get("access_state", ""),
            "target_completed_tasks": _number(target, "completed_tasks"),
            "target_credited_independent_tasks": _number(
                target, "credited_independent_tasks"
            ),
            "target_evidence_types": _number(target, "evidence_types"),
            "target_evidence_mass": _number(target, "evidence_mass"),
            "target_trust_score": _number(target, "trust_score"),
            "target_history_decision_score": _number(
                target, "history_decision_score"
            ),
            "target_history_std": _number(target, "history_std"),
            "target_history_evidence_maturity": _number(
                target, "history_evidence_maturity"
            ),
            "target_onboarding_duration_seconds": _number(
                target, "onboarding_duration_seconds"
            ),
            "target_structurally_accepted_training_updates": _number(
                target, structural_field
            ),
            "target_flower_fit_events": _number(target, "flower_fit_events"),
            "target_aggregated_fit_events": _number(
                target, "aggregated_fit_events", "flower_fit_events"
            ),
            "target_aggregated_weight_mass": _number(
                target, "aggregated_weight_mass"
            ),
            "normal_initial_admitted_rate": (
                sum(state == "ADMITTED" for state in normal_states) / len(normal)
            ),
            "normal_initial_limited_rate": (
                sum(state == "LIMITED" for state in normal_states) / len(normal)
            ),
            "normal_initial_training_eligible_rate": (
                sum(state in {"LIMITED", "ADMITTED"} for state in normal_states)
                / len(normal)
            ),
            "normal_final_admitted_rate": (
                sum(row.get("access_state") == "ADMITTED" for row in normal)
                / len(normal)
            ),
            "normal_completed_tasks_mean": _mean(normal, "completed_tasks"),
            "normal_evidence_types_mean": _mean(normal, "evidence_types"),
            "normal_evidence_mass_mean": _mean(normal, "evidence_mass"),
            "normal_onboarding_duration_seconds_mean": _mean(
                normal, "onboarding_duration_seconds"
            ),
            "normal_aggregated_fit_events_mean": statistics.fmean(
                _number(row, "aggregated_fit_events", "flower_fit_events")
                for row in normal
            ),
            "normal_aggregated_weight_mass_mean": _mean(
                normal, "aggregated_weight_mass"
            ),
        })
        output.append(result)
    return output


_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_ci95(values: list[float]) -> float | str:
    if len(values) < 2:
        return ""
    critical = _T975.get(len(values) - 1, 1.96)
    return critical * statistics.stdev(values) / math.sqrt(len(values))


def wilson(successes: int, trials: int) -> tuple[float | str, float | str]:
    if trials < 1:
        return "", ""
    z = 1.96
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize(runs: list[dict]) -> list[dict]:
    group_fields = KEYS[:-1]
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in runs:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    output: list[dict] = []
    for key, group in sorted(grouped.items()):
        n = len(group)
        eligible = sum(int(row["target_initial_training_eligible"]) for row in group)
        correct = sum(int(row["target_correct_access_decision"]) for row in group)
        eligible_interval = wilson(eligible, n)
        correct_interval = wilson(correct, n)
        result = dict(zip(group_fields, key))
        result.update({
            "scenario_claim": group[0]["scenario_claim"],
            "independent_runs": n,
            "client_count": group[0]["client_count"],
            "target_initial_training_eligible_rate": eligible / n,
            "target_initial_training_eligible_wilson_ci95_lower": eligible_interval[0],
            "target_initial_training_eligible_wilson_ci95_upper": eligible_interval[1],
            "target_initial_admitted_rate": statistics.fmean(
                float(row["target_initial_admitted"]) for row in group
            ),
            "target_initial_limited_rate": statistics.fmean(
                float(row["target_initial_limited"]) for row in group
            ),
            "target_correct_access_decision_rate": correct / n,
            "target_correct_access_decision_wilson_ci95_lower": correct_interval[0],
            "target_correct_access_decision_wilson_ci95_upper": correct_interval[1],
            "normal_node_records": sum(int(row["normal_nodes"]) for row in group),
        })
        mean_fields = {
            "normal_initial_training_eligible_rate_mean": "normal_initial_training_eligible_rate",
            "normal_initial_admitted_rate_mean": "normal_initial_admitted_rate",
            "normal_initial_limited_rate_mean": "normal_initial_limited_rate",
            "target_completed_tasks_mean": "target_completed_tasks",
            "normal_completed_tasks_mean": "normal_completed_tasks_mean",
            "target_evidence_types_mean": "target_evidence_types",
            "target_evidence_mass_mean": "target_evidence_mass",
            "target_history_evidence_maturity_mean": "target_history_evidence_maturity",
            "target_onboarding_duration_seconds_mean": "target_onboarding_duration_seconds",
            "normal_onboarding_duration_seconds_mean": "normal_onboarding_duration_seconds_mean",
            "target_aggregated_fit_events_mean": "target_aggregated_fit_events",
            "target_aggregated_weight_mass_mean": "target_aggregated_weight_mass",
            "normal_aggregated_fit_events_mean": "normal_aggregated_fit_events_mean",
            "normal_aggregated_weight_mass_mean": "normal_aggregated_weight_mass_mean",
        }
        for mean_field, source in mean_fields.items():
            values = [float(row[source]) for row in group]
            result[mean_field] = statistics.fmean(values)
            result[mean_field.removesuffix("_mean") + "_ci95"] = t_ci95(values)
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    if not rows:
        raise ValueError("no V39 access rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: analyze_access_v39 NODE_RESULTS RUN_OUT SUMMARY_OUT"
        )
    runs = analyze_runs(read_csv(Path(sys.argv[1])))
    write_csv(Path(sys.argv[2]), runs, RUN_FIELDS)
    write_csv(Path(sys.argv[3]), summarize(runs), SUMMARY_FIELDS)


if __name__ == "__main__":
    main()

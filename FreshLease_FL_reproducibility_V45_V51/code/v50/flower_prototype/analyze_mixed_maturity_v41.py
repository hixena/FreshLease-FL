"""Dependency-free analysis for the V41 mixed-maturity access experiment."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


CONFIG = (
    "dataset", "cohort_mode", "variant", "trust_estimator", "attack_scenario",
    "noniid_alpha", "repeat", "attack_start_round", "limited_observation_updates",
    "limited_aggregation_weight", "max_probation_tasks",
    "min_probation_completed_tasks", "min_evidence_mass",
)
TARGETS = {
    "single_type_farming": "single-type-farming",
    "three_type_repeat_farming": "three-type-repeat-farming",
    "false_quality_reporting": "false-quality-reporting",
    "attestation_camouflage": "attestation-camouflage",
    "diverse_then_repeat_farming": "diverse-repeat-farming",
    "diverse_then_repeat_backdoor": "diverse-repeat-backdoor",
}
BOUNDARY = {"diverse_then_repeat_farming", "diverse_then_repeat_backdoor"}

RUN_FIELDS = (*CONFIG, "stage", "scenario_claim", "target_node_id", "target_cohort_role",
    "target_initial_access_state", "target_initial_training_eligible",
    "target_correct_access_decision", "target_final_access_state",
    "target_completed_tasks", "target_evidence_types", "target_evidence_mass",
    "target_history_evidence_maturity", "normal_nodes",
    "normal_initial_training_eligible_rate", "normal_initial_admitted_rate",
    "normal_initial_limited_rate", "target_returned_updates_after_start",
    "target_aggregated_updates_after_start", "target_rejected_updates_after_start",
    "target_aggregation_weight_after_start", "normal_aggregation_weight_after_start",
    "target_effective_aggregation_share_after_start",
    "target_vs_normal_mean_aggregation_weight_ratio",
    "target_limited_state_rounds_after_start",
    "target_limited_weighted_updates_after_start", "target_promotion_round",
    "final_accuracy", "final_backdoor_asr")

SUMMARY_FIELDS = tuple(field for field in CONFIG if field != "repeat") + (
    "stage", "scenario_claim", "independent_runs",
    "target_initial_training_eligible_rate", "target_initial_training_eligible_wilson_ci95_lower",
    "target_initial_training_eligible_wilson_ci95_upper", "target_correct_access_decision_rate",
    "normal_initial_training_eligible_rate_mean", "normal_initial_training_eligible_rate_ci95",
    "normal_initial_admitted_rate_mean", "normal_initial_admitted_rate_ci95",
    "normal_initial_limited_rate_mean", "normal_initial_limited_rate_ci95",
    "target_completed_tasks_mean", "target_completed_tasks_ci95",
    "target_evidence_types_mean", "target_evidence_types_ci95",
    "target_evidence_mass_mean", "target_evidence_mass_ci95",
    "target_history_evidence_maturity_mean", "target_history_evidence_maturity_ci95",
    "target_returned_updates_after_start_mean", "target_returned_updates_after_start_ci95",
    "target_aggregated_updates_after_start_mean", "target_aggregated_updates_after_start_ci95",
    "target_rejected_updates_after_start_mean", "target_rejected_updates_after_start_ci95",
    "target_aggregation_weight_after_start_mean", "target_aggregation_weight_after_start_ci95",
    "normal_aggregation_weight_after_start_mean", "normal_aggregation_weight_after_start_ci95",
    "target_effective_aggregation_share_after_start_mean",
    "target_effective_aggregation_share_after_start_ci95",
    "target_vs_normal_mean_aggregation_weight_ratio_mean",
    "target_vs_normal_mean_aggregation_weight_ratio_ci95",
    "target_limited_state_rounds_after_start_mean", "target_limited_state_rounds_after_start_ci95",
    "target_limited_weighted_updates_after_start_mean",
    "target_limited_weighted_updates_after_start_ci95",
    "target_promotion_round_mean", "target_promotion_round_ci95",
    "final_accuracy_mean", "final_accuracy_ci95", "final_backdoor_asr_mean",
    "final_backdoor_asr_ci95",
)

PAIR_FIELDS = ("dataset", "attack_scenario", "noniid_alpha", "repeat",
    "limited_observation_updates", "limited_aggregation_weight",
    "full_target_aggregation_weight_after_start", "control_target_aggregation_weight_after_start",
    "target_exposure_reduction", "full_final_accuracy", "control_final_accuracy",
    "accuracy_difference", "full_final_backdoor_asr", "control_final_backdoor_asr",
    "backdoor_asr_reduction", "full_target_effective_aggregation_share",
    "control_target_effective_aggregation_share", "target_share_reduction",
    "full_target_limited_state_rounds_after_start",
    "full_target_limited_weighted_updates_after_start", "full_target_promotion_round")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    defaults = {"dataset": "digits", "cohort_mode": "synchronous_cold_start",
                "trust_estimator": "dirichlet_lcb", "attack_start_round": "1",
                "limited_aggregation_weight": "0.25",
                "max_probation_tasks": "20", "min_probation_completed_tasks": "9",
                "min_evidence_mass": "1.50"}
    for row in rows:
        row.setdefault(
            "limited_observation_updates",
            row.get("limited_clean_updates", "2") or "2",
        )
        for key, value in defaults.items():
            row.setdefault(key, value)
    return rows


def number(row: dict, field: str) -> float:
    return float(row.get(field, "") or 0.0)


def key(row: dict) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in CONFIG)


def t_ci(values: list[float]) -> float | str:
    if len(values) < 2:
        return ""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    return table.get(len(values) - 1, 1.96) * statistics.stdev(values) / math.sqrt(len(values))


def wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.96
    p = successes / trials
    d = 1 + z * z / trials
    c = (p + z * z / (2 * trials)) / d
    h = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / d
    return max(0.0, c - h), min(1.0, c + h)


def analyze(stage: str, nodes: list[dict], events: list[dict], metrics: list[dict]) -> list[dict]:
    node_groups, event_groups, metric_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in nodes: node_groups[key(row)].append(row)
    for row in events: event_groups[key(row)].append(row)
    for row in metrics: metric_groups[key(row)].append(row)
    output = []
    for run_key, group in sorted(node_groups.items()):
        scenario = run_key[4]
        target_id = TARGETS.get(scenario)
        if not target_id:
            raise ValueError(f"unsupported V40 scenario: {scenario}")
        targets = [row for row in group if row.get("node_id") == target_id]
        normal = [row for row in group if row.get("profile") == "honest"]
        if len(targets) != 1 or not normal:
            raise ValueError(f"invalid target/normal membership for {run_key}")
        target = targets[0]
        initial = target.get("initial_access_state") or target.get("access_state", "")
        eligible = initial in {"LIMITED", "ADMITTED"}
        claim = "post_admission_boundary" if scenario in BOUNDARY else "pre_access_detectable"
        correct = eligible if claim == "post_admission_boundary" else not eligible
        start = int(float(target.get("attack_start_round", 1)))
        post = [row for row in event_groups[run_key] if int(float(row.get("round", 0))) >= start]
        target_events = [row for row in post if row.get("node_id") == target_id]
        normal_events = [row for row in post if row.get("profile") == "honest" or row.get("node_id", "").startswith("honest-")]
        finals = sorted(metric_groups[run_key], key=lambda row: int(float(row.get("round", 0))))
        final = finals[-1] if finals else {}
        states = [row.get("initial_access_state") or row.get("access_state", "") for row in normal]
        if stage in {"primary", "observation_sensitivity"}:
            if target.get("cohort_role") != "newcomer":
                raise ValueError(f"V41 target must be a newcomer: {run_key}")
            if any(row.get("cohort_role") != "incumbent" for row in normal):
                raise ValueError(f"V41 normal nodes must be incumbents: {run_key}")
            if any(state != "ADMITTED" for state in states):
                raise ValueError(f"V41 incumbents must start ADMITTED: {run_key}")
            expected_target_state = (
                "ADMITTED" if run_key[2] == "access_no_limited" else "LIMITED"
            )
            if initial != expected_target_state:
                raise ValueError(
                    f"V41 target state {initial} != {expected_target_state}: {run_key}"
                )
        result = dict(zip(CONFIG, run_key))
        target_weight = sum(number(r, "aggregation_weight") for r in target_events)
        normal_weight = sum(number(r, "aggregation_weight") for r in normal_events)
        total_weight = target_weight + normal_weight
        limited_rounds = sum(
            r.get("access_state_before") == "LIMITED" for r in target_events
        )
        promotion_rounds = [
            int(float(r.get("round", 0))) for r in target_events
            if r.get("access_state_before") == "LIMITED"
            and r.get("access_state_after") == "ADMITTED"
        ]
        result.update({"stage": stage, "scenario_claim": claim, "target_node_id": target_id,
            "target_cohort_role": target.get("cohort_role", ""),
            "target_initial_access_state": initial, "target_initial_training_eligible": int(eligible),
            "target_correct_access_decision": int(correct), "target_final_access_state": target.get("access_state", ""),
            "target_completed_tasks": number(target, "completed_tasks"),
            "target_evidence_types": number(target, "evidence_types"),
            "target_evidence_mass": number(target, "evidence_mass"),
            "target_history_evidence_maturity": number(target, "history_evidence_maturity"),
            "normal_nodes": len(normal),
            "normal_initial_training_eligible_rate": sum(s in {"LIMITED", "ADMITTED"} for s in states) / len(states),
            "normal_initial_admitted_rate": sum(s == "ADMITTED" for s in states) / len(states),
            "normal_initial_limited_rate": sum(s == "LIMITED" for s in states) / len(states),
            "target_returned_updates_after_start": sum(int(float(r.get("returned", 0))) for r in target_events),
            "target_aggregated_updates_after_start": sum(int(float(r.get("aggregated", 0))) for r in target_events),
            "target_rejected_updates_after_start": sum(int(float(r.get("returned", 0))) and not int(float(r.get("aggregated", 0))) for r in target_events),
            "target_aggregation_weight_after_start": target_weight,
            "normal_aggregation_weight_after_start": normal_weight,
            "target_effective_aggregation_share_after_start": (
                target_weight / total_weight if total_weight else 0.0
            ),
            "target_vs_normal_mean_aggregation_weight_ratio": (
                target_weight / (normal_weight / len(normal)) if normal_weight else 0.0
            ),
            "target_limited_state_rounds_after_start": limited_rounds,
            "target_limited_weighted_updates_after_start": sum(
                number(r, "aggregation_weight") < 1.0 for r in target_events
                if int(float(r.get("aggregated", 0)))
            ),
            "target_promotion_round": min(promotion_rounds) if promotion_rounds else "",
            "final_accuracy": number(final, "accuracy"), "final_backdoor_asr": number(final, "backdoor_asr")})
        output.append(result)
    return output


def summarize(runs: list[dict]) -> list[dict]:
    group_fields = tuple(field for field in CONFIG if field != "repeat")
    groups = defaultdict(list)
    for row in runs: groups[tuple(str(row[field]) for field in group_fields)].append(row)
    fields = ("normal_initial_training_eligible_rate", "normal_initial_admitted_rate",
              "normal_initial_limited_rate", "target_completed_tasks", "target_evidence_types",
              "target_evidence_mass", "target_history_evidence_maturity",
              "target_returned_updates_after_start", "target_aggregated_updates_after_start",
              "target_rejected_updates_after_start", "target_aggregation_weight_after_start",
              "normal_aggregation_weight_after_start",
              "target_effective_aggregation_share_after_start",
              "target_vs_normal_mean_aggregation_weight_ratio",
              "target_limited_state_rounds_after_start",
              "target_limited_weighted_updates_after_start",
              "final_accuracy", "final_backdoor_asr")
    output = []
    for group_key, group in sorted(groups.items()):
        result = dict(zip(group_fields, group_key)); n = len(group)
        eligible = sum(int(r["target_initial_training_eligible"]) for r in group)
        correct = sum(int(r["target_correct_access_decision"]) for r in group)
        lo, hi = wilson(eligible, n)
        result.update({"stage": group[0]["stage"], "scenario_claim": group[0]["scenario_claim"],
            "independent_runs": n, "target_initial_training_eligible_rate": eligible / n,
            "target_initial_training_eligible_wilson_ci95_lower": lo,
            "target_initial_training_eligible_wilson_ci95_upper": hi,
            "target_correct_access_decision_rate": correct / n})
        for field in fields:
            values = [float(r[field]) for r in group]
            result[field + "_mean"] = statistics.fmean(values)
            result[field + "_ci95"] = t_ci(values)
        promotion_values = [
            float(r["target_promotion_round"]) for r in group
            if r["target_promotion_round"] not in {"", None}
        ]
        result["target_promotion_round_mean"] = (
            statistics.fmean(promotion_values) if promotion_values else ""
        )
        result["target_promotion_round_ci95"] = (
            t_ci(promotion_values) if promotion_values else ""
        )
        output.append(result)
    return output


def pairs(runs: list[dict]) -> list[dict]:
    controls = {}
    for row in runs:
        if row["variant"] == "access_no_limited":
            controls[(row["dataset"], row["attack_scenario"], row["noniid_alpha"], row["repeat"])] = row
    output = []
    for row in runs:
        if row["variant"] != "access_full": continue
        pair_key = (row["dataset"], row["attack_scenario"], row["noniid_alpha"], row["repeat"])
        if pair_key not in controls: continue
        control = controls[pair_key]
        full_weight = float(row["target_aggregation_weight_after_start"])
        control_weight = float(control["target_aggregation_weight_after_start"])
        full_acc, control_acc = float(row["final_accuracy"]), float(control["final_accuracy"])
        full_asr, control_asr = float(row["final_backdoor_asr"]), float(control["final_backdoor_asr"])
        output.append(dict(zip(PAIR_FIELDS[:4], pair_key)) | {
            "limited_observation_updates": row["limited_observation_updates"],
            "limited_aggregation_weight": row["limited_aggregation_weight"],
            "full_target_aggregation_weight_after_start": full_weight,
            "control_target_aggregation_weight_after_start": control_weight,
            "target_exposure_reduction": control_weight - full_weight,
            "full_final_accuracy": full_acc, "control_final_accuracy": control_acc,
            "accuracy_difference": full_acc - control_acc,
            "full_final_backdoor_asr": full_asr, "control_final_backdoor_asr": control_asr,
            "backdoor_asr_reduction": control_asr - full_asr,
            "full_target_effective_aggregation_share": row[
                "target_effective_aggregation_share_after_start"
            ],
            "control_target_effective_aggregation_share": control[
                "target_effective_aggregation_share_after_start"
            ],
            "target_share_reduction": (
                float(control["target_effective_aggregation_share_after_start"])
                - float(row["target_effective_aggregation_share_after_start"])
            ),
            "full_target_limited_state_rounds_after_start": row[
                "target_limited_state_rounds_after_start"
            ],
            "full_target_limited_weighted_updates_after_start": row[
                "target_limited_weighted_updates_after_start"
            ],
            "full_target_promotion_round": row["target_promotion_round"],
        })
    return output


def write(path: Path, rows: list[dict], fields: tuple[str, ...], allow_empty: bool = False) -> None:
    if not rows and not allow_empty: raise ValueError(f"no V41 rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    if len(sys.argv) != 8:
        raise SystemExit("usage: analyze_mixed_maturity_v41 STAGE NODES EVENTS METRICS RUNS SUMMARY PAIRS")
    stage = sys.argv[1]
    runs = analyze(stage, read(Path(sys.argv[2])), read(Path(sys.argv[3])), read(Path(sys.argv[4])))
    write(Path(sys.argv[5]), runs, RUN_FIELDS)
    write(Path(sys.argv[6]), summarize(runs), SUMMARY_FIELDS)
    write(Path(sys.argv[7]), pairs(runs), PAIR_FIELDS, allow_empty=True)


if __name__ == "__main__":
    main()

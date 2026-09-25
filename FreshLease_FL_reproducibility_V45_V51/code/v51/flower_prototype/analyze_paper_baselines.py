"""Summarize V31 attack and benign-drift system baselines by dataset."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


KEYS = ("dataset", "variant", "attack_scenario", "noniid_alpha", "repeat")
TARGETS = {
    "gradual_drift_betrayal": "gradual-drift-betrayal",
    "benign_concept_drift": "benign-concept-drift",
    "diverse_then_repeat_backdoor": "diverse-repeat-backdoor",
}
RUN_FIELDS = (
    *KEYS, "attack_start_round", "target_profile", "initial_access_state",
    "initial_training_eligible", "target_revoked", "revoked_round",
    "revocation_reason", "normal_false_revocations", "target_updates_after_start",
    "normal_node_records",
    "target_weight_after_start", "target_effective_aggregation_share_after_start",
    "target_fltrust_trust_score_mean", "target_fltrust_zero_trust_updates",
    "target_trimmed_coordinate_retention_mean", "target_validation_loss_flags",
    "normal_validation_loss_flagged_nodes", "normal_validation_loss_flagged_updates",
    "target_norm_rejected_updates", "target_norm_hard_revocations",
    "normal_norm_flagged_nodes", "normal_norm_rejected_updates",
    "normal_norm_rejected_update_rate", "normal_norm_hard_revocations",
    "normal_norm_hard_revocation_rate",
    "normal_returned_updates", "detection_delay_rounds", "final_accuracy",
    "final_backdoor_asr",
)
SUMMARY_FIELDS = (
    "dataset", "variant", "attack_scenario", "noniid_alpha", "independent_runs",
    "target_revocation_rate", "target_revocation_wilson_ci95_half_width",
    "normal_node_records", "normal_false_revocation_rate",
    "normal_false_revocation_wilson_ci95_half_width",
    "normal_validation_loss_flagged_node_rate", "normal_validation_loss_flagged_update_rate",
    "target_validation_loss_flags_mean",
    "target_norm_rejected_updates_mean", "target_norm_hard_revocations_mean",
    "normal_norm_flagged_node_rate", "normal_norm_rejected_update_rate",
    "normal_norm_hard_revocation_rate",
    "detection_delay_rounds_mean", "detection_delay_rounds_ci95",
    "target_updates_after_start_mean", "target_updates_after_start_ci95",
    "target_weight_after_start_mean", "target_weight_after_start_ci95",
    "target_effective_aggregation_share_after_start_mean",
    "target_effective_aggregation_share_after_start_ci95",
    "target_fltrust_trust_score_mean", "target_fltrust_trust_score_ci95",
    "target_fltrust_zero_trust_update_rate",
    "target_trimmed_coordinate_retention_mean",
    "target_trimmed_coordinate_retention_ci95",
    "final_accuracy_mean", "final_accuracy_ci95", "final_backdoor_asr_mean",
    "final_backdoor_asr_ci95",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.setdefault("dataset", "digits")
    return rows


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "digits" if field == "dataset" else "") for field in KEYS)


def ci95(values: list[float]) -> float:
    return 0.0 if len(values) < 2 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))


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


def analyze_runs(
    nodes: list[dict[str, str]], events: list[dict[str, str]],
    metrics: list[dict[str, str]], attack_start_round: int,
) -> list[dict]:
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
        dataset_name, variant, scenario, alpha, repeat = run_key
        if scenario not in TARGETS:
            raise ValueError(f"unexpected paper-baseline scenario: {scenario}")
        target_rows = [row for row in group if row["node_id"] == TARGETS[scenario]]
        normal = [row for row in group if row["profile"] == "honest"]
        if len(target_rows) != 1 or len(normal) < 3 or len(group) != len(normal) + 1:
            raise ValueError(f"expected one target and at least three honest nodes: {run_key}")
        target = target_rows[0]
        target_events = [
            row for row in grouped_events[run_key]
            if row["node_id"] == TARGETS[scenario]
            and int(row["round"]) >= attack_start_round
            and row["aggregated"] == "1"
        ]
        effective_shares: list[float] = []
        for event in target_events:
            explicit = event.get("aggregation_effective_weight", "").strip()
            if explicit:
                effective_shares.append(float(explicit))
                continue
            # Backward-compatible recovery for V35 FLTrust CSVs written before
            # aggregation_effective_weight was added.  Trust is normalized over
            # the actually aggregated clients in the same round.
            if event.get("aggregation_rule") == "fltrust":
                same_round = [
                    row for row in grouped_events[run_key]
                    if row.get("round") == event.get("round")
                    and row.get("aggregated") == "1"
                ]
                denominator = sum(
                    float(row.get("aggregation_trust_score") or 0.0)
                    for row in same_round
                )
                trust = float(event.get("aggregation_trust_score") or 0.0)
                effective_shares.append(trust / denominator if denominator > 0 else 0.0)
        fltrust_scores = [
            float(row["aggregation_trust_score"])
            for row in target_events
            if row.get("aggregation_trust_score", "").strip()
        ]
        trimmed_retention = [
            float(row["aggregation_coordinate_retention_rate"])
            for row in target_events
            if row.get("aggregation_coordinate_retention_rate", "").strip()
        ]
        normal_ids = {row["node_id"] for row in normal}
        normal_events = [
            row for row in grouped_events[run_key]
            if row.get("node_id") in normal_ids and row.get("returned") == "1"
        ]
        normal_flagged_events = [
            row for row in normal_events
            if row.get("validation_loss_flag") == "1"
        ]
        target_flagged_events = [
            row for row in grouped_events[run_key]
            if row.get("node_id") == TARGETS[scenario]
            and row.get("validation_loss_flag") == "1"
        ]
        rejected_norm_outcomes = {
            "MODERATE_REJECTED", "REPEATED_SHADOW_REJECTED",
            "REPEATED_REVOKED", "EXTREME_REVOKED", "NONFINITE_REVOKED",
        }
        hard_norm_outcomes = {
            "REPEATED_REVOKED", "EXTREME_REVOKED", "NONFINITE_REVOKED",
        }
        target_norm_events = [
            row for row in grouped_events[run_key]
            if row.get("node_id") == TARGETS[scenario]
            and row.get("norm_screening_outcome") in rejected_norm_outcomes
        ]
        normal_norm_events = [
            row for row in normal_events
            if row.get("norm_screening_outcome") in rejected_norm_outcomes
        ]
        revoked_round = target.get("revoked_round", "").strip()
        delay = max(0, int(revoked_round) - attack_start_round) if revoked_round else ""
        final = max(grouped_metrics[run_key], key=lambda row: int(row["round"]))
        initial_state = target["initial_access_state"]
        output.append({
            "dataset": dataset_name,
            "variant": variant,
            "attack_scenario": scenario,
            "noniid_alpha": alpha,
            "repeat": repeat,
            "attack_start_round": attack_start_round,
            "target_profile": target["profile"],
            "initial_access_state": initial_state,
            "initial_training_eligible": int(initial_state in {"LIMITED", "ADMITTED"}),
            "target_revoked": int(bool(revoked_round)),
            "revoked_round": revoked_round,
            "revocation_reason": target.get("revocation_reason", ""),
            "normal_false_revocations": sum(
                bool(row.get("revoked_round", "").strip()) for row in normal
            ),
            "normal_node_records": len(normal),
            "target_updates_after_start": len(target_events),
            "target_weight_after_start": sum(
                float(row.get("aggregation_weight") or 1.0) for row in target_events
            ),
            "target_effective_aggregation_share_after_start": (
                sum(effective_shares) if len(effective_shares) == len(target_events) else ""
            ),
            "target_fltrust_trust_score_mean": (
                statistics.fmean(fltrust_scores) if fltrust_scores else ""
            ),
            "target_fltrust_zero_trust_updates": sum(
                score <= 1e-12 for score in fltrust_scores
            ),
            "target_trimmed_coordinate_retention_mean": (
                statistics.fmean(trimmed_retention) if trimmed_retention else ""
            ),
            "target_validation_loss_flags": len(target_flagged_events),
            "normal_validation_loss_flagged_nodes": len({
                row["node_id"] for row in normal_flagged_events
            }),
            "normal_validation_loss_flagged_updates": len(normal_flagged_events),
            "target_norm_rejected_updates": len(target_norm_events),
            "target_norm_hard_revocations": sum(
                row.get("norm_screening_outcome") in hard_norm_outcomes
                for row in target_norm_events
            ),
            "normal_norm_flagged_nodes": len({
                row["node_id"] for row in normal_norm_events
            }),
            "normal_norm_rejected_updates": len(normal_norm_events),
            "normal_norm_rejected_update_rate": (
                len(normal_norm_events) / len(normal_events) if normal_events else 0.0
            ),
            "normal_norm_hard_revocations": sum(
                row.get("norm_screening_outcome") in hard_norm_outcomes
                for row in normal_norm_events
            ),
            "normal_norm_hard_revocation_rate": sum(
                row.get("norm_screening_outcome") in hard_norm_outcomes
                for row in normal_norm_events
            ) / len(normal),
            "normal_returned_updates": len(normal_events),
            "detection_delay_rounds": delay,
            "final_accuracy": final["accuracy"],
            "final_backdoor_asr": final["backdoor_asr"],
        })
    return output


def summarize(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in runs:
        grouped[tuple(str(row[field]) for field in KEYS[:4])].append(row)
    output = []
    for (dataset_name, variant, scenario, alpha), group in sorted(grouped.items()):
        revoked = sum(int(row["target_revoked"]) for row in group)
        normal_revoked = sum(int(row["normal_false_revocations"]) for row in group)
        normal_records = sum(int(row.get("normal_node_records", 3)) for row in group)
        normal_flagged_nodes = sum(int(row["normal_validation_loss_flagged_nodes"]) for row in group)
        normal_flagged_updates = sum(int(row["normal_validation_loss_flagged_updates"]) for row in group)
        normal_returned_updates = sum(int(row["normal_returned_updates"]) for row in group)
        normal_norm_flagged_nodes = sum(int(row["normal_norm_flagged_nodes"]) for row in group)
        normal_norm_rejected_updates = sum(int(row["normal_norm_rejected_updates"]) for row in group)
        normal_norm_hard_revocations = sum(int(row["normal_norm_hard_revocations"]) for row in group)
        delays = [float(row["detection_delay_rounds"]) for row in group if row["detection_delay_rounds"] != ""]
        updates = [float(row["target_updates_after_start"]) for row in group]
        weights = [float(row["target_weight_after_start"]) for row in group]
        effective_shares = [
            float(row["target_effective_aggregation_share_after_start"])
            for row in group
            if row["target_effective_aggregation_share_after_start"] != ""
        ]
        fltrust_scores = [
            float(row["target_fltrust_trust_score_mean"])
            for row in group if row["target_fltrust_trust_score_mean"] != ""
        ]
        fltrust_updates = sum(
            int(row["target_updates_after_start"])
            for row in group if row["target_fltrust_trust_score_mean"] != ""
        )
        fltrust_zero_updates = sum(
            int(row["target_fltrust_zero_trust_updates"])
            for row in group if row["target_fltrust_trust_score_mean"] != ""
        )
        trimmed_retention = [
            float(row["target_trimmed_coordinate_retention_mean"])
            for row in group
            if row["target_trimmed_coordinate_retention_mean"] != ""
        ]
        accuracies = [float(row["final_accuracy"]) for row in group]
        asrs = [float(row["final_backdoor_asr"]) for row in group]
        output.append({
            "dataset": dataset_name,
            "variant": variant,
            "attack_scenario": scenario,
            "noniid_alpha": alpha,
            "independent_runs": len(group),
            "target_revocation_rate": revoked / len(group),
            "target_revocation_wilson_ci95_half_width": wilson_half_width(revoked, len(group)),
            "normal_node_records": normal_records,
            "normal_false_revocation_rate": normal_revoked / normal_records,
            "normal_false_revocation_wilson_ci95_half_width": wilson_half_width(normal_revoked, normal_records),
            "normal_validation_loss_flagged_node_rate": normal_flagged_nodes / normal_records,
            "normal_validation_loss_flagged_update_rate": (
                normal_flagged_updates / normal_returned_updates
                if normal_returned_updates else 0.0
            ),
            "target_validation_loss_flags_mean": statistics.fmean(
                float(row["target_validation_loss_flags"]) for row in group
            ),
            "target_norm_rejected_updates_mean": statistics.fmean(
                float(row["target_norm_rejected_updates"]) for row in group
            ),
            "target_norm_hard_revocations_mean": statistics.fmean(
                float(row["target_norm_hard_revocations"]) for row in group
            ),
            "normal_norm_flagged_node_rate": normal_norm_flagged_nodes / normal_records,
            "normal_norm_rejected_update_rate": (
                normal_norm_rejected_updates / normal_returned_updates
                if normal_returned_updates else 0.0
            ),
            "normal_norm_hard_revocation_rate": normal_norm_hard_revocations / normal_records,
            "detection_delay_rounds_mean": statistics.fmean(delays) if delays else "",
            "detection_delay_rounds_ci95": ci95(delays) if delays else "",
            "target_updates_after_start_mean": statistics.fmean(updates),
            "target_updates_after_start_ci95": ci95(updates),
            "target_weight_after_start_mean": statistics.fmean(weights),
            "target_weight_after_start_ci95": ci95(weights),
            "target_effective_aggregation_share_after_start_mean": (
                statistics.fmean(effective_shares) if effective_shares else ""
            ),
            "target_effective_aggregation_share_after_start_ci95": (
                ci95(effective_shares) if effective_shares else ""
            ),
            "target_fltrust_trust_score_mean": (
                statistics.fmean(fltrust_scores) if fltrust_scores else ""
            ),
            "target_fltrust_trust_score_ci95": (
                ci95(fltrust_scores) if fltrust_scores else ""
            ),
            "target_fltrust_zero_trust_update_rate": (
                fltrust_zero_updates / fltrust_updates if fltrust_updates else ""
            ),
            "target_trimmed_coordinate_retention_mean": (
                statistics.fmean(trimmed_retention) if trimmed_retention else ""
            ),
            "target_trimmed_coordinate_retention_ci95": (
                ci95(trimmed_retention) if trimmed_retention else ""
            ),
            "final_accuracy_mean": statistics.fmean(accuracies),
            "final_accuracy_ci95": ci95(accuracies),
            "final_backdoor_asr_mean": statistics.fmean(asrs),
            "final_backdoor_asr_ci95": ci95(asrs),
        })
    return output


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    if not rows:
        raise ValueError("no V31 baseline runs to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit(
            "usage: analyze_paper_baselines NODES EVENTS METRICS RUN_OUT "
            "SUMMARY_OUT ATTACK_START_ROUND"
        )
    runs = analyze_runs(
        *(read_csv(Path(argument)) for argument in sys.argv[1:4]), int(sys.argv[6])
    )
    write_csv(Path(sys.argv[4]), runs, RUN_FIELDS)
    write_csv(Path(sys.argv[5]), summarize(runs), SUMMARY_FIELDS)


if __name__ == "__main__":
    main()

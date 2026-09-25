from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mean_available(rows: list[dict[str, str]], field: str) -> float | str:
    """Leave unavailable legacy columns blank instead of inventing measurements."""
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return statistics.fmean(values) if values else ""


def estimator(row: dict[str, str]) -> str:
    return row.get("trust_estimator", "dirichlet_lcb") or "dirichlet_lcb"


def dataset(row: dict[str, str]) -> str:
    return row.get("dataset", "digits") or "digits"


def write_access_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            dataset(row), row["variant"], estimator(row), row["attack_scenario"],
            row["noniid_alpha"], row["profile"],
        )
        grouped[key].append(row)

    output_rows = []
    for (dataset_name, variant, trust_estimator, scenario, alpha, profile), group in sorted(grouped.items()):
        initial_admitted = sum(
            row.get("initial_access_state", row["access_state"]) == "ADMITTED"
            for row in group
        )
        rate = initial_admitted / len(group)
        initial_limited = sum(
            row.get("initial_access_state", row["access_state"]) == "LIMITED"
            for row in group
        )
        eligible_rate = (initial_admitted + initial_limited) / len(group)
        final_rate = sum(row["access_state"] == "ADMITTED" for row in group) / len(group)
        final_limited_rate = sum(row["access_state"] == "LIMITED" for row in group) / len(group)
        revoked_rate = sum(bool(row.get("revoked_round", "")) for row in group) / len(group)
        z = 1.96
        denominator = 1.0 + z * z / len(group)
        half_width = (
            z
            * math.sqrt(
                rate * (1.0 - rate) / len(group)
                + z * z / (4.0 * len(group) ** 2)
            )
            / denominator
        )
        output_rows.append({
            "dataset": dataset_name,
            "variant": variant,
            "trust_estimator": trust_estimator,
            "attack_scenario": scenario,
            "noniid_alpha": alpha,
            "profile": profile,
            "runs": len(group),
            "initial_admitted_rate": rate,
            "initial_limited_rate": initial_limited / len(group),
            "initial_training_eligible_rate": eligible_rate,
            "admitted_rate": rate,
            "admitted_rate_wilson_ci95_half_width": half_width,
            "final_admitted_rate": final_rate,
            "final_limited_rate": final_limited_rate,
            "revoked_rate": revoked_rate,
            "mean_flower_fit_events": statistics.fmean(
                float(row["flower_fit_events"]) for row in group
            ),
            "mean_aggregated_fit_events": statistics.fmean(
                float(row.get("aggregated_fit_events", row["flower_fit_events"])) for row in group
            ),
            "mean_completed_tasks": mean_available(group, "completed_tasks"),
            "mean_credited_independent_tasks": mean_available(
                group, "credited_independent_tasks"
            ),
            "mean_verified_source_records": mean_available(group, "verified_source_records"),
            "mean_evidence_types": mean_available(group, "evidence_types"),
            "mean_evidence_mass": mean_available(group, "evidence_mass"),
            "mean_onboarding_duration_seconds": mean_available(
                group, "onboarding_duration_seconds"
            ),
        })

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: analyze_real_fl_results INPUT_ROUND_CSV INPUT_NODE_CSV "
            "OUTPUT_MODEL_SUMMARY_CSV OUTPUT_ACCESS_SUMMARY_CSV"
        )
    rows = read_rows(Path(sys.argv[1]))
    if not rows:
        raise SystemExit("no real-FL round metrics were found")

    final_by_run: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            dataset(row), row["variant"], estimator(row), row["attack_scenario"],
            row["noniid_alpha"], row["repeat"],
        )
        previous = final_by_run.get(key)
        if previous is None or int(row["round"]) > int(previous["round"]):
            final_by_run[key] = row

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for (dataset_name, variant, trust_estimator, scenario, alpha, _), row in final_by_run.items():
        grouped[(dataset_name, variant, trust_estimator, scenario, alpha)].append(row)

    output_rows = []
    for (dataset_name, variant, trust_estimator, scenario, alpha), group in sorted(grouped.items()):
        accuracies = [float(row["accuracy"]) for row in group]
        losses = [float(row["loss"]) for row in group]
        attack_rates = [float(row["backdoor_asr"]) for row in group]
        output_rows.append({
            "dataset": dataset_name,
            "variant": variant,
            "trust_estimator": trust_estimator,
            "attack_scenario": scenario,
            "noniid_alpha": alpha,
            "runs": len(group),
            "final_accuracy_mean": statistics.fmean(accuracies),
            "final_accuracy_ci95": ci95(accuracies),
            "final_loss_mean": statistics.fmean(losses),
            "final_loss_ci95": ci95(losses),
            "final_backdoor_asr_mean": statistics.fmean(attack_rates),
            "final_backdoor_asr_ci95": ci95(attack_rates),
        })

    output_path = Path(sys.argv[3])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    write_access_summary(read_rows(Path(sys.argv[2])), Path(sys.argv[4]))


if __name__ == "__main__":
    main()

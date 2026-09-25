from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ESTIMATORS = ("dirichlet_mean", "beta_mean")
NUMERIC_METRICS = (
    "initial_training_eligible",
    "final_admitted",
    "revoked",
    "trust_score",
    "history_mean",
    "history_std",
    "completed_tasks",
    "evidence_mass",
    "onboarding_duration_seconds",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def derived(row: dict[str, str]) -> dict[str, float]:
    initial = row.get("initial_access_state", row.get("access_state", ""))
    return {
        "initial_training_eligible": float(initial in {"LIMITED", "ADMITTED"}),
        "final_admitted": float(row.get("access_state") == "ADMITTED"),
        "revoked": float(bool(row.get("revoked_round", ""))),
        "trust_score": float(row["trust_score"]),
        "history_mean": float(row["history_mean"]),
        "history_std": float(row["history_std"]),
        "completed_tasks": float(row["completed_tasks"]),
        "evidence_mass": float(row["evidence_mass"]),
        "onboarding_duration_seconds": float(row["onboarding_duration_seconds"]),
    }


def aggregate_run_profile(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        estimator = row.get("trust_estimator", "")
        if estimator not in ESTIMATORS:
            continue
        grouped[(
            estimator, row["attack_scenario"], row["noniid_alpha"],
            row["repeat"], row["profile"],
        )].append(row)

    output: list[dict[str, str | float]] = []
    for key, group in sorted(grouped.items()):
        estimator, scenario, alpha, repeat, profile = key
        values = [derived(row) for row in group]
        item: dict[str, str | float] = {
            "trust_estimator": estimator,
            "attack_scenario": scenario,
            "noniid_alpha": alpha,
            "repeat": repeat,
            "profile": profile,
            "node_records": len(group),
        }
        for metric in NUMERIC_METRICS:
            item[metric] = statistics.fmean(value[metric] for value in values)
        output.append(item)
    return output


def summarize(run_rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str | float]]] = defaultdict(list)
    for row in run_rows:
        grouped[(
            str(row["trust_estimator"]), str(row["attack_scenario"]),
            str(row["noniid_alpha"]), str(row["profile"]),
        )].append(row)

    output: list[dict[str, str | float]] = []
    for key, group in sorted(grouped.items()):
        estimator, scenario, alpha, profile = key
        item: dict[str, str | float] = {
            "trust_estimator": estimator,
            "attack_scenario": scenario,
            "noniid_alpha": alpha,
            "profile": profile,
            "independent_repeats": len(group),
            "node_records": sum(int(row["node_records"]) for row in group),
        }
        for metric in NUMERIC_METRICS:
            samples = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.fmean(samples)
            item[f"{metric}_ci95"] = ci95(samples)
        output.append(item)
    return output


def paired_effects(run_rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    indexed = {
        (
            str(row["trust_estimator"]), str(row["attack_scenario"]),
            str(row["noniid_alpha"]), str(row["repeat"]), str(row["profile"]),
        ): row
        for row in run_rows
    }
    groups: dict[tuple[str, str, str], list[tuple[dict, dict]]] = defaultdict(list)
    for key, left in indexed.items():
        estimator, scenario, alpha, repeat, profile = key
        if estimator != "dirichlet_mean":
            continue
        right = indexed.get(("beta_mean", scenario, alpha, repeat, profile))
        if right is not None:
            groups[(scenario, alpha, profile)].append((left, right))

    output: list[dict[str, str | float]] = []
    for (scenario, alpha, profile), pairs in sorted(groups.items()):
        for metric in NUMERIC_METRICS:
            differences = [float(left[metric]) - float(right[metric]) for left, right in pairs]
            output.append({
                "attack_scenario": scenario,
                "noniid_alpha": alpha,
                "profile": profile,
                "metric": metric,
                "independent_pairs": len(differences),
                "dirichlet_minus_beta_mean": statistics.fmean(differences),
                "difference_ci95": ci95(differences),
            })
    return output


def write_rows(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        raise ValueError(f"no rows available for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: analyze_trust_estimator_comparison NODE_CSV RUN_CSV "
            "SUMMARY_CSV PAIRED_EFFECTS_CSV"
        )
    run_rows = aggregate_run_profile(read_rows(Path(sys.argv[1])))
    write_rows(Path(sys.argv[2]), run_rows)
    write_rows(Path(sys.argv[3]), summarize(run_rows))
    write_rows(Path(sys.argv[4]), paired_effects(run_rows))


if __name__ == "__main__":
    main()

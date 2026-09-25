from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


NUMERIC_FIELDS = (
    "admitted_rate", "quarantine_rate", "mean_completed_tasks",
    "mean_onboarding_duration_seconds", "mean_flower_fit_events",
)
EXPECTED_NODES_PER_RUN = 6
EXPECTED_FLOWER_FIT_EVENTS = 20


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def ci95(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return float("nan")
    sample_mean = statistics.fmean(finite)
    variance = sum((value - sample_mean) ** 2 for value in finite) / (len(finite) - 1)
    return 1.96 * math.sqrt(variance) / math.sqrt(len(finite))


def valid_run(rows: list[dict[str, str]]) -> tuple[bool, str]:
    if len(rows) != EXPECTED_NODES_PER_RUN:
        return False, f"expected {EXPECTED_NODES_PER_RUN} nodes, found {len(rows)}"
    if any(not row.get("onboarding_duration_seconds") for row in rows):
        return False, "missing onboarding completion"
    for row in rows:
        fits = int(float(row["flower_fit_events"]))
        admitted = row["access_state"] == "ADMITTED"
        if admitted and fits != EXPECTED_FLOWER_FIT_EVENTS:
            return False, f"admitted node {row['node_id']} has {fits} FIT events"
        if not admitted and fits != 0:
            return False, f"non-admitted node {row['node_id']} has {fits} FIT events"
    return True, "ok"


def read_per_repeat(
    path: Path,
) -> tuple[list[dict[str, float | str | int]], list[dict[str, str | int]]]:
    run_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            run_groups[(row["variant"], int(row["repeat"]))].append(row)
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    invalid_runs = []
    for (variant, repeat), run_rows in sorted(run_groups.items()):
        valid, reason = valid_run(run_rows)
        if not valid:
            invalid_runs.append({"variant": variant, "repeat": repeat, "reason": reason})
            continue
        for row in run_rows:
            grouped[(variant, row["profile"], repeat)].append(row)
    results = []
    for (variant, profile, repeat), rows in sorted(grouped.items()):
        durations = [
            float(row["onboarding_duration_seconds"])
            for row in rows if row.get("onboarding_duration_seconds") not in {None, ""}
        ]
        results.append({
            "variant": variant,
            "profile": profile,
            "repeat": repeat,
            "admitted_rate": mean([
                float(row["access_state"] == "ADMITTED") for row in rows
            ]),
            "quarantine_rate": mean([
                float(row["access_state"] == "QUARANTINE") for row in rows
            ]),
            "mean_completed_tasks": mean([
                float(row["completed_tasks"]) for row in rows
            ]),
            "mean_onboarding_duration_seconds": mean(durations),
            "mean_flower_fit_events": mean([
                float(row["flower_fit_events"]) for row in rows
            ]),
        })
    return results, invalid_runs


def write_validity(rows: list[dict[str, str | int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "repeat", "reason"))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["profile"])].append(row)
    output = []
    for (variant, profile), values in sorted(grouped.items()):
        item = {"variant": variant, "profile": profile, "repeats": len(values)}
        for field in NUMERIC_FIELDS:
            samples = [float(row[field]) for row in values]
            item[f"{field}_mean"] = mean(samples)
            item[f"{field}_ci95"] = ci95(samples)
        output.append(item)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def write_paired(rows: list[dict], path: Path) -> None:
    indexed = {
        (row["variant"], row["profile"], row["repeat"]): row for row in rows
    }
    comparisons = (
        ("single_type_farming", "no_diversity", "type coverage"),
        ("diverse_then_repeat_farming", "no_repeat_decay", "repeat decay"),
        ("attestation_camouflage", "no_semantic_separation", "semantic separation"),
        ("single_type_farming", "no_budget", "bounded probation"),
    )
    output = []
    repeats = sorted({int(row["repeat"]) for row in rows})
    for profile, ablation, component in comparisons:
        for metric in NUMERIC_FIELDS:
            differences = []
            for repeat in repeats:
                full = indexed.get(("full", profile, repeat))
                other = indexed.get((ablation, profile, repeat))
                if full is not None and other is not None:
                    differences.append(float(full[metric]) - float(other[metric]))
            output.append({
                "component": component,
                "profile": profile,
                "ablation": ablation,
                "metric": metric,
                "pairs": len(differences),
                "full_minus_ablation_mean": mean(differences),
                "difference_ci95": ci95(differences),
                "same_direction_fraction": (
                    max(
                        sum(value > 0 for value in differences),
                        sum(value < 0 for value in differences),
                    ) / len(differences) if differences else float("nan")
                ),
            })
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: analyze_container_results INPUT SUMMARY PAIRED_EFFECTS"
        )
    rows, invalid_runs = read_per_repeat(Path(sys.argv[1]))
    if not rows:
        raise SystemExit("no node results found")
    summary_path = Path(sys.argv[2])
    write_summary(rows, summary_path)
    write_paired(rows, Path(sys.argv[3]))
    validity_path = summary_path.with_name("invalid_runs.csv")
    write_validity(invalid_runs, validity_path)
    if invalid_runs:
        print(f"excluded {len(invalid_runs)} incomplete run(s); see {validity_path}")


if __name__ == "__main__":
    main()

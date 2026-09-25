"""V37 reporting for robust aggregation under explicit cohort initial conditions."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from flower_prototype import analyze_paper_baselines as base


def _run_fields() -> tuple[str, ...]:
    fields: list[str] = []
    for field in base.RUN_FIELDS:
        if field == "repeat":
            fields.extend(("cohort_mode", "repeat", "target_cohort_role"))
        elif field == "target_updates_after_start":
            fields.extend((
                "target_returned_updates_after_start",
                "target_aggregated_updates_after_start",
                "target_rejected_updates_after_start",
            ))
        else:
            fields.append(field)
    return tuple(fields)


def _summary_fields() -> tuple[str, ...]:
    fields: list[str] = []
    for field in base.SUMMARY_FIELDS:
        if field == "independent_runs":
            fields.extend(("cohort_mode", "independent_runs"))
        elif field == "target_revocation_wilson_ci95_half_width":
            fields.extend((
                "target_revocation_wilson_ci95_lower",
                "target_revocation_wilson_ci95_upper",
            ))
        elif field == "normal_false_revocation_wilson_ci95_half_width":
            fields.extend((
                "normal_false_revocation_wilson_ci95_lower",
                "normal_false_revocation_wilson_ci95_upper",
            ))
        elif field == "detection_delay_rounds_ci95":
            fields.extend((field, "detection_delay_observed_runs"))
        elif field == "target_updates_after_start_mean":
            fields.extend((
                "target_returned_updates_after_start_mean",
                "target_aggregated_updates_after_start_mean",
                "target_rejected_updates_after_start_mean",
            ))
        elif field == "target_updates_after_start_ci95":
            fields.extend((
                "target_returned_updates_after_start_ci95",
                "target_aggregated_updates_after_start_ci95",
                "target_rejected_updates_after_start_ci95",
            ))
        else:
            fields.append(field)
    return tuple(fields)


RUN_FIELDS = _run_fields()
SUMMARY_FIELDS = _summary_fields()


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float | str, float | str]:
    if trials < 1:
        return "", ""
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def ci95_or_blank(values: list[float]) -> float | str:
    if len(values) < 2:
        return ""
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def read_csv(path: Path) -> list[dict[str, str]]:
    rows = base.read_csv(path)
    for row in rows:
        row.setdefault("cohort_mode", "synchronous_cold_start")
        if not row.get("cohort_role"):
            row["cohort_role"] = "newcomer"
    return rows


def _identity(row: dict) -> tuple[str, str, str, str, str]:
    return tuple(str(row[field]) for field in base.KEYS)  # type: ignore[return-value]


def analyze_runs(
    nodes: list[dict[str, str]], events: list[dict[str, str]],
    metrics: list[dict[str, str]], attack_start_round: int,
) -> list[dict]:
    cohort_modes = {row.get("cohort_mode", "synchronous_cold_start") for row in nodes}
    if len(cohort_modes) != 1:
        raise ValueError("V37 analyzer expects one cohort mode per result directory")
    cohort_mode = next(iter(cohort_modes))
    base_runs = base.analyze_runs(nodes, events, metrics, attack_start_round)
    event_groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    node_groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in events:
        event_groups[_identity(row)].append(row)
    for row in nodes:
        node_groups[_identity(row)].append(row)

    output: list[dict] = []
    for row in base_runs:
        identity = _identity(row)
        target_id = base.TARGETS[str(row["attack_scenario"])]
        target_nodes = [item for item in node_groups[identity] if item["node_id"] == target_id]
        if len(target_nodes) != 1:
            raise ValueError(f"expected one V37 target node: {identity}")
        returned = [
            item for item in event_groups[identity]
            if item.get("node_id") == target_id
            and int(item["round"]) >= attack_start_round
            and item.get("returned") == "1"
        ]
        aggregated = [item for item in returned if item.get("aggregated") == "1"]
        enriched = dict(row)
        enriched["cohort_mode"] = cohort_mode
        enriched["target_cohort_role"] = (
            target_nodes[0].get("cohort_role") or "newcomer"
        )
        enriched["target_returned_updates_after_start"] = len(returned)
        enriched["target_aggregated_updates_after_start"] = len(aggregated)
        enriched["target_rejected_updates_after_start"] = len(returned) - len(aggregated)
        enriched.pop("target_updates_after_start", None)
        output.append(enriched)
    return output


def summarize(runs: list[dict]) -> list[dict]:
    cohort_modes = {str(row["cohort_mode"]) for row in runs}
    if len(cohort_modes) != 1:
        raise ValueError("V37 summary expects one cohort mode per result directory")
    cohort_mode = next(iter(cohort_modes))
    legacy_runs = []
    for row in runs:
        legacy = dict(row)
        legacy["target_updates_after_start"] = row["target_aggregated_updates_after_start"]
        legacy_runs.append(legacy)
    base_summary = base.summarize(legacy_runs)
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in runs:
        grouped[tuple(str(row[field]) for field in base.KEYS[:4])].append(row)

    output: list[dict] = []
    for row in base_summary:
        group = grouped[tuple(str(row[field]) for field in base.KEYS[:4])]
        result = dict(row)
        result["cohort_mode"] = cohort_mode
        target_successes = sum(int(item["target_revoked"]) for item in group)
        target_lower, target_upper = wilson_interval(target_successes, len(group))
        normal_records = sum(int(item["normal_node_records"]) for item in group)
        normal_failures = sum(int(item["normal_false_revocations"]) for item in group)
        normal_lower, normal_upper = wilson_interval(normal_failures, normal_records)
        result["target_revocation_wilson_ci95_lower"] = target_lower
        result["target_revocation_wilson_ci95_upper"] = target_upper
        result["normal_false_revocation_wilson_ci95_lower"] = normal_lower
        result["normal_false_revocation_wilson_ci95_upper"] = normal_upper
        result.pop("target_revocation_wilson_ci95_half_width", None)
        result.pop("normal_false_revocation_wilson_ci95_half_width", None)

        for prefix in ("returned", "aggregated", "rejected"):
            values = [
                float(item[f"target_{prefix}_updates_after_start"])
                for item in group
            ]
            result[f"target_{prefix}_updates_after_start_mean"] = statistics.fmean(values)
            result[f"target_{prefix}_updates_after_start_ci95"] = ci95_or_blank(values)
        result.pop("target_updates_after_start_mean", None)
        result.pop("target_updates_after_start_ci95", None)
        result["detection_delay_observed_runs"] = sum(
            item["detection_delay_rounds"] != "" for item in group
        )

        if len(group) < 2:
            for field in list(result):
                if field.endswith("_ci95"):
                    result[field] = ""
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    if not rows:
        raise ValueError("no V37 hybrid runs to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit(
            "usage: analyze_hybrid_v37 NODES EVENTS METRICS RUN_OUT "
            "SUMMARY_OUT ATTACK_START_ROUND"
        )
    nodes, events, metrics = (read_csv(Path(argument)) for argument in sys.argv[1:4])
    runs = analyze_runs(nodes, events, metrics, int(sys.argv[6]))
    write_csv(Path(sys.argv[4]), runs, RUN_FIELDS)
    write_csv(Path(sys.argv[5]), summarize(runs), SUMMARY_FIELDS)


if __name__ == "__main__":
    main()

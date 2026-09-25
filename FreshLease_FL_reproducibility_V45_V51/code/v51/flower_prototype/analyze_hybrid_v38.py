"""V38 hybrid reporting with explicit mode labels and Student-t intervals."""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import t as student_t

from flower_prototype import analyze_hybrid_v37 as v37
from flower_prototype import analyze_paper_baselines as base


def _insert_after(fields: tuple[str, ...], anchor: str, value: str) -> tuple[str, ...]:
    output: list[str] = []
    for field in fields:
        output.append(field)
        if field == anchor:
            output.append(value)
    return tuple(output)


RUN_FIELDS = _insert_after(v37.RUN_FIELDS, "cohort_mode", "norm_escalation_mode")
SUMMARY_FIELDS = _insert_after(
    v37.SUMMARY_FIELDS, "cohort_mode", "norm_escalation_mode",
)


def t_ci95_or_blank(values: list[float]) -> float | str:
    """Return the two-sided Student-t 95% half-width for independent runs."""
    if len(values) < 2:
        return ""
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    return critical * statistics.stdev(values) / len(values) ** 0.5


def _identity(row: dict) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in base.KEYS)


def analyze_runs(
    nodes: list[dict[str, str]], events: list[dict[str, str]],
    metrics: list[dict[str, str]], attack_start_round: int,
) -> list[dict]:
    runs = v37.analyze_runs(nodes, events, metrics, attack_start_round)
    modes: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in events:
        modes[_identity(row)].add(row.get("norm_escalation_mode") or "enforce")
    for row in runs:
        observed = modes[_identity(row)] or {"enforce"}
        if len(observed) != 1:
            raise ValueError(f"mixed norm escalation modes in one run: {_identity(row)}")
        row["norm_escalation_mode"] = next(iter(observed))
        if not row.get("target_cohort_role"):
            row["target_cohort_role"] = "newcomer"
    return runs


CI_SOURCE_FIELDS = {
    "detection_delay_rounds_ci95": "detection_delay_rounds",
    "target_returned_updates_after_start_ci95": (
        "target_returned_updates_after_start"
    ),
    "target_aggregated_updates_after_start_ci95": (
        "target_aggregated_updates_after_start"
    ),
    "target_rejected_updates_after_start_ci95": (
        "target_rejected_updates_after_start"
    ),
    "target_weight_after_start_ci95": "target_weight_after_start",
    "target_effective_aggregation_share_after_start_ci95": (
        "target_effective_aggregation_share_after_start"
    ),
    "target_fltrust_trust_score_ci95": "target_fltrust_trust_score_mean",
    "target_trimmed_coordinate_retention_ci95": (
        "target_trimmed_coordinate_retention_mean"
    ),
    "final_accuracy_ci95": "final_accuracy",
    "final_backdoor_asr_ci95": "final_backdoor_asr",
}


def summarize(runs: list[dict]) -> list[dict]:
    output = v37.summarize(runs)
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in runs:
        grouped[tuple(str(row[field]) for field in base.KEYS[:4])].append(row)
    for result in output:
        key = tuple(str(result[field]) for field in base.KEYS[:4])
        group = grouped[key]
        modes = {str(row["norm_escalation_mode"]) for row in group}
        if len(modes) != 1:
            raise ValueError(f"mixed norm escalation modes in summary group: {key}")
        result["norm_escalation_mode"] = next(iter(modes))
        for ci_field, source_field in CI_SOURCE_FIELDS.items():
            values = [
                float(row[source_field]) for row in group
                if row.get(source_field, "") not in {"", None}
            ]
            result[ci_field] = t_ci95_or_blank(values)
    return output


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit(
            "usage: analyze_hybrid_v38 NODES EVENTS METRICS RUN_OUT "
            "SUMMARY_OUT ATTACK_START_ROUND"
        )
    nodes, events, metrics = (
        v37.read_csv(Path(argument)) for argument in sys.argv[1:4]
    )
    runs = analyze_runs(nodes, events, metrics, int(sys.argv[6]))
    v37.write_csv(Path(sys.argv[4]), runs, RUN_FIELDS)
    v37.write_csv(Path(sys.argv[5]), summarize(runs), SUMMARY_FIELDS)


if __name__ == "__main__":
    main()

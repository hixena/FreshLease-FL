"""Summarize persistence of ordinary norm excesses in the V38 shadow matrix."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from flower_prototype import analyze_hybrid_v37 as v37
from flower_prototype import analyze_hybrid_v38 as v38
from flower_prototype import analyze_paper_baselines as base


MODERATE_OUTCOMES = {
    "MODERATE_REJECTED", "REPEATED_SHADOW_REJECTED", "REPEATED_REVOKED",
}
REPEATED_OUTCOMES = {"REPEATED_SHADOW_REJECTED", "REPEATED_REVOKED"}
HARD_NORM_OUTCOMES = {
    "REPEATED_REVOKED", "EXTREME_REVOKED", "NONFINITE_REVOKED",
}

NODE_FIELDS = (
    "dataset", "variant", "attack_scenario", "noniid_alpha", "cohort_mode",
    "repeat", "norm_escalation_mode", "node_id", "evaluation_role", "profile",
    "cohort_role", "initial_access_state", "final_access_state",
    "returned_updates", "moderate_excess_updates", "repeated_shadow_rejections",
    "extreme_or_nonfinite_events", "hard_norm_revocations",
    "max_consecutive_moderate_excess", "first_excess_round", "last_excess_round",
    "moderate_excess_update_rate", "mean_excess_norm_ratio", "max_norm_ratio",
)

SUMMARY_FIELDS = (
    "dataset", "variant", "attack_scenario", "noniid_alpha", "cohort_mode",
    "norm_escalation_mode", "evaluation_role", "independent_runs", "nodes",
    "returned_updates", "moderate_excess_updates", "moderate_excess_update_rate",
    "nodes_with_repeat_shadow", "repeat_shadow_node_rate",
    "repeat_shadow_wilson_ci95_lower", "repeat_shadow_wilson_ci95_upper",
    "max_consecutive_moderate_excess_mean",
    "max_consecutive_moderate_excess_ci95", "hard_norm_revocations",
    "hard_norm_revocation_rate", "max_norm_ratio_mean", "max_norm_ratio_ci95",
)


def _key(row: dict) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in base.KEYS)


def _cohort_role(node: dict, cohort_mode: str, evaluation_role: str) -> str:
    if node.get("cohort_role"):
        return str(node["cohort_role"])
    if cohort_mode == "mixed_maturity":
        return "newcomer" if evaluation_role == "target" else "incumbent"
    return "newcomer"


def analyze_nodes(
    nodes: list[dict[str, str]], events: list[dict[str, str]],
) -> list[dict]:
    grouped_events: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for event in events:
        if event.get("node_id"):
            grouped_events[(*_key(event), str(event["node_id"]))].append(event)
    output: list[dict] = []
    for node in nodes:
        scenario = str(node["attack_scenario"])
        node_id = str(node["node_id"])
        evaluation_role = (
            "target" if node_id == base.TARGETS[scenario] else "normal"
        )
        observed = grouped_events.get((*_key(node), node_id), [])
        returned = [row for row in observed if row.get("returned") == "1"]
        moderate = [
            row for row in returned
            if row.get("norm_screening_outcome") in MODERATE_OUTCOMES
        ]
        repeated = [
            row for row in returned
            if row.get("norm_screening_outcome") in REPEATED_OUTCOMES
        ]
        extreme_or_nonfinite = [
            row for row in returned
            if row.get("norm_screening_outcome") in {
                "EXTREME_REVOKED", "NONFINITE_REVOKED",
            }
        ]
        hard = [
            row for row in returned
            if row.get("norm_screening_outcome") in HARD_NORM_OUTCOMES
        ]
        strikes = [
            int(row["norm_strike_count"]) for row in moderate
            if row.get("norm_strike_count") not in {"", None}
        ]
        ratios = [
            float(row["norm_ratio"]) for row in moderate
            if row.get("norm_ratio") not in {"", None}
        ]
        rounds = [int(row["round"]) for row in moderate]
        modes = {
            row.get("norm_escalation_mode") or "enforce" for row in returned
        }
        if not modes:
            modes = {node.get("norm_escalation_mode") or "enforce"}
        if len(modes) != 1:
            raise ValueError(f"mixed norm modes for {_key(node)} node={node_id}")
        cohort_mode = str(node.get("cohort_mode") or "synchronous_cold_start")
        output.append({
            **{field: node[field] for field in base.KEYS},
            "cohort_mode": cohort_mode,
            "norm_escalation_mode": next(iter(modes)),
            "node_id": node_id,
            "evaluation_role": evaluation_role,
            "profile": node.get("profile", ""),
            "cohort_role": _cohort_role(node, cohort_mode, evaluation_role),
            "initial_access_state": node.get("initial_access_state", ""),
            "final_access_state": node.get("access_state", ""),
            "returned_updates": len(returned),
            "moderate_excess_updates": len(moderate),
            "repeated_shadow_rejections": len(repeated),
            "extreme_or_nonfinite_events": len(extreme_or_nonfinite),
            "hard_norm_revocations": len(hard),
            "max_consecutive_moderate_excess": max(strikes, default=0),
            "first_excess_round": min(rounds) if rounds else "",
            "last_excess_round": max(rounds) if rounds else "",
            "moderate_excess_update_rate": (
                len(moderate) / len(returned) if returned else 0.0
            ),
            "mean_excess_norm_ratio": statistics.fmean(ratios) if ratios else "",
            "max_norm_ratio": max(ratios) if ratios else "",
        })
    return output


def summarize(rows: list[dict]) -> list[dict]:
    group_fields = (
        "dataset", "variant", "attack_scenario", "noniid_alpha", "cohort_mode",
        "norm_escalation_mode", "evaluation_role",
    )
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    output: list[dict] = []
    for key, group in sorted(grouped.items()):
        returned = sum(int(row["returned_updates"]) for row in group)
        moderate = sum(int(row["moderate_excess_updates"]) for row in group)
        repeat_nodes = sum(
            int(row["repeated_shadow_rejections"]) > 0 for row in group
        )
        hard_nodes = sum(int(row["hard_norm_revocations"]) > 0 for row in group)
        repeat_lower, repeat_upper = v37.wilson_interval(repeat_nodes, len(group))
        max_strikes = [
            float(row["max_consecutive_moderate_excess"]) for row in group
        ]
        max_ratios = [
            float(row["max_norm_ratio"]) for row in group
            if row["max_norm_ratio"] not in {"", None}
        ]
        result = dict(zip(group_fields, key))
        result.update({
            "independent_runs": len({str(row["repeat"]) for row in group}),
            "nodes": len(group),
            "returned_updates": returned,
            "moderate_excess_updates": moderate,
            "moderate_excess_update_rate": moderate / returned if returned else 0.0,
            "nodes_with_repeat_shadow": repeat_nodes,
            "repeat_shadow_node_rate": repeat_nodes / len(group),
            "repeat_shadow_wilson_ci95_lower": repeat_lower,
            "repeat_shadow_wilson_ci95_upper": repeat_upper,
            "max_consecutive_moderate_excess_mean": statistics.fmean(max_strikes),
            "max_consecutive_moderate_excess_ci95": v38.t_ci95_or_blank(max_strikes),
            "hard_norm_revocations": hard_nodes,
            "hard_norm_revocation_rate": hard_nodes / len(group),
            "max_norm_ratio_mean": statistics.fmean(max_ratios) if max_ratios else "",
            "max_norm_ratio_ci95": v38.t_ci95_or_blank(max_ratios),
        })
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    if not rows:
        raise ValueError("no V38 norm-shadow rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: analyze_norm_shadow_v38 NODES EVENTS NODE_OUT SUMMARY_OUT"
        )
    nodes = v37.read_csv(Path(sys.argv[1]))
    events = v37.read_csv(Path(sys.argv[2]))
    rows = analyze_nodes(nodes, events)
    write_csv(Path(sys.argv[3]), rows, NODE_FIELDS)
    write_csv(Path(sys.argv[4]), summarize(rows), SUMMARY_FIELDS)


if __name__ == "__main__":
    main()

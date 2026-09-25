"""Auditable two-stage outcome for delayed, initially honest node attacks."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


KEYS = ("variant", "attack_scenario", "noniid_alpha", "repeat")
TARGETS = {
    "diverse_then_repeat_farming": "diverse-repeat-farming",
    "diverse_then_repeat_backdoor": "diverse-repeat-backdoor",
    "gradual_drift_betrayal": "gradual-drift-betrayal",
}
RUN_FIELDS = (
    *KEYS, "attack_start_round", "initial_access_state", "initial_admitted",
    "initial_limited", "initial_training_eligible", "completed_tasks",
    "credited_independent_tasks", "first_attack_returned",
    "first_attack_signed", "first_attack_aggregated", "first_attack_reason",
    "revoked_round", "revocation_reason", "revoked_before_attack", "normal_initial_admitted",
    "normal_initial_limited", "normal_initial_training_eligible",
    "normal_revoked", "normal_aggregated_updates", "attack_aggregated_updates",
    "normal_aggregated_weight_mass", "attack_aggregated_weight_mass",
    "attack_aggregated_after_start", "detection_delay_rounds",
    "attack_aggregated_weight_after_start",
    "final_accuracy", "final_backdoor_asr",
)
PAIR_FIELDS = (
    "attack_scenario", "noniid_alpha", "repeat", "attack_start_round",
    "full_initial_admitted", "control_initial_admitted",
    "full_first_attack_returned", "control_first_attack_returned",
    "full_first_attack_aggregated", "control_first_attack_aggregated",
    "full_first_attack_reason", "control_first_attack_reason",
    "full_revoked_round", "control_revoked_round",
    "full_normal_initial_admitted", "control_normal_initial_admitted",
    "full_normal_revoked", "control_normal_revoked",
    "full_attack_aggregated_updates", "control_attack_aggregated_updates",
    "full_final_accuracy", "control_final_accuracy",
    "full_final_backdoor_asr", "control_final_backdoor_asr",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    if not rows:
        raise ValueError("no complete experimental runs to analyze")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in KEYS)


def analyze_runs(nodes: list[dict[str, str]], events: list[dict[str, str]],
                 metrics: list[dict[str, str]], attack_start_round: int) -> list[dict]:
    if attack_start_round < 2:
        raise ValueError("delayed-betrayal attack_start_round must be at least 2")
    grouped_nodes: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    grouped_events: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    grouped_metrics: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in nodes:
        grouped_nodes[run_key(row)].append(row)
    for row in events:
        if row["aggregated"] == "1" and row["returned"] != "1":
            raise ValueError(f"aggregation recorded without returned update: {run_key(row)}")
        grouped_events[run_key(row)].append(row)
    for row in metrics:
        grouped_metrics[run_key(row)].append(row)

    runs: list[dict] = []
    for key, group in sorted(grouped_nodes.items()):
        variant, scenario, alpha, repeat = key
        if variant not in {
            "full", "no_online_revalidation", "progressive_full",
            "progressive_no_cumulative",
        } or scenario not in TARGETS:
            raise ValueError(f"unexpected variant/scenario in delayed experiment: {key}")
        target_rows = [node for node in group if node["node_id"] == TARGETS[scenario]]
        normal = [node for node in group if node["profile"] == "honest"]
        if len(group) != 4 or len(target_rows) != 1 or len(normal) != 3:
            raise ValueError(f"expected one target and three honest nodes: {key}")
        target = target_rows[0]
        rounds = grouped_metrics[key]
        if not rounds or max(int(m["round"]) for m in rounds) < attack_start_round:
            raise ValueError(f"missing attack-start model round: {key}")
        end = max(rounds, key=lambda m: int(m["round"]))
        target_events = [row for row in grouped_events[key]
                         if row["node_id"] == target["node_id"]
                         and int(row["round"]) == attack_start_round]
        initial_state = target["initial_access_state"]
        initial = initial_state == "ADMITTED"
        eligible = initial_state in {"LIMITED", "ADMITTED"}
        revoked = target.get("revoked_round", "").strip()
        revoked_before_attack = bool(revoked and int(revoked) < attack_start_round)
        if eligible and not revoked_before_attack and len(target_events) != 1:
            raise ValueError(f"admitted target missing first attack event: {key}")
        if len(target_events) > 1:
            raise ValueError(f"duplicate first attack event: {key}")
        event = target_events[0] if target_events else None
        if event and event["aggregated"] == "1" and event["returned"] != "1":
            raise ValueError(f"aggregation recorded without returned update: {key}")
        # Missing update / prior revocation cannot be credited as detection.
        observed = event is not None and event["returned"] == "1"
        all_target_events = [row for row in grouped_events[key]
                             if row["node_id"] == target["node_id"]]
        aggregated_after_start = sum(
            row["aggregated"] == "1" and int(row["round"]) >= attack_start_round
            for row in all_target_events
        )
        aggregated_weight_after_start = sum(
            float(row.get("aggregation_weight") or 1.0)
            for row in all_target_events
            if row["aggregated"] == "1" and int(row["round"]) >= attack_start_round
        )
        runs.append(dict(zip(KEYS, key), **{
            "attack_start_round": attack_start_round,
            "initial_access_state": initial_state,
            "initial_admitted": int(initial),
            "initial_limited": int(initial_state == "LIMITED"),
            "initial_training_eligible": int(eligible),
            "completed_tasks": target["completed_tasks"],
            "credited_independent_tasks": target.get("credited_independent_tasks", ""),
            "first_attack_returned": int(observed),
            "first_attack_signed": int(event["signature_verified"] == "1") if observed else "",
            "first_attack_aggregated": int(event["aggregated"] == "1") if observed else "",
            "first_attack_reason": event["reason"] if event else "",
            "revoked_round": revoked,
            "revocation_reason": target.get("revocation_reason", ""),
            "revoked_before_attack": int(revoked_before_attack),
            "normal_initial_admitted": sum(n["initial_access_state"] == "ADMITTED" for n in normal),
            "normal_initial_limited": sum(n["initial_access_state"] == "LIMITED" for n in normal),
            "normal_initial_training_eligible": sum(
                n["initial_access_state"] in {"LIMITED", "ADMITTED"} for n in normal
            ),
            "normal_revoked": sum(bool(n.get("revoked_round", "").strip()) for n in normal),
            "normal_aggregated_updates": sum(int(n["aggregated_fit_events"]) for n in normal),
            "attack_aggregated_updates": int(target["aggregated_fit_events"]),
            "normal_aggregated_weight_mass": sum(
                float(n.get("aggregated_weight_mass", n["aggregated_fit_events"]))
                for n in normal
            ),
            "attack_aggregated_weight_mass": float(
                target.get("aggregated_weight_mass", target["aggregated_fit_events"])
            ),
            "attack_aggregated_after_start": aggregated_after_start,
            "detection_delay_rounds": (
                int(revoked) - attack_start_round if revoked else ""
            ),
            "attack_aggregated_weight_after_start": aggregated_weight_after_start,
            "final_accuracy": end["accuracy"],
            "final_backdoor_asr": end["backdoor_asr"],
        }))
    if set(grouped_metrics) != set(grouped_nodes) or set(grouped_events) != set(grouped_nodes):
        raise ValueError("node, event and metric run keys differ")
    return runs


def pair_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, ...], dict[str, dict]] = defaultdict(dict)
    for row in runs:
        key = (row["attack_scenario"], row["noniid_alpha"], row["repeat"])
        if row["variant"] in grouped[key]:
            raise ValueError(f"duplicate paired run: {key}")
        grouped[key][row["variant"]] = row
    pairs = []
    field_map = {
        "initial_admitted": "initial_admitted",
        "first_attack_returned": "first_attack_returned",
        "first_attack_aggregated": "first_attack_aggregated",
        "first_attack_reason": "first_attack_reason",
        "revoked_round": "revoked_round",
        "normal_initial_admitted": "normal_initial_admitted",
        "normal_revoked": "normal_revoked",
        "attack_aggregated_updates": "attack_aggregated_updates",
        "final_accuracy": "final_accuracy",
        "final_backdoor_asr": "final_backdoor_asr",
    }
    for key, variants in sorted(grouped.items()):
        if set(variants) != {"full", "no_online_revalidation"}:
            raise ValueError(f"unpaired variants: {key}")
        full, control = variants["full"], variants["no_online_revalidation"]
        if full["attack_start_round"] != control["attack_start_round"]:
            raise ValueError(f"different attack start rounds: {key}")
        item = dict(zip(("attack_scenario", "noniid_alpha", "repeat"), key))
        item["attack_start_round"] = full["attack_start_round"]
        for label, source in (("full", full), ("control", control)):
            for output_field, input_field in field_map.items():
                item[f"{label}_{output_field}"] = source[input_field]
        pairs.append(item)
    return pairs


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit("usage: analyze_third_scenario NODES EVENTS METRICS RUN_OUT PAIR_OUT ATTACK_START_ROUND")
    runs = analyze_runs(*(read_csv(Path(arg)) for arg in sys.argv[1:4]), int(sys.argv[6]))
    pairs = pair_runs(runs)
    write_csv(Path(sys.argv[4]), runs, RUN_FIELDS)
    write_csv(Path(sys.argv[5]), pairs, PAIR_FIELDS)


if __name__ == "__main__":
    main()

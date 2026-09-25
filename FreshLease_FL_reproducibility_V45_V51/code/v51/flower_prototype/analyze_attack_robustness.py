"""Summarize attack-onset and gradual-strength robustness experiments."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from flower_prototype.analyze_parameter_tradeoff import wilson_interval
from flower_prototype.analyze_third_scenario import analyze_runs as analyze_attack


RUN_FIELDS = (
    "variant", "noniid_alpha", "repeat", "attack_start_round",
    "gradual_sign_flip_scale", "initial_access_state", "first_attack_returned",
    "first_attack_aggregated", "first_attack_blocked", "first_attack_reason",
    "attack_revoked", "revoked_round", "revocation_reason",
    "detection_delay_rounds", "attack_updates_after_start",
    "attack_weight_after_start", "normal_false_revocations",
    "final_accuracy", "final_backdoor_asr",
)

SUMMARY_FIELDS = (
    "variant", "noniid_alpha", "attack_start_round", "gradual_sign_flip_scale",
    "independent_runs", "first_attack_block_rate",
    "first_attack_block_wilson_ci95_low", "first_attack_block_wilson_ci95_high",
    "attack_revocation_rate", "attack_revocation_wilson_ci95_low",
    "attack_revocation_wilson_ci95_high", "detection_delay_rounds_mean",
    "detection_delay_rounds_ci95", "attack_updates_after_start_mean",
    "attack_updates_after_start_ci95", "attack_weight_after_start_mean",
    "attack_weight_after_start_ci95", "attack_weight_reduction_vs_full_mean",
    "attack_weight_reduction_vs_full_ci95", "normal_false_revocation_rate",
    "final_accuracy_mean", "final_accuracy_ci95", "final_backdoor_asr_mean",
    "final_backdoor_asr_ci95",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    if not rows:
        raise ValueError("no attack-robustness rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def load_runs(root: Path, manifest: list[dict[str, str]]) -> list[dict]:
    output: list[dict] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for entry in manifest:
        child = root / entry["child_dir"]
        parsed = analyze_attack(
            read_csv(child / "real_fl_node_results.csv"),
            read_csv(child / "real_fl_round_node_events.csv"),
            read_csv(child / "real_fl_round_metrics.csv"),
            int(entry["attack_start_round"]),
        )
        for row in parsed:
            key = (
                row["variant"], row["noniid_alpha"], row["repeat"],
                entry["attack_start_round"], entry["gradual_sign_flip_scale"],
            )
            if key in seen:
                raise ValueError(f"duplicate robustness run: {key}")
            seen.add(key)
            returned = int(row["first_attack_returned"])
            aggregated = int(row["first_attack_aggregated"]) if returned else 0
            output.append({
                "variant": row["variant"],
                "noniid_alpha": row["noniid_alpha"],
                "repeat": row["repeat"],
                "attack_start_round": entry["attack_start_round"],
                "gradual_sign_flip_scale": entry["gradual_sign_flip_scale"],
                "initial_access_state": row["initial_access_state"],
                "first_attack_returned": returned,
                "first_attack_aggregated": aggregated if returned else "",
                "first_attack_blocked": int(returned and not aggregated),
                "first_attack_reason": row["first_attack_reason"],
                "attack_revoked": int(bool(str(row["revoked_round"]))),
                "revoked_round": row["revoked_round"],
                "revocation_reason": row["revocation_reason"],
                "detection_delay_rounds": row["detection_delay_rounds"],
                "attack_updates_after_start": row["attack_aggregated_after_start"],
                "attack_weight_after_start": row["attack_aggregated_weight_after_start"],
                "normal_false_revocations": row["normal_revoked"],
                "final_accuracy": row["final_accuracy"],
                "final_backdoor_asr": row["final_backdoor_asr"],
            })
    return output


def _condition_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row["noniid_alpha"]), str(row["repeat"]),
        str(row["attack_start_round"]), str(row["gradual_sign_flip_scale"]),
    )


def summarize(rows: list[dict]) -> list[dict]:
    full = {
        _condition_key(row): row for row in rows if row["variant"] == "full"
    }
    if not full:
        raise ValueError("full baseline runs are required")

    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["variant"]), str(row["noniid_alpha"]),
            str(row["attack_start_round"]), str(row["gradual_sign_flip_scale"]),
        )
        grouped[key].append(row)

    output = []
    for key, group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0], float(item[0][1]), int(item[0][2]), float(item[0][3])
        ),
    ):
        variant, alpha, start, scale = key
        n = len(group)
        returned = sum(int(row["first_attack_returned"]) for row in group)
        blocked = sum(int(row["first_attack_blocked"]) for row in group)
        revoked = sum(int(row["attack_revoked"]) for row in group)
        block_low, block_high = wilson_interval(blocked, returned)
        revoke_low, revoke_high = wilson_interval(revoked, n)
        delays = [
            float(row["detection_delay_rounds"]) for row in group
            if str(row["detection_delay_rounds"]) != ""
        ]
        updates = [float(row["attack_updates_after_start"]) for row in group]
        weights = [float(row["attack_weight_after_start"]) for row in group]
        accuracies = [float(row["final_accuracy"]) for row in group]
        asr = [float(row["final_backdoor_asr"]) for row in group]
        reductions = []
        if variant != "full":
            for row in group:
                base = full.get(_condition_key(row))
                if base is None:
                    raise ValueError("missing matched full baseline")
                base_weight = float(base["attack_weight_after_start"])
                reductions.append(
                    1.0 - float(row["attack_weight_after_start"]) / base_weight
                    if base_weight else 0.0
                )
        output.append({
            "variant": variant,
            "noniid_alpha": alpha,
            "attack_start_round": start,
            "gradual_sign_flip_scale": scale,
            "independent_runs": n,
            "first_attack_block_rate": blocked / returned if returned else "",
            "first_attack_block_wilson_ci95_low": block_low if returned else "",
            "first_attack_block_wilson_ci95_high": block_high if returned else "",
            "attack_revocation_rate": revoked / n,
            "attack_revocation_wilson_ci95_low": revoke_low,
            "attack_revocation_wilson_ci95_high": revoke_high,
            "detection_delay_rounds_mean": statistics.mean(delays) if delays else "",
            "detection_delay_rounds_ci95": ci95(delays) if delays else "",
            "attack_updates_after_start_mean": statistics.mean(updates),
            "attack_updates_after_start_ci95": ci95(updates),
            "attack_weight_after_start_mean": statistics.mean(weights),
            "attack_weight_after_start_ci95": ci95(weights),
            "attack_weight_reduction_vs_full_mean": (
                statistics.mean(reductions) if reductions else ""
            ),
            "attack_weight_reduction_vs_full_ci95": ci95(reductions) if reductions else "",
            "normal_false_revocation_rate": (
                sum(int(row["normal_false_revocations"]) for row in group) / (3 * n)
            ),
            "final_accuracy_mean": statistics.mean(accuracies),
            "final_accuracy_ci95": ci95(accuracies),
            "final_backdoor_asr_mean": statistics.mean(asr),
            "final_backdoor_asr_ci95": ci95(asr),
        })
    return output


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: analyze_attack_robustness ROOT MANIFEST RUN_OUT SUMMARY_OUT"
        )
    root = Path(sys.argv[1])
    manifest = read_csv(Path(sys.argv[2]))
    if not manifest:
        raise ValueError("empty attack-robustness manifest")
    rows = load_runs(root, manifest)
    write_csv(Path(sys.argv[3]), rows, RUN_FIELDS)
    write_csv(Path(sys.argv[4]), summarize(rows), SUMMARY_FIELDS)


if __name__ == "__main__":
    main()

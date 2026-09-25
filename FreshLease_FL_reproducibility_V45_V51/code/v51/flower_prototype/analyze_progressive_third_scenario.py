from __future__ import annotations

import sys
from pathlib import Path

from flower_prototype.analyze_third_scenario import (
    RUN_FIELDS, analyze_runs, read_csv, write_csv,
)


COMPARISON_FIELDS = (
    "attack_scenario", "noniid_alpha", "repeat", "attack_start_round",
    "variant", "initial_access_state", "initial_training_eligible",
    "first_attack_returned", "first_attack_aggregated", "first_attack_reason",
    "revoked_round", "revocation_reason", "normal_initial_admitted", "normal_initial_limited",
    "normal_initial_training_eligible", "normal_revoked",
    "normal_aggregated_updates", "attack_aggregated_updates",
    "normal_aggregated_weight_mass", "attack_aggregated_weight_mass",
    "attack_aggregated_after_start", "detection_delay_rounds",
    "attack_aggregated_weight_after_start",
    "final_accuracy", "final_backdoor_asr",
)


def comparison_rows(runs: list[dict]) -> list[dict]:
    expected = {
        "progressive_full", "progressive_no_cumulative",
        "full", "no_online_revalidation",
    }
    grouped: dict[tuple[str, str, str], dict[str, dict]] = {}
    for row in runs:
        key = (row["attack_scenario"], row["noniid_alpha"], row["repeat"])
        grouped.setdefault(key, {})[row["variant"]] = row
    output = []
    for key, variants in sorted(grouped.items()):
        if set(variants) != expected:
            raise ValueError(f"incomplete progressive comparison: {key}")
        for variant in (
            "progressive_full", "progressive_no_cumulative",
            "full", "no_online_revalidation",
        ):
            source = variants[variant]
            output.append({field: source[field] for field in COMPARISON_FIELDS})
    return output


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit("usage: analyze_progressive_third_scenario NODES EVENTS METRICS RUN_OUT COMPARISON_OUT ATTACK_START_ROUND")
    runs = analyze_runs(*(read_csv(Path(arg)) for arg in sys.argv[1:4]), int(sys.argv[6]))
    write_csv(Path(sys.argv[4]), runs, RUN_FIELDS)
    write_csv(Path(sys.argv[5]), comparison_rows(runs), COMPARISON_FIELDS)


if __name__ == "__main__":
    main()

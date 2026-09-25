"""Independent CIFAR-10 matrix with the attack starting after access promotion.

The published V47/V48 matrices begin the attack in round 2. This analyzer
refuses to label a new run as post-promotion unless every progressive-access
target reached ADMITTED before the fixed round-6 attack, with full aggregation
weight on its first attacking round. No V47/V48 observations are recycled.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from flower_prototype.analyze_cifar10_v47 import paired_effects_v47
from flower_prototype.analyze_freshness_lease_v45 import (
    enrich_round_runs, enrich_runs, summarize_rounds_v45, summarize_v45,
)
from flower_prototype.analyze_long_horizon_v44 import (
    build_long_runs, build_round_runs, effective_share,
)
from flower_prototype.analyze_q3_v43 import IDENTITY, read, write
from flower_prototype.analyze_v46_evidence import annotate_lease_setting, paired_summary


ATTACK_START_ROUND = 6
PROGRESSIVE_VARIANTS = {"access_freshness_lease", "access_full"}


def identity(row: dict) -> tuple:
    return tuple(row.get(field, "") for field in IDENTITY)


def audit_post_promotion(
    nodes: list[dict], events: list[dict], configs: list[dict],
    *, attack_start_round: int = ATTACK_START_ROUND,
    metrics: list[dict] | None = None, expected_rounds: int | None = None,
) -> list[dict]:
    """Reject a matrix unless the attack begins after actual target promotion."""
    by_nodes: dict[tuple, list[dict]] = defaultdict(list)
    by_events: dict[tuple, list[dict]] = defaultdict(list)
    by_configs: dict[tuple, list[dict]] = defaultdict(list)
    for row in nodes:
        by_nodes[identity(row)].append(row)
    for row in events:
        by_events[identity(row)].append(row)
    for row in configs:
        by_configs[identity(row)].append(row)

    by_metrics: dict[tuple, set[int]] = defaultdict(set)
    metric_count: dict[tuple, int] = defaultdict(int)
    if metrics is not None:
        if not metrics or expected_rounds is None:
            raise ValueError("metrics and expected rounds are required together")
        for row in metrics:
            by_metrics[identity(row)].add(int(float(row["round"])))
            metric_count[identity(row)] += 1

    if not by_configs or set(by_configs) != set(by_nodes) or set(by_configs) != set(by_events):
        raise ValueError("incomplete node, event or configuration matrix")
    if metrics is not None and set(by_configs) != set(by_metrics):
        raise ValueError("round metrics do not match configuration matrix")

    result = []
    for key in sorted(by_configs):
        rows = by_nodes[key]
        cfg_rows = by_configs[key]
        if len(cfg_rows) != 1:
            raise ValueError(f"missing or duplicated configuration: {key}")
        if metrics is not None and (by_metrics[key] != set(range(1, expected_rounds + 1))
                                    or metric_count[key] != expected_rounds):
            raise ValueError(f"configuration missing required rounds 1–{expected_rounds}: {key}")
        if key[0] != "cifar10" or key[4] not in PROGRESSIVE_VARIANTS | {"rffl_reputation"}:
            raise ValueError(f"unexpected dataset or comparator: {key}")
        if key[6] not in {"diverse_then_repeat_backdoor", "benign_concept_drift"}:
            raise ValueError(f"unexpected scenario: {key}")
        if any(int(float(row["attack_start_round"])) != attack_start_round
               for row in rows + cfg_rows):
            raise ValueError(f"attack start differs from round {attack_start_round}: {key}")
        targets = [row for row in rows if row["profile"] != "honest"]
        expected = int(float(key[2]))
        if len(targets) != expected or len({row["node_id"] for row in targets}) != expected:
            raise ValueError(f"target membership mismatch: {key}")

        target_first_full = []
        for target in targets:
            node_id = target["node_id"]
            per_target = [row for row in by_events[key] if row["node_id"] == node_id]
            admitted = [int(float(row["round"])) for row in per_target
                        if row.get("access_state_after") == "ADMITTED"]
            first_full = (1 if target.get("initial_access_state") == "ADMITTED"
                          else min(admitted) if admitted else None)
            if first_full is None or first_full >= attack_start_round:
                raise ValueError(f"target was not ADMITTED before attack: {key}, {node_id}")
            if key[4] in PROGRESSIVE_VARIANTS:
                if target.get("initial_access_state") != "LIMITED":
                    raise ValueError(f"expected LIMITED newcomer at admission: {key}, {node_id}")
                before_attack = [row for row in per_target
                                 if int(float(row["round"])) < attack_start_round]
                if not before_attack:
                    raise ValueError(f"no preattack training events: {key}, {node_id}")
                latest = max(before_attack, key=lambda row: int(float(row["round"])))
                if (latest.get("access_state_after") != "ADMITTED"
                    or latest.get("aggregated") != "1"
                    or float(latest.get("aggregation_weight") or 0.0) < 0.999999):
                    raise ValueError(f"no full-weight clean update before attack: {key}, {node_id}")
            target_first_full.append(first_full)

        result.append({
            **dict(zip(IDENTITY, key)),
            "attack_start_round": attack_start_round,
            "target_first_full_access_round_max": max(target_first_full),
            "target_full_access_before_attack_rate": sum(
                rnd < attack_start_round for rnd in target_first_full
            ) / expected,
            "progressive_clean_full_weight_before_attack_verified": (
                1 if key[4] in PROGRESSIVE_VARIANTS else "not_applicable"
            ),
        })
    return result


def adjust_windows_to_attack_start(
    runs: list[dict], events: list[dict], nodes: list[dict],
) -> None:
    """V44's rounds 2–4 early window is empty for a round-6 attack."""
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    target_ids_by_key: dict[tuple, set[str]] = defaultdict(set)
    for row in nodes:
        if row.get("profile") != "honest":
            target_ids_by_key[identity(row)].add(row["node_id"])
    for row in events:
        by_key[identity(row)].append(row)
    for run in runs:
        group = by_key[identity(run)]
        targets = target_ids_by_key[identity(run)]
        if not targets:
            raise ValueError(f"no target events for {identity(run)}")
        early = [row for row in group
                 if ATTACK_START_ROUND <= int(float(row["round"])) < 10]
        late = [row for row in group if int(float(row["round"])) >= 10]
        run["target_effective_aggregation_share_early"] = effective_share(early, targets)
        run["target_effective_aggregation_share_late"] = effective_share(late, targets)


def main() -> None:
    if len(sys.argv) == 8 and sys.argv[1] == "validate":
        _, _, node_path, event_path, metric_path, config_path, guard_path, rounds = sys.argv
        write(guard_path, audit_post_promotion(
            read(node_path), read(event_path), read(config_path),
            metrics=read(metric_path), expected_rounds=int(rounds),
        ))
        return
    if len(sys.argv) != 14:
        raise SystemExit(
            "usage: analyze_post_promotion_v49 STAGE nodes.csv events.csv "
            "metrics.csv configs.csv runs.csv summary.csv pair_runs.csv "
            "pair_summary.csv round_runs.csv round_summary.csv guard.csv ROUNDS"
        )
    (
        _, stage, node_path, event_path, metric_path, config_path,
        run_path, summary_path, pair_path, pair_summary_path,
        round_run_path, round_summary_path, guard_path, rounds,
    ) = sys.argv
    nodes, events = read(node_path), read(event_path)
    metrics, configs = read(metric_path), read(config_path)
    guard = audit_post_promotion(
        nodes, events, configs, metrics=metrics, expected_rounds=int(rounds),
    )
    runs = annotate_lease_setting(enrich_runs(
        build_long_runs(nodes, events, metrics, configs, stage), nodes, events,
    ))
    round_runs = annotate_lease_setting(enrich_round_runs(
        build_round_runs(nodes, events, metrics), nodes, events,
    ))
    if len(runs) != len(guard):
        raise ValueError("summarized run count does not match audited configurations")
    adjust_windows_to_attack_start(runs, events, nodes)
    paired = paired_effects_v47(runs)
    write(run_path, runs)
    write(summary_path, annotate_lease_setting(summarize_v45(runs)))
    write(pair_path, paired)
    write(pair_summary_path, paired_summary(paired))
    write(round_run_path, round_runs)
    write(round_summary_path, annotate_lease_setting(summarize_rounds_v45(round_runs)))
    write(guard_path, guard)


if __name__ == "__main__":
    main()

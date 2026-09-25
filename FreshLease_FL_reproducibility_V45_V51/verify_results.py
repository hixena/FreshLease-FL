#!/usr/bin/env python3
"""Check V45, V48, V49, V50 and V51 results against paired rows and raw ledgers."""

import argparse
import csv
import gzip
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED = {
    "v45": {"configs": 30, "rounds": 1500, "nodes": 600, "events": 30000},
    "v48": {"configs": 60, "rounds": 3000, "nodes": 1200, "events": 59744},
    "v49": {"configs": 60, "rounds": 3000, "nodes": 1200, "events": 59718},
    "v50": {"configs": 20, "rounds": 1000, "nodes": 400, "events": 20000},
    "v51": {"configs": 180, "rounds": 9000, "nodes": 3000, "events": 149372},
}


def check_sha256(release_root: Path):
    manifest = release_root / "SHA256SUMS"
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = release_root / name
        if not path.is_file() or not path.resolve().is_relative_to(release_root.resolve()):
            raise AssertionError(f"missing or escaped release file: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        check_equal(digest.hexdigest(), expected, f"SHA-256 {name}")
        checked += 1
    print(f"SHA-256 verified: {checked} files")


def location(root: Path, name: str) -> Path:
    path = root / name
    if not path.exists():
        path = root / (name + ".gz")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_rows(root: Path, name: str):
    path = location(root, name)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as fh:
        yield from csv.DictReader(fh)


def check_equal(actual, expected, what):
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected}, found {actual}")


def approx(actual, expected, what, tolerance=1e-10):
    if not math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0):
        raise AssertionError(f"{what}: expected {expected}, found {actual}")


def check_raw(root: Path, stage: str):
    spec = EXPECTED[stage]
    base = root / stage
    configs = list(read_rows(base, "real_fl_configuration_metrics.csv"))
    check_equal(len(configs), spec["configs"], f"{stage} configurations")
    fields = ("dataset", "client_count", "attacker_count", "cohort_mode",
              "variant", "trust_estimator", "attack_scenario", "noniid_alpha", "repeat")
    keys = {tuple(row.get(field) for field in fields) for row in configs}
    check_equal(len(keys), spec["configs"], f"{stage} unique configuration keys")
    check_equal(sum(1 for _ in read_rows(base, "real_fl_node_results.csv")),
                spec["nodes"], f"{stage} node records")
    check_equal(sum(1 for _ in read_rows(base, "real_fl_round_metrics.csv")),
                spec["rounds"], f"{stage} round records")
    check_equal(sum(1 for _ in read_rows(base, "real_fl_round_node_events.csv")),
                spec["events"], f"{stage} node events")
    if stage == "v51":
        audit = list(read_rows(base, "real_fl_audit_integrity.csv"))
        check_equal(len(audit), 180, "V51 audit rows")
        check_equal(sum(all(int(a[c]) == 1 for c in
                            ("audit_valid", "chain_valid", "signatures_valid"))
                        for a in audit), 180, "V51 recorded audit validity")
        guard = list(read_rows(base, "q3_access_v51_promotion_guard.csv"))
        check_equal(len(guard), 180, "V51 guard rows")
        check_equal(sum(int(g["progressive_clean_full_weight_before_attack_verified"]) == 1
                        for g in guard if g["variant"] != "rffl_reputation"),
                    120, "V51 progressive configurations with preattack full update")
    print(f"{stage}: {spec['configs']} configs, {spec['rounds']} rounds, "
          f"{spec['nodes']} nodes, {spec['events']} node events OK")


def verify_grouped_means(root, pairs_filename, summary_filename,
                         group_fields, fields, expected_count):
    pairs = list(read_rows(root, pairs_filename))
    summary = list(read_rows(root, summary_filename))
    groups = defaultdict(list)
    for row in pairs:
        groups[tuple(row[field] for field in group_fields)].append(row)
    check_equal(len(pairs), expected_count, f"{pairs_filename} count")
    check_equal(len(summary), len(groups), f"{summary_filename} group count")
    for row in summary:
        key = tuple(row[field] for field in group_fields)
        group = groups[key]
        count = int(row["independent_pairs"])
        check_equal(count, len(group), f"{summary_filename}: {key} independent pairs")
        for field in fields:
            mean = sum(float(x[field]) for x in group) / count
            approx(row[field + "_mean"], mean, f"{summary_filename}: {key} {field}")
    return summary


def verify_claims(root: Path):
    v45 = list(read_rows(root / "v45", "freshness_lease_v45_paired_effects.csv"))
    check_equal(len(v45), 30, "V45 paired rows")
    v45_center = [float(r["asr_auc_reduction"]) for r in v45
                  if r["attack_scenario"] == "diverse_then_repeat_backdoor"
                  and r["control_variant"] == "access_full"]
    check_equal(len(v45_center), 5, "V45 Fashion-MNIST backdoor paired seeds")
    approx(sum(v45_center)/5, 0.07375787037037039, "V45 Fashion-MNIST ASR-AUC reduction")
    print("V45 Fashion-MNIST: ASR-AUC reduction 7.3758 pp across five paired seeds")

    v48 = verify_grouped_means(
        root / "v48", "cifar10_v48_paired_effects.csv", "cifar10_v48_paired_summary.csv",
        ("dataset", "client_count", "attacker_count", "cohort_mode", "trust_estimator",
         "attack_scenario", "noniid_alpha", "method_variant", "control_variant"),
        ("asr_auc_reduction", "final_accuracy_difference",
         "sustained_target_share_reduction"), 40)
    v48_center = next(r for r in v48 if r["attack_scenario"] == "diverse_then_repeat_backdoor"
                      and r["control_variant"] == "access_full")
    check_equal(int(v48_center["independent_pairs"]), 10, "V48 main paired seeds")
    approx(v48_center["asr_auc_reduction_mean"], 0.05072800925925922,
           "V48 main ASR-AUC reduction")
    print("V48 CIFAR-10 main matrix: ASR-AUC reduction 5.0728 pp across ten paired seeds")

    v49 = verify_grouped_means(
        root / "v49", "post_promotion_v49_paired_effects.csv",
        "post_promotion_v49_paired_summary.csv",
        ("dataset", "client_count", "attacker_count", "cohort_mode", "trust_estimator",
         "attack_scenario", "noniid_alpha", "method_variant", "control_variant"),
        ("asr_auc_reduction", "final_accuracy_difference"), 40)
    v50 = verify_grouped_means(
        root / "v50", "weight_matched_v50_paired_runs.csv",
        "weight_matched_v50_paired_summary.csv", ("attack_scenario",),
        ("asr_auc_reduction", "final_accuracy_difference"), 20)
    v51 = verify_grouped_means(
        root / "v51", "q3_access_v51_paired_runs.csv",
        "q3_access_v51_paired_summary.csv",
        ("dataset", "client_count", "attacker_count", "cohort_mode", "trust_estimator",
         "attack_scenario", "noniid_alpha", "method_variant", "control_variant"),
        ("asr_auc_reduction", "final_accuracy_difference",
         "sustained_target_share_reduction"), 120)
    b = "diverse_then_repeat_backdoor"
    x = next(r for r in v49 if r["attack_scenario"] == b and r["control_variant"] == "access_full")
    approx(x["asr_auc_reduction_mean"], 0.0382733585858586, "V49 primary ASR-AUC reduction")
    approx(x["final_accuracy_difference_mean"], -0.014, "V49 clean accuracy difference")
    print("V49 center: ASR-AUC reduction 3.8273 pp; clean accuracy difference -1.400 pp")

    x = next(r for r in v50 if r["attack_scenario"] == b)
    approx(x["asr_auc_reduction_mean"], -0.00933648989898988, "V50 weight-matched ASR-AUC")
    rows = list(read_rows(root / "v50", "weight_matched_v50_paired_runs.csv"))
    check_equal(sum(abs(float(r["target_aggregation_mass_difference"])) < 1e-8 for r in rows),
                20, "V50 nominally matched aggregation mass")
    print("V50 center: lease-minus-fixed ASR-AUC +0.9336 pp (no detected improvement)")

    expected = {("20", "0.1"): 0.03514141414141414,
                ("20", "10"): 0.06731376262626258,
                ("10", "0.5"): 0.06005681818181816}
    for (clients, alpha), value in expected.items():
        x = next(r for r in v51 if r["attack_scenario"] == b
                 and r["control_variant"] == "access_full"
                 and r["client_count"] == clients and r["noniid_alpha"] == alpha)
        check_equal(int(x["independent_pairs"]), 10, "V51 paired seeds")
        approx(x["asr_auc_reduction_mean"], value, "V51 primary ASR-AUC")
        print(f"V51 {clients} clients, alpha={alpha}: ASR-AUC reduction {100*value:.4f} pp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--skip-event-count", action="store_true",
                        help="Skip reading the large node event ledgers")
    args = parser.parse_args()
    check_sha256(Path(__file__).parent)
    if args.skip_event_count:
        for stage, spec in EXPECTED.items():
            root = args.results_dir / stage
            check_equal(sum(1 for _ in read_rows(root, "real_fl_round_metrics.csv")),
                        spec["rounds"], f"{stage} round records")
    else:
        for stage in EXPECTED:
            check_raw(args.results_dir, stage)
    verify_claims(args.results_dir)
    print("All checks passed. These are recorded-result checks, not fresh Docker reruns.")


if __name__ == "__main__":
    main()

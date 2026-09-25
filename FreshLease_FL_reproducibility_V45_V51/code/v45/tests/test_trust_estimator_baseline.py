from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import (
    AccessLedger,
    challenge_answer,
    public_key_b64,
    sign_payload,
)
from flower_prototype.analyze_trust_estimator_comparison import (
    aggregate_run_profile,
    paired_effects,
    summarize,
)
from flower_prototype.analyze_real_fl_results import write_access_summary


def populated_ledger(root: Path, estimator: str, qualities: list[float]) -> AccessLedger:
    ledger = AccessLedger(
        root / f"{estimator}.sqlite",
        root / f"{estimator}.pem",
        1.0,
        mechanism_variant="progressive_full",
        min_completed_tasks=9,
        trust_estimator=estimator,
    )
    key = Ed25519PrivateKey.generate()
    registration = {
        "action": "REGISTER",
        "node_id": "candidate",
        "profile": "honest",
        "public_key": public_key_b64(key),
    }
    ledger.register_node(
        "candidate", "honest", registration["public_key"],
        sign_payload(key, registration), now=0.0,
    )
    for index, quality in enumerate(qualities):
        now = index * 2.0 + 1.0
        assignment = ledger.issue_task("candidate", now=now)
        task = assignment["task"]
        receipt = {
            "action": "ACK", "task_id": task["task_id"],
            "assignment_hash": assignment["assignment_hash"],
        }
        ledger.acknowledge(
            "candidate", task["task_id"], assignment["assignment_hash"],
            sign_payload(key, receipt), now=now + 0.1,
        )
        result = {
            "action": "RESULT", "task_id": task["task_id"],
            "quality": 0.95, "result_hash": f"result-{index}",
            "work_product": challenge_answer(task),
        }
        ledger.submit_result(
            "candidate", task["task_id"], 0.95, result["result_hash"],
            sign_payload(key, result), result["work_product"], now=now + 0.2,
        )
        # The current protocol challenge is binary.  Inject independently
        # verified graded observations here to unit-test estimator semantics.
        ledger.connection.execute(
            "UPDATE tasks SET quality=? WHERE task_id=?", (quality, task["task_id"])
        )
    return ledger


class TrustEstimatorTests(unittest.TestCase):
    def test_unknown_estimator_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "unknown trust estimator"):
                AccessLedger(
                    root / "bad.sqlite", root / "bad.pem",
                    trust_estimator="not_an_estimator",
                )

    def test_plain_beta_uses_posterior_mean_while_default_uses_lcb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            beta = populated_ledger(root, "beta_mean", [0.95] * 9)
            default = populated_ledger(root, "dirichlet_lcb", [0.95] * 9)
            beta_status = beta.node_status("candidate", expire=False)
            default_status = default.node_status("candidate", expire=False)
            self.assertEqual(beta_status["trust_estimator"], "beta_mean")
            self.assertEqual(
                beta_status["history_decision_score"], beta_status["history_mean"]
            )
            self.assertEqual(
                default_status["history_decision_score"], default_status["history_lcb"]
            )
            beta.connection.close()
            default.connection.close()

    def test_beta_and_dirichlet_distinguish_binary_and_graded_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualities = [0.95, 0.85, 0.79, 0.75, 0.65, 0.95, 0.85, 0.75, 0.65]
            beta = populated_ledger(root, "beta_mean", qualities)
            directory_model = populated_ledger(root, "dirichlet_mean", qualities)
            beta_status = beta.node_status("candidate", expire=False)
            dirichlet_status = directory_model.node_status("candidate", expire=False)
            self.assertNotAlmostEqual(
                beta_status["history_mean"], dirichlet_status["history_mean"]
            )
            self.assertEqual(
                dirichlet_status["history_decision_score"],
                dirichlet_status["history_mean"],
            )
            beta.connection.close()
            directory_model.connection.close()

    def test_matched_estimators_are_equivalent_on_current_binary_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            beta = populated_ledger(root, "beta_mean", [1.0] * 9)
            directory_model = populated_ledger(root, "dirichlet_mean", [1.0] * 9)
            beta_status = beta.node_status("candidate", expire=False)
            dirichlet_status = directory_model.node_status("candidate", expire=False)
            self.assertAlmostEqual(
                beta_status["history_mean"], dirichlet_status["history_mean"]
            )
            self.assertAlmostEqual(
                beta_status["trust_score"], dirichlet_status["trust_score"]
            )
            self.assertEqual(
                beta_status["access_state"], dirichlet_status["access_state"]
            )
            beta.connection.close()
            directory_model.connection.close()


class TrustEstimatorAnalysisTests(unittest.TestCase):
    @staticmethod
    def row(estimator: str, repeat: int, trust: float) -> dict[str, str]:
        return {
            "variant": "progressive_full",
            "trust_estimator": estimator,
            "attack_scenario": "diverse_then_repeat_farming",
            "noniid_alpha": "0.5",
            "repeat": str(repeat),
            "profile": "honest",
            "node_id": f"honest-{repeat}",
            "initial_access_state": "LIMITED",
            "access_state": "ADMITTED",
            "revoked_round": "",
            "trust_score": str(trust),
            "history_mean": str(trust),
            "history_std": "0.1",
            "completed_tasks": "9",
            "evidence_mass": "4.8",
            "onboarding_duration_seconds": "2.5",
        }

    def test_analysis_keeps_repeats_as_independent_units_and_pairs_estimators(self):
        rows = []
        for repeat in range(3):
            rows.append(self.row("dirichlet_mean", repeat, 0.8))
            rows.append(self.row("beta_mean", repeat, 0.7))
        runs = aggregate_run_profile(rows)
        summaries = summarize(runs)
        effects = paired_effects(runs)
        self.assertEqual(len(runs), 6)
        self.assertTrue(all(row["independent_repeats"] == 3 for row in summaries))
        trust_effect = next(row for row in effects if row["metric"] == "trust_score")
        self.assertEqual(trust_effect["independent_pairs"], 3)
        self.assertAlmostEqual(trust_effect["dirichlet_minus_beta_mean"], 0.1)

    def test_standard_access_summary_never_mixes_estimators(self):
        rows = []
        for estimator, state in (
            ("dirichlet_mean", "ADMITTED"),
            ("beta_mean", "QUARANTINE"),
        ):
            row = self.row(estimator, 0, 0.7)
            row.update({
                "initial_access_state": state,
                "access_state": state,
                "flower_fit_events": "0",
                "aggregated_fit_events": "0",
            })
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "access.csv"
            write_access_summary(rows, output)
            with output.open(newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
        self.assertEqual(len(summary), 2)
        by_estimator = {row["trust_estimator"]: row for row in summary}
        self.assertEqual(
            by_estimator["dirichlet_mean"]["initial_admitted_rate"], "1.0"
        )
        self.assertEqual(by_estimator["beta_mean"]["initial_admitted_rate"], "0.0")


if __name__ == "__main__":
    unittest.main()

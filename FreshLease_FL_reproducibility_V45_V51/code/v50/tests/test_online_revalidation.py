from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import AccessLedger, public_key_b64, sign_payload
from flower_prototype.online_checks import (
    NORM_OUTCOME_EXTREME, NORM_OUTCOME_MODERATE, NORM_OUTCOME_NONFINITE,
    NORM_OUTCOME_NORMAL, classify_update_norm, norm_escalation_decision,
    screen_update, update_norm,
)
from flower_prototype.real_data import (
    apply_training_attack, client_partition, initial_parameters,
    load_validation_test_sets, local_train,
    metrics, parameter_digest, sign_flip_update,
)


class RevalidationTests(unittest.TestCase):
    def test_norm_classifier_retains_v33_threshold_and_adds_extreme_boundary(self):
        self.assertEqual(classify_update_norm(0.49, 0.1)[0], NORM_OUTCOME_NORMAL)
        outcome, threshold, extreme, ratio = classify_update_norm(0.75, 0.1)
        self.assertEqual(outcome, NORM_OUTCOME_MODERATE)
        self.assertEqual((threshold, extreme, ratio), (0.5, 1.0, 1.5))
        self.assertEqual(classify_update_norm(1.01, 0.1)[0], NORM_OUTCOME_EXTREME)
        self.assertEqual(classify_update_norm(float("inf"), 0.1)[0], NORM_OUTCOME_NONFINITE)

    def test_norm_escalation_requires_two_consecutive_moderate_excesses(self):
        first = norm_escalation_decision(NORM_OUTCOME_MODERATE, 0)
        self.assertEqual(first, ("MODERATE_REJECTED", 1, False, True))
        second = norm_escalation_decision(NORM_OUTCOME_MODERATE, first[1])
        self.assertEqual(second, ("REPEATED_REVOKED", 2, True, False))

    def test_extreme_and_nonfinite_norms_remain_immediate_hard_failures(self):
        self.assertEqual(
            norm_escalation_decision(NORM_OUTCOME_EXTREME, 0),
            ("EXTREME_REVOKED", 1, True, False),
        )
        self.assertEqual(
            norm_escalation_decision(NORM_OUTCOME_NONFINITE, 0),
            ("NONFINITE_REVOKED", 0, True, False),
        )

    def test_backdoor_is_not_automatically_caught_by_sign_flip_screen(self):
        """One-step screening cannot be used as evidence of universal detection."""
        x_val, y_val, _, _ = load_validation_test_sets()
        base = initial_parameters()
        base_loss, _ = metrics(base, x_val, y_val)
        alpha, seed = 0.5, 0
        updates = []
        for partition_id in (0, 1, 2, 4):
            x, y = client_partition(partition_id, 6, alpha, seed)
            if partition_id == 4:
                x, y = apply_training_attack(
                    "backdoor", x, y, poison_fraction=0.4,
                    seed=seed * 10000 + partition_id * 100 + 1,
                )
            updates.append(local_train(
                base, x, y, epochs=2, learning_rate=0.15, batch_size=32,
                seed=seed * 10000 + partition_id * 100 + 1,
            ))
        median_norm = float(np.median([update_norm(base, u) for u in updates]))
        suspicious = updates[3]
        reason = screen_update(
            update_norm(base, suspicious), median_norm,
            metrics(suspicious, x_val, y_val)[0], base_loss,
        )
        self.assertIsNone(reason)

    def test_sign_flip_is_rejected_before_aggregation_on_three_paired_seeds(self):
        x_test, y_test, _, _ = load_validation_test_sets()
        initial = initial_parameters()
        global_loss, _ = metrics(initial, x_test, y_test)
        for alpha in (0.1, 0.5, 10.0):
            for seed in range(3):
                proposed = []
                for partition_id in (0, 1, 2, 4):
                    local_x, local_y = client_partition(partition_id, 6, alpha, seed)
                    trained = local_train(
                        initial, local_x, local_y, epochs=2,
                        learning_rate=0.15, batch_size=32,
                        seed=seed * 10000 + partition_id * 100 + 1,
                    )
                    proposed.append(
                        sign_flip_update(initial, trained, 3.0)
                        if partition_id == 4 else trained
                    )
                median_norm = float(np.median([
                    update_norm(initial, values) for values in proposed
                ]))
                decisions = [
                    screen_update(
                        update_norm(initial, values), median_norm,
                        metrics(values, x_test, y_test)[0], global_loss,
                    ) for values in proposed
                ]
                self.assertEqual(decisions[:3], [None, None, None], (alpha, seed))
                self.assertIsNotNone(decisions[3], (alpha, seed))

    def test_signed_commitment_can_be_accepted_only_once_and_revoked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ledger = AccessLedger(
                path / "ledger.sqlite", path / "sign.pem", 1,
                mechanism_variant="oracle_benign",
            )
            key = Ed25519PrivateKey.generate()
            public_key = public_key_b64(key)
            register = {
                "action": "REGISTER", "node_id": "node",
                "profile": "honest", "public_key": public_key,
            }
            ledger.register_node("node", "honest", public_key, sign_payload(key, register))
            fit = {
                "action": "FIT_COMMITMENT", "node_id": "node", "server_round": 1,
                "update_hash": parameter_digest(initial_parameters()),
                "parent_hash": "a" * 64,
            }
            with self.assertRaises(Exception):
                ledger.verify_fit_commitment(
                    "node", 1, fit["update_hash"], fit["parent_hash"],
                    sign_payload(Ed25519PrivateKey.generate(), fit),
                )
            ledger.verify_fit_commitment(
                "node", 1, fit["update_hash"], fit["parent_hash"], sign_payload(key, fit),
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                ledger.verify_fit_commitment(
                    "node", 1, fit["update_hash"], fit["parent_hash"], sign_payload(key, fit),
                )
            ledger.accept_fit_update("node", 1, fit["update_hash"])
            self.assertEqual(ledger.experiment_summary()["nodes"][0]["aggregated_fit_events"], 0)
            ledger.record_fit_aggregation("node", 1, fit["update_hash"])
            self.assertEqual(ledger.experiment_summary()["nodes"][0]["aggregated_fit_events"], 1)
            ledger.revoke_training_access("node", 2, "VALIDATION_LOSS")
            self.assertEqual(ledger.node_status("node")["access_state"], "QUARANTINE")
            self.assertEqual(ledger.node_status("node")["revoked_round"], 2)
            with self.assertRaisesRegex(ValueError, "not currently eligible"):
                ledger.verify_fit_commitment("node", 3, "b" * 64, "c" * 64, "invalid")
            self.assertTrue(ledger.verify_integrity()["valid"])
            ledger.connection.close()


if __name__ == "__main__":
    unittest.main()

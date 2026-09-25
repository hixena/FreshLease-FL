import unittest
import csv
import tempfile
from pathlib import Path

import numpy as np

from flower_prototype.real_data import (
    add_backdoor,
    apply_training_attack,
    clean_distribution_drift,
    client_partition,
    dataset_dimensions,
    dirichlet_partition_indices,
    initial_parameters,
    is_trained_round,
    load_validation_test_sets,
    load_dataset_data,
    local_train,
    metrics,
    sign_flip_update,
    trigger_indices,
)
from flower_prototype.analyze_real_fl_results import write_access_summary


class RealDataExperimentTests(unittest.TestCase):
    def test_ten_client_partitions_cover_training_set(self):
        _, labels, _, _ = load_dataset_data("digits")
        partitions = dirichlet_partition_indices(labels, 10, 0.5, 35)
        self.assertEqual(len(partitions), 10)
        self.assertEqual(sum(len(partition) for partition in partitions), len(labels))
        self.assertEqual(len(np.unique(np.concatenate(partitions))), len(labels))

    def test_ten_client_partitions_cover_training_set(self):
        _, labels, _, _ = load_dataset_data("digits")
        partitions = dirichlet_partition_indices(labels, 10, 0.5, 35)
        self.assertEqual(len(partitions), 10)
        self.assertEqual(sum(len(partition) for partition in partitions), len(labels))
        self.assertEqual(len(np.unique(np.concatenate(partitions))), len(labels))

    def test_common_dataset_contract_supports_a_second_feature_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fashion_mnist.npz"
            np.savez(
                path,
                x_train=np.zeros((40, 784), dtype=np.float32),
                y_train=np.tile(np.arange(10), 4),
                x_test=np.zeros((20, 784), dtype=np.float32),
                y_test=np.tile(np.arange(10), 2),
            )
            x_train, y_train, x_test, y_test = load_dataset_data(
                "fashion_mnist", path
            )
            self.assertEqual(x_train.shape, (40, 784))
            self.assertEqual(x_test.shape, (20, 784))
            self.assertEqual(len(y_train), 40)
            self.assertEqual(len(y_test), 20)
            self.assertEqual(dataset_dimensions("fashion_mnist", path), (784, 10))
            self.assertEqual(trigger_indices(784), [0, 27, 756, 783])

    def test_dataset_contract_rejects_noncontiguous_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fashion_mnist.npz"
            np.savez(
                path,
                x_train=np.zeros((2, 4)), y_train=np.array([0, 2]),
                x_test=np.zeros((2, 4)), y_test=np.array([0, 2]),
            )
            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_dataset_data("fashion_mnist", path)

    def test_validation_test_split_is_disjoint_and_stratified(self):
        _, validation_labels, _, test_labels = load_validation_test_sets()
        self.assertEqual(len(validation_labels) + len(test_labels), 360)
        self.assertTrue(all(np.sum(validation_labels == i) >= 10 for i in range(10)))
        self.assertTrue(all(np.sum(test_labels == i) >= 10 for i in range(10)))

    def test_dirichlet_partition_is_complete_and_reproducible(self):
        labels = np.repeat(np.arange(10), 30)
        first = dirichlet_partition_indices(labels, 6, 0.5, 7, min_size=10)
        second = dirichlet_partition_indices(labels, 6, 0.5, 7, min_size=10)
        self.assertEqual(sum(map(len, first)), len(labels))
        self.assertEqual(len(np.unique(np.concatenate(first))), len(labels))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_real_digits_local_training_reduces_loss(self):
        x, y = client_partition(0, 6, 0.5, 11)
        parameters = initial_parameters(x.shape[1])
        before, _ = metrics(parameters, x, y)
        trained = local_train(parameters, x, y, 3, 0.2, 32, 11)
        after, _ = metrics(trained, x, y)
        self.assertLess(after, before)

    def test_sign_flip_moves_opposite_to_local_update(self):
        global_parameters = [np.zeros((2, 2)), np.zeros(2)]
        local_parameters = [np.ones((2, 2)), np.ones(2)]
        flipped = sign_flip_update(global_parameters, local_parameters, 2.0)
        np.testing.assert_array_equal(flipped[0], -2.0 * np.ones((2, 2)))

    def test_backdoor_sets_four_corner_pixels(self):
        x = np.zeros((2, 64), dtype=np.float32)
        y = np.array([1, 2])
        triggered, targets = add_backdoor(x, y)
        np.testing.assert_array_equal(triggered[:, [0, 7, 56, 63]], 1.0)
        np.testing.assert_array_equal(targets, 0)

    def test_backdoor_is_data_poisoning_not_sign_flip(self):
        x, y = client_partition(4, 6, 0.5, 0)
        changed_x, changed_y = apply_training_attack(
            "backdoor", x, y, poison_fraction=0.4, seed=401
        )
        self.assertEqual(changed_x.shape, x.shape)
        self.assertEqual(changed_y.shape, y.shape)
        self.assertTrue(np.any(changed_x != x))
        self.assertTrue(np.any(changed_y != y))

    def test_clean_distribution_drift_is_deterministic_and_keeps_labels_attached(self):
        base_x = np.arange(24, dtype=np.float32).reshape(6, 4)
        base_y = np.arange(6, dtype=np.int64)
        shifted_x = np.arange(100, 140, dtype=np.float32).reshape(10, 4)
        shifted_y = np.arange(10, dtype=np.int64)
        first_x, first_y = clean_distribution_drift(
            base_x, base_y, shifted_x, shifted_y, 0.5, 9
        )
        second_x, second_y = clean_distribution_drift(
            base_x, base_y, shifted_x, shifted_y, 0.5, 9
        )
        np.testing.assert_array_equal(first_x, second_x)
        np.testing.assert_array_equal(first_y, second_y)
        self.assertEqual(len(first_y), len(base_y))
        valid_pairs = {
            (tuple(features), int(label))
            for features, label in zip(
                np.concatenate((base_x, shifted_x)),
                np.concatenate((base_y, shifted_y)),
            )
        }
        self.assertTrue(all(
            (tuple(features), int(label)) in valid_pairs
            for features, label in zip(first_x, first_y)
        ))

    def test_clean_distribution_drift_rejects_invalid_fraction(self):
        values = np.zeros((2, 4), dtype=np.float32)
        labels = np.zeros(2, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "fraction"):
            clean_distribution_drift(values, labels, values, labels, 1.1, 0)

    def test_access_summary_reports_admission_and_fit_events(self):
        rows = [
            {
                "variant": "full", "attack_scenario": "single_type_farming",
                "noniid_alpha": "0.5", "profile": "single_type_farming",
                "access_state": "QUARANTINE", "flower_fit_events": "0",
            },
            {
                "variant": "full", "attack_scenario": "single_type_farming",
                "noniid_alpha": "0.5", "profile": "single_type_farming",
                "access_state": "ADMITTED", "flower_fit_events": "20",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "access.csv"
            write_access_summary(rows, path)
            with path.open(newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))
        self.assertEqual(float(summary["admitted_rate"]), 0.5)
        self.assertEqual(float(summary["mean_flower_fit_events"]), 10.0)

    def test_access_summary_keeps_datasets_separate(self):
        rows = []
        for dataset_name in ("digits", "fashion_mnist"):
            rows.append({
                "dataset": dataset_name,
                "variant": "progressive_full",
                "attack_scenario": "gradual_drift_betrayal",
                "noniid_alpha": "0.5",
                "profile": "gradual_drift_betrayal",
                "access_state": "QUARANTINE",
                "flower_fit_events": "1",
                "aggregated_fit_events": "0",
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "access.csv"
            write_access_summary(rows, path)
            with path.open(newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
        self.assertEqual(len(summary), 2)
        self.assertEqual({row["dataset"] for row in summary}, {"digits", "fashion_mnist"})

    def test_revalidated_node_is_counted_as_initially_admitted_then_revoked(self):
        rows = [{
            "variant": "full", "attack_scenario": "diverse_then_repeat_farming",
            "noniid_alpha": "0.5", "profile": "diverse_then_repeat_farming",
            "initial_access_state": "ADMITTED", "access_state": "QUARANTINE",
            "revoked_round": "1", "flower_fit_events": "1",
            "aggregated_fit_events": "0",
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "access.csv"
            write_access_summary(rows, path)
            with path.open(newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))
        self.assertEqual(float(summary["initial_admitted_rate"]), 1.0)
        self.assertEqual(float(summary["final_admitted_rate"]), 0.0)
        self.assertEqual(float(summary["revoked_rate"]), 1.0)
        self.assertEqual(float(summary["mean_aggregated_fit_events"]), 0.0)

    def test_access_summary_preserves_onboarding_evidence_measurements(self):
        rows = [{
            "variant": "full", "attack_scenario": "single_type_farming",
            "noniid_alpha": "0.5", "profile": "single_type_farming",
            "initial_access_state": "QUARANTINE", "access_state": "QUARANTINE",
            "revoked_round": "", "flower_fit_events": "0", "aggregated_fit_events": "0",
            "completed_tasks": "14", "evidence_types": "1", "evidence_mass": "3.0",
            "credited_independent_tasks": "4", "verified_source_records": "14",
            "onboarding_duration_seconds": "5.5",
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "access.csv"
            write_access_summary(rows, path)
            with path.open(newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))
        self.assertEqual(float(summary["mean_completed_tasks"]), 14.0)
        self.assertEqual(float(summary["mean_credited_independent_tasks"]), 4.0)
        self.assertEqual(float(summary["mean_verified_source_records"]), 14.0)
        self.assertEqual(float(summary["mean_evidence_types"]), 1.0)
        self.assertEqual(float(summary["mean_evidence_mass"]), 3.0)
        self.assertEqual(float(summary["mean_onboarding_duration_seconds"]), 5.5)

    def test_round_zero_is_excluded_from_formal_metrics(self):
        self.assertFalse(is_trained_round(0))
        self.assertTrue(is_trained_round(1))


if __name__ == "__main__":
    unittest.main()

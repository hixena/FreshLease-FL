from __future__ import annotations

import unittest

import numpy as np

from flower_prototype.robust_aggregation import (
    coordinate_median, coordinate_trimmed_mean,
    coordinate_trimmed_mean_with_retention, fedavg_effective_delta_weights,
    fltrust, normalized_trust_weights,
)


class RobustAggregationTests(unittest.TestCase):
    def test_trimmed_mean_removes_one_extreme_per_tail(self):
        models = [[np.array([value], dtype=np.float32)] for value in (-100, 1, 2, 3, 100)]
        result = coordinate_trimmed_mean(models, trim_count=1)
        np.testing.assert_allclose(result[0], [2.0])

    def test_coordinate_median_ignores_single_outlier(self):
        models = [[np.array([value], dtype=np.float32)] for value in (1, 2, 3, 100)]
        result = coordinate_median(models)
        np.testing.assert_allclose(result[0], [2.5])

    def test_trimmed_mean_reports_coordinate_retention(self):
        models = [
            [np.array(values, dtype=np.float32)]
            for values in ((-100, 2), (1, 100), (2, 1), (3, 3), (100, -100))
        ]
        result, retention = coordinate_trimmed_mean_with_retention(models, trim_count=1)
        np.testing.assert_allclose(result[0], [2.0, 2.0])
        self.assertEqual(retention, [0.5, 0.5, 1.0, 1.0, 0.0])
        self.assertEqual(sum(retention), 3.0)

    def test_fltrust_zeros_opposed_update_and_normalizes_magnitude(self):
        base = [np.array([0.0, 0.0], dtype=np.float32)]
        root = [np.array([1.0, 0.0], dtype=np.float32)]
        clients = [
            [np.array([10.0, 0.0], dtype=np.float32)],
            [np.array([-100.0, 0.0], dtype=np.float32)],
        ]
        result, trust, scales = fltrust(base, clients, root)
        np.testing.assert_allclose(result[0], [1.0, 0.0])
        self.assertAlmostEqual(trust[0], 1.0)
        self.assertEqual(trust[1], 0.0)
        self.assertAlmostEqual(scales[0], 0.1)

    def test_normalized_fltrust_weights_measure_effective_share(self):
        self.assertEqual(normalized_trust_weights([0.0, 1.0, 3.0]), [0.0, 0.25, 0.75])
        self.assertEqual(normalized_trust_weights([0.0, 0.0]), [0.0, 0.0])

    def test_fltrust_combines_trust_with_progressive_access_weight(self):
        effective = normalized_trust_weights([1.0, 1.0], [0.25, 1.0])
        self.assertEqual(effective, [0.2, 0.8])
        base = [np.array([0.0, 0.0], dtype=np.float32)]
        root = [np.array([1.0, 0.0], dtype=np.float32)]
        clients = [
            [np.array([1.0, 1.0], dtype=np.float32)],
            [np.array([1.0, -1.0], dtype=np.float32)],
        ]
        result, _, _ = fltrust(base, clients, root, [0.25, 1.0])
        self.assertLess(float(result[0][1]), 0.0)

    def test_fedavg_effective_delta_weights_include_limited_weight(self):
        effective = fedavg_effective_delta_weights([10, 10], [0.75, 1.0])
        self.assertEqual(effective, [0.375, 0.5])

    def test_trim_count_must_leave_a_model(self):
        with self.assertRaises(ValueError):
            coordinate_trimmed_mean([[np.array([1.0])], [np.array([2.0])]], 1)


if __name__ == "__main__":
    unittest.main()

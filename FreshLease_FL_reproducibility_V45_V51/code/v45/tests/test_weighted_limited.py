from __future__ import annotations

import unittest

import numpy as np

from flower_prototype.online_checks import (
    delta_opposition, directional_opposition, shrink_update,
)


class WeightedLimitedTests(unittest.TestCase):
    def test_limited_delta_is_shrunk_without_changing_reference(self):
        reference = [np.asarray([0.0, 2.0], dtype=np.float32)]
        proposed = [np.asarray([4.0, -2.0], dtype=np.float32)]
        weighted = shrink_update(reference, proposed, 0.25)
        np.testing.assert_allclose(weighted[0], [1.0, 1.0])
        np.testing.assert_allclose(reference[0], [0.0, 2.0])

    def test_directional_opposition_only_scores_opposite_updates(self):
        reference = [np.zeros(2, dtype=np.float32)]
        cohort = [np.asarray([1.0, 0.0], dtype=np.float32)]
        self.assertAlmostEqual(
            directional_opposition(reference, [np.asarray([-1.0, 0.0])], cohort),
            1.0,
        )
        self.assertAlmostEqual(
            directional_opposition(reference, [np.asarray([1.0, 0.0])], cohort),
            0.0,
        )

    def test_node_conditioned_reversal_ignores_consistent_direction(self):
        baseline = [np.asarray([1.0, 0.0], dtype=np.float32)]
        self.assertAlmostEqual(
            delta_opposition([np.asarray([-0.5, 0.0])], baseline), 1.0
        )
        self.assertAlmostEqual(
            delta_opposition([np.asarray([0.5, 0.1])], baseline), 0.0
        )


if __name__ == "__main__":
    unittest.main()

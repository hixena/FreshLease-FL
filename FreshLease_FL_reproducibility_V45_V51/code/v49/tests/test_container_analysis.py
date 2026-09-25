from __future__ import annotations

import math
import unittest

from flower_prototype.analyze_container_results import ci95, valid_run


class ContainerAnalysisTests(unittest.TestCase):
    def test_ci95_ignores_non_finite_values(self):
        self.assertAlmostEqual(ci95([0.0, float("nan"), 1.0]), 0.98)

    def test_incomplete_onboarding_run_is_rejected(self):
        rows = [
            {
                "node_id": f"node-{index}",
                "access_state": "ADMITTED",
                "onboarding_duration_seconds": "" if index == 0 else "1.0",
                "flower_fit_events": "20",
            }
            for index in range(6)
        ]
        valid, reason = valid_run(rows)
        self.assertFalse(valid)
        self.assertEqual(reason, "missing onboarding completion")


if __name__ == "__main__":
    unittest.main()

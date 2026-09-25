from __future__ import annotations

import unittest
from pathlib import Path


class V48PrecisionExtensionTests(unittest.TestCase):
    def test_matrix_runner_supports_nonzero_repeat_start(self):
        root = Path(__file__).resolve().parents[1]
        matrix = (
            root / "flower_prototype" / "run_real_fl_matrix.ps1"
        ).read_text()
        self.assertIn("[int]$RepeatStart = 0", matrix)
        self.assertIn("$repeat = $RepeatStart", matrix)
        self.assertIn("$repeat -lt ($RepeatStart + $Repeats)", matrix)

    def test_v48_reuses_frozen_v47_and_runs_only_seeds_five_to_nine(self):
        root = Path(__file__).resolve().parents[1]
        wrapper = (
            root / "flower_prototype" / "run_cifar10_v48.ps1"
        ).read_text()
        self.assertIn("[string]$BaseResultDirectory", wrapper)
        self.assertIn("[int]$RepeatStart = 5", wrapper)
        self.assertIn("[int]$AdditionalRepeats = 5", wrapper)
        self.assertIn("Repeats=$AdditionalRepeats; RepeatStart=$RepeatStart", wrapper)
        self.assertIn("AccessLeaseFullUpdates=5", wrapper)
        self.assertIn("MaxLocalTrainSamples=512", wrapper)
        self.assertIn("analyze_cifar10_v47 formal10", wrapper)
        self.assertIn("duplicate experiment keys", wrapper)
        self.assertIn("cifar10_v48_paired_summary.csv", wrapper)


if __name__ == "__main__":
    unittest.main()

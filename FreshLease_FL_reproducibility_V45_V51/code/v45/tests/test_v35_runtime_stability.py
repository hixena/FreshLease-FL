from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class V35RuntimeStabilityTests(unittest.TestCase):
    def test_ten_client_timeouts_are_configurable_with_safe_defaults(self):
        compose = (ROOT / "flower_prototype" / "docker-compose.evidence-farming.yml").read_text(
            encoding="utf-8",
        )
        runner = (ROOT / "flower_prototype" / "run_real_fl_matrix.ps1").read_text(
            encoding="utf-8",
        )
        wrapper = (ROOT / "flower_prototype" / "run_external_baselines_v35.ps1").read_text(
            encoding="utf-8",
        )

        self.assertIn("TASK_DEADLINE_SECONDS: ${TASK_DEADLINE_SECONDS:-30}", compose)
        self.assertIn("BARRIER_TIMEOUT_SECONDS: ${BARRIER_TIMEOUT_SECONDS:-600}", compose)
        self.assertIn("CLIENT_JOIN_TIMEOUT_SECONDS: ${CLIENT_JOIN_TIMEOUT_SECONDS:-300}", compose)
        self.assertIn(
            'os.getenv("TASK_DEADLINE_SECONDS", "30")',
            (ROOT / "flower_prototype" / "api.py").read_text(encoding="utf-8"),
        )
        self.assertIn(
            'os.getenv("BARRIER_TIMEOUT_SECONDS", "600")',
            (ROOT / "flower_prototype" / "client.py").read_text(encoding="utf-8"),
        )
        server = (ROOT / "flower_prototype" / "server.py").read_text(encoding="utf-8")
        self.assertIn('os.getenv("BARRIER_TIMEOUT_SECONDS", "600")', server)
        self.assertIn('os.getenv("CLIENT_JOIN_TIMEOUT_SECONDS", "300")', server)
        for script in (runner, wrapper):
            self.assertIn("[int]$TaskDeadlineSeconds = 30", script)
            self.assertIn("[int]$BarrierTimeoutSeconds = 600", script)
            self.assertIn("[int]$ClientJoinTimeoutSeconds = 300", script)

    def test_client_http_error_preserves_controller_detail(self):
        source = (ROOT / "flower_prototype" / "client.py").read_text(
            encoding="utf-8",
        )
        self.assertIn("response.raise_for_status()", source)
        self.assertIn("except requests.HTTPError as exc:", source)
        self.assertIn("f\"POST {path} failed with HTTP {response.status_code}: \"", source)
        self.assertIn("f\"{response.text[:500]}\"", source)

    def test_external_baseline_wrapper_exposes_delayed_backdoor(self):
        wrapper = (ROOT / "flower_prototype" / "run_external_baselines_v35.ps1").read_text(
            encoding="utf-8",
        )
        compose = (ROOT / "flower_prototype" / "docker-compose.evidence-farming.yml").read_text(
            encoding="utf-8",
        )
        self.assertIn("[string[]]$AttackScenarios", wrapper)
        self.assertIn('"diverse_then_repeat_backdoor"', wrapper)
        backdoor_service = compose.split("  diverse-repeat-backdoor:", 1)[1].split(
            "\n  shadow-exact-replay:", 1,
        )[0]
        self.assertIn("PARTITION_ID: ${TARGET_PARTITION_ID:-9}", backdoor_service)


if __name__ == "__main__":
    unittest.main()

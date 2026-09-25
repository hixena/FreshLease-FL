import unittest
from dataclasses import replace

import numpy as np

from trust_experiment.models import (
    BetaTrust,
    ControlledProbationV1,
    DirichletTrust,
    DirichletTrustV1JointGate,
    DirichletTrustV1MinEvidence,
    DirichletTrustV2,
    ModelConfig,
    build_models,
)
from trust_experiment.runner import run_models
from trust_experiment.simulation import SimulationConfig, simulate_events


def model_config() -> ModelConfig:
    return ModelConfig(
        prior_strength=2.0,
        prior_mean=0.5,
        forgetting_factor=0.97,
        ema_alpha=0.2,
        fixed_current_weight=0.5,
        lambda_min=0.35,
        kappa=20.0,
        lower_quantile=0.05,
        utility=[0.0, 0.25, 0.5, 0.75, 1.0],
    )


class TrustExperimentTests(unittest.TestCase):
    def test_scores_stay_in_unit_interval(self):
        cfg = model_config()
        models = [
            BetaTrust(cfg, "b", True, True, True, True),
            DirichletTrust(cfg, "d", True, True, True),
        ]
        for _ in range(100):
            for model in models:
                result = model.update(0.8, 1, 4, 0.7)
                self.assertTrue(0.0 <= result["trust_score"] <= 1.0)
                self.assertTrue(0.0 <= result["history_score"] <= 1.0)

    def test_reliable_positive_feedback_increases_beta_mean(self):
        model = BetaTrust(model_config(), "b", True, True, False, False)
        before = model.a / (model.a + model.b)
        result = model.update(0.8, 1, 4, 1.0)
        self.assertGreater(result["history_mean"], before)

    def test_simulation_is_reproducible(self):
        cfg = SimulationConfig(
            rounds=20,
            profiles=["stable_benign", "betrayal"],
            observation_noise_std=0.08,
            current_evidence_noise_std=0.12,
            current_evidence_signal=0.45,
            low_quality_source_rate=0.2,
            reliable_source_quality=0.95,
            low_source_quality=0.65,
            max_feedback_delay=5,
            freshness_tau=5.0,
            beta_binary_threshold=0.6,
            dirichlet_bins=[0.2, 0.4, 0.6, 0.8],
        )
        a = simulate_events(cfg, 123, 0)
        b = simulate_events(cfg, 123, 0)
        self.assertTrue(np.allclose(a["observed_quality"], b["observed_quality"]))
        self.assertTrue(np.allclose(a["current_confidence"], b["current_confidence"]))

    def test_current_round_feedback_does_not_leak_into_access_score(self):
        cfg = SimulationConfig(
            rounds=2,
            profiles=["stable_benign"],
            observation_noise_std=0.0,
            current_evidence_noise_std=0.0,
            current_evidence_signal=0.0,
            low_quality_source_rate=0.0,
            reliable_source_quality=1.0,
            low_source_quality=0.5,
            max_feedback_delay=0,
            freshness_tau=5.0,
            beta_binary_threshold=0.6,
            dirichlet_bins=[0.2, 0.4, 0.6, 0.8],
        )
        events = simulate_events(cfg, 321, 0)
        changed = events.copy()
        changed.loc[0, "beta_feedback"] = 0
        changed.loc[0, "dirichlet_level"] = 0
        original_scores = run_models(events, model_config(), ["proposed_dirichlet"])
        changed_scores = run_models(changed, model_config(), ["proposed_dirichlet"])
        # Round 0 is identical; changed post-access feedback only affects round 1.
        self.assertAlmostEqual(
            original_scores.loc[0, "trust_score"], changed_scores.loc[0, "trust_score"]
        )
        self.assertNotAlmostEqual(
            original_scores.loc[1, "trust_score"], changed_scores.loc[1, "trust_score"]
        )

    def test_full_factorial_ablation_models_exist(self):
        names = {model.name for model in build_models(model_config())}
        self.assertTrue({
            "ablation_no_reliability",
            "ablation_no_lcb",
            "ablation_no_adaptive",
            "proposed_dirichlet",
            "proposed_dirichlet_v1_min_evidence",
            "proposed_dirichlet_v1_joint_gate",
            "proposed_dirichlet_v1_controlled_probation",
            "proposed_dirichlet_v1_selection_aware",
            "score_current_only_access",
            "score_ema_access",
            "score_dirichlet_decay_mean_access",
            "score_v1_access",
            "probation_no_diversity",
            "probation_no_repeat_decay",
            "probation_no_semantic_separation",
            "probation_naive",
            "probation_unbounded",
            "proposed_dirichlet_v2",
        }.issubset(names))

    def test_selection_aware_probation_counts_only_acknowledged_missing_results(self):
        model = ControlledProbationV1(
            model_config(), use_selection_awareness=True
        )
        model.record_probation_attempt(
            0, 0, 0.9, "shadow_update", False, 1.0, acknowledged=True
        )
        model.record_probation_attempt(
            0, 0, 0.9, "shadow_update", False, 1.0, acknowledged=False
        )
        result = model.decide(0.9, 1.0)
        self.assertEqual(result["probation_acknowledged_tasks"], 1)
        self.assertEqual(result["probation_missing_results"], 1)
        self.assertAlmostEqual(result["acknowledged_missing_rate"], 1.0)
        self.assertFalse(result["missingness_ready"])

    def test_score_access_baseline_denies_below_threshold(self):
        cfg = replace(
            model_config(), limited_access_threshold=0.80,
            full_access_threshold=0.90,
        )
        model = build_models(cfg, ["score_dirichlet_decay_mean_access"])[0]
        before = model.base.alpha.copy()
        result = model.decide(current=0.10, current_confidence=1.0)
        self.assertEqual(result["access_state"], "deny")
        self.assertFalse(result["aggregation_eligible"])
        self.assertTrue(result["blocks_formal_observation_when_ineligible"])
        self.assertTrue(np.allclose(before, model.base.alpha))

    def test_v1_min_evidence_stays_in_probation_before_gate(self):
        model = DirichletTrustV1MinEvidence(model_config())
        for _ in range(5):
            result = model.decide(current=0.95, current_confidence=1.0)
            self.assertFalse(result["history_ready"])
            self.assertEqual(result["access_state"], "probation")
            self.assertEqual(result["decision_score"], 0.0)
            model.observe(beta_feedback=1, level=4, reliability=0.95)

    def test_v1_min_evidence_activates_after_enough_completed_history(self):
        model = DirichletTrustV1MinEvidence(model_config())
        for _ in range(5):
            model.observe(beta_feedback=1, level=4, reliability=1.0)
        result = model.decide(current=0.95, current_confidence=1.0)
        self.assertTrue(result["history_ready"])
        self.assertNotEqual(result["access_state"], "probation")
        self.assertEqual(result["decision_score"], result["trust_score"])

    def test_v1_joint_gate_requires_all_maturity_conditions(self):
        model = DirichletTrustV1JointGate(model_config())
        for _ in range(5):
            model.observe(beta_feedback=1, level=4, reliability=1.0)
        result = model.decide(current=0.95, current_confidence=1.0)
        self.assertEqual(
            result["history_ready"],
            result["count_ready"]
            and result["uncertainty_ready"]
            and result["confidence_ready"],
        )

    def test_v1_joint_gate_caps_current_weight_after_maturity(self):
        model = DirichletTrustV1JointGate(model_config())
        for _ in range(30):
            model.observe(beta_feedback=1, level=4, reliability=1.0)
        result = model.decide(current=0.95, current_confidence=1.0)
        self.assertTrue(result["history_ready"])
        self.assertLessEqual(
            result["current_weight"],
            model_config().max_current_weight_after_activation,
        )

    def test_probation_attestation_does_not_inflate_behavior_history(self):
        model = ControlledProbationV1(model_config())
        before = model.alpha.copy()
        model.observe(1, 4, 1.0, evidence_type="attestation")
        self.assertTrue(np.allclose(before, model.alpha))
        self.assertGreater(model.probation_evidence_mass, 0.0)

    def test_repeated_probation_evidence_has_diminishing_contribution(self):
        model = ControlledProbationV1(model_config())
        model.observe(1, 4, 1.0, evidence_type="protocol_check")
        first = model.probation_evidence_mass
        model.observe(1, 4, 1.0, evidence_type="protocol_check")
        second = model.probation_evidence_mass - first
        self.assertLess(second, first)

    def test_repeat_decay_ablation_counts_repeated_evidence_equally(self):
        model = ControlledProbationV1(
            model_config(), use_repeat_decay=False
        )
        model.observe(1, 4, 1.0, evidence_type="protocol_check")
        first = model.probation_evidence_mass
        model.observe(1, 4, 1.0, evidence_type="protocol_check")
        second = model.probation_evidence_mass - first
        self.assertAlmostEqual(first, second)

    def test_semantic_separation_ablation_counts_attestation_as_behavior(self):
        model = ControlledProbationV1(
            model_config(), separate_attestation=False
        )
        before = model.alpha.copy()
        model.observe(1, 4, 1.0, evidence_type="attestation")
        self.assertFalse(np.allclose(before, model.alpha))

    def test_controlled_probation_requires_completed_diverse_tasks(self):
        model = ControlledProbationV1(model_config())
        tasks = [
            "attestation", "protocol_check", "canary_training",
            "shadow_update", "shadow_update", "canary_training",
            "protocol_check", "shadow_update", "canary_training",
        ]
        for task in tasks:
            model.observe(1, 4, 1.0, evidence_type=task)
        result = model.decide(current=0.95, current_confidence=1.0)
        self.assertTrue(result["history_ready"])
        self.assertFalse(result["requires_probation_task"])
        self.assertTrue(result["aggregation_eligible"])

    def test_probation_budget_triggers_quarantine_then_reattest(self):
        cfg = replace(
            model_config(), probation_max_attempts=2,
            probation_max_cost=100.0,
            probation_max_consecutive_failures=99,
            probation_cooldown_rounds=2,
        )
        model = ControlledProbationV1(cfg)
        for _ in range(2):
            model.decide(0.9, 1.0)
            model.record_probation_attempt(1, 4, 1.0, "shadow_update", True, 1.0)
        quarantined = model.decide(0.9, 1.0)
        self.assertEqual(quarantined["access_state"], "quarantine")
        self.assertFalse(quarantined["requires_probation_task"])
        model.decide(0.9, 1.0)
        reattest = model.decide(0.9, 1.0)
        self.assertEqual(reattest["required_probation_evidence_type"], "attestation")

    def test_successful_reattest_starts_fresh_evidence_cycle(self):
        cfg = replace(
            model_config(), probation_max_attempts=1,
            probation_max_cost=100.0,
            probation_max_consecutive_failures=99,
            probation_cooldown_rounds=1,
        )
        model = ControlledProbationV1(cfg)
        model.record_probation_attempt(1, 4, 1.0, "shadow_update", True, 1.0)
        model.decide(0.9, 1.0)  # enters and consumes the one-round cooldown
        request = model.decide(0.9, 1.0)
        self.assertTrue(request["reattest_required"])
        model.record_probation_attempt(1, 4, 1.0, "attestation", True, 0.2)
        self.assertEqual(model.probation_type_counts["shadow_update"], 0)
        self.assertEqual(model.probation_type_counts["attestation"], 1)

    def test_retry_limit_causes_terminal_denial(self):
        cfg = replace(
            model_config(), probation_max_attempts=1,
            probation_max_cost=100.0,
            probation_max_consecutive_failures=99,
            probation_max_reapplications=0,
        )
        model = ControlledProbationV1(cfg)
        model.record_probation_attempt(1, 4, 1.0, "shadow_update", True, 1.0)
        result = model.decide(0.9, 1.0)
        self.assertTrue(result["permanently_denied"])
        self.assertEqual(result["reason_code"], "PROBATION_RETRY_LIMIT_EXCEEDED")

    def test_unbounded_ablation_does_not_enter_quarantine(self):
        cfg = replace(model_config(), probation_max_attempts=1)
        model = ControlledProbationV1(cfg, use_budget_guard=False)
        for _ in range(3):
            model.record_probation_attempt(1, 4, 1.0, "shadow_update", True, 1.0)
        result = model.decide(0.9, 1.0)
        self.assertEqual(result["access_state"], "probation")
        self.assertEqual(result["probation_budget_exhaustions"], 0)

    def test_incomplete_probation_task_creates_no_history(self):
        cfg = SimulationConfig(
            rounds=2,
            profiles=["stable_benign"],
            observation_noise_std=0.0,
            current_evidence_noise_std=0.0,
            current_evidence_signal=0.0,
            low_quality_source_rate=0.0,
            reliable_source_quality=1.0,
            low_source_quality=0.5,
            max_feedback_delay=0,
            freshness_tau=5.0,
            beta_binary_threshold=0.6,
            dirichlet_bins=[0.2, 0.4, 0.6, 0.8],
        )
        events = simulate_events(cfg, 456, 0)
        events["probation_task_completed"] = 0
        scores = run_models(
            events, model_config(),
            ["proposed_dirichlet_v1_controlled_probation"],
        )
        self.assertEqual(scores.iloc[-1]["probation_completed_tasks"], 0)
        self.assertEqual(scores.iloc[-1]["history_observation_count"], 0)

    def test_v2_uses_current_evidence_confidence(self):
        cfg = model_config()
        model = DirichletTrustV2(cfg, "v2")
        for _ in range(5):
            model.observe(beta_feedback=0, level=0, reliability=1.0)
        low = model.decide(current=0.9, current_confidence=0.0)
        high = model.decide(current=0.9, current_confidence=1.0)
        self.assertLess(low["current_weight"], high["current_weight"])
        self.assertLess(low["trust_score"], high["trust_score"])

    def test_v2_separates_trust_score_and_risk_bound(self):
        model = DirichletTrustV2(model_config(), "v2")
        result = model.decide(current=0.7, current_confidence=0.2)
        self.assertGreaterEqual(result["trust_score"], result["risk_lower_bound"])
        self.assertEqual(result["decision_score"], result["trust_score"])
        self.assertEqual(result["access_state"], "probation")


if __name__ == "__main__":
    unittest.main()

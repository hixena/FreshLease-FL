from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist


@dataclass(frozen=True)
class ModelConfig:
    prior_strength: float
    prior_mean: float
    forgetting_factor: float
    ema_alpha: float
    fixed_current_weight: float
    lambda_min: float
    kappa: float
    lower_quantile: float
    utility: list[float]
    history_confidence_kappa: float = 10.0
    current_uncertainty_scale: float = 0.15
    confidence_min: float = 0.25
    limited_access_threshold: float = 0.50
    full_access_threshold: float = 0.60
    min_history_evidence: float = 5.0
    min_history_confidence: float = 0.15
    max_history_std: float = 0.16
    max_current_weight_after_activation: float = 0.65
    probation_min_completed_tasks: int = 5
    probation_min_evidence_mass: float = 1.50
    probation_min_evidence_types: int = 3
    probation_repeat_decay_power: float = 0.50
    probation_max_attempts: int = 40
    probation_max_cost: float = 25.0
    probation_max_consecutive_failures: int = 5
    probation_cooldown_rounds: int = 10
    probation_max_reapplications: int = 2
    probation_reattest_min_reliability: float = 0.50
    probation_missing_penalty: float = 0.20
    probation_max_acknowledged_missing_rate: float = 0.25
    probation_missing_min_acknowledged: int = 4


class TrustModel:
    name = "base"

    def decide(self, current: float, current_confidence: float = 1.0) -> dict:
        """Use history through t-1 to score the access request at round t."""
        raise NotImplementedError

    def observe(
        self,
        beta_feedback: int,
        level: int,
        reliability: float,
        evidence_type: str = "formal_training",
    ) -> None:
        """Update history after the round-t access outcome becomes available."""
        return None

    def record_probation_attempt(
        self,
        beta_feedback: int,
        level: int,
        reliability: float,
        evidence_type: str,
        completed: bool,
        cost: float,
        acknowledged: bool = True,
    ) -> None:
        """Record an attempted low-risk task without inventing missing evidence."""
        if completed:
            self.observe(beta_feedback, level, reliability, evidence_type)

    def update(self, current: float, beta_feedback: int, level: int, reliability: float) -> dict:
        """Backward-compatible one-step helper used by simple unit tests.

        The experiment runner deliberately calls decide() before observe() so
        that post-access feedback from round t cannot leak into its decision.
        """
        self.observe(beta_feedback, level, reliability)
        return self.decide(current)


class CurrentOnly(TrustModel):
    name = "current_only"

    def decide(self, current: float, current_confidence: float = 1.0) -> dict:
        return {"history_score": np.nan, "current_weight": 1.0, "trust_score": current}


class EMA(TrustModel):
    name = "ema"

    def __init__(self, cfg: ModelConfig):
        self.alpha = cfg.ema_alpha
        self.value = cfg.prior_mean

    def decide(self, current: float, current_confidence: float = 1.0) -> dict:
        self.value = self.alpha * current + (1.0 - self.alpha) * self.value
        return {"history_score": self.value, "current_weight": 0.0, "trust_score": self.value}


class ScoreThresholdAccess(TrustModel):
    """Turn a continuous trust estimator into a causal access baseline.

    The wrapped estimator receives formal post-access feedback only after it
    has admitted the node. This avoids giving score-only baselines feedback
    that an actually rejected node could not have produced.
    """

    def __init__(self, base: TrustModel, cfg: ModelConfig, name: str):
        self.base = base
        self.cfg = cfg
        self.name = name

    def decide(self, current: float, current_confidence: float = 1.0) -> dict:
        result = self.base.decide(current, current_confidence)
        score = float(result["trust_score"])
        if score >= self.cfg.full_access_threshold:
            access_state = "full_access"
        elif score >= self.cfg.limited_access_threshold:
            access_state = "limited_access"
        else:
            access_state = "deny"
        result.update({
            "decision_score": score,
            "access_state": access_state,
            "aggregation_eligible": access_state in {"limited_access", "full_access"},
            "onboarding_complete": False,
            "history_ready": True,
            "requires_probation_task": False,
            "blocks_formal_observation_when_ineligible": True,
            "reason_code": "SCORE_THRESHOLD_BASELINE",
        })
        return result

    def observe(self, beta_feedback: int, level: int, reliability: float,
                evidence_type: str = "formal_training") -> None:
        self.base.observe(beta_feedback, level, reliability, evidence_type)


class BetaTrust(TrustModel):
    def __init__(
        self,
        cfg: ModelConfig,
        name: str,
        decay: bool,
        use_reliability: bool,
        use_lcb: bool,
        adaptive_fusion: bool,
    ):
        self.name = name
        self.cfg = cfg
        self.decay = decay
        self.use_reliability = use_reliability
        self.use_lcb = use_lcb
        self.adaptive_fusion = adaptive_fusion
        self.a0 = cfg.prior_strength * cfg.prior_mean
        self.b0 = cfg.prior_strength * (1.0 - cfg.prior_mean)
        self.a, self.b = self.a0, self.b0

    def observe(self, beta_feedback: int, level: int, reliability: float,
                evidence_type: str = "formal_training") -> None:
        gamma = self.cfg.forgetting_factor if self.decay else 1.0
        rho = reliability if self.use_reliability else 1.0
        self.a = self.a0 + gamma * (self.a - self.a0) + rho * beta_feedback
        self.b = self.b0 + gamma * (self.b - self.b0) + rho * (1 - beta_feedback)

    def decide(self, current: float, current_confidence: float = 1.0) -> dict:
        mean = self.a / (self.a + self.b)
        history = (
            float(beta_dist.ppf(self.cfg.lower_quantile, self.a, self.b))
            if self.use_lcb else float(mean)
        )
        n_eff = max(0.0, self.a + self.b - self.a0 - self.b0)
        current_weight = (
            max(self.cfg.lambda_min, self.cfg.kappa / (self.cfg.kappa + n_eff))
            if self.adaptive_fusion else self.cfg.fixed_current_weight
        )
        score = current_weight * current + (1.0 - current_weight) * history
        return {
            "history_score": history,
            "history_mean": mean,
            "current_weight": current_weight,
            "effective_evidence": n_eff,
            "trust_score": float(np.clip(score, 0.0, 1.0)),
        }


class DirichletTrust(TrustModel):
    def __init__(
        self,
        cfg: ModelConfig,
        name: str,
        use_reliability: bool,
        use_lcb: bool,
        adaptive_fusion: bool,
    ):
        self.name = name
        self.cfg = cfg
        self.utility = np.asarray(cfg.utility, dtype=float)
        if len(self.utility) < 2:
            raise ValueError("Dirichlet至少需要两个等级")
        self.alpha0 = np.full(len(self.utility), cfg.prior_strength / len(self.utility))
        self.alpha = self.alpha0.copy()
        self.use_reliability = use_reliability
        self.use_lcb = use_lcb
        self.adaptive_fusion = adaptive_fusion

    def _mean_and_std(self) -> tuple[float, float]:
        total = float(self.alpha.sum())
        weighted_sum = float(self.utility @ self.alpha)
        mean = weighted_sum / total
        variance = (
            total * float((self.utility ** 2) @ self.alpha) - weighted_sum ** 2
        ) / (total ** 2 * (total + 1.0))
        return mean, float(np.sqrt(max(variance, 0.0)))

    def observe(self, beta_feedback: int, level: int, reliability: float,
                evidence_type: str = "formal_training") -> None:
        rho = reliability if self.use_reliability else 1.0
        self.alpha = self.alpha0 + self.cfg.forgetting_factor * (self.alpha - self.alpha0)
        self.alpha[level] += rho

    def decide(self, current: float, current_confidence: float = 1.0) -> dict:
        mean, std = self._mean_and_std()

        # 5%单侧正态近似。最终论文可用Monte Carlo分位点复核。
        z_by_quantile = {0.10: 1.281552, 0.05: 1.644854, 0.025: 1.959964, 0.01: 2.326348}
        z = z_by_quantile.get(self.cfg.lower_quantile, 1.644854)
        history = max(0.0, mean - z * std) if self.use_lcb else mean
        n_eff = max(0.0, float((self.alpha - self.alpha0).sum()))
        current_weight = (
            max(self.cfg.lambda_min, self.cfg.kappa / (self.cfg.kappa + n_eff))
            if self.adaptive_fusion else self.cfg.fixed_current_weight
        )
        score = current_weight * current + (1.0 - current_weight) * history
        return {
            "history_score": history,
            "history_mean": mean,
            "history_std": std,
            "current_weight": current_weight,
            "effective_evidence": n_eff,
            "trust_score": float(np.clip(score, 0.0, 1.0)),
        }


class DirichletTrustV1MinEvidence(DirichletTrust):
    """V1 with a minimum effective-history gate before ordinary admission."""

    def __init__(self, cfg: ModelConfig, name: str = "proposed_dirichlet_v1_min_evidence"):
        super().__init__(
            cfg,
            name,
            use_reliability=True,
            use_lcb=True,
            adaptive_fusion=True,
        )
        self.history_observation_count = 0

    def observe(self, beta_feedback: int, level: int, reliability: float,
                evidence_type: str = "formal_training") -> None:
        self.history_observation_count += 1
        super().observe(beta_feedback, level, reliability, evidence_type)

    def decide(self, current: float, current_confidence: float = 1.0) -> dict:
        result = super().decide(current, current_confidence)
        history_ready = self.history_observation_count >= self.cfg.min_history_evidence

        if not history_ready:
            # Preserve the V1 estimate for auditing, but prevent uncertain
            # cold-start history from authorizing ordinary/full participation.
            result["decision_score"] = 0.0
            result["access_state"] = "probation"
        elif result["trust_score"] >= self.cfg.full_access_threshold:
            result["decision_score"] = result["trust_score"]
            result["access_state"] = "full_access"
        elif result["trust_score"] >= self.cfg.limited_access_threshold:
            result["decision_score"] = result["trust_score"]
            result["access_state"] = "limited_access"
        else:
            result["decision_score"] = result["trust_score"]
            result["access_state"] = "deny"

        result["history_ready"] = history_ready
        result["history_observation_count"] = self.history_observation_count
        result["min_history_evidence"] = self.cfg.min_history_evidence
        return result


class DirichletTrustV1JointGate(DirichletTrust):
    """V1 with joint history-maturity gating and a current-weight cap."""

    def __init__(self, cfg: ModelConfig, name: str = "proposed_dirichlet_v1_joint_gate"):
        super().__init__(
            cfg,
            name,
            use_reliability=True,
            use_lcb=True,
            adaptive_fusion=True,
        )
        self.history_observation_count = 0

    def observe(self, beta_feedback: int, level: int, reliability: float,
                evidence_type: str = "formal_training") -> None:
        self.history_observation_count += 1
        super().observe(beta_feedback, level, reliability, evidence_type)

    def decide(self, current: float, current_confidence: float = 1.0) -> dict:
        result = super().decide(current, current_confidence)
        n_eff = float(result["effective_evidence"])
        history_std = float(result["history_std"])
        history_confidence = (
            n_eff / (n_eff + self.cfg.history_confidence_kappa)
            if n_eff > 0.0 else 0.0
        )
        count_ready = self.history_observation_count >= self.cfg.min_history_evidence
        uncertainty_ready = history_std <= self.cfg.max_history_std
        confidence_ready = history_confidence >= self.cfg.min_history_confidence
        history_ready = count_ready and uncertainty_ready and confidence_ready

        original_current_weight = float(result["current_weight"])
        current_weight = min(
            original_current_weight,
            self.cfg.max_current_weight_after_activation,
        )
        score = float(np.clip(
            current_weight * current
            + (1.0 - current_weight) * float(result["history_score"]),
            0.0,
            1.0,
        ))
        result["trust_score"] = score
        result["original_current_weight"] = original_current_weight
        result["current_weight"] = current_weight
        result["history_confidence"] = history_confidence
        result["history_observation_count"] = self.history_observation_count
        result["count_ready"] = count_ready
        result["uncertainty_ready"] = uncertainty_ready
        result["confidence_ready"] = confidence_ready
        result["history_ready"] = history_ready

        if not history_ready:
            result["decision_score"] = 0.0
            result["access_state"] = "probation"
        elif score >= self.cfg.full_access_threshold:
            result["decision_score"] = score
            result["access_state"] = "full_access"
        elif score >= self.cfg.limited_access_threshold:
            result["decision_score"] = score
            result["access_state"] = "limited_access"
        else:
            result["decision_score"] = score
            result["access_state"] = "deny"
        return result


class ControlledProbationV1(DirichletTrust):
    """V1 with explicit, non-aggregating probation evidence acquisition.

    New nodes cannot obtain ordinary training history merely by waiting.  They
    must complete several kinds of low-risk tasks.  Device attestation improves
    onboarding maturity only; it is deliberately not counted as behavioral
    evidence in the Dirichlet posterior.
    """

    TASK_WEIGHTS = {
        "attestation": 0.25,
        "protocol_check": 0.50,
        "canary_training": 0.75,
        "shadow_update": 1.00,
    }

    def __init__(
        self,
        cfg: ModelConfig,
        name: str = "proposed_dirichlet_v1_controlled_probation",
        use_diversity: bool = True,
        use_repeat_decay: bool = True,
        separate_attestation: bool = True,
        use_budget_guard: bool = True,
        use_selection_awareness: bool = False,
    ):
        super().__init__(cfg, name, use_reliability=True, use_lcb=True,
                         adaptive_fusion=True)
        self.use_diversity = use_diversity
        self.use_repeat_decay = use_repeat_decay
        self.separate_attestation = separate_attestation
        self.use_budget_guard = use_budget_guard
        self.use_selection_awareness = use_selection_awareness
        self.probation_completed_tasks = 0
        self.probation_evidence_mass = 0.0
        self.probation_type_counts = {key: 0 for key in self.TASK_WEIGHTS}
        self.behavior_observation_count = 0
        self.onboarding_complete = False
        self.probation_cycle_attempts = 0
        self.probation_cycle_cost = 0.0
        self.probation_consecutive_failures = 0
        self.probation_total_attempts = 0
        self.probation_total_cost = 0.0
        self.probation_budget_exhaustions = 0
        self.probation_reapplications = 0
        self.probation_acknowledged_tasks = 0
        self.probation_missing_results = 0
        self.cooldown_remaining = 0
        self.reattest_required = False
        self.permanently_denied = False
        self._budget_exhausted_pending = False

    def _reset_probation_evidence(self) -> None:
        """Start a fresh onboarding cycle after successful re-attestation."""
        self.alpha = self.alpha0.copy()
        self.probation_completed_tasks = 0
        self.probation_evidence_mass = 0.0
        self.probation_type_counts = {key: 0 for key in self.TASK_WEIGHTS}
        self.behavior_observation_count = 0
        self.onboarding_complete = False
        self.probation_acknowledged_tasks = 0
        self.probation_missing_results = 0

    def _start_quarantine(self) -> None:
        self._budget_exhausted_pending = False
        self.probation_budget_exhaustions += 1
        self.probation_reapplications += 1
        self.probation_cycle_attempts = 0
        self.probation_cycle_cost = 0.0
        self.probation_consecutive_failures = 0
        if self.probation_reapplications > self.cfg.probation_max_reapplications:
            self.permanently_denied = True
            self.cooldown_remaining = 0
            self.reattest_required = False
        else:
            self.cooldown_remaining = self.cfg.probation_cooldown_rounds
            self.reattest_required = False

    def record_probation_attempt(
        self,
        beta_feedback: int,
        level: int,
        reliability: float,
        evidence_type: str,
        completed: bool,
        cost: float,
        acknowledged: bool = True,
    ) -> None:
        if self.permanently_denied or self.cooldown_remaining > 0:
            return
        self.probation_cycle_attempts += 1
        self.probation_cycle_cost += max(0.0, float(cost))
        self.probation_total_attempts += 1
        self.probation_total_cost += max(0.0, float(cost))

        if acknowledged:
            self.probation_acknowledged_tasks += 1
            if not completed:
                self.probation_missing_results += 1

        if self.reattest_required:
            successful_reattest = (
                completed
                and evidence_type == "attestation"
                and beta_feedback == 1
                and reliability >= self.cfg.probation_reattest_min_reliability
            )
            if successful_reattest:
                self._reset_probation_evidence()
                self.reattest_required = False
                self.probation_consecutive_failures = 0
                self.observe(beta_feedback, level, reliability, "attestation")
            else:
                self.probation_consecutive_failures += 1
        elif completed and acknowledged:
            self.probation_consecutive_failures = 0
            self.observe(beta_feedback, level, reliability, evidence_type)
        elif acknowledged:
            self.probation_consecutive_failures += 1

        if self.use_budget_guard and (
            self.probation_cycle_attempts >= self.cfg.probation_max_attempts
            or self.probation_cycle_cost >= self.cfg.probation_max_cost
            or self.probation_consecutive_failures
            >= self.cfg.probation_max_consecutive_failures
        ):
            self._budget_exhausted_pending = True

    def observe(self, beta_feedback: int, level: int, reliability: float,
                evidence_type: str = "formal_training") -> None:
        if evidence_type == "formal_training":
            self.behavior_observation_count += 1
            super().observe(beta_feedback, level, reliability, evidence_type)
            return

        if evidence_type not in self.TASK_WEIGHTS:
            raise ValueError(f"Unknown probation evidence type: {evidence_type}")

        previous = self.probation_type_counts[evidence_type]
        repeat_discount = (
            (1.0 + previous) ** (-self.cfg.probation_repeat_decay_power)
            if self.use_repeat_decay else 1.0
        )
        contribution = float(np.clip(reliability, 0.0, 1.0)) \
            * self.TASK_WEIGHTS[evidence_type] * repeat_discount
        self.probation_type_counts[evidence_type] += 1
        self.probation_completed_tasks += 1
        self.probation_evidence_mass += contribution

        if evidence_type != "attestation" or not self.separate_attestation:
            # Only behavior-bearing tasks update the behavioral posterior.
            self.behavior_observation_count += 1
            super().observe(beta_feedback, level, contribution, evidence_type)

    def decide(self, current: float, current_confidence: float = 1.0) -> dict:
        result = super().decide(current, current_confidence)
        n_eff = float(result["effective_evidence"])
        history_std = float(result["history_std"])
        history_confidence = (
            n_eff / (n_eff + self.cfg.history_confidence_kappa)
            if n_eff > 0.0 else 0.0
        )
        evidence_types = sum(count > 0 for count in self.probation_type_counts.values())
        acknowledged_missing_rate = (
            self.probation_missing_results / self.probation_acknowledged_tasks
            if self.probation_acknowledged_tasks else 0.0
        )
        missingness_ready = (
            self.probation_acknowledged_tasks
            >= self.cfg.probation_missing_min_acknowledged
            and acknowledged_missing_rate
            <= self.cfg.probation_max_acknowledged_missing_rate
            if self.use_selection_awareness else True
        )
        task_ready = self.probation_completed_tasks >= self.cfg.probation_min_completed_tasks
        mass_ready = self.probation_evidence_mass >= self.cfg.probation_min_evidence_mass
        diversity_ready = (
            evidence_types >= self.cfg.probation_min_evidence_types
            if self.use_diversity else True
        )
        uncertainty_ready = history_std <= self.cfg.max_history_std
        confidence_ready = history_confidence >= self.cfg.min_history_confidence
        maturity_conditions_met = (
            task_ready and mass_ready and diversity_ready
            and uncertainty_ready and confidence_ready and missingness_ready
        )
        # Onboarding maturity is a one-time graduation condition.  Subsequent
        # behavioral deterioration is handled by limited/deny decisions rather
        # than forcing the node through cold start again.
        self.onboarding_complete = self.onboarding_complete or maturity_conditions_met
        history_ready = self.onboarding_complete

        if (
            not history_ready
            and self.use_budget_guard
            and self._budget_exhausted_pending
            and not self.permanently_denied
        ):
            self._start_quarantine()

        original_current_weight = float(result["current_weight"])
        current_weight = min(original_current_weight,
                             self.cfg.max_current_weight_after_activation)
        score = float(np.clip(
            current_weight * current
            + (1.0 - current_weight) * float(result["history_score"]),
            0.0, 1.0,
        ))
        if self.use_selection_awareness:
            score = float(np.clip(
                score
                - self.cfg.probation_missing_penalty * acknowledged_missing_rate,
                0.0,
                1.0,
            ))
        result.update({
            "trust_score": score,
            "original_current_weight": original_current_weight,
            "current_weight": current_weight,
            "history_confidence": history_confidence,
            "history_observation_count": self.behavior_observation_count,
            "probation_completed_tasks": self.probation_completed_tasks,
            "probation_evidence_mass": self.probation_evidence_mass,
            "probation_evidence_types": evidence_types,
            "task_ready": task_ready,
            "mass_ready": mass_ready,
            "diversity_ready": diversity_ready,
            "uncertainty_ready": uncertainty_ready,
            "confidence_ready": confidence_ready,
            "missingness_ready": missingness_ready,
            "acknowledged_missing_rate": acknowledged_missing_rate,
            "probation_acknowledged_tasks": self.probation_acknowledged_tasks,
            "probation_missing_results": self.probation_missing_results,
            "maturity_conditions_met": maturity_conditions_met,
            "onboarding_complete": self.onboarding_complete,
            "history_ready": history_ready,
            "requires_probation_task": not history_ready,
            "use_diversity": self.use_diversity,
            "use_repeat_decay": self.use_repeat_decay,
            "separate_attestation": self.separate_attestation,
            "use_budget_guard": self.use_budget_guard,
            "use_selection_awareness": self.use_selection_awareness,
            "probation_cycle_attempts": self.probation_cycle_attempts,
            "probation_cycle_cost": self.probation_cycle_cost,
            "probation_consecutive_failures": self.probation_consecutive_failures,
            "probation_total_attempts": self.probation_total_attempts,
            "probation_total_cost": self.probation_total_cost,
            "probation_budget_exhaustions": self.probation_budget_exhaustions,
            "probation_reapplications": self.probation_reapplications,
            "cooldown_remaining": self.cooldown_remaining,
            "reattest_required": self.reattest_required,
            "permanently_denied": self.permanently_denied,
            "blocks_formal_observation_when_ineligible": True,
        })

        if self.permanently_denied:
            result["decision_score"] = 0.0
            result["access_state"] = "deny"
            result["aggregation_eligible"] = False
            result["requires_probation_task"] = False
            result["reason_code"] = "PROBATION_RETRY_LIMIT_EXCEEDED"
        elif self.cooldown_remaining > 0:
            result["decision_score"] = 0.0
            result["access_state"] = "quarantine"
            result["aggregation_eligible"] = False
            result["requires_probation_task"] = False
            result["reason_code"] = "PROBATION_BUDGET_EXHAUSTED"
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0:
                self.reattest_required = True
        elif self.reattest_required:
            result["decision_score"] = 0.0
            result["access_state"] = "probation"
            result["aggregation_eligible"] = False
            result["requires_probation_task"] = True
            result["required_probation_evidence_type"] = "attestation"
            result["required_probation_task_cost"] = self.TASK_WEIGHTS["attestation"]
            result["reason_code"] = "RE_ATTESTATION_REQUIRED"
        elif not history_ready:
            result["decision_score"] = 0.0
            result["access_state"] = "probation"
            result["aggregation_eligible"] = False
            result["reason_code"] = "ONBOARDING_EVIDENCE_INSUFFICIENT"
        elif score >= self.cfg.full_access_threshold:
            result["decision_score"] = score
            result["access_state"] = "full_access"
            result["aggregation_eligible"] = True
        elif score >= self.cfg.limited_access_threshold:
            result["decision_score"] = score
            result["access_state"] = "limited_access"
            result["aggregation_eligible"] = True
        else:
            result["decision_score"] = score
            result["access_state"] = "deny"
            result["aggregation_eligible"] = False
        return result


class DirichletTrustV2(DirichletTrust):
    """Dual-confidence fusion with uncertainty separated from trust expectation."""

    def __init__(
        self,
        cfg: ModelConfig,
        name: str,
        use_reliability: bool = True,
        use_current_confidence: bool = True,
        lcb_as_score: bool = False,
        use_probation: bool = True,
    ):
        super().__init__(
            cfg,
            name,
            use_reliability=use_reliability,
            use_lcb=False,
            adaptive_fusion=False,
        )
        self.use_current_confidence = use_current_confidence
        self.lcb_as_score = lcb_as_score
        self.use_probation = use_probation

    def decide(self, current: float, current_confidence: float = 1.0) -> dict:
        mean, history_std = self._mean_and_std()
        n_eff = max(0.0, float((self.alpha - self.alpha0).sum()))
        history_confidence = n_eff / (
            n_eff + self.cfg.history_confidence_kappa
        ) if n_eff > 0.0 else 0.0
        evidence_confidence = (
            float(np.clip(current_confidence, 0.0, 1.0))
            if self.use_current_confidence else 1.0
        )
        confidence_sum = evidence_confidence + history_confidence
        current_weight = (
            evidence_confidence / confidence_sum if confidence_sum > 1e-12 else 0.5
        )
        score = float(np.clip(
            current_weight * current + (1.0 - current_weight) * mean,
            0.0,
            1.0,
        ))

        current_std = self.cfg.current_uncertainty_scale * (1.0 - evidence_confidence)
        fused_std = float(np.sqrt(
            current_weight ** 2 * current_std ** 2
            + (1.0 - current_weight) ** 2 * history_std ** 2
        ))
        z_by_quantile = {0.10: 1.281552, 0.05: 1.644854, 0.025: 1.959964, 0.01: 2.326348}
        z = z_by_quantile.get(self.cfg.lower_quantile, 1.644854)
        risk_lower_bound = float(max(0.0, score - z * fused_std))
        combined_confidence = float(
            current_weight * evidence_confidence
            + (1.0 - current_weight) * history_confidence
        )

        if self.use_probation and combined_confidence < self.cfg.confidence_min:
            access_state = "probation"
        elif risk_lower_bound >= self.cfg.full_access_threshold:
            access_state = "full_access"
        elif score >= self.cfg.limited_access_threshold:
            access_state = "limited_access"
        else:
            access_state = "deny"

        decision_score = risk_lower_bound if self.lcb_as_score else score
        return {
            "history_score": mean,
            "history_mean": mean,
            "history_std": history_std,
            "history_confidence": history_confidence,
            "evidence_confidence": evidence_confidence,
            "combined_confidence": combined_confidence,
            "current_weight": current_weight,
            "effective_evidence": n_eff,
            "risk_lower_bound": risk_lower_bound,
            "decision_score": decision_score,
            "access_state": access_state,
            "trust_score": score,
        }


def build_model_registry(cfg: ModelConfig) -> dict[str, TrustModel]:
    """Return fresh model instances, including full factorial ablations."""
    models: list[TrustModel] = [
        CurrentOnly(),
        EMA(cfg),
        ScoreThresholdAccess(CurrentOnly(), cfg, "score_current_only_access"),
        ScoreThresholdAccess(EMA(cfg), cfg, "score_ema_access"),
        ScoreThresholdAccess(
            DirichletTrust(cfg, "_dirichlet_decay_mean_access", False, False, False),
            cfg, "score_dirichlet_decay_mean_access",
        ),
        ScoreThresholdAccess(
            DirichletTrust(cfg, "_v1_score_access", True, True, True),
            cfg, "score_v1_access",
        ),
        BetaTrust(cfg, "beta_no_decay_mean", False, False, False, False),
        BetaTrust(cfg, "beta_decay_mean", True, False, False, False),
        DirichletTrust(cfg, "dirichlet_decay_mean", False, False, False),
        DirichletTrust(cfg, "dirichlet_plus_reliability", True, False, False),
        DirichletTrust(cfg, "dirichlet_plus_lcb", True, True, False),
        DirichletTrust(cfg, "ablation_no_reliability", False, True, True),
        DirichletTrust(cfg, "ablation_no_lcb", True, False, True),
        DirichletTrust(cfg, "ablation_no_adaptive", True, True, False),
        DirichletTrust(cfg, "proposed_dirichlet", True, True, True),
        DirichletTrustV1MinEvidence(cfg),
        DirichletTrustV1JointGate(cfg),
        ControlledProbationV1(cfg),
        ControlledProbationV1(
            cfg, "proposed_dirichlet_v1_selection_aware",
            use_selection_awareness=True,
        ),
        ControlledProbationV1(
            cfg, "probation_no_diversity", use_diversity=False,
        ),
        ControlledProbationV1(
            cfg, "probation_no_repeat_decay", use_repeat_decay=False,
        ),
        ControlledProbationV1(
            cfg, "probation_no_semantic_separation", separate_attestation=False,
        ),
        ControlledProbationV1(
            cfg, "probation_naive", use_diversity=False,
            use_repeat_decay=False, separate_attestation=False,
        ),
        ControlledProbationV1(
            cfg, "probation_unbounded", use_budget_guard=False,
        ),
        DirichletTrustV2(cfg, "v2_no_reliability", use_reliability=False),
        DirichletTrustV2(cfg, "v2_no_current_confidence", use_current_confidence=False),
        DirichletTrustV2(cfg, "v2_lcb_as_score", lcb_as_score=True),
        DirichletTrustV2(cfg, "v2_no_probation", use_probation=False),
        DirichletTrustV2(cfg, "proposed_dirichlet_v2"),
        BetaTrust(cfg, "proposed_beta", True, True, True, True),
    ]
    return {model.name: model for model in models}


def build_models(cfg: ModelConfig, names: list[str] | None = None) -> list[TrustModel]:
    """Build all models or an explicitly selected subset in requested order."""
    registry = build_model_registry(cfg)
    if names is None:
        return list(registry.values())
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    return [registry[name] for name in names]

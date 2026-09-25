"""Single source of truth for mechanism-to-policy composition."""

from __future__ import annotations


PROGRESSIVE_VARIANTS = frozenset({
    "progressive_full",
    "progressive_no_cumulative",
    "progressive_fltrust",
    "progressive_trimmed_mean",
    # V39 access-only variants share one staged authorization mechanism.
    "access_full", "access_no_diversity", "access_no_repeat_decay",
    "access_no_result_verification", "access_freshness_lease",
})

ACCESS_CORE_VARIANTS = frozenset({
    "access_full", "access_no_diversity", "access_no_repeat_decay",
    "access_no_result_verification", "access_no_limited", "access_naive",
    "access_attestation_only", "access_static_multisource",
    "access_freshness_lease",
    # V43 literature-grounded continued-participation comparator.  It uses
    # ordinary onboarding (no LIMITED state) and a gradient-reputation rule.
    "rffl_reputation",
})

CUMULATIVE_RISK_VARIANTS = frozenset({
    "progressive_full",
    "progressive_fltrust",
    "progressive_trimmed_mean",
})

AGGREGATION_RULE_BY_VARIANT = {
    "fltrust": "fltrust",
    "progressive_fltrust": "fltrust",
    "trimmed_mean": "trimmed_mean",
    "progressive_trimmed_mean": "trimmed_mean",
    "coordinate_median": "coordinate_median",
    "rffl_reputation": "rffl_reputation",
}


def is_progressive_variant(variant: str) -> bool:
    return variant in PROGRESSIVE_VARIANTS


def uses_freshness_lease(variant: str) -> bool:
    return variant == "access_freshness_lease"


def uses_cumulative_risk(variant: str) -> bool:
    return variant in CUMULATIVE_RISK_VARIANTS


def default_aggregation_rule(variant: str) -> str:
    return AGGREGATION_RULE_BY_VARIANT.get(variant, "fedavg")


def is_access_core_variant(variant: str) -> bool:
    return variant in ACCESS_CORE_VARIANTS


def uses_result_verification(variant: str) -> bool:
    return variant not in {"no_result_verification", "access_no_result_verification", "access_naive"}


def uses_evidence_diversity(variant: str) -> bool:
    return variant not in {"no_diversity", "naive", "access_no_diversity", "access_naive"}


def uses_repeat_decay(variant: str) -> bool:
    return variant not in {"no_repeat_decay", "naive", "access_no_repeat_decay", "access_naive"}


def uses_semantic_separation(variant: str) -> bool:
    return variant not in {"no_semantic_separation", "naive", "access_naive"}


def uses_probation_budget(variant: str) -> bool:
    return variant not in {"no_budget", "naive", "access_naive"}

"""Server-side screening before FedAvg: fail closed on unverifiable updates."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


NORM_OUTCOME_NORMAL = "NORMAL"
NORM_OUTCOME_MODERATE = "MODERATE_EXCESS"
NORM_OUTCOME_EXTREME = "EXTREME_EXCESS"
NORM_OUTCOME_NONFINITE = "NONFINITE"


def norm_escalation_decision(
    norm_class: str, prior_strikes: int, moderate_shadow: bool = False,
) -> tuple[str, int, bool, bool]:
    """Map a norm class to telemetry, strike count, hard/soft action.

    ``moderate_shadow`` changes only repeated *finite ordinary* excesses from
    identity revocation to update rejection.  Non-finite and extreme updates
    remain hard failures.  This mode is diagnostic: it exposes persistence
    under strong Non-IID without silently changing the frozen V34 policy.
    """
    if prior_strikes < 0:
        raise ValueError("prior strikes cannot be negative")
    if norm_class == NORM_OUTCOME_NONFINITE:
        return "NONFINITE_REVOKED", prior_strikes, True, False
    if norm_class == NORM_OUTCOME_EXTREME:
        return "EXTREME_REVOKED", prior_strikes + 1, True, False
    if norm_class == NORM_OUTCOME_MODERATE:
        strikes = prior_strikes + 1
        if strikes >= 2:
            if moderate_shadow:
                return "REPEATED_SHADOW_REJECTED", strikes, False, True
            return "REPEATED_REVOKED", strikes, True, False
        return "MODERATE_REJECTED", strikes, False, True
    if norm_class == NORM_OUTCOME_NORMAL:
        # Reset happens only after the strategy receives controller acceptance.
        return "NORMAL", prior_strikes, False, False
    raise ValueError(f"unknown norm class: {norm_class}")


def update_norm(reference: Iterable[np.ndarray], proposed: Iterable[np.ndarray]) -> float:
    base = list(reference)
    candidate = list(proposed)
    if len(base) != len(candidate) or not base:
        return math.inf
    squared = 0.0
    for old, new in zip(base, candidate):
        old_array, new_array = np.asarray(old), np.asarray(new)
        if old_array.shape != new_array.shape or not np.isfinite(new_array).all():
            return math.inf
        squared += float(np.sum((new_array.astype(np.float64) - old_array) ** 2))
    return math.sqrt(squared)


def shrink_update(
    reference: Iterable[np.ndarray], proposed: Iterable[np.ndarray], weight: float,
) -> list[np.ndarray]:
    """Shrink a verified LIMITED update toward the current global model."""
    if not 0.0 < weight <= 1.0:
        raise ValueError("aggregation weight must be in (0, 1]")
    base, candidate = list(reference), list(proposed)
    if len(base) != len(candidate) or not base:
        raise ValueError("incompatible model update")
    return [
        (np.asarray(old, dtype=np.float32)
         + weight * (np.asarray(new, dtype=np.float32)
                     - np.asarray(old, dtype=np.float32))).astype(np.float32)
        for old, new in zip(base, candidate)
    ]


def directional_opposition(
    reference: Iterable[np.ndarray], proposed: Iterable[np.ndarray],
    cohort_median: Iterable[np.ndarray],
) -> float:
    """Return the negative-cosine part of a node update against the cohort median."""
    base, candidate, median = list(reference), list(proposed), list(cohort_median)
    if not base or len(base) != len(candidate) or len(base) != len(median):
        return 1.0
    node_delta = np.concatenate([
        (np.asarray(new, dtype=np.float64) - np.asarray(old, dtype=np.float64)).ravel()
        for old, new in zip(base, candidate)
    ])
    median_delta = np.concatenate([
        (np.asarray(new, dtype=np.float64) - np.asarray(old, dtype=np.float64)).ravel()
        for old, new in zip(base, median)
    ])
    denominator = float(np.linalg.norm(node_delta) * np.linalg.norm(median_delta))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.dot(node_delta, median_delta) / denominator)
    return max(0.0, min(1.0, -cosine))


def delta_opposition(
    current_delta: Iterable[np.ndarray], baseline_delta: Iterable[np.ndarray],
) -> float:
    """Negative-cosine change against a node's own trusted update direction."""
    current, baseline = list(current_delta), list(baseline_delta)
    if not current or len(current) != len(baseline):
        return 1.0
    left = np.concatenate([
        np.asarray(value, dtype=np.float64).ravel() for value in current
    ])
    right = np.concatenate([
        np.asarray(value, dtype=np.float64).ravel() for value in baseline
    ])
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.dot(left, right) / denominator)
    return max(0.0, min(1.0, -cosine))


def screen_update(
    norm: float, median_norm: float, candidate_loss: float, global_loss: float,
    norm_multiplier: float = 2.5, loss_margin: float = 0.75,
) -> str | None:
    """Toy, explicit policy; thresholds are not calibrated guarantees for Non-IID."""
    if not all(math.isfinite(value) for value in (norm, median_norm, candidate_loss, global_loss)):
        return "VALIDATION_LOSS"
    if norm > max(0.5, norm_multiplier * median_norm):
        return "EXCESS_UPDATE_NORM"
    if candidate_loss > global_loss + max(0.5, loss_margin * global_loss):
        return "VALIDATION_LOSS"
    return None


def classify_update_norm(
    norm: float, median_norm: float, norm_multiplier: float = 2.5,
    extreme_multiplier: float = 2.0,
) -> tuple[str, float, float, float]:
    """Classify update magnitude without deciding node-level punishment.

    The ordinary threshold remains V33's ``max(0.5, 2.5 * median)``.  V34
    separates a finite ordinary exceedance from an extreme exceedance so the
    strategy can reject an isolated Non-IID outlier without revoking identity.
    """
    if norm_multiplier <= 0.0 or extreme_multiplier <= 1.0:
        raise ValueError("norm multipliers must be positive and extreme > 1")
    if not math.isfinite(median_norm):
        threshold = math.inf
    else:
        threshold = max(0.5, norm_multiplier * median_norm)
    extreme_threshold = extreme_multiplier * threshold
    ratio = norm / threshold if threshold > 0.0 else math.inf
    if not math.isfinite(norm):
        return NORM_OUTCOME_NONFINITE, threshold, extreme_threshold, ratio
    if norm > extreme_threshold:
        return NORM_OUTCOME_EXTREME, threshold, extreme_threshold, ratio
    if norm > threshold:
        return NORM_OUTCOME_MODERATE, threshold, extreme_threshold, ratio
    return NORM_OUTCOME_NORMAL, threshold, extreme_threshold, ratio

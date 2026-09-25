"""Transparent RFFL-style reputation aggregation used as an external baseline.

This is an independently implemented operationalization of Xu and Lyu's RFFL
idea, not the authors' official code.  It follows the paper's defining steps:
cosine contribution, historical fading, reputation-weighted aggregation, and
removal below a declared threshold.  The exact choices below are exported in
the experiment CSV so that the comparator is reproducible.
"""

from __future__ import annotations

import numpy as np


def _flatten_delta(reference: list[np.ndarray], model: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([
        (np.asarray(new, dtype=np.float64) - np.asarray(old, dtype=np.float64)).ravel()
        for old, new in zip(reference, model)
    ])


def cosine_contribution(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def rffl_reputation_step(
    reference: list[np.ndarray],
    client_models: list[list[np.ndarray]],
    node_ids: list[str],
    prior_reputations: dict[str, float],
    fade: float = 0.95,
    threshold_scale: float = 1.0 / 3.0,
) -> tuple[list[np.ndarray], list[float], list[float], list[bool], list[float]]:
    """Return model, updated reputation, contribution, removal, and used weight.

    Aggregation uses the prior reputation (uniform for unseen nodes).  The
    resulting direction is then used to update reputation by an EMA of the
    non-negative cosine contribution.  A node is removed when its normalized
    updated reputation is below ``threshold_scale / n``.
    """
    n = len(client_models)
    if n == 0 or len(node_ids) != n:
        raise ValueError("RFFL-style aggregation requires aligned non-empty clients")
    if not 0.0 <= fade < 1.0:
        raise ValueError("reputation fade must be in [0,1)")
    prior = np.asarray([
        max(0.0, float(prior_reputations.get(node_id, 1.0 / n)))
        for node_id in node_ids
    ], dtype=np.float64)
    if float(prior.sum()) <= 0.0:
        prior[:] = 1.0 / n
    else:
        prior /= prior.sum()
    deltas = [_flatten_delta(reference, model) for model in client_models]
    aggregate_delta = sum(weight * delta for weight, delta in zip(prior, deltas))
    contributions = [max(0.0, cosine_contribution(delta, aggregate_delta)) for delta in deltas]
    updated = fade * prior + (1.0 - fade) * np.asarray(contributions, dtype=np.float64)
    if float(updated.sum()) <= 0.0:
        updated[:] = 1.0 / n
    else:
        updated /= updated.sum()
    threshold = float(threshold_scale) / n
    removed = [bool(value < threshold) for value in updated]
    if all(removed):
        removed = [False] * n
    # Equation (4) in RFFL uses the previous-round reputation for the current
    # aggregation.  Updated reputation and removal therefore affect later
    # rounds, not the update that was just evaluated.
    model = [
        (np.asarray(old, dtype=np.float64) + sum(
            weight * (
                np.asarray(client[layer], dtype=np.float64)
                - np.asarray(old, dtype=np.float64)
            )
            for weight, client in zip(prior, client_models)
        )).astype(np.float32)
        for layer, old in enumerate(reference)
    ]
    return model, updated.tolist(), contributions, removed, prior.tolist()

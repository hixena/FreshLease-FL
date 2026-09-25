"""Pure aggregation rules used by the V36 aggregation experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


Model = Sequence[np.ndarray]


def fedavg_effective_delta_weights(
    sample_counts: Sequence[int], controller_weights: Sequence[float],
) -> list[float]:
    """Coefficients applied to client deltas by weighted-model FedAvg."""
    if len(sample_counts) != len(controller_weights):
        raise ValueError("sample counts and controller weights must align")
    if not sample_counts:
        return []
    if any(count < 0 for count in sample_counts) or any(
        weight < 0 for weight in controller_weights
    ):
        raise ValueError("aggregation weights must be non-negative")
    sample_mass = float(sum(sample_counts))
    if sample_mass <= 0:
        return [0.0] * len(sample_counts)
    return [
        float(count * weight / sample_mass)
        for count, weight in zip(sample_counts, controller_weights)
    ]


def normalized_trust_weights(
    trust_scores: Sequence[float], access_weights: Sequence[float] | None = None,
) -> list[float]:
    """Normalize FLTrust scores after applying access-controller weights."""
    if any(score < 0 for score in trust_scores):
        raise ValueError("trust scores must be non-negative")
    if access_weights is None:
        access_weights = [1.0] * len(trust_scores)
    if len(access_weights) != len(trust_scores):
        raise ValueError("trust scores and access weights must align")
    if any(weight < 0 for weight in access_weights):
        raise ValueError("access weights must be non-negative")
    weighted_scores = [
        float(score * weight)
        for score, weight in zip(trust_scores, access_weights)
    ]
    total = float(sum(weighted_scores))
    if total <= 1e-12:
        return [0.0] * len(trust_scores)
    return [float(score / total) for score in weighted_scores]


def _models(values: Iterable[Model]) -> list[list[np.ndarray]]:
    models = [[np.asarray(layer, dtype=np.float32) for layer in model] for model in values]
    if not models:
        raise ValueError("at least one client model is required")
    shapes = [layer.shape for layer in models[0]]
    if any([layer.shape for layer in model] != shapes for model in models):
        raise ValueError("all client models must have identical tensor shapes")
    return models


def coordinate_trimmed_mean(models: Iterable[Model], trim_count: int = 1) -> list[np.ndarray]:
    """Coordinate-wise trimmed mean, dropping ``trim_count`` values per tail."""
    result, _ = coordinate_trimmed_mean_with_retention(models, trim_count)
    return result


def coordinate_trimmed_mean_with_retention(
    models: Iterable[Model], trim_count: int = 1,
) -> tuple[list[np.ndarray], list[float]]:
    """Return the trimmed mean and each client's retained-coordinate fraction.

    A stable sort assigns tied coordinates deterministically.  Retention is
    telemetry for the implemented aggregation, not a claim that a retained
    coordinate is benign or that one scalar fully describes client influence.
    """
    values = _models(models)
    if trim_count < 0 or 2 * trim_count >= len(values):
        raise ValueError("trim_count must leave at least one client model")
    result = []
    retained = np.zeros(len(values), dtype=np.int64)
    coordinate_count = 0
    for layer in range(len(values[0])):
        stacked = np.stack([model[layer] for model in values], axis=0)
        order = np.argsort(stacked, axis=0, kind="stable")
        ordered = np.take_along_axis(stacked, order, axis=0)
        kept_order = (
            order[trim_count : len(values) - trim_count]
            if trim_count else order
        )
        kept = (
            ordered[trim_count : len(values) - trim_count]
            if trim_count else ordered
        )
        result.append(kept.mean(axis=0).astype(np.float32))
        coordinate_count += int(stacked[0].size)
        retained += np.bincount(kept_order.ravel(), minlength=len(values))
    if coordinate_count <= 0:
        raise ValueError("trimmed mean has no retained coordinates")
    # Divide first by the number of coordinates.  The sum of rates is the
    # number of contributors retained per coordinate.
    rates = [float(count / coordinate_count) for count in retained]
    return result, rates


def coordinate_median(models: Iterable[Model]) -> list[np.ndarray]:
    """Coordinate-wise median of complete client models."""
    values = _models(models)
    return [
        np.median(np.stack([model[layer] for model in values]), axis=0).astype(np.float32)
        for layer in range(len(values[0]))
    ]


def fltrust(
    base: Model, client_models: Iterable[Model], root_model: Model,
    access_weights: Sequence[float] | None = None,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Aggregate using FLTrust trust scores and hypersphere normalization.

    The server-root update defines both the trusted direction and target norm.
    ReLU cosine similarity supplies the trust score. Client sample counts are
    intentionally not used, matching the trust-weighted rule in FLTrust.
    """
    clients = _models(client_models)
    if access_weights is None:
        access_weights = [1.0] * len(clients)
    if len(access_weights) != len(clients):
        raise ValueError("client models and access weights must align")
    if any(weight < 0 for weight in access_weights):
        raise ValueError("access weights must be non-negative")
    base_values = [np.asarray(layer, dtype=np.float32) for layer in base]
    root_values = [np.asarray(layer, dtype=np.float32) for layer in root_model]
    if [x.shape for x in base_values] != [x.shape for x in root_values]:
        raise ValueError("root model and base model shapes differ")

    root_delta = [new - old for old, new in zip(base_values, root_values)]
    root_vector = np.concatenate([layer.ravel().astype(np.float64) for layer in root_delta])
    root_norm = float(np.linalg.norm(root_vector))
    if root_norm <= 1e-12:
        return [layer.copy() for layer in base_values], [0.0] * len(clients), [0.0] * len(clients)

    trust_scores: list[float] = []
    norm_scales: list[float] = []
    normalized_deltas: list[list[np.ndarray]] = []
    for model in clients:
        delta = [new - old for old, new in zip(base_values, model)]
        vector = np.concatenate([layer.ravel().astype(np.float64) for layer in delta])
        client_norm = float(np.linalg.norm(vector))
        if client_norm <= 1e-12:
            trust, scale = 0.0, 0.0
        else:
            trust = max(0.0, float(np.dot(vector, root_vector) / (client_norm * root_norm)))
            scale = root_norm / client_norm
        trust_scores.append(trust)
        norm_scales.append(scale)
        normalized_deltas.append([(layer * scale).astype(np.float32) for layer in delta])

    effective_weights = normalized_trust_weights(trust_scores, access_weights)
    if sum(effective_weights) <= 1e-12:
        return [layer.copy() for layer in base_values], trust_scores, norm_scales
    aggregate_delta = [
        sum(
            weight * delta[layer]
            for weight, delta in zip(effective_weights, normalized_deltas)
        )
        for layer in range(len(base_values))
    ]
    return [
        (old + delta).astype(np.float32)
        for old, delta in zip(base_values, aggregate_delta)
    ], trust_scores, norm_scales

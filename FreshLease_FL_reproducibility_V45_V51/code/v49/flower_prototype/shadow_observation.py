"""Observable digits-model update for a narrowly defined onboarding replay test.

The controller verifies numerical shape/finite range and hashes submitted bytes.
It cannot establish the client's training process or semantic independence.
"""
from __future__ import annotations

import numpy as np

from flower_prototype.real_data import parameter_digest


def inspect_shadow_artifact(artifact: object) -> str:
    """Return a server-computed exact float32 fingerprint; reject malformed work."""
    if not isinstance(artifact, dict) or set(artifact) != {"weights", "bias"}:
        raise ValueError("shadow update must contain weights and bias")
    weights, bias = artifact["weights"], artifact["bias"]
    if (not isinstance(weights, list) or len(weights) != 64
            or not isinstance(bias, list) or len(bias) != 10
            or any(not isinstance(row, list) or len(row) != 10 for row in weights)):
        raise ValueError("invalid shadow update dimensions")
    try:
        w = np.asarray(weights, dtype=np.float32)
        b = np.asarray(bias, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid numerical shadow update") from exc
    if not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite shadow update")
    magnitude = float(np.linalg.norm(w)) + float(np.linalg.norm(b))
    if not 1e-6 < magnitude < 20.0:
        raise ValueError("shadow update outside basic magnitude bounds")
    return parameter_digest((w, b))

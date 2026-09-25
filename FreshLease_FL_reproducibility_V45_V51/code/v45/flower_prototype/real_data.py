from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np


DATA_DIR = Path(__file__).with_name("data")
DATA_PATH = DATA_DIR / "digits.npz"
SUPPORTED_DATASETS = {"digits", "fashion_mnist"}


def dataset_path(
    dataset_name: str = "digits", path: str | Path | None = None,
) -> Path:
    """Resolve a bundled or explicitly mounted NPZ dataset."""
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    if path:
        return Path(path)
    configured = os.getenv("DATASET_PATH", "")
    return Path(configured) if configured else DATA_DIR / f"{dataset_name}.npz"


def load_dataset_data(
    dataset_name: str = "digits", path: str | Path | None = None,
) -> tuple[np.ndarray, ...]:
    """Load and validate the common flattened-classification NPZ contract."""
    resolved = dataset_path(dataset_name, path)
    if not resolved.exists():
        hint = (
            " Run `python flower_prototype/prepare_fashion_mnist.py` before "
            "building the Docker image."
            if dataset_name == "fashion_mnist" else ""
        )
        raise FileNotFoundError(f"dataset file not found: {resolved}.{hint}")
    with np.load(resolved) as bundle:
        required = {"x_train", "y_train", "x_test", "y_test"}
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"dataset is missing arrays: {sorted(missing)}")
        x_train = bundle["x_train"].astype(np.float32)
        y_train = bundle["y_train"].astype(np.int64)
        x_test = bundle["x_test"].astype(np.float32)
        y_test = bundle["y_test"].astype(np.int64)
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("features must be flattened two-dimensional arrays")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("train and test feature dimensions differ")
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("feature and label counts differ")
    labels = np.unique(np.concatenate((y_train, y_test)))
    if len(labels) < 2 or not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError("labels must be contiguous integers starting at zero")
    return x_train, y_train, x_test, y_test


def dataset_dimensions(
    dataset_name: str = "digits", path: str | Path | None = None,
) -> tuple[int, int]:
    x_train, y_train, _, y_test = load_dataset_data(dataset_name, path)
    num_classes = int(max(y_train.max(), y_test.max())) + 1
    return int(x_train.shape[1]), num_classes


def parameter_digest(values: Iterable[np.ndarray]) -> str:
    """Stable digest over tensor shapes, boundaries and float32 values."""
    hasher = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        hasher.update(str(array.shape).encode("ascii"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def is_trained_round(server_round: int) -> bool:
    """Round zero is untrained initialization and must not enter formal metrics."""
    return int(server_round) > 0


def load_digits_data(path: Path = DATA_PATH) -> tuple[np.ndarray, ...]:
    """Load the fixed public sklearn digits train/test split bundled with the prototype."""
    return load_dataset_data("digits", path)


def load_validation_test_sets(
    path: Path | None = None, dataset_name: str = "digits",
) -> tuple[np.ndarray, ...]:
    """Split the preexisting held-out data by class: update checks never see test labels."""
    _, _, x_heldout, y_heldout = load_dataset_data(dataset_name, path)
    rng = np.random.default_rng(2026)
    validation_indices, test_indices = [], []
    for label in np.unique(y_heldout):
        indices = np.flatnonzero(y_heldout == label)
        rng.shuffle(indices)
        midpoint = len(indices) // 2
        validation_indices.extend(indices[:midpoint])
        test_indices.extend(indices[midpoint:])
    return (x_heldout[validation_indices], y_heldout[validation_indices],
            x_heldout[test_indices], y_heldout[test_indices])


def dirichlet_partition_indices(
    labels: np.ndarray,
    num_partitions: int,
    alpha: float,
    seed: int,
    min_size: int = 20,
) -> list[np.ndarray]:
    """Create deterministic label-skew Non-IID partitions.

    A Dirichlet vector is sampled independently for every class. The routine retries
    until every client owns at least ``min_size`` examples, which prevents empty
    Flower clients without altering the requested alpha.
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if min_size * num_partitions > len(labels):
        raise ValueError("min_size is infeasible for this dataset")

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(1000):
        buckets: list[list[int]] = [[] for _ in range(num_partitions)]
        for class_id in classes:
            class_indices = np.flatnonzero(labels == class_id)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_partitions, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for bucket, shard in zip(buckets, np.split(class_indices, cuts)):
                bucket.extend(int(value) for value in shard)
        if min(map(len, buckets)) >= min_size:
            result = []
            for bucket in buckets:
                values = np.asarray(bucket, dtype=np.int64)
                rng.shuffle(values)
                result.append(values)
            return result
    raise RuntimeError("could not construct a feasible Dirichlet partition")


def client_partition(
    partition_id: int,
    num_partitions: int,
    alpha: float,
    seed: int,
    dataset_name: str = "digits",
    path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_train, y_train, _, _ = load_dataset_data(dataset_name, path)
    partitions = dirichlet_partition_indices(
        y_train, num_partitions=num_partitions, alpha=alpha, seed=seed
    )
    if not 0 <= partition_id < len(partitions):
        raise ValueError("partition_id is outside the configured partition range")
    indices = partitions[partition_id]
    return x_train[indices], y_train[indices]


def clean_distribution_drift(
    base_x: np.ndarray,
    base_y: np.ndarray,
    shifted_x: np.ndarray,
    shifted_y: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic clean-data mixture for a benign drift control.

    The labels stay attached to their original examples.  Only the share drawn
    from a second legitimate client distribution changes, so this control does
    not perform label poisoning, parameter manipulation, or sign reversal.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("drift fraction must be in [0,1]")
    if len(base_x) != len(base_y) or len(shifted_x) != len(shifted_y):
        raise ValueError("features and labels must have matching lengths")
    if len(base_y) == 0 or len(shifted_y) == 0:
        raise ValueError("drift distributions must not be empty")
    if fraction == 0.0:
        return base_x.copy(), base_y.copy()

    rng = np.random.default_rng(seed)
    size = len(base_y)
    shifted_count = int(round(size * fraction))
    base_count = size - shifted_count
    base_indices = rng.choice(
        len(base_y), size=base_count, replace=base_count > len(base_y)
    )
    shifted_indices = rng.choice(
        len(shifted_y), size=shifted_count,
        replace=shifted_count > len(shifted_y),
    )
    mixed_x = np.concatenate((base_x[base_indices], shifted_x[shifted_indices]))
    mixed_y = np.concatenate((base_y[base_indices], shifted_y[shifted_indices]))
    order = rng.permutation(size)
    return mixed_x[order].astype(np.float32), mixed_y[order].astype(np.int64)


def initial_parameters(num_features: int = 64, num_classes: int = 10) -> list[np.ndarray]:
    return [
        np.zeros((num_features, num_classes), dtype=np.float32),
        np.zeros((num_classes,), dtype=np.float32),
    ]


def _probabilities(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    logits = x @ weights + bias
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def metrics(
    parameters: Iterable[np.ndarray], x: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    weights, bias = [np.asarray(value, dtype=np.float32) for value in parameters]
    probabilities = _probabilities(x, weights, bias)
    loss = float(-np.log(probabilities[np.arange(len(y)), y] + 1e-12).mean())
    accuracy = float((probabilities.argmax(axis=1) == y).mean())
    return loss, accuracy


def local_train(
    parameters: Iterable[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> list[np.ndarray]:
    weights, bias = [np.asarray(value, dtype=np.float32).copy() for value in parameters]
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            xb, yb = x[batch], y[batch]
            probabilities = _probabilities(xb, weights, bias)
            probabilities[np.arange(len(batch)), yb] -= 1.0
            probabilities /= len(batch)
            weights -= learning_rate * (xb.T @ probabilities)
            bias -= learning_rate * probabilities.sum(axis=0)
    return [weights.astype(np.float32), bias.astype(np.float32)]


def trigger_indices(num_features: int) -> list[int]:
    """Return four deterministic corner-like coordinates for flattened images."""
    side = int(round(np.sqrt(num_features)))
    if side * side == num_features and side >= 2:
        return [0, side - 1, num_features - side, num_features - 1]
    return sorted({0, min(1, num_features - 1), max(0, num_features - 2), num_features - 1})


def add_backdoor(
    x: np.ndarray, y: np.ndarray, target: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    triggered = np.asarray(x, dtype=np.float32).copy()
    triggered[:, trigger_indices(triggered.shape[1])] = 1.0
    return triggered, np.full_like(y, target)


def apply_training_attack(
    attack: str,
    x: np.ndarray,
    y: np.ndarray,
    poison_fraction: float,
    seed: int,
    num_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    if attack in {"", "none"}:
        return x, y
    rng = np.random.default_rng(seed)
    count = max(1, int(round(len(y) * poison_fraction)))
    poisoned_indices = rng.choice(len(y), size=min(count, len(y)), replace=False)
    attacked_x, attacked_y = x.copy(), y.copy()
    if attack == "label_flip":
        attacked_y[poisoned_indices] = (
            attacked_y[poisoned_indices] + 1
        ) % num_classes
    elif attack == "backdoor":
        attacked_x[poisoned_indices], attacked_y[poisoned_indices] = add_backdoor(
            attacked_x[poisoned_indices], attacked_y[poisoned_indices]
        )
    elif attack not in {"sign_flip", "gradual_sign_flip"}:
        raise ValueError(f"unsupported training attack: {attack}")
    return attacked_x, attacked_y


def sign_flip_update(
    global_parameters: Iterable[np.ndarray],
    local_parameters: Iterable[np.ndarray],
    scale: float,
) -> list[np.ndarray]:
    global_values = [np.asarray(value, dtype=np.float32) for value in global_parameters]
    local_values = [np.asarray(value, dtype=np.float32) for value in local_parameters]
    return [
        (global_value - scale * (local_value - global_value)).astype(np.float32)
        for global_value, local_value in zip(global_values, local_values)
    ]


def backdoor_success_rate(
    parameters: Iterable[np.ndarray], x: np.ndarray, y: np.ndarray, target: int = 0
) -> float:
    keep = y != target
    triggered_x, _ = add_backdoor(x[keep], y[keep], target=target)
    weights, bias = [np.asarray(value, dtype=np.float32) for value in parameters]
    predictions = _probabilities(triggered_x, weights, bias).argmax(axis=1)
    return float((predictions == target).mean())


def append_server_metrics(
    path: Path,
    server_round: int,
    loss: float,
    accuracy: float,
    attack_success_rate: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["round", "loss", "accuracy", "backdoor_asr"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "round": server_round,
            "loss": loss,
            "accuracy": accuracy,
            "backdoor_asr": attack_success_rate,
        })

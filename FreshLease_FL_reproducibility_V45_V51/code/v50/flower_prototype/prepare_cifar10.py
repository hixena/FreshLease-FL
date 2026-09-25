from __future__ import annotations

import argparse
import hashlib
import io
import pickle
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
EXPECTED_MD5 = "c58f30108f718f92721af3b95e74349a"
TRAIN_MEMBERS = tuple(
    f"cifar-10-batches-py/data_batch_{index}" for index in range(1, 6)
)
TEST_MEMBER = "cifar-10-batches-py/test_batch"


def _download(attempts: int = 3) -> bytes:
    errors: list[str] = []
    for attempt in range(attempts):
        request = urllib.request.Request(
            URL, headers={"User-Agent": "node-access-v47"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            if hashlib.md5(payload).hexdigest() != EXPECTED_MD5:
                raise ValueError("CIFAR-10 archive MD5 mismatch")
            return payload
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            errors.append(str(exc))
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError("could not download verified CIFAR-10 archive: " + "; ".join(errors))


def _batch(archive: tarfile.TarFile, name: str) -> tuple[np.ndarray, np.ndarray]:
    member = archive.getmember(name)
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"missing CIFAR-10 member: {name}")
    record = pickle.load(source, encoding="bytes")
    features = np.asarray(record[b"data"], dtype=np.uint8)
    labels = np.asarray(record[b"labels"], dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != 3072 or len(features) != len(labels):
        raise ValueError(f"invalid CIFAR-10 batch: {name}")
    return features, labels


def arrays_from_archive(payload: bytes) -> dict[str, np.ndarray]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        train = [_batch(archive, name) for name in TRAIN_MEMBERS]
        x_test, y_test = _batch(archive, TEST_MEMBER)
    x_train = np.concatenate([features for features, _ in train])
    y_train = np.concatenate([labels for _, labels in train])
    if x_train.shape != (50000, 3072) or x_test.shape != (10000, 3072):
        raise ValueError("unexpected CIFAR-10 dataset shape")
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }


def prepare(output: Path, force: bool = False) -> Path:
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    payload = _download()
    arrays = arrays_from_archive(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **arrays,
        source_url=np.asarray([URL]),
        source_md5=np.asarray([EXPECTED_MD5]),
        feature_layout=np.asarray(["channel-first flattened RGB 32x32"]),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official CIFAR-10 for V47")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("data") / "cifar10.npz",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    path = prepare(args.output, args.force)
    print(f"prepared {path}")


if __name__ == "__main__":
    main()

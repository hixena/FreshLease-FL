from __future__ import annotations

import argparse
import gzip
import hashlib
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


BASE_URLS = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion",
    "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
)
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename: str, attempts: int = 3) -> bytes:
    errors: list[str] = []
    for attempt in range(attempts):
        for base_url in BASE_URLS:
            url = f"{base_url}/{filename}"
            request = urllib.request.Request(
                url, headers={"User-Agent": "node-access-v32"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                errors.append(f"{url}: {exc}")
        if attempt + 1 < attempts:
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"could not download {filename} from any configured mirror:\n"
        + "\n".join(errors)
    )


def _images(payload: bytes) -> np.ndarray:
    raw = gzip.decompress(payload)
    magic, count, rows, columns = struct.unpack(">IIII", raw[:16])
    if magic != 2051 or len(raw) != 16 + count * rows * columns:
        raise ValueError("invalid Fashion-MNIST image archive")
    values = np.frombuffer(raw, dtype=np.uint8, offset=16)
    return values.reshape(count, rows * columns).astype(np.float32) / 255.0


def _labels(payload: bytes) -> np.ndarray:
    raw = gzip.decompress(payload)
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049 or len(raw) != 8 + count:
        raise ValueError("invalid Fashion-MNIST label archive")
    return np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)


def prepare(output: Path, force: bool = False) -> Path:
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    payloads = {
        key: _download(filename)
        for key, filename in FILES.items()
    }
    arrays = {
        "x_train": _images(payloads["x_train"]),
        "y_train": _labels(payloads["y_train"]),
        "x_test": _images(payloads["x_test"]),
        "y_test": _labels(payloads["y_test"]),
    }
    if arrays["x_train"].shape != (60000, 784):
        raise ValueError("unexpected Fashion-MNIST training shape")
    if arrays["x_test"].shape != (10000, 784):
        raise ValueError("unexpected Fashion-MNIST test shape")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **arrays,
        source_sha256=np.asarray([
            f"{FILES[key]}:{hashlib.sha256(payloads[key]).hexdigest()}"
            for key in FILES
        ]),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Fashion-MNIST for V31")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("data") / "fashion_mnist.npz",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    path = prepare(args.output, args.force)
    print(f"prepared {path}")


if __name__ == "__main__":
    main()

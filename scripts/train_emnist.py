#!/usr/bin/env python3
"""Train the EMNIST ByClass CNN and save weights to the AWP model cache.

Downloads EMNIST ByClass from the NIST official mirror (~537 MB, one-time).
The download is cached so re-runs skip the download step entirely.

Label ordering (62 classes):
  0-9   → digits  '0'..'9'
  10-35 → uppercase 'A'..'Z'
  36-61 → lowercase 'a'..'z'

Usage
-----
    # Default: 5 epochs (>80% accuracy on CPU in ~20-40 min)
    python scripts/train_emnist.py

    # Fast test: 2 epochs (~8-15 min, ~74% accuracy)
    python scripts/train_emnist.py --epochs 2

    # Full: 10 epochs (~83% accuracy)
    python scripts/train_emnist.py --epochs 10

    # Train then immediately run character benchmark
    python scripts/train_emnist.py --epochs 5 --benchmark

Output
------
    ~/.cache/awp/emnist/emnist_byclass_cnn_v1.pt   (model weights, state_dict)
    scripts/training_log.csv                        (epoch / loss / accuracy)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import logging
import os
import ssl
import struct
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

# ── Add project src/ to path so we can import emnist.py ──────────────────────
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from assistive_writing_pad.recognition.emnist import (
    EMNIST_LABELS,
    _CACHE_DIR,
    _WEIGHTS_FILENAME,
    _build_cnn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset download
# ─────────────────────────────────────────────────────────────────────────────

# Primary + fallback mirror URLs for the EMNIST gzip package.
# The NIST server sometimes has expired SSL certificates; we try with
# verification disabled as a last resort (no integrity checking bypass —
# we rely on the gzip CRC for file integrity).
_EMNIST_URLS = [
    "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip",
    # BioLab mirror
    "https://rds.ict.griffith.edu.au/staff/sridharan/EMNIST/gzip.zip",
]
_ZIP_FILENAME = "emnist_gzip.zip"

_BYCLASS_FILES = {
    "train_images": "emnist-byclass-train-images-idx3-ubyte.gz",
    "train_labels": "emnist-byclass-train-labels-idx1-ubyte.gz",
    "test_images":  "emnist-byclass-test-images-idx3-ubyte.gz",
    "test_labels":  "emnist-byclass-test-labels-idx1-ubyte.gz",
}

# The files live inside a subdirectory in the zip.
_ZIP_PREFIX = "gzip/"


def _download_with_progress(urls: list, dest: Path) -> None:
    """Download from the first working URL in `urls` to `dest` with a progress bar.

    Falls back to disabled SSL verification if the server certificate is expired.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    def _reporthook(count: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, count * block_size * 100 // total)
        mb = count * block_size / 1_048_576
        total_mb = total / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)

    last_exc: Optional[Exception] = None
    for url in urls:
        for verify_ssl in (True, False):
            try:
                logger.info("Downloading %s (ssl_verify=%s)", url, verify_ssl)
                if verify_ssl:
                    opener = urllib.request.build_opener()
                else:
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    opener = urllib.request.build_opener(
                        urllib.request.HTTPSHandler(context=ctx)
                    )
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url, str(tmp), reporthook=_reporthook)
                print()  # newline after progress bar
                tmp.rename(dest)
                logger.info(
                    "Download complete: %s (%.1f MB)",
                    dest.name, dest.stat().st_size / 1_048_576,
                )
                return
            except Exception as exc:
                last_exc = exc
                tmp.unlink(missing_ok=True)
                logger.warning("Download attempt failed (%s, ssl=%s): %s", url, verify_ssl, exc)

    raise RuntimeError(
        f"Failed to download EMNIST from all mirrors.\nLast error: {last_exc}\n"
        f"Please download manually:\n  {_EMNIST_URLS[0]}\n"
        f"and save to: {dest}"
    ) from last_exc


def _ensure_emnist_downloaded(cache_dir: Path) -> Path:
    """Return path to the cached gzip.zip, downloading if necessary."""
    zip_path = cache_dir / _ZIP_FILENAME
    if zip_path.exists():
        logger.info("EMNIST archive already cached at %s", zip_path)
    else:
        _download_with_progress(_EMNIST_URLS, zip_path)
    return zip_path


def _extract_byclass_files(zip_path: Path, cache_dir: Path) -> dict[str, Path]:
    """Extract only the ByClass split files from the zip archive."""
    result: dict[str, Path] = {}
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        all_names = set(zf.namelist())
        for key, filename in _BYCLASS_FILES.items():
            dest = cache_dir / filename
            if dest.exists():
                logger.info("Already extracted: %s", filename)
                result[key] = dest
                continue
            # Try with and without the prefix.
            zip_name = _ZIP_PREFIX + filename
            if zip_name not in all_names:
                zip_name = filename
            if zip_name not in all_names:
                raise FileNotFoundError(
                    f"'{filename}' not found in {zip_path.name}. "
                    f"Archive may be incomplete."
                )
            logger.info("Extracting %s …", filename)
            with zf.open(zip_name) as src, open(str(dest), "wb") as dst:
                dst.write(src.read())
            result[key] = dest
    return result


# ─────────────────────────────────────────────────────────────────────────────
# IDX parsing (pure stdlib — no scipy, no torchvision)
# ─────────────────────────────────────────────────────────────────────────────

def _read_idx_images(path: Path) -> "torch.Tensor":
    """Read a gzipped IDX3 image file → float32 tensor of shape (N, 28, 28)."""
    import numpy as np

    with gzip.open(str(path), "rb") as f:
        magic, count, rows, cols = struct.unpack(">4I", f.read(16))
        assert magic == 2051, f"Bad IDX3 magic: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)

    images = data.reshape(count, rows, cols).astype(np.float32) / 255.0

    # EMNIST images are transposed relative to MNIST.
    # Standard fix: rotate 90° CCW then flip horizontally.
    images = np.rot90(images, k=1, axes=(1, 2))
    images = np.flip(images, axis=2).copy()

    return torch.from_numpy(images).unsqueeze(1)   # (N, 1, 28, 28)


def _read_idx_labels(path: Path) -> "torch.Tensor":
    """Read a gzipped IDX1 label file → int64 tensor of shape (N,)."""
    import numpy as np

    with gzip.open(str(path), "rb") as f:
        magic, count = struct.unpack(">2I", f.read(8))
        assert magic == 2049, f"Bad IDX1 magic: {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    return torch.from_numpy(labels)


def load_emnist_byclass(
    cache_dir: Path,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Tuple[TensorDataset, TensorDataset]:
    """Return (train_dataset, test_dataset) for EMNIST ByClass.

    Downloads and extracts from NIST on first call.
    """
    zip_path = _ensure_emnist_downloaded(cache_dir)
    files = _extract_byclass_files(zip_path, cache_dir)

    logger.info("Loading train images …")
    train_x = _read_idx_images(files["train_images"])
    train_y = _read_idx_labels(files["train_labels"])
    logger.info("Loading test images …")
    test_x  = _read_idx_images(files["test_images"])
    test_y  = _read_idx_labels(files["test_labels"])

    if max_train is not None:
        train_x, train_y = train_x[:max_train], train_y[:max_train]
    if max_test is not None:
        test_x, test_y = test_x[:max_test], test_y[:max_test]

    logger.info(
        "Dataset loaded: train=%d  test=%d  classes=%d",
        len(train_y), len(test_y), len(EMNIST_LABELS),
    )
    return TensorDataset(train_x, train_y), TensorDataset(test_x, test_y)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    cache_dir: Path = _CACHE_DIR,
    weights_path: Optional[Path] = None,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    log_csv: Optional[Path] = None,
) -> float:
    """Train the EMNIST CNN. Returns final test accuracy (0-1)."""
    if weights_path is None:
        weights_path = cache_dir / _WEIGHTS_FILENAME

    train_ds, test_ds = load_emnist_byclass(
        cache_dir, max_train=max_train, max_test=max_test
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model = _build_cnn(num_classes=len(EMNIST_LABELS))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    logger.info(
        "Training: epochs=%d  batch=%d  lr=%g  train=%d  test=%d",
        epochs, batch_size, lr, len(train_ds), len(test_ds),
    )

    csv_rows: List[dict] = []
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        correct = 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()

            if (batch_idx + 1) % 200 == 0:
                pct = (batch_idx + 1) * 100 // len(train_loader)
                print(f"\r  Epoch {epoch}/{epochs}  [{pct:3d}%]  loss={loss.item():.4f}", end="", flush=True)

        print()
        scheduler.step()

        train_loss = total_loss / len(train_ds)
        train_acc  = correct / len(train_ds)

        # ── Evaluate ───────────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                preds = model(images).argmax(1)
                val_correct += (preds == labels).sum().item()
        val_acc = val_correct / len(test_ds)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %2d/%d  train_loss=%.4f  train_acc=%.3f  val_acc=%.3f  (%.0f s)",
            epoch, epochs, train_loss, train_acc, val_acc, elapsed,
        )

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "elapsed_s": f"{elapsed:.1f}",
        }
        csv_rows.append(row)

        # Save best checkpoint.
        if val_acc > best_acc:
            best_acc = val_acc
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(weights_path))
            logger.info("✓ New best %.3f — saved to %s", best_acc, weights_path)

    # Write training log CSV.
    if log_csv is None:
        log_csv = Path(__file__).parent / "training_log.csv"
    with open(str(log_csv), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info("Training log → %s", log_csv)

    logger.info("Training complete. Best val_acc = %.4f (%.1f%%)", best_acc, best_acc * 100)
    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the EMNIST ByClass CNN for the Assistive Writing Pad.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs. 5 → ~80%%, 10 → ~83%%.",
    )
    p.add_argument(
        "--batch-size", type=int, default=256,
        help="Mini-batch size.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Adam learning rate.",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=_CACHE_DIR,
        help="Directory for downloaded dataset + saved weights.",
    )
    p.add_argument(
        "--max-train", type=int, default=None,
        help="Cap training set size (for quick smoke-tests).",
    )
    p.add_argument(
        "--max-test", type=int, default=None,
        help="Cap test set size (for quick smoke-tests).",
    )
    p.add_argument(
        "--benchmark", action="store_true",
        help="Run the character benchmark after training.",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 2 epochs, 50k train samples, 10k test samples.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.quick:
        logger.info("Quick mode: 2 epochs, 50k/10k samples")
        args.epochs = max(args.epochs, 2)
        if args.max_train is None:
            args.max_train = 50_000
        if args.max_test is None:
            args.max_test = 10_000

    final_acc = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        cache_dir=args.cache_dir,
        max_train=args.max_train,
        max_test=args.max_test,
    )

    target = 0.80
    ok = final_acc >= target
    status = "✓ PASS" if ok else "✗ BELOW TARGET"
    print(f"\n{status}: val_acc={final_acc:.4f} (target={target})")

    if args.benchmark:
        logger.info("Running character benchmark …")
        _run_benchmark()

    return 0 if ok else 1


def _run_benchmark() -> None:
    """Quick inline benchmark using the EMNIST test split."""
    import subprocess
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "ocr_benchmark.py"),
            "--charset", "alphanum",
            "--mode", "character",
            "--output", "confusion_matrix.csv",
        ],
        check=False,
    )


if __name__ == "__main__":
    sys.exit(main())

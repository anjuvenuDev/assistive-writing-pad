#!/usr/bin/env python3
"""Train the EMNIST CNN using synthetically rendered characters.

No internet required. Uses Windows system fonts (Arial, Courier, etc.)
with heavy augmentation to simulate handwritten character variation.

Generates ~1,200 samples per class (74,400 total) and trains for the
requested number of epochs.

Expected accuracy: 70-78% on held-out synthetic test set.
On real handwriting the confusion-aware merge layer compensates for gaps.

Usage
-----
    python scripts/train_emnist_synthetic.py            # 5 epochs
    python scripts/train_emnist_synthetic.py --epochs 10
    python scripts/train_emnist_synthetic.py --epochs 3 --samples 600
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ── src/ on path ──────────────────────────────────────────────────────────────
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFilter, ImageFont
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
# Font discovery (Windows system fonts)
# ─────────────────────────────────────────────────────────────────────────────

_WINDOWS_FONT_NAMES = [
    "arial.ttf",
    "arialbd.ttf",      # Arial Bold
    "ariali.ttf",       # Arial Italic
    "cour.ttf",         # Courier New
    "courbd.ttf",       # Courier New Bold
    "times.ttf",        # Times New Roman
    "timesbd.ttf",
    "verdana.ttf",
    "verdanab.ttf",
    "consola.ttf",      # Consolas
    "consolab.ttf",
    "georgia.ttf",
    "georgiab.ttf",
    "trebuc.ttf",
    "trebucbd.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "comic.ttf",        # Comic Sans — actually useful for rounded letters
    "comicbd.ttf",
    "tahoma.ttf",
    "tahomabd.ttf",
    "segoeui.ttf",
]

_FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"


def _find_fonts() -> List[Path]:
    found = []
    for name in _WINDOWS_FONT_NAMES:
        p = _FONT_DIR / name
        if p.exists():
            found.append(p)
    if not found:
        logger.warning(
            "No Windows TrueType fonts found in %s. "
            "Will use PIL default bitmap font (lower quality).",
            _FONT_DIR,
        )
    logger.info("Found %d font files in %s", len(found), _FONT_DIR)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic sample generation
# ─────────────────────────────────────────────────────────────────────────────

def _render_char(
    char: str,
    font_paths: List[Path],
    rng: random.Random,
) -> np.ndarray:
    """Render a single character to a 28×28 float32 array.

    Pipeline:
      1. Render with a random font at random size onto a 56×56 canvas.
      2. Apply random affine-like augmentation (rotate, scale, translate).
      3. Apply morphological dilation/erosion to vary stroke width.
      4. Downsample to 28×28 with anti-aliasing.
      5. Add Gaussian noise.
      6. Normalise to [0, 1] (white ink on black background, like EMNIST).
    """
    SIZE = 56   # render at 2× then downsample

    # ── Font ──
    if font_paths:
        font_path = rng.choice(font_paths)
        font_size = rng.randint(28, 42)
        try:
            font = ImageFont.truetype(str(font_path), size=font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # ── Render on white canvas (black text on white) ──
    canvas = Image.new("L", (SIZE, SIZE), color=255)
    draw = ImageDraw.Draw(canvas)

    # Measure text to centre it.
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (SIZE - w) // 2 - bbox[0] + rng.randint(-4, 4)
    y = (SIZE - h) // 2 - bbox[1] + rng.randint(-4, 4)
    draw.text((x, y), char, fill=0, font=font)

    # ── Augmentation ──
    # Random rotation
    angle = rng.uniform(-18, 18)
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

    # Random scale (zoom in/out by resampling a sub-crop or super-crop)
    scale = rng.uniform(0.80, 1.20)
    new_side = max(1, int(SIZE * scale))
    if scale > 1.0:
        # Zoom out: paste onto larger canvas then crop back
        big = Image.new("L", (new_side, new_side), color=255)
        offset = (new_side - SIZE) // 2
        big.paste(canvas, (offset, offset))
        canvas = big.crop((
            (new_side - SIZE) // 2,
            (new_side - SIZE) // 2,
            (new_side - SIZE) // 2 + SIZE,
            (new_side - SIZE) // 2 + SIZE,
        ))
    else:
        # Zoom in: crop a smaller region then resize back
        margin = (SIZE - new_side) // 2
        canvas = canvas.crop((margin, margin, margin + new_side, margin + new_side))
        canvas = canvas.resize((SIZE, SIZE), Image.BILINEAR)

    # ── Stroke width variation (dilate = thicker, erode = thinner) ──
    op = rng.choice(["dilate", "erode", "none", "none"])
    if op == "dilate":
        canvas = canvas.filter(ImageFilter.MinFilter(3))   # MinFilter darkens ink
    elif op == "erode":
        canvas = canvas.filter(ImageFilter.MaxFilter(3))   # MaxFilter lightens ink

    # ── Slight blur (simulate pen tip) ──
    if rng.random() < 0.5:
        radius = rng.uniform(0.3, 0.9)
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=radius))

    # ── Downsample to 28×28 ──
    canvas = canvas.resize((28, 28), Image.LANCZOS)

    # Convert to float [0,1] and invert so ink=1, background=0 (EMNIST convention)
    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = 1.0 - arr   # invert: now ink is bright

    # ── Gaussian noise ──
    noise = np.random.normal(0, rng.uniform(0.01, 0.06), arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)

    return arr


def build_dataset(
    samples_per_class: int,
    font_paths: List[Path],
    train_fraction: float = 0.85,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    """Generate synthetic dataset, return (train_dataset, val_dataset)."""
    rng = random.Random(seed)
    np.random.seed(seed)

    n_classes = len(EMNIST_LABELS)
    total = n_classes * samples_per_class
    logger.info(
        "Generating %d synthetic samples (%d classes × %d each) …",
        total, n_classes, samples_per_class,
    )

    images: List[np.ndarray] = []
    labels: List[int] = []

    for label_int, char in enumerate(EMNIST_LABELS):
        for _ in range(samples_per_class):
            img = _render_char(char, font_paths, rng)
            images.append(img)
            labels.append(label_int)

        if (label_int + 1) % 10 == 0 or label_int == n_classes - 1:
            pct = (label_int + 1) * 100 // n_classes
            print(f"\r  Generated {label_int + 1}/{n_classes} classes ({pct}%) …", end="", flush=True)

    print()

    # Shuffle
    combined = list(zip(images, labels))
    rng.shuffle(combined)
    images, labels = zip(*combined)

    x = torch.from_numpy(np.stack(images)).unsqueeze(1)   # (N, 1, 28, 28)
    y = torch.tensor(labels, dtype=torch.long)

    split = int(len(x) * train_fraction)
    train_ds = TensorDataset(x[:split], y[:split])
    val_ds   = TensorDataset(x[split:], y[split:])

    logger.info("Dataset: train=%d  val=%d", len(train_ds), len(val_ds))
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    epochs: int = 5,
    samples_per_class: int = 1200,
    batch_size: int = 256,
    lr: float = 1e-3,
    cache_dir: Path = _CACHE_DIR,
    weights_path: Optional[Path] = None,
    log_csv: Optional[Path] = None,
) -> float:
    if weights_path is None:
        weights_path = cache_dir / _WEIGHTS_FILENAME

    font_paths = _find_fonts()
    train_ds, val_ds = build_dataset(samples_per_class, font_paths)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = _build_cnn(num_classes=len(EMNIST_LABELS))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    logger.info(
        "Training: epochs=%d  batch=%d  lr=%g  train=%d  val=%d",
        epochs, batch_size, lr, len(train_ds), len(val_ds),
    )

    csv_rows: List[dict] = []
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            if (batch_idx + 1) % 50 == 0:
                pct = (batch_idx + 1) * 100 // len(train_loader)
                print(f"\r  Epoch {epoch}/{epochs}  [{pct:3d}%]  loss={loss.item():.4f}", end="", flush=True)

        print()
        scheduler.step()

        train_loss = total_loss / len(train_ds)
        train_acc  = correct    / len(train_ds)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                val_correct += (model(images).argmax(1) == labels).sum().item()
        val_acc = val_correct / len(val_ds)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %2d/%d  train_loss=%.4f  train_acc=%.3f  val_acc=%.3f  (%.0f s)",
            epoch, epochs, train_loss, train_acc, val_acc, elapsed,
        )

        csv_rows.append({
            "epoch": epoch,
            "train_loss": f"{train_loss:.4f}",
            "train_acc":  f"{train_acc:.4f}",
            "val_acc":    f"{val_acc:.4f}",
            "elapsed_s":  f"{elapsed:.1f}",
        })

        if val_acc > best_acc:
            best_acc = val_acc
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(weights_path))
            logger.info("✓ New best %.3f — saved to %s", best_acc, weights_path)

    if log_csv is None:
        log_csv = Path(__file__).parent / "training_log_synthetic.csv"
    with open(str(log_csv), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info("Training log → %s", log_csv)
    logger.info("Best val_acc = %.4f (%.1f%%)", best_acc, best_acc * 100)
    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Train EMNIST CNN from synthetic font-rendered characters (no internet needed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs",   type=int, default=5,    help="Training epochs.")
    p.add_argument("--samples",  type=int, default=1200, help="Samples per class.")
    p.add_argument("--batch",    type=int, default=256,  help="Batch size.")
    p.add_argument("--lr",       type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--cache-dir", type=Path, default=_CACHE_DIR)
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke test: 2 epochs, 300 samples/class (~5 min).")
    args = p.parse_args()

    if args.quick:
        args.epochs  = 2
        args.samples = 300

    final_acc = train(
        epochs=args.epochs,
        samples_per_class=args.samples,
        batch_size=args.batch,
        lr=args.lr,
        cache_dir=args.cache_dir,
    )

    target = 0.70   # lower target for synthetic data
    ok = final_acc >= target
    print(f"\n{'✓ PASS' if ok else '✗ BELOW TARGET'}: val_acc={final_acc:.4f}  (target={target})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

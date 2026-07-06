#!/usr/bin/env python3
"""OCR Benchmark tool for the Assistive Writing Pad.

Two modes of operation:

1. IMAGE MODE (default)
   Processes a folder of handwriting images through the full TrOCR preprocessing
   pipeline (auto-crop, enhance_for_ocr) and outputs per-image predictions.

2. CHARSET MODE (--charset)
   Generates synthetic single-character stroke data for every character in the
   target set (A-Z, a-z, 0-9 by default), runs each through the recognizer, and
   builds a confusion matrix.  Use this to measure and track character-level
   accuracy improvements without needing hand-labelled image files.

Usage
-----
    # Image folder benchmark
    python scripts/ocr_benchmark.py --images-dir data/examples --output report.csv

    # Single-character confusion matrix
    python scripts/ocr_benchmark.py --charset alphanum --output confusion_report.csv

    # A/B: preprocessing off
    python scripts/ocr_benchmark.py --no-preprocess --images-dir data/examples

CSV columns (image mode)
------------------------
filename, prediction, confidence

CSV columns (charset mode)
--------------------------
expected, predicted, confidence, correct

Exit codes
----------
0  Success
1  No images found / dependency error
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure src/ is on the path when running from the project root.
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ocr_benchmark")

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

CHARSET_ALPHANUM = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
CHARSET_ALPHA    = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
CHARSET_DIGITS   = list("0123456789")


# ---------------------------------------------------------------------------
# Synthetic stroke generation
# ---------------------------------------------------------------------------

def _synthetic_strokes_for_char(char: str) -> List[Tuple[float, float]]:
    """Return a minimal list of (x, y) points that traces a recognisable glyph.

    These are schematic paths — not pixel-perfect, but sufficient to give TrOCR
    a meaningful input image for benchmarking purposes.

    All coordinates are in a 100x200 local space (width x height).
    """
    # Map common similar-shape characters to their stroke paths.
    # Each entry is a list of (x, y) tuples representing a continuous stroke.
    # Multiple strokes are joined with None as a separator.
    c = char
    if c in "IilL1":
        return [(50, 10), (50, 190)]
    if c in "Tt":
        return [(20, 50), (80, 50), None, (50, 10), (50, 190)]
    if c in "Hh":
        return [(20, 10), (20, 190), None, (20, 100), (80, 100), None, (80, 10), (80, 190)]
    if c in "Nn":
        return [(20, 190), (20, 10), (80, 190), (80, 10)]
    if c in "Mm":
        return [(10, 190), (10, 10), (50, 100), (90, 10), (90, 190)]
    if c in "Vv":
        return [(10, 10), (50, 190), (90, 10)]
    if c in "Ww":
        return [(10, 10), (30, 190), (50, 100), (70, 190), (90, 10)]
    if c in "Uu":
        return [(20, 10), (20, 150), (50, 190), (80, 150), (80, 10)]
    if c in "OoQq0":
        # Approximate ellipse with 12 points
        cx, cy, rx, ry = 50, 100, 35, 80
        pts = []
        for i in range(13):
            angle = 2 * math.pi * i / 12
            pts.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
        return pts
    if c in "Cc":
        cx, cy, rx, ry = 55, 100, 35, 80
        pts = []
        for i in range(10):
            angle = math.pi * 0.25 + 2 * math.pi * 0.75 * i / 9
            pts.append((cx - rx * math.cos(angle), cy + ry * math.sin(angle)))
        return pts
    if c in "Gg":
        pts = _synthetic_strokes_for_char("C")
        pts.extend([None, (55, 100), (80, 100)])
        return pts
    if c in "Ss5":
        return [(80, 30), (40, 10), (20, 40), (50, 100), (80, 160), (60, 190), (20, 170)]
    if c in "Zz7":
        return [(20, 20), (80, 20), (20, 180), (80, 180)]
    if c in "Xx":
        return [(20, 20), (80, 180), None, (80, 20), (20, 180)]
    if c in "Kk":
        return [(20, 10), (20, 190), None, (80, 10), (20, 100), (80, 190)]
    if c in "Yy":
        return [(20, 20), (50, 100), (80, 20), None, (50, 100), (50, 190)]
    if c in "Pp":
        return [(20, 10), (20, 190), None, (20, 10), (70, 10), (80, 50), (70, 90), (20, 90)]
    if c in "Bb":
        return [(20, 10), (20, 190), None, (20, 10), (70, 10), (80, 50), (70, 100),
                (20, 100), None, (20, 100), (70, 100), (80, 150), (70, 190), (20, 190)]
    if c in "Rr":
        return [(20, 10), (20, 190), None, (20, 10), (70, 10), (80, 50), (70, 100), (20, 100), (80, 190)]
    if c in "Ff":
        return [(20, 10), (80, 10), None, (20, 10), (20, 190), None, (20, 100), (70, 100)]
    if c in "Ee":
        return [(80, 10), (20, 10), (20, 190), (80, 190), None, (20, 100), (70, 100)]
    if c in "Dd":
        return [(20, 10), (20, 190), (70, 190), (85, 140), (85, 60), (70, 10), (20, 10)]
    if c in "Aa":
        return [(50, 10), (20, 190), None, (50, 10), (80, 190), None, (30, 120), (70, 120)]
    if c == "i":
        return [(50, 60), (50, 190), None, (50, 20), (50, 35)]
    if c == "j":
        return [(50, 60), (50, 170), (40, 190), (30, 175), None, (50, 20), (50, 35)]
    if c in "89":
        cx, cy, rx, ry = 50, 100, 35, 80
        pts = []
        for i in range(13):
            angle = 2 * math.pi * i / 12
            pts.append((cx + rx * math.cos(angle), cy + ry / 2 * math.sin(angle) - 40))
        pts.append(None)
        for i in range(13):
            angle = 2 * math.pi * i / 12
            pts.append((cx + rx * math.cos(angle), cy + ry / 2 * math.sin(angle) + 40))
        return pts
    if c == "4":
        return [(60, 10), (20, 130), (80, 130), None, (60, 10), (60, 190)]
    if c == "6":
        return [(70, 20), (30, 80), (20, 130), (30, 170), (60, 190), (80, 160), (80, 120), (60, 100), (30, 105)]
    if c == "2":
        return [(20, 50), (40, 20), (70, 30), (75, 60), (20, 130), (20, 180), (80, 180)]
    if c == "3":
        return [(20, 30), (60, 10), (80, 50), (60, 100), (80, 150), (60, 190), (20, 170)]
    # Default: draw a vertical line for unrecognised characters.
    return [(50, 20), (50, 180)]


def _strokes_to_stroke_points(raw_points):
    """Convert raw (x, y) / None list to StrokePoint groups."""
    from assistive_writing_pad.contracts import StrokePoint
    groups = []
    current: List = []
    t = 0
    for pt in raw_points:
        if pt is None:
            if current:
                groups.append(current)
                current = []
        else:
            x, y = pt
            current.append(StrokePoint(x=x, y=y, timestamp_ms=t, pressure=1.0))
            t += 16
    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# Image mode helpers
# ---------------------------------------------------------------------------


def _load_image_as_rgb(path: Path):
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"cv2.imread returned None for {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _run_trocr(processor, model, torch_mod, image_rgb, max_new_tokens: int = 48):
    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch_mod.no_grad():
        generated = model.generate(
            inputs.pixel_values,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
    raw_text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
    scores = getattr(generated, "scores", None)
    if scores:
        import numpy as np
        token_confs = [float(torch_mod.softmax(s, dim=-1).max().item()) for s in scores]
        confidence = float(np.mean(token_confs)) if token_confs else 0.0
    else:
        confidence = 0.0
    return raw_text.strip(), confidence


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the Assistive Writing Pad recognizers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images-dir", default="data/examples",
                        help="Directory of PNG/JPG images (image mode).")
    parser.add_argument("--output", default="benchmark_report.csv",
                        help="CSV output path.")
    parser.add_argument("--model",
                        default=os.environ.get("AWP_TROCR_MODEL", "microsoft/trocr-small-handwritten"),
                        help="HuggingFace TrOCR checkpoint.")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip auto-crop + enhance_for_ocr (image mode A/B).")
    parser.add_argument("--max-tokens", type=int, default=48,
                        help="max_new_tokens for TrOCR generation (image mode).")
    parser.add_argument("--charset", choices=["alpha", "digits", "alphanum"],
                        default=None,
                        help="If set, run confusion-matrix benchmark over this character set.")
    parser.add_argument("--mode", choices=["auto", "character", "word"], default="character",
                        help="Recognition mode for charset benchmark (default: character).")
    parser.add_argument(
        "--samples-per-class", type=int, default=50,
        help="(character mode) Number of EMNIST test images to sample per class (default: 50)."
    )
    args = parser.parse_args()

    if args.charset:
        if args.mode == "character":
            # Use real EMNIST test images — no TrOCR involved.
            _run_emnist_image_benchmark(args)
        else:
            # Synthetic strokes through TrOCR (word mode).
            _run_charset_benchmark(args)
    else:
        _run_image_benchmark(args)


def _run_charset_benchmark(args) -> None:
    """Confusion-matrix benchmark using synthetic strokes → TrOCR (word mode)."""
    from assistive_writing_pad.recognition.trocr import TrOCRHandwritingRecognizer

    charset_map = {"alpha": CHARSET_ALPHA, "digits": CHARSET_DIGITS, "alphanum": CHARSET_ALPHANUM}
    charset = charset_map.get(args.charset, CHARSET_ALPHANUM)

    recognizer = TrOCRHandwritingRecognizer(model_name=args.model)
    mode = args.mode

    logger.info("Charset benchmark (synthetic strokes): %d characters, mode=%s", len(charset), mode)

    results = []
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    col_w = 10

    print()
    print(f"{'EXPECTED':<{col_w}}  {'PREDICTED':<{col_w}}  {'CONF':>6}  {'OK':>4}")
    print("-" * (col_w * 2 + 18))

    n_correct = 0
    for expected in charset:
        raw_pts = _synthetic_strokes_for_char(expected)
        stroke_groups = _strokes_to_stroke_points(raw_pts)
        try:
            result = recognizer.recognize_stroke_groups(stroke_groups, mode=mode)
            predicted = (result.text or "").strip()
            confidence = result.confidence
        except Exception as exc:
            logger.warning("Failed for %r: %s", expected, exc)
            predicted = "ERROR"
            confidence = 0.0

        correct = predicted == expected
        if correct:
            n_correct += 1
        confusion[expected][predicted] += 1

        ok_sym = "✓" if correct else "✗"
        print(f"{expected!r:<{col_w}}  {predicted!r:<{col_w}}  {confidence:>5.2f}  {ok_sym}")
        results.append({
            "expected": expected,
            "predicted": predicted,
            "confidence": round(confidence, 4),
            "correct": "1" if correct else "0",
        })

    _write_charset_csv(args.output, charset, results, confusion)
    accuracy = n_correct / len(charset) * 100 if charset else 0.0
    print(f"\nAccuracy: {n_correct}/{len(charset)} = {accuracy:.1f}%\n")


def _run_emnist_image_benchmark(args) -> None:
    """Character-mode benchmark: classify real EMNIST test images.

    Loads the EMNIST ByClass test split, samples `--samples-per-class`
    images per class, runs each through EMNISTCharacterRecognizer, and
    produces:
      - <output>.csv           per-sample results
      - <output>_confusion.csv 62×62 confusion matrix
      - <output>_accuracy.txt  accuracy-by-class report
    """
    import gzip
    import struct
    import random
    import torch
    import numpy as np

    from assistive_writing_pad.recognition.emnist import (
        EMNIST_LABELS,
        _CACHE_DIR,
        EMNISTCharacterRecognizer,
    )

    # ── Load EMNIST test split ───────────────────────────────────────────────
    cache_dir = _CACHE_DIR
    test_img_path  = cache_dir / "emnist-byclass-test-images-idx3-ubyte.gz"
    test_lbl_path  = cache_dir / "emnist-byclass-test-labels-idx1-ubyte.gz"

    if not test_img_path.exists() or not test_lbl_path.exists():
        logger.error(
            "EMNIST test split not found in %s.\n"
            "Run first: python scripts/train_emnist.py\n"
            "(downloads dataset as a side effect)",
            cache_dir,
        )
        sys.exit(1)

    logger.info("Reading EMNIST test images from %s …", cache_dir)

    def _read_images(path):
        with gzip.open(str(path), "rb") as f:
            magic, count, rows, cols = struct.unpack(">4I", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        imgs = data.reshape(count, rows, cols).astype(np.float32) / 255.0
        # Standard EMNIST orientation fix
        imgs = np.rot90(imgs, k=1, axes=(1, 2))
        imgs = np.flip(imgs, axis=2).copy()
        return imgs  # (N, 28, 28)

    def _read_labels(path):
        with gzip.open(str(path), "rb") as f:
            magic, count = struct.unpack(">2I", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    test_images = _read_images(test_img_path)
    test_labels = _read_labels(test_lbl_path)
    logger.info("Test set: %d images, %d classes", len(test_images), len(EMNIST_LABELS))

    # ── Sample per class ────────────────────────────────────────────────────
    charset_map = {"alpha": CHARSET_ALPHA, "digits": CHARSET_DIGITS, "alphanum": CHARSET_ALPHANUM}
    charset_chars = charset_map.get(args.charset, CHARSET_ALPHANUM)
    charset_set   = set(charset_chars)

    # Build index: label_int -> list of image indices
    label_to_indices: Dict[int, list] = defaultdict(list)
    for idx, lbl in enumerate(test_labels):
        if EMNIST_LABELS[lbl] in charset_set:
            label_to_indices[int(lbl)].append(idx)

    spc = getattr(args, "samples_per_class", 50)
    random.seed(42)
    sampled: list[tuple[int, int]] = []  # (image_idx, label_int)
    for lbl_int, indices in sorted(label_to_indices.items()):
        k = min(spc, len(indices))
        sampled.extend((i, lbl_int) for i in random.sample(indices, k))
    random.shuffle(sampled)

    logger.info(
        "Running %d samples (%d per class) through EMNISTCharacterRecognizer …",
        len(sampled), spc,
    )

    # ── Classify ────────────────────────────────────────────────────────────
    rec = EMNISTCharacterRecognizer()

    results   = []
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_correct = 0
    t_total   = 0.0

    col_w = 12
    print(f"\n{'EXPECTED':<{col_w}}  {'PREDICTED':<{col_w}}  {'CONF':>6}  {'OK':>4}")
    print("-" * (col_w * 2 + 20))

    for sample_idx, (img_idx, lbl_int) in enumerate(sampled):
        expected  = EMNIST_LABELS[lbl_int]
        img28     = test_images[img_idx]          # (28, 28) float32

        t0 = time.perf_counter()
        try:
            result = _classify_emnist_image(rec, img28)
            predicted  = result.text
            confidence = result.confidence
        except Exception as exc:
            logger.warning("Error classifying sample %d: %s", sample_idx, exc)
            predicted  = "ERROR"
            confidence = 0.0
        t_total += time.perf_counter() - t0

        correct = predicted == expected
        if correct:
            n_correct += 1
        confusion[expected][predicted] += 1

        if sample_idx < 30 or not correct:  # print first 30 + all errors
            ok_sym = "✓" if correct else "✗"
            print(f"{expected!r:<{col_w}}  {predicted!r:<{col_w}}  {confidence:>5.2f}  {ok_sym}")

        results.append({
            "expected":   expected,
            "predicted":  predicted,
            "confidence": round(confidence, 4),
            "correct":    "1" if correct else "0",
        })

    # ── Report ───────────────────────────────────────────────────────────────
    total     = len(results)
    accuracy  = n_correct / total * 100 if total else 0.0
    avg_ms    = (t_total / total * 1000) if total else 0.0
    print("-" * (col_w * 2 + 20))
    print(f"\nSamples:  {total}")
    print(f"Correct:  {n_correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Avg time: {avg_ms:.1f} ms/sample")

    # Accuracy by class
    print("\nAccuracy by class:")
    per_class: Dict[str, tuple] = {}
    for lbl_int in sorted(label_to_indices):
        ch  = EMNIST_LABELS[lbl_int]
        tot = sum(confusion[ch].values())
        ok  = confusion[ch].get(ch, 0)
        per_class[ch] = (ok, tot, ok / tot * 100 if tot else 0.0)
    for ch, (ok, tot, pct) in sorted(per_class.items(), key=lambda x: x[1][2]):
        bar = "█" * int(pct / 5) + " " * (20 - int(pct / 5))
        print(f"  {ch!r}  [{bar}] {pct:5.1f}%  ({ok}/{tot})")

    # Top confusions
    print("\nTop confusions (expected → predicted):")
    errors = [
        (exp, pred, cnt)
        for exp, preds in confusion.items()
        for pred, cnt in preds.items()
        if pred != exp
    ]
    errors.sort(key=lambda x: -x[2])
    for exp, pred, cnt in errors[:15]:
        print(f"  {exp!r:>4} → {pred!r:<4}  ({cnt}x)")

    _write_charset_csv(args.output, list(charset_set), results, confusion)

    # Accuracy report txt
    out_path = Path(args.output)
    acc_path = out_path.with_name(out_path.stem + "_accuracy.txt")
    with open(str(acc_path), "w", encoding="utf-8") as f:
        f.write(f"EMNIST ByClass Character Accuracy Report\n")
        f.write(f"Samples per class: {spc}\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Overall accuracy: {accuracy:.2f}%\n\n")
        f.write("Per-class accuracy:\n")
        for ch, (ok, tot, pct) in sorted(per_class.items()):
            f.write(f"  {ch!r:>4}  {pct:6.2f}%  ({ok}/{tot})\n")
        f.write("\nTop 15 confusions:\n")
        for exp, pred, cnt in errors[:15]:
            f.write(f"  {exp!r} → {pred!r}  ({cnt}x)\n")
    logger.info("Accuracy report → %s", acc_path)


def _classify_emnist_image(rec, img28) -> object:
    """Run one 28×28 numpy image through the EMNIST recognizer directly.

    Bypasses the StrokePreprocessor by injecting the pre-loaded image
    into _run_cnn() or _use_fallback path.
    """
    import numpy as np

    rec._ensure_loaded()

    if rec._use_fallback or rec._model is None:
        from assistive_writing_pad.recognition.emnist import _pixel_classify
        candidates = _pixel_classify(img28)
        top = candidates[0]
        from assistive_writing_pad.contracts import RecognitionResult, CharacterConfidence
        return RecognitionResult(
            text=top[0],
            confidence=top[1],
            character_confidences=tuple(
                CharacterConfidence(character=c, confidence=s) for c, s in candidates
            ),
            metadata={"recognizer": "emnist", "model": "heuristic_fallback"},
        )

    return rec._run_cnn(img28)


def _write_charset_csv(
    output: str,
    charset: list,
    results: list,
    confusion: Dict[str, Dict[str, int]],
) -> None:
    """Write per-sample CSV and 62×62 confusion matrix CSV."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["expected", "predicted", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(results)

    # Full confusion matrix CSV
    cm_path = output_path.with_name(output_path.stem + "_confusion.csv")
    all_chars = sorted(set(charset))
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["expected\\predicted"] + all_chars)
        for exp in all_chars:
            row = [exp] + [confusion[exp].get(pred, 0) for pred in all_chars]
            writer.writerow(row)

    logger.info("Report:           %s", output_path)
    logger.info("Confusion matrix: %s", cm_path)


def _run_image_benchmark(args) -> None:
    """Image folder benchmark (original functionality)."""
    import torch
    import numpy as np
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    if not image_paths:
        logger.error("No images found in %s", images_dir)
        sys.exit(1)

    logger.info("Found %d image(s) in %s", len(image_paths), images_dir)
    logger.info("Loading model: %s", args.model)

    processor = TrOCRProcessor.from_pretrained(args.model, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(args.model, low_cpu_mem_usage=False)
    model.to(torch.device("cpu"))
    model.eval()

    if not args.no_preprocess:
        from assistive_writing_pad.preprocessing.ocr_image_ops import (
            auto_crop_handwriting, enhance_for_ocr,
        )
        logger.info("Preprocessing: ENABLED")
    else:
        logger.info("Preprocessing: DISABLED")

    results = []
    col_w = max(len(p.name) for p in image_paths) + 2
    total_conf = 0.0

    print()
    print(f"{'FILENAME':<{col_w}}  {'PREDICTION':<30}  {'CONF':>6}")
    print("-" * (col_w + 42))

    for img_path in image_paths:
        t0 = time.perf_counter()
        try:
            image_rgb = _load_image_as_rgb(img_path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", img_path.name, exc)
            continue

        if not args.no_preprocess:
            image_rgb = auto_crop_handwriting(image_rgb, padding=20)
            image_rgb = enhance_for_ocr(image_rgb)

        prediction, confidence = _run_trocr(
            processor, model, torch, image_rgb, max_new_tokens=args.max_tokens
        )
        elapsed = time.perf_counter() - t0

        indicator = "[HIGH]" if confidence > 0.85 else ("[MED] " if confidence >= 0.65 else "[LOW] ")
        display_pred = prediction[:28] + ".." if len(prediction) > 30 else prediction
        print(f"{img_path.name:<{col_w}}  {indicator} {display_pred:<28}  {confidence:>5.2f}  ({elapsed:.1f}s)")

        total_conf += confidence
        results.append({"filename": img_path.name, "prediction": prediction, "confidence": round(confidence, 4)})

    print("-" * (col_w + 42))
    if results:
        print(f"\nSummary: {len(results)} image(s)  |  avg confidence: {total_conf / len(results):.3f}\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    logger.info("CSV report written to: %s", output_path)


if __name__ == "__main__":
    main()

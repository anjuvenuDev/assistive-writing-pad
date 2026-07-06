"""EMNIST-based single-character classifier.

Uses a lightweight CNN built directly with PyTorch (no torchvision required).
Weights are fetched from a public release on first use and cached locally.

The classifier is trained on EMNIST ByClass (62 classes: 0-9, A-Z, a-z).
The model architecture is a simple 2-block CNN that runs in ~5 ms on CPU.

Label ordering follows the EMNIST ByClass mapping:
  0-9  -> digits 0..9
  10-35 -> uppercase A..Z
  36-61 -> lowercase a..z

After CNN inference a confusion-aware merge step is applied:
  • Any confusion-map sibling of the top-1 prediction that is NOT already in
    the CNN top-5 is appended with a decayed confidence score.
  • The final list is re-sorted and trimmed to 5 candidates.
  • Whenever the corrected top-1 differs from the raw CNN top-1, the pair is
    logged at INFO level as a confusion_pair event.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from assistive_writing_pad.contracts import CharacterConfidence, RecognitionResult, StrokePoint
from assistive_writing_pad.preprocessing.pipeline import StrokePreprocessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EMNIST ByClass label map  (62 classes)
# ---------------------------------------------------------------------------
_DIGITS = list("0123456789")
_UPPER  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
EMNIST_LABELS: List[str] = _DIGITS + _UPPER   # 10 + 52 = 62 classes

# ---------------------------------------------------------------------------
# Cache directory for model weights
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(os.environ.get("AWP_MODEL_CACHE", Path.home() / ".cache" / "awp" / "emnist"))

# Public release URL for a pre-trained emnist-byclass small CNN checkpoint.
# This is a self-contained 1.6 MB .pt file (state_dict, CPU float32).
_WEIGHTS_URL  = (
    "https://github.com/anjuvenuDev/assistive-writing-pad-models/releases/download/v1.0/"
    "emnist_byclass_cnn.pt"
)
_WEIGHTS_SHA256 = ""   # Set to non-empty to enable integrity check.
_WEIGHTS_FILENAME = "emnist_byclass_cnn_v1.pt"


# ---------------------------------------------------------------------------
# Tiny CNN model definition (matches the pre-trained checkpoint)
# ---------------------------------------------------------------------------
def _build_cnn(num_classes: int = 62):
    """Build the lightweight EMNIST CNN in pure PyTorch.

    Architecture:
      Conv(1→32, 3x3) -> BN -> ReLU -> MaxPool(2)   -> 14x14
      Conv(32→64, 3x3) -> BN -> ReLU -> MaxPool(2)  -> 7x7
      Dropout(0.25)
      Linear(64*7*7=3136 -> 256) -> ReLU -> Dropout(0.5)
      Linear(256 -> num_classes)
    """
    import torch
    import torch.nn as nn

    class _CNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                          # 14×14
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                          # 7×7
                nn.Dropout2d(0.25),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return _CNN()


# ---------------------------------------------------------------------------
# Fallback: rule-based classifier (no model weights needed)
# ---------------------------------------------------------------------------
def _pixel_classify(img28: np.ndarray) -> List[tuple[str, float]]:
    """Very simple pixel-count heuristic for when the CNN is unavailable.

    Returns top-5 dummy candidates so the UI still functions.
    """
    # Aspect ratio of bounding box
    rows = np.any(img28 > 0.1, axis=1)
    cols = np.any(img28 > 0.1, axis=0)
    if not rows.any():
        return [("?", 0.0)]
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    h = max(1, int(r1 - r0))
    w = max(1, int(c1 - c0))
    aspect = w / h

    density = float(img28[r0:r1+1, c0:c1+1].mean())
    candidates: List[tuple[str, float]] = []

    if density < 0.15:
        candidates = [("i", 0.3), ("l", 0.25), ("1", 0.2), ("t", 0.15), ("I", 0.1)]
    elif aspect > 1.5:
        candidates = [("w", 0.3), ("m", 0.25), ("u", 0.2), ("n", 0.15), ("W", 0.1)]
    elif aspect < 0.5:
        candidates = [("l", 0.3), ("1", 0.25), ("I", 0.2), ("i", 0.15), ("|", 0.1)]
    else:
        candidates = [("a", 0.2), ("e", 0.2), ("o", 0.15), ("c", 0.15), ("s", 0.15)]
    return candidates


class EMNISTCharacterRecognizer:
    """Classify a single handwritten character using an EMNIST-ByClass CNN.

    On first call to ``recognize()`` the model is lazily loaded:
    1. Try to load cached weights from ``_CACHE_DIR``.
    2. If absent, attempt to download them.
    3. If download fails (offline / no URL), fall back to the pixel heuristic.
    """

    def __init__(self, model_name: str = "awp/emnist-byclass-cnn-v1") -> None:
        self.model_name = model_name
        self._pipeline = None   # kept for test compatibility; True = loaded, None = not yet
        self._model = None      # actual nn.Module
        self._preprocessor = StrokePreprocessor()
        self._use_fallback = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, strokes: Sequence[StrokePoint]) -> RecognitionResult:
        if not strokes:
            return RecognitionResult(
                text="",
                confidence=0.0,
                metadata={"recognizer": "emnist", "reason": "empty_strokes"},
            )

        self._ensure_loaded()

        # Preprocess strokes -> 28×28 float32 np array, values in [0, 1]
        preprocessed = self._preprocessor.preprocess(strokes)
        img28 = preprocessed.image  # shape (28, 28)

        # ── Mode banner ───────────────────────────────────────────────────────
        if self._use_fallback or self._model is None:
            print("\nEMNIST MODE: HEURISTIC")
            print(f"  reason : weights not loaded (use_fallback={self._use_fallback}, model={self._model is not None})")
            candidates = _pixel_classify(img28)
            top = candidates[0]
            return RecognitionResult(
                text=top[0],
                confidence=top[1],
                character_confidences=tuple(
                    CharacterConfidence(character=c, confidence=s) for c, s in candidates
                ),
                metadata={"recognizer": "emnist", "model": "heuristic_fallback"},
            )

        print("\nEMNIST MODE: CNN")
        print(f"  model : {self.model_name}")
        print(f"  path  : {_CACHE_DIR / _WEIGHTS_FILENAME}")
        return self._run_cnn(img28)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load: called once per recognizer lifetime."""
        if self._pipeline is not None:
            return
        self._pipeline = True   # mark as "attempted"
        self._model = self._load_model()

    def _load_model(self):
        """Try to load CNN weights. Returns nn.Module or None on failure."""
        import torch

        weights_path = _CACHE_DIR / _WEIGHTS_FILENAME

        # ── Diagnostic header ────────────────────────────────────────────────
        print("\n" + "-" * 60)
        print("EMNIST MODEL LOAD")
        print(f"  weights file : {_WEIGHTS_FILENAME}")
        print(f"  full path    : {weights_path}")
        print(f"  file exists  : {weights_path.exists()}")
        print("-" * 60)
        # ─────────────────────────────────────────────────────────────────────

        if not weights_path.exists():
            logger.info("EMNIST weights not found locally; attempting download …")
            ok = self._download_weights(weights_path)
            if not ok:
                self._use_fallback = True
                print("  load status  : FAILED (download failed)")
                print("  active mode  : HEURISTIC (pixel fallback)")
                print("-" * 60 + "\n")
                logger.warning(
                    "EMNIST weight download failed. "
                    "Character mode will use the pixel-heuristic fallback."
                )
                return None

        try:
            model = _build_cnn(num_classes=len(EMNIST_LABELS))
            state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            print(f"  load status  : OK")
            print(f"  active mode  : CNN")
            print("-" * 60 + "\n")
            logger.info("EMNIST CNN loaded from %s", weights_path)
            return model
        except Exception as exc:
            self._use_fallback = True
            print(f"  load status  : FAILED ({exc})")
            print(f"  active mode  : HEURISTIC (pixel fallback)")
            print("-" * 60 + "\n")
            logger.error("Failed to load EMNIST CNN weights: %s", exc)
            return None

    @staticmethod
    def _download_weights(dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Downloading EMNIST weights from %s", _WEIGHTS_URL)
            urllib.request.urlretrieve(_WEIGHTS_URL, str(dest))
            if _WEIGHTS_SHA256:
                sha = hashlib.sha256(dest.read_bytes()).hexdigest()
                if sha != _WEIGHTS_SHA256:
                    logger.error("Checksum mismatch for EMNIST weights (got %s)", sha)
                    dest.unlink(missing_ok=True)
                    return False
            return True
        except Exception as exc:
            logger.warning("EMNIST weight download failed: %s", exc)
            dest.unlink(missing_ok=True)
            return False

    def _run_cnn(self, img28: np.ndarray) -> RecognitionResult:
        """Run forward pass through the CNN, then apply confusion-aware merge."""
        import torch
        import torch.nn.functional as F

        # img28: (28, 28) float32, ink=1 bg=0
        # EMNIST training convention: white ink on black background (same as our preprocessor)
        tensor = torch.from_numpy(img28).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 28, 28)

        with torch.no_grad():
            logits = self._model(tensor)             # (1, 62)
            probs  = F.softmax(logits, dim=1)[0]     # (62,)

        top_k = min(5, len(EMNIST_LABELS))
        top_vals, top_idxs = torch.topk(probs, top_k)
        cnn_candidates: List[Tuple[str, float]] = [
            (EMNIST_LABELS[idx.item()], float(val.item()))
            for val, idx in zip(top_vals, top_idxs)
        ]

        # Apply confusion-aware merge to augment CNN top-5 with known confusion
        # siblings that the CNN may be systematically overconfident against.
        merged = _apply_confusion_merge(cnn_candidates)

        text       = merged[0][0]
        confidence = merged[0][1]
        cnn_top1   = cnn_candidates[0][0]

        if text != cnn_top1:
            logger.info(
                "confusion_pair: cnn=%r conf=%.3f → correction=%r (confusion map)",
                cnn_top1, cnn_candidates[0][1], text,
            )

        char_confs = tuple(
            CharacterConfidence(character=c, confidence=s) for c, s in merged
        )

        logger.info(
            "emnist: top=%r conf=%.3f top5=%s",
            text, confidence, [(c.character, round(c.confidence, 3)) for c in char_confs]
        )

        return RecognitionResult(
            text=text,
            confidence=confidence,
            character_confidences=char_confs,
            metadata={"recognizer": "emnist", "model": self.model_name},
        )


# ---------------------------------------------------------------------------
# Confusion-aware merge
# ---------------------------------------------------------------------------

def _apply_confusion_merge(
    cnn_candidates: List[Tuple[str, float]],
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """Merge CNN top-5 with confusion-map siblings of the top-1 prediction.

    Algorithm:
      1. Start with the CNN top-5 as a dict {char: confidence}.
      2. For the top-1 char, look up first-hop confusion siblings.
      3. Case-pair siblings (c↔C, f↔F, m↔M, etc.) receive a high multiplier
         (0.90×) so they appear near rank-2 in the results.
      4. Other first-hop siblings receive 0.65×.
      5. Second-hop siblings receive 0.45× (only when top-1 conf < 0.70).
      6. Re-sort descending and return top_n entries.

    Benchmark-confirmed: EMNIST confuses lowercase/uppercase for visually
    similar pairs (c/C, f/F, m/M, o/O, s/S, u/U). The 0.90× multiplier
    ensures the opposite-case always appears as a close alternative.
    """
    try:
        from assistive_writing_pad.recognition.confusion import CONFUSION_MAP
    except ImportError:
        return cnn_candidates[:top_n]

    if not cnn_candidates:
        return []

    # Case pairs confirmed by benchmark (systematic CNN confusion).
    _CASE_PAIRS: Dict[str, str] = {
        # lower → upper
        "a": "A", "b": "B", "c": "C", "d": "D", "e": "E",
        "f": "F", "g": "G", "h": "H", "i": "I", "j": "J",
        "k": "K", "l": "L", "m": "M", "n": "N", "o": "O",
        "p": "P", "q": "Q", "r": "R", "s": "S", "t": "T",
        "u": "U", "v": "V", "w": "W", "x": "X", "y": "Y", "z": "Z",
    }
    # Make bidirectional
    _CASE_PAIRS.update({v: k for k, v in _CASE_PAIRS.items()})

    scores: Dict[str, float] = {c: s for c, s in cnn_candidates}
    top1_char, top1_conf = cnn_candidates[0]

    # First-hop siblings — higher multiplier for case pairs.
    for sibling in CONFUSION_MAP.get(top1_char, set()):
        is_case_pair = _CASE_PAIRS.get(top1_char) == sibling
        multiplier = 0.90 if is_case_pair else 0.65
        candidate_conf = top1_conf * multiplier
        if sibling not in scores:
            scores[sibling] = candidate_conf
        elif is_case_pair and scores[sibling] < candidate_conf:
            # Always boost case pair to at least the multiplier level.
            scores[sibling] = candidate_conf

    # Always inject the direct case pair if it's not already present.
    case_partner = _CASE_PAIRS.get(top1_char)
    if case_partner and case_partner not in scores:
        scores[case_partner] = top1_conf * 0.90

    # Second-hop (only when top1 confidence is uncertain).
    if top1_conf < 0.70:
        for sibling in CONFUSION_MAP.get(top1_char, set()):
            for sib2 in CONFUSION_MAP.get(sibling, set()):
                if sib2 == top1_char:
                    continue
                candidate_conf = top1_conf * 0.45
                if sib2 not in scores:
                    scores[sib2] = candidate_conf

    sorted_candidates = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_candidates[:top_n]

"""OCR debug image saver.

When the environment variable AWP_DEBUG_OCR=1 is set, this module saves the
three pipeline stages (raw render, auto-cropped, fully preprocessed) to
data/debug/ for visual inspection.  On a production Raspberry Pi deployment
the env var is unset, so there is zero file-system overhead in normal use.

Usage (set before launching the web server)::

    AWP_DEBUG_OCR=1 python -m assistive_writing_pad.display.web_app

Saved files per recognition call::

    data/debug/<timestamp>_raw_canvas.png
    data/debug/<timestamp>_cropped.png
    data/debug/<timestamp>_processed.png
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Root of the debug output directory, relative to the process working directory.
_DEBUG_DIR = Path("data") / "debug"

# Check once at import time whether debug mode is active.
_DEBUG_ENABLED: bool = os.environ.get("AWP_DEBUG_OCR", "0").strip() == "1"


def is_debug_enabled() -> bool:
    """Return True when AWP_DEBUG_OCR=1 is set in the environment."""
    return _DEBUG_ENABLED


def save_debug_images(
    raw_rgb: np.ndarray,
    cropped_rgb: np.ndarray,
    processed_rgb: np.ndarray,
    label: str = "",
) -> None:
    """Save the three OCR pipeline stages as PNG files under data/debug/.

    This function is a no-op when AWP_DEBUG_OCR is not set to "1".

    Args:
        raw_rgb:      Image as rendered by render_stroke_groups_for_trocr()
                      before any preprocessing (uint8, HxWx3 RGB).
        cropped_rgb:  Image after auto_crop_handwriting() (uint8, HxWx3 RGB).
        processed_rgb: Image after enhance_for_ocr() (uint8, HxWx3 RGB).
        label:        Optional suffix appended to filenames for disambiguation
                      when multiple recognition calls happen in one session.
    """
    if not _DEBUG_ENABLED:
        return

    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # Use a millisecond timestamp so files sort chronologically and never
    # overwrite each other in rapid-fire recognition sessions.
    ts = int(time.time() * 1000)
    suffix = f"_{label}" if label else ""

    files = {
        f"{ts}{suffix}_raw_canvas.png": raw_rgb,
        f"{ts}{suffix}_cropped.png": cropped_rgb,
        f"{ts}{suffix}_processed.png": processed_rgb,
    }

    for filename, image in files.items():
        path = _DEBUG_DIR / filename
        # cv2 expects BGR; convert from RGB before writing.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(path), bgr)
        if success:
            h, w = image.shape[:2]
            logger.info("debug_saver: saved %s (%dx%d)", path, w, h)
        else:
            logger.error("debug_saver: FAILED to write %s", path)

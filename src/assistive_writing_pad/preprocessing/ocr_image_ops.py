"""TrOCR-specific image preprocessing operations.

These functions operate on *RGB NumPy arrays* (dtype=uint8, shape HxWx3) that
have already been rendered by render_stroke_groups_for_trocr() and are about to
be fed into the TrOCR processor.  They are intentionally separate from the
rasterize / image_ops pipeline used by the template recogniser so that we can
tune them independently without risk of regression.

All operations are CPU-only and Raspberry Pi 4 compatible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InkCropResult:
    """Validated handwriting crop and diagnostics used by the OCR boundary."""

    image: np.ndarray
    bbox: Tuple[int, int, int, int]
    ink_pixels: int
    ink_coverage: float
    original_ink_coverage: float
    valid: bool
    original_size: Tuple[int, int]

    @property
    def crop_size(self) -> Tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    def __getattr__(self, name: str):
        # Preserve the old ndarray-shaped API for callers that only inspect shape/dtype.
        return getattr(self.image, name)


def _as_white_background_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = np.clip(image, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("OCR image must be a grayscale, RGB, or RGBA array")

    source = np.clip(image, 0, 255).astype(np.uint8)
    if source.shape[2] == 3:
        return source.copy()

    rgb = source[:, :, :3].astype(np.float32)
    alpha = source[:, :, 3:4].astype(np.float32) / 255.0
    composited = rgb * alpha + 255.0 * (1.0 - alpha)
    return np.rint(composited).astype(np.uint8)


def auto_crop_handwriting(
    image_rgb: np.ndarray,
    padding: Optional[int] = None,
) -> InkCropResult:
    """Detect meaningful ink and crop it with bounded, proportional padding.

    Empty whitespace sent to TrOCR confuses the model because it sees a very
    small ink region surrounded by irrelevant background tokens.  Cropping
    ensures the ink fills as much of the input as possible.

    Args:
        image_rgb: White-background RGB image (uint8, HxWx3).
        padding:   Explicit border size. If omitted, padding is proportional to
                   the ink bounding box and clamped to a useful range.

    Returns:
        Crop image, bounding box, coverage, and validity diagnostics.
    """
    image = _as_white_background_rgb(image_rgb)
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Include anti-aliased/pressure-light pixels while excluding near-white noise.
    ink_mask = (gray < 245).astype(np.uint8)

    total_pixels = h * w
    raw_ink_pixels = int(np.count_nonzero(ink_mask))
    components, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, 8)
    min_component_pixels = max(10, int(round(total_pixels * 0.00005)))
    meaningful = np.zeros_like(ink_mask)
    for component in range(1, components):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area >= min_component_pixels:
            meaningful[labels == component] = 1

    ink_pixels = int(np.count_nonzero(meaningful))
    ink_ratio = ink_pixels / max(total_pixels, 1)
    rows, cols = np.where(meaningful)
    if ink_pixels == 0:
        if raw_ink_pixels:
            logger.warning(
                "auto_crop: rejected insignificant ink in %dx%d image; "
                "large whitespace may hurt OCR accuracy (pixels=%d)",
                w, h, raw_ink_pixels,
            )
        logger.info("auto_crop: no meaningful ink detected in %dx%d image", w, h)
        return InkCropResult(image.copy(), (0, 0, w, h), 0, 0.0, 0.0, False, (w, h))

    x_min, x_max = int(cols.min()), int(cols.max())
    y_min, y_max = int(rows.min()), int(rows.max())
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    if ink_pixels < 16 or bbox_width < 4 or bbox_height < 4:
        logger.warning(
            "auto_crop: rejected insignificant ink in %dx%d image; "
            "large whitespace may hurt OCR accuracy (pixels=%d bbox=%dx%d)",
            w, h, ink_pixels, bbox_width, bbox_height,
        )
        return InkCropResult(
            image.copy(), (0, 0, w, h), ink_pixels, ink_ratio, ink_ratio, False, (w, h)
        )

    border = (
        max(0, int(padding))
        if padding is not None
        else max(8, min(32, int(round(max(bbox_width, bbox_height) * 0.12))))
    )
    y1 = max(0, y_min - border)
    y2 = min(h, y_max + border + 1)
    x1 = max(0, x_min - border)
    x2 = min(w, x_max + border + 1)
    cropped = image[y1:y2, x1:x2].copy()
    cropped_coverage = ink_pixels / max(cropped.shape[0] * cropped.shape[1], 1)
    logger.debug(
        "auto_crop: %dx%d -> %dx%d box=(%d,%d,%d,%d) ink=%d coverage=%.2f%%",
        w, h, cropped.shape[1], cropped.shape[0], x1, y1, x2, y2,
        ink_pixels, cropped_coverage * 100,
    )
    return InkCropResult(
        cropped, (x1, y1, x2, y2), ink_pixels, cropped_coverage, ink_ratio, True, (w, h)
    )


def enhance_for_ocr(image_rgb: np.ndarray) -> np.ndarray:
    """Sharpen and normalise a handwriting image to improve TrOCR accuracy.

    TrOCR was trained on clean, high-contrast document images.  Raw canvas
    renders are often mid-grey strokes on white, which the model handles less
    reliably than crisp black-on-white text.  This pipeline brings the image
    closer to the training distribution:

      1. Grayscale -- remove colour noise; ink is always dark.
      2. Adaptive thresholding -- binarises locally so that thin strokes and
         thick strokes are equally preserved regardless of global contrast.
      3. Contrast enhancement -- mild alpha/beta boost to make ink darker.
      4. Morphological opening -- removes isolated noise pixels (1-px salt) that
         can confuse the encoder; kernel is intentionally tiny (3x3) to avoid
         erasing thin stroke segments.
      5. Back to 3-channel RGB -- TrOCR processor expects colour input.

    Args:
        image_rgb: Input RGB image (uint8, HxWx3).

    Returns:
        Preprocessed RGB image (uint8, HxWx3) ready for TrOCR.
    """
    # Step 1 -- Grayscale conversion.
    # Colour channels carry no meaningful information for ink-on-white drawings.
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Step 2 -- Adaptive thresholding (Gaussian weighted, block size 31, C=10).
    # We use THRESH_BINARY_INV so ink -> white (255) and background -> black (0),
    # then invert back.  This is a standard technique for handwriting binarisation
    # because it handles uneven illumination and varying stroke pressure far better
    # than global Otsu thresholding.
    binary_inv = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=31,  # neighbourhood size; must be odd
        C=10,          # constant subtracted from mean; higher = more aggressive
    )
    # Invert: ink is dark (0) on white (255) -- TrOCR's expected polarity.
    binary = cv2.bitwise_not(binary_inv)

    # Step 3 -- Contrast enhancement.
    # alpha > 1 increases contrast; beta adds brightness to keep background white.
    # These mild values avoid over-darkening while making strokes crisper.
    enhanced = cv2.convertScaleAbs(binary, alpha=1.3, beta=10)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # Step 4 -- Morphological opening (erosion then dilation) with a tiny kernel.
    # Opening removes isolated noise pixels that survive thresholding without
    # thinning genuine handwriting strokes.  A 3x3 kernel is the smallest
    # structuring element that eliminates single-pixel salt noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 5 -- Convert back to 3-channel RGB.
    # TrOCRProcessor.from_pretrained() expects a PIL image or 3-channel array.
    rgb_out = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)

    logger.debug(
        "enhance_for_ocr: input %dx%d -> output %dx%d",
        image_rgb.shape[1], image_rgb.shape[0],
        rgb_out.shape[1], rgb_out.shape[0],
    )
    return rgb_out


def segment_words_horizontally(
    image_rgb: np.ndarray,
    min_gap_px: int = 20,
) -> List[np.ndarray]:
    """Split a line image into per-word regions using horizontal projection.

    TrOCR accuracy can improve when each word is recognised independently,
    because the model's attention does not have to span the full line width.
    This function detects large horizontal gaps (columns of all-white pixels)
    and splits the image at those gaps.

    If no qualifying gap is found (e.g. tightly written text), the original
    full-width image is returned as a single-element list so the caller does
    not need to special-case the output.

    Args:
        image_rgb:  RGB image of a single handwritten line (uint8, HxWx3).
        min_gap_px: Minimum consecutive all-white columns to count as a word
                    boundary.  Default 20 px works well at 768-px render width.

    Returns:
        List of RGB sub-images, one per detected word region.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Horizontal projection: for each column, count ink pixels (< 250).
    # A column with zero ink pixels is a candidate gap column.
    col_ink = np.sum(gray < 250, axis=0)  # shape: (w,)
    is_gap_col = col_ink == 0             # True where no ink

    # Find contiguous runs of gap columns and split at qualifying gaps.
    regions: List[np.ndarray] = []
    word_start = 0
    gap_start: int = -1

    for col in range(w):
        if is_gap_col[col]:
            if gap_start == -1:
                gap_start = col
        else:
            if gap_start != -1:
                gap_length = col - gap_start
                if gap_length >= min_gap_px and gap_start > word_start:
                    # Emit the word region to the left of this gap.
                    regions.append(image_rgb[:, word_start:gap_start])
                    word_start = col
                gap_start = -1

    # Emit the final word region (or the entire image if no gaps were found).
    regions.append(image_rgb[:, word_start:w])

    if len(regions) == 1:
        logger.debug(
            "segment_words: no gap >= %d px found in %dx%d image -- using full image",
            min_gap_px, w, h,
        )
    else:
        logger.debug(
            "segment_words: split %dx%d image into %d word regions",
            w, h, len(regions),
        )

    return regions

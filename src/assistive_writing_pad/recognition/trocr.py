"""Pretrained TrOCR handwritten text recognizer.

The default checkpoint is Microsoft's IAM-finetuned handwritten TrOCR base
model. Imports are lazy so the rest of the application remains usable before
model dependencies are installed.

Environment variables
---------------------
AWP_TROCR_MODEL
    HuggingFace checkpoint name (default: microsoft/trocr-base-handwritten).
AWP_TROCR_RENDER_SIZE
    Render canvas size in WxH format, e.g. "768x256" (default: 768x256).
    Increase for higher accuracy; decrease on memory-constrained devices.
AWP_DEBUG_OCR
    Set to "1" to save raw/cropped/processed images to data/debug/ on every
    recognition call.  Off by default.
AWP_WORD_SEGMENT
    Set to "0" to disable stroke-geometry word segmentation before OCR.  Each
    detected word is recognized independently and results are joined with
    spaces.  On by default for better sentence spacing.
AWP_TROCR_NUM_BEAMS
    Beam count for TrOCR generation (default: 3). Set to 1 on constrained
    devices if latency is more important than alternatives.
AWP_TROCR_CANDIDATES
    Number of decoded alternatives to return in metadata/top predictions
    (default: 3, capped by beam count).
AWP_OCR_MODE
    Accepted for backward compatibility; every web mode uses OCR.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from assistive_writing_pad.contracts import RecognitionResult, StrokePoint

logger = logging.getLogger(__name__)

# Recognition mode type. Values are accepted for backward compatibility with
# the UI/API, but TrOCR is now used for every live recognition path.
RecognitionMode = Literal["auto", "character", "word", "ocr"]

_DEFAULT_OCR_MODE: RecognitionMode = os.environ.get(  # type: ignore[assignment]
    "AWP_OCR_MODE", "auto"
).strip() or "auto"

DEFAULT_TROCR_MODEL = os.environ.get("AWP_TROCR_MODEL", "microsoft/trocr-base-handwritten")

# Default render canvas: 768x256 gives TrOCR roughly 2x the horizontal
# resolution compared to the previous 384x128, which significantly improves
# recognition of narrow letters and connected script.
_DEFAULT_RENDER_W = 768
_DEFAULT_RENDER_H = 256

# Whether stroke-geometry word segmentation is enabled.
_WORD_SEGMENT_ENABLED: bool = os.environ.get("AWP_WORD_SEGMENT", "1").strip() != "0"
_DEFAULT_NUM_BEAMS = 3
_DEFAULT_NUM_CANDIDATES = 3
_DEFAULT_MAX_WORD_SEGMENTS = 6


@dataclass(frozen=True)
class _OCRCandidate:
    text: str
    confidence: float
    raw_text: str = ""


@dataclass(frozen=True)
class _StrokeBounds:
    stroke: Sequence[StrokePoint]
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    center_x: float
    center_y: float
    width: float
    height: float
    point_count: int


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _normalize_requested_mode(mode: str) -> str:
    if mode in {"auto", "character", "word", "ocr"}:
        return mode
    if _DEFAULT_OCR_MODE in {"character", "word", "ocr"}:
        return _DEFAULT_OCR_MODE
    return "ocr"


def _parse_render_size() -> tuple:
    """Return (width, height) for the TrOCR render canvas.

    Reads AWP_TROCR_RENDER_SIZE env var (format: "WxH", e.g. "768x256").
    Falls back to (_DEFAULT_RENDER_W, _DEFAULT_RENDER_H) on any parse error.
    """
    raw = os.environ.get("AWP_TROCR_RENDER_SIZE", "").strip()
    if raw:
        try:
            w_str, h_str = raw.lower().split("x")
            w, h = int(w_str), int(h_str)
            if w > 0 and h > 0:
                logger.debug("render size from AWP_TROCR_RENDER_SIZE: %dx%d", w, h)
                return (w, h)
        except (ValueError, AttributeError):
            logger.warning(
                "AWP_TROCR_RENDER_SIZE='%s' is not valid WxH format -- using default %dx%d",
                raw, _DEFAULT_RENDER_W, _DEFAULT_RENDER_H,
            )
    return (_DEFAULT_RENDER_W, _DEFAULT_RENDER_H)


_RENDER_SIZE: tuple = _parse_render_size()


class RecognitionUnavailable(RuntimeError):
    """Raised when pretrained recognition cannot run in the current environment."""


@dataclass
class TrOCRHandwritingRecognizer:
    model_name: str = DEFAULT_TROCR_MODEL
    max_new_tokens: int = 48
    num_beams: int = _int_env("AWP_TROCR_NUM_BEAMS", _DEFAULT_NUM_BEAMS)
    num_return_sequences: int = _int_env("AWP_TROCR_CANDIDATES", _DEFAULT_NUM_CANDIDATES)
    max_word_segments: int = _int_env("AWP_TROCR_MAX_WORD_SEGMENTS", _DEFAULT_MAX_WORD_SEGMENTS)

    def __post_init__(self) -> None:
        self.num_beams = max(1, int(self.num_beams))
        self.num_return_sequences = max(1, min(int(self.num_return_sequences), self.num_beams))
        self.max_word_segments = max(1, int(self.max_word_segments))
        self._processor = None
        self._model = None
        self._torch = None

    def recognize(self, strokes: Sequence[StrokePoint], mode: RecognitionMode = "auto") -> RecognitionResult:
        if not strokes:
            return RecognitionResult(
                text="",
                confidence=0.0,
                metadata={"recognizer": "trocr", "reason": "empty_strokes"},
            )

        return self.recognize_stroke_groups([strokes], mode=mode)

    def recognize_stroke_groups(
        self,
        stroke_groups: Sequence[Sequence[StrokePoint]],
        mode: RecognitionMode = "auto",
    ) -> RecognitionResult:
        non_empty_groups = [stroke for stroke in stroke_groups if stroke]
        if not non_empty_groups:
            return RecognitionResult(
                text="",
                confidence=0.0,
                metadata={"recognizer": "trocr", "reason": "empty_strokes"},
            )

        requested_mode = _normalize_requested_mode(mode)
        logger.info(
            "recognize_stroke_groups: requested_mode=%s effective=ocr",
            requested_mode,
        )

        self._ensure_loaded()

        from assistive_writing_pad.recognition.debug_saver import save_debug_images

        if requested_mode in {"auto", "character"} and _looks_like_single_character_input(
            non_empty_groups
        ):
            return self._recognize_single_character_groups(
                non_empty_groups,
                requested_mode=requested_mode,
                save_debug_images=save_debug_images,
            )

        lines = segment_strokes_into_lines(non_empty_groups)

        logger.info(
            "recognize_stroke_groups: %d stroke group(s) -> %d line(s)",
            len(non_empty_groups), len(lines),
        )

        line_results = []
        line_candidate_groups: List[List[_OCRCandidate]] = []

        for line_index, line_groups in enumerate(lines):
            word_groups = [line_groups]
            segmentation = "line"
            if _WORD_SEGMENT_ENABLED and requested_mode in {"auto", "word", "ocr"}:
                segmented_groups = segment_strokes_into_words(line_groups)
                if 1 < len(segmented_groups) <= self.max_word_segments:
                    word_groups = segmented_groups
                    segmentation = "stroke_word"

            logger.info(
                "line %d: %d stroke group(s), segmentation=%s, segments=%d",
                line_index,
                len(line_groups),
                segmentation,
                len(word_groups),
            )

            word_candidate_groups: List[List[_OCRCandidate]] = []
            word_debug = []
            for word_index, current_groups in enumerate(word_groups):
                label = f"line{line_index}"
                if segmentation == "stroke_word":
                    label = f"{label}_word{word_index}"
                candidates = self._recognize_group_image(
                    current_groups,
                    debug_label=label,
                    save_debug_images=save_debug_images,
                )
                word_candidate_groups.append(candidates)
                primary = candidates[0] if candidates else _OCRCandidate("", 0.0, "")
                word_debug.append(
                    {
                        "word_index": word_index,
                        "text": primary.text,
                        "raw_text": primary.raw_text,
                        "confidence": primary.confidence,
                        "stroke_groups": len(current_groups),
                        "candidates": _top_payload(candidates),
                    }
                )

            line_candidates = _merge_segment_candidate_groups(word_candidate_groups)
            line_candidate_groups.append(line_candidates)
            primary_line = line_candidates[0] if line_candidates else _OCRCandidate("", 0.0, "")
            logger.info(
                "line %d: result=%r confidence=%.3f",
                line_index,
                primary_line.text.strip(),
                primary_line.confidence,
            )

            line_results.append(
                {
                    "line_index": line_index,
                    "text": primary_line.text.strip(),
                    "raw_text": primary_line.raw_text.strip(),
                    "confidence": primary_line.confidence,
                    "stroke_groups": len(line_groups),
                    "word_count": len(word_groups),
                    "segmentation": segmentation,
                    "words": word_debug,
                }
            )

        final_candidates = _merge_line_candidate_groups(line_candidate_groups)
        primary = final_candidates[0] if final_candidates else _OCRCandidate("", 0.0, "")
        effective_mode = (
            "word"
            if any(item.get("segmentation") == "stroke_word" for item in line_results)
            else "ocr"
        )
        return RecognitionResult(
            text=primary.text,
            confidence=primary.confidence,
            metadata={
                "recognizer": "trocr",
                "model": self.model_name,
                "lines": str(len(lines)),
                "line_results": json.dumps(line_results),
                "mode": effective_mode,
                "requested_mode": requested_mode,
                "top3": json.dumps(_top_payload(final_candidates)),
                "word_segmentation": "enabled" if _WORD_SEGMENT_ENABLED else "disabled",
            },
        )

    def _recognize_single_character_groups(
        self,
        stroke_groups: Sequence[Sequence[StrokePoint]],
        requested_mode: str,
        save_debug_images,
    ) -> RecognitionResult:
        candidates = self._recognize_group_image(
            stroke_groups,
            debug_label="single_character",
            save_debug_images=save_debug_images,
        )
        text, confidence, top_candidates, reason = _single_character_candidates(
            stroke_groups,
            candidates,
        )
        raw_text = candidates[0].raw_text if candidates else ""

        logger.info(
            "single-character result=%r confidence=%.3f reason=%s raw=%r",
            text,
            confidence,
            reason,
            raw_text,
        )

        return RecognitionResult(
            text=text,
            confidence=confidence,
            metadata={
                "recognizer": "trocr",
                "model": self.model_name,
                "raw_text": raw_text.strip(),
                "mode": "character",
                "requested_mode": requested_mode,
                "single_character": "true",
                "single_character_reason": reason,
                "top3": json.dumps(_top_payload(top_candidates)),
            },
        )

    def _recognize_group_image(
        self,
        stroke_groups: Sequence[Sequence[StrokePoint]],
        debug_label: str,
        save_debug_images,
    ) -> List[_OCRCandidate]:
        raw_image = render_stroke_groups_for_trocr(stroke_groups)
        logger.debug(
            "%s: raw render %dx%d (%d stroke groups)",
            debug_label,
            raw_image.shape[1],
            raw_image.shape[0],
            len(stroke_groups),
        )

        proc_image, cropped_image, processed_image = _preprocess_image(raw_image)
        save_debug_images(raw_image, cropped_image, processed_image, label=debug_label)
        return self._run_ocr(proc_image)

    def _run_ocr(self, image: np.ndarray) -> List[_OCRCandidate]:
        inputs = self._processor(images=image, return_tensors="pt")
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if self.num_beams > 1:
            generation_kwargs["num_beams"] = self.num_beams
        if self.num_return_sequences > 1:
            generation_kwargs["num_return_sequences"] = self.num_return_sequences

        with self._torch.no_grad():
            generated = self._model.generate(inputs.pixel_values, **generation_kwargs)

        raw_texts = self._processor.batch_decode(generated.sequences, skip_special_tokens=True)
        confidences = _candidate_confidences(generated, self._torch, len(raw_texts))
        seen: Dict[str, _OCRCandidate] = {}
        for raw_text, confidence in zip(raw_texts, confidences):
            text = _clean_ocr_text(raw_text).strip()
            key = text.lower()
            existing = seen.get(key)
            candidate = _OCRCandidate(text=text, confidence=confidence, raw_text=raw_text.strip())
            if existing is None or candidate.confidence > existing.confidence:
                seen[key] = candidate

        candidates = sorted(seen.values(), key=lambda item: item.confidence, reverse=True)
        if candidates:
            return candidates[: self.num_return_sequences]

        fallback_confidence = confidences[0] if confidences else 0.0
        fallback_raw = raw_texts[0].strip() if raw_texts else ""
        return [_OCRCandidate(text="", confidence=fallback_confidence, raw_text=fallback_raw)]

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RecognitionUnavailable(
                "Pretrained OCR dependencies are missing. Install them with "
                "`scripts/setup_model_env.sh`, or install CPU PyTorch first and then "
                "`pip install -e '.[models]'`."
            ) from exc

        self._torch = torch
        self._processor = TrOCRProcessor.from_pretrained(self.model_name, use_fast=False)
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=False,
        )
        self._model.to(torch.device("cpu"))
        self._model.eval()


def render_strokes_for_trocr(
    strokes: Sequence[StrokePoint],
    size: Optional[tuple] = None,
    padding: int = 14,
) -> np.ndarray:
    """Render captured strokes as a white-background RGB image for TrOCR.

    The default render size has been increased from 384x128 to 768x256 to give
    TrOCR twice the horizontal resolution.  Higher resolution preserves more
    stroke detail and significantly reduces character confusion errors.
    Override with AWP_TROCR_RENDER_SIZE env var (e.g. "512x192" for Pi).
    """
    if size is None:
        size = _RENDER_SIZE
    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    if not strokes:
        return image

    points = _scale_points(strokes, width, height, padding)
    if len(points) == 1:
        x, y = points[0]
        # Radius 3 matches the larger canvas; thick enough to be visible but
        # not so thick that letters bleed into each other.
        _draw_dot(image, x, y, radius=3)
        return image

    for start, end in zip(points, points[1:]):
        _draw_line(image, start, end, radius=3)
    return image


def render_stroke_groups_for_trocr(
    stroke_groups: Sequence[Sequence[StrokePoint]],
    size: Optional[tuple] = None,
    padding: int = 14,
) -> np.ndarray:
    """Render multiple pen strokes into one OCR image without joining stroke gaps.

    See render_strokes_for_trocr() for notes on the default size increase.
    """
    if size is None:
        size = _RENDER_SIZE
    flattened = [point for stroke in stroke_groups for point in stroke]
    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    if not flattened:
        return image

    bounds = _bounds(flattened)
    all_points = _scale_points(flattened, width, height, padding, bounds=bounds)
    point_lookup = iter(all_points)

    for stroke in stroke_groups:
        if not stroke:
            continue
        scaled = [next(point_lookup) for _ in stroke]
        if len(scaled) == 1:
            x, y = scaled[0]
            _draw_dot(image, x, y, radius=3)
            continue
        for start, end in zip(scaled, scaled[1:]):
            _draw_line(image, start, end, radius=3)

    return image


def _preprocess_image(
    raw_image: np.ndarray,
) -> tuple:
    """Run auto-crop and OCR enhancement on a rendered stroke image.

    Returns a tuple of (final_image, cropped_image, processed_image) where
    final_image is what gets fed into TrOCR and the other two are kept for
    the debug saver.  All three are uint8 RGB arrays.
    """
    from assistive_writing_pad.preprocessing.ocr_image_ops import (
        auto_crop_handwriting,
        enhance_for_ocr,
    )

    # Stage 1 -- Auto-crop: remove the large whitespace margins so that the
    # model's attention is focused on the actual handwriting region.
    cropped = auto_crop_handwriting(raw_image, padding=20)
    logger.debug(
        "_preprocess_image: after crop %dx%d -> %dx%d",
        raw_image.shape[1], raw_image.shape[0],
        cropped.shape[1], cropped.shape[0],
    )

    # Stage 2 -- Enhance: adaptive threshold + contrast boost + morphological
    # opening to produce a crisp binary image closer to TrOCR's training data.
    processed = enhance_for_ocr(cropped)
    logger.debug(
        "_preprocess_image: after enhance %dx%d",
        processed.shape[1], processed.shape[0],
    )

    return processed, cropped, processed


def segment_strokes_into_lines(
    stroke_groups: Sequence[Sequence[StrokePoint]],
    gap_threshold: Optional[float] = None,
) -> List[List[Sequence[StrokePoint]]]:
    """Cluster stroke groups by vertical position into reading lines."""

    stroke_bounds = [_stroke_bounds(stroke) for stroke in stroke_groups if stroke]
    if not stroke_bounds:
        return []

    threshold = gap_threshold if gap_threshold is not None else _adaptive_line_gap(stroke_bounds)
    stroke_bounds.sort(key=lambda item: (item.center_y, item.center_x))

    line_groups: List[List[_StrokeBounds]] = [[stroke_bounds[0]]]

    for bounds in stroke_bounds[1:]:
        current_line = line_groups[-1]
        current_center = _line_center_y(current_line)
        if abs(bounds.center_y - current_center) > threshold and not _vertically_overlaps_line(
            bounds,
            current_line,
        ):
            line_groups.append([bounds])
            continue

        current_line.append(bounds)

    lines: List[List[Sequence[StrokePoint]]] = []
    for line in line_groups:
        line.sort(key=lambda item: (item.min_x, item.center_y))
        lines.append([item.stroke for item in line])
    return lines


def segment_strokes_into_words(
    stroke_groups: Sequence[Sequence[StrokePoint]],
    gap_threshold: Optional[float] = None,
) -> List[List[Sequence[StrokePoint]]]:
    """Split one line of stroke groups into words using raw stroke geometry.

    The split is intentionally conservative: a gap must be large relative to
    both the line height and the typical stroke width before it is treated as a
    space. This avoids splitting multi-stroke letters such as dotted i/j or
    crossed t/f.
    """

    stroke_bounds = [_stroke_bounds(stroke) for stroke in stroke_groups if stroke]
    if not stroke_bounds:
        return []
    if len(stroke_bounds) == 1:
        return [[stroke_bounds[0].stroke]]

    stroke_bounds.sort(key=lambda item: (item.min_x, item.center_y))
    threshold = gap_threshold if gap_threshold is not None else _adaptive_word_gap(stroke_bounds)

    words: List[List[_StrokeBounds]] = [[stroke_bounds[0]]]
    current_right = stroke_bounds[0].max_x

    for bounds in stroke_bounds[1:]:
        gap = bounds.min_x - current_right
        if gap > threshold and not _is_likely_attached_mark(bounds, words[-1]):
            words.append([bounds])
        else:
            words[-1].append(bounds)
        current_right = max(current_right, bounds.max_x)

    result: List[List[Sequence[StrokePoint]]] = []
    for word in words:
        word.sort(key=lambda item: (item.min_x, item.center_y))
        result.append([item.stroke for item in word])
    return result


def _stroke_bounds(stroke: Sequence[StrokePoint]) -> _StrokeBounds:
    min_x, max_x, min_y, max_y = _bounds(stroke)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    return _StrokeBounds(
        stroke=stroke,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        center_x=(min_x + max_x) / 2.0,
        center_y=(min_y + max_y) / 2.0,
        width=width,
        height=height,
        point_count=len(stroke),
    )


def _adaptive_line_gap(stroke_bounds: Sequence[_StrokeBounds]) -> float:
    heights = [bounds.height for bounds in stroke_bounds]
    median_height = float(np.median(heights)) if heights else 1.0
    tall_height = float(np.percentile(heights, 75)) if len(heights) > 1 else median_height
    return max(28.0, median_height * 1.8, tall_height * 1.25)


def _line_center_y(line: Sequence[_StrokeBounds]) -> float:
    weights = [max(bounds.point_count, 1) for bounds in line]
    centers = [bounds.center_y for bounds in line]
    return float(np.average(centers, weights=weights))


def _vertically_overlaps_line(bounds: _StrokeBounds, line: Sequence[_StrokeBounds]) -> bool:
    line_min_y = min(item.min_y for item in line)
    line_max_y = max(item.max_y for item in line)
    overlap = min(bounds.max_y, line_max_y) - max(bounds.min_y, line_min_y)
    return overlap >= min(bounds.height, max(line_max_y - line_min_y, 1.0)) * 0.20


def _adaptive_word_gap(stroke_bounds: Sequence[_StrokeBounds]) -> float:
    line_height = max(bounds.max_y for bounds in stroke_bounds) - min(
        bounds.min_y for bounds in stroke_bounds
    )
    widths = [bounds.width for bounds in stroke_bounds]
    median_width = float(np.median(widths)) if widths else 1.0
    positive_gaps = []
    ordered = sorted(stroke_bounds, key=lambda item: (item.min_x, item.center_y))
    current_right = ordered[0].max_x
    for bounds in ordered[1:]:
        gap = bounds.min_x - current_right
        if gap > 0:
            positive_gaps.append(gap)
        current_right = max(current_right, bounds.max_x)
    median_gap = float(np.median(positive_gaps)) if len(positive_gaps) >= 3 else 0.0
    return max(18.0, line_height * 0.30, median_width * 0.90, median_gap * 1.75)


def _is_likely_attached_mark(
    bounds: _StrokeBounds,
    current_word: Sequence[_StrokeBounds],
) -> bool:
    if not current_word:
        return False
    word_min_x = min(item.min_x for item in current_word)
    word_max_x = max(item.max_x for item in current_word)
    word_width = max(word_max_x - word_min_x, 1.0)
    word_max_height = max(item.height for item in current_word)
    small_mark = bounds.point_count <= 8 and bounds.height <= max(word_max_height * 0.35, 4.0)
    near_word = bounds.min_x <= word_max_x + max(word_width * 0.25, 8.0)
    x_overlaps = bounds.min_x <= word_max_x and bounds.max_x >= word_min_x
    return small_mark and (near_word or x_overlaps)


def _scale_points(
    strokes: Sequence[StrokePoint],
    width: int,
    height: int,
    padding: int,
    bounds: Optional[tuple] = None,
) -> list:
    min_x, max_x, min_y, max_y = bounds or _bounds(strokes)

    x_range = max(max_x - min_x, 1.0)
    y_range = max(max_y - min_y, 1.0)
    scale = min((width - padding * 2) / x_range, (height - padding * 2) / y_range)
    content_width = x_range * scale
    content_height = y_range * scale
    x_offset = (width - content_width) / 2.0
    y_offset = (height - content_height) / 2.0

    points = []
    for point in strokes:
        x = int(round((point.x - min_x) * scale + x_offset))
        y = int(round((point.y - min_y) * scale + y_offset))
        points.append((max(0, min(width - 1, x)), max(0, min(height - 1, y))))
    return points


def _bounds(strokes: Sequence[StrokePoint]) -> tuple:
    xs = [point.x for point in strokes]
    ys = [point.y for point in strokes]
    return min(xs), max(xs), min(ys), max(ys)


def _draw_line(image: np.ndarray, start: tuple, end: tuple, radius: int) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        alpha = step / steps
        x = int(round(x0 + (x1 - x0) * alpha))
        y = int(round(y0 + (y1 - y0) * alpha))
        _draw_dot(image, x, y, radius)


def _draw_dot(image: np.ndarray, x: int, y: int, radius: int) -> None:
    height, width, _channels = image.shape
    for row in range(max(0, y - radius), min(height, y + radius + 1)):
        for col in range(max(0, x - radius), min(width, x + radius + 1)):
            image[row, col] = 0


def _generation_confidence(generated, torch_module) -> float:
    scores = getattr(generated, "scores", None)
    if not scores:
        return 0.0

    token_confidences = []
    for score in scores:
        probabilities = torch_module.softmax(score, dim=-1)
        token_confidences.append(float(probabilities.max().item()))

    if not token_confidences:
        return 0.0
    return float(np.mean(token_confidences))


def _candidate_confidences(generated, torch_module, count: int) -> List[float]:
    if count <= 0:
        return []

    base_confidence = _generation_confidence(generated, torch_module)
    if base_confidence <= 0.0:
        base_confidence = 0.50

    sequence_scores = getattr(generated, "sequences_scores", None)
    if sequence_scores is not None and len(sequence_scores) >= count:
        scores = sequence_scores[:count]
        best_score = float(scores[0].item())
        confidences = []
        for score in scores:
            relative = float(torch_module.exp(score - best_score).item())
            confidences.append(_clamp_confidence(base_confidence * relative))
        return confidences

    return [_clamp_confidence(base_confidence * (0.92 ** index)) for index in range(count)]


def _top_payload(candidates: Sequence[_OCRCandidate], limit: int = 5) -> List[Tuple[str, float]]:
    return [
        (candidate.text, round(float(candidate.confidence), 4))
        for candidate in candidates[:limit]
        if candidate.text
    ]


def _merge_segment_candidate_groups(
    candidate_groups: Sequence[Sequence[_OCRCandidate]],
    limit: int = 5,
) -> List[_OCRCandidate]:
    non_empty_groups = [list(group) for group in candidate_groups if group]
    if not non_empty_groups:
        return []
    if len(non_empty_groups) == 1:
        return list(non_empty_groups[0])[:limit]

    primary_parts = [group[0].text.strip() for group in non_empty_groups]
    primary_confidences = [group[0].confidence for group in non_empty_groups]
    return _merge_alternative_parts(primary_parts, primary_confidences, non_empty_groups, " ", limit)


def _merge_line_candidate_groups(
    candidate_groups: Sequence[Sequence[_OCRCandidate]],
    limit: int = 5,
) -> List[_OCRCandidate]:
    non_empty_groups = [list(group) for group in candidate_groups if group]
    if not non_empty_groups:
        return []
    if len(non_empty_groups) == 1:
        return list(non_empty_groups[0])[:limit]

    primary_parts = [group[0].text.strip() for group in non_empty_groups]
    primary_confidences = [group[0].confidence for group in non_empty_groups]
    return _merge_alternative_parts(primary_parts, primary_confidences, non_empty_groups, "\n", limit)


def _merge_alternative_parts(
    primary_parts: Sequence[str],
    primary_confidences: Sequence[float],
    candidate_groups: Sequence[Sequence[_OCRCandidate]],
    separator: str,
    limit: int,
) -> List[_OCRCandidate]:
    candidates: Dict[str, _OCRCandidate] = {}
    primary_text = separator.join(part for part in primary_parts if part).strip()
    primary_confidence = _mean_confidence(primary_confidences)
    if primary_text:
        candidates[primary_text] = _OCRCandidate(primary_text, primary_confidence, primary_text)

    for index, group in enumerate(candidate_groups):
        if index >= len(primary_parts):
            continue
        for alternative in list(group)[1:]:
            alt_text = alternative.text.strip()
            if not alt_text or alt_text == primary_parts[index]:
                continue
            variant_parts = list(primary_parts)
            variant_confidences = list(primary_confidences)
            variant_parts[index] = alt_text
            variant_confidences[index] = alternative.confidence
            text = separator.join(part for part in variant_parts if part).strip()
            confidence = _mean_confidence(variant_confidences)
            existing = candidates.get(text)
            if existing is None or confidence > existing.confidence:
                candidates[text] = _OCRCandidate(text, confidence, text)

    return sorted(candidates.values(), key=lambda item: item.confidence, reverse=True)[:limit]


def _single_character_candidates(
    stroke_groups: Sequence[Sequence[StrokePoint]],
    ocr_candidates: Sequence[_OCRCandidate],
) -> Tuple[str, float, List[_OCRCandidate], str]:
    from assistive_writing_pad.recognition.confusion import apply_confusion_correction

    scored: Dict[str, _OCRCandidate] = {}
    primary = ocr_candidates[0] if ocr_candidates else _OCRCandidate("", 0.0, "")
    primary_guess = ""

    for candidate in ocr_candidates:
        guess = _single_character_guess(candidate.text) or _single_character_guess(candidate.raw_text)
        if not guess:
            continue
        corrected, confusion_candidates = apply_confusion_correction(
            guess,
            candidate.confidence,
            mode="character",
        )
        if corrected:
            _put_best_candidate(scored, corrected, candidate.confidence, candidate.raw_text)
            if not primary_guess:
                primary_guess = corrected
        for char, confidence in confusion_candidates:
            if char:
                _put_best_candidate(scored, char, confidence, candidate.raw_text)

    shape_hint = _shape_hint_for_single_character(stroke_groups, primary_guess)
    reason = "ocr_single_guess"
    if shape_hint is not None:
        reason = "shape_hint"
        promoted_confidence = min(max(primary.confidence + 0.04, 0.82), 0.98)
        _put_best_candidate(scored, shape_hint, promoted_confidence, primary.raw_text)

    if not scored:
        fallback_text = _single_character_guess(primary.raw_text) or _single_character_guess(primary.text)
        if not fallback_text:
            return "", 0.0, [], "empty_single_character"
        _put_best_candidate(scored, fallback_text, primary.confidence, primary.raw_text)

    candidates = sorted(scored.values(), key=lambda item: item.confidence, reverse=True)[:5]
    top = candidates[0]
    return top.text, top.confidence, candidates, reason


def _put_best_candidate(
    candidates: Dict[str, _OCRCandidate],
    text: str,
    confidence: float,
    raw_text: str,
) -> None:
    clean_text = text.strip()
    if not clean_text:
        return
    key = clean_text
    candidate = _OCRCandidate(clean_text, _clamp_confidence(confidence), raw_text.strip())
    existing = candidates.get(key)
    if existing is None or candidate.confidence > existing.confidence:
        candidates[key] = candidate


def _mean_confidence(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return _clamp_confidence(float(np.mean(values)))


def _clamp_confidence(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _looks_like_single_character_input(stroke_groups: Sequence[Sequence[StrokePoint]]) -> bool:
    if not stroke_groups:
        return False
    total_points = sum(len(stroke) for stroke in stroke_groups)
    if len(stroke_groups) > 3 or total_points > 220:
        return False

    points = [point for stroke in stroke_groups for point in stroke]
    if not points:
        return False

    min_x, max_x, min_y, max_y = _bounds(points)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    if width > max(height * 1.8, 48.0):
        return False

    if len(stroke_groups) > 1:
        centers = [_stroke_bounds(stroke).center_x for stroke in stroke_groups if stroke]
        if centers and (max(centers) - min(centers)) > max(height * 1.25, 42.0):
            return False

    return True


def _clean_ocr_text(text: str) -> str:
    """Normalize frequent TrOCR artifacts for handwritten alphabet input."""

    if not text:
        return ""

    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = text.replace("#", " ")
    tokens = text.split()
    input_has_alpha = any(any(char.isalpha() for char in token) for token in tokens)

    cleaned_tokens = []
    for token in tokens:
        token = re.sub(r"[^0-9A-Za-z'\-]+", "", token)
        if not token:
            continue

        has_alpha = any(char.isalpha() for char in token)
        has_digit = any(char.isdigit() for char in token)

        if has_alpha and has_digit:
            token = "".join(char for char in token if char.isalpha() or char in "'-")
            token = token.strip("-'")
            if not token:
                continue
        elif input_has_alpha and has_digit and len(token) <= 4:
            # Standalone short number tokens are usually OCR noise for letters.
            continue

        cleaned_tokens.append(token)

    return " ".join(cleaned_tokens).strip()


def _single_character_guess(text: str) -> str:
    """Collapse noisy line OCR output into a likely single alphabet character."""

    text = text.strip()
    if len(text) <= 1:
        return text

    letters = [char.lower() for char in text if char.isalpha()]
    if not letters:
        digits = [char for char in text if char.isdigit()]
        if not digits:
            return ""
        digit_to_letter = {
            "0": "o",
            "1": "l",
            "2": "z",
            "5": "s",
            "6": "g",
            "8": "b",
        }
        mapped = [digit_to_letter.get(char, "") for char in digits]
        mapped = [char for char in mapped if char]
        return mapped[-1] if mapped else ""

    if len(letters) == 1:
        return letters[0]

    counts = Counter(letters)
    best_count = max(counts.values())
    tied = {char for char, count in counts.items() if count == best_count}

    consonant_tied = {char for char in tied if char not in {"a", "e", "i", "o", "u"}}
    if consonant_tied:
        for char in reversed(letters):
            if char in consonant_tied:
                return char

    for char in reversed(letters):
        if char in tied:
            return char
    return letters[-1]


def _shape_hint_for_single_character(
    stroke_groups: Sequence[Sequence[StrokePoint]],
    current_guess: str,
) -> Optional[str]:
    """Override specific single-letter confusions using stroke geometry."""

    if not stroke_groups:
        return None

    guess = current_guess.strip().lower()
    if guess not in {"", "a", "m", "t", "l", "i", "j", "o", "q", "g", "r"}:
        return None

    dot_hint = _dot_above_stem_hint(stroke_groups)
    if dot_hint is not None:
        return dot_hint

    if guess in {"o", "q", "g"}:
        stem_hint = _open_stem_hint(stroke_groups)
        if stem_hint is not None:
            return stem_hint

    points = [point for stroke in stroke_groups for point in stroke]
    if len(points) < 8:
        return None

    min_x, max_x, min_y, max_y = _bounds(points)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    aspect = height / width
    if aspect < 1.3:
        return None

    first_stroke = next((stroke for stroke in stroke_groups if stroke), [])
    if len(first_stroke) < 6:
        return None

    prefix_len = max(3, int(len(first_stroke) * 0.35))
    prefix = first_stroke[:prefix_len]
    prefix_x_span = max(point.x for point in prefix) - min(point.x for point in prefix)
    prefix_y_span = max(point.y for point in prefix) - min(point.y for point in prefix)

    stem_like = prefix_x_span <= width * 0.22 and prefix_y_span >= height * 0.45
    end_point = first_stroke[-1]
    right_leg = (end_point.x - min_x) >= width * 0.5 and (end_point.y - min_y) >= height * 0.55

    if stem_like and right_leg:
        return "h"

    return None


def _dot_above_stem_hint(stroke_groups: Sequence[Sequence[StrokePoint]]) -> Optional[str]:
    """Detect dotted lowercase glyphs such as i/j from detached tiny strokes."""

    non_empty = [stroke for stroke in stroke_groups if stroke]
    if len(non_empty) < 2:
        return None

    flattened = [point for stroke in non_empty for point in stroke]
    min_x, max_x, min_y, max_y = _bounds(flattened)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    group_boxes = []
    for stroke in non_empty:
        sx0, sx1, sy0, sy1 = _bounds(stroke)
        group_boxes.append(
            {
                "stroke": stroke,
                "min_x": sx0,
                "max_x": sx1,
                "min_y": sy0,
                "max_y": sy1,
                "width": max(sx1 - sx0, 1.0),
                "height": max(sy1 - sy0, 1.0),
                "size": len(stroke),
                "center_x": (sx0 + sx1) / 2.0,
            }
        )

    dot_candidates = [
        box
        for box in group_boxes
        if box["size"] <= 8
        and box["width"] <= max(width * 0.35, 3.0)
        and box["height"] <= max(height * 0.22, 3.0)
    ]
    if not dot_candidates:
        return None

    main = max(group_boxes, key=lambda box: box["height"] * box["size"])
    for dot in dot_candidates:
        above_main = dot["max_y"] < (main["min_y"] - height * 0.08)
        x_aligned = abs(dot["center_x"] - main["center_x"]) <= max(width * 0.35, 3.0)
        if above_main and x_aligned:
            end_point = main["stroke"][-1]
            deep_descender = (main["max_y"] - min_y) >= height * 0.82
            left_hook = end_point.x <= (main["center_x"] - max(width * 0.08, 1.5))
            if deep_descender and left_hook:
                return "j"
            return "i"

    return None


def _open_stem_hint(stroke_groups: Sequence[Sequence[StrokePoint]]) -> Optional[str]:
    """Detect open stem letters (for example r/l/t) that OCR may confuse with loop letters."""

    stroke = _dominant_stroke(stroke_groups)
    if not stroke or len(stroke) < 6:
        return None

    min_x, max_x, min_y, max_y = _bounds(stroke)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    if height / width < 1.1:
        return None

    if _is_closed_loop(stroke, width, height):
        return None

    start = stroke[0]
    end = stroke[-1]

    top_band = min_y + height * 0.35
    top_points = [point for point in stroke if point.y <= top_band]
    top_span = (max(point.x for point in top_points) - min(point.x for point in top_points)) if top_points else 0.0

    has_top_arm = top_span >= width * 0.45
    starts_near_left = (start.x - min_x) <= width * 0.35
    has_tall_stem = _max_vertical_column_span(stroke, width) >= height * 0.62
    stem_x = _dominant_stem_x(stroke, width)

    descends_to_bottom = (max_y - end.y) <= height * 0.2

    if has_top_arm and has_tall_stem:
        if top_points:
            left_reach = stem_x - min(point.x for point in top_points)
            right_reach = max(point.x for point in top_points) - stem_x
        else:
            left_reach = 0.0
            right_reach = 0.0

        looks_like_crossbar = left_reach >= width * 0.18 and right_reach >= width * 0.22
        if top_span >= width * 0.60 and looks_like_crossbar:
            return "t"
        if starts_near_left or descends_to_bottom:
            return "r"
        return "r"

    if not has_top_arm and descends_to_bottom and starts_near_left and (end.x - min_x) <= width * 0.25:
        return "l"

    return None


def _dominant_stroke(stroke_groups: Sequence[Sequence[StrokePoint]]) -> Sequence[StrokePoint]:
    non_empty = [stroke for stroke in stroke_groups if stroke]
    if not non_empty:
        return []
    return max(non_empty, key=lambda stroke: len(stroke))


def _is_closed_loop(stroke: Sequence[StrokePoint], width: float, height: float) -> bool:
    if len(stroke) < 6:
        return False
    start = stroke[0]
    end = stroke[-1]
    dx = abs(end.x - start.x)
    dy = abs(end.y - start.y)
    return dx <= width * 0.22 and dy <= height * 0.22


def _max_vertical_column_span(stroke: Sequence[StrokePoint], width: float) -> float:
    if not stroke:
        return 0.0
    tolerance = max(width * 0.08, 1.5)
    best = 0.0
    for anchor in stroke:
        column = [point for point in stroke if abs(point.x - anchor.x) <= tolerance]
        if len(column) < 2:
            continue
        span = max(point.y for point in column) - min(point.y for point in column)
        if span > best:
            best = span
    return best


def _dominant_stem_x(stroke: Sequence[StrokePoint], width: float) -> float:
    if not stroke:
        return 0.0
    tolerance = max(width * 0.08, 1.5)
    best_span = -1.0
    best_x = stroke[0].x
    for anchor in stroke:
        column = [point for point in stroke if abs(point.x - anchor.x) <= tolerance]
        if len(column) < 2:
            continue
        span = max(point.y for point in column) - min(point.y for point in column)
        if span > best_span:
            best_span = span
            best_x = float(np.mean([point.x for point in column]))
    return best_x

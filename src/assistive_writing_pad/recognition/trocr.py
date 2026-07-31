"""Pretrained TrOCR handwritten text recognizer.

The default checkpoint is Microsoft's IAM-finetuned handwritten TrOCR base
model. Imports are lazy so the rest of the application remains usable before
model dependencies are installed.

Environment variables
---------------------
AWP_TROCR_MODEL
    HuggingFace checkpoint name (default: microsoft/trocr-small-handwritten).
AWP_TROCR_RENDER_SIZE
    Render canvas size in WxH format, e.g. "768x256" (default: 768x256).
    Increase for higher accuracy; decrease on memory-constrained devices.
AWP_DEBUG_OCR
    Set to "1" to save raw/cropped/processed images to data/debug/ on every
    recognition call.  Off by default.
AWP_WORD_SEGMENT
    Set to "1" to enable horizontal word segmentation before OCR.  Each
    detected word is recognised independently and results are joined with
    spaces.  Off by default (safer for tightly written text).
AWP_OCR_MODE
    Default recognition mode: "auto" (default), "character", or "word".
    Can be overridden per-request via the API payload.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np

from assistive_writing_pad.contracts import RecognitionResult, StrokePoint

logger = logging.getLogger(__name__)

# Recognition mode type.
# "auto"      -- use single-character path when strokes look like one char,
#                word path otherwise.
# "character" -- always use single-character path + confusion correction.
# "word"      -- always use full-line TrOCR path, no confusion correction.
RecognitionMode = Literal["auto", "character", "word"]

_DEFAULT_OCR_MODE: RecognitionMode = os.environ.get(  # type: ignore[assignment]
    "AWP_OCR_MODE", "auto"
).strip() or "auto"

DEFAULT_TROCR_MODEL = os.environ.get("AWP_TROCR_MODEL", "microsoft/trocr-small-handwritten")

# Default render canvas: 768x256 gives TrOCR roughly 2x the horizontal
# resolution compared to the previous 384x128, which significantly improves
# recognition of narrow letters and connected script.
_DEFAULT_RENDER_W = 768
_DEFAULT_RENDER_H = 256

# Whether horizontal word segmentation is enabled (off by default).
_WORD_SEGMENT_ENABLED: bool = os.environ.get("AWP_WORD_SEGMENT", "0").strip() == "1"


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

    def __post_init__(self) -> None:
        self._processor = None
        self._model = None
        self._torch = None
        self._emnist_recognizer = None

    def recognize(self, strokes: Sequence[StrokePoint], mode: RecognitionMode = "auto") -> RecognitionResult:
        if not strokes:
            return RecognitionResult(
                text="",
                confidence=0.0,
                metadata={"recognizer": "trocr", "reason": "empty_strokes"},
            )

        # Resolve effective mode.
        effective_mode: RecognitionMode = mode if mode in ("auto", "character", "word") else _DEFAULT_OCR_MODE
        is_single_char = _looks_like_single_character_input([strokes])
        use_char_mode = effective_mode == "character" or (effective_mode == "auto" and is_single_char)
        logger.info(
            "recognize: mode=%s effective=%s is_single_char=%s",
            mode, "character" if use_char_mode else "word", is_single_char,
        )

        if use_char_mode:
            if self._emnist_recognizer is None:
                from assistive_writing_pad.recognition.emnist import EMNISTCharacterRecognizer
                self._emnist_recognizer = EMNISTCharacterRecognizer()
            res = self._emnist_recognizer.recognize(strokes)
            # top5 carries up to 5 candidates (CNN + confusion-map siblings).
            top5 = [(c.character, c.confidence) for c in res.character_confidences]
            # Log confusion pairs: when rank-1 differs between raw CNN and merged output
            # this is already logged inside emnist.py; surface it here too for the API.
            confusion_pairs = [
                f"{top5[0][0]}\u2194{c}" for c, _ in top5[1:4]
            ] if len(top5) > 1 else []
            logger.info(
                "recognize [character]: result=%r conf=%.3f top5=%r confusion=%r",
                res.text, res.confidence, top5[:3], confusion_pairs,
            )
            return RecognitionResult(
                text=res.text,
                confidence=res.confidence,
                character_confidences=res.character_confidences,
                metadata={
                    "recognizer": "emnist",
                    "model": res.metadata.get("model", ""),
                    "mode": "character",
                    "top3": json.dumps(top5),
                    "confusion_pairs": json.dumps(confusion_pairs),
                }
            )

        self._ensure_loaded()

        # --- Render ---
        raw_image = render_strokes_for_trocr(strokes)
        logger.info(
            "recognize: raw render %dx%d (strokes=%d points)",
            raw_image.shape[1], raw_image.shape[0], len(strokes),
        )

        # --- Preprocess ---
        image, cropped_image, processed_image = _preprocess_image(raw_image)

        # --- Debug save ---
        from assistive_writing_pad.recognition.debug_saver import save_debug_images
        save_debug_images(raw_image, cropped_image, processed_image)

        # --- OCR ---
        # In character mode use a smaller token budget: a single character
        # takes at most 2 tokens (char + EOS), so capping at 4 prevents the
        # model from emitting long noisy sequences.
        max_tokens = 4 if use_char_mode else self.max_new_tokens
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values

        with self._torch.no_grad():
            generated = self._model.generate(
                pixel_values,
                max_new_tokens=max_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

        raw_text = self._processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
        confidence = _generation_confidence(generated, self._torch)

        text = _clean_ocr_text(raw_text)
        if is_single_char:
            text = _single_character_guess(text)
            hinted = _shape_hint_for_single_character([strokes], text)
            if hinted is not None:
                text = hinted
        top3: List[Tuple[str, float]] = [(text, confidence)]
        logger.info(
            "recognize [word]: result=%r confidence=%.3f raw=%r",
            text.strip(), confidence, raw_text.strip(),
        )

        return RecognitionResult(
            text=text.strip(),
            confidence=confidence,
            metadata={
                "recognizer": "trocr",
                "model": self.model_name,
                "raw_text": raw_text.strip(),
                "mode": "character" if use_char_mode else "word",
                "top3": json.dumps(top3),
            },
        )

    def recognize_stroke_groups(
        self,
        stroke_groups: Sequence[Sequence[StrokePoint]],
        mode: RecognitionMode = "auto",
    ) -> RecognitionResult:
        if not stroke_groups:
            return RecognitionResult(
                text="",
                confidence=0.0,
                metadata={"recognizer": "trocr", "reason": "empty_strokes"},
            )

        # Resolve effective mode.
        effective_mode: RecognitionMode = mode if mode in ("auto", "character", "word") else _DEFAULT_OCR_MODE
        is_single_char_input = _looks_like_single_character_input(stroke_groups)
        use_char_mode = effective_mode == "character" or (effective_mode == "auto" and is_single_char_input)
        logger.info(
            "recognize_stroke_groups: mode=%s effective=%s is_single_char=%s",
            mode, "character" if use_char_mode else "word", is_single_char_input,
        )

        if use_char_mode:
            if self._emnist_recognizer is None:
                from assistive_writing_pad.recognition.emnist import EMNISTCharacterRecognizer
                self._emnist_recognizer = EMNISTCharacterRecognizer()
            flattened = [point for stroke in stroke_groups for point in stroke]
            res = self._emnist_recognizer.recognize(flattened)
            top5 = [(c.character, c.confidence) for c in res.character_confidences]
            return RecognitionResult(
                text=res.text,
                confidence=res.confidence,
                character_confidences=res.character_confidences,
                metadata={
                    "recognizer": "emnist",
                    "model": res.metadata.get("model", ""),
                    "mode": "character",
                    "top3": json.dumps(top5),
                    "lines": "1",
                    "line_results": json.dumps([{
                        "line_index": 0,
                        "text": res.text,
                        "raw_text": res.text,
                        "confidence": res.confidence,
                        "stroke_groups": len(stroke_groups)
                    }])
                }
            )

        self._ensure_loaded()

        # Import here to keep recognition module decoupled from preprocessing
        # at the class level while still using the new pipeline at runtime.
        from assistive_writing_pad.preprocessing.ocr_image_ops import (
            segment_words_horizontally,
        )
        from assistive_writing_pad.recognition.confusion import (
            apply_confusion_correction,
            restrict_to_charset,
        )
        from assistive_writing_pad.recognition.debug_saver import save_debug_images

        if use_char_mode or is_single_char_input:
            lines = [list(stroke_groups)]
        else:
            lines = segment_strokes_into_lines(stroke_groups)

        logger.info(
            "recognize_stroke_groups: %d stroke group(s) -> %d line(s)",
            len(stroke_groups), len(lines),
        )

        line_results = []
        line_texts = []
        confidences = []
        # top3 for the whole result (only meaningful in character mode)
        result_top3: List[Tuple[str, float]] = []

        for line_index, line_groups in enumerate(lines):
            # --- Render ---
            raw_image = render_stroke_groups_for_trocr(line_groups)
            logger.info(
                "line %d: raw render %dx%d (%d stroke groups)",
                line_index, raw_image.shape[1], raw_image.shape[0], len(line_groups),
            )

            # --- Preprocess ---
            proc_image, cropped_image, processed_image = _preprocess_image(raw_image)

            # --- Debug save ---
            save_debug_images(
                raw_image, cropped_image, processed_image,
                label=f"line{line_index}",
            )

            # In character mode use a small token budget.
            max_tokens = 4 if use_char_mode else self.max_new_tokens

            # --- Optional word segmentation (word mode only) ---
            if _WORD_SEGMENT_ENABLED and not use_char_mode:
                word_regions = segment_words_horizontally(proc_image)
                logger.info(
                    "line %d: word segmentation -> %d region(s)",
                    line_index, len(word_regions),
                )
            else:
                word_regions = [proc_image]

            # --- OCR per region ---
            word_texts: List[str] = []
            word_confidences: List[float] = []
            for region in word_regions:
                inputs = self._processor(images=region, return_tensors="pt")
                with self._torch.no_grad():
                    generated = self._model.generate(
                        inputs.pixel_values,
                        max_new_tokens=max_tokens,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                raw_text = self._processor.batch_decode(
                    generated.sequences, skip_special_tokens=True
                )[0]
                word_conf = _generation_confidence(generated, self._torch)

                if use_char_mode:
                    # Character mode: restrict charset and apply confusion correction.
                    clean_raw = restrict_to_charset(raw_text, charset="alphanum")
                    if not clean_raw:
                        clean_raw = _single_character_guess(_clean_ocr_text(raw_text))
                        hinted = _shape_hint_for_single_character(line_groups, clean_raw)
                        if hinted is not None:
                            clean_raw = hinted
                    word_text, top3 = apply_confusion_correction(clean_raw, word_conf, mode="character")
                    result_top3 = top3  # save for final result
                    logger.info(
                        "line %d [char]: result=%r conf=%.3f top3=%r",
                        line_index, word_text, word_conf, top3,
                    )
                else:
                    word_text = _clean_ocr_text(raw_text)
                    word_text = word_text.strip()

                word_texts.append(word_text)
                word_confidences.append(word_conf)

            # Merge word results back into a line.
            line_raw_text = " ".join(w for w in word_texts if w)
            text = line_raw_text

            # Single-character correction path for auto-mode (non-char-mode).
            if not use_char_mode and _looks_like_single_character_input(line_groups):
                text = _single_character_guess(text)
                hinted = _shape_hint_for_single_character(line_groups, text)
                if hinted is not None:
                    text = hinted

            confidence = float(np.mean(word_confidences)) if word_confidences else 0.0
            logger.info(
                "line %d: result=%r confidence=%.3f",
                line_index, text.strip(), confidence,
            )

            line_texts.append(text.strip())
            confidences.append(confidence)
            line_results.append(
                {
                    "line_index": line_index,
                    "text": text.strip(),
                    "raw_text": line_raw_text.strip(),
                    "confidence": confidence,
                    "stroke_groups": len(line_groups),
                }
            )

        final_confidence = float(np.mean(confidences)) if confidences else 0.0
        return RecognitionResult(
            text="\n".join(text for text in line_texts if text),
            confidence=final_confidence,
            metadata={
                "recognizer": "trocr",
                "model": self.model_name,
                "lines": str(len(lines)),
                "line_results": json.dumps(line_results),
                "mode": "character" if use_char_mode else "word",
                "top3": json.dumps(result_top3),
            },
        )

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
    gap_threshold: float = 56.0,
) -> List[List[Sequence[StrokePoint]]]:
    """Cluster stroke groups by vertical position into reading lines."""

    strokes_with_centers = []
    for stroke in stroke_groups:
        if not stroke:
            continue
        ys = [point.y for point in stroke]
        xs = [point.x for point in stroke]
        strokes_with_centers.append(
            {
                "stroke": stroke,
                "center_y": float(np.mean(ys)),
                "center_x": float(np.mean(xs)),
            }
        )

    if not strokes_with_centers:
        return []

    strokes_with_centers.sort(key=lambda item: (item["center_y"], item["center_x"]))
    lines: List[List[Sequence[StrokePoint]]] = [[strokes_with_centers[0]["stroke"]]]
    line_centers = [strokes_with_centers[0]["center_y"]]

    for item in strokes_with_centers[1:]:
        center_y = item["center_y"]
        if abs(center_y - line_centers[-1]) > gap_threshold:
            lines.append([item["stroke"]])
            line_centers.append(center_y)
            continue

        lines[-1].append(item["stroke"])
        line_centers[-1] = float(np.mean([line_centers[-1], center_y]))

    return lines


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


def _looks_like_single_character_input(stroke_groups: Sequence[Sequence[StrokePoint]]) -> bool:
    if not stroke_groups:
        return False
    total_points = sum(len(stroke) for stroke in stroke_groups)
    return len(stroke_groups) <= 3 and total_points <= 220


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

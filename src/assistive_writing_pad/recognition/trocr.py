"""Pretrained TrOCR handwritten text recognizer.

The default checkpoint is Microsoft's IAM-finetuned handwritten TrOCR model.
Imports are lazy so the rest of the application remains usable before model
dependencies are installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from assistive_writing_pad.contracts import RecognitionResult, StrokePoint

DEFAULT_TROCR_MODEL = "microsoft/trocr-small-handwritten"


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

    def recognize(self, strokes: Sequence[StrokePoint]) -> RecognitionResult:
        if not strokes:
            return RecognitionResult(
                text="",
                confidence=0.0,
                metadata={"recognizer": "trocr", "reason": "empty_strokes"},
            )

        self._ensure_loaded()
        image = render_strokes_for_trocr(strokes)

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values

        with self._torch.no_grad():
            generated = self._model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

        text = self._processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
        confidence = _generation_confidence(generated, self._torch)
        return RecognitionResult(
            text=text.strip(),
            confidence=confidence,
            metadata={"recognizer": "trocr", "model": self.model_name},
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
                "`pip install -e '.[models]'`."
            ) from exc

        self._torch = torch
        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self._model.eval()


def render_strokes_for_trocr(
    strokes: Sequence[StrokePoint],
    size: tuple = (384, 128),
    padding: int = 14,
) -> np.ndarray:
    """Render captured strokes as a white-background RGB image for TrOCR."""

    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    if not strokes:
        return image

    points = _scale_points(strokes, width, height, padding)
    if len(points) == 1:
        x, y = points[0]
        _draw_dot(image, x, y, radius=2)
        return image

    for start, end in zip(points, points[1:]):
        _draw_line(image, start, end, radius=2)
    return image


def _scale_points(
    strokes: Sequence[StrokePoint],
    width: int,
    height: int,
    padding: int,
) -> list:
    xs = [point.x for point in strokes]
    ys = [point.y for point in strokes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

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

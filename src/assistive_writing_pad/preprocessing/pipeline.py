"""Preprocessing pipeline from tablet strokes to model-ready arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.preprocessing.image_ops import (
    crop_to_content,
    normalize_unit,
    pad_to_square,
    resize_nearest,
)
from assistive_writing_pad.preprocessing.rasterize import RasterizerConfig, rasterize_strokes


@dataclass(frozen=True)
class PreprocessingConfig:
    rasterizer: RasterizerConfig = RasterizerConfig()
    output_size: Tuple[int, int] = (28, 28)


@dataclass(frozen=True)
class PreprocessedImage:
    image: np.ndarray
    original_point_count: int
    output_size: Tuple[int, int]


class StrokePreprocessor:
    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()) -> None:
        self.config = config

    def preprocess(self, points: Sequence[StrokePoint]) -> PreprocessedImage:
        raster = rasterize_strokes(points, self.config.rasterizer)
        if not points:
            return PreprocessedImage(
                image=np.zeros(self.config.output_size, dtype=np.float32),
                original_point_count=0,
                output_size=self.config.output_size,
            )

        cropped = crop_to_content(raster)
        squared = pad_to_square(cropped)
        resized = resize_nearest(squared, self.config.output_size)
        normalized = normalize_unit(resized)
        return PreprocessedImage(
            image=normalized.astype(np.float32),
            original_point_count=len(points),
            output_size=self.config.output_size,
        )

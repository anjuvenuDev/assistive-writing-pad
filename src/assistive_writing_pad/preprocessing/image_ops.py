"""Small image operations for CPU-only preprocessing."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def crop_to_content(image: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    rows, cols = np.where(image > threshold)
    if rows.size == 0 or cols.size == 0:
        return image.copy()

    return image[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def pad_to_square(image: np.ndarray, value: float = 0.0) -> np.ndarray:
    height, width = image.shape
    size = max(height, width)
    output = np.full((size, size), value, dtype=image.dtype)

    y_offset = (size - height) // 2
    x_offset = (size - width) // 2
    output[y_offset : y_offset + height, x_offset : x_offset + width] = image
    return output


def resize_nearest(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_height, target_width = size
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target size must be positive")

    source_height, source_width = image.shape
    y_indices = np.linspace(0, source_height - 1, target_height).round().astype(int)
    x_indices = np.linspace(0, source_width - 1, target_width).round().astype(int)
    return image[y_indices][:, x_indices]


def normalize_unit(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return image.astype(np.float32)

    max_value = float(image.max())
    if max_value <= 0.0:
        return image.astype(np.float32)
    return (image.astype(np.float32) / max_value).clip(0.0, 1.0)

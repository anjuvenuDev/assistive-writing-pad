"""Rasterize tablet stroke points into grayscale model input arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from assistive_writing_pad.contracts import StrokePoint


@dataclass(frozen=True)
class RasterizerConfig:
    canvas_size: Tuple[int, int] = (128, 128)
    padding_px: int = 8
    line_radius_px: int = 1
    pressure_sensitive: bool = True


def rasterize_strokes(
    points: Sequence[StrokePoint],
    config: RasterizerConfig = RasterizerConfig(),
) -> np.ndarray:
    height, width = config.canvas_size
    if height <= 0 or width <= 0:
        raise ValueError("canvas_size values must be positive")

    image = np.zeros((height, width), dtype=np.float32)
    if not points:
        return image

    scaled = _scale_points(points, width, height, config.padding_px)
    for start, end in zip(scaled, scaled[1:]):
        _draw_segment(image, start, end, config.line_radius_px, config.pressure_sensitive)

    if len(scaled) == 1:
        point = scaled[0]
        _draw_dot(image, point[0], point[1], config.line_radius_px, _pressure_intensity(point[2]))

    return image.clip(0.0, 1.0)


def _scale_points(
    points: Sequence[StrokePoint],
    width: int,
    height: int,
    padding_px: int,
) -> Tuple[Tuple[int, int, float], ...]:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    x_range = max(max_x - min_x, 1.0)
    y_range = max(max_y - min_y, 1.0)
    draw_width = max(width - padding_px * 2 - 1, 1)
    draw_height = max(height - padding_px * 2 - 1, 1)
    scale = min(draw_width / x_range, draw_height / y_range)

    content_width = x_range * scale
    content_height = y_range * scale
    x_offset = (width - content_width) / 2.0
    y_offset = (height - content_height) / 2.0

    scaled = []
    for point in points:
        x = int(round((point.x - min_x) * scale + x_offset))
        y = int(round((point.y - min_y) * scale + y_offset))
        scaled.append((min(max(x, 0), width - 1), min(max(y, 0), height - 1), point.pressure))
    return tuple(scaled)


def _draw_segment(
    image: np.ndarray,
    start: Tuple[int, int, float],
    end: Tuple[int, int, float],
    radius: int,
    pressure_sensitive: bool,
) -> None:
    x0, y0, p0 = start
    x1, y1, p1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)

    for step in range(steps + 1):
        alpha = step / steps
        x = int(round(x0 + (x1 - x0) * alpha))
        y = int(round(y0 + (y1 - y0) * alpha))
        pressure = p0 + (p1 - p0) * alpha
        intensity = _pressure_intensity(pressure) if pressure_sensitive else 1.0
        _draw_dot(image, x, y, radius, intensity)


def _draw_dot(image: np.ndarray, x: int, y: int, radius: int, intensity: float) -> None:
    height, width = image.shape
    for row in range(max(0, y - radius), min(height, y + radius + 1)):
        for col in range(max(0, x - radius), min(width, x + radius + 1)):
            image[row, col] = max(image[row, col], intensity)


def _pressure_intensity(pressure: float) -> float:
    if pressure <= 0.0:
        return 0.5
    return min(max(pressure, 0.05), 1.0)

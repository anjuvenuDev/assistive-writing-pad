import numpy as np
import pytest

from assistive_writing_pad.capture.simulator import StrokeSimulator
from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.preprocessing.image_ops import resize_nearest
from assistive_writing_pad.preprocessing.pipeline import StrokePreprocessor
from assistive_writing_pad.preprocessing.rasterize import RasterizerConfig, rasterize_strokes


def test_preprocessor_returns_empty_model_image_for_empty_input() -> None:
    result = StrokePreprocessor().preprocess([])

    assert result.image.shape == (28, 28)
    assert result.image.dtype == np.float32
    assert float(result.image.max()) == 0.0
    assert result.original_point_count == 0


def test_preprocessor_normalizes_variable_size_writing() -> None:
    points = [
        StrokePoint(x=1000.0, y=1000.0, timestamp_ms=0, pressure=0.7),
        StrokePoint(x=2400.0, y=1800.0, timestamp_ms=16, pressure=0.9),
        StrokePoint(x=3200.0, y=900.0, timestamp_ms=32, pressure=0.8),
    ]

    result = StrokePreprocessor().preprocess(points)

    assert result.image.shape == (28, 28)
    assert 0.0 <= float(result.image.min()) <= float(result.image.max()) <= 1.0
    assert float(result.image.max()) == pytest.approx(1.0)


def test_rasterizer_uses_pressure_for_intensity() -> None:
    low_pressure = [
        StrokePoint(x=0.0, y=0.0, timestamp_ms=0, pressure=0.2),
        StrokePoint(x=10.0, y=0.0, timestamp_ms=16, pressure=0.2),
    ]
    high_pressure = [
        StrokePoint(x=0.0, y=0.0, timestamp_ms=0, pressure=0.9),
        StrokePoint(x=10.0, y=0.0, timestamp_ms=16, pressure=0.9),
    ]

    config = RasterizerConfig(canvas_size=(32, 32), pressure_sensitive=True)

    assert rasterize_strokes(high_pressure, config).max() > rasterize_strokes(
        low_pressure, config
    ).max()


def test_preprocessor_accepts_simulated_capture_points() -> None:
    points = StrokeSimulator().write_text("teh")
    result = StrokePreprocessor().preprocess(points)

    assert result.original_point_count == len(points)
    assert result.image.shape == (28, 28)
    assert result.image.max() > 0


def test_resize_rejects_invalid_target_size() -> None:
    with pytest.raises(ValueError, match="target size"):
        resize_nearest(np.zeros((2, 2), dtype=np.float32), (0, 28))

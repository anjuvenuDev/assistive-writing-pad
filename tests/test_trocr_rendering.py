from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.recognition.trocr import render_strokes_for_trocr


def test_trocr_renderer_returns_rgb_numpy_line_image() -> None:
    image = render_strokes_for_trocr(
        [
            StrokePoint(x=10, y=20, timestamp_ms=0, pressure=1.0),
            StrokePoint(x=60, y=40, timestamp_ms=16, pressure=1.0),
        ]
    )

    assert image.shape == (128, 384, 3)
    assert image.dtype.name == "uint8"
    assert image.min() == 0
    assert image.max() == 255


def test_trocr_renderer_handles_empty_strokes() -> None:
    image = render_strokes_for_trocr([])

    assert image.shape == (128, 384, 3)
    assert image.min() == 255

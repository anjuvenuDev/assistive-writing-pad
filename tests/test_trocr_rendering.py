from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.contracts import RecognitionResult
from assistive_writing_pad.recognition.trocr import (
    TrOCRHandwritingRecognizer,
    _DEFAULT_RENDER_H,
    _DEFAULT_RENDER_W,
    render_strokes_for_trocr,
)


class StubEMNISTRecognizer:
    def recognize(self, strokes):
        return RecognitionResult(
            text="a",
            confidence=0.91,
            metadata={"recognizer": "emnist", "model": "stub"},
        )


def test_trocr_renderer_returns_rgb_numpy_line_image() -> None:
    image = render_strokes_for_trocr(
        [
            StrokePoint(x=10, y=20, timestamp_ms=0, pressure=1.0),
            StrokePoint(x=60, y=40, timestamp_ms=16, pressure=1.0),
        ]
    )

    # Shape must match the configured default render canvas.
    assert image.shape == (_DEFAULT_RENDER_H, _DEFAULT_RENDER_W, 3)
    assert image.dtype.name == "uint8"
    assert image.min() == 0
    assert image.max() == 255


def test_trocr_renderer_handles_empty_strokes() -> None:
    image = render_strokes_for_trocr([])

    assert image.shape == (_DEFAULT_RENDER_H, _DEFAULT_RENDER_W, 3)
    assert image.min() == 255


def test_character_mode_does_not_load_trocr(monkeypatch) -> None:
    recognizer = TrOCRHandwritingRecognizer()
    monkeypatch.setattr(
        "assistive_writing_pad.recognition.emnist.EMNISTCharacterRecognizer",
        StubEMNISTRecognizer,
    )

    def fail_if_loaded() -> None:
        raise AssertionError("TrOCR should not load for character mode")

    monkeypatch.setattr(recognizer, "_ensure_loaded", fail_if_loaded)
    result = recognizer.recognize([StrokePoint(x=10, y=20, timestamp_ms=0)], mode="character")

    assert result.text == "a"
    assert result.metadata["recognizer"] == "emnist"


def test_character_stroke_groups_do_not_load_trocr(monkeypatch) -> None:
    recognizer = TrOCRHandwritingRecognizer()
    monkeypatch.setattr(
        "assistive_writing_pad.recognition.emnist.EMNISTCharacterRecognizer",
        StubEMNISTRecognizer,
    )

    def fail_if_loaded() -> None:
        raise AssertionError("TrOCR should not load for character mode")

    monkeypatch.setattr(recognizer, "_ensure_loaded", fail_if_loaded)
    result = recognizer.recognize_stroke_groups(
        [[StrokePoint(x=10, y=20, timestamp_ms=0)]],
        mode="character",
    )

    assert result.text == "a"
    assert result.metadata["recognizer"] == "emnist"

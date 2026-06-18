from pathlib import Path

from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.recognition.template import TemplateGlyphRecognizer, TemplateStore


def diagonal_stroke(offset: float = 0.0):
    return [
        StrokePoint(x=0.0 + offset, y=0.0, timestamp_ms=0, pressure=1.0),
        StrokePoint(x=10.0 + offset, y=10.0, timestamp_ms=16, pressure=1.0),
        StrokePoint(x=20.0 + offset, y=20.0, timestamp_ms=32, pressure=1.0),
    ]


def horizontal_stroke():
    return [
        StrokePoint(x=0.0, y=10.0, timestamp_ms=0, pressure=1.0),
        StrokePoint(x=10.0, y=10.0, timestamp_ms=16, pressure=1.0),
        StrokePoint(x=20.0, y=10.0, timestamp_ms=32, pressure=1.0),
    ]


def test_template_recognizer_reports_no_templates(tmp_path: Path) -> None:
    recognizer = TemplateGlyphRecognizer(store=TemplateStore(tmp_path / "templates.json"))

    result = recognizer.recognize(diagonal_stroke())

    assert result.text == ""
    assert result.confidence == 0.0
    assert result.metadata["reason"] == "no_templates"


def test_template_recognizer_learns_and_recognizes_same_stroke(tmp_path: Path) -> None:
    recognizer = TemplateGlyphRecognizer(store=TemplateStore(tmp_path / "templates.json"))

    recognizer.learn("a", diagonal_stroke())
    result = recognizer.recognize(diagonal_stroke(offset=4.0))

    assert result.text == "a"
    assert result.confidence > 0.95


def test_template_store_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "templates.json"
    recognizer = TemplateGlyphRecognizer(store=TemplateStore(path))
    recognizer.learn("a", diagonal_stroke())
    recognizer.learn("b", horizontal_stroke())

    loaded = TemplateStore.load(path)

    assert loaded.labels() == ["a", "b"]
    assert len(loaded.samples) == 2

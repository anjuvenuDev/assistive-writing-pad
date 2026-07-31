import pytest

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import RecognitionResult
from assistive_writing_pad.correction.contextual import ContextualCorrector
from assistive_writing_pad.display.web_app import RecognitionService, stroke_groups_from_payload


class StubStrokeGroupRecognizer:
    def recognize_stroke_groups(self, stroke_groups, mode="auto") -> RecognitionResult:
        assert len(stroke_groups) == 1
        return RecognitionResult(
            text="teh cat sat on a chaier",
            confidence=0.92,
            metadata={"recognizer": "stub", "mode": mode, "top3": "[]"},
        )


def test_stroke_groups_from_payload_parses_strokes() -> None:
    points = stroke_groups_from_payload(
        {
            "strokes": [
                [
                    {"x": 1, "y": 2, "timestamp_ms": 0, "pressure": 0.5},
                    {"x": 3.2, "y": 4.5, "timestamp_ms": 16},
                ],
                [{"x": 10, "y": 20, "timestamp_ms": 0}],
            ]
        }
    )

    assert len(points) == 2
    assert len(points[0]) == 2
    assert points[0][0].x == 1.0
    assert points[0][0].pressure == 0.5
    assert points[0][1].pressure == 1.0
    assert points[1][0].x == 10.0


def test_stroke_groups_from_payload_rejects_missing_list() -> None:
    with pytest.raises(ValueError, match="strokes list"):
        stroke_groups_from_payload({})


def test_recognition_service_returns_realtime_correction_metadata() -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=ContextualCorrector(model_enabled=False),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )

    result = service.recognize_payload(
        {"strokes": [[{"x": 1, "y": 2, "timestamp_ms": 0}]]}
    )

    assert result["recognized_text"] == "teh cat sat on a chaier"
    assert result["corrected_text"] == "the cat sat on a chair"
    assert result["text"] == "the cat sat on a chair"
    assert result["needs_review"] is False
    assert result["mode"] == "ocr"
    assert result["corrections"][0]["original"] == "teh"


def test_recognition_service_accepts_legacy_mode_values() -> None:
    service = RecognitionService(
        recognizer=StubStrokeGroupRecognizer(),
        corrector=ContextualCorrector(model_enabled=False),
        settings=RuntimeSettings(contextual_model_enabled=False),
    )

    result = service.recognize_payload(
        {"strokes": [[{"x": 1, "y": 2, "timestamp_ms": 0}]], "mode": "word"}
    )

    assert result["mode"] == "word"

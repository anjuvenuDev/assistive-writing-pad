import json

from assistive_writing_pad.contracts import RecognitionResult, StrokePoint
from assistive_writing_pad.eval.recognition_eval import (
    RecognitionCase,
    evaluate_recognition_cases,
    load_recognition_cases,
    normalize_for_recognition_eval,
    parse_stroke_groups,
)


class SequenceRecognizer:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def recognize_stroke_groups(self, stroke_groups, mode="ocr") -> RecognitionResult:
        assert stroke_groups
        assert mode == "auto"
        return self.outputs.pop(0)


def test_parse_stroke_groups_builds_immutable_stroke_points() -> None:
    groups = parse_stroke_groups(
        [
            [{"x": 1, "y": 2, "timestamp_ms": 0}],
            [{"x": 3.5, "y": 4.5}],
        ]
    )

    assert groups == (
        (StrokePoint(x=1.0, y=2.0, timestamp_ms=0, pressure=1.0),),
        (StrokePoint(x=3.5, y=4.5, timestamp_ms=0, pressure=1.0),),
    )


def test_load_recognition_cases_from_jsonl(tmp_path) -> None:
    manifest = tmp_path / "recognition.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "case_1",
                "category": "single_character",
                "expected": "i",
                "strokes": [[{"x": 1, "y": 2, "timestamp_ms": 0}]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_recognition_cases(manifest)

    assert cases == [
        RecognitionCase(
            id="case_1",
            category="single_character",
            expected_text="i",
            stroke_groups=((StrokePoint(x=1.0, y=2.0, timestamp_ms=0, pressure=1.0),),),
        )
    ]


def test_evaluate_recognition_cases_summarizes_accuracy_error_rates_and_latency() -> None:
    cases = [
        RecognitionCase(
            id="exact",
            category="word",
            expected_text="cat",
            stroke_groups=((StrokePoint(1, 2, 0),),),
        ),
        RecognitionCase(
            id="wrong",
            category="word",
            expected_text="dog",
            stroke_groups=((StrokePoint(3, 4, 0),),),
        ),
    ]
    recognizer = SequenceRecognizer(
        [
            RecognitionResult(text="Cat", confidence=0.91),
            RecognitionResult(text="dig", confidence=0.42),
        ]
    )

    report = evaluate_recognition_cases(
        cases,
        recognizer,
        mode="auto",
        confidence_threshold=0.65,
    )

    assert report.summary.total == 2
    assert report.summary.exact == 1
    assert report.summary.accuracy == 0.5
    assert report.summary.low_confidence == 1
    assert report.summary.average_cer > 0.0
    assert report.summary.by_category["word"].low_confidence == 1
    assert [row.id for row in report.rows if not row.exact_match] == ["wrong"]


def test_normalize_for_recognition_eval_ignores_case_spacing_and_curly_apostrophes() -> None:
    assert normalize_for_recognition_eval("  I\u2019m  Here . ") == "i'm here."

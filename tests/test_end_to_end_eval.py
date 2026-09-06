import json

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import CorrectionResult, RecognitionResult, StrokePoint
from assistive_writing_pad.eval.end_to_end_eval import (
    EndToEndCase,
    evaluate_end_to_end_cases,
    load_end_to_end_cases,
)


class SequenceRecognizer:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def recognize_stroke_groups(self, stroke_groups, mode="ocr") -> RecognitionResult:
        assert stroke_groups
        assert mode == "auto"
        return self.outputs.pop(0)


class SequenceCorrector:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def correct(self, text: str) -> CorrectionResult:
        result = self.outputs.pop(0)
        assert result.original_text == text
        return result


def test_load_end_to_end_cases_from_jsonl(tmp_path) -> None:
    manifest = tmp_path / "end_to_end.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "case_1",
                "category": "single_character",
                "expected": "i",
                "expected_recognized": "I",
                "strokes": [[{"x": 1, "y": 2, "timestamp_ms": 0}]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_end_to_end_cases(manifest)

    assert cases == [
        EndToEndCase(
            id="case_1",
            category="single_character",
            expected_corrected_text="i",
            expected_recognized_text="I",
            stroke_groups=((StrokePoint(x=1.0, y=2.0, timestamp_ms=0, pressure=1.0),),),
        )
    ]


def test_evaluate_end_to_end_cases_summarizes_recognition_and_correction() -> None:
    cases = [
        EndToEndCase(
            id="clean",
            category="word",
            expected_recognized_text="teh",
            expected_corrected_text="the",
            stroke_groups=((StrokePoint(1, 2, 0),),),
        ),
        EndToEndCase(
            id="wrong",
            category="word",
            expected_recognized_text="cat",
            expected_corrected_text="cat",
            stroke_groups=((StrokePoint(3, 4, 0),),),
        ),
    ]
    recognizer = SequenceRecognizer(
        [
            RecognitionResult(text="teh", confidence=0.91),
            RecognitionResult(text="cot", confidence=0.42),
        ]
    )
    corrector = SequenceCorrector(
        [
            CorrectionResult(original_text="teh", corrected_text="the", confidence=0.88),
            CorrectionResult(original_text="cot", corrected_text="cot", confidence=0.72),
        ]
    )

    report = evaluate_end_to_end_cases(
        cases,
        recognizer,
        corrector,
        settings=RuntimeSettings(confidence_threshold=0.65),
        mode="auto",
        confidence_threshold=0.65,
    )

    assert report.summary.total == 2
    assert report.summary.recognition_exact == 1
    assert report.summary.corrected_exact == 1
    assert report.summary.low_confidence == 1
    assert report.summary.needs_review == 1
    assert report.summary.by_category["word"].corrected_exact == 1
    assert [row.id for row in report.rows if not row.corrected_exact_match] == ["wrong"]

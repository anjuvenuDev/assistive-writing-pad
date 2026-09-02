import json

from assistive_writing_pad.contracts import Correction, CorrectionResult, RecognitionResult, StrokePoint
from assistive_writing_pad.display.handwriting_app import (
    TextAlternative,
    collect_text_alternatives,
    flatten_strokes,
    format_correction_lines,
    parse_json_list,
)


def test_flatten_strokes_preserves_point_order() -> None:
    first = StrokePoint(x=1, y=2, timestamp_ms=0)
    second = StrokePoint(x=3, y=4, timestamp_ms=1)
    third = StrokePoint(x=5, y=6, timestamp_ms=2)

    assert flatten_strokes(((first, second), (third,))) == [first, second, third]


def test_collect_text_alternatives_merges_ocr_and_model_options() -> None:
    recognition = RecognitionResult(
        text="teh cat",
        confidence=0.82,
        metadata={"top3": json.dumps([["teh cat", 0.82], ["the cat", 0.74]])},
    )
    correction = CorrectionResult(
        original_text="teh cat",
        corrected_text="the cat",
        metadata={
            "stages": json.dumps(
                [
                    {
                        "stage": "spelling",
                        "alternatives": [
                            {"text": "the cat", "confidence": 0.91},
                            {"text": "ten cat", "confidence": 0.55},
                        ],
                    }
                ]
            )
        },
    )

    alternatives = collect_text_alternatives(recognition, correction)

    assert alternatives == [
        TextAlternative("teh cat", 0.82),
        TextAlternative("the cat", 0.74),
        TextAlternative("ten cat", 0.55),
    ]


def test_collect_text_alternatives_applies_limit() -> None:
    recognition = RecognitionResult(
        text="one",
        confidence=0.9,
        metadata={"top3": json.dumps([["two", 0.8], ["three", 0.7]])},
    )

    assert collect_text_alternatives(
        recognition,
        CorrectionResult(original_text="one", corrected_text="one"),
        limit=2,
    ) == [
        TextAlternative("one", 0.9),
        TextAlternative("two", 0.8),
    ]


def test_format_correction_lines_renders_changes_for_review() -> None:
    result = CorrectionResult(
        original_text="teh cat",
        corrected_text="the cat.",
        corrections=(
            Correction("teh", "the", 0.91, "hf_spelling_model"),
            Correction("", ".", 0.88, "hf_grammar_model"),
        ),
    )

    assert format_correction_lines(result) == "teh -> the  91%\n[insert] -> .  88%"


def test_format_correction_lines_handles_clean_text() -> None:
    assert (
        format_correction_lines(CorrectionResult(original_text="The cat.", corrected_text="The cat."))
        == "No changes."
    )


def test_parse_json_list_rejects_non_list_values() -> None:
    assert parse_json_list("{not json") == []
    assert parse_json_list('{"text": "cat"}') == []
    assert parse_json_list("[1, 2]") == [1, 2]

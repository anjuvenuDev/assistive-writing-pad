from assistive_writing_pad.contracts import (
    CorrectionResult,
    RecognitionHypothesisSelection,
    RecognitionResult,
)
from assistive_writing_pad.correction.rule_based import RuleBasedCorrector
from assistive_writing_pad.pipeline import WritingPipeline
from assistive_writing_pad.recognition.demo import DemoRecognizer


class BrokenCorrector:
    def correct(self, text: str):
        raise RuntimeError("correction model failed")


class SelectingCorrector:
    def select_recognition_candidate(self, candidates):
        assert ("I fed the cat", 0.72) in candidates
        return RecognitionHypothesisSelection(
            text="I fed the cat",
            confidence=0.88,
            rankings=(("I fed the cat", 0.88), ("I fed the cot", 0.70)),
            metadata={"selector": "test-selector"},
        )

    def correct(self, text: str) -> CorrectionResult:
        return CorrectionResult(original_text=text, corrected_text=text)


def test_pipeline_applies_correction_when_confidence_is_high() -> None:
    pipeline = WritingPipeline(
        recognizer=DemoRecognizer(text="teh cat sat on a chaier", confidence=0.92),
        corrector=RuleBasedCorrector(),
    )

    result = pipeline.process_strokes([])

    assert result.needs_review is False
    assert result.recognition.text == "teh cat sat on a chaier"
    assert result.correction.corrected_text == "the cat sat on a chair"
    assert [correction.reason for correction in result.correction.corrections] == [
        "letter_swap",
        "phonetic_or_insertion_error",
    ]


def test_pipeline_flags_low_confidence_recognition_for_review() -> None:
    pipeline = WritingPipeline(
        recognizer=DemoRecognizer(text="teh cat sat on a chaier", confidence=0.52),
        corrector=RuleBasedCorrector(),
    )

    result = pipeline.process_strokes([])

    assert result.needs_review is True
    assert result.review_reason == "recognition_confidence_below_threshold"
    assert result.correction.corrected_text == "the cat sat on a chair"


def test_pipeline_does_not_fail_when_correction_model_fails() -> None:
    pipeline = WritingPipeline(
        recognizer=DemoRecognizer(text="the cat", confidence=0.92),
        corrector=BrokenCorrector(),
    )

    result = pipeline.process_strokes([])

    assert result.needs_review is False
    assert result.correction.corrected_text == "the cat"
    assert result.correction.confidence == 0.0


def test_pipeline_automatically_selects_best_ocr_hypothesis_before_correction() -> None:
    pipeline = WritingPipeline(
        recognizer=DemoRecognizer(text="unused", confidence=1.0),
        corrector=SelectingCorrector(),
    )
    recognition = RecognitionResult(
        text="I fed the cot",
        confidence=0.91,
        metadata={"top3": '[["I fed the cot", 0.91], ["I fed the cat", 0.72]]'},
    )

    result = pipeline.process_recognition(recognition)

    assert result.recognition.text == "I fed the cat"
    assert result.correction.corrected_text == "I fed the cat"
    assert result.recognition.metadata["selected_ocr_text"] == "I fed the cat"

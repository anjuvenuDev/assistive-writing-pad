from assistive_writing_pad.correction.rule_based import RuleBasedCorrector
from assistive_writing_pad.pipeline import WritingPipeline
from assistive_writing_pad.recognition.demo import DemoRecognizer


class BrokenCorrector:
    def correct(self, text: str):
        raise RuntimeError("correction model failed")


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

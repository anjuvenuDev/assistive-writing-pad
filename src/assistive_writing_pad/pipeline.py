"""Core orchestration for handwriting recognition and text correction."""

from dataclasses import dataclass, field
from typing import Sequence

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import (
    CorrectionResult,
    HandwritingRecognizer,
    PipelineResult,
    RecognitionResult,
    StrokePoint,
    TextCorrector,
)


@dataclass
class WritingPipeline:
    recognizer: HandwritingRecognizer
    corrector: TextCorrector
    settings: RuntimeSettings = field(default_factory=RuntimeSettings)

    def __post_init__(self) -> None:
        self.settings.validate()

    def process_strokes(self, strokes: Sequence[StrokePoint]) -> PipelineResult:
        recognition = self.recognizer.recognize(strokes)

        if recognition.confidence < self.settings.confidence_threshold:
            return PipelineResult(
                recognition=recognition,
                correction=CorrectionResult(
                    original_text=recognition.text,
                    corrected_text=recognition.text,
                    confidence=recognition.confidence,
                ),
                needs_review=True,
                review_reason="recognition_confidence_below_threshold",
            )

        if recognition.is_empty:
            return PipelineResult(
                recognition=recognition,
                correction=CorrectionResult(original_text="", corrected_text="", confidence=1.0),
                needs_review=False,
            )

        correction = self.corrector.correct(recognition.text)
        return PipelineResult(
            recognition=recognition,
            correction=correction,
            needs_review=False,
        )

"""Core orchestration for handwriting recognition and text correction."""

from dataclasses import dataclass, field
import logging
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

logger = logging.getLogger(__name__)


@dataclass
class WritingPipeline:
    recognizer: HandwritingRecognizer
    corrector: TextCorrector
    settings: RuntimeSettings = field(default_factory=RuntimeSettings)

    def __post_init__(self) -> None:
        self.settings.validate()

    def process_strokes(self, strokes: Sequence[StrokePoint]) -> PipelineResult:
        recognition = self.recognizer.recognize(strokes)
        return self.process_recognition(recognition)

    def process_recognition(self, recognition: RecognitionResult) -> PipelineResult:
        logger.info("RAW OCR: %s", recognition.text)
        logger.info("OCR CONFIDENCE: %.3f", recognition.confidence)
        if recognition.confidence < self.settings.confidence_threshold:
            correction = CorrectionResult(
                original_text=recognition.text,
                corrected_text=recognition.text,
                confidence=recognition.confidence,
                metadata={
                    "correction_guardrail": "skipped_low_ocr_confidence",
                    "ocr_confidence": f"{recognition.confidence:.4f}",
                },
            )
            logger.info("CORRECTED: %s (preserved raw OCR; review required)", correction.corrected_text)
            return PipelineResult(
                recognition=recognition,
                correction=correction,
                needs_review=True,
                review_reason="recognition_confidence_below_threshold",
            )

        correction = self._correct_recognition(recognition)
        logger.info("CORRECTED: %s", correction.corrected_text)
        return PipelineResult(
            recognition=recognition,
            correction=correction,
            needs_review=False,
        )

    def _correct_recognition(self, recognition: RecognitionResult) -> CorrectionResult:
        if recognition.is_empty:
            return CorrectionResult(original_text="", corrected_text="", confidence=1.0)

        try:
            return self.corrector.correct(recognition.text)
        except Exception:
            logger.exception("text correction failed; returning raw recognition")
            return CorrectionResult(
                original_text=recognition.text,
                corrected_text=recognition.text,
                confidence=0.0,
            )

"""Core orchestration for handwriting recognition and text correction."""

from dataclasses import dataclass, field
import json
import logging
from typing import List, Sequence, Tuple

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
        recognition = self._select_recognition_hypothesis(recognition)
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

    def _select_recognition_hypothesis(
        self,
        recognition: RecognitionResult,
    ) -> RecognitionResult:
        selector = getattr(self.corrector, "select_recognition_candidate", None)
        if not callable(selector) or recognition.is_empty:
            return recognition
        candidates = recognition_candidates(recognition)
        if len(candidates) < 2:
            return recognition
        try:
            selection = selector(candidates)
        except Exception:
            logger.exception("recognition hypothesis selection failed; using TrOCR primary")
            return recognition
        if not selection.text.strip():
            return recognition

        metadata = dict(recognition.metadata)
        metadata.update(selection.metadata)
        metadata["selected_ocr_text"] = selection.text
        metadata["candidate_rankings"] = json.dumps(selection.rankings)
        return RecognitionResult(
            text=selection.text,
            confidence=selection.confidence,
            character_confidences=recognition.character_confidences,
            metadata=metadata,
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


def recognition_candidates(recognition: RecognitionResult) -> Tuple[Tuple[str, float], ...]:
    candidates: List[Tuple[str, float]] = [(recognition.text, recognition.confidence)]
    try:
        alternatives = json.loads(recognition.metadata.get("top3", "[]"))
    except (json.JSONDecodeError, TypeError):
        alternatives = []
    for item in alternatives:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            candidates.append((str(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    unique = {}
    for text, confidence in candidates:
        key = text.strip().casefold()
        if not key:
            continue
        current = unique.get(key)
        if current is None or confidence > current[1]:
            unique[key] = (text.strip(), confidence)
    return tuple(unique.values())

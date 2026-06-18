"""Shared data contracts and component protocols."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class StrokePoint:
    """Single tablet sample from a pen stroke."""

    x: float
    y: float
    timestamp_ms: int
    pressure: float = 0.0


Stroke = Sequence[StrokePoint]


@dataclass(frozen=True)
class CharacterConfidence:
    character: str
    confidence: float


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    confidence: float
    character_confidences: Tuple[CharacterConfidence, ...] = ()
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.text.strip() == ""


@dataclass(frozen=True)
class Correction:
    original: str
    corrected: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class CorrectionResult:
    original_text: str
    corrected_text: str
    corrections: Tuple[Correction, ...] = ()
    confidence: float = 1.0

    @property
    def changed(self) -> bool:
        return self.original_text != self.corrected_text


@dataclass(frozen=True)
class PipelineResult:
    recognition: RecognitionResult
    correction: CorrectionResult
    needs_review: bool
    review_reason: Optional[str] = None


class HandwritingRecognizer(Protocol):
    def recognize(self, strokes: Sequence[StrokePoint]) -> RecognitionResult:
        """Recognize text from tablet stroke points."""


class TextCorrector(Protocol):
    def correct(self, text: str) -> CorrectionResult:
        """Return corrected text and correction metadata."""

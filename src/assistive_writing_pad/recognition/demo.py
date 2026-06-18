"""Deterministic recognizer used before model integration."""

from dataclasses import dataclass
from typing import Sequence

from assistive_writing_pad.contracts import CharacterConfidence, RecognitionResult, StrokePoint


@dataclass(frozen=True)
class DemoRecognizer:
    text: str
    confidence: float

    def recognize(self, strokes: Sequence[StrokePoint]) -> RecognitionResult:
        del strokes
        return RecognitionResult(
            text=self.text,
            confidence=self.confidence,
            character_confidences=tuple(
                CharacterConfidence(character=character, confidence=self.confidence)
                for character in self.text
                if character != " "
            ),
            metadata={"recognizer": "demo"},
        )

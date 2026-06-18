"""Small rule-based corrector for deterministic demos and early tests."""

from dataclasses import dataclass, field
from typing import Dict, List

from assistive_writing_pad.contracts import Correction, CorrectionResult


DEFAULT_REPLACEMENTS = {
    "teh": ("the", "letter_swap"),
    "chaier": ("chair", "phonetic_or_insertion_error"),
    "fone": ("phone", "phonetic_error"),
    "sed": ("said", "phonetic_error"),
    "rite": ("right", "phonetic_error"),
    "becaus": ("because", "omission"),
    "recieve": ("receive", "letter_order"),
    "writting": ("writing", "doubling_error"),
    "occured": ("occurred", "doubling_error"),
}


@dataclass(frozen=True)
class RuleBasedCorrector:
    replacements: Dict[str, tuple] = field(default_factory=lambda: dict(DEFAULT_REPLACEMENTS))

    def correct(self, text: str) -> CorrectionResult:
        words = text.split()
        corrected_words: List[str] = []
        corrections: List[Correction] = []

        for word in words:
            replacement = self.replacements.get(word.lower())
            if replacement is None:
                corrected_words.append(word)
                continue

            corrected, reason = replacement
            corrected_words.append(_preserve_basic_case(word, corrected))
            corrections.append(
                Correction(
                    original=word,
                    corrected=_preserve_basic_case(word, corrected),
                    confidence=0.98,
                    reason=reason,
                )
            )

        corrected_text = " ".join(corrected_words)
        confidence = min((item.confidence for item in corrections), default=1.0)
        return CorrectionResult(
            original_text=text,
            corrected_text=corrected_text,
            corrections=tuple(corrections),
            confidence=confidence,
        )


def _preserve_basic_case(original: str, corrected: str) -> str:
    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected.capitalize()
    return corrected

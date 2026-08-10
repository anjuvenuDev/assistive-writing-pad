"""Compatibility wrapper for the intelligent post-OCR correction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import CorrectionResult
from assistive_writing_pad.correction.memory import CorrectionMemory
from assistive_writing_pad.correction.pipeline import IntelligentCorrectionPipeline
from assistive_writing_pad.correction.rule_based import DEFAULT_REPLACEMENTS
from assistive_writing_pad.correction.spell_corrector import SpellCorrector
from assistive_writing_pad.correction.context_corrector import LocalContextScorer

DEFAULT_CONTEXT_MODEL = "distilbert/distilbert-base-uncased"


@dataclass
class ContextualCorrector:
    """Backward-compatible entrypoint used by the pipeline and web service."""

    replacements: Dict[str, Tuple[str, str]] = field(default_factory=lambda: dict(DEFAULT_REPLACEMENTS))
    model_enabled: bool = True
    model_name: str = DEFAULT_CONTEXT_MODEL
    max_candidates: int = 8
    max_transpositions: int = 2
    confidence_threshold: float = 0.70
    auto_threshold: float = 0.90
    memory: Optional[CorrectionMemory] = None
    dictionary_path: str = ""
    debug: bool = False

    def __post_init__(self) -> None:
        self._pipeline = IntelligentCorrectionPipeline(
            spell_corrector=SpellCorrector(dictionary_path=self.dictionary_path),
            # This is a tiny local scorer, not the optional legacy transformer.
            # Keep it available even when model_enabled=False so offline context
            # corrections retain their intended behaviour.
            context_scorer=LocalContextScorer(enabled=True),
            auto_threshold=self.auto_threshold,
            suggestion_threshold=self.confidence_threshold,
            max_candidates=self.max_candidates,
            max_transpositions=self.max_transpositions,
            replacements=self.replacements,
            memory=self.memory,
            debug=self.debug,
        )

    @classmethod
    def from_settings(cls, settings: RuntimeSettings) -> "ContextualCorrector":
        engine = IntelligentCorrectionPipeline.from_settings(settings)
        return cls(
            replacements=engine.replacements,
            model_enabled=settings.correction_mode == "contextual",
            model_name=settings.contextual_model_name,
            max_candidates=settings.max_correction_candidates,
            max_transpositions=settings.correction_max_transpositions,
            confidence_threshold=settings.correction_suggestion_threshold,
            auto_threshold=settings.correction_auto_threshold,
            memory=engine.memory,
            dictionary_path=settings.correction_dictionary_path,
            debug=settings.debug_correction,
        )

    def correct(self, text: str) -> CorrectionResult:
        return self._pipeline.correct(text)

    def record_feedback(self, original: str, corrected: str) -> None:
        self._pipeline.record_feedback(original, corrected)

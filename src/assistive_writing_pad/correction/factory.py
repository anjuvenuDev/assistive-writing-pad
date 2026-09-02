"""Correction backend selection."""

from __future__ import annotations

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import TextCorrector
from assistive_writing_pad.correction.contextual import ContextualCorrector
from assistive_writing_pad.correction.huggingface import HuggingFaceCorrectionPipeline
from assistive_writing_pad.correction.rule_based import RuleBasedCorrector


def corrector_from_settings(settings: RuntimeSettings) -> TextCorrector:
    if settings.correction_mode == "hf":
        return HuggingFaceCorrectionPipeline.from_settings(settings)
    if settings.correction_mode == "contextual":
        return ContextualCorrector.from_settings(settings)
    if settings.correction_mode == "rules":
        return RuleBasedCorrector()
    raise ValueError(f"unsupported correction mode: {settings.correction_mode}")

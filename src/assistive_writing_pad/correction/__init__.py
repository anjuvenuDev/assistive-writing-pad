"""Text correction components."""

from assistive_writing_pad.correction.contextual import ContextualCorrector
from assistive_writing_pad.correction.factory import corrector_from_settings
from assistive_writing_pad.correction.huggingface import HuggingFaceCorrectionPipeline
from assistive_writing_pad.correction.rule_based import RuleBasedCorrector
from assistive_writing_pad.correction.semantic import SemanticCorrectionRunner

__all__ = [
    "ContextualCorrector",
    "HuggingFaceCorrectionPipeline",
    "RuleBasedCorrector",
    "SemanticCorrectionRunner",
    "corrector_from_settings",
]

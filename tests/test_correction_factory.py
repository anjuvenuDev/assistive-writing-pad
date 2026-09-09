from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.correction.contextual import ContextualCorrector
from assistive_writing_pad.correction.factory import corrector_from_settings
from assistive_writing_pad.correction.huggingface import HuggingFaceCorrectionPipeline
from assistive_writing_pad.correction.rule_based import RuleBasedCorrector


def test_correction_factory_uses_huggingface_backend_by_default() -> None:
    corrector = corrector_from_settings(RuntimeSettings())

    assert isinstance(corrector, HuggingFaceCorrectionPipeline)
    assert corrector.lexical_runner is None
    assert corrector.semantic_runner is not None


def test_correction_factory_keeps_legacy_modes_available() -> None:
    assert isinstance(
        corrector_from_settings(RuntimeSettings(correction_mode="contextual")),
        ContextualCorrector,
    )
    assert isinstance(
        corrector_from_settings(RuntimeSettings(correction_mode="rules")),
        RuleBasedCorrector,
    )

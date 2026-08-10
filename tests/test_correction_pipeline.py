from pathlib import Path

from assistive_writing_pad.correction.contextual import ContextualCorrector
from assistive_writing_pad.correction.memory import JsonCorrectionMemory


def make_corrector(**kwargs) -> ContextualCorrector:
    defaults = {
        "model_enabled": True,
        "confidence_threshold": 0.65,
        "auto_threshold": 0.90,
    }
    defaults.update(kwargs)
    return ContextualCorrector(**defaults)


def test_correct_word_remains_unchanged() -> None:
    result = make_corrector().correct("ball")
    assert result.corrected_text == "ball"
    assert result.changed is False


def test_simple_spelling_error_blal_to_ball() -> None:
    result = make_corrector().correct("Blal")
    assert result.corrected_text == "Ball"
    assert result.corrections[0].corrected == "Ball"


def test_ocr_confusion_bali_to_ball() -> None:
    result = make_corrector().correct("balI")
    assert result.corrected_text == "ball"


def test_missing_character_bll_to_ball() -> None:
    result = make_corrector().correct("bll")
    assert result.corrected_text == "ball"


def test_extra_character_balll_to_ball() -> None:
    result = make_corrector().correct("balll")
    assert result.corrected_text == "ball"


def test_transposition_taht_to_that() -> None:
    result = make_corrector().correct("taht")
    assert result.corrected_text == "that"


def test_adjacent_transposition_bgi_to_big() -> None:
    result = make_corrector().correct("bgi")
    assert result.corrected_text == "big"
    assert result.corrections[0].reason == "adjacent_transposition"


def test_adjacent_transposition_works_in_sentence_context() -> None:
    result = make_corrector().correct("Blal is bgi")
    assert result.corrected_text == "Ball is big"
    assert result.corrections[1].reason == "adjacent_transposition"


def test_split_word_reconstruction_requires_strong_dictionary_evidence() -> None:
    corrector = make_corrector()
    assert corrector.correct("C hristmas").corrected_text == "Christmas"
    result = corrector.correct("C rhismtas")
    assert result.corrected_text == "Christmas"
    assert result.corrections[0].reason == "token_merge+adjacent_transposition"
    assert corrector.correct("the cat").corrected_text == "the cat"


def test_spaced_fragment_not_merged() -> None:
    result = make_corrector().correct("ba ll")
    assert result.corrected_text == "ba ll"


def test_context_kicked_the_bal_to_ball() -> None:
    result = make_corrector().correct("I kicked the bal")
    assert result.corrected_text == "I kicked the ball"


def test_numbers_unchanged() -> None:
    result = make_corrector().correct("12345")
    assert result.corrected_text == "12345"


def test_numeric_structure_is_preserved() -> None:
    result = make_corrector().correct("Class 5 is at 10:30.")
    assert result.corrected_text == "Class 5 is at 10:30."
    assert result.corrections == ()


def test_ambiguous_glyphs_are_not_blindly_replaced() -> None:
    result = make_corrector().correct("O 0 l I 1")
    assert result.corrected_text == "O 0 l I 1"
    assert result.corrections == ()


def test_punctuation_preserved() -> None:
    result = make_corrector().correct("Blal.")
    assert result.corrected_text == "Ball."


def test_capitalization_preserved() -> None:
    result = make_corrector().correct("Blal")
    assert result.corrected_text == "Ball"


def test_lower_and_uppercase_correction_patterns_are_preserved() -> None:
    corrector = make_corrector()
    assert corrector.correct("blal").corrected_text == "ball"
    assert corrector.correct("BLAL").corrected_text == "BALL"


def test_apostrophes_newlines_and_multiple_sentences_are_preserved() -> None:
    result = make_corrector().correct("Don't stop.\nBlal! Teh cat.")
    assert result.corrected_text == "Don't stop.\nBall! The cat."


def test_empty_and_whitespace_text_are_preserved() -> None:
    corrector = make_corrector()
    assert corrector.correct("").corrected_text == ""
    assert corrector.correct(" \n\t ").corrected_text == " \n\t "


def test_valid_uncommon_word_not_aggressively_replaced() -> None:
    result = make_corrector().correct("quokka")
    assert result.corrected_text == "quokka"


def test_low_confidence_preserves_original() -> None:
    result = make_corrector(confidence_threshold=0.95, auto_threshold=0.99).correct("bqll")
    assert result.corrected_text == "bqll"
    assert result.status == "preserved"


def test_multiple_candidates_are_returned() -> None:
    result = make_corrector().correct("bqll")
    assert len(result.alternatives) >= 1
    assert len(result.alternatives) <= 3


def test_personalized_correction_memory_boosts_candidate(tmp_path: Path) -> None:
    memory = JsonCorrectionMemory(tmp_path / "user_corrections.json")
    corrector = make_corrector(memory=memory)

    baseline = corrector.correct("blorx")
    assert baseline.corrected_text == "blorx"

    corrector.record_feedback("blorx", "ball")
    learned = corrector.correct("blorx")

    assert learned.status in {"suggestion", "automatic"}
    assert any(c.corrected.lower() == "ball" for c in learned.corrections)

from assistive_writing_pad.correction.contextual import ContextualCorrector


def make_corrector() -> ContextualCorrector:
    return ContextualCorrector(model_enabled=False)


def test_contextual_corrector_handles_rule_and_punctuation() -> None:
    result = make_corrector().correct("Teh cat sat on a chaier.")

    assert result.corrected_text == "The cat sat on a chair."
    assert [(item.original, item.corrected) for item in result.corrections] == [
        ("Teh", "The"),
        ("chaier", "chair"),
    ]


def test_contextual_corrector_handles_jumbled_spelling() -> None:
    result = make_corrector().correct("The word is jumlbed.")

    assert result.corrected_text == "The word is jumbled."
    assert result.corrections[0].reason in {"letter_swap", "fuzzy_spelling"}


def test_contextual_corrector_preserves_clean_text() -> None:
    result = make_corrector().correct("The cat sat on the chair.")

    assert result.corrected_text == "The cat sat on the chair."
    assert result.corrections == ()


def test_contextual_corrector_uses_semantic_context_heuristics() -> None:
    result = make_corrector().correct("I went too school.")

    assert result.corrected_text == "I went to school."
    assert result.corrections[0].reason in {"semantic_context", "symspell", "omission"}


def test_contextual_corrector_avoids_unsupported_semantic_change() -> None:
    result = make_corrector().correct("It is too big.")

    assert result.corrected_text == "It is too big."
    assert result.corrections == ()


def test_contextual_corrector_does_not_fuzzy_correct_short_fragments() -> None:
    result = make_corrector().correct("o")

    assert result.corrected_text == "o"
    assert result.corrections == ()


def test_contextual_corrector_preserves_valid_progressive_phrase_words() -> None:
    result = make_corrector().correct("doing well")

    assert result.corrected_text == "doing well"
    assert result.corrections == ()


def test_contextual_corrector_repairs_are_misread_as_the_before_ing_word() -> None:
    result = make_corrector().correct("Hi hope you\nthe doing well")

    assert result.corrected_text == "Hi hope you\nare doing well"
    assert [(item.original, item.corrected, item.reason) for item in result.corrections] == [
        ("the", "are", "semantic_context")
    ]

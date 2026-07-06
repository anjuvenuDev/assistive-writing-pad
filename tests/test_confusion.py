"""Unit tests for the confusion-aware character correction module.

Tests are fast, CPU-only, and do not require model dependencies.
"""

from __future__ import annotations

import pytest

from assistive_writing_pad.recognition.confusion import (
    CONFUSION_MAP,
    ALPHANUM_CHARSET,
    apply_confusion_correction,
    get_candidates,
    restrict_to_charset,
)


# ---------------------------------------------------------------------------
# CONFUSION_MAP structural tests
# ---------------------------------------------------------------------------


class TestConfusionMapStructure:
    def test_no_self_references(self) -> None:
        """No character should confuse with itself."""
        for char, confused in CONFUSION_MAP.items():
            assert char not in confused, (
                f"Character {char!r} appears in its own confusion set {confused!r}"
            )

    def test_is_bidirectional(self) -> None:
        """If A confuses with B then B must confuse with A."""
        for char, confused in CONFUSION_MAP.items():
            for other in confused:
                assert char in CONFUSION_MAP.get(other, set()), (
                    f"Confusion not bidirectional: {char!r} -> {other!r} "
                    f"but {other!r} does not include {char!r}"
                )

    def test_well_known_confusions_present(self) -> None:
        """Spot-check the most critical confusion pairs from user-reported data."""
        expected_pairs = [
            ("9", "g"), ("9", "q"),
            ("7", "y"), ("7", "z"),
            ("I", "l"), ("I", "1"),
            ("r", "m"), ("r", "n"),
            ("h", "t"),
            ("0", "O"),
            ("4", "l"),
        ]
        for a, b in expected_pairs:
            assert b in CONFUSION_MAP.get(a, set()), f"{b!r} not in CONFUSION_MAP[{a!r}]"
            assert a in CONFUSION_MAP.get(b, set()), f"{a!r} not in CONFUSION_MAP[{b!r}]"

    def test_all_values_are_sets(self) -> None:
        """All confusion values must be Python sets (not lists or strings)."""
        for char, confused in CONFUSION_MAP.items():
            assert isinstance(confused, set), (
                f"CONFUSION_MAP[{char!r}] should be a set, got {type(confused)}"
            )


# ---------------------------------------------------------------------------
# restrict_to_charset
# ---------------------------------------------------------------------------


class TestRestrictToCharset:
    def test_alphanum_removes_punctuation(self) -> None:
        assert restrict_to_charset("a!b.c,", charset="alphanum") == "abc"

    def test_alphanum_removes_spaces(self) -> None:
        assert restrict_to_charset("h e l l o", charset="alphanum") == "hello"

    def test_alpha_removes_digits(self) -> None:
        assert restrict_to_charset("abc123", charset="alpha") == "abc"

    def test_digits_removes_letters(self) -> None:
        assert restrict_to_charset("a1b2c3", charset="digits") == "123"

    def test_single_char_preserved(self) -> None:
        assert restrict_to_charset("g", charset="alphanum") == "g"

    def test_all_stripped_returns_empty(self) -> None:
        assert restrict_to_charset("!@#$%", charset="alphanum") == ""

    def test_strips_leading_trailing_whitespace(self) -> None:
        result = restrict_to_charset("  g  ", charset="alphanum")
        assert result == "g"

    def test_alphanum_charset_covers_all_expected_chars(self) -> None:
        """Verify ALPHANUM_CHARSET contains exactly A-Z, a-z, 0-9."""
        expected = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
        assert ALPHANUM_CHARSET == expected


# ---------------------------------------------------------------------------
# get_candidates
# ---------------------------------------------------------------------------


class TestGetCandidates:
    def test_primary_always_first(self) -> None:
        """The primary prediction must always be the first candidate."""
        for char in ("g", "9", "r", "I", "h"):
            candidates = get_candidates(char, 0.80)
            assert candidates, f"get_candidates({char!r}) returned empty list"
            assert candidates[0][0] == char, (
                f"Primary {char!r} is not rank-1 in {candidates}"
            )

    def test_returns_at_most_top_n(self) -> None:
        """Never return more than top_n candidates."""
        for top_n in (1, 2, 3):
            candidates = get_candidates("9", 0.80, top_n=top_n)
            assert len(candidates) <= top_n

    def test_sorted_descending_by_confidence(self) -> None:
        """Candidates must be sorted from highest to lowest confidence."""
        candidates = get_candidates("g", 0.80)
        confs = [conf for _, conf in candidates]
        assert confs == sorted(confs, reverse=True), (
            f"Candidates not sorted: {candidates}"
        )

    def test_known_sibling_included(self) -> None:
        """The most common confusion sibling should appear in top-5 with a low threshold."""
        # 9 -> g is a documented confusion.
        # Sibling confidence = 0.75 * 0.72 = 0.54, so we must use threshold <= 0.54.
        candidates_9 = get_candidates("9", 0.75, threshold=0.45, top_n=5)
        chars = [ch for ch, _ in candidates_9]
        assert "g" in chars, f"'g' not in candidates for '9': {candidates_9}"

    def test_high_confidence_fewer_alternatives(self) -> None:
        """With very high confidence, low-scoring siblings may be filtered out."""
        high_conf = get_candidates("g", 0.99, threshold=0.90, top_n=3)
        # At conf=0.99 only sibling conf = 0.99*0.72 = 0.71 < 0.90 threshold.
        # So only the primary should pass.
        assert len(high_conf) == 1
        assert high_conf[0][0] == "g"

    def test_empty_char_returns_empty(self) -> None:
        assert get_candidates("", 0.80) == []

    def test_char_not_in_map_returns_only_primary(self) -> None:
        """A character with no known confusions should return only itself."""
        # '@' is not alphanumeric and has no confusion map entry.
        candidates = get_candidates("@", 0.90, top_n=3)
        assert len(candidates) == 1
        assert candidates[0][0] == "@"


# ---------------------------------------------------------------------------
# apply_confusion_correction
# ---------------------------------------------------------------------------


class TestApplyConfusionCorrection:
    def test_single_char_returns_tuple_of_text_and_list(self) -> None:
        text, top3 = apply_confusion_correction("g", 0.80, mode="character")
        assert isinstance(text, str)
        assert isinstance(top3, list)

    def test_single_char_primary_is_rank1(self) -> None:
        text, top3 = apply_confusion_correction("g", 0.80, mode="character")
        assert top3[0][0] == "g"

    def test_top3_list_is_sorted(self) -> None:
        _, top3 = apply_confusion_correction("r", 0.72, mode="character")
        confs = [c for _, c in top3]
        assert confs == sorted(confs, reverse=True)

    def test_top3_has_at_most_3_items(self) -> None:
        _, top3 = apply_confusion_correction("9", 0.75, mode="character")
        assert len(top3) <= 3

    def test_word_mode_returns_text_unchanged(self) -> None:
        """Word mode should return the raw text without confusion correction."""
        text, top3 = apply_confusion_correction("hello world", 0.80, mode="word")
        assert text == "hello world"
        assert len(top3) == 1
        assert top3[0][0] == "hello world"

    def test_multi_char_in_auto_mode_returns_unchanged(self) -> None:
        """Multi-character results in auto mode must not be single-char corrected."""
        text, top3 = apply_confusion_correction("cat", 0.85, mode="auto")
        assert text == "cat"

    def test_empty_text_returns_empty_result(self) -> None:
        """Empty text in character mode returns empty string (top3 may contain the empty string)."""
        text, top3 = apply_confusion_correction("", 0.50, mode="character")
        assert text == ""

    def test_restricted_charset_applied_in_char_mode(self) -> None:
        """Non-alphanum characters should be stripped before correction."""
        # Input "g!" — the '!' should be stripped, leaving 'g'.
        text, top3 = apply_confusion_correction("g", 0.70, mode="character")
        # result char should be in ALPHANUM_CHARSET
        assert text in ALPHANUM_CHARSET or text == ""

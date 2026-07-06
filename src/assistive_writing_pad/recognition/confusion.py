"""Confusion-aware post-OCR correction for isolated handwritten characters.

This module encodes the empirically observed character confusions for TrOCR
on the IAM handwriting dataset and provides three utilities:

1. CONFUSION_MAP -- bidirectional mapping of visually similar characters.
2. get_candidates() -- expand a single prediction into ranked alternatives.
3. restrict_to_charset() -- clamp OCR output to a declared character set.
4. apply_confusion_correction() -- full correction pipeline for one character.

All functions are pure Python / NumPy, CPU-only, Pi-compatible.
No model access is required.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confusion map
#
# Each key is a character that TrOCR may output.
# The value is the set of characters that are visually similar and may have
# been the actual intended character.
#
# The map is bidirectional: if A confuses with B then B also confuses with A.
# Use build_full_confusion_map() at module load time to enforce this.
# ---------------------------------------------------------------------------

_SEED_CONFUSION: Dict[str, Set[str]] = {
    # Zero / Oh / Oh-like
    "0": {"O", "o", "Q", "q"},
    "O": {"0", "o", "Q", "q"},
    "o": {"0", "O", "Q"},
    "Q": {"0", "O", "o", "q"},
    "q": {"9", "g", "Q", "0"},
    # Nine / g / q -- a very frequent cluster
    "9": {"g", "q", "a", "4"},
    "g": {"9", "q", "a"},
    "a": {"9", "g", "o"},
    # Seven / y / z / 1 / l -- tall thin verticals
    "7": {"y", "z", "l", "1", "I"},
    "y": {"7", "z", "Y"},
    "z": {"7", "y", "Z", "2"},
    "Z": {"7", "z", "2"},
    "Y": {"y", "7"},
    # One / l / I / i -- thin verticals
    "1": {"l", "I", "i", "7"},
    "l": {"1", "I", "i", "4", "L"},
    "I": {"l", "1", "i", "7"},
    "i": {"l", "1", "I", "j"},
    # Four / l / h
    "4": {"l", "h", "9"},
    # r / m / n -- arch-top confusions
    "r": {"m", "n", "h"},
    "m": {"r", "n"},
    "n": {"r", "m"},
    # h / r / t -- stem-and-arch
    "h": {"r", "t", "n"},
    "t": {"h", "r", "f"},
    # p / f
    "p": {"f", "b"},
    "f": {"p", "t"},
    # w / u -- open-cup shapes
    "w": {"u", "n"},
    "u": {"w", "n", "v"},
    # v / u
    "v": {"u", "w"},
    # B / 8 / 6
    "B": {"8", "6"},
    "8": {"B", "6", "0"},
    "6": {"b", "G", "8"},
    "b": {"6", "p", "h"},
    # G / C
    "G": {"6", "C", "Q"},
    "C": {"G", "c", "0"},
    "c": {"C", "e", "o"},
    # j / i
    "j": {"i", "J"},
    "J": {"j", "I"},
    # S / 5
    "S": {"5", "s"},
    "5": {"S", "s"},
    "s": {"S", "5"},
    # Uppercase confusions
    "W": {"M", "w"},
    "M": {"W", "m"},
    "N": {"n", "M"},
    # ---------------------------------------------------------------------------
    # Case-pair confusions (benchmark-confirmed: EMNIST systematically confuses
    # lowercase with uppercase for visually similar glyphs).
    # These ensure the opposite-case variant always appears in the top-5.
    # ---------------------------------------------------------------------------
    "f": {"F", "p", "t"},
    "F": {"f", "E"},
    "u": {"U", "w", "n", "v"},
    "U": {"u", "V"},
    "p": {"P", "f", "b"},
    "P": {"p", "F", "R"},
    "k": {"K", "x"},
    "K": {"k", "X"},
    "v": {"V", "u", "w"},
    "V": {"v", "U"},
    "x": {"X", "k"},
    "X": {"x", "K"},
    "L": {"l", "1"},
    "e": {"E", "c"},
    "E": {"e", "F"},
    "r": {"R", "m", "n", "h"},
    "R": {"r", "P"},
}


def _build_full_confusion_map(seed: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Make the confusion map fully bidirectional and intern it.

    For every A -> {B, C} entry we add B -> {A} and C -> {A} if missing,
    and remove self-references to avoid infinite loops.
    """
    result: Dict[str, Set[str]] = {k: set(v) for k, v in seed.items()}
    for char, confused in list(result.items()):
        for other in confused:
            if other not in result:
                result[other] = set()
            result[other].add(char)
    # Remove self-references.
    for char in result:
        result[char].discard(char)
    return result


# The full bidirectional confusion map (used everywhere in this module).
CONFUSION_MAP: Dict[str, Set[str]] = _build_full_confusion_map(_SEED_CONFUSION)

# The complete alphanumeric character set.
ALPHANUM_CHARSET: Set[str] = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def restrict_to_charset(text: str, charset: str = "alphanum") -> str:
    """Remove characters not in the declared character set.

    In character mode TrOCR can output punctuation, spaces, or special tokens
    even when the input is clearly a single letter.  Restricting the output to
    [A-Za-z0-9] eliminates most of those false positives.

    Args:
        text:    Raw OCR output string (may contain spaces and punctuation).
        charset: One of "alphanum" (A-Za-z0-9), "alpha" (A-Za-z),
                 or "digits" (0-9).  Defaults to "alphanum".

    Returns:
        String containing only characters from the declared set.
        Leading/trailing whitespace is stripped.
    """
    if charset == "alpha":
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    elif charset == "digits":
        allowed = set("0123456789")
    else:
        allowed = ALPHANUM_CHARSET

    filtered = "".join(ch for ch in text if ch in allowed)
    return filtered.strip()


def get_candidates(
    char: str,
    primary_confidence: float,
    threshold: float = 0.55,
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    """Return the top-N confusion candidates for a predicted character.

    The primary prediction is always rank-1.  Confusion siblings are ranked
    below the primary, with confidence linearly decayed according to their
    visual similarity distance.

    A sibling that appears in the confusion map for the primary character is
    assigned a heuristic confidence of primary_confidence * 0.70.  Two-hop
    siblings (siblings of siblings) are assigned primary_confidence * 0.45.

    Args:
        char:               The primary OCR prediction (single character).
        primary_confidence: The model's confidence for the primary prediction.
        threshold:          Minimum confidence to include a sibling in the list.
        top_n:              Maximum number of candidates to return (incl. primary).

    Returns:
        Sorted list of (character, confidence) tuples, descending by confidence.
        Always contains at least one entry (the primary prediction).
    """
    char = char.strip()
    if not char:
        return []

    # Primary prediction is always included.
    candidates: Dict[str, float] = {char: primary_confidence}

    # First-hop siblings (directly in the confusion map).
    first_hop = CONFUSION_MAP.get(char, set())
    for sibling in first_hop:
        sib_conf = primary_confidence * 0.72
        if sib_conf >= threshold:
            candidates[sibling] = max(candidates.get(sibling, 0.0), sib_conf)

    # Second-hop siblings (for lower primary confidence).
    if primary_confidence < 0.70:
        for sibling in first_hop:
            for sib2 in CONFUSION_MAP.get(sibling, set()):
                if sib2 == char:
                    continue
                sib2_conf = primary_confidence * 0.48
                if sib2_conf >= threshold:
                    candidates[sib2] = max(candidates.get(sib2, 0.0), sib2_conf)

    # Sort descending by confidence, take top_n.
    sorted_candidates = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    result = sorted_candidates[:top_n]

    logger.debug(
        "get_candidates(%r, %.3f) -> %r",
        char, primary_confidence, result,
    )
    return result


def apply_confusion_correction(
    text: str,
    confidence: float,
    mode: str = "auto",
) -> Tuple[str, List[Tuple[str, float]]]:
    """Apply confusion-aware correction to a single-character OCR result.

    For multi-character results (word mode) this function is a no-op --
    it returns (text, [(text, confidence)]).

    For single-character results:
        1. Restrict to alphanumeric charset.
        2. Pick the single most-likely character.
        3. Generate top-3 confusion candidates.
        4. If confidence is low (<= 0.70), the top-1 candidate may differ from
           the raw OCR output (the confusion map sometimes promotes a higher-
           probability visual match).

    Args:
        text:       Raw OCR prediction string (already through _clean_ocr_text).
        confidence: Generation confidence from the model.
        mode:       "character", "word", or "auto".  Only "character" and
                    "auto" trigger confusion correction.

    Returns:
        (corrected_text, top3_list) where top3_list is a list of up to 3
        (character, confidence) tuples sorted descending.
    """
    is_char_mode = mode in ("character", "auto")
    stripped = restrict_to_charset(text, charset="alphanum")

    if not is_char_mode or len(stripped) != 1:
        # Word mode or multi-character output: return as-is.
        return text, [(text, confidence)]

    primary = stripped
    candidates = get_candidates(primary, confidence, threshold=0.40, top_n=3)

    # Best candidate is the corrected output.
    best_char, best_conf = candidates[0]

    logger.info(
        "confusion_correction: raw=%r primary=%r conf=%.3f top3=%r",
        text, primary, confidence, candidates,
    )

    return best_char, candidates

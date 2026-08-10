"""Frequency-aware spelling candidates using SymSpell where available."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, List, Optional

from importlib import resources

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?$", re.ASCII)


@dataclass(frozen=True)
class SpellCandidate:
    word: str
    distance: int
    frequency: int
    reason: str = "symspell"


class SpellCorrector:
    def __init__(self, dictionary_path: Optional[str] = None, max_dictionary_edit_distance: int = 2) -> None:
        self._max_dictionary_edit_distance = max_dictionary_edit_distance
        self._symspell = None
        self._fallback_freq: Dict[str, int] = {}
        self._load(dictionary_path)

    def _load(self, dictionary_path: Optional[str]) -> None:
        path = dictionary_path
        if not path:
            try:
                path = str(
                    resources.files("assistive_writing_pad.correction.resources")
                    .joinpath("word_frequency_en_small.txt")
                )
            except Exception:
                path = None

        # Keep a small local frequency map even when SymSpell is installed.
        # It enables explicit bounded Damerau matching for OCR transpositions.
        if path:
            self._fallback_freq = self._read_frequency_file(path)

        try:
            from symspellpy import SymSpell

            symspell = SymSpell(max_dictionary_edit_distance=self._max_dictionary_edit_distance)
            if path and symspell.load_dictionary(path, term_index=0, count_index=1):
                self._symspell = symspell
                return
        except Exception:
            self._symspell = None

    def lookup_damerau(
        self, token: str, max_distance: int, limit: int = 6
    ) -> List[SpellCandidate]:
        """Return dictionary candidates within a bounded Damerau distance.

        The bundled dictionary is intentionally small, so an exhaustive local
        scan is deterministic and inexpensive while making adjacent swaps
        (``bgi`` -> ``big``) explicit rather than incidental.
        """
        lowered = token.lower()
        if not WORD_RE.match(token):
            return []
        candidates = [
            SpellCandidate(
                word=word,
                distance=distance,
                frequency=frequency,
                reason=(
                    "adjacent_transposition"
                    if _levenshtein(lowered, word) > distance
                    else "damerau_edit"
                ),
            )
            for word, frequency in self._fallback_freq.items()
            if (distance := _damerau_levenshtein(lowered, word, max_distance)) <= max_distance
        ]
        candidates.sort(key=lambda item: (item.distance, -item.frequency, item.word))
        return candidates[:limit]

    def _read_frequency_file(self, path: str) -> Dict[str, int]:
        words: Dict[str, int] = {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    token = parts[0].lower()
                    try:
                        freq = int(parts[1])
                    except ValueError:
                        continue
                    words[token] = max(words.get(token, 0), freq)
        except OSError:
            return {}
        return words

    def is_known_word(self, token: str) -> bool:
        lowered = token.lower()
        if self._symspell is not None:
            return bool(self._symspell.word_frequency.lookup(lowered, 0))
        return lowered in self._fallback_freq

    def lookup(self, token: str, max_edit_distance: int, limit: int = 6) -> List[SpellCandidate]:
        lowered = token.lower()
        if not WORD_RE.match(token):
            return []

        if self._symspell is not None:
            from symspellpy import Verbosity

            suggestions = self._symspell.lookup(
                lowered,
                Verbosity.CLOSEST,
                max_edit_distance=max_edit_distance,
                include_unknown=False,
                transfer_casing=False,
            )
            results = [
                SpellCandidate(word=item.term, distance=item.distance, frequency=int(item.count))
                for item in suggestions
                if item.term
            ]
            return results[:limit]

        # Fallback: tiny frequency dictionary nearest neighbors.
        candidates: List[SpellCandidate] = []
        for candidate, freq in self._fallback_freq.items():
            distance = _levenshtein(lowered, candidate)
            if distance <= max_edit_distance:
                candidates.append(
                    SpellCandidate(word=candidate, distance=distance, frequency=freq, reason="fallback")
                )
        candidates.sort(key=lambda item: (item.distance, -item.frequency, item.word))
        return candidates[:limit]

    def frequency_score(self, word: str) -> float:
        lowered = word.lower()
        freq = 0
        if self._symspell is not None:
            found = self._symspell.word_frequency.lookup(lowered, 0)
            freq = int(found) if found else 0
        else:
            freq = int(self._fallback_freq.get(lowered, 0))

        if freq <= 0:
            return 0.0
        return min(1.0, math.log10(freq + 10) / 7.0)


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        curr = [i]
        for j, rch in enumerate(right, start=1):
            cost = 0 if lch == rch else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _damerau_levenshtein(left: str, right: str, maximum: int) -> int:
    """Optimal-string-alignment Damerau distance with an early length bound."""
    if abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous_previous = None
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_minimum = current[0]
        for right_index, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + cost)
            if (
                previous_previous is not None
                and left_index > 1
                and right_index > 1
                and left_char == right[right_index - 2]
                and left[left_index - 2] == right_char
            ):
                value = min(value, previous_previous[right_index - 2] + 1)
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > maximum:
            return maximum + 1
        previous_previous, previous = previous, current
    return previous[-1]

"""Lightweight offline local-context scoring."""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?$", re.ASCII)

_SEMANTIC_GROUPS = (
    ("there", "their", "they're"),
    ("to", "too", "two"),
    ("right", "write", "rite"),
    ("hear", "here"),
    ("no", "know"),
    ("are", "the"),
)

# Hand-tuned bigram priors for common classroom sentences.
_BIGRAM_PRIORS = {
    ("the", "cat"): 0.98,
    ("the", "ball"): 0.96,
    ("a", "ball"): 0.94,
    ("quick", "brown"): 0.97,
    ("brown", "fox"): 0.97,
    ("i", "have"): 0.96,
    ("have", "a"): 0.96,
    ("kicked", "the"): 0.95,
    ("the", "school"): 0.90,
    ("to", "school"): 0.96,
    ("is", "black"): 0.95,
    ("is", "big"): 0.95,
    ("is", "at"): 0.95,
}


def nearest_word(tokens: Sequence[str], start_index: int, step: int) -> Optional[str]:
    index = start_index + step
    while 0 <= index < len(tokens):
        token = tokens[index]
        if WORD_RE.match(token):
            return token.lower()
        if token.strip() and not WORD_RE.match(token):
            return None
        index += step
    return None


class LocalContextScorer:
    """Context scorer that works fully offline and on CPU."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def score(self, tokens: Sequence[str], token_index: int, candidate: str) -> float:
        if not self.enabled:
            return 0.5

        previous_word = nearest_word(tokens, token_index, -1)
        next_word = nearest_word(tokens, token_index, 1)
        cand = candidate.lower()

        score = 0.5

        if previous_word:
            score = max(score, _BIGRAM_PRIORS.get((previous_word, cand), 0.0))
        if next_word:
            score = max(score, _BIGRAM_PRIORS.get((cand, next_word), 0.0))

        # Semantic confusion disambiguation.
        if cand in {"to", "too", "two"}:
            if next_word in {"school", "class", "home", "the", "a", "an"}:
                score = max(score, 0.95 if cand == "to" else 0.30)
            if next_word in {"big", "small", "many", "much"}:
                score = max(score, 0.93 if cand == "too" else 0.30)

        if cand in {"there", "their", "they're"}:
            if next_word in {"book", "class", "teacher", "home"}:
                score = max(score, 0.93 if cand == "their" else 0.35)
            if next_word in {"is", "are", "was", "were"}:
                score = max(score, 0.92 if cand == "there" else 0.35)

        if cand in {"are", "the"}:
            if previous_word in {"you", "we", "they"} and (next_word or "").endswith("ing"):
                score = max(score, 0.95 if cand == "are" else 0.25)

        # Slight preference for keeping an existing semantic group member unless context says otherwise.
        for group in _SEMANTIC_GROUPS:
            if cand in group:
                score = max(score, 0.56)
                break

        return min(max(score, 0.0), 0.99)

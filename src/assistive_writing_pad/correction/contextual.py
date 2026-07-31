"""Realtime spelling and contextual correction for dysgraphia support."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import Correction, CorrectionResult
from assistive_writing_pad.correction.rule_based import DEFAULT_REPLACEMENTS

logger = logging.getLogger(__name__)

try:  # rapidfuzz is a normal project dependency, but keep import failure non-fatal.
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - only used in incomplete environments.
    fuzz = None
    process = None


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|\s+|[^\w\s]", re.ASCII)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?$", re.ASCII)

DEFAULT_CONTEXT_MODEL = "distilbert/distilbert-base-uncased"

COMMON_WORDS: Set[str] = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "around",
    "as",
    "ask",
    "at",
    "away",
    "back",
    "be",
    "because",
    "been",
    "before",
    "big",
    "book",
    "boy",
    "but",
    "by",
    "came",
    "can",
    "cat",
    "chair",
    "child",
    "children",
    "class",
    "come",
    "could",
    "day",
    "did",
    "do",
    "does",
    "dog",
    "down",
    "eat",
    "for",
    "friend",
    "from",
    "gave",
    "get",
    "girl",
    "go",
    "good",
    "got",
    "had",
    "has",
    "have",
    "he",
    "hear",
    "help",
    "her",
    "here",
    "him",
    "his",
    "home",
    "house",
    "i",
    "in",
    "is",
    "it",
    "jumped",
    "jumbled",
    "know",
    "learn",
    "left",
    "letter",
    "like",
    "little",
    "look",
    "made",
    "make",
    "many",
    "me",
    "my",
    "new",
    "no",
    "not",
    "now",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "over",
    "page",
    "paper",
    "pen",
    "phone",
    "play",
    "put",
    "read",
    "receive",
    "right",
    "room",
    "said",
    "sat",
    "saw",
    "school",
    "see",
    "sentence",
    "she",
    "small",
    "so",
    "some",
    "spelling",
    "story",
    "teacher",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "thing",
    "this",
    "time",
    "to",
    "too",
    "two",
    "up",
    "use",
    "very",
    "want",
    "was",
    "we",
    "went",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "word",
    "work",
    "write",
    "writing",
    "you",
    "your",
}

DYSLEXIA_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "b": ("d", "p"),
    "d": ("b",),
    "p": ("q", "b"),
    "q": ("p",),
    "m": ("w",),
    "w": ("m",),
    "n": ("u",),
    "u": ("n",),
}

SEMANTIC_CONFUSIONS: Tuple[Tuple[str, ...], ...] = (
    ("there", "their", "they're"),
    ("to", "too", "two"),
    ("right", "write", "rite"),
    ("hear", "here"),
    ("no", "know"),
    ("was", "saw"),
)


@dataclass(frozen=True)
class Candidate:
    text: str
    confidence: float
    reason: str


@dataclass
class ContextualCorrector:
    """Correct spelling and context errors without blocking Pi migration."""

    vocabulary: Set[str] = field(default_factory=lambda: set(COMMON_WORDS))
    replacements: Mapping[str, Tuple[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_REPLACEMENTS)
    )
    semantic_confusions: Sequence[Tuple[str, ...]] = SEMANTIC_CONFUSIONS
    model_enabled: bool = True
    model_name: str = DEFAULT_CONTEXT_MODEL
    max_candidates: int = 8
    confidence_threshold: float = 0.70
    _context_scorer: Optional["_MaskedLanguageContextScorer"] = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def from_settings(cls, settings: RuntimeSettings) -> "ContextualCorrector":
        return cls(
            model_enabled=(
                settings.contextual_model_enabled and settings.correction_mode == "contextual"
            ),
            model_name=settings.contextual_model_name,
            max_candidates=settings.max_correction_candidates,
            confidence_threshold=settings.correction_confidence_threshold,
        )

    def correct(self, text: str) -> CorrectionResult:
        tokens = tokenize_preserving_layout(text)
        if not tokens:
            return CorrectionResult(original_text=text, corrected_text=text, confidence=1.0)

        corrected_tokens = list(tokens)
        corrections: List[Correction] = []

        for index, token in enumerate(tokens):
            if not WORD_RE.match(token):
                continue
            candidates = self._candidates_for(token)
            ranked = self._rank_candidates(tokens, index, candidates)
            if not ranked:
                continue
            best = ranked[0]
            if best.text.lower() == token.lower() or best.confidence < self.confidence_threshold:
                continue

            corrected = preserve_case(token, best.text)
            corrected_tokens[index] = corrected
            corrections.append(
                Correction(
                    original=token,
                    corrected=corrected,
                    confidence=round(best.confidence, 4),
                    reason=best.reason,
                )
            )

        corrected_text = "".join(corrected_tokens)
        confidence = min((item.confidence for item in corrections), default=1.0)
        return CorrectionResult(
            original_text=text,
            corrected_text=corrected_text,
            corrections=tuple(corrections),
            confidence=confidence,
        )

    def _candidates_for(self, token: str) -> List[Candidate]:
        lowered = token.lower()
        candidates: Dict[str, Candidate] = {}

        replacement = self.replacements.get(lowered)
        if replacement is not None:
            corrected, reason = replacement
            candidates[corrected] = Candidate(corrected, 0.98, reason)

        for variant in self._semantic_variants(lowered):
            if variant != lowered:
                candidates[variant] = Candidate(variant, 0.78, "semantic_context")

        if lowered in self.vocabulary:
            return sorted(candidates.values(), key=lambda item: item.confidence, reverse=True)

        if len(lowered) < 3:
            return sorted(candidates.values(), key=lambda item: item.confidence, reverse=True)

        for candidate, score, reason in self._fuzzy_candidates(lowered):
            current = candidates.get(candidate)
            if current is None or score > current.confidence:
                candidates[candidate] = Candidate(candidate, score, reason)

        return sorted(candidates.values(), key=lambda item: item.confidence, reverse=True)[
            : self.max_candidates
        ]

    def _semantic_variants(self, word: str) -> Tuple[str, ...]:
        for group in self.semantic_confusions:
            if word in group:
                return group
        return ()

    def _fuzzy_candidates(self, word: str) -> Iterable[Tuple[str, float, str]]:
        known_variants = dyslexia_variants(word)
        for variant in known_variants:
            if variant in self.vocabulary:
                yield variant, 0.92, spelling_reason(word, variant)

        if process is not None and fuzz is not None:
            matches = process.extract(
                word,
                self.vocabulary,
                scorer=fuzz.WRatio,
                limit=max(10, self.max_candidates * 2),
            )
            for candidate, raw_score, _ in matches:
                if candidate == word:
                    continue
                score = raw_score / 100.0
                if score >= 0.78:
                    yield candidate, score, spelling_reason(word, candidate)
            return

        for candidate in self.vocabulary:  # pragma: no cover - rapidfuzz fallback.
            score = simple_similarity(word, candidate)
            if score >= 0.78:
                yield candidate, score, spelling_reason(word, candidate)

    def _rank_candidates(
        self,
        tokens: Sequence[str],
        token_index: int,
        candidates: Sequence[Candidate],
    ) -> List[Candidate]:
        if not candidates:
            return []

        scored = list(candidates)
        context_scores = self._context_scores(tokens, token_index, [item.text for item in scored])
        rescored = []
        for candidate in scored:
            context_score = context_scores.get(candidate.text.lower())
            if context_score is None:
                if candidate.reason == "semantic_context":
                    rescored.append(Candidate(candidate.text, 0.60, candidate.reason))
                else:
                    rescored.append(candidate)
                continue
            combined = (candidate.confidence * 0.72) + (context_score * 0.28)
            reason = (
                "semantic_context"
                if candidate.reason == "semantic_context"
                else f"{candidate.reason}+context"
            )
            rescored.append(Candidate(candidate.text, min(combined, 0.995), reason))
        scored = rescored

        return sorted(scored, key=lambda item: item.confidence, reverse=True)

    def _context_scores(
        self,
        tokens: Sequence[str],
        token_index: int,
        candidate_words: Sequence[str],
    ) -> Dict[str, float]:
        if not candidate_words:
            return {}

        scores = heuristic_context_scores(tokens, token_index, candidate_words)
        if not self.model_enabled:
            return scores

        scorer = self._get_context_scorer()
        if scorer is None:
            return scores

        model_scores = scorer.score_candidates(tokens, token_index, candidate_words)
        return merge_context_scores(scores, model_scores)

    def _get_context_scorer(self) -> Optional["_MaskedLanguageContextScorer"]:
        if self._context_scorer is not None:
            return self._context_scorer
        try:
            self._context_scorer = _MaskedLanguageContextScorer(self.model_name)
        except Exception as exc:  # pragma: no cover - depends on local model env/cache.
            logger.warning("contextual correction model unavailable: %s", exc)
            self.model_enabled = False
            return None
        return self._context_scorer


class _MaskedLanguageContextScorer:
    """Small wrapper around a pretrained fill-mask model."""

    def __init__(self, model_name: str) -> None:
        from transformers import pipeline

        self._pipeline = pipeline("fill-mask", model=model_name, tokenizer=model_name)
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        self._mask_token = getattr(tokenizer, "mask_token", "[MASK]")

    def score_candidates(
        self,
        tokens: Sequence[str],
        token_index: int,
        candidate_words: Sequence[str],
    ) -> Dict[str, float]:
        masked_tokens = list(tokens)
        masked_tokens[token_index] = self._mask_token
        masked_text = "".join(masked_tokens)
        targets = [word.lower() for word in candidate_words if WORD_RE.match(word)]
        if not targets:
            return {}

        try:
            results = self._pipeline(masked_text, targets=targets)
        except Exception as exc:  # pragma: no cover - model/tokenizer edge cases.
            logger.warning("contextual correction scoring failed: %s", exc)
            return {}

        if isinstance(results, dict):
            results = [results]

        scores: Dict[str, float] = {}
        for item in results:
            token = str(item.get("token_str", "")).strip().lower()
            if token:
                scores[token] = float(item.get("score", 0.0))
        return scores


def heuristic_context_scores(
    tokens: Sequence[str],
    token_index: int,
    candidate_words: Sequence[str],
) -> Dict[str, float]:
    current = tokens[token_index].lower()
    previous_word = nearest_word(tokens, token_index, -1)
    next_word = nearest_word(tokens, token_index, 1)
    candidates = {word.lower() for word in candidate_words}
    scores: Dict[str, float] = {}

    if {"to", "too", "two"} & candidates:
        if next_word in {"go", "school", "work", "home", "the", "a", "an", "class"}:
            scores["to"] = 0.94
        if previous_word in {"one", "three", "four"} or next_word in {"cats", "dogs", "books"}:
            scores["two"] = 0.92
        if next_word in {"big", "small", "good", "many", "much"}:
            scores["too"] = 0.92

    if {"there", "their", "they're"} & candidates:
        if next_word in {"book", "chair", "class", "home", "house", "paper", "pen", "teacher"}:
            scores["their"] = 0.92
        if next_word in {"is", "are", "was", "were"} or previous_word in {"over", "in"}:
            scores["there"] = 0.92

    if {"hear", "here"} & candidates:
        if previous_word in {"can", "could", "will"}:
            scores["hear"] = 0.92
        if previous_word in {"is", "was"} or next_word in {"is", "are"}:
            scores["here"] = 0.90

    if {"right", "write", "rite"} & candidates:
        if next_word in {"word", "sentence", "story", "letter"}:
            scores["write"] = 0.92
        if previous_word in {"the", "my", "your"} or next_word in {"answer", "side"}:
            scores["right"] = 0.90

    if current in scores:
        scores[current] = max(scores[current], 0.96)
    return scores


def merge_context_scores(left: Mapping[str, float], right: Mapping[str, float]) -> Dict[str, float]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = max(value, merged.get(key, 0.0))
    return merged


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


def tokenize_preserving_layout(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def preserve_case(original: str, corrected: str) -> str:
    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected.capitalize()
    return corrected


def dyslexia_variants(word: str) -> Set[str]:
    variants: Set[str] = set()

    for index in range(len(word) - 1):
        variants.add(word[:index] + word[index + 1] + word[index] + word[index + 2 :])

    for index, char in enumerate(word):
        for replacement in DYSLEXIA_VARIANTS.get(char, ()):
            variants.add(word[:index] + replacement + word[index + 1 :])

    for index in range(len(word) - 1):
        if word[index] == word[index + 1]:
            variants.add(word[:index] + word[index + 1 :])

    for index in range(len(word)):
        variants.add(word[:index] + word[index + 1 :])

    return {variant for variant in variants if variant}


def spelling_reason(original: str, candidate: str) -> str:
    if has_adjacent_swap(original, candidate):
        return "letter_swap"
    if abs(len(original) - len(candidate)) == 1:
        return "omission_or_insertion"
    if removes_doubling(original) == removes_doubling(candidate):
        return "doubling_error"
    if visual_confusion_distance(original, candidate) == 1:
        return "visual_or_reversal_error"
    return "fuzzy_spelling"


def has_adjacent_swap(original: str, candidate: str) -> bool:
    if len(original) != len(candidate):
        return False
    for index in range(len(original) - 1):
        swapped = original[:index] + original[index + 1] + original[index] + original[index + 2 :]
        if swapped == candidate:
            return True
    return False


def removes_doubling(word: str) -> str:
    if not word:
        return word
    chars = [word[0]]
    for char in word[1:]:
        if char != chars[-1]:
            chars.append(char)
    return "".join(chars)


def visual_confusion_distance(original: str, candidate: str) -> float:
    if len(original) != len(candidate):
        return math.inf
    distance = 0
    for left, right in zip(original, candidate):
        if left == right:
            continue
        if right in DYSLEXIA_VARIANTS.get(left, ()):
            distance += 1
        else:
            return math.inf
    return distance


def simple_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    common = sum(1 for char in left if char in right)
    return common / max(len(left), len(right))

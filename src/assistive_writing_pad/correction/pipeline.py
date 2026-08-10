"""Post-OCR intelligent correction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import Correction, CorrectionAlternative, CorrectionResult
from assistive_writing_pad.correction.context_corrector import LocalContextScorer
from assistive_writing_pad.correction.memory import CorrectionMemory, JsonCorrectionMemory
from assistive_writing_pad.correction.rule_based import DEFAULT_REPLACEMENTS
from assistive_writing_pad.correction.spell_corrector import SpellCandidate, SpellCorrector

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|\s+|[^\w\s]", re.ASCII)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?$", re.ASCII)
DIGIT_RE = re.compile(r"\d", re.ASCII)

# A compact guard for common classroom vocabulary omitted from the deliberately
# small frequency file.  These are known words, not correction targets.
_COMMON_SAFE_WORDS = frozenset({"sat", "stop"})

_CONFUSION_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("0", "o", "O"),
    ("1", "l", "I"),
    ("2", "z", "Z"),
    ("4", "a", "A"),
    ("5", "s", "S"),
    ("7", "t", "T"),
    ("9", "g", "q"),
    ("m", "M"),
    ("c", "C"),
    ("u", "U"),
    ("f", "F"),
    ("s", "S"),
    ("p", "P"),
    ("v", "V"),
    ("y", "Y"),
    ("k", "K"),
    ("z", "Z"),
    ("x", "X"),
)

_SEMANTIC_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("there", "their", "they're"),
    ("to", "too", "two"),
    ("right", "write", "rite"),
    ("hear", "here"),
    ("no", "know"),
    ("are", "the"),
)

_CONFUSION_MAP: Dict[str, Set[str]] = {}
for group in _CONFUSION_GROUPS:
    for token in group:
        values = _CONFUSION_MAP.setdefault(token, set())
        values.update({item for item in group if item != token})
        lowered = token.lower()
        values_lower = _CONFUSION_MAP.setdefault(lowered, set())
        values_lower.update({item.lower() for item in group if item.lower() != lowered})


@dataclass(frozen=True)
class _RankedCandidate:
    word: str
    confidence: float
    reason: str
    edit_distance: int
    automatic: bool


@dataclass
class IntelligentCorrectionPipeline:
    spell_corrector: SpellCorrector
    context_scorer: LocalContextScorer
    auto_threshold: float = 0.90
    suggestion_threshold: float = 0.65
    max_candidates: int = 8
    max_transpositions: int = 2
    replacements: Dict[str, Tuple[str, str]] = field(default_factory=lambda: dict(DEFAULT_REPLACEMENTS))
    memory: Optional[CorrectionMemory] = None
    debug: bool = False

    @classmethod
    def from_settings(cls, settings: RuntimeSettings) -> "IntelligentCorrectionPipeline":
        memory: Optional[CorrectionMemory] = JsonCorrectionMemory(settings.correction_memory_path)
        return cls(
            spell_corrector=SpellCorrector(dictionary_path=settings.correction_dictionary_path),
            context_scorer=LocalContextScorer(enabled=settings.correction_mode == "contextual"),
            auto_threshold=settings.correction_auto_threshold,
            suggestion_threshold=settings.correction_suggestion_threshold,
            max_candidates=settings.max_correction_candidates,
            max_transpositions=settings.correction_max_transpositions,
            memory=memory,
            debug=settings.debug_correction,
        )

    def correct(self, text: str) -> CorrectionResult:
        tokens = tokenize_preserving_layout(text)
        if not tokens:
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                confidence=1.0,
                method="symspell+ocr+context",
                status="preserved",
            )

        corrected_tokens = list(tokens)
        corrections: List[Correction] = []
        text_alternatives: List[CorrectionAlternative] = []
        reconstructed_indexes = self._reconstruct_split_words(tokens, corrected_tokens, corrections)

        for index, token in enumerate(tokens):
            if index in reconstructed_indexes:
                continue
            if not WORD_RE.match(token):
                continue
            if DIGIT_RE.search(token):
                continue

            ranked = self._rank_token_candidates(tokens, index, token)
            if not ranked:
                continue

            best = ranked[0]
            alternatives = tuple(
                CorrectionAlternative(text=preserve_case(token, item.word), confidence=item.confidence, reason=item.reason)
                for item in ranked[:3]
                if item.word.lower() != token.lower()
            )
            if not alternatives:
                continue

            if self.debug or os.environ.get("AWP_DEBUG_CORRECTION", "0").strip() == "1":
                logger.info("OCR token=%r candidates=%s selected=%r conf=%.3f reason=%s", token, [
                    (item.word, round(item.confidence, 3), item.edit_distance, item.reason) for item in ranked[:5]
                ], best.word, best.confidence, best.reason)

            corrected = preserve_case(token, best.word)
            status = "automatic" if best.automatic else "suggestion"

            if best.automatic and best.word.lower() != token.lower():
                corrected_tokens[index] = corrected
            else:
                corrected = token

            corrections.append(
                Correction(
                    original=token,
                    corrected=preserve_case(token, best.word),
                    confidence=round(best.confidence, 4),
                    reason=best.reason,
                    alternatives=alternatives,
                    edit_distance=best.edit_distance,
                    automatic=best.automatic,
                    status=status,
                )
            )

            if status == "suggestion":
                text_alternatives.extend(alternatives)

        corrected_text = "".join(corrected_tokens)
        changed = corrected_text != text

        if corrections:
            confidence = min(item.confidence for item in corrections)
            if changed:
                status = "automatic"
            elif any(item.status == "suggestion" for item in corrections):
                status = "suggestion"
            else:
                status = "preserved"
        else:
            confidence = 1.0
            status = "preserved"

        # Keep top-3 unique alternatives across the full sentence.
        unique: Dict[str, CorrectionAlternative] = {}
        for alt in text_alternatives:
            key = alt.text.lower()
            if key not in unique or alt.confidence > unique[key].confidence:
                unique[key] = alt
        sorted_alts = sorted(unique.values(), key=lambda item: item.confidence, reverse=True)[:3]

        return CorrectionResult(
            original_text=text,
            corrected_text=corrected_text,
            corrections=tuple(corrections),
            confidence=confidence,
            method="symspell+ocr+context",
            alternatives=tuple(sorted_alts),
            status=status,
        )

    def record_feedback(self, original: str, corrected: str) -> None:
        if self.memory is None:
            return
        self.memory.record_correction(original, corrected)

    def _rank_token_candidates(
        self,
        tokens: Sequence[str],
        token_index: int,
        token: str,
    ) -> List[_RankedCandidate]:
        lowered = token.lower()
        is_known_word = self.spell_corrector.is_known_word(lowered) or lowered in _COMMON_SAFE_WORDS
        replacement = self.replacements.get(lowered)
        # An interior capital (for example ``balI``) is a useful OCR signal,
        # but a normal title-cased word such as ``Bali`` must remain intact.
        has_ocr_case_signal = (
            token != token.lower()
            and not (token[:1].isupper() and token[1:].islower())
            and not token.isupper()
        )

        max_edit_distance = max_edit_distance_for_token(lowered)
        if max_edit_distance == 0 and is_known_word and replacement is None:
            return []

        candidates: Dict[str, _RankedCandidate] = {
            lowered: _RankedCandidate(
                word=lowered,
                confidence=0.72 if is_known_word else 0.45,
                reason="original",
                edit_distance=0,
                automatic=False,
            )
        }

        if replacement is not None:
            rep_word, rep_reason = replacement
            candidates[rep_word] = _RankedCandidate(
                word=rep_word,
                confidence=0.96,
                reason=rep_reason,
                edit_distance=word_edit_distance(lowered, rep_word),
                automatic=True,
            )

        for variant in semantic_variants(lowered):
            if variant == lowered:
                continue
            self._upsert_scored_candidate(
                candidates,
                tokens,
                token_index,
                lowered,
                variant,
                edit_distance=word_edit_distance(lowered, variant),
                reason="semantic_context",
                from_ocr_confusion=False,
                memory_boost=0.04,
            )

        # Valid dictionary words are normally left alone.  The only exceptions
        # are context-only homophone choices and a visible OCR case confusion.
        if max_edit_distance > 0 and (not is_known_word or has_ocr_case_signal):
            for match in self.spell_corrector.lookup(lowered, max_edit_distance=max_edit_distance, limit=self.max_candidates):
                self._upsert_scored_candidate(
                    candidates,
                    tokens,
                    token_index,
                    lowered,
                    match.word,
                    edit_distance=match.distance,
                    reason=match.reason,
                    from_ocr_confusion=False,
                )

            for match in self.spell_corrector.lookup_damerau(
                lowered,
                max_distance=min(max_edit_distance, self.max_transpositions),
                limit=self.max_candidates,
            ):
                self._upsert_scored_candidate(
                    candidates,
                    tokens,
                    token_index,
                    lowered,
                    match.word,
                    edit_distance=match.distance,
                    reason=match.reason,
                    from_ocr_confusion=False,
                )

            for variant, ocr_cost in generate_ocr_confusion_variants(token):
                if not self.spell_corrector.is_known_word(variant):
                    continue
                self._upsert_scored_candidate(
                    candidates,
                    tokens,
                    token_index,
                    lowered,
                    variant,
                    edit_distance=word_edit_distance(lowered, variant),
                    reason="ocr_confusion",
                    from_ocr_confusion=True,
                    ocr_cost=ocr_cost,
                )

        preferred = self.memory.get_preferred_correction(lowered) if self.memory is not None else None
        if preferred:
            self._upsert_scored_candidate(
                candidates,
                tokens,
                token_index,
                lowered,
                preferred,
                edit_distance=word_edit_distance(lowered, preferred),
                reason="personalized_memory",
                from_ocr_confusion=False,
                memory_boost=0.12,
            )

        ranked = sorted(candidates.values(), key=lambda item: item.confidence, reverse=True)

        if is_known_word and replacement is None and not has_ocr_case_signal and ranked:
            best = ranked[0]
            # Prevent aggressive replacements of valid words unless context is clearly better.
            if best.word != lowered and (best.confidence - candidates[lowered].confidence) < 0.18:
                return []

        final_ranked: List[_RankedCandidate] = []
        for item in ranked:
            if item.word == lowered:
                continue
            # Homophone substitutions are only safe when local context gives
            # affirmative evidence; a neutral semantic-group prior is not
            # enough to rewrite a student's valid word.
            if (
                item.reason == "semantic_context"
                and self.context_scorer.score(tokens, token_index, item.word) < 0.90
            ):
                continue
            automatic = item.confidence >= self.auto_threshold and max_edit_distance > 0
            if item.reason in {
                "letter_swap",
                "dysgraphia_transposition",
                "adjacent_transposition",
                "omission",
            }:
                automatic = True
            if len(lowered) <= 2:
                automatic = False
            if item.confidence < self.suggestion_threshold:
                continue
            final_ranked.append(
                _RankedCandidate(
                    word=item.word,
                    confidence=item.confidence,
                    reason=item.reason,
                    edit_distance=item.edit_distance,
                    automatic=automatic,
                )
            )

        return final_ranked[: self.max_candidates]

    def _upsert_scored_candidate(
        self,
        candidates: Dict[str, _RankedCandidate],
        tokens: Sequence[str],
        token_index: int,
        original: str,
        candidate_word: str,
        edit_distance: int,
        reason: str,
        from_ocr_confusion: bool,
        ocr_cost: int = 0,
        memory_boost: float = 0.0,
    ) -> None:
        if not candidate_word:
            return
        freq_score = self.spell_corrector.frequency_score(candidate_word)
        if edit_distance <= 0:
            edit_score = 1.0
        elif edit_distance == 1:
            edit_score = 0.90
        elif edit_distance == 2:
            edit_score = 0.75
        else:
            edit_score = 0.45
        context_score = self.context_scorer.score(tokens, token_index, candidate_word)
        ocr_score = 1.0 - min(ocr_cost, 2) * 0.25 if from_ocr_confusion else 0.8

        confidence = (
            0.45 * freq_score
            + 0.25 * edit_score
            + 0.20 * context_score
            + 0.10 * ocr_score
            + memory_boost
        )

        # A single, explicit OCR glyph confusion is stronger evidence than a
        # generic edit-distance neighbour.  It still has to resolve to a known
        # dictionary candidate, so this is candidate generation—not a blind
        # character substitution.
        if from_ocr_confusion and edit_distance <= 1:
            confidence += 0.10
        if reason == "adjacent_transposition":
            confidence += 0.10 if edit_distance == 1 else 0.04
        if (
            edit_distance == 1
            and len(original) >= 3
            and original[-1] == original[-2]
            and candidate_word == original[:-1]
        ):
            confidence += 0.08

        if candidate_word == original:
            confidence = max(confidence, 0.85)

        confidence = min(max(confidence, 0.0), 0.99)

        existing = candidates.get(candidate_word)
        ranked = _RankedCandidate(
            word=candidate_word,
            confidence=confidence,
            reason=reason,
            edit_distance=edit_distance,
            automatic=False,
        )
        if existing is None or ranked.confidence > existing.confidence:
            candidates[candidate_word] = ranked

    def _reconstruct_split_words(
        self,
        tokens: Sequence[str],
        corrected_tokens: List[str],
        corrections: List[Correction],
    ) -> Set[int]:
        """Merge only strong, likely OCR word splits (for example ``C rhismtas``).

        Requiring a one-character leading fragment avoids destructive merges of
        normal adjacent words such as ``the cat``.  The combined token must map
        to a known dictionary word within the configured low-cost Damerau
        budget, so spaces are never removed merely because concatenation exists.
        """
        consumed: Set[int] = set()
        for index in range(len(tokens) - 2):
            first, separator, second = tokens[index : index + 3]
            if not (
                len(first) == 1
                and WORD_RE.match(first)
                and separator.isspace()
                and "\n" not in separator
                and WORD_RE.match(second)
            ):
                continue
            raw = first + second
            candidates = self.spell_corrector.lookup_damerau(
                raw, max_distance=self.max_transpositions, limit=self.max_candidates
            )
            if not candidates:
                continue
            best = candidates[0]
            if not best.word.startswith(first.lower()):
                continue
            # Exact reconstruction and a one/two-swap reconstruction are
            # strong enough to apply automatically; longer transformations are
            # intentionally excluded by the bounded lookup.
            merged = best.word.capitalize() if first.isupper() else best.word
            confidence = min(0.98, 0.92 + (0.04 if best.distance == 0 else 0.02))
            reason = "token_merge" if best.distance == 0 else "token_merge+adjacent_transposition"
            corrected_tokens[index] = merged
            corrected_tokens[index + 1] = ""
            corrected_tokens[index + 2] = ""
            corrections.append(
                Correction(
                    original=first + separator + second,
                    corrected=merged,
                    confidence=confidence,
                    reason=reason,
                    alternatives=tuple(
                        CorrectionAlternative(
                            text=(candidate.word.capitalize() if first.isupper() else candidate.word),
                            confidence=max(0.65, confidence - 0.08 * candidate.distance),
                            reason=reason,
                        )
                        for candidate in candidates[:3]
                    ),
                    edit_distance=best.distance,
                    automatic=True,
                    status="automatic",
                )
            )
            if self.debug or os.environ.get("AWP_DEBUG_CORRECTION", "0").strip() == "1":
                logger.info(
                    "OCR merge=%r candidates=%s selected=%r conf=%.3f reason=%s",
                    first + separator + second,
                    [(candidate.word, candidate.distance) for candidate in candidates],
                    best.word,
                    confidence,
                    reason,
                )
            consumed.update({index, index + 1, index + 2})
        return consumed


def tokenize_preserving_layout(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def preserve_case(original: str, corrected: str) -> str:
    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected.capitalize()
    return corrected


def max_edit_distance_for_token(token: str) -> int:
    length = len(token)
    if length <= 2:
        return 0
    if length == 3:
        return 1
    return 2


def word_edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    rows = len(left) + 1
    cols = len(right) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + cost,
            )
    return table[-1][-1]


def generate_ocr_confusion_variants(token: str) -> List[Tuple[str, int]]:
    variants: Dict[str, int] = {}
    chars = list(token)
    for index, ch in enumerate(chars):
        replacements = _CONFUSION_MAP.get(ch, set()) | _CONFUSION_MAP.get(ch.lower(), set())
        for rep in replacements:
            candidate_chars = chars[:]
            candidate_chars[index] = rep
            candidate = "".join(candidate_chars).lower()
            if candidate:
                variants[candidate] = min(variants.get(candidate, 99), 1)

    # Two-position replacements for stronger OCR confusion patterns.
    for first in range(len(chars)):
        first_reps = _CONFUSION_MAP.get(chars[first], set()) | _CONFUSION_MAP.get(chars[first].lower(), set())
        if not first_reps:
            continue
        for second in range(first + 1, len(chars)):
            second_reps = _CONFUSION_MAP.get(chars[second], set()) | _CONFUSION_MAP.get(chars[second].lower(), set())
            if not second_reps:
                continue
            for left_rep in first_reps:
                for right_rep in second_reps:
                    candidate_chars = chars[:]
                    candidate_chars[first] = left_rep
                    candidate_chars[second] = right_rep
                    candidate = "".join(candidate_chars).lower()
                    if candidate:
                        variants[candidate] = min(variants.get(candidate, 99), 2)

    return sorted(variants.items(), key=lambda item: (item[1], item[0]))


def semantic_variants(word: str) -> Tuple[str, ...]:
    for group in _SEMANTIC_GROUPS:
        if word in group:
            return group
    return ()

"""Hugging Face model-backed correction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
from functools import lru_cache
from assistive_writing_pad.config.cpu_runtime import configure_cpu, optimize_cpu_model
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from rapidfuzz.distance import Levenshtein

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import (
    Correction,
    CorrectionResult,
    RecognitionHypothesisSelection,
)

logger = logging.getLogger(__name__)

DEFAULT_SPELLING_MODEL = "oliverguhr/spelling-correction-english-base"
DEFAULT_GRAMMAR_MODEL = "gotutiyan/gec-bart-base"

MODEL_PROMPTS: Mapping[str, str] = {
    DEFAULT_SPELLING_MODEL: "{text}",
    DEFAULT_GRAMMAR_MODEL: "{text}",
    "Unbabel/gec-t5_small": "gec: {text}",
    "pszemraj/flan-t5-large-grammar-synthesis": "{text}",
    "grammarly/coedit-large": "Fix grammatical errors in this sentence: {text}",
    "grammarly/coedit-xl": "Fix grammatical errors in this sentence: {text}",
}

MODEL_TOKENIZERS: Mapping[str, str] = {
    "Unbabel/gec-t5_small": "t5-small",
}

TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", re.ASCII)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?$", re.ASCII)
TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?,:;]+$", re.ASCII)
TRAILING_QUOTE_ARTIFACT_RE = re.compile(r"""['"`]+[.!?,:;]?$""", re.ASCII)


class ModelCorrectionUnavailable(RuntimeError):
    """Raised when a configured model cannot be loaded or executed."""


@dataclass(frozen=True)
class GeneratedCorrection:
    text: str
    confidence: float
    model_name: str
    stage: str


class CorrectionModelRunner(Protocol):
    model_name: str
    stage: str

    def generate(self, text: str) -> Sequence[GeneratedCorrection]:
        """Return ranked model generations for text."""


@dataclass
class HFSeq2SeqCorrectionRunner:
    """Lazy `transformers` runner for one seq2seq correction model."""

    model_name: str
    stage: str
    prompt_template: str = "{text}"
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[Path] = None
    local_files_only: bool = False
    device: str = "auto"
    num_beams: int = 6
    num_return_sequences: int = 6
    max_input_tokens: int = 128
    max_new_tokens: int = 128
    _tokenizer: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)
    _torch: object = field(default=None, init=False, repr=False)
    _device: str = field(default="cpu", init=False, repr=False)

    def warm_up(self) -> None:
        self.generate("The child is writing.")

    def generate(self, text: str) -> Sequence[GeneratedCorrection]:
        self._ensure_loaded()
        if self._tokenizer is None or self._model is None or self._torch is None:
            raise ModelCorrectionUnavailable(f"{self.model_name} did not load")

        prompt = self.prompt_template.format(text=text.strip())
        tokenizer = self._tokenizer
        torch = self._torch
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        beam_count = max(self.num_beams, self.num_return_sequences)
        try:
            with torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    num_beams=beam_count,
                    num_return_sequences=self.num_return_sequences,
                    max_new_tokens=self.max_new_tokens,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        except Exception as exc:  # pragma: no cover - depends on model runtime.
            raise ModelCorrectionUnavailable(
                f"{self.model_name} generation failed: {exc}"
            ) from exc

        confidences = self._sequence_confidences(generated)
        decoded = tokenizer.batch_decode(
            generated.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        results: List[GeneratedCorrection] = []
        seen: set[str] = set()
        for index, raw_text in enumerate(decoded):
            candidate = normalize_generated_text(raw_text)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            confidence = confidences[index] if index < len(confidences) else 0.75
            results.append(
                GeneratedCorrection(
                    text=candidate,
                    confidence=confidence,
                    model_name=self.model_name,
                    stage=self.stage,
                )
            )
        return results

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._torch = torch
            configure_cpu(torch)
            self._device = resolve_device(self.device, torch)
            tokenizer_source = self.tokenizer_name or MODEL_TOKENIZERS.get(
                self.model_name,
                self.model_name,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                local_files_only=self.local_files_only,
            )
            self._model.to(self._device)
            self._model.eval()
            if self._device == "cpu":
                self._model = optimize_cpu_model(self._model, torch)
        except Exception as exc:  # pragma: no cover - depends on local model env/cache.
            raise ModelCorrectionUnavailable(f"could not load {self.model_name}: {exc}") from exc

        logger.info(
            "loaded %s correction model %s on %s in %.2fs",
            self.stage,
            self.model_name,
            self._device,
            time.perf_counter() - started,
        )

    def _sequence_confidences(self, generated: object) -> List[float]:
        if self._model is None:
            return []
        try:
            beam_indices = getattr(generated, "beam_indices", None)
            if beam_indices is None:
                scores = self._model.compute_transition_scores(
                    generated.sequences,
                    generated.scores,
                    normalize_logits=True,
                )
            else:
                scores = self._model.compute_transition_scores(
                    generated.sequences,
                    generated.scores,
                    beam_indices=beam_indices,
                    normalize_logits=True,
                )
        except Exception:  # pragma: no cover - optional model helper.
            return []

        confidences: List[float] = []
        for row in scores:
            values = row.detach().float()
            values = values[values < 0]
            if values.numel() == 0:
                confidences.append(0.75)
                continue
            mean_logprob = float(values.mean().item())
            confidences.append(float(max(0.0, min(1.0, math.exp(mean_logprob)))))
        return confidences


@dataclass
class HuggingFaceCorrectionPipeline:
    """Model pipeline: spelling, semantic real-word correction, then GEC."""

    lexical_runner: Optional[CorrectionModelRunner] = None
    spelling_runner: Optional[CorrectionModelRunner] = None
    semantic_runner: Optional[CorrectionModelRunner] = None
    grammar_runner: Optional[CorrectionModelRunner] = None
    min_generation_confidence: float = 0.0
    max_change_ratio: float = 0.70

    @classmethod
    def from_settings(cls, settings: RuntimeSettings) -> "HuggingFaceCorrectionPipeline":
        cache_dir = settings.hf_cache_dir
        common_kwargs = {
            "cache_dir": cache_dir,
            "local_files_only": settings.hf_correction_local_files_only,
            "device": settings.hf_correction_device,
            "num_beams": settings.hf_correction_num_beams,
            "num_return_sequences": settings.hf_correction_candidates,
            "max_input_tokens": settings.hf_correction_max_input_tokens,
            "max_new_tokens": settings.hf_correction_max_new_tokens,
        }
        spelling_runner: Optional[CorrectionModelRunner] = None
        lexical_runner: Optional[CorrectionModelRunner] = None
        semantic_runner: Optional[CorrectionModelRunner] = None
        grammar_runner: Optional[CorrectionModelRunner] = None

        if settings.hf_spelling_model_enabled:
            spelling_runner = HFSeq2SeqCorrectionRunner(
                model_name=settings.hf_spelling_model,
                stage="spelling",
                prompt_template=prompt_for_model(settings.hf_spelling_model, "{text}"),
                tokenizer_name=MODEL_TOKENIZERS.get(settings.hf_spelling_model),
                **common_kwargs,
            )

        if settings.hf_semantic_model_enabled:
            from assistive_writing_pad.correction.semantic import SemanticCorrectionRunner

            semantic_runner = SemanticCorrectionRunner(
                model_name=settings.hf_semantic_model,
                cache_dir=cache_dir,
                local_files_only=settings.hf_correction_local_files_only,
                device=settings.hf_correction_device,
                max_input_tokens=settings.hf_correction_max_input_tokens,
                min_score=settings.hf_semantic_min_score,
                min_margin=settings.hf_semantic_min_margin,
                min_ratio=settings.hf_semantic_min_ratio,
            )

        if settings.hf_grammar_model_enabled:
            grammar_runner = HFSeq2SeqCorrectionRunner(
                model_name=settings.hf_grammar_model,
                stage="grammar",
                prompt_template=prompt_for_model(settings.hf_grammar_model, "gec: {text}"),
                tokenizer_name=MODEL_TOKENIZERS.get(settings.hf_grammar_model),
                **common_kwargs,
            )

        return cls(
            lexical_runner=lexical_runner,
            spelling_runner=spelling_runner,
            semantic_runner=semantic_runner,
            grammar_runner=grammar_runner,
            min_generation_confidence=settings.hf_correction_min_confidence,
            max_change_ratio=settings.hf_correction_max_change_ratio,
        )

    def warm_up(self) -> None:
        for runner in self._active_runners():
            warm_up = getattr(runner, "warm_up", None)
            if callable(warm_up):
                warm_up()

    def select_recognition_candidate(
        self,
        candidates: Sequence[Tuple[str, float]],
    ) -> RecognitionHypothesisSelection:
        unique: Dict[str, Tuple[str, float]] = {}
        for text, confidence in candidates:
            cleaned = normalize_generated_text(text)
            if not cleaned:
                continue
            key = cleaned.casefold()
            current = unique.get(key)
            if current is None or confidence > current[1]:
                unique[key] = (cleaned, max(0.0, min(1.0, float(confidence))))
        hypotheses = list(unique.values())[:5]
        if len(hypotheses) <= 1:
            text, confidence = hypotheses[0] if hypotheses else ("", 0.0)
            return RecognitionHypothesisSelection(text=text, confidence=confidence)

        prepared = [self._prepare_recognition_hypothesis(text) for text, _ in hypotheses]
        if max(len(normalized_word_tokens(text)) for text in prepared) < 3:
            text, confidence = hypotheses[0]
            return RecognitionHypothesisSelection(text=text, confidence=confidence)

        scored = []
        for index, ((text, ocr_confidence), normalized) in enumerate(zip(hypotheses, prepared)):
            lexical_score = recognition_lexical_score(normalized)
            repair_ratio = text_change_ratio(text, normalized)
            final_score = (
                (ocr_confidence * 0.50)
                + (lexical_score * 0.50)
                - (repair_ratio * 0.08)
            )
            scored.append((text, ocr_confidence, max(0.0, min(1.0, final_score))))
        scored.sort(key=lambda item: item[2], reverse=True)
        selected_text, selected_confidence, selected_score = scored[0]
        primary_text = hypotheses[0][0]
        primary = next(item for item in scored if item[0] == primary_text)
        if selected_text != primary_text and selected_score < primary[2] + 0.02:
            selected_text, selected_confidence, _ = primary
        return RecognitionHypothesisSelection(
            text=selected_text,
            confidence=selected_confidence,
            rankings=tuple((text, round(score, 4)) for text, _confidence, score in scored),
            metadata={
                "selector": "trocr+wordfreq_corpus",
                "candidate_count": str(len(scored)),
            },
        )

    def _prepare_recognition_hypothesis(self, text: str) -> str:
        if self.lexical_runner is None:
            return text
        try:
            generated = list(self.lexical_runner.generate(text))
        except ModelCorrectionUnavailable:
            return text
        return generated[0].text if generated else text

    def correct(self, text: str) -> CorrectionResult:
        if not text.strip():
            return CorrectionResult(original_text=text, corrected_text=text, confidence=1.0)
        if is_isolated_character_input(text):
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                confidence=1.0,
                metadata={
                    "backend": "huggingface",
                    "skipped": "isolated_character",
                    "stages": "[]",
                },
            )

        current = text
        stages: List[Dict[str, object]] = []
        applied: List[GeneratedCorrection] = []
        errors: List[Dict[str, str]] = []

        for runner in self._active_runners():
            before = current
            if is_single_word_fragment_input(before) and runner.stage != "spelling":
                stages.append(
                    {
                        "stage": runner.stage,
                        "model": runner.model_name,
                        "input": before,
                        "accepted": False,
                        "skipped": "single_word_fragment",
                        "alternatives": [],
                    }
                )
                continue
            try:
                generations = list(runner.generate(before))
            except ModelCorrectionUnavailable as exc:
                logger.warning("%s correction unavailable: %s", runner.stage, exc)
                errors.append(
                    {
                        "stage": runner.stage,
                        "model": runner.model_name,
                        "error": str(exc),
                    }
                )
                continue

            accepted = [
                item
                for item in generations
                if self._accept_generation(
                    before,
                    item.text,
                    item.confidence,
                    stage=runner.stage,
                )
            ]
            # A high-probability generation that preserves the words is evidence
            # for leaving them alone. Do not discard it for adding a period and
            # then choose a much weaker, meaning-changing beam instead.
            unchanged_words = [
                item for item in generations
                if normalized_word_tokens(before) == normalized_word_tokens(item.text)
                and not has_trailing_quote_artifact(before)
            ]
            if unchanged_words:
                unchanged_confidence = max(item.confidence for item in unchanged_words)
                accepted = [
                    item for item in accepted
                    if normalize_generated_text(before) == normalize_generated_text(item.text)
                    or item.confidence > unchanged_confidence
                ]
            accepted.sort(
                key=lambda item: self._generation_rank_score(before, item),
                reverse=True,
            )
            for item in generations:
                logger.info(
                    "correction candidate stage=%s text=%r model_confidence=%.4f "
                    "accepted=%s rank_score=%.4f",
                    runner.stage,
                    item.text,
                    item.confidence,
                    item in accepted,
                    self._generation_rank_score(before, item),
                )
            stage_record = {
                "stage": runner.stage,
                "model": runner.model_name,
                "input": before,
                "accepted": bool(accepted),
                "alternatives": [
                    {
                        "text": item.text,
                        "confidence": round(item.confidence, 4),
                        "rank_score": round(self._generation_rank_score(before, item), 4),
                    }
                    for item in generations
                ],
            }
            if accepted:
                best = accepted[0]
                best_text = best.text
                if runner.stage == "grammar":
                    best_text = finalize_grammar_output(before, best_text)
                current = preserve_outer_whitespace(before, best_text)
                if normalize_generated_text(current) != normalize_generated_text(before):
                    applied.append(best)
                    stage_record["output"] = current
            stages.append(stage_record)

        confidence = min((item.confidence for item in applied), default=1.0)
        reason = reason_for_applied_stages(applied)
        corrections = diff_corrections(text, current, confidence=confidence, reason=reason)
        metadata = {
            "backend": "huggingface",
            "stages": json.dumps(stages),
        }
        if errors:
            metadata["errors"] = json.dumps(errors)

        return CorrectionResult(
            original_text=text,
            corrected_text=current,
            corrections=tuple(corrections),
            confidence=confidence if corrections else 1.0,
            metadata=metadata,
        )

    def _active_runners(self) -> Tuple[CorrectionModelRunner, ...]:
        runners: List[CorrectionModelRunner] = []
        if self.spelling_runner is not None:
            if self.lexical_runner is not None:
                runners.append(self.lexical_runner)
            runners.append(self.spelling_runner)
        elif self.lexical_runner is not None:
            runners.append(self.lexical_runner)
        if self.semantic_runner is not None:
            runners.append(self.semantic_runner)
        if self.grammar_runner is not None:
            runners.append(self.grammar_runner)
        return tuple(runners)

    def _accept_generation(
        self,
        original: str,
        candidate: str,
        confidence: float,
        *,
        stage: str = "",
    ) -> bool:
        if confidence < self.min_generation_confidence:
            return False
        if normalize_generated_text(original) == normalize_generated_text(candidate):
            return True
        if is_split_word_fragment_input(original) and not is_word_or_ocr_fragment_output(
            candidate
        ):
            return False
        if is_unrequested_punctuation_or_case_only_change(original, candidate):
            return False
        if stage == "spelling" and not is_probable_spelling_change(original, candidate):
            return False
        if stage == "grammar" and not preserves_grammar_content(original, candidate):
            return False
        return is_acceptable_model_output(
            original,
            candidate,
            max_change_ratio=self.max_change_ratio,
        )

    def _generation_rank_score(self, original: str, candidate: GeneratedCorrection) -> float:
        score = candidate.confidence
        if candidate.stage == "spelling":
            character_similarity = difflib.SequenceMatcher(
                a=normalize_generated_text(original).casefold(),
                b=normalize_generated_text(candidate.text).casefold(),
                autojunk=False,
            ).ratio()
            score = (
                (candidate.confidence * 0.05)
                + (recognition_lexical_score(candidate.text) * 0.40)
                + (character_similarity * 0.55)
            )
        original_tokens = normalized_word_tokens(original)
        candidate_tokens = normalized_word_tokens(candidate.text)
        if original_tokens and candidate_tokens:
            score -= token_change_ratio(original_tokens, candidate_tokens) * 0.08

        if has_trailing_quote_artifact(original):
            terminal = terminal_punctuation(candidate.text)
            if terminal == ".":
                score += 0.14
            elif terminal in {",", ":", ";"}:
                score -= 0.16
        return score


def prompt_for_model(model_name: str, fallback: str) -> str:
    return MODEL_PROMPTS.get(model_name, fallback)


def resolve_device(device: str, torch: object) -> str:
    if device != "auto":
        return device
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class WordfreqFragmentCorrectionRunner:
    """Repair OCR token fragments using word-frequency and edit-distance scoring."""

    stage: str = "lexical"
    model_name: str = "wordfreq-en-zipf-rapidfuzz"
    language: str = "en"
    top_n: int = 100000
    min_zipf: float = 3.0
    min_margin: float = 0.05
    distance_penalty: float = 0.60

    def generate(self, text: str) -> Sequence[GeneratedCorrection]:
        corrected = repair_ocr_fragments(
            text,
            language=self.language,
            top_n=self.top_n,
            min_zipf=self.min_zipf,
            min_margin=self.min_margin,
            distance_penalty=self.distance_penalty,
        )
        if corrected == text:
            return ()
        return (
            GeneratedCorrection(
                text=corrected,
                confidence=0.78,
                model_name=self.model_name,
                stage=self.stage,
            ),
        )


def normalize_generated_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([({\[])\s+", r"\1", cleaned)
    return cleaned


def recognition_lexical_score(text: str, language: str = "en") -> float:
    words = normalized_word_tokens(text)
    if not words:
        return 0.0
    scores = [
        max(0.0, min(1.0, (zipf_frequency(word, language) - 2.0) / 5.0))
        for word in words
    ]
    token_score = float(sum(scores) / len(scores))
    phrase_score = max(
        0.0,
        min(1.0, (zipf_frequency(" ".join(words), language) - 2.0) / 3.0),
    )
    return (token_score * 0.55) + (phrase_score * 0.45)


def text_change_ratio(original: str, candidate: str) -> float:
    left = normalize_generated_text(original).casefold()
    right = normalize_generated_text(candidate).casefold()
    if not left and not right:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()


def repair_ocr_fragments(
    text: str,
    *,
    language: str = "en",
    top_n: int = 100000,
    min_zipf: float = 3.0,
    min_margin: float = 0.05,
    distance_penalty: float = 0.60,
) -> str:
    tokens = TEXT_TOKEN_RE.findall(text)
    if len([token for token in tokens if WORD_RE.match(token)]) < 2:
        return text

    repaired: List[str] = []
    changed = False
    index = 0
    while index < len(tokens):
        current = tokens[index]
        if index + 1 >= len(tokens):
            repaired.append(current)
            break
        next_token = tokens[index + 1]
        if not WORD_RE.match(current) or not WORD_RE.match(next_token):
            repaired.append(current)
            index += 1
            continue

        candidate = best_fragment_merge_candidate(
            current,
            next_token,
            language=language,
            top_n=top_n,
            min_zipf=min_zipf,
            min_margin=min_margin,
            distance_penalty=distance_penalty,
        )
        if candidate is None:
            repaired.append(current)
            index += 1
            continue
        repaired.append(_preserve_fragment_case(current, next_token, candidate))
        changed = True
        index += 2

    if not changed:
        return text
    return normalize_generated_text(" ".join(repaired))


def best_fragment_merge_candidate(
    left: str,
    right: str,
    *,
    language: str,
    top_n: int,
    min_zipf: float,
    min_margin: float,
    distance_penalty: float,
) -> Optional[str]:
    left_word = left.lower()
    right_word = right.lower()
    if not should_consider_fragment_merge(left_word, right_word, language=language):
        return None

    compact = f"{left_word}{right_word}"
    candidate_scores: Dict[str, float] = {}
    for candidate in direct_fragment_candidates(left_word, right_word):
        frequency = zipf_frequency(candidate, language)
        if frequency >= min_zipf:
            distance = Levenshtein.distance(compact, candidate)
            candidate_scores[candidate] = frequency - (distance * distance_penalty)

    max_distance = 2 if len(compact) <= 7 else 3
    for word in frequent_words(language, top_n):
        if abs(len(word) - len(compact)) > max_distance:
            continue
        distance = Levenshtein.distance(compact, word)
        if distance > max_distance:
            continue
        frequency = zipf_frequency(word, language)
        if frequency < min_zipf:
            continue
        score = frequency - (distance * distance_penalty)
        existing = candidate_scores.get(word)
        if existing is None or score > existing:
            candidate_scores[word] = score

    if not candidate_scores:
        return None

    best, best_score = max(candidate_scores.items(), key=lambda item: (item[1], item[0]))
    original_score = zipf_frequency(f"{left_word} {right_word}", language)
    if best_score < original_score + min_margin:
        return None
    return best


def should_consider_fragment_merge(left: str, right: str, *, language: str) -> bool:
    if len(left) + len(right) < 4:
        return False
    left_frequency = zipf_frequency(left, language)
    right_frequency = zipf_frequency(right, language)
    if left_frequency == 0.0 or right_frequency == 0.0:
        return True
    if len(left) == 1 or len(right) == 1:
        other_frequency = right_frequency if len(left) == 1 else left_frequency
        if other_frequency >= 4.5:
            return False
        return True
    return False


def direct_fragment_candidates(left: str, right: str) -> Sequence[str]:
    compact = f"{left}{right}"
    candidates = {compact}
    if right in {"i", "l", "1"}:
        candidates.add(f"{left}l")
    if left in {"i", "l", "1"}:
        candidates.add(f"i{right}")
    return tuple(candidates)


@lru_cache(maxsize=8)
def frequent_words(language: str, top_n: int) -> Tuple[str, ...]:
    try:
        from wordfreq import top_n_list
    except ImportError as exc:  # pragma: no cover - dependency is declared.
        raise ModelCorrectionUnavailable("wordfreq is not installed") from exc

    words = [
        word
        for word in top_n_list(language, top_n)
        if word.isalpha() and 3 <= len(word) <= 18
    ]
    return tuple(words)


def zipf_frequency(word: str, language: str) -> float:
    try:
        from wordfreq import zipf_frequency as score
    except ImportError as exc:  # pragma: no cover - dependency is declared.
        raise ModelCorrectionUnavailable("wordfreq is not installed") from exc
    return float(score(word, language))


def _preserve_fragment_case(left: str, right: str, candidate: str) -> str:
    if left[:1].isupper() and not right.isupper():
        return candidate.capitalize()
    if left.isupper() and right.isupper():
        return candidate.upper()
    return candidate


def preserve_outer_whitespace(original: str, corrected: str) -> str:
    prefix = original[: len(original) - len(original.lstrip())]
    suffix = original[len(original.rstrip()) :]
    return f"{prefix}{normalize_generated_text(corrected)}{suffix}"


def is_isolated_character_input(text: str) -> bool:
    cleaned = normalize_generated_text(text)
    return len(cleaned) == 1 and cleaned.isalpha()


def is_single_word_fragment_input(text: str) -> bool:
    cleaned = normalize_generated_text(text)
    return bool(WORD_RE.match(cleaned))


def is_word_or_ocr_fragment_input(text: str) -> bool:
    tokens = normalized_word_tokens(text)
    cleaned = normalize_generated_text(text)
    return bool(tokens) and len(tokens) <= 2 and not any(char in ".,!?;:" for char in cleaned)


def is_split_word_fragment_input(text: str) -> bool:
    tokens = normalized_word_tokens(text)
    cleaned = normalize_generated_text(text)
    if len(tokens) != 2 or any(char in ".,!?;:" for char in cleaned):
        return False
    return best_fragment_merge_candidate(
        tokens[0],
        tokens[1],
        language="en",
        top_n=100000,
        min_zipf=3.0,
        min_margin=0.05,
        distance_penalty=0.60,
    ) is not None


def is_word_or_ocr_fragment_output(text: str) -> bool:
    tokens = normalized_word_tokens(text)
    cleaned = normalize_generated_text(text)
    if not tokens:
        return False
    if len(tokens) > 1:
        return False
    if any(char in ".,!?;:" for char in cleaned):
        return False
    return True


def is_unrequested_punctuation_or_case_only_change(original: str, candidate: str) -> bool:
    cleaned_original = normalize_generated_text(original)
    cleaned_candidate = normalize_generated_text(candidate)
    if not cleaned_original or not cleaned_candidate:
        return False

    original_tokens = normalized_word_tokens(cleaned_original)
    candidate_tokens = normalized_word_tokens(cleaned_candidate)
    if original_tokens != candidate_tokens:
        return False

    if cleaned_original.lower() == cleaned_candidate.lower():
        return True

    if has_trailing_quote_artifact(cleaned_original):
        return False

    # Identical word tokens also catch a hallucinated leading quote or comma.
    return True


def preserves_grammar_content(original: str, candidate: str) -> bool:
    """Allow function-word and inflection fixes, reject unrelated content rewrites.

    Semantic alternatives belong to the separately scored closed-set semantic
    stage. A fluent grammar generation is not evidence for changing a noun.
    """
    function_words = set(
        "a an the am is are was were be been being do does did have has had "
        "to of in on at for from by with as and or but if then than that this "
        "these those it its i you he she we they me him her us them my your "
        "his our their there not no will would can could should shall may might".split()
    )
    left, right = normalized_word_tokens(original), normalized_word_tokens(candidate)
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        before, after = left[left_start:left_end], right[right_start:right_end]
        if all(token in function_words for token in before + after):
            continue
        if tag != "replace" or len(before) != len(after):
            return False
        for source, target in zip(before, after):
            if source in function_words and target in function_words:
                continue
            # Preserve inflection changes without allowing free substitution.
            def stems(word):
                values = {word}
                for suffix in ('s', 'es', 'ed', 'ing'):
                    if word.endswith(suffix) and len(word) > len(suffix) + 1:
                        base = word[:-len(suffix)]
                        values.update((base, base + 'e'))
                        if len(base) > 2 and base[-1] == base[-2]:
                            values.add(base[:-1])
                return values
            if not stems(source).intersection(stems(target)):
                return False
    return True


def has_terminal_punctuation(text: str) -> bool:
    cleaned = normalize_generated_text(text)
    return bool(cleaned) and cleaned[-1] in ".!?,:;"


def has_trailing_quote_artifact(text: str) -> bool:
    return bool(TRAILING_QUOTE_ARTIFACT_RE.search(normalize_generated_text(text)))


def terminal_punctuation(text: str) -> str:
    cleaned = normalize_generated_text(text)
    if not cleaned:
        return ""
    if cleaned[-1] in ".!?,:;":
        return cleaned[-1]
    return ""


def finalize_grammar_output(original: str, candidate: str) -> str:
    cleaned_candidate = normalize_generated_text(candidate)
    if terminal_punctuation(cleaned_candidate):
        return candidate
    if has_trailing_quote_artifact(original):
        return candidate

    original_tokens = normalized_word_tokens(original)
    candidate_tokens = normalized_word_tokens(cleaned_candidate)
    if len(candidate_tokens) < 4:
        return candidate
    if token_change_ratio(original_tokens, candidate_tokens) <= 0.0:
        return candidate
    return f"{cleaned_candidate}."


def is_probable_spelling_change(
    original: str,
    candidate: str,
    *,
    language: str = "en",
) -> bool:
    original_tokens = normalized_word_tokens(original)
    candidate_tokens = normalized_word_tokens(candidate)
    if not original_tokens or not candidate_tokens:
        return True
    if original_tokens == candidate_tokens:
        return True
    if len(original_tokens) != len(candidate_tokens):
        return False

    matcher = difflib.SequenceMatcher(
        a=original_tokens,
        b=candidate_tokens,
        autojunk=False,
    )
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or (left_end - left_start) != (right_end - right_start):
            return False
        for source, target in zip(
            original_tokens[left_start:left_end],
            candidate_tokens[right_start:right_end],
        ):
            if not is_probable_word_spelling_change(source, target, language=language):
                return False
    return True


def is_probable_word_spelling_change(
    source: str,
    target: str,
    *,
    language: str,
) -> bool:
    if source == target:
        return True
    distance = Levenshtein.distance(source, target)
    max_distance = max(2, math.ceil(len(source) * 0.35))
    if distance > max_distance:
        return False
    if len(source) <= 2 and len(target) <= 4:
        return True

    source_frequency = zipf_frequency(source, language)
    target_frequency = zipf_frequency(target, language)
    if target_frequency < source_frequency + 0.35:
        return False
    if source_frequency >= 4.5:
        return False
    return True


def is_acceptable_model_output(
    original: str,
    candidate: str,
    max_change_ratio: float = 0.70,
) -> bool:
    cleaned_original = normalize_generated_text(original)
    cleaned_candidate = normalize_generated_text(candidate)
    if not cleaned_candidate:
        return False
    if cleaned_candidate.lower() in {"none", "null", "n/a"}:
        return False
    if any(ord(char) < 32 and char not in "\n\t" for char in cleaned_candidate):
        return False
    if any(char.isalpha() for char in cleaned_original) and not any(
        char.isalpha() for char in cleaned_candidate
    ):
        return False
    if len(cleaned_candidate) > max(64, len(cleaned_original) * 3):
        return False
    if len(cleaned_candidate) < max(1, len(cleaned_original) // 4):
        return False

    original_tokens = normalized_word_tokens(cleaned_original)
    candidate_tokens = normalized_word_tokens(cleaned_candidate)
    if not original_tokens or not candidate_tokens:
        return True

    diff_ratio = token_change_ratio(original_tokens, candidate_tokens)
    if diff_ratio > max_change_ratio:
        character_similarity = difflib.SequenceMatcher(
            a=cleaned_original.lower(),
            b=cleaned_candidate.lower(),
            autojunk=False,
        ).ratio()
        if character_similarity >= 0.55:
            return True
        shared = len(set(original_tokens) & set(candidate_tokens))
        if shared / max(len(set(original_tokens)), 1) < 0.35:
            return False
    return True


def token_change_ratio(left: Sequence[str], right: Sequence[str]) -> float:
    matcher = difflib.SequenceMatcher(a=list(left), b=list(right), autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return changed / max(len(left), len(right), 1)


def normalized_word_tokens(text: str) -> List[str]:
    return [token.lower() for token in TEXT_TOKEN_RE.findall(text) if WORD_RE.match(token)]


def diff_corrections(
    original: str,
    corrected: str,
    confidence: float,
    reason: str,
) -> List[Correction]:
    left = TEXT_TOKEN_RE.findall(original)
    right = TEXT_TOKEN_RE.findall(corrected)
    matcher = difflib.SequenceMatcher(
        a=[token.lower() for token in left],
        b=[token.lower() for token in right],
        autojunk=False,
    )
    corrections: List[Correction] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        original_span = format_token_span(left[i1:i2])
        corrected_span = format_token_span(right[j1:j2])
        if not original_span and not corrected_span:
            continue
        corrections.append(
            Correction(
                original=original_span,
                corrected=corrected_span,
                confidence=round(confidence, 4),
                reason=reason,
            )
        )
    return corrections


def format_token_span(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    text = " ".join(tokens)
    return normalize_generated_text(text)


def reason_for_applied_stages(applied: Sequence[GeneratedCorrection]) -> str:
    stages = {item.stage for item in applied}
    if stages == {"spelling"}:
        return "hf_spelling_model"
    if stages == {"grammar"}:
        return "hf_grammar_model"
    if stages == {"semantic"}:
        return "hf_semantic_model"
    if stages == {"lexical"}:
        return "wordfreq_fragment_model"
    if stages:
        return "hf_model_pipeline"
    return "hf_model_pipeline"

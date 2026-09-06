"""Hugging Face model-backed correction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import Correction, CorrectionResult

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
    num_beams: int = 4
    num_return_sequences: int = 3
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
                if self._accept_generation(before, item.text, item.confidence)
            ]
            stage_record = {
                "stage": runner.stage,
                "model": runner.model_name,
                "input": before,
                "accepted": bool(accepted),
                "alternatives": [
                    {
                        "text": item.text,
                        "confidence": round(item.confidence, 4),
                    }
                    for item in generations
                ],
            }
            if accepted:
                best = accepted[0]
                current = preserve_outer_whitespace(before, best.text)
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
            runners.append(self.spelling_runner)
        if self.semantic_runner is not None:
            runners.append(self.semantic_runner)
        if self.grammar_runner is not None:
            runners.append(self.grammar_runner)
        return tuple(runners)

    def _accept_generation(self, original: str, candidate: str, confidence: float) -> bool:
        if confidence < self.min_generation_confidence:
            return False
        return is_acceptable_model_output(
            original,
            candidate,
            max_change_ratio=self.max_change_ratio,
        )


def prompt_for_model(model_name: str, fallback: str) -> str:
    return MODEL_PROMPTS.get(model_name, fallback)


def resolve_device(device: str, torch: object) -> str:
    if device != "auto":
        return device
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"
    return "cpu"


def normalize_generated_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([({\[])\s+", r"\1", cleaned)
    return cleaned


def preserve_outer_whitespace(original: str, corrected: str) -> str:
    prefix = original[: len(original) - len(original.lstrip())]
    suffix = original[len(original.rstrip()) :]
    return f"{prefix}{normalize_generated_text(corrected)}{suffix}"


def is_isolated_character_input(text: str) -> bool:
    cleaned = normalize_generated_text(text)
    return len(cleaned) == 1 and cleaned.isalpha()


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
    if stages:
        return "hf_model_pipeline"
    return "hf_model_pipeline"

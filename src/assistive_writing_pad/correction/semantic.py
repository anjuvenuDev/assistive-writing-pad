"""Model-scored semantic correction for real-word errors."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import re
import time
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from assistive_writing_pad.correction.huggingface import (
    GeneratedCorrection,
    ModelCorrectionUnavailable,
    normalize_generated_text,
    preserve_outer_whitespace,
    resolve_device,
)

logger = logging.getLogger(__name__)

DEFAULT_SEMANTIC_MODEL = "distilbert/distilbert-base-uncased"

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\s+|[^\w\s]", re.ASCII)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?$", re.ASCII)

DEFAULT_CONFUSION_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("to", "too", "two"),
    ("there", "their"),
    ("hear", "here"),
    ("write", "right", "rite"),
    ("no", "know"),
    ("than", "then"),
    ("your", "you're"),
    ("its", "it's"),
)


class SemanticScorer(Protocol):
    def score_candidates(self, masked_text: str, candidates: Sequence[str]) -> Mapping[str, float]:
        """Return context scores for candidate words in the masked position."""


@dataclass
class HFMaskedLMSemanticScorer:
    """Lazy Hugging Face masked-language-model scorer."""

    model_name: str = DEFAULT_SEMANTIC_MODEL
    cache_dir: Optional[Path] = None
    local_files_only: bool = False
    device: str = "auto"
    max_input_tokens: int = 128
    _tokenizer: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)
    _torch: object = field(default=None, init=False, repr=False)
    _device: str = field(default="cpu", init=False, repr=False)

    def warm_up(self) -> None:
        self._ensure_loaded()
        tokenizer = self._tokenizer
        mask_token = getattr(tokenizer, "mask_token", "[MASK]")
        self.score_candidates(
            f"I can {mask_token} the word.",
            ("write", "right"),
        )

    def score_candidates(self, masked_text: str, candidates: Sequence[str]) -> Mapping[str, float]:
        self._ensure_loaded()
        if self._tokenizer is None or self._model is None or self._torch is None:
            raise ModelCorrectionUnavailable(f"{self.model_name} did not load")

        tokenizer = self._tokenizer
        torch = self._torch
        encoded = tokenizer(
            masked_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        mask_positions = (encoded["input_ids"] == tokenizer.mask_token_id).nonzero(
            as_tuple=False
        )
        if mask_positions.numel() == 0:
            return {}
        batch_index = int(mask_positions[0][0].item())
        token_index = int(mask_positions[0][1].item())

        with torch.inference_mode():
            output = self._model(**encoded)
            probabilities = torch.softmax(output.logits[batch_index, token_index], dim=-1)

        scores: Dict[str, float] = {}
        for candidate in candidates:
            token_id = single_token_id(tokenizer, candidate)
            if token_id is None:
                continue
            scores[candidate] = float(probabilities[token_id].item())
        return scores

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self._torch = torch
            self._device = resolve_device(self.device, torch)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModelForMaskedLM.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                local_files_only=self.local_files_only,
            )
            self._model.to(self._device)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - depends on local model env/cache.
            raise ModelCorrectionUnavailable(f"could not load {self.model_name}: {exc}") from exc

        logger.info(
            "loaded semantic model %s on %s in %.2fs",
            self.model_name,
            self._device,
            time.perf_counter() - started,
        )


@dataclass
class SemanticCorrectionRunner:
    """Closed-set real-word correction scored by a masked language model."""

    model_name: str = DEFAULT_SEMANTIC_MODEL
    stage: str = "semantic"
    scorer: Optional[SemanticScorer] = None
    confusion_groups: Sequence[Tuple[str, ...]] = DEFAULT_CONFUSION_GROUPS
    min_score: float = 0.01
    min_margin: float = 0.04
    min_ratio: float = 1.55
    cache_dir: Optional[Path] = None
    local_files_only: bool = False
    device: str = "auto"
    max_input_tokens: int = 128

    def warm_up(self) -> None:
        scorer = self._get_scorer()
        warm_up = getattr(scorer, "warm_up", None)
        if callable(warm_up):
            warm_up()

    def generate(self, text: str) -> Sequence[GeneratedCorrection]:
        tokens = TOKEN_RE.findall(text)
        if not tokens:
            return []

        scorer = self._get_scorer()
        corrected_tokens = list(tokens)
        applied_scores: List[float] = []

        for index, token in enumerate(tokens):
            group = self._candidate_group(token)
            if group is None:
                continue
            masked_text = self._masked_text(tokens, index, scorer)
            try:
                scores = scorer.score_candidates(masked_text, group)
            except ModelCorrectionUnavailable:
                raise
            except Exception as exc:  # pragma: no cover - scorer-specific.
                raise ModelCorrectionUnavailable(
                    f"{self.model_name} semantic scoring failed: {exc}"
                ) from exc

            decision = choose_semantic_candidate(token, scores, self.min_score)
            if decision is None:
                continue
            best, best_score, current_score = decision
            if not semantic_margin_passes(best_score, current_score, self.min_margin, self.min_ratio):
                continue

            corrected_tokens[index] = preserve_case(token, best)
            applied_scores.append(semantic_decision_confidence(best_score, current_score))

        corrected = "".join(corrected_tokens)
        if normalize_generated_text(corrected) == normalize_generated_text(text):
            return [GeneratedCorrection(text=text, confidence=1.0, model_name=self.model_name, stage=self.stage)]

        confidence = min(applied_scores, default=0.0)
        return [
            GeneratedCorrection(
                text=preserve_outer_whitespace(text, corrected),
                confidence=confidence,
                model_name=self.model_name,
                stage=self.stage,
            )
        ]

    def _get_scorer(self) -> SemanticScorer:
        if self.scorer is None:
            self.scorer = HFMaskedLMSemanticScorer(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
                device=self.device,
                max_input_tokens=self.max_input_tokens,
            )
        return self.scorer

    def _candidate_group(self, token: str) -> Optional[Tuple[str, ...]]:
        if not WORD_RE.match(token):
            return None
        word = token.lower()
        for group in self.confusion_groups:
            if word in group:
                return group
        return None

    def _masked_text(self, tokens: Sequence[str], token_index: int, scorer: SemanticScorer) -> str:
        mask_token = getattr(scorer, "mask_token", None)
        if mask_token is None:
            tokenizer = getattr(scorer, "_tokenizer", None)
            mask_token = getattr(tokenizer, "mask_token", "[MASK]")
        masked = list(tokens)
        masked[token_index] = str(mask_token)
        return "".join(masked)


def single_token_id(tokenizer: object, word: str) -> Optional[int]:
    encoded = tokenizer.encode(word, add_special_tokens=False)
    if len(encoded) != 1:
        return None
    return int(encoded[0])


def choose_semantic_candidate(
    current_token: str,
    scores: Mapping[str, float],
    min_score: float,
) -> Optional[Tuple[str, float, float]]:
    if not scores:
        return None
    current = current_token.lower()
    current_score = float(scores.get(current, 0.0))
    best, best_score = max(scores.items(), key=lambda item: item[1])
    if best == current or best_score < min_score:
        return None
    return best, float(best_score), current_score


def semantic_margin_passes(
    best_score: float,
    current_score: float,
    min_margin: float,
    min_ratio: float,
) -> bool:
    if best_score - current_score >= min_margin:
        return True
    if current_score <= 0.0 and best_score >= min_margin:
        return True
    return best_score / max(current_score, 1e-9) >= min_ratio


def semantic_decision_confidence(best_score: float, current_score: float) -> float:
    margin = max(0.0, best_score - current_score)
    ratio = best_score / max(current_score, 1e-9)
    score_component = min(best_score * 1.5, 0.25)
    margin_component = min(margin * 2.0, 0.20)
    ratio_component = min(math.log(max(ratio, 1.0)) / 8.0, 0.20)
    return float(min(0.97, 0.45 + score_component + margin_component + ratio_component))


def preserve_case(original: str, corrected: str) -> str:
    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected.capitalize()
    return corrected

"""Runtime settings for laptop and Raspberry Pi execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RuntimeSettings:
    """Configuration values shared by the pipeline components."""

    confidence_threshold: float = 0.85
    max_sentence_latency_ms: int = 2000
    max_word_latency_ms: int = 500
    models_dir: Path = Path("models")
    data_dir: Path = Path("data")
    hf_cache_dir: Path = Path("models/cache/huggingface")
    evaluation_capture_enabled: bool = False
    evaluation_manifest_path: Path = Path("data/evaluation/end_to_end_cases.jsonl")
    api_correction_enabled: bool = False
    device_profile: str = "laptop"
    correction_mode: str = "hf"
    contextual_model_enabled: bool = False
    contextual_model_name: str = "distilbert/distilbert-base-uncased"
    preload_ocr_model: bool = True
    preload_correction_models: bool = True
    max_correction_candidates: int = 8
    correction_confidence_threshold: float = 0.70
    hf_spelling_model_enabled: bool = True
    hf_spelling_model: str = "oliverguhr/spelling-correction-english-base"
    hf_semantic_model_enabled: bool = True
    hf_semantic_model: str = "distilbert/distilbert-base-uncased"
    hf_semantic_min_score: float = 0.01
    hf_semantic_min_margin: float = 0.04
    hf_semantic_min_ratio: float = 1.55
    hf_grammar_model_enabled: bool = True
    hf_grammar_model: str = "gotutiyan/gec-bart-base"
    hf_correction_local_files_only: bool = False
    hf_correction_device: str = "auto"
    hf_correction_num_beams: int = 6
    hf_correction_candidates: int = 6
    hf_correction_max_input_tokens: int = 128
    hf_correction_max_new_tokens: int = 128
    hf_correction_min_confidence: float = 0.0
    hf_correction_max_change_ratio: float = 0.70

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        device_profile = os.environ.get("AWP_DEVICE_PROFILE", "laptop").strip() or "laptop"
        correction_mode = os.environ.get("AWP_CORRECTION_MODE", cls.correction_mode).strip()
        correction_mode = correction_mode or cls.correction_mode
        default_preload = "0" if device_profile == "raspberry_pi" else "1"
        settings = cls(
            confidence_threshold=_float_env("AWP_CONFIDENCE_THRESHOLD", cls.confidence_threshold),
            max_sentence_latency_ms=_int_env(
                "AWP_MAX_SENTENCE_LATENCY_MS", cls.max_sentence_latency_ms
            ),
            max_word_latency_ms=_int_env("AWP_MAX_WORD_LATENCY_MS", cls.max_word_latency_ms),
            models_dir=Path(os.environ.get("AWP_MODELS_DIR", str(cls.models_dir))),
            data_dir=Path(os.environ.get("AWP_DATA_DIR", str(cls.data_dir))),
            hf_cache_dir=huggingface_cache_dir_from_env(
                models_dir=Path(os.environ.get("AWP_MODELS_DIR", str(cls.models_dir))),
            ),
            evaluation_capture_enabled=_bool_env("AWP_EVALUATION_CAPTURE_ENABLED", False),
            evaluation_manifest_path=Path(
                os.environ.get(
                    "AWP_EVALUATION_MANIFEST",
                    str(cls.evaluation_manifest_path),
                )
            ),
            api_correction_enabled=_bool_env("AWP_API_CORRECTION_ENABLED", False),
            device_profile=device_profile,
            correction_mode=correction_mode,
            contextual_model_enabled=_bool_env(
                "AWP_CONTEXTUAL_MODEL_ENABLED", cls.contextual_model_enabled
            ),
            contextual_model_name=os.environ.get(
                "AWP_CONTEXTUAL_MODEL", cls.contextual_model_name
            ).strip()
            or cls.contextual_model_name,
            preload_ocr_model=_bool_env("AWP_PRELOAD_OCR_MODEL", default_preload == "1"),
            preload_correction_models=_bool_env(
                "AWP_PRELOAD_CORRECTION_MODELS",
                default_preload == "1",
            ),
            max_correction_candidates=_int_env(
                "AWP_MAX_CORRECTION_CANDIDATES", cls.max_correction_candidates
            ),
            correction_confidence_threshold=_float_env(
                "AWP_CORRECTION_CONFIDENCE_THRESHOLD",
                cls.correction_confidence_threshold,
            ),
            hf_spelling_model_enabled=_bool_env(
                "AWP_HF_SPELLING_MODEL_ENABLED",
                cls.hf_spelling_model_enabled,
            ),
            hf_spelling_model=os.environ.get(
                "AWP_HF_SPELLING_MODEL",
                cls.hf_spelling_model,
            ).strip()
            or cls.hf_spelling_model,
            hf_semantic_model_enabled=_bool_env(
                "AWP_HF_SEMANTIC_MODEL_ENABLED",
                cls.hf_semantic_model_enabled,
            ),
            hf_semantic_model=os.environ.get(
                "AWP_HF_SEMANTIC_MODEL",
                cls.hf_semantic_model,
            ).strip()
            or cls.hf_semantic_model,
            hf_semantic_min_score=_float_env(
                "AWP_HF_SEMANTIC_MIN_SCORE",
                cls.hf_semantic_min_score,
            ),
            hf_semantic_min_margin=_float_env(
                "AWP_HF_SEMANTIC_MIN_MARGIN",
                cls.hf_semantic_min_margin,
            ),
            hf_semantic_min_ratio=_float_env(
                "AWP_HF_SEMANTIC_MIN_RATIO",
                cls.hf_semantic_min_ratio,
            ),
            hf_grammar_model_enabled=_bool_env(
                "AWP_HF_GRAMMAR_MODEL_ENABLED",
                cls.hf_grammar_model_enabled,
            ),
            hf_grammar_model=os.environ.get(
                "AWP_HF_GRAMMAR_MODEL",
                cls.hf_grammar_model,
            ).strip()
            or cls.hf_grammar_model,
            hf_correction_local_files_only=_bool_env(
                "AWP_HF_CORRECTION_LOCAL_FILES_ONLY",
                cls.hf_correction_local_files_only,
            ),
            hf_correction_device=os.environ.get(
                "AWP_HF_CORRECTION_DEVICE",
                cls.hf_correction_device,
            ).strip()
            or cls.hf_correction_device,
            hf_correction_num_beams=_int_env(
                "AWP_HF_CORRECTION_NUM_BEAMS",
                cls.hf_correction_num_beams,
            ),
            hf_correction_candidates=_int_env(
                "AWP_HF_CORRECTION_CANDIDATES",
                cls.hf_correction_candidates,
            ),
            hf_correction_max_input_tokens=_int_env(
                "AWP_HF_CORRECTION_MAX_INPUT_TOKENS",
                cls.hf_correction_max_input_tokens,
            ),
            hf_correction_max_new_tokens=_int_env(
                "AWP_HF_CORRECTION_MAX_NEW_TOKENS",
                cls.hf_correction_max_new_tokens,
            ),
            hf_correction_min_confidence=_float_env(
                "AWP_HF_CORRECTION_MIN_CONFIDENCE",
                cls.hf_correction_min_confidence,
            ),
            hf_correction_max_change_ratio=_float_env(
                "AWP_HF_CORRECTION_MAX_CHANGE_RATIO",
                cls.hf_correction_max_change_ratio,
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.correction_confidence_threshold <= 1.0:
            raise ValueError("correction_confidence_threshold must be between 0 and 1")
        if self.max_word_latency_ms <= 0:
            raise ValueError("max_word_latency_ms must be positive")
        if self.max_sentence_latency_ms <= 0:
            raise ValueError("max_sentence_latency_ms must be positive")
        if self.device_profile not in {"laptop", "raspberry_pi"}:
            raise ValueError("device_profile must be 'laptop' or 'raspberry_pi'")
        if self.correction_mode not in {"rules", "contextual", "hf"}:
            raise ValueError("correction_mode must be 'rules', 'contextual', or 'hf'")
        if self.max_correction_candidates <= 0:
            raise ValueError("max_correction_candidates must be positive")
        if self.hf_correction_num_beams <= 0:
            raise ValueError("hf_correction_num_beams must be positive")
        if self.hf_correction_candidates <= 0:
            raise ValueError("hf_correction_candidates must be positive")
        if self.hf_correction_max_input_tokens <= 0:
            raise ValueError("hf_correction_max_input_tokens must be positive")
        if self.hf_correction_max_new_tokens <= 0:
            raise ValueError("hf_correction_max_new_tokens must be positive")
        if not 0.0 <= self.hf_correction_min_confidence <= 1.0:
            raise ValueError("hf_correction_min_confidence must be between 0 and 1")
        if not 0.0 <= self.hf_correction_max_change_ratio <= 1.0:
            raise ValueError("hf_correction_max_change_ratio must be between 0 and 1")
        if not 0.0 <= self.hf_semantic_min_score <= 1.0:
            raise ValueError("hf_semantic_min_score must be between 0 and 1")
        if not 0.0 <= self.hf_semantic_min_margin <= 1.0:
            raise ValueError("hf_semantic_min_margin must be between 0 and 1")
        if self.hf_semantic_min_ratio <= 0.0:
            raise ValueError("hf_semantic_min_ratio must be positive")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def huggingface_cache_dir_from_env(*, models_dir: Optional[Path] = None) -> Path:
    raw_cache = os.environ.get("AWP_HF_CACHE_DIR", "").strip()
    if raw_cache:
        return Path(raw_cache)
    raw_root = os.environ.get("AWP_MODEL_CACHE", "").strip()
    if raw_root:
        return Path(raw_root) / "huggingface"
    root = models_dir or Path("models")
    return root / "cache" / "huggingface"

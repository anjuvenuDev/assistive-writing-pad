"""Runtime settings for laptop and Raspberry Pi execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class RuntimeSettings:
    """Configuration values shared by the pipeline components."""

    confidence_threshold: float = 0.85
    max_sentence_latency_ms: int = 2000
    max_word_latency_ms: int = 500
    models_dir: Path = Path("models")
    data_dir: Path = Path("data")
    api_correction_enabled: bool = False
    device_profile: str = "laptop"
    correction_mode: str = "contextual"
    contextual_model_enabled: bool = False
    contextual_model_name: str = "distilbert/distilbert-base-uncased"
    preload_ocr_model: bool = True
    max_correction_candidates: int = 8
    correction_max_transpositions: int = 2
    correction_confidence_threshold: float = 0.70
    correction_auto_threshold: float = 0.90
    correction_suggestion_threshold: float = 0.65
    correction_dictionary_path: str = ""
    correction_memory_path: Path = Path("data/user_corrections.json")
    debug_correction: bool = False

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        device_profile = os.environ.get("AWP_DEVICE_PROFILE", "laptop").strip() or "laptop"
        correction_mode = os.environ.get("AWP_CORRECTION_MODE", "contextual").strip() or "contextual"
        default_preload = "0" if device_profile == "raspberry_pi" else "1"
        settings = cls(
            confidence_threshold=_float_env("AWP_CONFIDENCE_THRESHOLD", cls.confidence_threshold),
            max_sentence_latency_ms=_int_env(
                "AWP_MAX_SENTENCE_LATENCY_MS", cls.max_sentence_latency_ms
            ),
            max_word_latency_ms=_int_env("AWP_MAX_WORD_LATENCY_MS", cls.max_word_latency_ms),
            models_dir=Path(os.environ.get("AWP_MODELS_DIR", str(cls.models_dir))),
            data_dir=Path(os.environ.get("AWP_DATA_DIR", str(cls.data_dir))),
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
            max_correction_candidates=_int_env(
                "AWP_MAX_CORRECTION_CANDIDATES", cls.max_correction_candidates
            ),
            correction_max_transpositions=_int_env(
                "AWP_CORRECTION_MAX_TRANSPOSITIONS", cls.correction_max_transpositions
            ),
            correction_confidence_threshold=_float_env(
                "AWP_CORRECTION_CONFIDENCE_THRESHOLD",
                cls.correction_confidence_threshold,
            ),
            correction_auto_threshold=_float_env(
                "AWP_CORRECTION_AUTO_THRESHOLD",
                cls.correction_auto_threshold,
            ),
            correction_suggestion_threshold=_float_env(
                "AWP_CORRECTION_SUGGESTION_THRESHOLD",
                cls.correction_suggestion_threshold,
            ),
            correction_dictionary_path=os.environ.get("AWP_CORRECTION_DICTIONARY", "").strip(),
            correction_memory_path=Path(
                os.environ.get("AWP_CORRECTION_MEMORY_PATH", str(cls.correction_memory_path))
            ),
            debug_correction=_bool_env("AWP_DEBUG_CORRECTION", cls.debug_correction),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.correction_confidence_threshold <= 1.0:
            raise ValueError("correction_confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.correction_auto_threshold <= 1.0:
            raise ValueError("correction_auto_threshold must be between 0 and 1")
        if not 0.0 <= self.correction_suggestion_threshold <= 1.0:
            raise ValueError("correction_suggestion_threshold must be between 0 and 1")
        if self.correction_suggestion_threshold > self.correction_auto_threshold:
            raise ValueError("correction_suggestion_threshold must be <= correction_auto_threshold")
        if self.max_word_latency_ms <= 0:
            raise ValueError("max_word_latency_ms must be positive")
        if self.max_sentence_latency_ms <= 0:
            raise ValueError("max_sentence_latency_ms must be positive")
        if self.device_profile not in {"laptop", "raspberry_pi"}:
            raise ValueError("device_profile must be 'laptop' or 'raspberry_pi'")
        if self.correction_mode not in {"rules", "contextual"}:
            raise ValueError("correction_mode must be 'rules' or 'contextual'")
        if self.max_correction_candidates <= 0:
            raise ValueError("max_correction_candidates must be positive")
        if not 0 <= self.correction_max_transpositions <= 2:
            raise ValueError("correction_max_transpositions must be between 0 and 2")


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

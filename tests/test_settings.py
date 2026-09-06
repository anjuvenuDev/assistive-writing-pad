import pytest

from assistive_writing_pad.config.settings import RuntimeSettings


def test_runtime_settings_accept_raspberry_pi_profile() -> None:
    RuntimeSettings(device_profile="raspberry_pi").validate()


def test_runtime_settings_reject_invalid_confidence_threshold() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        RuntimeSettings(confidence_threshold=1.5).validate()


def test_runtime_settings_reject_invalid_correction_mode() -> None:
    with pytest.raises(ValueError, match="correction_mode"):
        RuntimeSettings(correction_mode="cloud_only").validate()


def test_runtime_settings_use_huggingface_correction_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AWP_CORRECTION_MODE", raising=False)

    assert RuntimeSettings.from_env().correction_mode == "hf"


def test_runtime_settings_read_huggingface_correction_flags(monkeypatch) -> None:
    monkeypatch.setenv("AWP_HF_SEMANTIC_MODEL", "distilbert/distilbert-base-uncased")
    monkeypatch.setenv("AWP_HF_SEMANTIC_MIN_RATIO", "2.0")
    monkeypatch.setenv("AWP_HF_GRAMMAR_MODEL", "pszemraj/flan-t5-large-grammar-synthesis")
    monkeypatch.setenv("AWP_HF_CORRECTION_LOCAL_FILES_ONLY", "1")
    monkeypatch.setenv("AWP_HF_CORRECTION_CANDIDATES", "2")

    settings = RuntimeSettings.from_env()

    assert settings.hf_semantic_model == "distilbert/distilbert-base-uncased"
    assert settings.hf_semantic_min_ratio == 2.0
    assert settings.hf_grammar_model == "pszemraj/flan-t5-large-grammar-synthesis"
    assert settings.hf_correction_local_files_only is True
    assert settings.hf_correction_candidates == 2


def test_runtime_settings_disable_contextual_model_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AWP_CONTEXTUAL_MODEL_ENABLED", raising=False)

    assert RuntimeSettings.from_env().contextual_model_enabled is False


def test_runtime_settings_disable_ocr_preload_on_raspberry_pi(monkeypatch) -> None:
    monkeypatch.setenv("AWP_DEVICE_PROFILE", "raspberry_pi")
    monkeypatch.delenv("AWP_PRELOAD_OCR_MODEL", raising=False)

    assert RuntimeSettings.from_env().preload_ocr_model is False


def test_runtime_settings_preload_correction_models_by_laptop_default(monkeypatch) -> None:
    monkeypatch.setenv("AWP_DEVICE_PROFILE", "laptop")
    monkeypatch.delenv("AWP_PRELOAD_CORRECTION_MODELS", raising=False)

    assert RuntimeSettings.from_env().preload_correction_models is True


def test_runtime_settings_disable_correction_preload_on_raspberry_pi(monkeypatch) -> None:
    monkeypatch.setenv("AWP_DEVICE_PROFILE", "raspberry_pi")
    monkeypatch.delenv("AWP_PRELOAD_CORRECTION_MODELS", raising=False)

    assert RuntimeSettings.from_env().preload_correction_models is False


def test_runtime_settings_allow_correction_preload_override(monkeypatch) -> None:
    monkeypatch.setenv("AWP_DEVICE_PROFILE", "raspberry_pi")
    monkeypatch.setenv("AWP_PRELOAD_CORRECTION_MODELS", "1")

    assert RuntimeSettings.from_env().preload_correction_models is True


def test_runtime_settings_read_explicit_huggingface_cache(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "hf-cache"
    monkeypatch.setenv("AWP_HF_CACHE_DIR", str(cache_dir))

    assert RuntimeSettings.from_env().hf_cache_dir == cache_dir


def test_runtime_settings_derive_huggingface_cache_from_model_cache(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "model-cache"
    monkeypatch.delenv("AWP_HF_CACHE_DIR", raising=False)
    monkeypatch.setenv("AWP_MODEL_CACHE", str(cache_root))

    assert RuntimeSettings.from_env().hf_cache_dir == cache_root / "huggingface"


def test_runtime_settings_read_evaluation_capture_flags(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "cases.jsonl"
    monkeypatch.setenv("AWP_EVALUATION_CAPTURE_ENABLED", "1")
    monkeypatch.setenv("AWP_EVALUATION_MANIFEST", str(manifest))

    settings = RuntimeSettings.from_env()

    assert settings.evaluation_capture_enabled is True
    assert settings.evaluation_manifest_path == manifest

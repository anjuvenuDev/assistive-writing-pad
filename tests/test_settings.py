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


def test_runtime_settings_disable_contextual_model_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AWP_CONTEXTUAL_MODEL_ENABLED", raising=False)

    assert RuntimeSettings.from_env().contextual_model_enabled is False


def test_runtime_settings_disable_ocr_preload_on_raspberry_pi(monkeypatch) -> None:
    monkeypatch.setenv("AWP_DEVICE_PROFILE", "raspberry_pi")
    monkeypatch.delenv("AWP_PRELOAD_OCR_MODEL", raising=False)

    assert RuntimeSettings.from_env().preload_ocr_model is False

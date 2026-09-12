from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.recognition.trocr import (
    DEFAULT_TROCR_MODEL,
    TrOCRHandwritingRecognizer,
    _DEFAULT_RENDER_H,
    _DEFAULT_RENDER_W,
    _normalize_requested_mode,
    default_huggingface_cache_dir,
    render_strokes_for_trocr,
)
from assistive_writing_pad.recognition.trocr import _preprocess_image


def test_trocr_renderer_returns_rgb_numpy_line_image() -> None:
    image = render_strokes_for_trocr(
        [
            StrokePoint(x=10, y=20, timestamp_ms=0, pressure=1.0),
            StrokePoint(x=60, y=40, timestamp_ms=16, pressure=1.0),
        ]
    )

    # Shape must match the configured default render canvas.
    assert image.shape == (_DEFAULT_RENDER_H, _DEFAULT_RENDER_W, 3)
    assert image.dtype.name == "uint8"
    assert image.min() == 0
    assert image.max() == 255


def test_trocr_renderer_handles_empty_strokes() -> None:
    image = render_strokes_for_trocr([])

    assert image.shape == (_DEFAULT_RENDER_H, _DEFAULT_RENDER_W, 3)
    assert image.min() == 255


def test_blank_render_is_rejected_before_ocr() -> None:
    image = render_strokes_for_trocr([])

    processed, crop, _ = _preprocess_image(image)

    assert crop.valid is False
    assert processed.shape == image.shape


def test_default_trocr_model_prioritizes_base_handwriting_ocr() -> None:
    recognizer = TrOCRHandwritingRecognizer()

    assert DEFAULT_TROCR_MODEL == "microsoft/trocr-base-handwritten"
    assert recognizer.model_name == "microsoft/trocr-base-handwritten"


def test_trocr_uses_project_huggingface_cache_by_default(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "cache-root"
    monkeypatch.delenv("AWP_HF_CACHE_DIR", raising=False)
    monkeypatch.setenv("AWP_MODEL_CACHE", str(cache_root))

    recognizer = TrOCRHandwritingRecognizer()

    assert default_huggingface_cache_dir() == cache_root / "huggingface"
    assert recognizer.cache_dir == cache_root / "huggingface"


def test_trocr_accepts_explicit_cache_and_local_files_only(tmp_path) -> None:
    cache_dir = tmp_path / "hf"
    recognizer = TrOCRHandwritingRecognizer(
        cache_dir=cache_dir,
        local_files_only=True,
    )

    assert recognizer.cache_dir == cache_dir
    assert recognizer.local_files_only is True


def test_trocr_reads_local_files_only_from_env(monkeypatch) -> None:
    monkeypatch.setenv("AWP_TROCR_LOCAL_FILES_ONLY", "1")

    assert TrOCRHandwritingRecognizer().local_files_only is True


def test_legacy_modes_are_accepted_as_ocr_requests() -> None:
    assert _normalize_requested_mode("auto") == "auto"
    assert _normalize_requested_mode("character") == "character"
    assert _normalize_requested_mode("word") == "word"
    assert _normalize_requested_mode("ocr") == "ocr"


def test_unknown_mode_falls_back_to_ocr() -> None:
    assert _normalize_requested_mode("glyph") == "ocr"

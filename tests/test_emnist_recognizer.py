"""Unit tests for the EMNIST character recognizer.

All tests are CPU-only, fast, and avoid any network access or real model loading.
The CNN model is mocked so no weights need to be downloaded.
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.recognition.emnist import (
    EMNIST_LABELS,
    EMNISTCharacterRecognizer,
    _pixel_classify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stroke(n_points: int = 20) -> List[StrokePoint]:
    """Generate a simple diagonal stroke from (10,10) to (50,50)."""
    pts = []
    for i in range(n_points):
        t = i / max(n_points - 1, 1)
        pts.append(StrokePoint(x=10 + 40 * t, y=10 + 40 * t, timestamp_ms=i * 16, pressure=1.0))
    return pts


def _make_fake_model(logits: List[float] = None) -> MagicMock:
    """Return a mock nn.Module whose forward() returns fixed logits as a tensor.

    logits should have length == len(EMNIST_LABELS) (62).
    If not provided, a default all-zero tensor is used.
    """
    n = len(EMNIST_LABELS)
    if logits is None:
        logits = [0.0] * n
    logit_tensor = torch.tensor([logits], dtype=torch.float32)  # shape (1, 62)
    mock_model = MagicMock()
    mock_model.return_value = logit_tensor
    mock_model.eval.return_value = mock_model
    return mock_model


def _rec_with_model(logits: List[float] = None) -> EMNISTCharacterRecognizer:
    """Create a recognizer whose CNN is pre-loaded with a mock model."""
    rec = EMNISTCharacterRecognizer()
    rec._pipeline = True          # mark as 'loaded'
    rec._use_fallback = False
    rec._model = _make_fake_model(logits)
    return rec


def _make_logits_for(char: str, confidence: float = 0.9) -> List[float]:
    """Build a logit vector that puts `char` at approximately `confidence` after softmax."""
    idx = EMNIST_LABELS.index(char)
    # log(p/(1-p)) ≈ large positive for target, 0 for rest
    import math
    # Use a large logit for the target so softmax ≈ confidence
    target_logit = math.log(confidence / ((1 - confidence) / (len(EMNIST_LABELS) - 1)))
    logits = [0.0] * len(EMNIST_LABELS)
    logits[idx] = target_logit
    return logits


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestEMNISTCharacterRecognizerInit:
    def test_default_model_name(self) -> None:
        rec = EMNISTCharacterRecognizer()
        assert rec.model_name == "awp/emnist-byclass-cnn-v1"

    def test_custom_model_name(self) -> None:
        rec = EMNISTCharacterRecognizer(model_name="some/other-model")
        assert rec.model_name == "some/other-model"

    def test_pipeline_not_loaded_at_init(self) -> None:
        rec = EMNISTCharacterRecognizer()
        assert rec._pipeline is None


# ---------------------------------------------------------------------------
# Empty-input guard
# ---------------------------------------------------------------------------


class TestEMNISTRecognizerEmptyInput:
    def test_empty_strokes_returns_empty_result(self) -> None:
        rec = EMNISTCharacterRecognizer()
        result = rec.recognize([])
        assert result.text == ""
        assert result.confidence == 0.0
        assert result.metadata.get("reason") == "empty_strokes"

    def test_empty_strokes_does_not_load_model(self) -> None:
        rec = EMNISTCharacterRecognizer()
        rec.recognize([])
        assert rec._pipeline is None


# ---------------------------------------------------------------------------
# Core recognition (with mocked CNN)
# ---------------------------------------------------------------------------


class TestEMNISTRecognitionWithMock:
    def test_returns_top_label_as_text(self) -> None:
        rec = _rec_with_model(_make_logits_for("g"))
        result = rec.recognize(_make_stroke())
        assert result.text == "g"

    def test_confidence_matches_top_score(self) -> None:
        rec = _rec_with_model(_make_logits_for("a", confidence=0.92))
        result = rec.recognize(_make_stroke())
        # The softmax of our constructed logits should be close to 0.92
        assert abs(result.confidence - 0.92) < 0.02

    def test_top5_character_confidences_returned(self) -> None:
        rec = _rec_with_model()
        result = rec.recognize(_make_stroke())
        assert len(result.character_confidences) == 5

    def test_character_confidences_sorted_descending(self) -> None:
        rec = _rec_with_model()
        result = rec.recognize(_make_stroke())
        confs = [c.confidence for c in result.character_confidences]
        assert confs == sorted(confs, reverse=True)

    def test_recognizer_field_in_metadata(self) -> None:
        rec = _rec_with_model()
        result = rec.recognize(_make_stroke())
        assert result.metadata.get("recognizer") == "emnist"

    def test_model_name_in_metadata(self) -> None:
        rec = _rec_with_model()
        result = rec.recognize(_make_stroke())
        assert "awp" in result.metadata.get("model", "")

    def test_custom_model_name_in_metadata(self) -> None:
        rec = EMNISTCharacterRecognizer(model_name="test/my-model")
        rec._pipeline = True
        rec._use_fallback = False
        rec._model = _make_fake_model()
        result = rec.recognize(_make_stroke())
        assert "test/my-model" in result.metadata.get("model", "")


# ---------------------------------------------------------------------------
# Fallback (no model available)
# ---------------------------------------------------------------------------


class TestEMNISTFallback:
    def test_fallback_returns_non_empty_result(self) -> None:
        rec = EMNISTCharacterRecognizer()
        rec._pipeline = True
        rec._use_fallback = True
        rec._model = None
        result = rec.recognize(_make_stroke())
        assert result.text != ""

    def test_fallback_metadata_indicates_heuristic(self) -> None:
        rec = EMNISTCharacterRecognizer()
        rec._pipeline = True
        rec._use_fallback = True
        rec._model = None
        result = rec.recognize(_make_stroke())
        assert "heuristic" in result.metadata.get("model", "")

    def test_fallback_returns_5_candidates(self) -> None:
        rec = EMNISTCharacterRecognizer()
        rec._pipeline = True
        rec._use_fallback = True
        rec._model = None
        result = rec.recognize(_make_stroke())
        assert len(result.character_confidences) == 5

    def test_pixel_classify_all_black(self) -> None:
        img = np.zeros((28, 28), dtype=np.float32)
        candidates = _pixel_classify(img)
        assert len(candidates) >= 1  # must return something

    def test_pixel_classify_tall_thin_stroke(self) -> None:
        # Tall narrow stroke should lean towards 'l', '1', etc.
        img = np.zeros((28, 28), dtype=np.float32)
        img[4:24, 13:15] = 1.0  # tall thin column
        candidates = _pixel_classify(img)
        assert len(candidates) >= 1


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


class TestEMNISTLazyLoading:
    @patch("assistive_writing_pad.recognition.emnist.EMNISTCharacterRecognizer._ensure_loaded")
    def test_ensure_loaded_called_on_recognize(self, mock_ensure) -> None:
        rec = EMNISTCharacterRecognizer()
        rec._pipeline = True
        rec._use_fallback = True
        rec._model = None
        rec.recognize(_make_stroke())
        mock_ensure.assert_called_once()

    def test_pipeline_flag_set_after_ensure_loaded(self) -> None:
        rec = EMNISTCharacterRecognizer()
        # Patch _load_model to avoid hitting disk/network
        rec._load_model = lambda: None  # type: ignore
        rec._ensure_loaded()
        assert rec._pipeline is True

    def test_model_not_reloaded_on_second_call(self) -> None:
        rec = EMNISTCharacterRecognizer()
        rec._pipeline = True
        rec._use_fallback = True
        rec._model = None
        load_count = {"n": 0}
        original_load = rec._load_model
        def counting_load():
            load_count["n"] += 1
            return original_load()
        rec._load_model = counting_load  # type: ignore
        rec.recognize(_make_stroke())
        rec.recognize(_make_stroke())
        # _load_model should not be called at all since _pipeline is already True
        assert load_count["n"] == 0

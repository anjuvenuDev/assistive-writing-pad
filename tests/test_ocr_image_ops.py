"""Unit tests for the TrOCR-specific image preprocessing module.

These tests use only numpy and opencv (both already in the dependency list) and
run entirely on CPU, making them fast and Raspberry Pi compatible.

All tests are independent and produce no side-effects on the filesystem.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from assistive_writing_pad.preprocessing.ocr_image_ops import (
    auto_crop_handwriting,
    enhance_for_ocr,
    segment_words_horizontally,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _white_rgb(h: int, w: int) -> np.ndarray:
    """Create a fully white RGB image."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _place_black_rect(image: np.ndarray, y1: int, x1: int, y2: int, x2: int) -> None:
    """Draw a filled black rectangle on an RGB image in-place."""
    image[y1:y2, x1:x2] = 0


# ---------------------------------------------------------------------------
# auto_crop_handwriting
# ---------------------------------------------------------------------------


class TestAutoCrop:
    def test_crops_tightly_around_ink(self) -> None:
        """A small black square inside a large white canvas should be cropped
        to just the square plus the configured padding."""
        h, w = 200, 400
        padding = 10
        image = _white_rgb(h, w)
        # Place a 20x20 black square starting at (80, 100).
        _place_black_rect(image, 80, 100, 100, 120)

        result = auto_crop_handwriting(image, padding=padding)

        # The crop should cover the square plus padding on each side.
        expected_h = (100 - 80) + 2 * padding  # 20 + 20 = 40
        expected_w = (120 - 100) + 2 * padding  # 20 + 20 = 40
        assert result.shape == (expected_h, expected_w, 3), (
            f"Expected crop shape ({expected_h}, {expected_w}, 3), got {result.shape}"
        )

    def test_returns_original_on_empty_image(self) -> None:
        """A fully white image (no ink) should be returned unchanged."""
        image = _white_rgb(100, 200)
        result = auto_crop_handwriting(image, padding=10)
        assert result.shape == image.shape

    def test_clamps_padding_at_borders(self) -> None:
        """Padding that would exceed image boundaries must be clamped (no IndexError)."""
        image = _white_rgb(50, 50)
        # Ink in the top-left corner — padding would go out of bounds without clamping.
        _place_black_rect(image, 0, 0, 5, 5)
        result = auto_crop_handwriting(image, padding=30)
        # Should not raise and result must be a valid sub-image.
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_warns_when_ink_below_5_percent(self, caplog) -> None:
        """Should emit a WARNING log when ink covers less than 5% of the canvas."""
        image = _white_rgb(200, 200)
        # Place a tiny 3x3 ink dot (<< 5% of 200x200 = 40000 px).
        _place_black_rect(image, 100, 100, 103, 103)  # 9 px = 0.02%

        with caplog.at_level(logging.WARNING, logger="assistive_writing_pad.preprocessing.ocr_image_ops"):
            auto_crop_handwriting(image, padding=2)

        assert any("whitespace" in record.message.lower() or "ink covers" in record.message.lower()
                   for record in caplog.records), (
            "Expected a WARNING about sparse ink coverage"
        )

    def test_output_dtype_preserved(self) -> None:
        """Output should always be uint8 RGB."""
        image = _white_rgb(100, 100)
        _place_black_rect(image, 20, 20, 80, 80)
        result = auto_crop_handwriting(image)
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3


# ---------------------------------------------------------------------------
# enhance_for_ocr
# ---------------------------------------------------------------------------


class TestEnhanceForOcr:
    def test_returns_3_channel_rgb_uint8(self) -> None:
        """Output must always be uint8 HxWx3 regardless of input content."""
        image = _white_rgb(128, 384)
        _place_black_rect(image, 40, 50, 90, 330)
        result = enhance_for_ocr(image)
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_output_shape_matches_input(self) -> None:
        """enhance_for_ocr must not change image dimensions."""
        image = _white_rgb(64, 256)
        _place_black_rect(image, 10, 20, 54, 236)
        result = enhance_for_ocr(image)
        assert result.shape == image.shape

    def test_ink_remains_dark_on_white_background(self) -> None:
        """After enhancement the background should be brighter than the ink pixels."""
        image = _white_rgb(100, 100)
        # Solid black bar — should still appear as dark pixels after binarisation.
        _place_black_rect(image, 30, 10, 70, 90)
        result = enhance_for_ocr(image)
        # Background pixels (top rows) should be brighter than ink pixels.
        background_mean = float(result[:20, :, 0].mean())
        ink_mean = float(result[30:70, 10:90, 0].mean())
        assert background_mean > ink_mean, (
            f"Expected background ({background_mean:.1f}) brighter than ink ({ink_mean:.1f})"
        )

    def test_all_white_image_does_not_crash(self) -> None:
        """A completely white image (no ink) should produce a white output without error."""
        image = _white_rgb(64, 128)
        result = enhance_for_ocr(image)
        assert result is not None
        assert result.shape == image.shape


# ---------------------------------------------------------------------------
# segment_words_horizontally
# ---------------------------------------------------------------------------


class TestSegmentWordsHorizontally:
    def test_splits_two_blobs_with_wide_gap(self) -> None:
        """Two ink regions separated by a gap wider than min_gap_px should produce 2 regions."""
        image = _white_rgb(64, 200)
        # Left word: columns 10-50
        _place_black_rect(image, 20, 10, 50, 50)
        # Right word: columns 150-190
        _place_black_rect(image, 20, 150, 50, 190)
        # Gap between 50 and 150 = 100 all-white columns >> min_gap_px=20

        regions = segment_words_horizontally(image, min_gap_px=20)
        assert len(regions) == 2, f"Expected 2 word regions, got {len(regions)}"

    def test_no_split_when_gap_is_too_small(self) -> None:
        """Two blobs with a gap smaller than min_gap_px should return 1 region."""
        image = _white_rgb(64, 100)
        _place_black_rect(image, 20, 5, 50, 45)
        _place_black_rect(image, 20, 50, 50, 95)
        # Gap between 45 and 50 = 5 columns < min_gap_px=20

        regions = segment_words_horizontally(image, min_gap_px=20)
        assert len(regions) == 1, f"Expected 1 region, got {len(regions)}"

    def test_full_image_returned_when_no_gap(self) -> None:
        """Solid ink covering the whole width should return a single region."""
        image = _white_rgb(64, 100)
        _place_black_rect(image, 10, 0, 54, 100)

        regions = segment_words_horizontally(image, min_gap_px=5)
        assert len(regions) == 1
        # The single region should span the full width.
        assert regions[0].shape[1] == 100

    def test_region_heights_match_input(self) -> None:
        """All returned regions must have the same height as the input image."""
        image = _white_rgb(80, 300)
        _place_black_rect(image, 10, 20, 70, 100)
        _place_black_rect(image, 10, 200, 70, 280)

        regions = segment_words_horizontally(image, min_gap_px=30)
        for i, region in enumerate(regions):
            assert region.shape[0] == 80, (
                f"Region {i} height {region.shape[0]} != input height 80"
            )

    def test_three_words_produce_three_regions(self) -> None:
        """Three well-separated blobs should produce exactly 3 word regions."""
        image = _white_rgb(64, 360)
        _place_black_rect(image, 15, 10, 50, 80)     # word 1
        _place_black_rect(image, 15, 150, 50, 220)   # word 2
        _place_black_rect(image, 15, 290, 50, 350)   # word 3

        regions = segment_words_horizontally(image, min_gap_px=20)
        assert len(regions) == 3, f"Expected 3 regions, got {len(regions)}"

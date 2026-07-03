from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.recognition.trocr import (
    _clean_ocr_text,
    _dot_above_stem_hint,
    _looks_like_single_character_input,
    segment_strokes_into_lines,
    _shape_hint_for_single_character,
    _single_character_guess,
)


def test_clean_ocr_text_removes_hash_noise() -> None:
    assert _clean_ocr_text("###") == ""


def test_clean_ocr_text_strips_digit_prefix_from_letters() -> None:
    assert _clean_ocr_text("2b 2c") == "b c"


def test_clean_ocr_text_removes_short_numeric_noise_tokens() -> None:
    assert _clean_ocr_text("2258 b") == "b"


def test_clean_ocr_text_keeps_regular_words() -> None:
    assert _clean_ocr_text("hello world") == "hello world"


def test_single_character_guess_collapses_multiword_noise() -> None:
    assert _single_character_guess("alder of death") == "d"


def test_single_character_guess_preserves_single_letter() -> None:
    assert _single_character_guess("b") == "b"


def test_single_character_guess_maps_digit_noise_to_letter() -> None:
    assert _single_character_guess("0 0") == "o"


def test_looks_like_single_character_input_for_small_strokes() -> None:
    stroke_groups = [
        [
            StrokePoint(x=10, y=10, timestamp_ms=0, pressure=1.0),
            StrokePoint(x=20, y=20, timestamp_ms=16, pressure=1.0),
        ]
    ]
    assert _looks_like_single_character_input(stroke_groups) is True


def test_looks_like_single_character_input_rejects_large_input() -> None:
    long_stroke = [
        StrokePoint(x=float(i), y=float(i % 30), timestamp_ms=i, pressure=1.0)
        for i in range(260)
    ]
    assert _looks_like_single_character_input([long_stroke]) is False


def test_shape_hint_detects_h_from_tall_stem_and_right_leg() -> None:
    h_stroke = [
        StrokePoint(x=10, y=2, timestamp_ms=0, pressure=1.0),
        StrokePoint(x=10, y=10, timestamp_ms=10, pressure=1.0),
        StrokePoint(x=10, y=22, timestamp_ms=20, pressure=1.0),
        StrokePoint(x=10, y=36, timestamp_ms=30, pressure=1.0),
        StrokePoint(x=12, y=30, timestamp_ms=40, pressure=1.0),
        StrokePoint(x=18, y=24, timestamp_ms=50, pressure=1.0),
        StrokePoint(x=24, y=24, timestamp_ms=60, pressure=1.0),
        StrokePoint(x=26, y=30, timestamp_ms=70, pressure=1.0),
        StrokePoint(x=26, y=36, timestamp_ms=80, pressure=1.0),
    ]
    assert _shape_hint_for_single_character([h_stroke], "a") == "h"


def test_shape_hint_does_not_force_non_h_shapes() -> None:
    a_like_stroke = [
        StrokePoint(x=10, y=26, timestamp_ms=0, pressure=1.0),
        StrokePoint(x=14, y=14, timestamp_ms=10, pressure=1.0),
        StrokePoint(x=20, y=10, timestamp_ms=20, pressure=1.0),
        StrokePoint(x=26, y=16, timestamp_ms=30, pressure=1.0),
        StrokePoint(x=26, y=24, timestamp_ms=40, pressure=1.0),
        StrokePoint(x=20, y=30, timestamp_ms=50, pressure=1.0),
        StrokePoint(x=14, y=28, timestamp_ms=60, pressure=1.0),
        StrokePoint(x=10, y=22, timestamp_ms=70, pressure=1.0),
    ]
    assert _shape_hint_for_single_character([a_like_stroke], "a") is None


def test_dot_above_stem_hint_detects_i_shape() -> None:
    dot = [
        StrokePoint(x=20, y=6, timestamp_ms=0, pressure=1.0),
        StrokePoint(x=21, y=7, timestamp_ms=8, pressure=1.0),
        StrokePoint(x=20, y=8, timestamp_ms=16, pressure=1.0),
    ]
    stem = [
        StrokePoint(x=20, y=16, timestamp_ms=24, pressure=1.0),
        StrokePoint(x=20, y=24, timestamp_ms=32, pressure=1.0),
        StrokePoint(x=20, y=32, timestamp_ms=40, pressure=1.0),
        StrokePoint(x=22, y=36, timestamp_ms=48, pressure=1.0),
    ]
    assert _dot_above_stem_hint([dot, stem]) == "i"


def test_shape_hint_detects_j_from_dot_and_descender_hook() -> None:
    dot = [
        StrokePoint(x=20, y=6, timestamp_ms=0, pressure=1.0),
        StrokePoint(x=21, y=7, timestamp_ms=8, pressure=1.0),
        StrokePoint(x=20, y=8, timestamp_ms=16, pressure=1.0),
    ]
    j_stem = [
        StrokePoint(x=21, y=16, timestamp_ms=24, pressure=1.0),
        StrokePoint(x=21, y=26, timestamp_ms=32, pressure=1.0),
        StrokePoint(x=21, y=36, timestamp_ms=40, pressure=1.0),
        StrokePoint(x=20, y=44, timestamp_ms=48, pressure=1.0),
        StrokePoint(x=17, y=48, timestamp_ms=56, pressure=1.0),
    ]
    assert _shape_hint_for_single_character([dot, j_stem], "o") == "j"


def test_shape_hint_detects_r_from_open_stem_shape() -> None:
    r_like = [
        StrokePoint(x=10, y=8, timestamp_ms=0, pressure=1.0),
        StrokePoint(x=10, y=14, timestamp_ms=8, pressure=1.0),
        StrokePoint(x=10, y=22, timestamp_ms=16, pressure=1.0),
        StrokePoint(x=10, y=30, timestamp_ms=24, pressure=1.0),
        StrokePoint(x=10, y=36, timestamp_ms=32, pressure=1.0),
        StrokePoint(x=12, y=10, timestamp_ms=40, pressure=1.0),
        StrokePoint(x=16, y=8, timestamp_ms=48, pressure=1.0),
        StrokePoint(x=20, y=8, timestamp_ms=56, pressure=1.0),
    ]
    assert _shape_hint_for_single_character([r_like], "o") == "r"


def test_shape_hint_detects_t_from_wide_top_bar_and_tall_stem() -> None:
    t_like = [
        StrokePoint(x=18, y=8, timestamp_ms=0, pressure=1.0),
        StrokePoint(x=14, y=10, timestamp_ms=8, pressure=1.0),
        StrokePoint(x=10, y=13, timestamp_ms=16, pressure=1.0),
        StrokePoint(x=18, y=9, timestamp_ms=24, pressure=1.0),
        StrokePoint(x=24, y=9, timestamp_ms=32, pressure=1.0),
        StrokePoint(x=30, y=10, timestamp_ms=40, pressure=1.0),
        StrokePoint(x=18, y=12, timestamp_ms=48, pressure=1.0),
        StrokePoint(x=18, y=20, timestamp_ms=56, pressure=1.0),
        StrokePoint(x=18, y=30, timestamp_ms=64, pressure=1.0),
        StrokePoint(x=18, y=38, timestamp_ms=72, pressure=1.0),
    ]
    assert _shape_hint_for_single_character([t_like], "o") == "t"


def test_dot_above_stem_hint_ignores_non_dotted_shapes() -> None:
    stroke_a = [
        StrokePoint(x=10, y=10, timestamp_ms=0, pressure=1.0),
        StrokePoint(x=15, y=20, timestamp_ms=16, pressure=1.0),
        StrokePoint(x=20, y=28, timestamp_ms=32, pressure=1.0),
    ]
    stroke_b = [
        StrokePoint(x=20, y=28, timestamp_ms=48, pressure=1.0),
        StrokePoint(x=25, y=20, timestamp_ms=64, pressure=1.0),
        StrokePoint(x=30, y=12, timestamp_ms=80, pressure=1.0),
    ]
    assert _dot_above_stem_hint([stroke_a, stroke_b]) is None


def test_line_segmentation_would_split_dotted_i_without_single_character_override() -> None:
    dot = [
        StrokePoint(x=20, y=6, timestamp_ms=0, pressure=1.0),
        StrokePoint(x=21, y=7, timestamp_ms=8, pressure=1.0),
    ]
    stem = [
        StrokePoint(x=20, y=16, timestamp_ms=24, pressure=1.0),
        StrokePoint(x=20, y=24, timestamp_ms=32, pressure=1.0),
        StrokePoint(x=20, y=32, timestamp_ms=40, pressure=1.0),
    ]
    lines = segment_strokes_into_lines([dot, stem], gap_threshold=8.0)
    assert len(lines) == 2
    assert _looks_like_single_character_input([dot, stem]) is True

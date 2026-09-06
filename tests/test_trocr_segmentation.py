import json

from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.recognition.trocr import (
    TrOCRHandwritingRecognizer,
    _OCRCandidate,
    _looks_like_single_character_input,
    _single_character_candidates,
    segment_strokes_into_lines,
    segment_strokes_into_words,
)


def _stroke(points):
    return [
        StrokePoint(x=float(x), y=float(y), timestamp_ms=index * 16, pressure=1.0)
        for index, (x, y) in enumerate(points)
    ]


def test_default_line_segmentation_keeps_dotted_i_together() -> None:
    dot = _stroke([(20, 6), (21, 7), (20, 8)])
    stem = _stroke([(20, 16), (20, 24), (20, 32)])

    lines = segment_strokes_into_lines([dot, stem])

    assert len(lines) == 1
    assert len(lines[0]) == 2


def test_word_segmentation_splits_large_horizontal_space() -> None:
    left_a = _stroke([(0, 10), (10, 30)])
    left_b = _stroke([(15, 10), (25, 30)])
    right = _stroke([(85, 10), (95, 30)])

    words = segment_strokes_into_words([left_a, left_b, right])

    assert [len(word) for word in words] == [2, 1]


def test_word_segmentation_keeps_crossed_letter_strokes_together() -> None:
    crossbar = _stroke([(30, 20), (70, 20)])
    stem = _stroke([(50, 5), (50, 55)])

    words = segment_strokes_into_words([stem, crossbar])

    assert len(words) == 1
    assert len(words[0]) == 2


def test_single_character_gate_rejects_short_wide_words() -> None:
    word_like = [
        _stroke([(0, 10), (10, 30)]),
        _stroke([(60, 10), (70, 30)]),
        _stroke([(120, 10), (130, 30)]),
    ]

    assert _looks_like_single_character_input(word_like) is False


def test_single_character_candidates_promote_shape_hint() -> None:
    h_stroke = _stroke(
        [
            (10, 2),
            (10, 10),
            (10, 22),
            (10, 36),
            (12, 30),
            (18, 24),
            (24, 24),
            (26, 30),
            (26, 36),
        ]
    )

    text, confidence, candidates, reason = _single_character_candidates(
        [h_stroke],
        [_OCRCandidate(text="a", confidence=0.70, raw_text="a")],
    )

    assert text == "h"
    assert confidence >= 0.82
    assert reason == "shape_hint"
    assert any(candidate.text == "a" for candidate in candidates)


def test_single_character_candidates_promote_h_when_ocr_reads_s() -> None:
    h_stroke = _stroke(
        [
            (10, 2),
            (10, 10),
            (10, 22),
            (10, 36),
            (12, 30),
            (18, 24),
            (24, 24),
            (26, 30),
            (26, 36),
        ]
    )

    text, confidence, candidates, reason = _single_character_candidates(
        [h_stroke],
        [_OCRCandidate(text="s", confidence=0.70, raw_text="s")],
    )

    assert text == "h"
    assert confidence >= 0.82
    assert reason == "shape_hint"
    assert any(candidate.text == "s" for candidate in candidates)


def test_single_character_candidates_use_dot_hint_for_i() -> None:
    dot = _stroke([(20, 6), (21, 7), (20, 8)])
    stem = _stroke([(20, 16), (20, 24), (20, 32), (22, 36)])

    text, _confidence, _candidates, reason = _single_character_candidates(
        [dot, stem],
        [_OCRCandidate(text="0 0", confidence=0.62, raw_text="0 0")],
    )

    assert text == "i"
    assert reason == "shape_hint"


class StubTrOCRRecognizer(TrOCRHandwritingRecognizer):
    def __init__(self, outputs):
        super().__init__(num_beams=3, num_return_sequences=3)
        self.outputs = list(outputs)

    def _ensure_loaded(self) -> None:
        return None

    def _run_ocr(self, image):
        del image
        return self.outputs.pop(0)


def test_recognizer_segments_words_and_returns_text_alternatives() -> None:
    left = _stroke([(0, 10), (10, 30), (20, 10)])
    right = _stroke([(90, 10), (100, 30), (110, 10)])
    recognizer = StubTrOCRRecognizer(
        [
            [
                _OCRCandidate(text="cat", confidence=0.91, raw_text="cat"),
                _OCRCandidate(text="cot", confidence=0.74, raw_text="cot"),
            ],
            [
                _OCRCandidate(text="dog", confidence=0.88, raw_text="dog"),
                _OCRCandidate(text="dig", confidence=0.70, raw_text="dig"),
            ],
        ]
    )

    result = recognizer.recognize_stroke_groups([left, right], mode="auto")
    line_results = json.loads(result.metadata["line_results"])
    top = json.loads(result.metadata["top3"])

    assert result.text == "cat dog"
    assert result.confidence == 0.895
    assert result.metadata["mode"] == "word"
    assert line_results[0]["segmentation"] == "stroke_word"
    assert line_results[0]["word_count"] == 2
    assert top[0] == ["cat dog", 0.895]
    assert ["cot dog", 0.81] in top
    assert ["cat dig", 0.805] in top

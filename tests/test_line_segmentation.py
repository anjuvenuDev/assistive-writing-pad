from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.recognition.trocr import segment_strokes_into_lines


def test_segment_strokes_into_lines_splits_by_vertical_gap() -> None:
    strokes = [
        [StrokePoint(x=0, y=10, timestamp_ms=0), StrokePoint(x=10, y=12, timestamp_ms=10)],
        [StrokePoint(x=12, y=14, timestamp_ms=20), StrokePoint(x=20, y=11, timestamp_ms=30)],
        [StrokePoint(x=0, y=120, timestamp_ms=40), StrokePoint(x=10, y=122, timestamp_ms=50)],
    ]

    lines = segment_strokes_into_lines(strokes, gap_threshold=40.0)

    assert len(lines) == 2
    assert len(lines[0]) == 2
    assert len(lines[1]) == 1

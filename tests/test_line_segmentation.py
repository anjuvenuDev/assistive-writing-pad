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


def test_temporal_line_segmentation_ignores_descender_overlap() -> None:
    strokes = [
        [StrokePoint(x=10, y=20, timestamp_ms=0), StrokePoint(x=80, y=22, timestamp_ms=1)],
        [StrokePoint(x=90, y=18, timestamp_ms=2), StrokePoint(x=160, y=85, timestamp_ms=3)],
        [StrokePoint(x=12, y=78, timestamp_ms=4), StrokePoint(x=75, y=80, timestamp_ms=5)],
        [StrokePoint(x=90, y=76, timestamp_ms=6), StrokePoint(x=150, y=82, timestamp_ms=7)],
    ]

    lines = segment_strokes_into_lines(strokes)

    assert [len(line) for line in lines] == [2, 2]


def test_temporal_line_segmentation_does_not_split_late_mark_in_same_line() -> None:
    strokes = [
        [StrokePoint(x=10, y=20, timestamp_ms=0), StrokePoint(x=80, y=22, timestamp_ms=1)],
        [StrokePoint(x=90, y=18, timestamp_ms=2), StrokePoint(x=160, y=85, timestamp_ms=3)],
        [StrokePoint(x=125, y=55, timestamp_ms=4), StrokePoint(x=130, y=60, timestamp_ms=5)],
    ]

    lines = segment_strokes_into_lines(strokes)

    assert len(lines) == 1


def test_temporal_line_segmentation_accepts_centered_second_line() -> None:
    strokes = [
        [StrokePoint(x=10, y=20, timestamp_ms=0), StrokePoint(x=220, y=24, timestamp_ms=1)],
        [StrokePoint(x=250, y=18, timestamp_ms=2), StrokePoint(x=400, y=25, timestamp_ms=3)],
        [StrokePoint(x=145, y=100, timestamp_ms=4), StrokePoint(x=360, y=105, timestamp_ms=5)],
    ]

    lines = segment_strokes_into_lines(strokes)

    assert [len(line) for line in lines] == [2, 1]

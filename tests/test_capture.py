from pathlib import Path

import pytest

from assistive_writing_pad.capture.huion_probe import InputDeviceSummary, find_huion_device
from assistive_writing_pad.capture.simulator import StrokeSimulator
from assistive_writing_pad.capture.stroke_io import load_strokes, save_strokes, strokes_from_record


def test_simulator_generates_monotonic_stroke_points() -> None:
    points = StrokeSimulator().write_text("ab")

    assert len(points) == 6
    assert all(point.pressure > 0 for point in points)
    assert [point.timestamp_ms for point in points] == sorted(
        point.timestamp_ms for point in points
    )


def test_stroke_record_round_trip(tmp_path: Path) -> None:
    points = StrokeSimulator().write_text("hi")
    path = tmp_path / "strokes.json"

    save_strokes(path, points, source="test")
    loaded = load_strokes(path)

    assert loaded == points


def test_stroke_record_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="unsupported stroke schema"):
        strokes_from_record({"schema_version": 999, "points": []})


def test_huion_probe_finds_likely_device() -> None:
    devices = [
        InputDeviceSummary("/dev/input/event0", "Keyboard", "", False),
        InputDeviceSummary("/dev/input/event7", "HUION Huion Tablet_HS64", "", True),
    ]

    assert find_huion_device(devices) == "/dev/input/event7"

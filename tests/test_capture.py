from pathlib import Path
from types import SimpleNamespace

import pytest

import assistive_writing_pad.capture.huion_reader as huion_reader
import assistive_writing_pad.capture.huion_probe as huion_probe
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


def test_huion_probe_reads_device_axis_ranges(monkeypatch) -> None:
    class FakeDevice:
        def absinfo(self, code):
            return SimpleNamespace(min=10 if code == 0 else 20, max=32010 if code == 0 else 20420)

    fake_evdev = SimpleNamespace(
        InputDevice=lambda path: FakeDevice(),
        ecodes=SimpleNamespace(ABS_X=0, ABS_Y=1),
    )
    monkeypatch.setattr(huion_probe, "_load_evdev", lambda: fake_evdev)

    assert huion_probe.huion_axis_ranges("/dev/input/event4") == (
        (10.0, 32010.0),
        (20.0, 20420.0),
    )


def test_huion_reader_groups_touch_events_without_hover_points(monkeypatch) -> None:
    events = [
        SimpleNamespace(type=3, code=0, value=100),
        SimpleNamespace(type=3, code=1, value=200),
        SimpleNamespace(type=1, code=320, value=1),
        SimpleNamespace(type=1, code=330, value=1),
        SimpleNamespace(type=3, code=24, value=80),
        SimpleNamespace(type=0, code=0, value=0),
        SimpleNamespace(type=3, code=0, value=120),
        SimpleNamespace(type=3, code=24, value=80),
        SimpleNamespace(type=0, code=0, value=0),
        SimpleNamespace(type=3, code=24, value=0),
        SimpleNamespace(type=0, code=0, value=0),
        SimpleNamespace(type=1, code=330, value=0),
        SimpleNamespace(type=0, code=0, value=0),
        SimpleNamespace(type=3, code=0, value=500),
        SimpleNamespace(type=0, code=0, value=0),
    ]

    class FakeDevice:
        def read_loop(self):
            return iter(events)

    fake_evdev = SimpleNamespace(
        InputDevice=lambda path: FakeDevice(),
        ecodes=SimpleNamespace(
            EV_ABS=3,
            EV_KEY=1,
            ABS_X=0,
            ABS_Y=1,
            ABS_PRESSURE=24,
            BTN_TOUCH=330,
            BTN_TOOL_PEN=320,
            EV_SYN=0,
        ),
    )
    monkeypatch.setattr(huion_reader, "_load_evdev", lambda: fake_evdev)

    result = list(huion_reader.HuionEventReader("/dev/input/event4").iter_stroke_events())

    assert [event["type"] for event in result] == [
        "stroke_start",
        "stroke_point",
        "stroke_point",
        "stroke_point",
        "stroke_end",
    ]
    assert result[1]["x"] == 100.0
    assert result[1]["y"] == 200.0
    assert result[1]["pressure"] == 80.0

"""Huion/Linux event reader with contact-aware stroke boundaries."""

from __future__ import annotations

import time
from dataclasses import dataclass
import select
import logging
from typing import Dict, Iterator, List, Optional

from assistive_writing_pad.contracts import StrokePoint

logger = logging.getLogger(__name__)


@dataclass
class HuionEventReader:
    device_path: str

    def iter_points(self) -> Iterator[StrokePoint]:
        evdev = _load_evdev()
        device = evdev.InputDevice(self.device_path)

        x: Optional[float] = None
        y: Optional[float] = None
        pressure = 0.0
        started_at = time.monotonic()

        for event in device.read_loop():
            if event.type != evdev.ecodes.EV_ABS:
                continue

            if event.code == evdev.ecodes.ABS_X:
                x = float(event.value)
            elif event.code == evdev.ecodes.ABS_Y:
                y = float(event.value)
            elif event.code == evdev.ecodes.ABS_PRESSURE:
                pressure = float(event.value)

            if x is not None and y is not None:
                yield StrokePoint(
                    x=x,
                    y=y,
                    pressure=pressure,
                    timestamp_ms=int((time.monotonic() - started_at) * 1000),
                )

    def iter_stroke_events(self) -> Iterator[Dict[str, object]]:
        """Yield contact strokes, excluding hover movement."""
        evdev = _load_evdev()
        device = evdev.InputDevice(self.device_path)

        x: Optional[float] = None
        y: Optional[float] = None
        pressure = 0.0
        touch_state = False
        tool_state: Optional[bool] = None
        drawing = False
        started_at = time.monotonic()
        last_point_log = 0.0

        for event in device.read_loop():
            if event.type == evdev.ecodes.EV_ABS:
                if event.code == evdev.ecodes.ABS_X:
                    x = float(event.value)
                elif event.code == evdev.ecodes.ABS_Y:
                    y = float(event.value)
                elif event.code == evdev.ecodes.ABS_PRESSURE:
                    pressure = float(event.value)
            elif event.type == evdev.ecodes.EV_KEY and event.code == evdev.ecodes.BTN_TOUCH:
                touch_state = bool(event.value)
            elif event.type == evdev.ecodes.EV_KEY and event.code == evdev.ecodes.BTN_TOOL_PEN:
                tool_state = bool(event.value)
            elif event.type != evdev.ecodes.EV_SYN:
                continue

            if event.type != evdev.ecodes.EV_SYN:
                continue

            # BTN_TOUCH is the contact source of truth. Pressure can lag or
            # briefly report zero on Huion devices while the pen is down.
            now = time.monotonic()
            should_log_point = False
            if event.type == evdev.ecodes.EV_SYN and (
                x is not None or y is not None or touch_state or tool_state is not None
            ):
                if now - last_point_log >= 0.25:
                    logger.info(
                        "HUION EVENT: x=%s y=%s pressure=%s touch=%s tool=%s",
                        x, y, pressure, touch_state, tool_state,
                    )
                    last_point_log = now
                    should_log_point = True

            # BTN_TOUCH is the contact source of truth. Some HS64 interfaces
            # report BTN_TOOL_PEN proximity separately or late.
            is_drawing = touch_state
            if is_drawing and not drawing:
                drawing = True
                logger.info("HUION PEN DOWN")
                yield {"type": "stroke_start"}
            if drawing and is_drawing and x is not None and y is not None:
                if should_log_point:
                    logger.info("HUION POINT x=%s y=%s", x, y)
                yield {
                    "type": "stroke_point",
                    "x": x,
                    "y": y,
                    "pressure": pressure,
                    "timestamp_ms": int((time.monotonic() - started_at) * 1000),
                }
            elif drawing and not is_drawing:
                drawing = False
                logger.info("HUION PEN UP")
                yield {"type": "stroke_end"}

    def capture_strokes(
        self,
        *,
        duration_seconds: float = 15.0,
        idle_timeout_seconds: float = 2.0,
    ) -> List[List[StrokePoint]]:
        """Capture one pen session, grouping points while the pen is down."""
        evdev = _load_evdev()
        device = evdev.InputDevice(self.device_path)
        started_at = time.monotonic()
        last_event_at = started_at
        last_point_at: Optional[float] = None
        x: Optional[float] = None
        y: Optional[float] = None
        pressure = 0.0
        strokes: List[List[StrokePoint]] = []
        current: List[StrokePoint] = []

        while time.monotonic() - started_at < duration_seconds:
            remaining = duration_seconds - (time.monotonic() - started_at)
            ready, _, _ = select.select([device.fd], [], [], min(0.25, remaining))
            if not ready:
                if current and last_point_at is not None:
                    if time.monotonic() - last_point_at >= idle_timeout_seconds:
                        strokes.append(current)
                        current = []
                if (
                    not current
                    and strokes
                    and time.monotonic() - last_event_at >= idle_timeout_seconds
                ):
                    break
                continue

            for event in device.read():
                last_event_at = time.monotonic()
                if event.type != evdev.ecodes.EV_ABS:
                    continue
                if event.code == evdev.ecodes.ABS_X:
                    x = float(event.value)
                elif event.code == evdev.ecodes.ABS_Y:
                    y = float(event.value)
                elif event.code == evdev.ecodes.ABS_PRESSURE:
                    pressure = float(event.value)

                if x is None or y is None:
                    continue

                point = StrokePoint(
                    x=x,
                    y=y,
                    pressure=pressure,
                    timestamp_ms=int((time.monotonic() - started_at) * 1000),
                )
                if pressure > 0:
                    if not current:
                        current = []
                    current.append(point)
                    last_point_at = time.monotonic()
                elif current:
                    strokes.append(current)
                    current = []
                    last_point_at = None

        if current:
            strokes.append(current)
        return [stroke for stroke in strokes if stroke]


def _load_evdev():
    try:
        import evdev
    except ImportError as exc:
        raise RuntimeError(
            "evdev is not installed. Install the hardware extra with "
            "`pip install -e '.[hardware]'` on Linux."
        ) from exc
    return evdev

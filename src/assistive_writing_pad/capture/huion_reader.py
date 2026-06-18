"""Minimal Huion/Linux event reader.

The reader maps absolute x/y/pressure events into StrokePoint samples. It is
kept small for Phase 2 and will be expanded after hardware validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

from assistive_writing_pad.contracts import StrokePoint


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


def _load_evdev():
    try:
        import evdev
    except ImportError as exc:
        raise RuntimeError(
            "evdev is not installed. Install the hardware extra with "
            "`pip install -e '.[hardware]'` on Linux."
        ) from exc
    return evdev

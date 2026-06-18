"""Probe Huion-compatible Linux input devices.

This module intentionally imports evdev lazily so laptop development and tests do
not require tablet dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class InputDeviceSummary:
    path: str
    name: str
    physical_location: str
    is_likely_huion: bool


def list_input_devices() -> List[InputDeviceSummary]:
    evdev = _load_evdev()
    summaries: List[InputDeviceSummary] = []

    for path in evdev.list_devices():
        device = evdev.InputDevice(path)
        name = device.name or ""
        summaries.append(
            InputDeviceSummary(
                path=path,
                name=name,
                physical_location=device.phys or "",
                is_likely_huion=_looks_like_huion(name),
            )
        )

    return summaries


def find_huion_device(devices: Optional[Iterable[InputDeviceSummary]] = None) -> Optional[str]:
    candidates = list(devices) if devices is not None else list_input_devices()
    for device in candidates:
        if device.is_likely_huion:
            return device.path
    return None


def main() -> None:
    try:
        devices = list_input_devices()
    except RuntimeError as exc:
        print(str(exc))
        return

    if not devices:
        print("No Linux input devices found.")
        return

    for device in devices:
        marker = "*" if device.is_likely_huion else "-"
        print(f"{marker} {device.path} | {device.name} | {device.physical_location}")

    huion_path = find_huion_device(devices)
    if huion_path is None:
        print("No likely Huion tablet detected.")
    else:
        print(f"Likely Huion tablet: {huion_path}")


def _looks_like_huion(name: str) -> bool:
    normalized = name.lower()
    return "huion" in normalized or "hs64" in normalized


def _load_evdev():
    try:
        import evdev
    except ImportError as exc:
        raise RuntimeError(
            "evdev is not installed. Install the hardware extra with "
            "`pip install -e '.[hardware]'` on Linux."
        ) from exc
    return evdev


if __name__ == "__main__":
    main()

"""JSON recording and playback for tablet stroke data."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from assistive_writing_pad.contracts import StrokePoint

SCHEMA_VERSION = 1


def strokes_to_record(points: Sequence[StrokePoint], source: str = "unknown") -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "points": [asdict(point) for point in points],
    }


def save_strokes(path: Path, points: Sequence[StrokePoint], source: str = "unknown") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(strokes_to_record(points, source), indent=2), encoding="utf-8")


def load_strokes(path: Path) -> List[StrokePoint]:
    record = json.loads(path.read_text(encoding="utf-8"))
    return strokes_from_record(record)


def strokes_from_record(record: Dict[str, Any]) -> List[StrokePoint]:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported stroke schema version: {record.get('schema_version')}")

    points = record.get("points")
    if not isinstance(points, list):
        raise ValueError("stroke record must contain a points list")

    return [_point_from_json(point) for point in points]


def _point_from_json(point: Dict[str, Any]) -> StrokePoint:
    return StrokePoint(
        x=float(point["x"]),
        y=float(point["y"]),
        timestamp_ms=int(point["timestamp_ms"]),
        pressure=float(point.get("pressure", 0.0)),
    )

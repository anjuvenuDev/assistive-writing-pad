"""Utilities for building evaluation manifests from captured stroke payloads."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Sequence

from assistive_writing_pad.contracts import StrokePoint
from assistive_writing_pad.eval.recognition_eval import parse_stroke_groups

CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def build_end_to_end_case_record(
    *,
    case_id: str,
    category: str,
    expected: str,
    stroke_payload: Dict[str, Any],
    expected_recognized: Optional[str] = None,
    notes: str = "",
) -> Dict[str, Any]:
    clean_case_id = validate_manifest_text("id", case_id)
    if not CASE_ID_RE.match(clean_case_id):
        raise ValueError("id must start with a letter or number and contain only A-Z, 0-9, _, ., or -")

    clean_category = validate_manifest_text("category", category)
    clean_expected = validate_manifest_text("expected", expected)
    stroke_groups = stroke_groups_from_capture_payload(stroke_payload)

    record: Dict[str, Any] = {
        "id": clean_case_id,
        "category": clean_category,
        "expected": clean_expected,
        "strokes": stroke_groups_to_jsonable(stroke_groups),
    }
    if expected_recognized is not None:
        record["expected_recognized"] = validate_manifest_text(
            "expected_recognized",
            expected_recognized,
        )
    clean_notes = notes.strip()
    if clean_notes:
        record["notes"] = clean_notes
    return record


def stroke_groups_from_capture_payload(
    stroke_payload: Dict[str, Any],
) -> Sequence[Sequence[StrokePoint]]:
    raw_strokes = stroke_payload.get("strokes")
    if raw_strokes is None and isinstance(stroke_payload.get("points"), list):
        raw_strokes = [stroke_payload["points"]]
    if raw_strokes is None:
        raise ValueError("stroke payload must contain strokes or points")
    return parse_stroke_groups(raw_strokes)


def stroke_groups_to_jsonable(
    stroke_groups: Sequence[Sequence[StrokePoint]],
) -> Sequence[Sequence[Dict[str, float]]]:
    return tuple(tuple(asdict(point) for point in stroke) for stroke in stroke_groups)


def append_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")))
        handle.write("\n")


def validate_manifest_text(field: str, value: str) -> str:
    cleaned = " ".join(str(value).strip().split())
    if not cleaned:
        raise ValueError(f"{field} must not be empty")
    return cleaned

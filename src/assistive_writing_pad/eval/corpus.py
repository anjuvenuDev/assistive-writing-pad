"""Utilities for building evaluation manifests from captured stroke payloads."""

from __future__ import annotations

from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlencode
from urllib.request import urlopen

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
    source: str = "manual",
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
        "source": validate_manifest_text("source", source),
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
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing_records = [
            json.loads(line)
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if any(existing.get("id") == record.get("id") for existing in existing_records):
            raise ValueError(f"evaluation case id {record.get('id')!r} already exists")

        sample_fields = ("category", "expected", "expected_recognized", "strokes")
        is_labeled_sample = bool(record.get("category") and record.get("expected") and record.get("strokes"))
        if is_labeled_sample and any(
            all(existing.get(field) == record.get(field) for field in sample_fields)
            for existing in existing_records
        ):
            raise ValueError("this labeled stroke sample has already been saved")

        handle.seek(0, 2)
        handle.write(json.dumps(record, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def validate_manifest_text(field: str, value: str) -> str:
    cleaned = " ".join(str(value).strip().split())
    if not cleaned:
        raise ValueError(f"{field} must not be empty")
    return cleaned


def penpal_rows_url(*, offset: int = 0, length: int = 20) -> str:
    query = urlencode(
        {
            "dataset": "breitburg/penpal",
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    return f"https://datasets-server.huggingface.co/rows?{query}"


def fetch_penpal_rows(*, offset: int = 0, length: int = 20, timeout: float = 30.0) -> Sequence[Dict[str, Any]]:
    with urlopen(penpal_rows_url(offset=offset, length=length), timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("Penpal Dataset Viewer response did not contain rows")
    return rows


def build_penpal_case_records(
    rows: Sequence[Dict[str, Any]],
    *,
    max_sentence_cases: int,
    max_word_cases: int,
) -> Sequence[Dict[str, Any]]:
    records = []
    sentence_count = 0
    word_count = 0
    for row in rows:
        raw = row.get("row", {})
        if not isinstance(raw, dict):
            continue
        text = validate_manifest_text("text", str(raw.get("text", "")))
        word_groups = raw.get("strokes")
        if not isinstance(word_groups, list):
            continue
        file_id = str(raw.get("file", row.get("row_idx", "unknown")))[:12]

        if sentence_count < max_sentence_cases:
            records.append(
                {
                    "id": f"penpal_sentence_{file_id}",
                    "category": "sentence",
                    "source": "hf_penpal_synthetic",
                    "expected": text,
                    "expected_recognized": text,
                    "strokes": penpal_word_groups_to_strokes(
                        word_groups,
                        expected_words=text.split(),
                    ),
                    "notes": "Synthetic Hugging Face Penpal stroke sample.",
                }
            )
            sentence_count += 1

        if word_count < max_word_cases:
            words = text.split()
            for index, word in enumerate(words):
                if index >= len(word_groups):
                    break
                if not word or not any(char.isalnum() for char in word):
                    continue
                records.append(
                    {
                        "id": f"penpal_word_{file_id}_{index}",
                        "category": "word",
                        "source": "hf_penpal_synthetic",
                        "expected": word,
                        "expected_recognized": word,
                        "strokes": penpal_word_groups_to_strokes(
                            [word_groups[index]],
                            expected_words=[word],
                        ),
                        "notes": "Synthetic Hugging Face Penpal word-level stroke sample.",
                    }
                )
                word_count += 1
                if word_count >= max_word_cases:
                    break

        if sentence_count >= max_sentence_cases and word_count >= max_word_cases:
            break
    return tuple(records)


def penpal_word_groups_to_strokes(
    word_groups: Sequence[object],
    *,
    expected_words: Optional[Sequence[str]] = None,
) -> Sequence[Sequence[Dict[str, float]]]:
    strokes = []
    timestamp_ms = 0
    for word_index, word_group in enumerate(word_groups):
        if not isinstance(word_group, list):
            continue
        expected_word = expected_words[word_index] if expected_words and word_index < len(expected_words) else ""
        for stroke in word_group:
            if not isinstance(stroke, dict):
                continue
            points = stroke.get("points", [])
            if not isinstance(points, list) or not points:
                continue
            if is_penpal_artifact_stroke(points, expected_word=expected_word):
                continue
            converted = []
            for point in points:
                if not isinstance(point, dict):
                    continue
                converted.append(
                    {
                        "x": float(point["x"]),
                        "y": float(point["y"]),
                        "timestamp_ms": timestamp_ms,
                        "pressure": 1.0,
                    }
                )
                timestamp_ms += 8
            if converted:
                strokes.append(tuple(converted))
            timestamp_ms += 24
        timestamp_ms += 120
    return tuple(strokes)


def is_penpal_artifact_stroke(points: Sequence[object], *, expected_word: str) -> bool:
    if len(points) > 1:
        return False
    return not any(char in expected_word.lower() for char in "ij!?:;.")

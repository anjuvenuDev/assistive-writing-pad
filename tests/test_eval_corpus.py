import json

import pytest

from assistive_writing_pad.eval.corpus import (
    append_jsonl_record,
    build_end_to_end_case_record,
    stroke_groups_from_capture_payload,
)


def test_build_end_to_end_case_record_from_browser_payload() -> None:
    record = build_end_to_end_case_record(
        case_id="word_the_001",
        category="word",
        expected="the",
        expected_recognized="teh",
        notes="transposed letters",
        stroke_payload={
            "strokes": [
                [
                    {"x": 1, "y": 2, "timestamp_ms": 0, "pressure": 0.5},
                    {"x": 3, "y": 4, "timestamp_ms": 16},
                ]
            ],
            "mode": "auto",
        },
    )

    assert record["id"] == "word_the_001"
    assert record["source"] == "manual"
    assert record["expected"] == "the"
    assert record["expected_recognized"] == "teh"
    assert record["strokes"][0][0]["pressure"] == 0.5
    assert record["strokes"][0][1]["pressure"] == 1.0


def test_stroke_groups_from_capture_payload_accepts_legacy_points_payload() -> None:
    groups = stroke_groups_from_capture_payload(
        {"points": [{"x": 1, "y": 2, "timestamp_ms": 0}]}
    )

    assert len(groups) == 1
    assert groups[0][0].x == 1.0


def test_build_end_to_end_case_record_rejects_invalid_id_and_empty_expected() -> None:
    with pytest.raises(ValueError, match="id must start"):
        build_end_to_end_case_record(
            case_id="../bad",
            category="word",
            expected="the",
            stroke_payload={"strokes": [[{"x": 1, "y": 2}]]},
        )

    with pytest.raises(ValueError, match="expected must not be empty"):
        build_end_to_end_case_record(
            case_id="word_001",
            category="word",
            expected=" ",
            stroke_payload={"strokes": [[{"x": 1, "y": 2}]]},
        )


def test_append_jsonl_record_writes_one_compact_record_per_line(tmp_path) -> None:
    manifest = tmp_path / "cases.jsonl"
    append_jsonl_record(manifest, {"id": "case_1", "strokes": []})
    append_jsonl_record(manifest, {"id": "case_2", "strokes": []})

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["case_1", "case_2"]


def test_append_jsonl_record_rejects_duplicate_id(tmp_path) -> None:
    manifest = tmp_path / "cases.jsonl"
    append_jsonl_record(manifest, {"id": "sentence_001", "strokes": []})

    with pytest.raises(ValueError, match="already exists"):
        append_jsonl_record(manifest, {"id": "sentence_001", "strokes": [[{"x": 2}]]})


def test_append_jsonl_record_rejects_same_labeled_strokes_under_new_id(tmp_path) -> None:
    manifest = tmp_path / "cases.jsonl"
    original = {
        "id": "sentence_001",
        "category": "sentence",
        "expected": "one",
        "expected_recognized": "won",
        "strokes": [[{"x": 1}]],
    }
    append_jsonl_record(manifest, original)

    with pytest.raises(ValueError, match="already been saved"):
        append_jsonl_record(manifest, {**original, "id": "sentence_002"})

import importlib.util
from pathlib import Path

from assistive_writing_pad.eval.corpus import (
    build_penpal_case_records,
    is_penpal_artifact_stroke,
    penpal_rows_url,
    penpal_word_groups_to_strokes,
)

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "import_penpal_samples.py"
_SPEC = importlib.util.spec_from_file_location("import_penpal_samples_script", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
import_penpal_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(import_penpal_script)


def penpal_row():
    return {
        "row_idx": 0,
        "row": {
            "text": "the cat",
            "file": "abcdef1234567890",
            "strokes": [
                [
                    {
                        "points": [
                            {"x": 0.0, "y": 0.0},
                            {"x": 1.0, "y": 2.0},
                        ]
                    }
                ],
                [
                    {
                        "points": [
                            {"x": 10.0, "y": 0.0},
                            {"x": 11.0, "y": 1.0},
                        ]
                    }
                ],
            ],
        },
    }


def test_penpal_rows_url_uses_dataset_viewer_api() -> None:
    url = penpal_rows_url(offset=10, length=5)

    assert url.startswith("https://datasets-server.huggingface.co/rows?")
    assert "dataset=breitburg%2Fpenpal" in url
    assert "offset=10" in url
    assert "length=5" in url


def test_penpal_word_groups_to_strokes_flattens_words_to_stroke_groups() -> None:
    strokes = penpal_word_groups_to_strokes(
        penpal_row()["row"]["strokes"],
        expected_words=["the", "cat"],
    )

    assert len(strokes) == 2
    assert strokes[0][0]["x"] == 0.0
    assert strokes[1][0]["x"] == 10.0
    assert strokes[1][0]["timestamp_ms"] > strokes[0][-1]["timestamp_ms"]


def test_penpal_word_groups_to_strokes_drops_single_point_artifacts() -> None:
    strokes = penpal_word_groups_to_strokes(
        [[{"points": [{"x": 5.0, "y": 5.0}]}]],
        expected_words=["the"],
    )

    assert strokes == ()


def test_penpal_artifact_filter_preserves_dot_like_word_marks() -> None:
    assert is_penpal_artifact_stroke([{"x": 1, "y": 2}], expected_word="the")
    assert not is_penpal_artifact_stroke([{"x": 1, "y": 2}], expected_word="idea")
    assert not is_penpal_artifact_stroke([{"x": 1, "y": 2}], expected_word="text.")


def test_build_penpal_case_records_creates_sentence_and_word_cases() -> None:
    records = build_penpal_case_records(
        [penpal_row()],
        max_sentence_cases=1,
        max_word_cases=2,
    )

    assert [record["category"] for record in records] == ["sentence", "word", "word"]
    assert all(record["source"] == "hf_penpal_synthetic" for record in records)
    assert records[0]["expected"] == "the cat"
    assert records[1]["expected"] == "the"


def test_import_penpal_script_replace_rewrites_manifest(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "penpal.jsonl"
    manifest.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(import_penpal_script, "fetch_penpal_rows", lambda offset, length: [penpal_row()])

    result = import_penpal_script.main(
        [
            "--manifest",
            str(manifest),
            "--max-sentence-cases",
            "1",
            "--max-word-cases",
            "1",
            "--replace",
        ]
    )

    content = manifest.read_text(encoding="utf-8")
    assert result == 0
    assert "old" not in content
    assert content.count("\n") == 2

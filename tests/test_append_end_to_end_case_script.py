import importlib.util
import json
from pathlib import Path

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "append_end_to_end_case.py"
_SPEC = importlib.util.spec_from_file_location("append_end_to_end_case_script", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
append_case_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(append_case_script)


def test_append_end_to_end_case_script_dry_run_outputs_record(tmp_path, capsys) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({"strokes": [[{"x": 1, "y": 2, "timestamp_ms": 0}]]}),
        encoding="utf-8",
    )

    result = append_case_script.main_with_args(
        [
            "--payload",
            str(payload),
            "--id",
            "word_001",
            "--category",
            "word",
            "--expected",
            "the",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    record = json.loads(captured.out)
    assert record["id"] == "word_001"
    assert record["expected"] == "the"

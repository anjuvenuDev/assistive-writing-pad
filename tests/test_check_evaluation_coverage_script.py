import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "check_evaluation_coverage.py"
_SPEC = importlib.util.spec_from_file_location("check_evaluation_coverage_script", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
coverage_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coverage_script)


def test_print_report_lists_findings(capsys) -> None:
    report = SimpleNamespace(
        total_cases=1,
        manual_cases=0,
        by_category={"single_character": 1},
        by_source={"smoke": 1},
        findings=[
            SimpleNamespace(
                requirement="word.min_cases",
                actual=0,
                expected=50,
                passed=False,
            )
        ],
    )

    coverage_script.print_report(report, profile="production")

    captured = capsys.readouterr()
    assert "Evaluation coverage (production)" in captured.out
    assert "fail: word.min_cases 0/50" in captured.out

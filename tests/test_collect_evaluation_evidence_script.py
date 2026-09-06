import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "collect_evaluation_evidence.py"
_SPEC = importlib.util.spec_from_file_location("collect_evaluation_evidence_script", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
evidence_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evidence_script)


def test_print_report_lists_blocking_findings(capsys) -> None:
    report = SimpleNamespace(
        smoke_ready=True,
        production_ready=False,
        findings=[
            SimpleNamespace(
                gate="coverage",
                metric="production_profile",
                actual=False,
                expected=True,
                passed=False,
            )
        ],
    )

    evidence_script.print_report(report)

    captured = capsys.readouterr()
    assert "smoke ready: true" in captured.out
    assert "production ready: false" in captured.out
    assert "coverage.production_profile" in captured.out

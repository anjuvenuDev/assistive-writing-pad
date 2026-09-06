import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_evaluation_suite.py"
_SPEC = importlib.util.spec_from_file_location("run_evaluation_suite_script", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
suite_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = suite_script
_SPEC.loader.exec_module(suite_script)


def test_build_suite_steps_runs_fresh_reports_before_readiness() -> None:
    steps = suite_script.build_suite_steps(".venv/bin/python", profile="smoke")

    assert [step.name for step in steps] == [
        "correction",
        "end_to_end",
        "coverage_smoke",
        "readiness",
    ]
    assert steps[0].command[1] == "scripts/evaluate_correction.py"
    assert steps[1].command[1] == "scripts/evaluate_end_to_end.py"
    assert steps[-1].command[1] == "scripts/collect_evaluation_evidence.py"
    assert steps[-1].env["AWP_TROCR_LOCAL_FILES_ONLY"] == "1"


def test_build_suite_steps_includes_production_coverage_for_production_profile() -> None:
    steps = suite_script.build_suite_steps(".venv/bin/python", profile="production")

    assert [step.name for step in steps] == [
        "correction",
        "end_to_end",
        "coverage_smoke",
        "coverage_production",
        "readiness",
    ]


def test_print_report_lists_step_status(capsys) -> None:
    report = SimpleNamespace(
        profile="smoke",
        passed=False,
        steps=[
            SimpleNamespace(name="correction", passed=True, returncode=0),
            SimpleNamespace(name="readiness", passed=False, returncode=2),
        ],
    )

    suite_script.print_report(report)

    captured = capsys.readouterr()
    assert "profile: smoke" in captured.out
    assert "pass: correction" in captured.out
    assert "fail: readiness" in captured.out

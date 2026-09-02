import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "evaluate_correction.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_correction_script", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
evaluate_correction_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_correction_script)

correction_gate_failures = evaluate_correction_script.correction_gate_failures
warm_up_corrector = evaluate_correction_script.warm_up_corrector


class WarmableCorrector:
    def __init__(self) -> None:
        self.warm_up_count = 0

    def warm_up(self) -> None:
        self.warm_up_count += 1


def fake_report(
    *,
    accuracy: float = 1.0,
    false_positive_rate: float = 0.0,
    missed_corrections: int = 0,
    p95_latency_ms: float = 100.0,
):
    return SimpleNamespace(
        summary=SimpleNamespace(
            accuracy=accuracy,
            false_positive_rate=false_positive_rate,
            missed_corrections=missed_corrections,
            p95_latency_ms=p95_latency_ms,
        )
    )


def test_correction_gate_passes_when_metrics_are_in_bounds() -> None:
    assert correction_gate_failures(
        fake_report(),
        min_accuracy=0.98,
        max_false_positive_rate=0.0,
        max_missed_corrections=0,
        max_p95_latency_ms=200.0,
    ) == []


def test_correction_gate_reports_accuracy_false_positive_miss_and_latency_failures() -> None:
    failures = correction_gate_failures(
        fake_report(
            accuracy=0.75,
            false_positive_rate=0.5,
            missed_corrections=2,
            p95_latency_ms=250.0,
        ),
        min_accuracy=0.98,
        max_false_positive_rate=0.0,
        max_missed_corrections=0,
        max_p95_latency_ms=200.0,
    )

    assert len(failures) == 4
    assert "accuracy" in failures[0]
    assert "false-positive" in failures[1]
    assert "missed corrections" in failures[2]
    assert "p95 latency" in failures[3]


def test_warm_up_corrector_delegates_to_warmable_corrector() -> None:
    corrector = WarmableCorrector()

    elapsed_ms = warm_up_corrector(corrector)

    assert corrector.warm_up_count == 1
    assert elapsed_ms >= 0.0

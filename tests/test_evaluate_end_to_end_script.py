import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "evaluate_end_to_end.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_end_to_end_script", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
evaluate_end_to_end_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_end_to_end_script)

end_to_end_gate_failures = evaluate_end_to_end_script.end_to_end_gate_failures
warm_up_components = evaluate_end_to_end_script.warm_up_components


class Warmable:
    def __init__(self) -> None:
        self.warm_up_count = 0

    def warm_up(self) -> None:
        self.warm_up_count += 1


def fake_report(
    *,
    recognition_accuracy: float = 1.0,
    corrected_accuracy: float = 1.0,
    average_recognition_cer: float = 0.0,
    average_corrected_cer: float = 0.0,
    low_confidence: int = 0,
    needs_review: int = 0,
    p95_total_latency_ms: float = 100.0,
):
    return SimpleNamespace(
        summary=SimpleNamespace(
            recognition_accuracy=recognition_accuracy,
            corrected_accuracy=corrected_accuracy,
            average_recognition_cer=average_recognition_cer,
            average_corrected_cer=average_corrected_cer,
            low_confidence=low_confidence,
            needs_review=needs_review,
            p95_total_latency_ms=p95_total_latency_ms,
        )
    )


def test_end_to_end_gate_passes_when_metrics_are_in_bounds() -> None:
    assert (
        end_to_end_gate_failures(
            fake_report(),
            min_recognition_accuracy=0.95,
            min_corrected_accuracy=0.95,
            max_average_recognition_cer=0.05,
            max_average_corrected_cer=0.05,
            max_low_confidence=0,
            max_needs_review=0,
            max_p95_total_latency_ms=500.0,
        )
        == []
    )


def test_end_to_end_gate_reports_accuracy_error_confidence_review_and_latency() -> None:
    failures = end_to_end_gate_failures(
        fake_report(
            recognition_accuracy=0.5,
            corrected_accuracy=0.25,
            average_recognition_cer=0.3,
            average_corrected_cer=0.4,
            low_confidence=2,
            needs_review=3,
            p95_total_latency_ms=900.0,
        ),
        min_recognition_accuracy=0.95,
        min_corrected_accuracy=0.95,
        max_average_recognition_cer=0.05,
        max_average_corrected_cer=0.05,
        max_low_confidence=0,
        max_needs_review=0,
        max_p95_total_latency_ms=500.0,
    )

    assert len(failures) == 7
    assert "recognition accuracy" in failures[0]
    assert "corrected accuracy" in failures[1]
    assert "recognition CER" in failures[2]
    assert "corrected CER" in failures[3]
    assert "low-confidence" in failures[4]
    assert "review cases" in failures[5]
    assert "p95 total latency" in failures[6]


def test_warm_up_components_delegates_to_warmable_components() -> None:
    recognizer = Warmable()
    corrector = Warmable()

    elapsed_ms = warm_up_components(recognizer, corrector)

    assert recognizer.warm_up_count == 1
    assert corrector.warm_up_count == 1
    assert elapsed_ms >= 0.0

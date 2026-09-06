import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "evaluate_recognition.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_recognition_script", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
evaluate_recognition_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_recognition_script)

recognition_gate_failures = evaluate_recognition_script.recognition_gate_failures
warm_up_recognizer = evaluate_recognition_script.warm_up_recognizer


class WarmableRecognizer:
    def __init__(self) -> None:
        self.warm_up_count = 0

    def warm_up(self) -> None:
        self.warm_up_count += 1


def fake_report(
    *,
    accuracy: float = 1.0,
    average_cer: float = 0.0,
    average_wer: float = 0.0,
    low_confidence: int = 0,
    p95_latency_ms: float = 100.0,
):
    return SimpleNamespace(
        summary=SimpleNamespace(
            accuracy=accuracy,
            average_cer=average_cer,
            average_wer=average_wer,
            low_confidence=low_confidence,
            p95_latency_ms=p95_latency_ms,
        )
    )


def test_recognition_gate_passes_when_metrics_are_in_bounds() -> None:
    assert recognition_gate_failures(
        fake_report(),
        min_exact_accuracy=0.95,
        max_average_cer=0.05,
        max_average_wer=0.10,
        max_low_confidence=0,
        max_p95_latency_ms=500.0,
    ) == []


def test_recognition_gate_reports_accuracy_error_confidence_and_latency_failures() -> None:
    failures = recognition_gate_failures(
        fake_report(
            accuracy=0.5,
            average_cer=0.25,
            average_wer=0.5,
            low_confidence=2,
            p95_latency_ms=700.0,
        ),
        min_exact_accuracy=0.95,
        max_average_cer=0.05,
        max_average_wer=0.10,
        max_low_confidence=0,
        max_p95_latency_ms=500.0,
    )

    assert len(failures) == 5
    assert "accuracy" in failures[0]
    assert "average CER" in failures[1]
    assert "average WER" in failures[2]
    assert "low-confidence" in failures[3]
    assert "p95 latency" in failures[4]


def test_warm_up_recognizer_delegates_to_warmable_recognizer() -> None:
    recognizer = WarmableRecognizer()

    elapsed_ms = warm_up_recognizer(recognizer)

    assert recognizer.warm_up_count == 1
    assert elapsed_ms >= 0.0

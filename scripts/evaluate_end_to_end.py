#!/usr/bin/env python
"""Evaluate the complete stroke recognition and correction path."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Optional

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from assistive_writing_pad.config.settings import RuntimeSettings  # noqa: E402
from assistive_writing_pad.correction.factory import corrector_from_settings  # noqa: E402
from assistive_writing_pad.eval.end_to_end_eval import (  # noqa: E402
    evaluate_end_to_end_cases,
    load_end_to_end_cases,
)
from assistive_writing_pad.recognition.trocr import (  # noqa: E402
    DEFAULT_TROCR_MODEL,
    RecognitionUnavailable,
    TrOCRHandwritingRecognizer,
    default_huggingface_cache_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/end_to_end_cases.jsonl"),
        help="JSONL file with id, category, expected, and strokes fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TROCR_MODEL,
        help="Hugging Face TrOCR model ID.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override shared Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only cached Hugging Face model files for OCR and correction.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "character", "word", "ocr"],
        default="auto",
        help="Recognition mode to pass to the recognizer.",
    )
    parser.add_argument(
        "--correction-mode",
        choices=["hf", "contextual", "rules"],
        default=None,
        help="Override AWP_CORRECTION_MODE for this run.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit cases for quick smoke tests.",
    )
    parser.add_argument(
        "--no-warm-up",
        action="store_true",
        help="Skip OCR and correction warm-up before timed evaluation.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
        help="Count outputs below this confidence as low-confidence cases.",
    )
    parser.add_argument(
        "--min-recognition-accuracy",
        type=float,
        default=0.0,
        help="Fail if exact-match recognition accuracy is below this value.",
    )
    parser.add_argument(
        "--min-corrected-accuracy",
        type=float,
        default=0.98,
        help="Fail if exact-match corrected accuracy is below this value.",
    )
    parser.add_argument(
        "--max-average-recognition-cer",
        type=float,
        default=None,
        help="Fail if average recognition CER is above this value.",
    )
    parser.add_argument(
        "--max-average-corrected-cer",
        type=float,
        default=None,
        help="Fail if average corrected CER is above this value.",
    )
    parser.add_argument(
        "--max-low-confidence",
        type=int,
        default=None,
        help="Fail if low-confidence cases exceed this count.",
    )
    parser.add_argument(
        "--max-needs-review",
        type=int,
        default=None,
        help="Fail if pipeline review cases exceed this count.",
    )
    parser.add_argument(
        "--max-p95-total-latency-ms",
        type=float,
        default=None,
        help="Fail if p95 total recognition-plus-correction latency is above this value.",
    )
    args = parser.parse_args()

    cases = load_end_to_end_cases(args.manifest)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    settings = RuntimeSettings.from_env()
    updates = {"confidence_threshold": args.confidence_threshold}
    cache_dir = args.cache_dir or default_huggingface_cache_dir()
    if args.correction_mode is not None:
        updates["correction_mode"] = args.correction_mode
    if args.cache_dir is not None:
        updates["hf_cache_dir"] = args.cache_dir
    if args.local_files_only:
        updates["hf_correction_local_files_only"] = True
    if updates:
        settings = replace(settings, **updates)
        settings.validate()

    recognizer = TrOCRHandwritingRecognizer(
        model_name=args.model,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    corrector = corrector_from_settings(settings)

    warm_up_ms = 0.0
    try:
        if not args.no_warm_up:
            warm_up_ms = warm_up_components(recognizer, corrector)
        report = evaluate_end_to_end_cases(
            cases,
            recognizer,
            corrector,
            settings=settings,
            mode=args.mode,
            confidence_threshold=args.confidence_threshold,
        )
    except RecognitionUnavailable as exc:
        print(f"Recognition unavailable: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"End-to-end evaluation unavailable: {exc}", file=sys.stderr)
        return 1

    print_report(report)
    if warm_up_ms > 0.0:
        print()
        print(f"Warm-up time: {warm_up_ms:.1f} ms")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")

    gate_failures = end_to_end_gate_failures(
        report,
        min_recognition_accuracy=args.min_recognition_accuracy,
        min_corrected_accuracy=args.min_corrected_accuracy,
        max_average_recognition_cer=args.max_average_recognition_cer,
        max_average_corrected_cer=args.max_average_corrected_cer,
        max_low_confidence=args.max_low_confidence,
        max_needs_review=args.max_needs_review,
        max_p95_total_latency_ms=args.max_p95_total_latency_ms,
    )
    if gate_failures:
        print()
        print("Gate failures")
        for failure in gate_failures:
            print(f"- {failure}")
        return 2

    print()
    print("Gate passed")
    return 0


def warm_up_components(recognizer: object, corrector: object) -> float:
    started = time.perf_counter()
    warm_up_recognizer = getattr(recognizer, "warm_up", None)
    if callable(warm_up_recognizer):
        print("Warming OCR model before timed evaluation...")
        warm_up_recognizer()
    warm_up_corrector = getattr(corrector, "warm_up", None)
    if callable(warm_up_corrector):
        print("Warming correction models before timed evaluation...")
        warm_up_corrector()
    return (time.perf_counter() - started) * 1000.0


def end_to_end_gate_failures(
    report,
    *,
    min_recognition_accuracy: float,
    min_corrected_accuracy: float,
    max_average_recognition_cer: Optional[float],
    max_average_corrected_cer: Optional[float],
    max_low_confidence: Optional[int],
    max_needs_review: Optional[int],
    max_p95_total_latency_ms: Optional[float],
) -> list[str]:
    summary = report.summary
    failures = []
    if summary.recognition_accuracy < min_recognition_accuracy:
        failures.append(
            "recognition accuracy "
            f"{summary.recognition_accuracy:.1%} is below {min_recognition_accuracy:.1%}"
        )
    if summary.corrected_accuracy < min_corrected_accuracy:
        failures.append(
            f"corrected accuracy {summary.corrected_accuracy:.1%} is below "
            f"{min_corrected_accuracy:.1%}"
        )
    if (
        max_average_recognition_cer is not None
        and summary.average_recognition_cer > max_average_recognition_cer
    ):
        failures.append(
            "average recognition CER "
            f"{summary.average_recognition_cer:.1%} is above "
            f"{max_average_recognition_cer:.1%}"
        )
    if (
        max_average_corrected_cer is not None
        and summary.average_corrected_cer > max_average_corrected_cer
    ):
        failures.append(
            "average corrected CER "
            f"{summary.average_corrected_cer:.1%} is above "
            f"{max_average_corrected_cer:.1%}"
        )
    if max_low_confidence is not None and summary.low_confidence > max_low_confidence:
        failures.append(
            f"low-confidence cases {summary.low_confidence} exceed {max_low_confidence}"
        )
    if max_needs_review is not None and summary.needs_review > max_needs_review:
        failures.append(f"review cases {summary.needs_review} exceed {max_needs_review}")
    if (
        max_p95_total_latency_ms is not None
        and summary.p95_total_latency_ms > max_p95_total_latency_ms
    ):
        failures.append(
            f"p95 total latency {summary.p95_total_latency_ms:.1f} ms is above "
            f"{max_p95_total_latency_ms:.1f} ms"
        )
    return failures


def print_report(report) -> None:
    summary = report.summary
    print("End-to-end evaluation")
    print(f"cases: {summary.total}")
    print(
        "recognition exact: "
        f"{summary.recognition_exact}/{summary.total} ({summary.recognition_accuracy:.1%})"
    )
    print(
        "corrected exact: "
        f"{summary.corrected_exact}/{summary.total} ({summary.corrected_accuracy:.1%})"
    )
    print(
        "average recognition CER/WER: "
        f"{summary.average_recognition_cer:.1%} / {summary.average_recognition_wer:.1%}"
    )
    print(
        "average corrected CER/WER: "
        f"{summary.average_corrected_cer:.1%} / {summary.average_corrected_wer:.1%}"
    )
    print(f"low confidence: {summary.low_confidence}")
    print(f"needs review: {summary.needs_review}")
    print(
        "total latency avg/p95: "
        f"{summary.average_total_latency_ms:.1f} / {summary.p95_total_latency_ms:.1f} ms"
    )
    print()
    print("By category")
    for category, metrics in summary.by_category.items():
        print(
            f"- {category}: recognition {metrics.recognition_exact}/{metrics.total} "
            f"({metrics.recognition_accuracy:.1%}), corrected "
            f"{metrics.corrected_exact}/{metrics.total} ({metrics.corrected_accuracy:.1%}), "
            f"total avg {metrics.average_total_latency_ms:.1f} ms, "
            f"p95 {metrics.p95_total_latency_ms:.1f} ms"
        )

    failures = [row for row in report.rows if not row.corrected_exact_match]
    if not failures:
        return
    print()
    print("Failures")
    for row in failures:
        print(
            f"- {row.id}: expected={row.expected_corrected_text!r} "
            f"recognized={row.recognized_text!r} corrected={row.corrected_text!r}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

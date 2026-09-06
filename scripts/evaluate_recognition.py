#!/usr/bin/env python
"""Evaluate handwriting recognition accuracy and latency from stroke fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Optional

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from assistive_writing_pad.eval.recognition_eval import (  # noqa: E402
    evaluate_recognition_cases,
    load_recognition_cases,
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
        default=Path("data/evaluation/recognition_cases.jsonl"),
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
        help="Override Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only cached Hugging Face model files.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "character", "word", "ocr"],
        default="auto",
        help="Recognition mode to pass to the recognizer.",
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
        help="Skip OCR model warm-up before timed evaluation.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
        help="Count outputs below this confidence as low-confidence cases.",
    )
    parser.add_argument(
        "--min-exact-accuracy",
        type=float,
        default=0.0,
        help="Fail if exact-match recognition accuracy is below this value.",
    )
    parser.add_argument(
        "--max-average-cer",
        type=float,
        default=None,
        help="Fail if average character error rate is above this value.",
    )
    parser.add_argument(
        "--max-average-wer",
        type=float,
        default=None,
        help="Fail if average word error rate is above this value.",
    )
    parser.add_argument(
        "--max-low-confidence",
        type=int,
        default=None,
        help="Fail if low-confidence cases exceed this count.",
    )
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        default=None,
        help="Fail if p95 recognition latency is above this value.",
    )
    args = parser.parse_args()

    cases = load_recognition_cases(args.manifest)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    recognizer = TrOCRHandwritingRecognizer(
        model_name=args.model,
        cache_dir=args.cache_dir or default_huggingface_cache_dir(),
        local_files_only=args.local_files_only,
    )
    warm_up_ms = 0.0
    try:
        if not args.no_warm_up:
            warm_up_ms = warm_up_recognizer(recognizer)
        report = evaluate_recognition_cases(
            cases,
            recognizer,
            mode=args.mode,
            confidence_threshold=args.confidence_threshold,
        )
    except RecognitionUnavailable as exc:
        print(f"Recognition unavailable: {exc}", file=sys.stderr)
        return 1
    print_report(report)
    if warm_up_ms > 0.0:
        print()
        print(f"Warm-up time: {warm_up_ms:.1f} ms")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")

    gate_failures = recognition_gate_failures(
        report,
        min_exact_accuracy=args.min_exact_accuracy,
        max_average_cer=args.max_average_cer,
        max_average_wer=args.max_average_wer,
        max_low_confidence=args.max_low_confidence,
        max_p95_latency_ms=args.max_p95_latency_ms,
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


def warm_up_recognizer(recognizer: object) -> float:
    warm_up = getattr(recognizer, "warm_up", None)
    if not callable(warm_up):
        return 0.0
    print("Warming OCR model before timed evaluation...")
    started = time.perf_counter()
    warm_up()
    return (time.perf_counter() - started) * 1000.0


def recognition_gate_failures(
    report,
    *,
    min_exact_accuracy: float,
    max_average_cer: Optional[float],
    max_average_wer: Optional[float],
    max_low_confidence: Optional[int],
    max_p95_latency_ms: Optional[float],
) -> list[str]:
    summary = report.summary
    failures = []
    if summary.accuracy < min_exact_accuracy:
        failures.append(f"accuracy {summary.accuracy:.1%} is below {min_exact_accuracy:.1%}")
    if max_average_cer is not None and summary.average_cer > max_average_cer:
        failures.append(
            f"average CER {summary.average_cer:.1%} is above {max_average_cer:.1%}"
        )
    if max_average_wer is not None and summary.average_wer > max_average_wer:
        failures.append(
            f"average WER {summary.average_wer:.1%} is above {max_average_wer:.1%}"
        )
    if max_low_confidence is not None and summary.low_confidence > max_low_confidence:
        failures.append(
            f"low-confidence cases {summary.low_confidence} exceed {max_low_confidence}"
        )
    if max_p95_latency_ms is not None and summary.p95_latency_ms > max_p95_latency_ms:
        failures.append(
            f"p95 latency {summary.p95_latency_ms:.1f} ms is above "
            f"{max_p95_latency_ms:.1f} ms"
        )
    return failures


def print_report(report) -> None:
    summary = report.summary
    print("Recognition evaluation")
    print(f"cases: {summary.total}")
    print(f"exact: {summary.exact}/{summary.total} ({summary.accuracy:.1%})")
    print(f"average CER/WER: {summary.average_cer:.1%} / {summary.average_wer:.1%}")
    print(f"low confidence: {summary.low_confidence}")
    print(f"latency avg/p95: {summary.average_latency_ms:.1f} / {summary.p95_latency_ms:.1f} ms")
    print()
    print("By category")
    for category, metrics in summary.by_category.items():
        print(
            f"- {category}: {metrics.exact}/{metrics.total} "
            f"({metrics.accuracy:.1%}), CER {metrics.average_cer:.1%}, "
            f"WER {metrics.average_wer:.1%}, avg {metrics.average_latency_ms:.1f} ms, "
            f"p95 {metrics.p95_latency_ms:.1f} ms"
        )

    failures = [row for row in report.rows if not row.exact_match]
    if not failures:
        return
    print()
    print("Failures")
    for row in failures:
        print(f"- {row.id}: expected={row.expected_text!r} output={row.output_text!r}")


if __name__ == "__main__":
    raise SystemExit(main())

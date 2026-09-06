#!/usr/bin/env python
"""Evaluate correction accuracy, false positives, and latency."""

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
from assistive_writing_pad.eval.correction_eval import (  # noqa: E402
    evaluate_correction_cases,
    load_correction_cases,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/correction_cases.jsonl"),
        help="JSONL file with id, category, input, and expected fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--correction-mode",
        choices=["hf", "contextual", "rules"],
        default=None,
        help="Override AWP_CORRECTION_MODE for this run.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only cached Hugging Face model files.",
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
        help="Skip model warm-up before timed evaluation.",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.98,
        help="Fail if exact-match accuracy is below this value.",
    )
    parser.add_argument(
        "--max-false-positive-rate",
        type=float,
        default=0.0,
        help="Fail if clean-text false-positive rate is above this value.",
    )
    parser.add_argument(
        "--max-missed-corrections",
        type=int,
        default=0,
        help="Fail if missed corrections exceed this count.",
    )
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        default=None,
        help="Optional p95 latency ceiling for the timed correction pass.",
    )
    args = parser.parse_args()

    cases = load_correction_cases(args.manifest)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    settings = RuntimeSettings.from_env()
    updates = {}
    if args.correction_mode is not None:
        updates["correction_mode"] = args.correction_mode
    if args.local_files_only:
        updates["hf_correction_local_files_only"] = True
    if updates:
        settings = replace(settings, **updates)
        settings.validate()

    corrector = corrector_from_settings(settings)
    warm_up_ms = 0.0
    if not args.no_warm_up:
        warm_up_ms = warm_up_corrector(corrector)

    report = evaluate_correction_cases(cases, corrector)
    print_report(report)
    if warm_up_ms > 0.0:
        print()
        print(f"Warm-up time: {warm_up_ms:.1f} ms")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")

    gate_failures = correction_gate_failures(
        report,
        min_accuracy=args.min_accuracy,
        max_false_positive_rate=args.max_false_positive_rate,
        max_missed_corrections=args.max_missed_corrections,
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


def warm_up_corrector(corrector: object) -> float:
    warm_up = getattr(corrector, "warm_up", None)
    if not callable(warm_up):
        return 0.0
    print("Warming correction models before timed evaluation...")
    started = time.perf_counter()
    warm_up()
    return (time.perf_counter() - started) * 1000.0


def correction_gate_failures(
    report,
    *,
    min_accuracy: float,
    max_false_positive_rate: float,
    max_missed_corrections: int,
    max_p95_latency_ms: Optional[float],
) -> list[str]:
    summary = report.summary
    failures = []
    if summary.accuracy < min_accuracy:
        failures.append(f"accuracy {summary.accuracy:.1%} is below {min_accuracy:.1%}")
    if summary.false_positive_rate > max_false_positive_rate:
        failures.append(
            "false-positive rate "
            f"{summary.false_positive_rate:.1%} is above {max_false_positive_rate:.1%}"
        )
    if summary.missed_corrections > max_missed_corrections:
        failures.append(
            f"missed corrections {summary.missed_corrections} exceed "
            f"{max_missed_corrections}"
        )
    if max_p95_latency_ms is not None and summary.p95_latency_ms > max_p95_latency_ms:
        failures.append(
            f"p95 latency {summary.p95_latency_ms:.1f} ms is above "
            f"{max_p95_latency_ms:.1f} ms"
        )
    return failures


def print_report(report) -> None:
    summary = report.summary
    print("Correction evaluation")
    print(f"cases: {summary.total}")
    print(f"exact: {summary.exact}/{summary.total} ({summary.accuracy:.1%})")
    print(
        "false positives: "
        f"{summary.false_positives} ({summary.false_positive_rate:.1%} of clean cases)"
    )
    print(f"missed corrections: {summary.missed_corrections}")
    print(f"latency avg/p95: {summary.average_latency_ms:.1f} / {summary.p95_latency_ms:.1f} ms")
    print()
    print("By category")
    for category, metrics in summary.by_category.items():
        print(
            f"- {category}: {metrics.exact}/{metrics.total} "
            f"({metrics.accuracy:.1%}), avg {metrics.average_latency_ms:.1f} ms, "
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

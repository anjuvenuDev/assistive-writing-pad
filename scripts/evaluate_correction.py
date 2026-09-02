#!/usr/bin/env python
"""Evaluate correction accuracy, false positives, and latency."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

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

    report = evaluate_correction_cases(cases, corrector_from_settings(settings))
    print_report(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")

    return 0 if report.summary.false_positives == 0 else 2


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

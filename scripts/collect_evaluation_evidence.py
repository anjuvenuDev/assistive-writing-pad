#!/usr/bin/env python
"""Collect evaluation reports into one readiness evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from assistive_writing_pad.config.settings import RuntimeSettings  # noqa: E402
from assistive_writing_pad.eval.coverage import (  # noqa: E402
    evaluate_coverage,
    requirements_for_profile,
)
from assistive_writing_pad.eval.end_to_end_eval import load_end_to_end_cases  # noqa: E402
from assistive_writing_pad.eval.readiness import (  # noqa: E402
    ReadinessThresholds,
    build_readiness_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correction-report",
        type=Path,
        default=Path("data/evaluation/correction_report.json"),
        help="Correction evaluation JSON report.",
    )
    parser.add_argument(
        "--end-to-end-report",
        type=Path,
        default=Path("data/evaluation/end_to_end_report.json"),
        help="End-to-end evaluation JSON report.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/end_to_end_cases.jsonl"),
        help="End-to-end manifest used for coverage checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/readiness_report.json"),
        help="Readiness evidence output path.",
    )
    parser.add_argument(
        "--min-correction-accuracy",
        type=float,
        default=0.98,
        help="Minimum correction exact-match accuracy.",
    )
    parser.add_argument(
        "--min-end-to-end-recognition-accuracy",
        type=float,
        default=0.95,
        help="Minimum end-to-end raw recognition accuracy.",
    )
    parser.add_argument(
        "--min-end-to-end-corrected-accuracy",
        type=float,
        default=0.98,
        help="Minimum end-to-end corrected accuracy.",
    )
    args = parser.parse_args()

    try:
        correction_report = read_json(args.correction_report)
        end_to_end_report = read_json(args.end_to_end_report)
        cases = load_end_to_end_cases(args.manifest)
        smoke_coverage = evaluate_coverage(
            cases,
            requirements=requirements_for_profile("smoke"),
        )
        production_coverage = evaluate_coverage(
            cases,
            requirements=requirements_for_profile("production"),
        )
        report = build_readiness_report(
            correction_report=correction_report,
            end_to_end_report=end_to_end_report,
            smoke_coverage=smoke_coverage,
            production_coverage=production_coverage,
            settings=RuntimeSettings.from_env(),
            thresholds=ReadinessThresholds(
                min_correction_accuracy=args.min_correction_accuracy,
                min_end_to_end_recognition_accuracy=args.min_end_to_end_recognition_accuracy,
                min_end_to_end_corrected_accuracy=args.min_end_to_end_corrected_accuracy,
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not collect evaluation evidence: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print_report(report)
    print(f"Wrote {args.output}")
    return 0 if report.production_ready else 2


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def print_report(report) -> None:
    print("Evaluation evidence")
    print(f"smoke ready: {str(report.smoke_ready).lower()}")
    print(f"production ready: {str(report.production_ready).lower()}")
    failed = [finding for finding in report.findings if not finding.passed]
    if not failed:
        print("all gates passed")
        return
    print("blocking findings:")
    for finding in failed:
        print(
            f"- {finding.gate}.{finding.metric}: actual={finding.actual!r}, "
            f"expected={finding.expected!r}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

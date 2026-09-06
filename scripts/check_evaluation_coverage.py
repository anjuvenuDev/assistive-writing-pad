#!/usr/bin/env python
"""Check whether the end-to-end evaluation manifest has enough coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from assistive_writing_pad.eval.coverage import (  # noqa: E402
    evaluate_coverage,
    requirements_for_profile,
)
from assistive_writing_pad.eval.end_to_end_eval import load_end_to_end_cases  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/end_to_end_cases.jsonl"),
        help="End-to-end JSONL manifest to check.",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "production"],
        default="production",
        help="Coverage requirement profile.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON coverage report output path.",
    )
    args = parser.parse_args()

    try:
        cases = load_end_to_end_cases(args.manifest)
        report = evaluate_coverage(
            cases,
            requirements=requirements_for_profile(args.profile),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not check evaluation coverage: {exc}", file=sys.stderr)
        return 1

    print_report(report, profile=args.profile)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")

    if not report.passed:
        return 2
    return 0


def print_report(report, *, profile: str) -> None:
    print(f"Evaluation coverage ({profile})")
    print(f"cases: {report.total_cases}")
    print(f"manual cases: {report.manual_cases}")
    print("by category:")
    for category, count in report.by_category.items():
        print(f"- {category}: {count}")
    print("by source:")
    for source, count in report.by_source.items():
        print(f"- {source}: {count}")
    print("requirements:")
    for finding in report.findings:
        status = "pass" if finding.passed else "fail"
        print(
            f"- {status}: {finding.requirement} "
            f"{finding.actual}/{finding.expected}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

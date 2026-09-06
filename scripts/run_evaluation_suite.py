#!/usr/bin/env python
"""Run evaluation gates and collect a fresh readiness evidence report."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class EvaluationStep:
    name: str
    command: Sequence[str]
    env: Mapping[str, str]


@dataclass(frozen=True)
class EvaluationStepResult:
    name: str
    command: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class EvaluationSuiteReport:
    profile: str
    passed: bool
    steps: Sequence[EvaluationStepResult]

    def to_dict(self) -> Dict[str, object]:
        return {
            "profile": self.profile,
            "passed": self.passed,
            "steps": [
                {
                    **asdict(step),
                    "passed": step.passed,
                }
                for step in self.steps
            ],
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["smoke", "production"],
        default="smoke",
        help="Run smoke gates or require production readiness.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for evaluation scripts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/evaluation_suite_report.json"),
        help="Evaluation suite report output path.",
    )
    args = parser.parse_args(argv)

    steps = build_suite_steps(args.python, profile=args.profile)
    results = [run_step(step) for step in steps]
    report = EvaluationSuiteReport(
        profile=args.profile,
        passed=all(result.passed for result in results),
        steps=tuple(results),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print_report(report)
    print(f"Wrote {args.output}")
    return 0 if report.passed else 2


def build_suite_steps(python_executable: str, *, profile: str) -> Sequence[EvaluationStep]:
    local_only_env = {
        "AWP_TROCR_LOCAL_FILES_ONLY": "1",
        "AWP_HF_CORRECTION_LOCAL_FILES_ONLY": "1",
    }
    steps = [
        EvaluationStep(
            name="correction",
            command=(
                python_executable,
                "scripts/evaluate_correction.py",
                "--local-files-only",
                "--max-p95-latency-ms",
                "1200",
                "--output",
                "data/evaluation/correction_report.json",
            ),
            env=local_only_env,
        ),
        EvaluationStep(
            name="end_to_end",
            command=(
                python_executable,
                "scripts/evaluate_end_to_end.py",
                "--local-files-only",
                "--mode",
                "auto",
                "--output",
                "data/evaluation/end_to_end_report.json",
                "--min-recognition-accuracy",
                "1.0",
                "--min-corrected-accuracy",
                "1.0",
                "--max-average-recognition-cer",
                "0.0",
                "--max-average-corrected-cer",
                "0.0",
                "--max-low-confidence",
                "0",
                "--max-needs-review",
                "0",
                "--max-p95-total-latency-ms",
                "5000",
            ),
            env=local_only_env,
        ),
        EvaluationStep(
            name="coverage_smoke",
            command=(
                python_executable,
                "scripts/check_evaluation_coverage.py",
                "--profile",
                "smoke",
                "--output",
                "data/evaluation/coverage_report.json",
            ),
            env={},
        ),
        EvaluationStep(
            name="readiness",
            command=(
                python_executable,
                "scripts/collect_evaluation_evidence.py",
                "--profile",
                profile,
                "--output",
                "data/evaluation/readiness_report.json",
            ),
            env=local_only_env,
        ),
    ]
    if profile == "production":
        steps.insert(
            3,
            EvaluationStep(
                name="coverage_production",
                command=(
                    python_executable,
                    "scripts/check_evaluation_coverage.py",
                    "--profile",
                    "production",
                ),
                env={},
            ),
        )
    return tuple(steps)


def run_step(step: EvaluationStep) -> EvaluationStepResult:
    print(f"Running {step.name}...")
    process = subprocess.run(
        list(step.command),
        check=False,
        capture_output=True,
        env={**os.environ, **dict(step.env)},
        text=True,
    )
    if process.stdout:
        print(process.stdout.rstrip())
    if process.stderr:
        print(process.stderr.rstrip(), file=sys.stderr)
    return EvaluationStepResult(
        name=step.name,
        command=tuple(step.command),
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )


def print_report(report: EvaluationSuiteReport) -> None:
    print("Evaluation suite")
    print(f"profile: {report.profile}")
    print(f"passed: {str(report.passed).lower()}")
    for step in report.steps:
        status = "pass" if step.passed else "fail"
        print(f"- {status}: {step.name} ({step.returncode})")


if __name__ == "__main__":
    raise SystemExit(main())

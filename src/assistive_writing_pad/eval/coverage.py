"""Evaluation corpus coverage checks for release readiness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

from assistive_writing_pad.eval.end_to_end_eval import EndToEndCase


@dataclass(frozen=True)
class CategoryRequirement:
    category: str
    min_cases: int
    min_manual_cases: int = 0


@dataclass(frozen=True)
class CoverageFinding:
    requirement: str
    actual: int
    expected: int
    passed: bool


@dataclass(frozen=True)
class CoverageReport:
    total_cases: int
    manual_cases: int
    by_category: Mapping[str, int]
    by_source: Mapping[str, int]
    findings: Sequence[CoverageFinding] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return all(finding.passed for finding in self.findings)

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_cases": self.total_cases,
            "manual_cases": self.manual_cases,
            "by_category": dict(self.by_category),
            "by_source": dict(self.by_source),
            "passed": self.passed,
            "findings": [asdict(finding) for finding in self.findings],
        }


SMOKE_REQUIREMENTS = (
    CategoryRequirement(category="single_character", min_cases=2, min_manual_cases=0),
)

PRODUCTION_REQUIREMENTS = (
    CategoryRequirement(category="single_character", min_cases=20, min_manual_cases=20),
    CategoryRequirement(category="word", min_cases=50, min_manual_cases=50),
    CategoryRequirement(category="sentence", min_cases=50, min_manual_cases=50),
)


def evaluate_coverage(
    cases: Iterable[EndToEndCase],
    *,
    requirements: Sequence[CategoryRequirement],
) -> CoverageReport:
    case_list = list(cases)
    by_category = count_by(case_list, "category")
    by_source = count_by(case_list, "source")
    findings: List[CoverageFinding] = []

    for requirement in requirements:
        category_cases = [case for case in case_list if case.category == requirement.category]
        findings.append(
            CoverageFinding(
                requirement=f"{requirement.category}.min_cases",
                actual=len(category_cases),
                expected=requirement.min_cases,
                passed=len(category_cases) >= requirement.min_cases,
            )
        )
        manual_count = sum(1 for case in category_cases if case.source == "manual")
        findings.append(
            CoverageFinding(
                requirement=f"{requirement.category}.min_manual_cases",
                actual=manual_count,
                expected=requirement.min_manual_cases,
                passed=manual_count >= requirement.min_manual_cases,
            )
        )

    manual_cases = sum(1 for case in case_list if case.source == "manual")
    return CoverageReport(
        total_cases=len(case_list),
        manual_cases=manual_cases,
        by_category=by_category,
        by_source=by_source,
        findings=tuple(findings),
    )


def requirements_for_profile(profile: str) -> Sequence[CategoryRequirement]:
    if profile == "smoke":
        return SMOKE_REQUIREMENTS
    if profile == "production":
        return PRODUCTION_REQUIREMENTS
    raise ValueError("profile must be 'smoke' or 'production'")


def count_by(cases: Sequence[EndToEndCase], field_name: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for case in cases:
        value = str(getattr(case, field_name))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))

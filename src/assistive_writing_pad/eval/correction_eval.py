"""Correction evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import statistics
import time
from typing import Dict, Iterable, List, Protocol, Sequence

from assistive_writing_pad.contracts import CorrectionResult


class CorrectorLike(Protocol):
    def correct(self, text: str) -> CorrectionResult:
        """Return corrected text for an input string."""


@dataclass(frozen=True)
class CorrectionCase:
    id: str
    input_text: str
    expected_text: str
    category: str
    notes: str = ""

    @property
    def expects_change(self) -> bool:
        return normalize_for_eval(self.input_text) != normalize_for_eval(self.expected_text)


@dataclass(frozen=True)
class CorrectionEvalRow:
    id: str
    category: str
    input_text: str
    expected_text: str
    output_text: str
    exact_match: bool
    expected_change: bool
    changed: bool
    false_positive: bool
    missed_correction: bool
    latency_ms: float
    confidence: float
    correction_count: int


@dataclass(frozen=True)
class CategoryMetrics:
    total: int
    exact: int
    accuracy: float
    false_positives: int
    false_positive_rate: float
    missed_corrections: int
    average_latency_ms: float
    p95_latency_ms: float


@dataclass(frozen=True)
class CorrectionEvalSummary:
    total: int
    exact: int
    accuracy: float
    false_positives: int
    false_positive_rate: float
    missed_corrections: int
    average_latency_ms: float
    p95_latency_ms: float
    by_category: Dict[str, CategoryMetrics] = field(default_factory=dict)


@dataclass(frozen=True)
class CorrectionEvalReport:
    summary: CorrectionEvalSummary
    rows: Sequence[CorrectionEvalRow]

    def to_dict(self) -> Dict[str, object]:
        return {
            "summary": summary_to_dict(self.summary),
            "rows": [asdict(row) for row in self.rows],
        }


def load_correction_cases(path: Path) -> List[CorrectionCase]:
    cases: List[CorrectionCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            data = json.loads(stripped)
            try:
                cases.append(
                    CorrectionCase(
                        id=str(data["id"]),
                        input_text=str(data["input"]),
                        expected_text=str(data["expected"]),
                        category=str(data["category"]),
                        notes=str(data.get("notes", "")),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} missing field {exc}") from exc
    return cases


def evaluate_correction_cases(
    cases: Iterable[CorrectionCase],
    corrector: CorrectorLike,
) -> CorrectionEvalReport:
    rows: List[CorrectionEvalRow] = []
    for case in cases:
        started = time.perf_counter()
        result = corrector.correct(case.input_text)
        latency_ms = (time.perf_counter() - started) * 1000.0

        expected = normalize_for_eval(case.expected_text)
        output = normalize_for_eval(result.corrected_text)
        input_text = normalize_for_eval(case.input_text)
        exact_match = output == expected
        expected_change = input_text != expected
        changed = input_text != output

        rows.append(
            CorrectionEvalRow(
                id=case.id,
                category=case.category,
                input_text=case.input_text,
                expected_text=case.expected_text,
                output_text=result.corrected_text,
                exact_match=exact_match,
                expected_change=expected_change,
                changed=changed,
                false_positive=not expected_change and changed,
                missed_correction=expected_change and not changed,
                latency_ms=round(latency_ms, 3),
                confidence=round(result.confidence, 4),
                correction_count=len(result.corrections),
            )
        )

    return CorrectionEvalReport(summary=summarize_rows(rows), rows=tuple(rows))


def summarize_rows(rows: Sequence[CorrectionEvalRow]) -> CorrectionEvalSummary:
    by_category = {
        category: summarize_category([row for row in rows if row.category == category])
        for category in sorted({row.category for row in rows})
    }
    category_all = summarize_category(rows)
    return CorrectionEvalSummary(
        total=category_all.total,
        exact=category_all.exact,
        accuracy=category_all.accuracy,
        false_positives=category_all.false_positives,
        false_positive_rate=category_all.false_positive_rate,
        missed_corrections=category_all.missed_corrections,
        average_latency_ms=category_all.average_latency_ms,
        p95_latency_ms=category_all.p95_latency_ms,
        by_category=by_category,
    )


def summarize_category(rows: Sequence[CorrectionEvalRow]) -> CategoryMetrics:
    total = len(rows)
    exact = sum(1 for row in rows if row.exact_match)
    false_positives = sum(1 for row in rows if row.false_positive)
    clean_total = sum(1 for row in rows if not row.expected_change)
    missed_corrections = sum(1 for row in rows if row.missed_correction)
    latencies = [row.latency_ms for row in rows]
    return CategoryMetrics(
        total=total,
        exact=exact,
        accuracy=round(exact / total, 4) if total else 0.0,
        false_positives=false_positives,
        false_positive_rate=round(false_positives / clean_total, 4) if clean_total else 0.0,
        missed_corrections=missed_corrections,
        average_latency_ms=round(statistics.fmean(latencies), 3) if latencies else 0.0,
        p95_latency_ms=round(percentile(latencies, 95), 3) if latencies else 0.0,
    )


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile_value / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)


def normalize_for_eval(text: str) -> str:
    cleaned = " ".join(text.strip().split()).lower()
    cleaned = cleaned.replace("\u2019", "'")
    cleaned = cleaned.replace(" .", ".").replace(" ,", ",")
    return cleaned


def summary_to_dict(summary: CorrectionEvalSummary) -> Dict[str, object]:
    data = asdict(summary)
    data["by_category"] = {
        category: asdict(metrics) for category, metrics in summary.by_category.items()
    }
    return data

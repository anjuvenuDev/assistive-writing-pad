"""Handwriting recognition evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import statistics
import time
from typing import Dict, Iterable, List, Protocol, Sequence

from jiwer import cer, wer

from assistive_writing_pad.contracts import RecognitionResult, StrokePoint
from assistive_writing_pad.eval.correction_eval import percentile


class StrokeGroupRecognizerLike(Protocol):
    def recognize_stroke_groups(
        self,
        stroke_groups: Sequence[Sequence[StrokePoint]],
        mode: str = "ocr",
    ) -> RecognitionResult:
        """Recognize grouped pen strokes."""


@dataclass(frozen=True)
class RecognitionCase:
    id: str
    expected_text: str
    stroke_groups: Sequence[Sequence[StrokePoint]]
    category: str
    notes: str = ""


@dataclass(frozen=True)
class RecognitionEvalRow:
    id: str
    category: str
    expected_text: str
    output_text: str
    exact_match: bool
    char_error_rate: float
    word_error_rate: float
    latency_ms: float
    confidence: float
    low_confidence: bool


@dataclass(frozen=True)
class RecognitionCategoryMetrics:
    total: int
    exact: int
    accuracy: float
    average_cer: float
    average_wer: float
    average_latency_ms: float
    p95_latency_ms: float
    low_confidence: int


@dataclass(frozen=True)
class RecognitionEvalSummary:
    total: int
    exact: int
    accuracy: float
    average_cer: float
    average_wer: float
    average_latency_ms: float
    p95_latency_ms: float
    low_confidence: int
    by_category: Dict[str, RecognitionCategoryMetrics] = field(default_factory=dict)


@dataclass(frozen=True)
class RecognitionEvalReport:
    summary: RecognitionEvalSummary
    rows: Sequence[RecognitionEvalRow]

    def to_dict(self) -> Dict[str, object]:
        return {
            "summary": recognition_summary_to_dict(self.summary),
            "rows": [asdict(row) for row in self.rows],
        }


def load_recognition_cases(path: Path) -> List[RecognitionCase]:
    cases: List[RecognitionCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            data = json.loads(stripped)
            try:
                cases.append(
                    RecognitionCase(
                        id=str(data["id"]),
                        category=str(data["category"]),
                        expected_text=str(data["expected"]),
                        stroke_groups=parse_stroke_groups(data["strokes"]),
                        notes=str(data.get("notes", "")),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} missing field {exc}") from exc
    return cases


def parse_stroke_groups(value: object) -> Sequence[Sequence[StrokePoint]]:
    if not isinstance(value, list):
        raise ValueError("strokes must be a list of stroke point lists")
    stroke_groups: List[List[StrokePoint]] = []
    for stroke in value:
        if not isinstance(stroke, list):
            raise ValueError("each stroke must be a list")
        points = []
        for raw_point in stroke:
            if not isinstance(raw_point, dict):
                raise ValueError("each point must be an object")
            points.append(
                StrokePoint(
                    x=float(raw_point["x"]),
                    y=float(raw_point["y"]),
                    timestamp_ms=int(raw_point.get("timestamp_ms", 0)),
                    pressure=float(raw_point.get("pressure", 1.0)),
                )
            )
        stroke_groups.append(points)
    return tuple(tuple(stroke) for stroke in stroke_groups)


def evaluate_recognition_cases(
    cases: Iterable[RecognitionCase],
    recognizer: StrokeGroupRecognizerLike,
    *,
    mode: str = "ocr",
    confidence_threshold: float = 0.65,
) -> RecognitionEvalReport:
    rows: List[RecognitionEvalRow] = []
    for case in cases:
        started = time.perf_counter()
        result = recognizer.recognize_stroke_groups(case.stroke_groups, mode=mode)
        latency_ms = (time.perf_counter() - started) * 1000.0

        expected = normalize_for_recognition_eval(case.expected_text)
        output = normalize_for_recognition_eval(result.text)
        rows.append(
            RecognitionEvalRow(
                id=case.id,
                category=case.category,
                expected_text=case.expected_text,
                output_text=result.text,
                exact_match=output == expected,
                char_error_rate=round(safe_cer(expected, output), 4),
                word_error_rate=round(safe_wer(expected, output), 4),
                latency_ms=round(latency_ms, 3),
                confidence=round(result.confidence, 4),
                low_confidence=result.confidence < confidence_threshold,
            )
        )

    return RecognitionEvalReport(summary=summarize_recognition_rows(rows), rows=tuple(rows))


def summarize_recognition_rows(rows: Sequence[RecognitionEvalRow]) -> RecognitionEvalSummary:
    by_category = {
        category: summarize_recognition_category([row for row in rows if row.category == category])
        for category in sorted({row.category for row in rows})
    }
    category_all = summarize_recognition_category(rows)
    return RecognitionEvalSummary(
        total=category_all.total,
        exact=category_all.exact,
        accuracy=category_all.accuracy,
        average_cer=category_all.average_cer,
        average_wer=category_all.average_wer,
        average_latency_ms=category_all.average_latency_ms,
        p95_latency_ms=category_all.p95_latency_ms,
        low_confidence=category_all.low_confidence,
        by_category=by_category,
    )


def summarize_recognition_category(
    rows: Sequence[RecognitionEvalRow],
) -> RecognitionCategoryMetrics:
    total = len(rows)
    exact = sum(1 for row in rows if row.exact_match)
    latencies = [row.latency_ms for row in rows]
    return RecognitionCategoryMetrics(
        total=total,
        exact=exact,
        accuracy=round(exact / total, 4) if total else 0.0,
        average_cer=round(statistics.fmean(row.char_error_rate for row in rows), 4)
        if rows
        else 0.0,
        average_wer=round(statistics.fmean(row.word_error_rate for row in rows), 4)
        if rows
        else 0.0,
        average_latency_ms=round(statistics.fmean(latencies), 3) if latencies else 0.0,
        p95_latency_ms=round(percentile(latencies, 95), 3) if latencies else 0.0,
        low_confidence=sum(1 for row in rows if row.low_confidence),
    )


def normalize_for_recognition_eval(text: str) -> str:
    cleaned = " ".join(text.strip().split()).lower()
    cleaned = cleaned.replace("\u2019", "'")
    cleaned = cleaned.replace(" .", ".").replace(" ,", ",")
    return cleaned


def safe_cer(expected: str, output: str) -> float:
    if not expected and not output:
        return 0.0
    if not expected:
        return 1.0 if output else 0.0
    return float(cer(expected, output))


def safe_wer(expected: str, output: str) -> float:
    if not expected and not output:
        return 0.0
    if not expected:
        return 1.0 if output else 0.0
    return float(wer(expected, output))


def recognition_summary_to_dict(summary: RecognitionEvalSummary) -> Dict[str, object]:
    data = asdict(summary)
    data["by_category"] = {
        category: asdict(metrics) for category, metrics in summary.by_category.items()
    }
    return data

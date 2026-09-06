"""End-to-end handwriting-to-correction evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import statistics
import time
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.contracts import CorrectionResult, RecognitionResult, StrokePoint
from assistive_writing_pad.eval.correction_eval import (
    normalize_for_eval,
    percentile,
)
from assistive_writing_pad.eval.recognition_eval import (
    normalize_for_recognition_eval,
    parse_stroke_groups,
    safe_cer,
    safe_wer,
)
from assistive_writing_pad.pipeline import WritingPipeline


class StrokeGroupRecognizerLike(Protocol):
    def recognize_stroke_groups(
        self,
        stroke_groups: Sequence[Sequence[StrokePoint]],
        mode: str = "ocr",
    ) -> RecognitionResult:
        """Recognize grouped pen strokes."""


class CorrectorLike(Protocol):
    def correct(self, text: str) -> CorrectionResult:
        """Return corrected text for an input string."""


@dataclass(frozen=True)
class EndToEndCase:
    id: str
    expected_corrected_text: str
    stroke_groups: Sequence[Sequence[StrokePoint]]
    category: str
    expected_recognized_text: Optional[str] = None
    source: str = "unknown"
    notes: str = ""


@dataclass(frozen=True)
class EndToEndEvalRow:
    id: str
    category: str
    source: str
    expected_recognized_text: str
    expected_corrected_text: str
    recognized_text: str
    corrected_text: str
    recognition_exact_match: bool
    corrected_exact_match: bool
    recognition_char_error_rate: float
    corrected_char_error_rate: float
    recognition_word_error_rate: float
    corrected_word_error_rate: float
    recognition_latency_ms: float
    correction_latency_ms: float
    total_latency_ms: float
    recognition_confidence: float
    correction_confidence: float
    low_confidence: bool
    needs_review: bool
    correction_count: int


@dataclass(frozen=True)
class EndToEndCategoryMetrics:
    total: int
    recognition_exact: int
    corrected_exact: int
    recognition_accuracy: float
    corrected_accuracy: float
    average_recognition_cer: float
    average_corrected_cer: float
    average_recognition_wer: float
    average_corrected_wer: float
    average_total_latency_ms: float
    p95_total_latency_ms: float
    low_confidence: int
    needs_review: int


@dataclass(frozen=True)
class EndToEndEvalSummary:
    total: int
    recognition_exact: int
    corrected_exact: int
    recognition_accuracy: float
    corrected_accuracy: float
    average_recognition_cer: float
    average_corrected_cer: float
    average_recognition_wer: float
    average_corrected_wer: float
    average_total_latency_ms: float
    p95_total_latency_ms: float
    low_confidence: int
    needs_review: int
    by_category: Dict[str, EndToEndCategoryMetrics] = field(default_factory=dict)


@dataclass(frozen=True)
class EndToEndEvalReport:
    summary: EndToEndEvalSummary
    rows: Sequence[EndToEndEvalRow]

    def to_dict(self) -> Dict[str, object]:
        return {
            "summary": end_to_end_summary_to_dict(self.summary),
            "rows": [asdict(row) for row in self.rows],
        }


def load_end_to_end_cases(path: Path) -> List[EndToEndCase]:
    cases: List[EndToEndCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            data = json.loads(stripped)
            try:
                expected_corrected = str(data.get("expected_corrected", data["expected"]))
                expected_recognized = data.get("expected_recognized")
                cases.append(
                    EndToEndCase(
                        id=str(data["id"]),
                        category=str(data["category"]),
                        expected_corrected_text=expected_corrected,
                        expected_recognized_text=(
                            str(expected_recognized)
                            if expected_recognized is not None
                            else expected_corrected
                        ),
                        source=str(data.get("source", "unknown")),
                        stroke_groups=parse_stroke_groups(data["strokes"]),
                        notes=str(data.get("notes", "")),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} missing field {exc}") from exc
    return cases


def evaluate_end_to_end_cases(
    cases: Iterable[EndToEndCase],
    recognizer: StrokeGroupRecognizerLike,
    corrector: CorrectorLike,
    *,
    settings: Optional[RuntimeSettings] = None,
    mode: str = "ocr",
    confidence_threshold: float = 0.65,
) -> EndToEndEvalReport:
    runtime_settings = settings or RuntimeSettings()
    pipeline = WritingPipeline(
        recognizer=recognizer,  # type: ignore[arg-type]
        corrector=corrector,
        settings=runtime_settings,
    )
    rows: List[EndToEndEvalRow] = []
    for case in cases:
        recognition_started = time.perf_counter()
        recognition = recognizer.recognize_stroke_groups(case.stroke_groups, mode=mode)
        recognition_latency_ms = (time.perf_counter() - recognition_started) * 1000.0

        correction_started = time.perf_counter()
        pipeline_result = pipeline.process_recognition(recognition)
        correction_latency_ms = (time.perf_counter() - correction_started) * 1000.0

        correction = pipeline_result.correction
        expected_recognized = normalize_for_recognition_eval(
            case.expected_recognized_text or case.expected_corrected_text
        )
        expected_corrected = normalize_for_eval(case.expected_corrected_text)
        recognized = normalize_for_recognition_eval(recognition.text)
        corrected = normalize_for_eval(correction.corrected_text)

        rows.append(
            EndToEndEvalRow(
                id=case.id,
                category=case.category,
                source=case.source,
                expected_recognized_text=case.expected_recognized_text
                or case.expected_corrected_text,
                expected_corrected_text=case.expected_corrected_text,
                recognized_text=recognition.text,
                corrected_text=correction.corrected_text,
                recognition_exact_match=recognized == expected_recognized,
                corrected_exact_match=corrected == expected_corrected,
                recognition_char_error_rate=round(safe_cer(expected_recognized, recognized), 4),
                corrected_char_error_rate=round(safe_cer(expected_corrected, corrected), 4),
                recognition_word_error_rate=round(safe_wer(expected_recognized, recognized), 4),
                corrected_word_error_rate=round(safe_wer(expected_corrected, corrected), 4),
                recognition_latency_ms=round(recognition_latency_ms, 3),
                correction_latency_ms=round(correction_latency_ms, 3),
                total_latency_ms=round(recognition_latency_ms + correction_latency_ms, 3),
                recognition_confidence=round(recognition.confidence, 4),
                correction_confidence=round(correction.confidence, 4),
                low_confidence=recognition.confidence < confidence_threshold,
                needs_review=pipeline_result.needs_review,
                correction_count=len(correction.corrections),
            )
        )

    return EndToEndEvalReport(summary=summarize_end_to_end_rows(rows), rows=tuple(rows))


def summarize_end_to_end_rows(rows: Sequence[EndToEndEvalRow]) -> EndToEndEvalSummary:
    by_category = {
        category: summarize_end_to_end_category([row for row in rows if row.category == category])
        for category in sorted({row.category for row in rows})
    }
    category_all = summarize_end_to_end_category(rows)
    return EndToEndEvalSummary(
        total=category_all.total,
        recognition_exact=category_all.recognition_exact,
        corrected_exact=category_all.corrected_exact,
        recognition_accuracy=category_all.recognition_accuracy,
        corrected_accuracy=category_all.corrected_accuracy,
        average_recognition_cer=category_all.average_recognition_cer,
        average_corrected_cer=category_all.average_corrected_cer,
        average_recognition_wer=category_all.average_recognition_wer,
        average_corrected_wer=category_all.average_corrected_wer,
        average_total_latency_ms=category_all.average_total_latency_ms,
        p95_total_latency_ms=category_all.p95_total_latency_ms,
        low_confidence=category_all.low_confidence,
        needs_review=category_all.needs_review,
        by_category=by_category,
    )


def summarize_end_to_end_category(
    rows: Sequence[EndToEndEvalRow],
) -> EndToEndCategoryMetrics:
    total = len(rows)
    recognition_exact = sum(1 for row in rows if row.recognition_exact_match)
    corrected_exact = sum(1 for row in rows if row.corrected_exact_match)
    latencies = [row.total_latency_ms for row in rows]
    return EndToEndCategoryMetrics(
        total=total,
        recognition_exact=recognition_exact,
        corrected_exact=corrected_exact,
        recognition_accuracy=round(recognition_exact / total, 4) if total else 0.0,
        corrected_accuracy=round(corrected_exact / total, 4) if total else 0.0,
        average_recognition_cer=round(
            statistics.fmean(row.recognition_char_error_rate for row in rows), 4
        )
        if rows
        else 0.0,
        average_corrected_cer=round(
            statistics.fmean(row.corrected_char_error_rate for row in rows), 4
        )
        if rows
        else 0.0,
        average_recognition_wer=round(
            statistics.fmean(row.recognition_word_error_rate for row in rows), 4
        )
        if rows
        else 0.0,
        average_corrected_wer=round(
            statistics.fmean(row.corrected_word_error_rate for row in rows), 4
        )
        if rows
        else 0.0,
        average_total_latency_ms=round(statistics.fmean(latencies), 3)
        if latencies
        else 0.0,
        p95_total_latency_ms=round(percentile(latencies, 95), 3) if latencies else 0.0,
        low_confidence=sum(1 for row in rows if row.low_confidence),
        needs_review=sum(1 for row in rows if row.needs_review),
    )


def end_to_end_summary_to_dict(summary: EndToEndEvalSummary) -> Dict[str, object]:
    data = asdict(summary)
    data["by_category"] = {
        category: asdict(metrics) for category, metrics in summary.by_category.items()
    }
    return data

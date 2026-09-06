"""Aggregate evaluation evidence into a release-readiness report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
from typing import Dict, Mapping, Sequence

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.eval.coverage import CoverageReport


@dataclass(frozen=True)
class GateFinding:
    gate: str
    metric: str
    actual: object
    expected: object
    passed: bool


@dataclass(frozen=True)
class ReadinessReport:
    generated_at: str
    production_ready: bool
    smoke_ready: bool
    model_configuration: Mapping[str, object]
    correction_summary: Mapping[str, object]
    end_to_end_summary: Mapping[str, object]
    smoke_coverage: Mapping[str, object]
    production_coverage: Mapping[str, object]
    findings: Sequence[GateFinding] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "production_ready": self.production_ready,
            "smoke_ready": self.smoke_ready,
            "model_configuration": dict(self.model_configuration),
            "correction_summary": dict(self.correction_summary),
            "end_to_end_summary": dict(self.end_to_end_summary),
            "smoke_coverage": dict(self.smoke_coverage),
            "production_coverage": dict(self.production_coverage),
            "findings": [asdict(finding) for finding in self.findings],
        }


@dataclass(frozen=True)
class ReadinessThresholds:
    min_correction_accuracy: float = 0.98
    max_correction_false_positives: int = 0
    max_correction_missed_corrections: int = 0
    min_end_to_end_recognition_accuracy: float = 0.95
    min_end_to_end_corrected_accuracy: float = 0.98
    max_end_to_end_average_recognition_cer: float = 0.05
    max_end_to_end_average_corrected_cer: float = 0.02
    max_end_to_end_low_confidence: int = 0
    max_end_to_end_needs_review: int = 0


def build_readiness_report(
    *,
    correction_report: Mapping[str, object],
    end_to_end_report: Mapping[str, object],
    smoke_coverage: CoverageReport,
    production_coverage: CoverageReport,
    settings: RuntimeSettings,
    thresholds: ReadinessThresholds = ReadinessThresholds(),
) -> ReadinessReport:
    correction_summary = dict(correction_report.get("summary", {}))
    end_to_end_summary = dict(end_to_end_report.get("summary", {}))

    findings = [
        GateFinding(
            gate="correction",
            metric="accuracy",
            actual=correction_summary.get("accuracy", 0.0),
            expected=f">= {thresholds.min_correction_accuracy}",
            passed=float(correction_summary.get("accuracy", 0.0))
            >= thresholds.min_correction_accuracy,
        ),
        GateFinding(
            gate="correction",
            metric="false_positives",
            actual=correction_summary.get("false_positives", 0),
            expected=f"<= {thresholds.max_correction_false_positives}",
            passed=int(correction_summary.get("false_positives", 0))
            <= thresholds.max_correction_false_positives,
        ),
        GateFinding(
            gate="correction",
            metric="missed_corrections",
            actual=correction_summary.get("missed_corrections", 0),
            expected=f"<= {thresholds.max_correction_missed_corrections}",
            passed=int(correction_summary.get("missed_corrections", 0))
            <= thresholds.max_correction_missed_corrections,
        ),
        GateFinding(
            gate="end_to_end",
            metric="recognition_accuracy",
            actual=end_to_end_summary.get("recognition_accuracy", 0.0),
            expected=f">= {thresholds.min_end_to_end_recognition_accuracy}",
            passed=float(end_to_end_summary.get("recognition_accuracy", 0.0))
            >= thresholds.min_end_to_end_recognition_accuracy,
        ),
        GateFinding(
            gate="end_to_end",
            metric="corrected_accuracy",
            actual=end_to_end_summary.get("corrected_accuracy", 0.0),
            expected=f">= {thresholds.min_end_to_end_corrected_accuracy}",
            passed=float(end_to_end_summary.get("corrected_accuracy", 0.0))
            >= thresholds.min_end_to_end_corrected_accuracy,
        ),
        GateFinding(
            gate="end_to_end",
            metric="average_recognition_cer",
            actual=end_to_end_summary.get("average_recognition_cer", 1.0),
            expected=f"<= {thresholds.max_end_to_end_average_recognition_cer}",
            passed=float(end_to_end_summary.get("average_recognition_cer", 1.0))
            <= thresholds.max_end_to_end_average_recognition_cer,
        ),
        GateFinding(
            gate="end_to_end",
            metric="average_corrected_cer",
            actual=end_to_end_summary.get("average_corrected_cer", 1.0),
            expected=f"<= {thresholds.max_end_to_end_average_corrected_cer}",
            passed=float(end_to_end_summary.get("average_corrected_cer", 1.0))
            <= thresholds.max_end_to_end_average_corrected_cer,
        ),
        GateFinding(
            gate="end_to_end",
            metric="low_confidence",
            actual=end_to_end_summary.get("low_confidence", 0),
            expected=f"<= {thresholds.max_end_to_end_low_confidence}",
            passed=int(end_to_end_summary.get("low_confidence", 0))
            <= thresholds.max_end_to_end_low_confidence,
        ),
        GateFinding(
            gate="end_to_end",
            metric="needs_review",
            actual=end_to_end_summary.get("needs_review", 0),
            expected=f"<= {thresholds.max_end_to_end_needs_review}",
            passed=int(end_to_end_summary.get("needs_review", 0))
            <= thresholds.max_end_to_end_needs_review,
        ),
        GateFinding(
            gate="coverage",
            metric="smoke_profile",
            actual=smoke_coverage.passed,
            expected=True,
            passed=smoke_coverage.passed,
        ),
        GateFinding(
            gate="coverage",
            metric="production_profile",
            actual=production_coverage.passed,
            expected=True,
            passed=production_coverage.passed,
        ),
    ]
    smoke_ready = all(
        finding.passed
        for finding in findings
        if not (finding.gate == "coverage" and finding.metric == "production_profile")
    )
    production_ready = smoke_ready and production_coverage.passed

    return ReadinessReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        production_ready=production_ready,
        smoke_ready=smoke_ready,
        model_configuration=model_configuration(settings),
        correction_summary=correction_summary,
        end_to_end_summary=end_to_end_summary,
        smoke_coverage=smoke_coverage.to_dict(),
        production_coverage=production_coverage.to_dict(),
        findings=tuple(findings),
    )


def model_configuration(settings: RuntimeSettings) -> Dict[str, object]:
    return {
        "device_profile": settings.device_profile,
        "hf_cache_dir": str(settings.hf_cache_dir),
        "trocr_model": os.environ.get("AWP_TROCR_MODEL", "microsoft/trocr-base-handwritten"),
        "trocr_local_files_only": _bool_env("AWP_TROCR_LOCAL_FILES_ONLY", False),
        "correction_mode": settings.correction_mode,
        "hf_spelling_model_enabled": settings.hf_spelling_model_enabled,
        "hf_spelling_model": settings.hf_spelling_model,
        "hf_semantic_model_enabled": settings.hf_semantic_model_enabled,
        "hf_semantic_model": settings.hf_semantic_model,
        "hf_grammar_model_enabled": settings.hf_grammar_model_enabled,
        "hf_grammar_model": settings.hf_grammar_model,
        "hf_correction_local_files_only": settings.hf_correction_local_files_only,
        "evaluation_capture_enabled": settings.evaluation_capture_enabled,
        "evaluation_manifest_path": str(settings.evaluation_manifest_path),
    }


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

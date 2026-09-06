from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.eval.coverage import (
    CategoryRequirement,
    CoverageReport,
    evaluate_coverage,
)
from assistive_writing_pad.eval.end_to_end_eval import EndToEndCase
from assistive_writing_pad.eval.readiness import build_readiness_report, model_configuration


def case(case_id: str, category: str, source: str) -> EndToEndCase:
    return EndToEndCase(
        id=case_id,
        category=category,
        source=source,
        expected_corrected_text="x",
        expected_recognized_text="x",
        stroke_groups=(),
    )


def passing_correction_report():
    return {
        "summary": {
            "accuracy": 1.0,
            "false_positives": 0,
            "missed_corrections": 0,
        }
    }


def passing_end_to_end_report():
    return {
        "summary": {
            "recognition_accuracy": 1.0,
            "corrected_accuracy": 1.0,
            "average_recognition_cer": 0.0,
            "average_corrected_cer": 0.0,
            "low_confidence": 0,
            "needs_review": 0,
        }
    }


def test_readiness_report_is_smoke_ready_but_not_production_ready_without_manual_coverage():
    smoke_coverage = CoverageReport(
        total_cases=2,
        manual_cases=0,
        by_category={"single_character": 2},
        by_source={"smoke": 2},
    )
    production_coverage = CoverageReport(
        total_cases=2,
        manual_cases=0,
        by_category={"single_character": 2},
        by_source={"smoke": 2},
        findings=(
            # Lightweight object shape is enough because the report only needs passed.
        ),
    )
    production_coverage = evaluate_coverage(
        [case("single_h_001", "single_character", "smoke")],
        requirements=(CategoryRequirement("word", min_cases=1, min_manual_cases=1),),
    )

    report = build_readiness_report(
        correction_report=passing_correction_report(),
        end_to_end_report=passing_end_to_end_report(),
        smoke_coverage=smoke_coverage,
        production_coverage=production_coverage,
        settings=RuntimeSettings(),
    )

    assert report.smoke_ready is True
    assert report.production_ready is False
    assert any(
        finding.gate == "coverage" and finding.metric == "production_profile"
        for finding in report.findings
        if not finding.passed
    )


def test_readiness_report_fails_smoke_when_model_metrics_fail():
    report = build_readiness_report(
        correction_report={"summary": {"accuracy": 0.5, "false_positives": 0, "missed_corrections": 0}},
        end_to_end_report=passing_end_to_end_report(),
        smoke_coverage=CoverageReport(
            total_cases=2,
            manual_cases=0,
            by_category={"single_character": 2},
            by_source={"smoke": 2},
        ),
        production_coverage=CoverageReport(
            total_cases=2,
            manual_cases=0,
            by_category={"single_character": 2},
            by_source={"smoke": 2},
        ),
        settings=RuntimeSettings(),
    )

    assert report.smoke_ready is False
    assert report.production_ready is False
    assert report.to_dict()["findings"][0]["passed"] is False


def test_model_configuration_includes_ocr_model_flags(monkeypatch):
    monkeypatch.setenv("AWP_TROCR_MODEL", "microsoft/trocr-small-handwritten")
    monkeypatch.setenv("AWP_TROCR_LOCAL_FILES_ONLY", "1")

    configuration = model_configuration(RuntimeSettings())

    assert configuration["trocr_model"] == "microsoft/trocr-small-handwritten"
    assert configuration["trocr_local_files_only"] is True

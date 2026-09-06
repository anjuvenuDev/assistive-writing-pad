from assistive_writing_pad.eval.coverage import (
    CategoryRequirement,
    evaluate_coverage,
    requirements_for_profile,
)
from assistive_writing_pad.eval.end_to_end_eval import EndToEndCase


def case(case_id: str, category: str, source: str) -> EndToEndCase:
    return EndToEndCase(
        id=case_id,
        category=category,
        source=source,
        expected_corrected_text="x",
        expected_recognized_text="x",
        stroke_groups=(),
    )


def test_smoke_coverage_passes_current_minimum_shape() -> None:
    report = evaluate_coverage(
        [
            case("single_h_001", "single_character", "smoke"),
            case("single_i_001", "single_character", "smoke"),
        ],
        requirements=requirements_for_profile("smoke"),
    )

    assert report.passed
    assert report.total_cases == 2
    assert report.by_category["single_character"] == 2
    assert report.by_source["smoke"] == 2


def test_production_coverage_requires_manual_word_and_sentence_cases() -> None:
    report = evaluate_coverage(
        [case("single_h_001", "single_character", "smoke")],
        requirements=requirements_for_profile("production"),
    )

    assert not report.passed
    failed = [finding.requirement for finding in report.findings if not finding.passed]
    assert "single_character.min_manual_cases" in failed
    assert "word.min_cases" in failed
    assert "sentence.min_cases" in failed


def test_custom_coverage_requirement_counts_manual_cases() -> None:
    report = evaluate_coverage(
        [
            case("word_001", "word", "manual"),
            case("word_002", "word", "synthetic"),
        ],
        requirements=(CategoryRequirement("word", min_cases=2, min_manual_cases=1),),
    )

    assert report.passed

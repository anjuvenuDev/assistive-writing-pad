import json

from assistive_writing_pad.contracts import CorrectionResult
from assistive_writing_pad.eval.correction_eval import (
    CorrectionCase,
    evaluate_correction_cases,
    load_correction_cases,
    normalize_for_eval,
)


class MappingCorrector:
    def __init__(self, outputs):
        self.outputs = outputs

    def correct(self, text: str) -> CorrectionResult:
        output = self.outputs.get(text, text)
        return CorrectionResult(
            original_text=text,
            corrected_text=output,
            confidence=0.9,
        )


def test_load_correction_cases_from_jsonl(tmp_path) -> None:
    manifest = tmp_path / "cases.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "case_1",
                "category": "spelling",
                "input": "teh cat",
                "expected": "the cat",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_correction_cases(manifest)

    assert cases == [
        CorrectionCase(
            id="case_1",
            category="spelling",
            input_text="teh cat",
            expected_text="the cat",
        )
    ]


def test_evaluate_correction_cases_summarizes_accuracy_and_false_positives() -> None:
    cases = [
        CorrectionCase("spell_ok", "teh cat", "the cat", "spelling"),
        CorrectionCase("missed", "I can here it", "I can hear it", "semantic"),
        CorrectionCase("clean_bad", "The cat sat.", "The cat sat.", "clean"),
    ]
    corrector = MappingCorrector(
        {
            "teh cat": "the cat",
            "The cat sat.": "The dog sat.",
        }
    )

    report = evaluate_correction_cases(cases, corrector)

    assert report.summary.total == 3
    assert report.summary.exact == 1
    assert report.summary.accuracy == 0.3333
    assert report.summary.false_positives == 1
    assert report.summary.missed_corrections == 1
    assert report.summary.by_category["clean"].false_positive_rate == 1.0
    assert [row.id for row in report.rows if row.false_positive] == ["clean_bad"]


def test_normalize_for_eval_ignores_case_spacing_and_curly_apostrophes() -> None:
    assert normalize_for_eval("  I\u2019m  here . ") == "i'm here."

"""Evaluation helpers."""

from assistive_writing_pad.eval.correction_eval import (
    CategoryMetrics,
    CorrectionCase,
    CorrectionEvalReport,
    CorrectionEvalRow,
    CorrectionEvalSummary,
    evaluate_correction_cases,
    load_correction_cases,
    normalize_for_eval,
)
from assistive_writing_pad.eval.corpus import (
    append_jsonl_record,
    build_end_to_end_case_record,
    stroke_groups_from_capture_payload,
    stroke_groups_to_jsonable,
)
from assistive_writing_pad.eval.coverage import (
    CategoryRequirement,
    CoverageFinding,
    CoverageReport,
    evaluate_coverage,
    requirements_for_profile,
)
from assistive_writing_pad.eval.end_to_end_eval import (
    EndToEndCase,
    EndToEndCategoryMetrics,
    EndToEndEvalReport,
    EndToEndEvalRow,
    EndToEndEvalSummary,
    evaluate_end_to_end_cases,
    load_end_to_end_cases,
)
from assistive_writing_pad.eval.recognition_eval import (
    RecognitionCase,
    RecognitionCategoryMetrics,
    RecognitionEvalReport,
    RecognitionEvalRow,
    RecognitionEvalSummary,
    evaluate_recognition_cases,
    load_recognition_cases,
    normalize_for_recognition_eval,
)

__all__ = [
    "CategoryMetrics",
    "CorrectionCase",
    "CorrectionEvalReport",
    "CorrectionEvalRow",
    "CorrectionEvalSummary",
    "evaluate_correction_cases",
    "load_correction_cases",
    "normalize_for_eval",
    "append_jsonl_record",
    "build_end_to_end_case_record",
    "stroke_groups_from_capture_payload",
    "stroke_groups_to_jsonable",
    "CategoryRequirement",
    "CoverageFinding",
    "CoverageReport",
    "evaluate_coverage",
    "requirements_for_profile",
    "EndToEndCase",
    "EndToEndCategoryMetrics",
    "EndToEndEvalReport",
    "EndToEndEvalRow",
    "EndToEndEvalSummary",
    "evaluate_end_to_end_cases",
    "load_end_to_end_cases",
    "RecognitionCase",
    "RecognitionCategoryMetrics",
    "RecognitionEvalReport",
    "RecognitionEvalRow",
    "RecognitionEvalSummary",
    "evaluate_recognition_cases",
    "load_recognition_cases",
    "normalize_for_recognition_eval",
]

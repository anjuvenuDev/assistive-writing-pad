import json
from typing import Dict, Sequence

from assistive_writing_pad.correction.huggingface import (
    GeneratedCorrection,
    HuggingFaceCorrectionPipeline,
    ModelCorrectionUnavailable,
    diff_corrections,
    is_acceptable_model_output,
    normalize_generated_text,
)


class FakeRunner:
    def __init__(
        self,
        stage: str,
        outputs: Dict[str, Sequence[GeneratedCorrection]],
        model_name: str = "fake-model",
    ) -> None:
        self.stage = stage
        self.model_name = model_name
        self.outputs = outputs

    def generate(self, text: str) -> Sequence[GeneratedCorrection]:
        return self.outputs.get(
            text,
            [GeneratedCorrection(text=text, confidence=0.99, model_name=self.model_name, stage=self.stage)],
        )


class BrokenRunner:
    stage = "grammar"
    model_name = "broken-model"

    def generate(self, text: str) -> Sequence[GeneratedCorrection]:
        raise ModelCorrectionUnavailable("missing cached model")


def generated(text: str, confidence: float, stage: str, model: str = "fake-model") -> GeneratedCorrection:
    return GeneratedCorrection(text=text, confidence=confidence, model_name=model, stage=stage)


def test_huggingface_pipeline_applies_spelling_then_grammar() -> None:
    spelling = FakeRunner(
        stage="spelling",
        model_name="spelling-model",
        outputs={
            "teh chlid writng": [
                generated("the child writing", 0.91, "spelling", "spelling-model"),
                generated("the child writting", 0.70, "spelling", "spelling-model"),
            ]
        },
    )
    grammar = FakeRunner(
        stage="grammar",
        model_name="grammar-model",
        outputs={
            "the child writing": [
                generated("the child is writing.", 0.88, "grammar", "grammar-model")
            ]
        },
    )

    result = HuggingFaceCorrectionPipeline(
        spelling_runner=spelling,
        grammar_runner=grammar,
    ).correct("teh chlid writng")

    assert result.corrected_text == "the child is writing."
    assert result.confidence == 0.88
    assert result.corrections == (
        result.corrections[0],
    )
    assert result.corrections[0].original == "teh chlid writng"
    assert result.corrections[0].corrected == "the child is writing."
    assert result.corrections[0].reason == "hf_model_pipeline"
    assert result.metadata["backend"] == "huggingface"

    stages = json.loads(result.metadata["stages"])
    assert [(item["stage"], item["accepted"]) for item in stages] == [
        ("spelling", True),
        ("grammar", True),
    ]
    assert stages[0]["alternatives"][0]["text"] == "the child writing"


def test_huggingface_pipeline_preserves_clean_text_when_models_agree() -> None:
    spelling = FakeRunner(stage="spelling", outputs={})
    grammar = FakeRunner(stage="grammar", outputs={})

    result = HuggingFaceCorrectionPipeline(
        spelling_runner=spelling,
        grammar_runner=grammar,
    ).correct("The child is writing.")

    assert result.corrected_text == "The child is writing."
    assert result.corrections == ()
    assert result.confidence == 1.0


def test_huggingface_pipeline_rejects_hallucinated_generation() -> None:
    spelling = FakeRunner(
        stage="spelling",
        outputs={
            "cat": [
                generated(
                    "The complete history of astronomy is unrelated to the input.",
                    0.99,
                    "spelling",
                )
            ]
        },
    )

    result = HuggingFaceCorrectionPipeline(spelling_runner=spelling).correct("cat")

    assert result.corrected_text == "cat"
    assert result.corrections == ()
    assert json.loads(result.metadata["stages"])[0]["accepted"] is False


def test_huggingface_pipeline_reports_model_errors_without_fake_corrections() -> None:
    result = HuggingFaceCorrectionPipeline(grammar_runner=BrokenRunner()).correct("teh cat")

    assert result.corrected_text == "teh cat"
    assert result.corrections == ()
    assert json.loads(result.metadata["errors"])[0]["model"] == "broken-model"


def test_model_output_guard_rejects_unrelated_long_text() -> None:
    assert not is_acceptable_model_output(
        "I lik swiming",
        "This generated paragraph talks about something completely different.",
    )


def test_diff_corrections_formats_insertions_and_punctuation() -> None:
    corrections = diff_corrections(
        "I like to swimming",
        "I like swimming.",
        confidence=0.9,
        reason="hf_grammar_model",
    )

    assert [(item.original, item.corrected) for item in corrections] == [
        ("to", ""),
        ("", "."),
    ]


def test_normalize_generated_text_repairs_spacing_before_punctuation() -> None:
    assert normalize_generated_text(" I like swimming . ") == "I like swimming."

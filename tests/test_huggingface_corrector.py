import json
from typing import Dict, Sequence

from assistive_writing_pad.correction.huggingface import (
    GeneratedCorrection,
    HuggingFaceCorrectionPipeline,
    ModelCorrectionUnavailable,
    WordfreqFragmentCorrectionRunner,
    best_fragment_merge_candidate,
    diff_corrections,
    finalize_grammar_output,
    is_acceptable_model_output,
    is_isolated_character_input,
    is_probable_word_spelling_change,
    is_single_word_fragment_input,
    is_word_or_ocr_fragment_output,
    normalize_generated_text,
    repair_ocr_fragments,
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


class WarmableRunner(FakeRunner):
    def __init__(
        self,
        stage: str,
        outputs: Dict[str, Sequence[GeneratedCorrection]],
        warm_log: list[str],
        model_name: str = "fake-model",
    ) -> None:
        super().__init__(stage=stage, outputs=outputs, model_name=model_name)
        self.warm_log = warm_log

    def warm_up(self) -> None:
        self.warm_log.append(self.stage)


def generated(text: str, confidence: float, stage: str, model: str = "fake-model") -> GeneratedCorrection:
    return GeneratedCorrection(text=text, confidence=confidence, model_name=model, stage=stage)


def test_huggingface_pipeline_applies_spelling_then_grammar() -> None:
    lexical = FakeRunner(
        stage="lexical",
        model_name="lexical-model",
        outputs={
            "teh chlid writng": [
                generated("teh chlid writng", 0.99, "lexical", "lexical-model")
            ]
        },
    )
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
        lexical_runner=lexical,
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
        ("lexical", True),
        ("spelling", True),
        ("grammar", True),
    ]
    assert stages[1]["alternatives"][0]["text"] == "the child writing"


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


def test_huggingface_pipeline_rejects_unrequested_punctuation_only_change() -> None:
    spelling = FakeRunner(
        stage="spelling",
        outputs={
            "Please analyze the central idea conveyed": [
                generated(
                    "Please analyze the central idea conveyed.",
                    0.99,
                    "spelling",
                )
            ]
        },
    )

    result = HuggingFaceCorrectionPipeline(spelling_runner=spelling).correct(
        "Please analyze the central idea conveyed"
    )

    assert result.corrected_text == "Please analyze the central idea conveyed"
    assert result.corrections == ()
    assert json.loads(result.metadata["stages"])[0]["accepted"] is False


def test_huggingface_pipeline_prefers_period_for_trailing_quote_artifact() -> None:
    grammar = FakeRunner(
        stage="grammar",
        outputs={
            "By this text '.": [
                generated("By this text:", 0.57, "grammar"),
                generated("By this text.", 0.48, "grammar"),
            ]
        },
    )

    result = HuggingFaceCorrectionPipeline(grammar_runner=grammar).correct("By this text '.")

    assert result.corrected_text == "By this text."
    stages = json.loads(result.metadata["stages"])
    assert stages[0]["accepted"] is True


def test_huggingface_pipeline_skips_isolated_character_input() -> None:
    grammar = FakeRunner(
        stage="grammar",
        outputs={"h": [generated("H.", 0.99, "grammar")]},
    )

    result = HuggingFaceCorrectionPipeline(grammar_runner=grammar).correct("h")

    assert result.corrected_text == "h"
    assert result.corrections == ()
    assert result.confidence == 1.0
    assert result.metadata["skipped"] == "isolated_character"


def test_huggingface_pipeline_uses_spelling_only_for_single_word_fragments() -> None:
    spelling = FakeRunner(
        stage="spelling",
        outputs={"teh": [generated("the", 0.91, "spelling")]},
    )
    grammar = FakeRunner(
        stage="grammar",
        outputs={"the": [generated("The.", 0.99, "grammar")]},
    )

    result = HuggingFaceCorrectionPipeline(
        spelling_runner=spelling,
        grammar_runner=grammar,
    ).correct("teh")

    assert result.corrected_text == "the"
    stages = json.loads(result.metadata["stages"])
    assert stages[0]["accepted"] is True
    assert stages[1]["skipped"] == "single_word_fragment"


def test_huggingface_pipeline_repairs_ocr_fragments_before_spelling() -> None:
    spelling = FakeRunner(stage="spelling", outputs={})

    result = HuggingFaceCorrectionPipeline(
        lexical_runner=WordfreqFragmentCorrectionRunner(),
        spelling_runner=spelling,
    ).correct("a nalyze")

    assert result.corrected_text == "analyze"
    assert result.corrections[0].reason == "wordfreq_fragment_model"


def test_huggingface_pipeline_rejects_sentence_output_for_ocr_word_fragment() -> None:
    spelling = FakeRunner(
        stage="spelling",
        outputs={"a nalyze": [generated("A smile.", 0.91, "spelling")]},
    )

    result = HuggingFaceCorrectionPipeline(spelling_runner=spelling).correct("a nalyze")

    assert result.corrected_text == "a nalyze"
    assert result.corrections == ()
    assert json.loads(result.metadata["stages"])[0]["accepted"] is False


def test_huggingface_pipeline_rejects_valid_word_spelling_rewrite() -> None:
    spelling = FakeRunner(
        stage="spelling",
        outputs={
            "Please analyze the central idea conveyed": [
                generated(
                    "Please analyze the central ideas conveyed.",
                    0.91,
                    "spelling",
                )
            ]
        },
    )

    result = HuggingFaceCorrectionPipeline(spelling_runner=spelling).correct(
        "Please analyze the central idea conveyed"
    )

    assert result.corrected_text == "Please analyze the central idea conveyed"
    assert result.corrections == ()


def test_huggingface_pipeline_adds_period_after_substantive_grammar_fix() -> None:
    grammar = FakeRunner(
        stage="grammar",
        outputs={
            "There book is on table": [
                generated("There is a book on the table", 0.82, "grammar")
            ]
        },
    )

    result = HuggingFaceCorrectionPipeline(grammar_runner=grammar).correct(
        "There book is on table"
    )

    assert result.corrected_text == "There is a book on the table."


def test_huggingface_pipeline_applies_semantic_stage_between_models() -> None:
    semantic = FakeRunner(
        stage="semantic",
        model_name="semantic-model",
        outputs={
            "I can here the bell.": [
                generated("I can hear the bell.", 0.82, "semantic", "semantic-model")
            ]
        },
    )
    grammar = FakeRunner(stage="grammar", outputs={})

    result = HuggingFaceCorrectionPipeline(
        semantic_runner=semantic,
        grammar_runner=grammar,
    ).correct("I can here the bell.")

    assert result.corrected_text == "I can hear the bell."
    assert result.corrections[0].reason == "hf_semantic_model"
    assert [item["stage"] for item in json.loads(result.metadata["stages"])] == [
        "semantic",
        "grammar",
    ]


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


def test_huggingface_pipeline_warms_active_runners_in_stage_order() -> None:
    warm_log: list[str] = []
    spelling = WarmableRunner("spelling", {}, warm_log)
    semantic = WarmableRunner("semantic", {}, warm_log)
    grammar = WarmableRunner("grammar", {}, warm_log)

    HuggingFaceCorrectionPipeline(
        spelling_runner=spelling,
        semantic_runner=semantic,
        grammar_runner=grammar,
    ).warm_up()

    assert warm_log == ["spelling", "semantic", "grammar"]


def test_huggingface_pipeline_automatically_selects_corpus_ranked_ocr_hypothesis() -> None:
    pipeline = HuggingFaceCorrectionPipeline()

    selection = pipeline.select_recognition_candidate(
        (("I fed the cot", 0.91), ("I fed the cat", 0.76))
    )

    assert selection.text == "I fed the cat"
    assert selection.metadata["selector"] == "trocr+wordfreq_corpus"
    assert selection.rankings[0][0] == "I fed the cat"


def test_huggingface_pipeline_preserves_primary_for_context_free_word_hypotheses() -> None:
    pipeline = HuggingFaceCorrectionPipeline(
        lexical_runner=WordfreqFragmentCorrectionRunner()
    )

    selection = pipeline.select_recognition_candidate(
        (("i plea", 0.88), ("i clear", 0.84))
    )

    assert selection.text == "i plea"


def test_huggingface_pipeline_requires_material_score_gain_to_replace_primary() -> None:
    pipeline = HuggingFaceCorrectionPipeline()

    selection = pipeline.select_recognition_candidate(
        (
            ("Please a nalyze the central i plea conveyed", 0.9998),
            ("Please a na lyze the central i plea conveyed", 0.9905),
        )
    )

    assert selection.text == "Please a nalyze the central i plea conveyed"


def test_huggingface_pipeline_uses_configured_cache_dir(tmp_path) -> None:
    from assistive_writing_pad.config.settings import RuntimeSettings

    cache_dir = tmp_path / "hf-cache"
    pipeline = HuggingFaceCorrectionPipeline.from_settings(
        RuntimeSettings(
            hf_cache_dir=cache_dir,
            hf_semantic_model_enabled=False,
            hf_grammar_model_enabled=False,
        )
    )

    assert pipeline.spelling_runner is not None
    assert getattr(pipeline.spelling_runner, "cache_dir") == cache_dir


def test_model_output_guard_rejects_unrelated_long_text() -> None:
    assert not is_acceptable_model_output(
        "I lik swiming",
        "This generated paragraph talks about something completely different.",
    )


def test_isolated_character_input_detects_letters_only() -> None:
    assert is_isolated_character_input(" h ")
    assert not is_isolated_character_input("hi")
    assert not is_isolated_character_input("1")
    assert not is_isolated_character_input("?")


def test_single_word_fragment_input_detects_plain_words_only() -> None:
    assert is_single_word_fragment_input("the")
    assert is_single_word_fragment_input("can't")
    assert not is_single_word_fragment_input("the cat")
    assert not is_single_word_fragment_input("the.")


def test_word_or_ocr_fragment_output_allows_only_one_unpunctuated_word() -> None:
    assert is_word_or_ocr_fragment_output("analyze")
    assert not is_word_or_ocr_fragment_output("A smile.")
    assert not is_word_or_ocr_fragment_output("Please.")
    assert not is_word_or_ocr_fragment_output("a nalyze")


def test_repair_ocr_fragments_merges_frequency_ranked_word_fragments() -> None:
    assert repair_ocr_fragments("a nalyze") == "analyze"
    assert repair_ocr_fragments("centra I") == "central"
    assert repair_ocr_fragments("Please a nalyze this") == "Please analyze this"


def test_best_fragment_merge_candidate_uses_frequency_and_edit_distance() -> None:
    runner = WordfreqFragmentCorrectionRunner()

    assert (
        best_fragment_merge_candidate(
            "i",
            "plea",
            language="en",
            top_n=runner.top_n,
            min_zipf=runner.min_zipf,
            min_margin=runner.min_margin,
            distance_penalty=runner.distance_penalty,
        )
        == "idea"
    )


def test_probable_word_spelling_change_uses_frequency_and_edit_distance() -> None:
    assert is_probable_word_spelling_change("teh", "the", language="en")
    assert is_probable_word_spelling_change("chlid", "child", language="en")
    assert not is_probable_word_spelling_change("idea", "ideas", language="en")
    assert not is_probable_word_spelling_change("conveyed", "conversely", language="en")


def test_finalize_grammar_output_only_punctuates_substantive_sentence_fixes() -> None:
    assert (
        finalize_grammar_output("There book is on table", "There is a book on the table")
        == "There is a book on the table."
    )
    assert (
        finalize_grammar_output(
            "Please analyze the central idea conveyed",
            "Please analyze the central idea conveyed",
        )
        == "Please analyze the central idea conveyed"
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

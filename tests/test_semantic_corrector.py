from typing import Mapping, Sequence

from assistive_writing_pad.correction.semantic import SemanticCorrectionRunner


class StaticSemanticScorer:
    mask_token = "[MASK]"

    def __init__(self, scores: Mapping[str, Mapping[str, float]]) -> None:
        self.scores = scores

    def score_candidates(self, masked_text: str, candidates: Sequence[str]) -> Mapping[str, float]:
        return {
            candidate: self.scores.get(masked_text, {}).get(candidate, 0.0)
            for candidate in candidates
        }


class WarmableSemanticScorer(StaticSemanticScorer):
    def __init__(self, scores: Mapping[str, Mapping[str, float]]) -> None:
        super().__init__(scores)
        self.warm_up_count = 0

    def warm_up(self) -> None:
        self.warm_up_count += 1


def test_semantic_runner_corrects_real_word_error_with_context_score() -> None:
    runner = SemanticCorrectionRunner(
        model_name="fake-semantic",
        scorer=StaticSemanticScorer(
            {
                "I can [MASK] the bell.": {
                    "hear": 0.24,
                    "here": 0.01,
                }
            }
        ),
    )

    result = runner.generate("I can here the bell.")

    assert result[0].text == "I can hear the bell."
    assert result[0].stage == "semantic"
    assert result[0].confidence > 0.80


def test_semantic_runner_preserves_text_when_margin_is_weak() -> None:
    runner = SemanticCorrectionRunner(
        model_name="fake-semantic",
        scorer=StaticSemanticScorer(
            {
                "Please [MASK] a sentence.": {
                    "write": 0.12,
                    "right": 0.10,
                    "rite": 0.01,
                }
            }
        ),
    )

    result = runner.generate("Please right a sentence.")

    assert result[0].text == "Please right a sentence."
    assert result[0].confidence == 1.0


def test_semantic_runner_handles_multiple_context_choices() -> None:
    runner = SemanticCorrectionRunner(
        model_name="fake-semantic",
        scorer=StaticSemanticScorer(
            {
                "I went [MASK] school and their is a book.": {
                    "to": 0.28,
                    "too": 0.03,
                    "two": 0.01,
                },
                "I went too school and [MASK] is a book.": {
                    "there": 0.22,
                    "their": 0.01,
                }
            }
        ),
    )

    result = runner.generate("I went too school and their is a book.")

    assert result[0].text == "I went to school and there is a book."


def test_semantic_runner_warm_up_delegates_to_scorer() -> None:
    scorer = WarmableSemanticScorer({})
    runner = SemanticCorrectionRunner(model_name="fake-semantic", scorer=scorer)

    runner.warm_up()

    assert scorer.warm_up_count == 1

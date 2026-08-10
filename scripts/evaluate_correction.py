#!/usr/bin/env python3
"""Evaluate post-OCR correction quality and latency."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, List, Sequence, Tuple

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.correction.contextual import ContextualCorrector


@dataclass(frozen=True)
class Example:
    raw_text: str
    expected_text: str
    category: str


@dataclass(frozen=True)
class Decision:
    raw_text: str
    expected_text: str
    corrected_text: str
    status: str
    proposed_text: str
    category: str
    latency_ms: float


DEFAULT_EXAMPLES: Tuple[Example, ...] = (
    # Corrections preserve source capitalization, hence ``Blal`` becomes
    # ``Ball`` while the underlying lexical correction is ``ball``.
    Example("Blal", "Ball", "capitalization"),
    Example("quik", "quick", "missing_letters"),
    Example("recieve", "receive", "transpositions"),
    Example("taht", "that", "transpositions"),
    Example("bgi", "big", "transpositions"),
    Example("balll", "ball", "repeated_letters"),
    Example("bll", "ball", "missing_letters"),
    Example("balI", "ball", "ocr_substitutions"),
    Example("I hav a bal", "I have a ball", "sentence_context"),
    Example("Blal is bgi", "Ball is big", "sentence_context"),
    Example("teh cat is blak", "the cat is black", "spelling_errors"),
    Example("The quik brown fox", "The quick brown fox", "sentence_context"),
    Example("C hristmas", "Christmas", "token_splitting"),
    Example("C rhismtas", "Christmas", "token_merging"),
    Example("Blal.", "Ball.", "punctuation"),
    Example("123 456 789", "123 456 789", "numbers"),
    Example("the cat", "the cat", "ambiguous_words"),
    Example("school", "school", "unchanged_words"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate correction layer")
    parser.add_argument(
        "--input",
        type=Path,
        help="CSV file containing raw_text,expected_text rows",
    )
    return parser.parse_args()


def load_examples(path: Path | None) -> Sequence[Example]:
    if path is None:
        return DEFAULT_EXAMPLES
    rows: List[Example] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or len(row) < 2:
                continue
            if row[0].strip().lower() == "raw_text" and row[1].strip().lower() == "expected_text":
                continue
            rows.append(Example(row[0], row[1], "external"))
    return rows


def evaluate(examples: Iterable[Example]) -> List[Decision]:
    settings = RuntimeSettings.from_env()
    corrector = ContextualCorrector.from_settings(settings)
    decisions: List[Decision] = []
    for item in examples:
        start = time.perf_counter()
        result = corrector.correct(item.raw_text)
        latency_ms = (time.perf_counter() - start) * 1000.0
        decisions.append(
            Decision(
                raw_text=item.raw_text,
                expected_text=item.expected_text,
                corrected_text=result.corrected_text,
                status=result.status,
                proposed_text=(
                    result.corrections[0].corrected if result.corrections else result.corrected_text
                ),
                category=item.category,
                latency_ms=latency_ms,
            )
        )
    return decisions


def summarize(decisions: Sequence[Decision]) -> None:
    total = len(decisions)
    if total == 0:
        print("No examples to evaluate.")
        return

    exact = sum(1 for item in decisions if item.corrected_text == item.expected_text)
    expected_words = sum(len(item.expected_text.split()) for item in decisions)
    matching_words = sum(
        sum(left == right for left, right in zip(item.corrected_text.split(), item.expected_text.split()))
        for item in decisions
    )
    unchanged_expected = [item for item in decisions if item.raw_text == item.expected_text]
    unchanged_correct = sum(
        1 for item in unchanged_expected if item.corrected_text == item.expected_text
    )
    changed_expected = [item for item in decisions if item.raw_text != item.expected_text]
    false_corrections = sum(
        1
        for item in decisions
        if item.raw_text == item.expected_text and item.corrected_text != item.expected_text
    )
    automatic = [item for item in decisions if item.status == "automatic"]
    automatic_hits = sum(item.corrected_text == item.expected_text for item in automatic)
    suggestions = [item for item in decisions if item.status == "suggestion"]
    suggestion_hits = sum(
        1
        for item in suggestions
        if item.proposed_text == item.expected_text
    )
    transpositions = [item for item in decisions if item.category == "transpositions"]
    merges = [item for item in decisions if item.category in {"token_splitting", "token_merging"}]
    avg_latency = sum(item.latency_ms for item in decisions) / total

    print(f"Examples: {total}")
    print(f"Exact correction accuracy: {exact / total:.2%}")
    print(f"Word correction accuracy: {matching_words / expected_words:.2%}")
    if unchanged_expected:
        print(f"Unchanged-word accuracy: {unchanged_correct / len(unchanged_expected):.2%}")
    else:
        print("Unchanged-word accuracy: n/a")
    print(f"False correction rate: {false_corrections / total:.2%}")
    if automatic:
        print(f"Automatic correction precision: {automatic_hits / len(automatic):.2%}")
    else:
        print("Automatic correction precision: n/a")
    if suggestions:
        print(f"Suggestion precision: {suggestion_hits / len(suggestions):.2%}")
    else:
        print("Suggestion precision: n/a")
    print(
        "Transposition correction accuracy: "
        + (f"{sum(item.corrected_text == item.expected_text for item in transpositions) / len(transpositions):.2%}" if transpositions else "n/a")
    )
    print(
        "Token merge accuracy: "
        + (f"{sum(item.corrected_text == item.expected_text for item in merges) / len(merges):.2%}" if merges else "n/a")
    )
    print(f"Average correction latency (ms): {avg_latency:.2f}")

    wrong = [item for item in decisions if item.corrected_text != item.expected_text]
    if wrong:
        print("\nIncorrect decisions:")
        for item in wrong:
            print(
                f"- raw={item.raw_text!r} expected={item.expected_text!r} "
                f"got={item.corrected_text!r} status={item.status} latency_ms={item.latency_ms:.2f}"
            )


def main() -> None:
    args = parse_args()
    examples = load_examples(args.input)
    decisions = evaluate(examples)
    summarize(decisions)


if __name__ == "__main__":
    main()

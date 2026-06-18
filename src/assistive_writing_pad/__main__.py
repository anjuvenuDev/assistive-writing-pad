"""Command-line demo for the assistive writing pipeline."""

from assistive_writing_pad.correction.rule_based import RuleBasedCorrector
from assistive_writing_pad.capture.simulator import StrokeSimulator
from assistive_writing_pad.pipeline import WritingPipeline
from assistive_writing_pad.recognition.demo import DemoRecognizer


def main() -> None:
    simulated_strokes = StrokeSimulator().write_text("teh cat sat on a chaier")
    pipeline = WritingPipeline(
        recognizer=DemoRecognizer(text="teh cat sat on a chaier", confidence=0.92),
        corrector=RuleBasedCorrector(),
    )
    result = pipeline.process_strokes(simulated_strokes)

    print(f"stroke points: {len(simulated_strokes)}")
    print(f"recognized: {result.recognition.text} ({result.recognition.confidence:.2f})")
    print(f"corrected:  {result.correction.corrected_text}")
    if result.needs_review:
        print("status:     manual review required")


if __name__ == "__main__":
    main()

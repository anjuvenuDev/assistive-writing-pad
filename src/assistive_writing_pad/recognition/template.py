"""From-scratch template recognizer for early handwriting capture.

This is not a pretrained model. It is a small local learner that stores user
examples and recognizes later strokes by comparing normalized raster images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from assistive_writing_pad.contracts import CharacterConfidence, RecognitionResult, StrokePoint
from assistive_writing_pad.preprocessing.pipeline import StrokePreprocessor


DEFAULT_TEMPLATE_PATH = Path("data/user_templates.json")


@dataclass(frozen=True)
class TemplateSample:
    label: str
    vector: Tuple[float, ...]


class TemplateStore:
    """Persistent JSON store for locally taught handwriting templates."""

    def __init__(self, path: Path = DEFAULT_TEMPLATE_PATH) -> None:
        self.path = path
        self.samples: List[TemplateSample] = []

    @classmethod
    def load(cls, path: Path = DEFAULT_TEMPLATE_PATH) -> "TemplateStore":
        store = cls(path)
        if not path.exists():
            return store

        data = json.loads(path.read_text(encoding="utf-8"))
        samples = data.get("samples", [])
        if not isinstance(samples, list):
            raise ValueError("template store must contain a samples list")

        store.samples = [
            TemplateSample(
                label=str(item["label"]),
                vector=tuple(float(value) for value in item["vector"]),
            )
            for item in samples
        ]
        return store

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "samples": [
                {"label": sample.label, "vector": list(sample.vector)} for sample in self.samples
            ],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add(self, sample: TemplateSample) -> None:
        self.samples.append(sample)

    def labels(self) -> List[str]:
        return sorted({sample.label for sample in self.samples})


class TemplateGlyphRecognizer:
    """Recognize one captured stroke group using nearest taught template."""

    def __init__(
        self,
        store: Optional[TemplateStore] = None,
        preprocessor: Optional[StrokePreprocessor] = None,
    ) -> None:
        self.store = store or TemplateStore.load()
        self.preprocessor = preprocessor or StrokePreprocessor()

    def recognize(self, strokes: Sequence[StrokePoint]) -> RecognitionResult:
        if not strokes:
            return RecognitionResult(
                text="",
                confidence=0.0,
                metadata={"recognizer": "template", "reason": "empty_strokes"},
            )

        if not self.store.samples:
            return RecognitionResult(
                text="",
                confidence=0.0,
                metadata={"recognizer": "template", "reason": "no_templates"},
            )

        vector = self.vectorize(strokes)
        best_label = ""
        best_score = -1.0

        for sample in self.store.samples:
            score = _cosine_similarity(vector, np.asarray(sample.vector, dtype=np.float32))
            if score > best_score:
                best_label = sample.label
                best_score = score

        confidence = float(max(0.0, min(best_score, 1.0)))
        return RecognitionResult(
            text=best_label,
            confidence=confidence,
            character_confidences=(
                CharacterConfidence(character=best_label, confidence=confidence),
            )
            if best_label
            else (),
            metadata={"recognizer": "template", "templates": str(len(self.store.samples))},
        )

    def learn(self, label: str, strokes: Sequence[StrokePoint]) -> TemplateSample:
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("label is required")
        if not strokes:
            raise ValueError("cannot learn from empty strokes")

        sample = TemplateSample(
            label=clean_label,
            vector=tuple(float(value) for value in self.vectorize(strokes)),
        )
        self.store.add(sample)
        self.store.save()
        return sample

    def vectorize(self, strokes: Sequence[StrokePoint]) -> np.ndarray:
        image = self.preprocessor.preprocess(strokes).image
        vector = image.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            return vector
        return vector / norm


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("template vector shapes do not match")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def count_samples_by_label(samples: Iterable[TemplateSample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    return counts

"""Data models for post-OCR correction output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CorrectionAlternative:
    text: str
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class TokenCorrection:
    original: str
    corrected: str
    confidence: float
    reason: str
    alternatives: Tuple[CorrectionAlternative, ...] = ()
    edit_distance: int = 0
    automatic: bool = False
    status: str = "suggestion"

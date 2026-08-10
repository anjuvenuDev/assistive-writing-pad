"""Personalization hooks for correction feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Optional, Protocol


class CorrectionMemory(Protocol):
    def record_correction(self, original: str, corrected: str) -> None:
        """Persist a user-approved correction pair."""

    def get_preferred_correction(self, original: str) -> Optional[str]:
        """Return a preferred correction for a previously seen token."""


@dataclass
class JsonCorrectionMemory:
    path: Path
    _pairs: Dict[str, str] = field(default_factory=dict, init=False)
    _loaded: bool = field(default=False, init=False)

    def record_correction(self, original: str, corrected: str) -> None:
        self._load_if_needed()
        key = original.strip().lower()
        value = corrected.strip().lower()
        if not key or not value:
            return
        self._pairs[key] = value
        self._persist()

    def get_preferred_correction(self, original: str) -> Optional[str]:
        self._load_if_needed()
        key = original.strip().lower()
        if not key:
            return None
        return self._pairs.get(key)

    def _load_if_needed(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            self._pairs = {}
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._pairs = {
                    str(key).lower(): str(value).lower()
                    for key, value in payload.items()
                    if str(key).strip() and str(value).strip()
                }
            else:
                self._pairs = {}
        except Exception:
            self._pairs = {}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(dict(sorted(self._pairs.items())), indent=2, sort_keys=True),
            encoding="utf-8",
        )

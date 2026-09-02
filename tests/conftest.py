"""Shared pytest defaults for deterministic offline test runs."""

import os
from pathlib import Path


os.environ.setdefault("AWP_MODEL_CACHE", str(Path("models") / "cache"))
os.environ.setdefault("AWP_EMNIST_AUTO_DOWNLOAD", "0")

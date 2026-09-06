#!/usr/bin/env python
"""Download the Hugging Face TrOCR model into the project cache."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from assistive_writing_pad.recognition.trocr import (  # noqa: E402
    DEFAULT_TROCR_MODEL,
    default_huggingface_cache_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_TROCR_MODEL,
        help="Hugging Face TrOCR model ID.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory. Defaults to models/cache/huggingface.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or default_huggingface_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_trocr_model(args.model, cache_dir)
    print(f"Cached OCR model {args.model} in {cache_dir}")
    return 0


def cache_trocr_model(model_id: str, cache_dir: Path) -> None:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    print(f"Downloading processor: {model_id}")
    TrOCRProcessor.from_pretrained(model_id, use_fast=False, cache_dir=str(cache_dir))

    print(f"Downloading model: {model_id}")
    VisionEncoderDecoderModel.from_pretrained(
        model_id,
        low_cpu_mem_usage=False,
        cache_dir=str(cache_dir),
    )


if __name__ == "__main__":
    raise SystemExit(main())

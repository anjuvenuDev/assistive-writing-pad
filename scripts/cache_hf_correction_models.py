#!/usr/bin/env python
"""Download Hugging Face correction models into the project cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from assistive_writing_pad.config.settings import RuntimeSettings
from assistive_writing_pad.correction.huggingface import MODEL_TOKENIZERS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory. Defaults to models/cache/huggingface.",
    )
    parser.add_argument(
        "--spelling-model",
        default=None,
        help="Override spelling model ID.",
    )
    parser.add_argument(
        "--grammar-model",
        default=None,
        help="Override grammar/GEC model ID.",
    )
    parser.add_argument(
        "--semantic-model",
        default=None,
        help="Override semantic masked-LM model ID.",
    )
    args = parser.parse_args()

    settings = RuntimeSettings.from_env()
    cache_dir = args.cache_dir or settings.models_dir / "cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_ids = [
        args.spelling_model or settings.hf_spelling_model,
        args.grammar_model or settings.hf_grammar_model,
    ]
    for model_id in dict.fromkeys(model_ids):
        cache_seq2seq_model(model_id, cache_dir)

    semantic_model = args.semantic_model or settings.hf_semantic_model
    cache_masked_lm_model(semantic_model, cache_dir)

    print(f"Cached correction models in {cache_dir}")
    return 0


def cache_seq2seq_model(model_id: str, cache_dir: Path) -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer_id = MODEL_TOKENIZERS.get(model_id, model_id)
    print(f"Downloading tokenizer: {tokenizer_id}")
    AutoTokenizer.from_pretrained(tokenizer_id, cache_dir=str(cache_dir))

    print(f"Downloading model: {model_id}")
    AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=str(cache_dir))


def cache_masked_lm_model(model_id: str, cache_dir: Path) -> None:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"Downloading tokenizer: {model_id}")
    AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))

    print(f"Downloading masked LM model: {model_id}")
    AutoModelForMaskedLM.from_pretrained(model_id, cache_dir=str(cache_dir))


if __name__ == "__main__":
    raise SystemExit(main())

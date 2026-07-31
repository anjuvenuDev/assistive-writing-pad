# Models

Place downloaded or exported model artifacts here during development.

Large model files are intentionally ignored by Git. Record each model's source,
license, expected input shape, quantization mode, and Raspberry Pi performance
notes in `docs/SKILL_LOG.md` when it is added.

Current model-backed paths:

- Handwriting OCR: `microsoft/trocr-small-handwritten`
- Optional contextual correction reranker:
  `distilbert/distilbert-base-uncased`

Keep correction model loading disabled unless a local benchmark proves the model
meets the word and sentence latency budgets. This is mandatory for Raspberry Pi
runs and recommended for realtime laptop demos until the cache and dependency
stack are validated.

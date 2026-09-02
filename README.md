# Assistive Writing Pad

Real-time handwriting recognition and intelligent correction system for children with dysgraphia.

The project is designed for laptop-first development and Raspberry Pi 4 migration:

- Python 3.9+
- CPU-only inference
- Offline local handwriting recognition
- Hugging Face seq2seq spelling and grammatical-error correction
- Modular pipeline for capture, preprocessing, recognition, correction, display, and evaluation

## Current Status

The project has moved beyond the initial scaffold. So far, the repo includes a testable end-to-end pipeline with capture, preprocessing, recognition, and display components.

What has been done so far:

- Project scaffold, package metadata, and core data contracts for stroke, recognition, correction, and pipeline results
- Runtime settings with Raspberry Pi validation and a deterministic demo recognizer for early testing
- Huion HS64 input probing, stroke simulation, JSON save/load helpers, and a minimal event reader
- CPU-only stroke rasterization and preprocessing into 28x28 grayscale model inputs
- Legacy rule/contextual correction layers for deterministic debugging
- A Tkinter handwriting app and a browser-based web UI for writing, recognition, and basic text actions
- Pretrained handwritten OCR support through TrOCR, with lazy loading and local cache usage
- Real-time correction after recognition, using separate model-backed Hugging
  Face spelling, semantic real-word, and grammatical-error-correction stages
- Tests covering capture, preprocessing, template recognition, TrOCR rendering, web payload parsing, and pipeline behavior

The next major gaps are real handwriting accuracy benchmarking, broader sentence-level
grammar evaluation, word/line segmentation refinement, and Raspberry Pi performance validation.

## Repository Layout

```text
/src      Main application package
/models   Pretrained or exported model artifacts, not committed by default
/tests    Unit and validation tests
/docs     Project documentation and session memory
/data     Example inputs and evaluation datasets
```

## Local Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
python -m assistive_writing_pad
PYTHONPATH=src python -m assistive_writing_pad.display.handwriting_app
```

This workspace currently uses `.git-local` because the mounted `.git` directory is a read-only placeholder.
Use this command pattern for local Git operations in this environment:

```bash
git --git-dir=.git-local --work-tree=. status
```

## Handwriting Interface

Run the browser-based writing-pad interface:

```bash
PYTHONPATH=src python -m assistive_writing_pad.display.web_app
```

Then open `http://127.0.0.1:8000` in a browser.

The older Tkinter interface is still available, but the browser UI is the
preferred path because it works more reliably across laptop, tablet, and
Raspberry Pi setups.

The main screen is intentionally small: write on the canvas, use `Recognize`
for immediate OCR/correction, `Try Next` to cycle through OCR/model
alternatives when the first result is wrong, and `Clear Screen` to reset ink,
text, corrections, confidence, and alternatives together. Raw OCR and pointer
diagnostics are collapsed under technical details.

The main recognizer is the pretrained handwritten OCR model
`microsoft/trocr-base-handwritten`. Manual template learning is only fallback
support, not the expected user workflow.

Install model dependencies before using pretrained recognition. Use the setup
script so PyTorch is installed from the CPU-only wheel index:

```bash
scripts/setup_model_env.sh
```

After setup, run the app with:

```bash
.venv/bin/python -m assistive_writing_pad.display.web_app
```

The first run downloads the model from Hugging Face and can take time. After
that, the UI runs it from the local cache.

For best compatibility with PyTorch, use Python 3.9-3.11 for the model
environment.

Model and diagnostic artifacts default to project-local paths so laptop and
Raspberry Pi setup can be copied or backed up predictably:

- Generic model cache root: `models/cache`
- EMNIST character-model cache: `models/cache/emnist`
- Override cache root with `AWP_MODEL_CACHE`
- Override EMNIST cache with `AWP_EMNIST_CACHE_DIR`
- Disable EMNIST first-use download with `AWP_EMNIST_AUTO_DOWNLOAD=0`

Verbose image dumps are off by default because they slow down real-time use.
Enable them only while debugging:

```bash
AWP_DEBUG_OCR=1
AWP_DEBUG_EMNIST=1
AWP_DEBUG_PREPROCESSING=1
```

## Real-Time Correction

Recognition responses now continue through the correction pipeline before the
browser UI is updated. The API returns both `recognized_text` and
`corrected_text`; the textarea shows the corrected text, while raw OCR stays in
the debug panel.

The recognition path now uses stroke-geometry segmentation before model
inference:

- single-character inputs are post-processed with OCR cleanup, confusion
  candidates, and deterministic shape hints for common dotted/stem letters
- lines are clustered from stroke bounds instead of fixed canvas assumptions
- clear word gaps are recognized independently, then rejoined with spaces
- TrOCR beam outputs are preserved as ranked text alternatives for later
  "try next match" UI recovery

The default realtime path uses a Hugging Face model pipeline:

- spelling/typo correction with `oliverguhr/spelling-correction-english-base`
- semantic real-word correction with `distilbert/distilbert-base-uncased`
- grammatical-error correction with `gotutiyan/gec-bart-base`
- beam-search alternatives returned in correction metadata for review and
  future "try next match" UI flows
- hallucination guardrails that reject empty, unrelated, or oversized model
  generations rather than silently inventing corrections

Useful runtime flags:

```bash
AWP_DEVICE_PROFILE=laptop
AWP_CORRECTION_MODE=hf
AWP_HF_SPELLING_MODEL=oliverguhr/spelling-correction-english-base
AWP_HF_SEMANTIC_MODEL=distilbert/distilbert-base-uncased
AWP_HF_GRAMMAR_MODEL=gotutiyan/gec-bart-base
AWP_HF_CORRECTION_CANDIDATES=3
AWP_HF_CORRECTION_NUM_BEAMS=4
AWP_HF_CORRECTION_LOCAL_FILES_ONLY=0
AWP_CONTEXTUAL_MODEL_ENABLED=0
AWP_CONTEXTUAL_MODEL=distilbert/distilbert-base-uncased
AWP_PRELOAD_OCR_MODEL=1
AWP_WORD_SEGMENT=1
AWP_TROCR_NUM_BEAMS=3
AWP_TROCR_CANDIDATES=3
```

For Raspberry Pi latency experiments, `AWP_HF_GRAMMAR_MODEL` can be switched to
`Unbabel/gec-t5_small`. For higher quality laptop experiments, it can be
switched to `pszemraj/flan-t5-large-grammar-synthesis` or `grammarly/coedit-large`,
but those models are much larger and should be benchmarked for latency before
use on the writing pad.

Recognition should still work if correction model loading fails; the API falls
back to raw recognized text and returns model error metadata instead of applying
fake corrections.

For Raspberry Pi migration, keep the same backend but run from a prefilled local
model cache:

```bash
AWP_DEVICE_PROFILE=raspberry_pi
AWP_TROCR_MODEL=microsoft/trocr-small-handwritten
AWP_CORRECTION_MODE=hf
AWP_HF_SEMANTIC_MODEL=distilbert/distilbert-base-uncased
AWP_HF_GRAMMAR_MODEL=gotutiyan/gec-bart-base
AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1
AWP_PRELOAD_OCR_MODEL=0
```

The project tracks "nearly 100%" accuracy as an evaluation target. It should be
reported with measured correction accuracy, false-positive rate, and latency on
curated handwriting samples rather than treated as a guaranteed runtime claim.

The older `contextual` and `rules` correction modes remain available only for
debugging:

```bash
AWP_CORRECTION_MODE=contextual
AWP_CORRECTION_MODE=rules
```

Fallback template mode is still available for debugging:

1. Write one character on the pad.
2. Enter the correct label, for example `a`.
3. Click `Save Template`.
4. Write the character again to see recognized text update on the right.

The saved templates live in `data/user_templates.json`, which is ignored by Git.

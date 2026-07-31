# Assistive Writing Pad

Real-time handwriting recognition and intelligent correction system for children with dysgraphia.

The project is designed for laptop-first development and Raspberry Pi 4 migration:

- Python 3.9+
- CPU-only inference
- Offline local handwriting recognition
- Hybrid grammar/semantic correction with offline fallback
- Modular pipeline for capture, preprocessing, recognition, correction, display, and evaluation

## Current Status

The project has moved beyond the initial scaffold. So far, the repo includes a testable end-to-end pipeline with capture, preprocessing, recognition, and display components.

What has been done so far:

- Project scaffold, package metadata, and core data contracts for stroke, recognition, correction, and pipeline results
- Runtime settings with Raspberry Pi validation and a deterministic demo recognizer for early testing
- Huion HS64 input probing, stroke simulation, JSON save/load helpers, and a minimal event reader
- CPU-only stroke rasterization and preprocessing into 28x28 grayscale model inputs
- A rule-based correction layer for early dysgraphia-style error patterns
- A Tkinter handwriting app and a browser-based web UI for writing, recognition, and basic text actions
- Pretrained handwritten OCR support through TrOCR, with lazy loading and local cache usage
- Real-time correction after recognition, including spelling, dysgraphia-style error patterns,
  semantic context heuristics, and an optional pretrained contextual language model
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

## Real-Time Correction

Recognition responses now continue through the correction pipeline before the
browser UI is updated. The API returns both `recognized_text` and
`corrected_text`; the textarea shows the corrected text, while raw OCR stays in
the debug panel.

The default realtime path uses the contextual corrector with lightweight local
checks enabled and Transformer reranking disabled:

- deterministic spelling and dysgraphia-pattern checks
- fuzzy candidate generation for swaps, omissions, insertions, doubling, and
  visual confusions
- semantic context heuristics for common real-word mistakes
- optional pretrained masked-language reranking through
  `distilbert/distilbert-base-uncased`

Useful runtime flags:

```bash
AWP_DEVICE_PROFILE=laptop
AWP_CORRECTION_MODE=contextual
AWP_CONTEXTUAL_MODEL_ENABLED=0
AWP_CONTEXTUAL_MODEL=distilbert/distilbert-base-uncased
AWP_PRELOAD_OCR_MODEL=1
```

Enable `AWP_CONTEXTUAL_MODEL_ENABLED=1` only after the model is installed and
verified on the target laptop. Recognition should still work if correction
model loading fails; the API falls back to raw recognized text.

For Raspberry Pi migration, keep the same pipeline but disable the contextual
Transformer by default:

```bash
AWP_DEVICE_PROFILE=raspberry_pi
AWP_TROCR_MODEL=microsoft/trocr-small-handwritten
AWP_CONTEXTUAL_MODEL_ENABLED=0
AWP_PRELOAD_OCR_MODEL=0
```

The project tracks "nearly 100%" accuracy as an evaluation target. It should be
reported with measured correction accuracy, false-positive rate, and latency on
curated handwriting samples rather than treated as a guaranteed runtime claim.

Fallback template mode is still available for debugging:

1. Write one character on the pad.
2. Enter the correct label, for example `a`.
3. Click `Save Template`.
4. Write the character again to see recognized text update on the right.

The saved templates live in `data/user_templates.json`, which is ignored by Git.

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
- Tkinter and browser handwriting UIs for writing, recognition, correction, and alternatives
- Pretrained handwritten OCR support through TrOCR, with lazy loading and local cache usage
- Real-time correction after recognition, using separate model-backed Hugging
  Face spelling, semantic real-word, and grammatical-error-correction stages
- Tests covering capture, preprocessing, template recognition, TrOCR rendering, UI helpers, web payload parsing, and pipeline behavior

The next major gaps are real handwriting sample collection, broader sentence-level
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

The browser UI is the preferred path because it works more reliably across
laptop, tablet, and Raspberry Pi setups. The Tkinter UI is still available and
uses the same production OCR/correction pipeline with `Recognize`, `Try Next`,
and `Clear Screen` controls.

The main screen is intentionally small: write on the canvas, use `Recognize`
for immediate OCR/correction, `Try Next` to cycle through OCR/model
alternatives when the first result is wrong, and `Clear Screen` to reset ink,
text, corrections, confidence, and alternatives together. Raw OCR and pointer
diagnostics are collapsed under technical details.

The main recognizer is the pretrained handwritten OCR model
`microsoft/trocr-base-handwritten`. The template recognizer remains available
for isolated debugging tests, but it is not exposed in the production writing
interfaces.

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

To prefill the OCR cache explicitly:

```bash
.venv/bin/python scripts/cache_hf_ocr_model.py --model microsoft/trocr-base-handwritten
```

The correction cache can be prefilled separately:

```bash
.venv/bin/python scripts/cache_hf_correction_models.py
```

For best compatibility with PyTorch, use Python 3.9-3.11 for the model
environment.

Model and diagnostic artifacts default to project-local paths so laptop and
Raspberry Pi setup can be copied or backed up predictably:

- Generic model cache root: `models/cache`
- Hugging Face OCR/correction cache: `models/cache/huggingface`
- Override Hugging Face cache with `AWP_HF_CACHE_DIR`
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
AWP_HF_CACHE_DIR=models/cache/huggingface
AWP_HF_SPELLING_MODEL=oliverguhr/spelling-correction-english-base
AWP_HF_SEMANTIC_MODEL=distilbert/distilbert-base-uncased
AWP_HF_GRAMMAR_MODEL=gotutiyan/gec-bart-base
AWP_HF_CORRECTION_CANDIDATES=3
AWP_HF_CORRECTION_NUM_BEAMS=4
AWP_HF_CORRECTION_LOCAL_FILES_ONLY=0
AWP_TROCR_LOCAL_FILES_ONLY=0
AWP_CONTEXTUAL_MODEL_ENABLED=0
AWP_CONTEXTUAL_MODEL=distilbert/distilbert-base-uncased
AWP_PRELOAD_OCR_MODEL=1
AWP_PRELOAD_CORRECTION_MODELS=1
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
AWP_HF_CACHE_DIR=models/cache/huggingface
AWP_HF_SEMANTIC_MODEL=distilbert/distilbert-base-uncased
AWP_HF_GRAMMAR_MODEL=gotutiyan/gec-bart-base
AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1
AWP_TROCR_LOCAL_FILES_ONLY=1
AWP_PRELOAD_OCR_MODEL=0
AWP_PRELOAD_CORRECTION_MODELS=0
```

The project tracks "nearly 100%" accuracy as an evaluation target. It should be
reported with measured correction accuracy, false-positive rate, and latency on
curated handwriting samples rather than treated as a guaranteed runtime claim.

## Recognition Evaluation

Run the recognition gate against committed stroke fixtures:

```bash
AWP_TROCR_LOCAL_FILES_ONLY=1 .venv/bin/python scripts/evaluate_recognition.py --local-files-only --mode auto --output data/evaluation/recognition_report.json
```

Use `--min-exact-accuracy`, `--max-average-cer`, `--max-average-wer`,
`--max-low-confidence`, and `--max-p95-latency-ms` to turn the report into a
hardware-specific release gate. The committed fixture set is intentionally
small and should be expanded with real child handwriting captures before
claiming production recognition accuracy.

The current cached TrOCR recognition baseline on this machine is:

- 2/2 exact matches on the committed stroke fixture set
- average CER/WER 0.0% / 0.0%
- 0 low-confidence cases
- average latency 2039.6 ms, p95 latency 2241.6 ms after OCR warm-up
- OCR model warm-up time 11914.3 ms

This is only a smoke baseline for the recognition module. It verifies offline
cache loading, auto single-character routing, and the evaluation gate; it is not
large enough to claim handwriting recognition accuracy for arbitrary words or
sentences.

## End-to-End Evaluation

Run the complete stroke-to-correction gate against committed stroke fixtures:

```bash
AWP_TROCR_LOCAL_FILES_ONLY=1 AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 .venv/bin/python scripts/evaluate_end_to_end.py --local-files-only --mode auto --output data/evaluation/end_to_end_report.json
```

Use `--min-recognition-accuracy`, `--min-corrected-accuracy`,
`--max-average-recognition-cer`, `--max-average-corrected-cer`,
`--max-low-confidence`, `--max-needs-review`, and
`--max-p95-total-latency-ms` as release gates for the combined OCR plus
correction path.

The current cached end-to-end baseline on this machine is:

- 2/2 exact recognition matches and 2/2 exact corrected matches
- average recognition CER/WER 0.0% / 0.0%
- average corrected CER/WER 0.0% / 0.0%
- 0 low-confidence cases and 0 review cases at threshold 0.65
- average total latency 2016.3 ms, p95 total latency 2227.4 ms after warm-up
- combined OCR plus correction model warm-up time 13194.6 ms

This gate caught and fixed a real integration issue where the grammar model
rewrote isolated recognized letters such as `h` and `i` into punctuated sentence
fragments. The HF correction pipeline now preserves isolated character outputs
unchanged and records that correction was skipped for `isolated_character`.

## Correction Evaluation

Run the correction gate against the committed curated manifest:

```bash
AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 .venv/bin/python scripts/evaluate_correction.py --local-files-only --output data/evaluation/correction_report.json
```

The evaluator warms the correction models before timing by default. Use
`--no-warm-up` only when measuring cold-start behavior, and add
`--max-p95-latency-ms` when validating a specific hardware latency target.

The current cached HF correction baseline on this machine is:

- 12/12 exact matches on the curated correction set
- 0/2 false positives on clean text
- average latency 753.0 ms, p95 latency 899.0 ms after model warm-up
- correction model warm-up time 5950.7 ms

This is a small benchmark set, not a guarantee for all handwriting. The mixed
spelling-plus-grammar case is now within the laptop gate, but Raspberry Pi
latency still needs to be measured against the local model cache before calling
the hardware path production-ready.

The older `contextual` and `rules` correction modes remain available only for
debugging:

```bash
AWP_CORRECTION_MODE=contextual
AWP_CORRECTION_MODE=rules
```

The template recognizer remains unit-tested as an isolated debugging component.
It is not part of the child-facing writing-pad workflow.

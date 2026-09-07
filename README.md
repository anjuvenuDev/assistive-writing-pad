# Assistive Writing Pad

Real-time handwriting recognition and intelligent correction system for children with
dysgraphia.

This project is a software prototype for a Raspberry Pi-supported writing pad. A
child writes on a tablet/canvas, the system recognizes the handwriting, detects
spelling and language errors, corrects the text, and shows a clean child-friendly
output. The screen is treated as a development/demo substitute for a small LCD.

The current implementation is model-backed, offline-capable after model caching,
and organized into independently testable modules:

- stroke capture and replay
- handwriting image preprocessing
- handwritten OCR recognition
- spelling and OCR-fragment correction
- semantic real-word correction
- grammar correction
- web and Tk user interfaces
- evaluation, sample capture, and readiness reports

The project tracks near-perfect accuracy as a target, but production readiness is
reported only from measured benchmark evidence. The current production-readiness
gate intentionally remains blocked until enough real manual child handwriting
samples are collected and evaluated.

## Quick Demo

Use this flow for a laptop demo after dependencies and model cache are ready:

```bash
. .venv/bin/activate
AWP_TROCR_LOCAL_FILES_ONLY=1 AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 assistive-writing-web
```

Open:

```text
http://127.0.0.1:8000
```

Demo script:

1. Write a character, word, or sentence on the canvas.
2. Click `Recognize`.
3. Confirm that the main text area shows corrected text.
4. Open technical details only if you need raw OCR, confidence, model metadata, or alternatives.
5. Click `Try Next` when the first OCR/model alternative is wrong.
6. Click `Clear Screen` before the next sample.

## Problem Statement

Dysgraphia can affect handwriting quality, spelling, spacing, letter formation, and
written expression. A child may write:

- jumbled or badly spaced words
- incorrect spelling
- unclear handwriting
- real-word mistakes such as `too` vs `to` or `here` vs `hear`
- grammatically incomplete sentences

The project aims to assist the child by converting handwriting into corrected text
without forcing the child to manually type or select every correction.

## Goals

The target system should:

- recognize handwritten characters, words, and sentences
- detect spelling and OCR-fragment errors
- validate words using model-backed language evidence
- detect real-word semantic errors in context
- correct grammar while avoiding meaning-changing rewrites
- provide a simple UI for children and teachers
- support retry/retranslation when the top prediction is wrong
- support sample collection for measurable improvement
- run offline from a local model cache where possible
- be portable to Raspberry Pi with explicit latency measurements

## Current Status

Implemented:

- Python package and CLI entry points
- shared data contracts for stroke, recognition, correction, and pipeline results
- browser writing pad UI
- Tkinter writing pad UI
- hidden evaluator capture page for manual sample collection
- Huion HS64 probe and event reader utilities
- synthetic stroke simulator
- JSON stroke save/load helpers
- stroke rasterization and preprocessing
- TrOCR handwritten OCR through Hugging Face Transformers
- line segmentation and word segmentation from stroke geometry
- single-character routing and shape/confusion hints
- correction pipeline using Hugging Face spelling, semantic, and grammar stages
- lexical OCR-fragment repair using `wordfreq` plus RapidFuzz edit distance
- guardrails against unsafe model outputs
- retry/alternative support in the UI
- repeatable correction, recognition, end-to-end, coverage, and readiness evaluations
- curated synthetic Hugging Face Penpal stroke benchmark
- full automated test suite

Not finished:

- enough real child handwriting samples for production claims
- Raspberry Pi runtime and memory benchmarking on target hardware
- model quantization or distillation for low-latency Pi deployment
- broader sentence-level dysgraphia benchmark coverage
- teacher/parent usability trial evidence
- final hardware enclosure/LCD integration
- deployment service files for automatic boot on Raspberry Pi

## Repository Layout

```text
.
|-- README.md
|-- pyproject.toml
|-- data/
|   |-- examples/
|   `-- evaluation/
|-- docs/
|   |-- ROADMAP.md
|   `-- SKILL_LOG.md
|-- models/
|   `-- README.md
|-- scripts/
|   |-- append_end_to_end_case.py
|   |-- cache_hf_correction_models.py
|   |-- cache_hf_ocr_model.py
|   |-- check_evaluation_coverage.py
|   |-- collect_evaluation_evidence.py
|   |-- evaluate_correction.py
|   |-- evaluate_end_to_end.py
|   |-- evaluate_recognition.py
|   |-- import_penpal_samples.py
|   |-- run_evaluation_suite.py
|   `-- setup_model_env.sh
|-- src/assistive_writing_pad/
|   |-- capture/
|   |-- config/
|   |-- correction/
|   |-- display/
|   |-- eval/
|   |-- preprocessing/
|   |-- recognition/
|   |-- contracts.py
|   `-- pipeline.py
`-- tests/
```

Important directories:

- `src/assistive_writing_pad/capture`: hardware capture, simulator, and stroke IO
- `src/assistive_writing_pad/preprocessing`: raster and OCR image preprocessing
- `src/assistive_writing_pad/recognition`: TrOCR, EMNIST/template debug recognizers, cleanup, segmentation
- `src/assistive_writing_pad/correction`: Hugging Face correction, semantic scorer, contextual/rule debug modes
- `src/assistive_writing_pad/display`: browser and Tk writing interfaces
- `src/assistive_writing_pad/eval`: evaluation data loaders, metrics, coverage, readiness reports
- `data/evaluation`: committed evaluation manifests and JSON reports
- `models/cache`: ignored local model cache

## Architecture

The core runtime flow is:

```text
Pen/tablet/canvas stroke points
  -> stroke grouping
  -> preprocessing and OCR image rendering
  -> TrOCR handwriting recognition
  -> lexical OCR-fragment repair
  -> Hugging Face spelling correction
  -> Hugging Face semantic real-word correction
  -> Hugging Face grammar correction
  -> output guardrails and confidence checks
  -> child-facing corrected text + optional technical metadata
```

The shared contracts are defined in `src/assistive_writing_pad/contracts.py`:

- `StrokePoint`: x/y/timestamp/pressure sample
- `RecognitionResult`: recognized text, confidence, metadata
- `Correction`: one correction span with reason and confidence
- `CorrectionResult`: corrected text plus correction list
- `PipelineResult`: recognition, correction, review flag, review reason

The orchestrator is `WritingPipeline` in `src/assistive_writing_pad/pipeline.py`.
It runs correction after recognition and marks low-confidence recognition for
review instead of hiding uncertainty.

## Implemented Modules

### 1. Capture Module

Implemented files:

- `capture/huion_probe.py`
- `capture/huion_reader.py`
- `capture/simulator.py`
- `capture/stroke_io.py`

Responsibilities:

- read or simulate pen stroke points
- preserve x/y coordinates, pressure, and timing
- support replayable JSON stroke files
- keep hardware imports lazy so development works without the Huion tablet

Current hardware target:

- Huion HS64 on Linux/Raspberry Pi OS through `evdev`

### 2. Preprocessing Module

Implemented files:

- `preprocessing/rasterize.py`
- `preprocessing/pipeline.py`
- `preprocessing/image_ops.py`
- `preprocessing/ocr_image_ops.py`

Responsibilities:

- normalize stroke bounds
- crop empty whitespace
- preserve handwriting shape
- render clean black-on-white OCR images
- enhance OCR images with thresholding and morphology
- keep all image operations CPU-compatible

### 3. Recognition Module

Implemented files:

- `recognition/trocr.py`
- `recognition/confusion.py`
- `recognition/debug_saver.py`
- `recognition/emnist.py`
- `recognition/template.py`

Production recognizer:

- `microsoft/trocr-base-handwritten`

Raspberry Pi candidate recognizer:

- `microsoft/trocr-small-handwritten`

Recognition behavior:

- all production writing UI recognition goes through TrOCR
- character/word/ocr modes are accepted for compatibility
- single-character inputs use OCR cleanup and shape/confusion hints
- lines are clustered from stroke bounds
- clear word gaps are recognized independently and joined with spaces
- beam alternatives are kept for retry/retranslation
- debug images can be saved with `AWP_DEBUG_OCR=1`

### 4. Correction Module

Implemented files:

- `correction/huggingface.py`
- `correction/semantic.py`
- `correction/contextual.py`
- `correction/rule_based.py`
- `correction/factory.py`

Production correction mode:

```bash
AWP_CORRECTION_MODE=hf
```

Correction stages:

1. `wordfreq` + RapidFuzz lexical OCR-fragment repair
2. Hugging Face spelling correction
3. Hugging Face masked-language semantic real-word correction
4. Hugging Face grammar correction
5. output acceptance guardrails

Default models:

- spelling: `oliverguhr/spelling-correction-english-base`
- semantic scorer: `distilbert/distilbert-base-uncased`
- grammar: `gotutiyan/gec-bart-base`

Examples currently covered by tests/evaluation:

- `teh chlid writng` -> `The child is writing.`
- `a nalyze` -> `analyze`
- `centra I` -> `central`
- `i plea` -> `idea`
- `I went too school.` -> `I went to school.`
- `I can here the bell.` -> `I can hear the bell.`
- `There book is on table` -> `There is a book on the table.`

Guardrails:

- isolated characters are not rewritten into sentences
- single-word fragments skip grammar and semantic stages
- punctuation-only and case-only model rewrites are rejected when unsafe
- valid-word spelling rewrites require frequency and edit-distance evidence
- overlong, empty, unrelated, and low-confidence generations are rejected
- correction failure falls back to raw recognition instead of fake corrections

Debug-only modes:

```bash
AWP_CORRECTION_MODE=contextual
AWP_CORRECTION_MODE=rules
```

These remain for deterministic tests and debugging. They are not the primary
production correction path.

### 5. UI Module

Implemented files:

- `display/web_app.py`
- `display/handwriting_app.py`

Browser UI:

- primary interface for demos and Raspberry Pi LCD-style use
- canvas for writing
- corrected text output
- correction list
- recognition and correction confidence
- `Recognize`
- `Try Next`
- `Clear Screen`
- technical details collapsed away from the child-facing screen

Tk UI:

- desktop fallback interface
- same production recognizer/corrector stack
- useful for quick local manual checks

### 6. Evaluation Module

Implemented files:

- `eval/recognition_eval.py`
- `eval/correction_eval.py`
- `eval/end_to_end_eval.py`
- `eval/coverage.py`
- `eval/readiness.py`
- `eval/corpus.py`

Evaluation reports measure:

- exact match
- character error rate
- word error rate
- false positives
- missed corrections
- low confidence cases
- review cases
- average and p95 latency
- coverage by category and source

## Installation

Recommended Python:

- Python 3.9 to 3.11 for best PyTorch compatibility

Create environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Install model dependencies:

```bash
scripts/setup_model_env.sh
```

This installs CPU-compatible PyTorch and model dependencies for Hugging Face
Transformers.

Run tests:

```bash
.venv/bin/python -m pytest
```

Current full test status:

```text
224 passed
```

## Model Cache Setup

The first online model run can download large files from Hugging Face. For
repeatable demos and Raspberry Pi use, pre-cache the models.

Cache OCR model:

```bash
.venv/bin/python scripts/cache_hf_ocr_model.py --model microsoft/trocr-base-handwritten
```

Cache correction models:

```bash
.venv/bin/python scripts/cache_hf_correction_models.py
```

Default cache paths:

- generic model root: `models/cache`
- Hugging Face cache: `models/cache/huggingface`
- EMNIST cache: `models/cache/emnist`

Override cache location:

```bash
AWP_HF_CACHE_DIR=/path/to/huggingface/cache
AWP_MODEL_CACHE=/path/to/model/cache/root
```

Run fully offline from cache:

```bash
AWP_TROCR_LOCAL_FILES_ONLY=1 AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 assistive-writing-web
```

## Running The App

### Browser App

```bash
. .venv/bin/activate
assistive-writing-web
```

Default URL:

```text
http://127.0.0.1:8000
```

Use a different host/port:

```bash
assistive-writing-web --host 0.0.0.0 --port 8000
```

For Raspberry Pi access from another device on the same network:

```text
http://<raspberry-pi-ip>:8000
```

### Tk App

```bash
assistive-writing-ui
```

### Console Demo

```bash
python -m assistive_writing_pad
```

## Browser API

The browser server exposes three local endpoints.

### `POST /api/recognize`

Input:

```json
{
  "strokes": [
    [
      {"x": 12, "y": 30, "timestamp_ms": 0, "pressure": 1.0}
    ]
  ],
  "mode": "auto"
}
```

Output includes:

- `recognized_text`
- `corrected_text`
- `confidence`
- `correction_confidence`
- `corrections`
- `needs_review`
- `review_reason`
- `metadata`
- `top3`

### `POST /api/correct`

Input:

```json
{"text": "teh chlid writng"}
```

This runs text correction without handwriting recognition.

### `POST /api/evaluation/cases`

Enabled only when:

```bash
AWP_EVALUATION_CAPTURE_ENABLED=1
```

It appends manual stroke samples to the configured evaluation manifest.

## Runtime Configuration

Common laptop settings:

```bash
AWP_DEVICE_PROFILE=laptop
AWP_CORRECTION_MODE=hf
AWP_HF_CACHE_DIR=models/cache/huggingface
AWP_TROCR_MODEL=microsoft/trocr-base-handwritten
AWP_HF_SPELLING_MODEL=oliverguhr/spelling-correction-english-base
AWP_HF_SEMANTIC_MODEL=distilbert/distilbert-base-uncased
AWP_HF_GRAMMAR_MODEL=gotutiyan/gec-bart-base
AWP_TROCR_LOCAL_FILES_ONLY=0
AWP_HF_CORRECTION_LOCAL_FILES_ONLY=0
AWP_PRELOAD_OCR_MODEL=1
AWP_PRELOAD_CORRECTION_MODELS=1
AWP_WORD_SEGMENT=1
AWP_TROCR_NUM_BEAMS=3
AWP_TROCR_CANDIDATES=3
AWP_HF_CORRECTION_NUM_BEAMS=4
AWP_HF_CORRECTION_CANDIDATES=3
```

Raspberry Pi experiment settings:

```bash
AWP_DEVICE_PROFILE=raspberry_pi
AWP_TROCR_MODEL=microsoft/trocr-small-handwritten
AWP_CORRECTION_MODE=hf
AWP_HF_CACHE_DIR=models/cache/huggingface
AWP_TROCR_LOCAL_FILES_ONLY=1
AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1
AWP_PRELOAD_OCR_MODEL=0
AWP_PRELOAD_CORRECTION_MODELS=0
AWP_WORD_SEGMENT=1
AWP_TROCR_NUM_BEAMS=1
AWP_TROCR_CANDIDATES=1
AWP_HF_CORRECTION_NUM_BEAMS=2
AWP_HF_CORRECTION_CANDIDATES=2
```

Debug flags:

```bash
AWP_DEBUG_OCR=1
AWP_DEBUG_EMNIST=1
AWP_DEBUG_PREPROCESSING=1
```

Evaluation flags:

```bash
AWP_EVALUATION_CAPTURE_ENABLED=1
AWP_EVALUATION_MANIFEST=data/evaluation/end_to_end_cases.jsonl
```

## Sample Capture Workflow

Manual samples are required for production evidence. Synthetic or smoke samples
are useful for development, but they are not enough to claim real-world accuracy.

### Capture Samples In The Browser

Start evaluator capture:

```bash
AWP_EVALUATION_CAPTURE_ENABLED=1 assistive-writing-web
```

Open:

```text
http://127.0.0.1:8000/capture
```

For each sample:

1. Choose a stable ID.
2. Choose a category: `single_character`, `word`, or `sentence`.
3. Write naturally on the pad.
4. Enter the expected corrected text.
5. Optionally enter expected raw recognized text if it should differ from the corrected text.
6. Add notes such as child age group, handwriting condition, or error type.
7. Save the sample.
8. Clear the canvas and repeat.

The server forces `source` to `manual`. The browser cannot spoof manual coverage by
sending its own source value.

Default manual manifest:

```text
data/evaluation/end_to_end_cases.jsonl
```

### Capture Samples From The Main Pad Console

Run the web app, write on the main pad, then open the browser console:

```js
JSON.stringify(window.assistiveWritingPad.exportStrokePayload())
```

Save the JSON to a temporary file and append it:

```bash
.venv/bin/python scripts/append_end_to_end_case.py \
  --payload /tmp/payload.json \
  --id sentence_001 \
  --category sentence \
  --expected "The child is writing."
```

If needed:

```bash
.venv/bin/python scripts/append_end_to_end_case.py \
  --payload /tmp/payload.json \
  --id sentence_002 \
  --category sentence \
  --expected "The child is writing." \
  --expected-recognized "The child is writng" \
  --notes "manual dysgraphia spelling sample"
```

### Required Manual Coverage For Production Readiness

The production coverage gate currently requires:

- 20 manual `single_character` cases
- 50 manual `word` cases
- 50 manual `sentence` cases

Recommended sample categories:

- clean handwriting
- poor handwriting but correct spelling
- correct handwriting with spelling mistakes
- bad handwriting plus spelling mistakes
- jumbled spacing
- omitted letters
- swapped letters
- doubled letters
- phonetic spelling
- real-word semantic errors
- short grammar errors
- full sentence errors

## Evaluation Commands

### Recognition Evaluation

```bash
AWP_TROCR_LOCAL_FILES_ONLY=1 .venv/bin/python scripts/evaluate_recognition.py \
  --local-files-only \
  --mode auto \
  --output data/evaluation/recognition_report.json
```

Current smoke baseline:

- 2/2 exact matches
- average CER/WER: 0.0% / 0.0%
- 0 low-confidence cases
- average latency: 2039.6 ms
- p95 latency: 2241.6 ms
- OCR warm-up: 11914.3 ms

### Correction Evaluation

```bash
AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 .venv/bin/python scripts/evaluate_correction.py \
  --local-files-only \
  --output data/evaluation/correction_report.json
```

Current correction baseline:

- 12/12 exact matches
- 0/2 false positives on clean text
- average latency: 810.8 ms
- p95 latency: 1015.9 ms
- correction warm-up: 4425.2 ms

### End-To-End Evaluation

```bash
AWP_TROCR_LOCAL_FILES_ONLY=1 AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 \
  .venv/bin/python scripts/evaluate_end_to_end.py \
  --local-files-only \
  --mode auto \
  --output data/evaluation/end_to_end_report.json
```

Current smoke baseline:

- 2/2 exact recognition matches
- 2/2 exact corrected matches
- recognition CER/WER: 0.0% / 0.0%
- corrected CER/WER: 0.0% / 0.0%
- average total latency: 2038.8 ms
- p95 total latency: 2251.5 ms
- combined warm-up: 13095.2 ms

### Synthetic Hugging Face Penpal Benchmark

The repo includes a curated synthetic handwriting-stroke fixture imported from:

```text
https://huggingface.co/datasets/breitburg/penpal
```

Penpal is a synthetic handwriting dataset with generated pen strokes grouped by
word. It is useful for repeatable development, but it is not a substitute for
manual child handwriting samples.

Refresh the committed synthetic sample:

```bash
.venv/bin/python scripts/import_penpal_samples.py \
  --replace \
  --offset 0 \
  --length 4 \
  --max-sentence-cases 2 \
  --max-word-cases 6
```

Run the benchmark:

```bash
AWP_TROCR_LOCAL_FILES_ONLY=1 AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 \
  .venv/bin/python scripts/evaluate_end_to_end.py \
  --local-files-only \
  --mode auto \
  --manifest data/evaluation/penpal_end_to_end_cases.jsonl \
  --output data/evaluation/penpal_end_to_end_report.json \
  --min-recognition-accuracy 0.0 \
  --min-corrected-accuracy 0.0 \
  --max-p95-total-latency-ms 15000
```

Current Penpal synthetic result:

- 8 synthetic stroke cases
- 2 sentence cases
- 6 word cases
- raw TrOCR recognition exact: 3/8 (37.5%)
- corrected output exact: 8/8 (100.0%)
- recognition CER/WER: 17.9% / 91.7%
- corrected CER/WER: 0.0% / 0.0%
- average total latency: 4248.3 ms
- p95 total latency: 10300.7 ms

### Coverage Gate

Smoke coverage:

```bash
.venv/bin/python scripts/check_evaluation_coverage.py \
  --profile smoke \
  --output data/evaluation/coverage_report.json
```

Production coverage:

```bash
.venv/bin/python scripts/check_evaluation_coverage.py --profile production
```

Current state:

- smoke coverage passes
- production coverage fails because manual cases are missing

### Readiness Report

```bash
AWP_TROCR_LOCAL_FILES_ONLY=1 AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 \
  .venv/bin/python scripts/collect_evaluation_evidence.py \
  --output data/evaluation/readiness_report.json
```

The readiness report intentionally separates:

- smoke readiness
- production readiness
- correction metrics
- end-to-end metrics
- coverage metrics
- model configuration
- blocking findings

### Full Smoke Suite

```bash
AWP_TROCR_LOCAL_FILES_ONLY=1 AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 \
  .venv/bin/python scripts/run_evaluation_suite.py \
  --profile smoke \
  --output data/evaluation/evaluation_suite_report.json
```

Current smoke suite:

- correction gate: pass
- end-to-end gate: pass
- smoke coverage gate: pass
- readiness collection: pass

Use production suite only after enough manual samples exist:

```bash
AWP_TROCR_LOCAL_FILES_ONLY=1 AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 \
  .venv/bin/python scripts/run_evaluation_suite.py --profile production
```

## Demo Checklist

Before demo:

1. Activate `.venv`.
2. Confirm model cache exists under `models/cache/huggingface`.
3. Run `pytest` if code changed.
4. Run the smoke suite if benchmark evidence is needed.
5. Start `assistive-writing-web`.
6. Open `http://127.0.0.1:8000`.
7. Test `Recognize`, `Try Next`, and `Clear Screen`.

Good demo inputs:

- single characters: `h`, `i`
- spelling: `teh`
- OCR fragment style: `a nalyze`
- sentence: `teh chlid writng`
- semantic: `I can here the bell.`
- grammar: `There book is on table`

Demo talking points:

- The child only writes; they do not need to type.
- Recognition and correction are separate modules.
- Raw OCR is preserved in technical details.
- The main output shows corrected text.
- The app can retry alternatives instead of forcing the first guess.
- Manual samples become repeatable evaluation fixtures.
- Production claims are gated by measured evidence, not assumptions.

## Report/PPT Content

### Abstract

Assistive Writing Pad is a real-time handwriting recognition and correction
prototype for dysgraphia support. It captures pen strokes, recognizes handwritten
text using a pretrained TrOCR model, applies model-backed lexical, spelling,
semantic, and grammar correction, and displays corrected text in a simple web UI.
The system includes evaluation tools for recognition accuracy, correction
accuracy, error rates, latency, and production-readiness coverage.

### System Modules

- input capture: Huion/tablet/canvas stroke points
- preprocessing: crop, normalize, render, enhance
- recognition: Hugging Face TrOCR handwritten OCR
- error detection: confidence, spelling likelihood, semantic scoring, grammar model output
- correction: lexical repair, spelling model, masked language model, GEC model
- UI: browser writing pad and evaluator capture page
- evaluation: JSONL manifests, metrics, coverage gates, readiness reports

### Technology Stack

- Python 3.9+
- NumPy
- OpenCV
- Pillow
- PyTorch
- Hugging Face Transformers
- TrOCR
- RapidFuzz
- wordfreq
- jiwer
- pytest
- ruff
- Tkinter
- built-in Python HTTP server for the browser UI

### Main Models

- OCR: `microsoft/trocr-base-handwritten`
- OCR low-memory candidate: `microsoft/trocr-small-handwritten`
- spelling: `oliverguhr/spelling-correction-english-base`
- semantic: `distilbert/distilbert-base-uncased`
- grammar: `gotutiyan/gec-bart-base`

### Current Evidence

- automated tests: 224 passing
- correction smoke benchmark: 12/12 exact
- clean-text false positives: 0/2
- end-to-end smoke benchmark: 2/2 recognition and correction exact
- synthetic Penpal corrected benchmark: 8/8 exact
- production readiness: blocked by missing manual sample coverage

### Key Risks

- Dysgraphic handwriting differs from standard handwriting datasets.
- Raw TrOCR recognition is not yet strong enough on all synthetic sentence samples.
- Transformer inference may be slow on Raspberry Pi without smaller models or quantization.
- Semantic/grammar correction can change intended meaning unless guarded.
- Real child handwriting sample collection is mandatory before accuracy claims.

## Raspberry Pi Notes

Target hardware:

- Raspberry Pi 4 or newer
- 4 GB RAM minimum recommended
- Linux/Raspberry Pi OS
- Huion HS64 or compatible pen tablet
- LCD or browser-accessible screen

Recommended approach:

1. Build and cache models on laptop.
2. Copy repo and `models/cache/huggingface` to Raspberry Pi.
3. Use `microsoft/trocr-small-handwritten` first.
4. Reduce beams/candidates.
5. Disable preload if startup memory is too high.
6. Run smoke tests.
7. Run latency benchmarks.
8. Only then decide whether larger models are usable.

Pi command template:

```bash
AWP_DEVICE_PROFILE=raspberry_pi \
AWP_TROCR_MODEL=microsoft/trocr-small-handwritten \
AWP_TROCR_LOCAL_FILES_ONLY=1 \
AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 \
AWP_PRELOAD_OCR_MODEL=0 \
AWP_PRELOAD_CORRECTION_MODELS=0 \
AWP_TROCR_NUM_BEAMS=1 \
AWP_TROCR_CANDIDATES=1 \
AWP_HF_CORRECTION_NUM_BEAMS=2 \
AWP_HF_CORRECTION_CANDIDATES=2 \
assistive-writing-web --host 0.0.0.0 --port 8000
```

## Troubleshooting

Model cannot load:

```bash
scripts/setup_model_env.sh
.venv/bin/python scripts/cache_hf_ocr_model.py --model microsoft/trocr-base-handwritten
.venv/bin/python scripts/cache_hf_correction_models.py
```

Offline mode fails:

- confirm `models/cache/huggingface` exists
- confirm the exact model names were cached
- temporarily set `AWP_TROCR_LOCAL_FILES_ONLY=0` and `AWP_HF_CORRECTION_LOCAL_FILES_ONLY=0` on a networked laptop

OCR is inaccurate:

- enable `AWP_DEBUG_OCR=1`
- inspect generated raw/cropped/processed images under debug output
- capture the failing sample into `data/evaluation/end_to_end_cases.jsonl`
- run `scripts/evaluate_end_to_end.py`
- check whether the issue is recognition, correction, or both

Correction changes meaning:

- add the text to `data/evaluation/correction_cases.jsonl`
- run `scripts/evaluate_correction.py`
- inspect `correction_metadata` stages
- tighten acceptance thresholds or model stage policy

Browser cannot access Raspberry Pi:

- start with `--host 0.0.0.0`
- confirm both devices are on the same network
- check firewall rules
- open `http://<raspberry-pi-ip>:8000`

Git status in this workspace:

This environment may use `.git-local` because `.git` can be mounted read-only.
Use:

```bash
git --git-dir=.git-local --work-tree=. status
```

## Development Workflow

For code changes:

```bash
.venv/bin/ruff check src tests scripts pyproject.toml
.venv/bin/python -m pytest
```

For model/evaluation changes:

```bash
AWP_TROCR_LOCAL_FILES_ONLY=1 AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 \
  .venv/bin/python scripts/run_evaluation_suite.py --profile smoke
```

For release-readiness evidence:

```bash
.venv/bin/python scripts/check_evaluation_coverage.py --profile production
AWP_TROCR_LOCAL_FILES_ONLY=1 AWP_HF_CORRECTION_LOCAL_FILES_ONLY=1 \
  .venv/bin/python scripts/run_evaluation_suite.py --profile production
```

## Data And Privacy

Manual handwriting samples can be sensitive. Treat collected samples as project
data, not public demo content, unless consent and anonymization are handled.

Recommended practice:

- avoid child names in sample IDs
- use anonymous IDs such as `word_001`
- record age band or condition only in notes if needed
- do not commit private real child handwriting unless approved
- keep larger private datasets outside Git or in an approved private store

## Final Production Checklist

Before claiming production readiness:

- collect required manual sample coverage
- pass production coverage gate
- pass production correction benchmark
- pass production recognition benchmark
- pass production end-to-end benchmark
- benchmark Raspberry Pi latency and memory
- verify offline boot from local model cache
- validate Huion input on Raspberry Pi OS
- test UI on the actual LCD resolution
- document model licenses and dataset consent
- prepare teacher/parent demo instructions
- define failure handling for low-confidence output

## Known Limitations

- The current strongest evidence is still smoke plus synthetic benchmark evidence.
- Manual child handwriting coverage is missing.
- Sentence OCR can be slow because word segmentation may run TrOCR once per word.
- Raw recognition exact accuracy on the current synthetic Penpal set is 37.5%.
- Correction can recover the current synthetic failures, but that does not prove arbitrary handwriting accuracy.
- Raspberry Pi latency is not yet measured on target hardware.

## Current Commit/Evidence Summary

Recent completed module work includes:

- all live handwriting recognition routed through TrOCR
- UI simplified to core child-facing controls
- retry/retranslation support through alternatives
- Hugging Face spelling, semantic, and grammar correction
- model warm-up for realtime use
- correction evaluation gate
- recognition evaluation gate
- end-to-end evaluation gate
- manual capture workflow
- production coverage gate
- readiness evidence report
- Hugging Face Penpal synthetic sample benchmark
- lexical OCR-fragment repair stage

The next engineering step is manual sample collection followed by production-profile
evaluation. The next hardware step is Raspberry Pi installation and latency testing.

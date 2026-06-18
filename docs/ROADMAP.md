# Project Roadmap

## Phase 0 - Planning And Repository Memory

- Timeline: completed on 2026-06-18
- Complexity: low
- Dependencies: none
- Deliverables:
  - Project plan
  - Initial `docs/SKILL_LOG.md`
  - Risk and resource summary
- Success criteria:
  - Project context from review PDFs captured
  - Implementation gated until approval

## Phase 1 - Repository Scaffold And Core Contracts

- Timeline: 1-2 days
- Complexity: low
- Dependencies: Phase 0 approval
- Deliverables:
  - Python package scaffold under `src/assistive_writing_pad`
  - Shared data contracts for strokes, recognition, correction, and pipeline results
  - Runtime settings with Raspberry Pi profile validation
  - Deterministic demo recognizer and rule-based corrector
  - Unit tests for confidence gating and correction flow
- Success criteria:
  - `python -m pytest` passes
  - `PYTHONPATH=src python -m assistive_writing_pad` runs a demo pipeline
  - Low-confidence recognition is flagged for review before correction

## Phase 2 - Tablet And Simulator Input Capture

- Timeline: completed on 2026-06-18
- Complexity: medium
- Dependencies: Phase 1
- Deliverables:
  - Huion HS64 event probe for Linux: completed
  - Stroke capture abstraction using x, y, pressure, and timing: completed
  - Synthetic simulator for development without hardware: completed
  - Raw stroke JSON recording and playback: completed
- Success criteria:
  - Capture produces stable stroke records: met by simulator tests
  - Replay path feeds the same pipeline contracts as live hardware: met by JSON round-trip tests
  - Hardware absence does not block development: met by lazy `evdev` imports

## Phase 3 - Stroke And Image Preprocessing

- Timeline: completed on 2026-06-18
- Complexity: medium
- Dependencies: Phase 2
- Deliverables:
  - Stroke smoothing and normalization: initial scaling/raster normalization complete
  - Bounding box crop and padding: completed
  - Grayscale rasterization for model input: completed
  - Word and line segmentation heuristics: deferred to recognition/display phases
- Success criteria:
  - Preprocessing handles variable size and pressure: met by tests
  - Generated images match expected model input shape: met with 28x28 output
  - Unit tests cover empty strokes and oversized writing: met

## Phase 4 - Offline Handwriting Recognition Baseline

- Timeline: 1-2 weeks
- Complexity: high
- Dependencies: Phase 3
- Deliverables:
  - Local recognizer adapter
  - Pretrained or existing EMNIST/IAM/CVL-compatible model loading
  - Character or word confidence scores
  - Confidence threshold behavior at 0.85
- Success criteria:
  - Offline recognition works without network
  - Recognition output includes text and confidence metadata
  - Baseline WER and character accuracy are measured

## Phase 5 - Spell Correction And Dysgraphia Error Patterns

- Timeline: 4-7 days
- Complexity: medium
- Dependencies: Phase 4
- Deliverables:
  - Dictionary and fuzzy matching correction
  - Dysgraphia-specific confusion sets
  - Reversal, swap, omission, phonetic, and doubling error tests
- Success criteria:
  - Common examples such as `teh -> the`, `fone -> phone`, and `writting -> writing` pass
  - Corrections include confidence and reason codes
  - False-positive correction rate is measured on clean text

## Phase 6 - Grammar And Semantic Correction

- Timeline: 1-2 weeks
- Complexity: high
- Dependencies: Phase 5
- Deliverables:
  - Offline grammar fallback
  - Optional API-backed correction adapter
  - Contextual word-choice checks
  - Capitalization and punctuation refinement
- Success criteria:
  - Subject-verb, tense, article, and contextual examples pass
  - Offline mode remains usable on laptop and Raspberry Pi
  - API mode can be disabled without changing the UI or pipeline

## Phase 7 - Real-Time Display

- Timeline: 1 week
- Complexity: medium
- Dependencies: Phases 2-6
- Deliverables:
  - Side-by-side raw and corrected text display
  - Highlighted corrections
  - Confidence badges
  - Manual review indicators
- Success criteria:
  - Corrections update after word completion
  - UI remains responsive during recognition/correction
  - Display is readable for children and teachers

## Phase 8 - Evaluation Harness

- Timeline: 1 week
- Complexity: medium
- Dependencies: Phases 4-7
- Deliverables:
  - WER measurement
  - Grammar and semantic accuracy evaluation
  - Latency and memory profiling
  - Dataset manifest format
- Success criteria:
  - Accuracy reports can be reproduced from command line
  - Metrics are split by recognition, spelling, grammar, semantic correction, and end-to-end output
  - 98% target is tracked against defined benchmark sets

## Phase 9 - Raspberry Pi Optimization

- Timeline: 1-2 weeks
- Complexity: high
- Dependencies: Phase 8
- Deliverables:
  - Quantized model artifacts where practical
  - Raspberry Pi install notes
  - CPU and memory profile
  - Startup script
- Success criteria:
  - Runs on Raspberry Pi 4 with 4 GB RAM
  - Sentence correction target remains below 2 seconds
  - Word-level feedback path targets below 500 ms when recognition confidence is high

## Phase 10 - Hardware And User Testing Loop

- Timeline: ongoing
- Complexity: high
- Dependencies: Phase 9
- Deliverables:
  - Hardware trial checklist
  - Teacher/parent feedback form
  - Error review dataset
  - Iterative model and UX improvements
- Success criteria:
  - System reduces scribe dependency in structured writing tasks
  - Usability feedback is positive enough for continued trials
  - Accuracy and latency are tracked over real samples

## Resource Requirements

- Core Python: `numpy`, `opencv-python-headless`, `Pillow`, `pydantic`, `PyYAML`, `rapidfuzz`, `jiwer`
- Development: `pytest`, `ruff`
- Recognition/model experiments: `onnxruntime`, `torch`, `torchvision`, `transformers`
- Hardware input: `evdev` on Linux
- Datasets: EMNIST ByClass/Letters, IAM Handwriting Database, CVL where licensing allows
- Hardware: Huion HS64 tablet, laptop, Raspberry Pi 4 with 4 GB RAM, display, keyboard/mouse for setup
- Optional services: grammar/semantic correction API behind a configurable adapter

## Primary Risks

- Dysgraphic handwriting differs from standard handwriting datasets.
- Character-only recognition will not reach sentence-level accuracy alone.
- Semantic correction may overcorrect intended meaning.
- Raspberry Pi transformer inference may be too slow without quantization or API fallback.
- Huion driver behavior must be validated on both laptop Linux and Raspberry Pi OS.

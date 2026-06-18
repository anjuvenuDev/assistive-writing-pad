# Assistive Writing Pad Project Skill Log

Last updated: 2026-06-18

## Project Goal

Build a real-time handwriting recognition and intelligent correction system for children with dysgraphia. The system should capture Huion HS64 tablet input, recognize handwriting locally, correct spelling/grammar/semantic issues, and display original plus corrected text with confidence feedback. Development starts on laptop and must stay portable to Raspberry Pi 4, Python 3.9+, CPU-only inference.

## Source Context

- `H18_WritingPadAndWritingPenSpecificLearningDisability.pdf`: project objective is an affordable writing pad/pen for children with dysgraphia, focused on reducing scribe dependency, guided prompts, real-time feedback, pressure/orientation sensing, speech-to-text support, and performance analytics.
- `IFP Initial Review - Anjana and Nikhil.pdf`: initial proposal emphasizes dysgraphia-specific design, affordability, real-time feedback, guided writing, and user testing.
- `IFP Second Review - Anjana and Nikhil.pdf`: current baseline is approximately 80% software pipeline implementation with tablet-style handwriting capture, preprocessing, CNN recognition on EMNIST ByClass, and modular input/preprocessing/recognition/post-processing.

## Current Phase Status

- Phase: 2 - Tablet and simulator input capture
- Status: completed
- Completed this session:
  - Reviewed repository state and confirmed the GitHub remote has no refs yet.
  - Created project roadmap in `docs/ROADMAP.md`.
  - Created Python package scaffold under `src/assistive_writing_pad`.
  - Added shared contracts for stroke points, recognition results, correction results, and pipeline results.
  - Added runtime settings with Raspberry Pi profile validation.
  - Added a deterministic demo recognizer.
  - Added a small rule-based corrector for early demo/error-pattern tests.
  - Added tests for high-confidence correction flow and low-confidence manual review gating.
  - Added project metadata in `pyproject.toml`, `README.md`, `.gitignore`, and `models/README.md`.
  - Added `StrokeSimulator` for hardware-free capture development.
  - Added stroke JSON save/load helpers for recording and replay.
  - Added Huion/Linux input device probe with lazy `evdev` import.
  - Added minimal Huion event reader that maps ABS_X, ABS_Y, and ABS_PRESSURE into `StrokePoint`.
  - Added capture tests for simulator output, stroke replay, schema validation, and Huion device detection.
- Not yet implemented:
  - Stroke/image preprocessing
  - Real handwriting recognition model loading
  - Full spelling/grammar/semantic correction models
  - Real-time UI
  - Evaluation datasets and benchmark runner
  - Hardware-validated Huion coordinate scaling and pressure normalization

## Technical Decisions

- Architecture will be modular: capture -> preprocessing -> recognition -> correction -> display -> metrics.
- Recognition should run offline locally.
- Grammar and semantic correction may use a hybrid mode: offline fallback first, optional API mode later.
- Raspberry Pi constraints are mandatory from the start: CPU-only, quantized/lightweight models, low memory, minimal background services.
- The 98% corrected output target will be evaluated on controlled benchmark sets and project-specific handwriting samples; it should not be assumed for unconstrained child handwriting until measured.
- Confidence gating is enforced at the pipeline layer so low-confidence recognition is flagged before correction.
- In this sandbox, `.git` is a read-only empty placeholder. Local commits use `.git-local` with `git --git-dir=.git-local --work-tree=.`.
- Hardware dependencies are imported lazily so simulator, tests, and non-Linux development do not require `evdev`.
- Stroke recording uses versioned JSON so captured samples can become repeatable evaluation fixtures.

## Known Limitations And Risks

- Dysgraphic handwriting is more variable than standard IAM/EMNIST samples.
- EMNIST character recognition alone is insufficient for sentence-level handwriting recognition.
- Huion HS64 Linux event data may vary by driver stack and OS configuration.
- Real-time feedback below 500 ms depends on word-boundary detection, batching strategy, and model size.
- Semantic correction can introduce harmful overcorrections if context is too short.
- Raspberry Pi 4 may not support transformer inference fast enough without quantization or API fallback.

## Raspberry Pi Optimization Notes

- Prefer ONNX Runtime, TFLite, or OpenVINO-compatible exported models where practical.
- Prefer INT8 quantization for correction and recognition models.
- Keep correction pipeline staged so cheap spell/context rules run before heavier models.
- Cache model objects and dictionaries at startup.
- Avoid GPU-only dependencies.
- Maintain a laptop/Pi configuration switch through environment variables or config files.

## Commands To Test Components

Current commands:

```bash
python -m pytest
PYTHONPATH=src python -m assistive_writing_pad
PYTHONPATH=src python -m assistive_writing_pad.capture.huion_probe
git --git-dir=.git-local --work-tree=. status
```

Expected future commands:

```bash
python -m src.app
python -m src.capture.huion_probe
python -m src.eval.run_wer
```

## Accuracy And Evaluation Log

Phase 1:

- Unit tests: 4 passed.
- Recognition accuracy: not measured; demo recognizer is deterministic and not a model.
- Error patterns currently represented in the demo corrector:
  - Letter swap: `teh -> the`
  - Phonetic or insertion-style error: `chaier -> chair`
  - Phonetic examples: `fone -> phone`, `sed -> said`, `rite -> right`
  - Omission: `becaus -> because`
  - Letter order: `recieve -> receive`
  - Doubling errors: `writting -> writing`, `occured -> occurred`

Phase 2:

- Unit tests: 8 passed.
- Capture accuracy: not applicable yet; hardware capture has not been validated on a connected Huion HS64.
- Simulator behavior:
  - Deterministic stroke points with x/y/timestamp/pressure.
  - Monotonic timestamps.
  - JSON round-trip verified.
- Raspberry Pi compatibility:
  - `evdev` is optional and lazy-loaded.
  - Stroke JSON uses standard library only.

Planned metrics:

- Character accuracy
- Word Error Rate
- Correction precision and recall
- Grammar correction accuracy
- Semantic correction acceptance rate
- End-to-end latency
- Raspberry Pi memory and CPU usage

## Next Session Focus

1. Phase 3: implement stroke and image preprocessing.
2. Add rasterization from `StrokePoint` to grayscale image arrays.
3. Add bounding-box crop, padding, resizing, and normalization.
4. Add tests for empty input, variable-size writing, and pressure-aware rendering.

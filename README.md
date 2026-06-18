# Assistive Writing Pad

Real-time handwriting recognition and intelligent correction system for children with dysgraphia.

The project is designed for laptop-first development and Raspberry Pi 4 migration:

- Python 3.9+
- CPU-only inference
- Offline local handwriting recognition
- Hybrid grammar/semantic correction with offline fallback
- Modular pipeline for capture, preprocessing, recognition, correction, display, and evaluation

## Current Status

Phase 1 is in progress: repository scaffold, core contracts, configuration, and testable pipeline skeleton.

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

Run the first writing-pad interface:

```bash
PYTHONPATH=src python -m assistive_writing_pad.display.handwriting_app
```

The first recognizer is intentionally from scratch. It starts with no model and
learns local templates from samples:

1. Write one character on the pad.
2. Enter the correct label, for example `a`.
3. Click `Save Sample`.
4. Write the character again to see recognized text update on the right.

The saved templates live in `data/user_templates.json`, which is ignored by Git.

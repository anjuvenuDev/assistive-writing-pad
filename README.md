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

Fallback template mode is still available for debugging:

1. Write one character on the pad.
2. Enter the correct label, for example `a`.
3. Click `Save Template`.
4. Write the character again to see recognized text update on the right.

The saved templates live in `data/user_templates.json`, which is ignored by Git.

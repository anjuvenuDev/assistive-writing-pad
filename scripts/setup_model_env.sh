#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
case "$(uname -m)" in
  aarch64|arm64) .venv/bin/python -m pip install 'torch>=2.6,<3' ;;
  x86_64) .venv/bin/python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu ;;
  *) echo "Local model inference requires a 64-bit OS. Install Raspberry Pi OS 64-bit." >&2; exit 1 ;;
esac
.venv/bin/python -m pip install -e ".[models,dev,hardware]"

echo "Model environment ready."
echo "Cache OCR model: .venv/bin/python scripts/cache_hf_ocr_model.py --model microsoft/trocr-base-handwritten"
echo "Cache correction models: .venv/bin/python scripts/cache_hf_correction_models.py"
echo "Run: .venv/bin/python -m assistive_writing_pad.display.web_app"

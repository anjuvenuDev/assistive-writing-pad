#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/anj/.pyenv/versions/3.10.12/bin/python}"

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[models,dev]"

echo "Model environment ready."
echo "Run: .venv/bin/python -m assistive_writing_pad.display.web_app"

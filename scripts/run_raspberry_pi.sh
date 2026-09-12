#!/usr/bin/env bash
# Run locally on Raspberry Pi OS 64-bit; all inference stays on the Pi.
set -euo pipefail
cd "$(dirname "$0")/.."

export AWP_DEVICE_PROFILE=raspberry_pi
export AWP_TORCH_THREADS="${AWP_TORCH_THREADS:-4}"
export OMP_NUM_THREADS="$AWP_TORCH_THREADS"
export MKL_NUM_THREADS="$AWP_TORCH_THREADS"
export AWP_HF_CORRECTION_DEVICE=cpu
export AWP_TROCR_LOCAL_FILES_ONLY="${AWP_TROCR_LOCAL_FILES_ONLY:-1}"
export AWP_HF_CORRECTION_LOCAL_FILES_ONLY="${AWP_HF_CORRECTION_LOCAL_FILES_ONLY:-1}"
export AWP_PRELOAD_OCR_MODEL=1
export AWP_PRELOAD_CORRECTION_MODELS=1
export AWP_WORD_SEGMENT=0
# INT8 remains explicit because target-device accuracy must be measured.
export AWP_DYNAMIC_INT8="${AWP_DYNAMIC_INT8:-0}"
exec .venv/bin/python -m assistive_writing_pad.display.web_app "$@"

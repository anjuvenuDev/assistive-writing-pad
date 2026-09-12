#!/usr/bin/env bash
# Launch the experimental small TrOCR adapter without typing its path manually.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

adapter_dir="$repo_root/models/adapters/penpal-small-v2"
if [[ ! -f "$adapter_dir/adapter.json" || ! -f "$adapter_dir/adapter.safetensors" ]]; then
  echo "Adapter files not found in: $adapter_dir" >&2
  echo "Expected adapter.json and adapter.safetensors. Check that this checkout includes the experimental adapter." >&2
  exit 1
fi

export AWP_TROCR_MODEL="microsoft/trocr-small-handwritten"
export AWP_TROCR_ADAPTER="$adapter_dir"
exec bash scripts/run_raspberry_pi.sh "$@"

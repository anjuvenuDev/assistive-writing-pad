# Experimental TrOCR adapter v2

Base: `microsoft/trocr-small-handwritten`, downloaded separately. Inference is
local CPU-only; no remote service is used. This is not a production-approved model.

Trained on the committed MIT Penpal subset: 149 synthetic lines, styles 0–8.
Validation uses 25 lines/styles 9–10; test uses 26 lines/styles 11–12. These are
synthetic styles, not independent dysgraphic child writers. Manual pad captures
were not used for training. Evaluation texts are excluded from training.

Rank 8, alpha 16, 104,448 trainable decoder query/value projection parameters,
AdamW learning rate 0.0003, batch 2, seed 2026, 512 CPU updates. Validation CER
every 64 updates selected step 192. Test results did not select that checkpoint.
The baseline was eligible if no validation checkpoint improved.

The trainer repairs three alignment problems: generation/training start-token
mismatch, doubly shifted loss targets, and an unwanted leading BOS target.
Validation exact: 7/25 -> 13/25; mean CER: 15.07% -> 9.48%.
Test exact: 6/26 -> 10/26; mean CER: 12.94% -> 9.39%.
Previously inspected test styles remain a regression benchmark, not untouched
release certification. The run used 940.5 MiB peak process RSS on an x86 host;
this is not Pi memory/latency evidence.

Full predictions, validation history and dataset fingerprints:
`data/evaluation/adapter_v2_training_report.json`. The real-pad comparison is in
`data/evaluation/small_baseline_manual_report.json` and
`data/evaluation/adapter_v2_manual_report.json`. Consult the project README for
the release decision. The loader verifies SHA-256, base identity and tensor shapes.

On the 81 audited pad captures, corrected exact increases 40/81 -> 45/81, but
sentence corrected exact decreases 19/32 -> 15/32. This category regression and
the unmet 98% gate prevent default promotion. No deployment defaults are changed.

Explicit experimental use (the launcher resolves and checks the adapter path):

```bash
bash scripts/run_raspberry_pi_adapter.sh
```

Reproduce training in a fresh output directory:

```bash
.venv/bin/python scripts/train_trocr_adapter.py --steps 512 --eval-every 64 \
  --output /tmp/penpal-v2-reproduction --report /tmp/penpal-v2-report.json
```

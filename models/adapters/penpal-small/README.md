# Experimental TrOCR adapter — not promoted

Base checkpoint: `microsoft/trocr-small-handwritten` (not bundled).
Data: [breitburg/penpal](https://huggingface.co/datasets/breitburg/penpal), MIT,
synthetic digital-ink trajectories. Source-derived records and provenance are in
`data/training/penpal`. No private/manual child captures were used for training.

64 CPU AdamW updates, rank 8, alpha 16, 104,448 trainable parameters. Only decoder
query/value projection adapter weights are included. Integrity and base-model
identity are checked by the loader using `adapter.json`.

Validation exact improved 7/25 -> 8/25, but test exact regressed 6/26 -> 5/26.
This artifact is retained for reproducibility and must not be treated as an
accuracy-approved or production-ready recognizer. See
`data/evaluation/adapter_training_report.json` and
`docs/DATASET_AND_TRAINING_REVIEW.md` for the full evidence and limitations.

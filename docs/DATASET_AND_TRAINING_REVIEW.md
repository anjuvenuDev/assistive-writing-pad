# Public digital-ink dataset review and training decision

Reviewed 2026-09-12. Target: English handwriting recognition and correction,
executed locally on Raspberry Pi 4, 4 GB, microSD. No single dataset establishes
production accuracy for dysgraphic children. Selection below is based on input
format, real-writer coverage, transcription availability, and access conditions.

| Dataset | Fit for this pad | Access and use decision |
| --- | --- | --- |
| [IAM-OnDB](https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database) | Strong candidate for English online pen trajectories and transcriptions; relevant foundation for writer-disjoint recognition evaluation. | Use official access/registration and published task splits. Not downloaded in this run; access could not be established. |
| [DeepWriting](https://ait.ethz.ch/deepwriting) | Real digital ink, character segmentation and timestamp release; combines collected samples with IAM-OnDB, 294 authors. | Dataset is CC BY-NC-SA 4.0 with additional terms and IAM registration request. Code's MIT license does not relicense the data. Not included in the product training run. |
| [Penpal](https://huggingface.co/datasets/breitburg/penpal) | Immediately usable English x/y strokes, transcript, synthetic style IDs; MIT dataset license. Useful for rendering adaptation and reproducible experiments. | Selected for the bounded local training experiment. Synthetic, generated from 13 priming handwriting styles; cannot prove dysgraphia performance. |
| [UiTM potential dysgraphia dataset](https://data.uitm.edu.my/id/eprint/108/) | Closer to the intended children and handwriting difficulties, but scanned images rather than online pad trajectories. | Verify the archive's actual transcript labels, consent/use terms and writer identifiers before OCR training. A dysgraphia class label alone is not a transcription. The repository page was unavailable during this run. |

DeepWriting's data description and license are from its [official project
page](https://ait.ethz.ch/deepwriting), rather than the code license.
Penpal's [dataset card](https://huggingface.co/datasets/breitburg/penpal) explicitly
describes its synthetic generation and `author` IDs as style IDs.

## What was actually run

Hardware inspection found CPU-only PyTorch 2.6.0, no CUDA, eight logical CPUs,
and approximately 4 GB available RAM. No paid cloud training was launched.

The importer downloads 200 Penpal rows starting at offset 1000, keeps complete
strokes without label-dependent geometry filtering, and excludes normalized
texts already present in the committed evaluation corpora. Styles 0–8 train,
9–10 validate, and 11–12 test. Duplicate normalized text is excluded across all
splits. These are disjoint **synthetic styles**, not independent child writers.

The local experiment starts from cached `microsoft/trocr-small-handwritten`,
freezes the base checkpoint and trains rank-8 LoRA matrices on decoder attention
query/value projections. Frozen encoder outputs are cached to bound CPU work.
It runs 64 AdamW updates, batch size 2, seed 2026, learning rate 0.0003. Validation
and test predictions are measured before and after training, with no test-based
checkpoint selection. Training loss alone is never a release criterion.

```bash
.venv/bin/python scripts/prepare_penpal_training.py
.venv/bin/python scripts/train_trocr_adapter.py --steps 64
```

Both commands refuse to overwrite existing output directories. Training refuses
quantized or already-adapted bases. The checkpoint is stored as safetensors with
base-model identity and SHA-256 verification, and is explicitly experimental.
The runtime does not select it automatically:

```bash
AWP_TROCR_MODEL=microsoft/trocr-small-handwritten \
AWP_TROCR_ADAPTER=models/adapters/penpal-small \
AWP_TROCR_LOCAL_FILES_ONLY=1 bash scripts/run_raspberry_pi.sh
```

Use `data/evaluation/adapter_training_report.json` for actual loss curves,
before/after predictions, dataset fingerprints, elapsed time and memory. Decoder
timings in this report exclude the cached encoder, capture and correction and
are **not** end-to-end or Pi latency evidence.

## Release decision

A short synthetic adapter run is not a production training campaign. A release
requires representative, independently annotated dysgraphic handwriting,
writer-disjoint held-out evaluation, >=98% corrected accuracy, protection against
meaning-changing corrections, and measured latency/memory on the physical Pi.
None of these requirements is waived by a passing software test suite or a
deadline. The trained adapter remains a candidate unless that evidence exists.

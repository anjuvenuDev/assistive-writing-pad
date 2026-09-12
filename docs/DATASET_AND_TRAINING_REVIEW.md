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

## Follow-up: repair training alignment before expanding the corpus

Inspection of the cached small checkpoint and installed Transformers implementation
found two training defects. The checkpoint's generation start token is 2, whereas
the old trainer used tokenizer CLS (0) when the top-level training configuration
was unset. More seriously, the installed composite model's default loss fell back
to causal-LM loss: decoder inputs were already right-shifted, but that loss shifted
labels again. Low training loss under this objective did not measure correct OCR
learning. The original 64-step checkpoint remains historical, not recommended.

The trainer now matches generation special tokens (including legitimate token 0)
and explicitly computes cross-entropy at matching decoder/target positions, ignoring
padding. Regression tests cover both behaviors. Every 128 steps it evaluates the
validation styles and saves the lowest validation CER checkpoint, breaking ties
with exact accuracy. The unchanged baseline is also eligible. Test predictions
never choose the checkpoint. Previously inspected test styles are a regression
benchmark, not a newly untouched certification set.

An intermediate 512-step run fixing only the start token selected step 128 and
scored 8/25 validation, 3/26 test; it is rejected. Its report is preserved as
`data/evaluation/adapter_aligned_training_report.json`. The subsequent seq2seq-loss
run fixes both defects, using the same 149 training lines for a controlled
comparison instead of conflating the fix with additional data.

That aligned-loss run reached 12/25 validation and 11/26 test exact (baseline
7/25 and 6/26), selecting step 512. Its report is
`data/evaluation/adapter_seq2seq_training_report.json`. A token-level diagnostic
then showed an additional mismatch: the tokenizer prepended BOS to labels but
the pretrained decoder generated text immediately after its start token. On the
first validation line, baseline teacher-forced loss was 9.887 with that extra BOS
and 1.206 without it. The final trainer therefore tokenizes text without automatic
special tokens and appends EOS explicitly. Decoder start is supplied separately.
The `penpal-small-v2` experiment evaluates validation every 64 steps for 512 steps.

Final v2 selects step 192: validation exact 7/25 -> 13/25, mean CER 15.0723% ->
9.4839%; test exact 6/26 -> 10/26, mean CER 12.9432% -> 9.3904%. The CPU run took
195.2 seconds with 940.5 MiB peak RSS. Another benchmark ran concurrently for
part of that time, so elapsed times are not controlled performance comparisons.
The final checkpoint, model card and full report are committed. This establishes
an improvement on this small synthetic regression set, not >=98% accuracy or
generalization to dysgraphic children. No new corpus was added in this follow-up:
the controlled method comparison used the same data and excluded pad captures.

### Full pad-pipeline regression check

Both runs use `microsoft/trocr-small-handwritten`, FP32 CPU, default automatic
routing and the same correction pipeline/confidence threshold (0.85), on the
81 audited captures. Only v2 loads the adapter. Original conflicting-label ink
remains excluded by the existing audit, not relabeled.

| Exact outputs | Small baseline | Small + v2 |
| --- | ---: | ---: |
| Raw OCR, all captures | 36/81 | 44/81 |
| Corrected, all captures | 40/81 | 45/81 |
| Corrected characters | 5/25 | 11/25 |
| Corrected words | 16/24 | 19/24 |
| Corrected sentences | 19/32 | 15/32 |

Reports: `small_baseline_manual_report.json` and `adapter_v2_manual_report.json`
under `data/evaluation`. V2 mean corrected CER is 26.4% versus baseline 33.0%.
V2 x86 end-to-end average/p95 is 591.9/1449.9 ms. Baseline ran partly concurrently
with training; do not interpret timing differences as a controlled speedup.
Both accuracy gates fail the unchanged 98% requirement. The sentence regression
prevents default promotion despite aggregate improvement. No Pi measurements
or child-writer-independent dysgraphia accuracy are claimed. The production
default remains the existing base checkpoint, not the small model or adapter.

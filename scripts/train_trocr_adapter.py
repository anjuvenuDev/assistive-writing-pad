#!/usr/bin/env python
"""Train a bounded TrOCR adapter, evaluate held-out styles, never auto-promote."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from assistive_writing_pad.eval.recognition_eval import parse_stroke_groups, safe_cer  # noqa: E402
from assistive_writing_pad.recognition.adapters import install_adapter  # noqa: E402
from assistive_writing_pad.recognition.trocr import (  # noqa: E402
    TrOCRHandwritingRecognizer, _preprocess_image, render_stroke_groups_for_trocr,
)


def validate_splits(splits):
    texts, styles = set(), set()
    for name, records in splits.items():
        if not records:
            raise ValueError(f'{name} split is empty')
        current_texts = {' '.join(re.findall(r'[a-z0-9]+', r['expected'].casefold())) for r in records}
        current_styles = {r['style_id'] for r in records}
        if texts & current_texts or styles & current_styles:
            raise ValueError('training/evaluation text or style leakage')
        texts.update(current_texts)
        styles.update(current_styles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/training/penpal'))
    parser.add_argument('--output', type=Path, default=Path('models/adapters/penpal-small'))
    parser.add_argument('--report', type=Path, default=Path('data/evaluation/adapter_training_report.json'))
    parser.add_argument('--model', default='microsoft/trocr-small-handwritten')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error('steps, batch size and learning rate must be positive')
    if args.output.exists():
        parser.error('output exists; choose a new checkpoint directory')
    if os.environ.get('AWP_DYNAMIC_INT8', '0') != '0' or os.environ.get('AWP_TROCR_ADAPTER'):
        parser.error('train from an unquantized base without an already-loaded adapter')
    splits = {name: [json.loads(line) for line in (args.data / (name+'.jsonl')).read_text().splitlines()]
              for name in ('train', 'validation', 'test')}
    validate_splits(splits)
    import torch
    from safetensors.torch import save_file
    from transformers.modeling_outputs import BaseModelOutput
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    recognizer = TrOCRHandwritingRecognizer(model_name=args.model, local_files_only=True,
                                           num_beams=2, num_return_sequences=1)
    recognizer._ensure_loaded()
    model, processor = recognizer._model, recognizer._processor
    model.config.decoder_start_token_id = model.config.decoder_start_token_id or processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    started = time.perf_counter()
    encoded = {}
    for name, records in splits.items():
        encoded[name] = []
        for record in records:
            strokes = parse_stroke_groups(record['strokes'])
            image, crop, _ = _preprocess_image(render_stroke_groups_for_trocr(strokes))
            if not crop.valid:
                raise ValueError('invalid training ink: ' + record['id'])
            pixels = processor(images=image, return_tensors='pt').pixel_values
            # Cache the frozen encoder once. Normal tensors allow decoder backward.
            with torch.no_grad():
                hidden = model.encoder(pixel_values=pixels).last_hidden_state.detach()
            labels = processor.tokenizer(record['expected'], return_tensors='pt').input_ids
            if labels.shape[1] > 96:
                raise ValueError('target exceeds training token limit')
            encoded[name].append((hidden, labels, record))
        print(f'Encoded {name}: {len(records)} lines', flush=True)

    def evaluate(name):
        model.eval()
        rows = []
        for hidden, labels, record in encoded[name]:
            began = time.perf_counter()
            with torch.inference_mode():
                tokens = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                                        max_new_tokens=96, num_beams=2)
            predicted = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            expected = record['expected'].strip()
            rows.append(dict(id=record['id'], expected=expected, predicted=predicted,
                             exact=predicted.casefold() == expected.casefold(),
                             cer=safe_cer(expected.casefold(), predicted.casefold()),
                             decoder_ms=(time.perf_counter()-began)*1000))
        return dict(count=len(rows), exact=sum(r['exact'] for r in rows),
                    mean_cer=sum(r['cer'] for r in rows)/len(rows), rows=rows)

    baseline_validation = evaluate('validation')
    baseline_test = evaluate('test')
    print('Baseline validation:', baseline_validation['exact'], '/', baseline_validation['count'], flush=True)
    parameters = install_adapter(model, rank=8, alpha=16)
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=args.learning_rate)
    losses = []
    examples = encoded['train']
    model.eval()  # deterministic frozen backbone; LoRA gradients remain enabled
    for step in range(args.steps):
        batch = random.sample(examples, min(args.batch_size, len(examples)))
        hidden = torch.cat([item[0] for item in batch])
        labels = torch.nn.utils.rnn.pad_sequence(
            [item[1].squeeze(0) for item in batch], batch_first=True, padding_value=-100,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(encoder_outputs=BaseModelOutput(last_hidden_state=hidden), labels=labels,
                       use_cache=False)
        if not torch.isfinite(output.loss):
            raise RuntimeError('non-finite training loss')
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0)
        optimizer.step()
        losses.append(float(output.loss.detach()))
        if (step+1) % 8 == 0:
            print(f'Step {step+1}/{args.steps}: loss={losses[-1]:.4f}', flush=True)
    final_validation = evaluate('validation')
    final_test = evaluate('test')
    args.output.mkdir(parents=True)
    metadata = dict(base_model=args.model, rank=8, alpha=16, steps=args.steps, seed=args.seed,
                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                    trainable_parameters=sum(p.numel() for p in parameters.values()),
                    promoted=False, status='experimental_not_production')
    save_file({name: p.detach().cpu().contiguous() for name, p in parameters.items()},
              str(args.output / 'adapter.safetensors'))
    metadata['sha256'] = hashlib.sha256((args.output / 'adapter.safetensors').read_bytes()).hexdigest()
    (args.output / 'adapter.json').write_text(json.dumps(metadata, indent=2)+'\n')
    report = dict(configuration=metadata, provenance=json.loads((args.data/'provenance.json').read_text()),
                  losses=losses, baseline_validation=baseline_validation, baseline_test=baseline_test,
                  final_validation=final_validation, final_test=final_test,
                  elapsed_seconds=time.perf_counter()-started,
                  peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  ready_for_production=False,
                  limitations=['Small synthetic training subset; no dysgraphia validation',
                               'No physical Pi timing or memory measurements',
                               'Decoder timings exclude cached encoder, capture and correction',
                               'Existing manual and Penpal evaluation cases were excluded from training'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('configuration', 'elapsed_seconds',
                                                 'peak_process_rss_mib')}, indent=2))
    print(f'Final validation {final_validation["exact"]}/{final_validation["count"]}; '
          f'test {final_test["exact"]}/{final_test["count"]}', flush=True)


if __name__ == '__main__':
    main()

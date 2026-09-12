#!/usr/bin/env python
"""Download a bounded public stroke corpus with disjoint synthetic style splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from assistive_writing_pad.eval.corpus import fetch_penpal_rows  # noqa: E402


def text_key(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.casefold()))


def build_records(rows, excluded=()):
    seen = set(excluded)
    splits = {'train': [], 'validation': [], 'test': []}
    for row in rows:
        raw = row['row']
        text = raw['text'].strip()
        key = text_key(text)
        if not key or key in seen:
            continue
        author = int(raw['author'])
        if not 0 <= author <= 12:
            raise ValueError('unexpected Penpal style ID')
        split = 'train' if author <= 8 else 'validation' if author <= 10 else 'test'
        strokes = []
        tick = 0
        for word in raw['strokes']:
            for stroke in word:
                points = []
                for point in stroke['points']:
                    points.append(dict(x=float(point['x']), y=float(point['y']),
                                       timestamp_ms=tick, pressure=1.0))
                    tick += 8
                if points:
                    strokes.append(points)
                tick += 24
        if not strokes:
            continue
        seen.add(key)
        splits[split].append(dict(
            id='penpal_train_' + hashlib.sha256(text.encode()).hexdigest()[:16],
            category='sentence', source='hf_penpal_synthetic', expected=text,
            expected_recognized=text, style_id=author, strokes=strokes,
            dataset_row=row['row_idx'], dataset='breitburg/penpal', license='MIT',
        ))
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offset', type=int, default=1000)
    parser.add_argument('--rows', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('data/training/penpal'))
    args = parser.parse_args()
    if args.rows <= 0 or args.offset < 0:
        parser.error('rows must be positive and offset nonnegative')
    if args.output.exists():
        parser.error('output already exists; choose a new directory to preserve provenance')
    excluded = set()
    for path in Path('data/evaluation').glob('*cases.jsonl'):
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                record = json.loads(line)
                excluded.add(text_key(record.get('expected', '')))
    rows = []
    for offset in range(args.offset, args.offset + args.rows, 100):
        rows.extend(fetch_penpal_rows(offset=offset, length=min(100, args.offset + args.rows-offset)))
    splits = build_records(rows, excluded)
    if any(not records for records in splits.values()):
        raise ValueError('not enough style coverage to create all three splits')
    args.output.mkdir(parents=True)
    manifest = {'dataset': 'https://huggingface.co/datasets/breitburg/penpal',
                'license': 'MIT', 'offset': args.offset, 'requested_rows': args.rows,
                'source': 'synthetic', 'splits': {}}
    for split, records in splits.items():
        path = args.output / (split + '.jsonl')
        content = ''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in records)
        path.write_text(content)
        manifest['splits'][split] = dict(count=len(records),
            styles=dict(Counter(r['style_id'] for r in records)),
            sha256=hashlib.sha256(content.encode()).hexdigest())
    (args.output / 'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

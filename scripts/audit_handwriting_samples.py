#!/usr/bin/env python
"""Audit labels and produce a benchmark excluding ambiguous duplicate ink."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def audit(records):
    groups = defaultdict(list)
    ids = Counter(record['id'] for record in records)
    for index, record in enumerate(records):
        # Coordinates, not timestamps/pressure, define the visible handwriting.
        ink = [[(p['x'], p['y']) for p in stroke] for stroke in record['strokes']]
        key = hashlib.sha256(json.dumps(ink, separators=(',', ':')).encode()).hexdigest()
        groups[key].append(index)
    excluded = {}
    for indices in groups.values():
        labels = {(records[i]['expected'], records[i].get('expected_recognized')) for i in indices}
        if len(labels) > 1:
            for i in indices:
                excluded[i] = 'identical_ink_conflicting_labels'
        else:
            for i in indices[1:]:
                excluded[i] = 'duplicate_labeled_ink'
    for i, record in enumerate(records):
        if ids[record['id']] > 1:
            excluded[i] = 'duplicate_id'
    valid = [r for i, r in enumerate(records) if i not in excluded]
    report = {
        'input_cases': len(records), 'benchmark_cases': len(valid),
        'excluded': [{'id': records[i]['id'], 'reason': reason} for i, reason in excluded.items()],
        'categories': dict(Counter(r['category'] for r in valid)),
        'note': 'Original captures are preserved. No labels were inferred or rewritten.',
    }
    return valid, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/evaluation/end_to_end_cases.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases-output', type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.resolve() in {args.output.resolve(), args.cases_output.resolve()}:
        parser.error('outputs must not overwrite the source captures')
    records = [json.loads(line) for line in args.manifest.read_text().splitlines()
               if line.strip() and not line.startswith('#')]
    valid, report = audit(records)
    for path in (args.output, args.cases_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    args.cases_output.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in valid))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

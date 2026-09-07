#!/usr/bin/env python
"""Import synthetic Hugging Face Penpal stroke samples into an evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from assistive_writing_pad.eval.corpus import (  # noqa: E402
    append_jsonl_record,
    build_penpal_case_records,
    fetch_penpal_rows,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/penpal_end_to_end_cases.jsonl"),
        help="Manifest to write synthetic Penpal cases into.",
    )
    parser.add_argument("--offset", type=int, default=0, help="Dataset row offset.")
    parser.add_argument("--length", type=int, default=20, help="Dataset rows to inspect.")
    parser.add_argument(
        "--max-sentence-cases",
        type=int,
        default=2,
        help="Maximum full-line sentence cases to append.",
    )
    parser.add_argument(
        "--max-word-cases",
        type=int,
        default=6,
        help="Maximum word-level cases to append.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print records without modifying the manifest.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the manifest before writing imported samples.",
    )
    args = parser.parse_args(argv)

    try:
        rows = fetch_penpal_rows(offset=args.offset, length=args.length)
        records = build_penpal_case_records(
            rows,
            max_sentence_cases=args.max_sentence_cases,
            max_word_cases=args.max_word_cases,
        )
    except Exception as exc:
        print(f"Could not import Penpal samples: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("No Penpal samples matched the requested import limits.", file=sys.stderr)
        return 1

    if args.replace and not args.dry_run:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text("", encoding="utf-8")

    for record in records:
        if args.dry_run:
            print(json.dumps(record, separators=(",", ":")))
        else:
            append_jsonl_record(args.manifest, record)
    if not args.dry_run:
        print(f"Appended {len(records)} Penpal samples to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

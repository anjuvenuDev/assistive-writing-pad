#!/usr/bin/env python
"""Append a captured stroke payload to the end-to-end evaluation manifest."""

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
    build_end_to_end_case_record,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload",
        required=True,
        help="Path to exported stroke payload JSON, or '-' for stdin.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/end_to_end_cases.jsonl"),
        help="Evaluation JSONL manifest to append.",
    )
    parser.add_argument("--id", required=True, help="Stable case id.")
    parser.add_argument("--category", required=True, help="Case category.")
    parser.add_argument("--expected", required=True, help="Expected final corrected text.")
    parser.add_argument(
        "--expected-recognized",
        default=None,
        help="Expected raw OCR text when it should differ from corrected text.",
    )
    parser.add_argument("--notes", default="", help="Optional manifest note.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the JSONL record without modifying the manifest.",
    )
    args = parser.parse_args(argv)

    try:
        stroke_payload = load_payload(args.payload)
        record = build_end_to_end_case_record(
            case_id=args.id,
            category=args.category,
            expected=args.expected,
            expected_recognized=args.expected_recognized,
            notes=args.notes,
            stroke_payload=stroke_payload,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not build evaluation case: {exc}", file=sys.stderr)
        return 1

    line = json.dumps(record, separators=(",", ":"))
    if args.dry_run:
        print(line)
        return 0

    append_jsonl_record(args.manifest, record)
    print(f"Appended {record['id']} to {args.manifest}")
    return 0


def main_with_args(argv: List[str]) -> int:
    return main(argv)


def load_payload(path: str) -> dict:
    if path == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())

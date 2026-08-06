#!/usr/bin/env python3
"""Fail unless two MPFR validation artifacts have identical decoding paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    left = {(r["width"], r["prompt_idx"]): r for r in baseline["rows"]}
    right = {(r["width"], r["prompt_idx"]): r for r in candidate["rows"]}
    if left.keys() != right.keys():
        raise SystemExit("FAIL: benchmark cells differ")
    for key in sorted(left):
        for field in ("output_ids", "blocks", "tokens", "num_blocks", "AATPS"):
            if left[key][field] != right[key][field]:
                raise SystemExit(f"FAIL: {key} differs in {field}")
    print(f"PASS: {len(left)} cells have identical tokens, blocks, and AATPS")


if __name__ == "__main__":
    main()

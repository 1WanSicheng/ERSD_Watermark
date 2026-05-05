"""Compute pairwise ROUGE-L between drafter-T conditions.

Standard NLG metric (Lin 2004) used by Rowan et al. (2025) and many others
for output-similarity audits. For each (decoder, prompt_idx), we have an
output token sequence under each drafter-T condition; we compute the
pairwise ROUGE-L F1 across the conditions and report the per-prompt
minimum (worst pair) averaged over prompts.

We use a pure-Python ROUGE-L F1 (no rouge_score dependency) since we
operate on token-id sequences, not text.

Usage:
  python -m experiments.compute_pairwise_rouge \\
      outputs/single_draft_qwen_cnn_n100_DI_Td05.json \\
      outputs/single_draft_qwen_cnn_n100_DI_Td10.json \\
      outputs/single_draft_qwen_cnn_n100_DI_Td15.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import List, Tuple


def lcs_length(a: List[int], b: List[int]) -> int:
    """Length of longest common subsequence."""
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    if n < m:
        a, b = b, a
        n, m = m, n
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] > prev[j] else prev[j]
        prev = cur
    return prev[m]


def rouge_l_f1(a: List[int], b: List[int]) -> float:
    if not a or not b:
        return 0.0
    L = lcs_length(a, b)
    if L == 0:
        return 0.0
    p = L / len(b)
    r = L / len(a)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def _label_from_path(p: Path) -> str:
    name = p.stem
    # Drafter-invariance tags: temperatures (Td05, Td10, Td15) and model
    # variants (D0, D1, D2, D3) used in Exp 2 of ablation_plan_2026-05-05.md.
    for tag in ("Td05", "Td10", "Td15", "Td20", "Td01", "D0", "D1", "D2", "D3"):
        if tag in name:
            return tag
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()

    runs = []
    for p in args.paths:
        runs.append((_label_from_path(p), json.load(open(p))))

    # Index by (decoder, prompt_idx) -> {label: output_ids}
    seqs: dict = defaultdict(dict)
    decoders_seen: set = set()
    for label, d in runs:
        for r in d["rows"]:
            ids = r.get("output_ids")
            if ids is None:
                continue
            seqs[(r["decoder"], int(r["prompt_idx"]))][label] = list(ids)
            decoders_seen.add(r["decoder"])

    labels = [lab for lab, _ in runs]
    pairs = list(combinations(labels, 2))

    print(f"# Pairwise ROUGE-L between drafter-T conditions {labels}")
    print(f"{'decoder':<26}" + "".join(f"  ROUGE-L({a},{b})".rjust(20) for a, b in pairs)
          + f"  {'min-pair mean':>15}  {'min-pair median':>17}")
    print("-" * (26 + 20 * len(pairs) + 36))

    for dec in sorted(decoders_seen):
        per_prompt_mins = []
        per_pair_scores = {pair: [] for pair in pairs}
        for (d, idx), per_label in seqs.items():
            if d != dec:
                continue
            if not all(lab in per_label for lab in labels):
                continue
            pair_vals = []
            for a, b in pairs:
                v = rouge_l_f1(per_label[a], per_label[b])
                per_pair_scores[(a, b)].append(v)
                pair_vals.append(v)
            if pair_vals:
                per_prompt_mins.append(min(pair_vals))
        if not per_prompt_mins:
            continue
        line = f"{dec:<26}"
        for a, b in pairs:
            vals = per_pair_scores[(a, b)]
            line += f"  {statistics.mean(vals):>18.4f}"
        line += f"  {statistics.mean(per_prompt_mins):>15.4f}"
        line += f"  {statistics.median(per_prompt_mins):>17.4f}"
        print(line)


if __name__ == "__main__":
    main()

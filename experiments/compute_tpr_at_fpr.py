"""Compute TPR @ FPR=1% across drafter-T conditions.

Reads the 3 single-draft DI outputs (T_d ∈ {0.5, 1.0, 1.5}). For each
(decoder, T_d) cell, computes:

  - Detection score per prompt = ANLPPT_U (or its un-normalized log p-value)
  - H0 distribution = the pfr_no_watermark scores in that same cell
  - threshold τ such that P(score > τ | H0) = 1%
  - TPR = P(score > τ | H1=watermarked decoder)

Reports the watermark-detector TPR @ FPR=1% per (decoder, T_d) cell, and
the spread across drafter T (the key "drafter invariance" metric for
detection robustness).

Usage:
  python -m experiments.compute_tpr_at_fpr \\
      outputs/single_draft_qwen_cnn_n100_DI_Td05.json \\
      outputs/single_draft_qwen_cnn_n100_DI_Td10.json \\
      outputs/single_draft_qwen_cnn_n100_DI_Td15.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _label_from_path(p: Path) -> str:
    name = p.stem
    for tag in ("Td05", "Td10", "Td15", "Td20", "Td01"):
        if tag in name:
            return tag
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--h0-decoder", default="pfr_no_watermark",
                    help="decoder name to use as H0 (no-watermark) distribution")
    ap.add_argument("--score-key", default="ANLPPT_U")
    ap.add_argument("--fpr", type=float, default=0.01)
    args = ap.parse_args()

    runs = []
    for p in args.paths:
        runs.append((_label_from_path(p), json.load(open(p))))

    # Index by (decoder, label, prompt_idx) -> score
    indexed: Dict[Tuple[str, str, int], float] = {}
    decoders_seen = set()
    for label, d in runs:
        for r in d["rows"]:
            dec = r["decoder"]
            decoders_seen.add(dec)
            score = r.get(args.score_key)
            if score is None or score != score:  # NaN check
                continue
            indexed[(dec, label, int(r["prompt_idx"]))] = score

    labels = [lab for lab, _ in runs]

    # H0 scores per drafter-T condition (from h0-decoder)
    h0_per_label: Dict[str, List[float]] = defaultdict(list)
    for (dec, lab, idx), score in indexed.items():
        if dec == args.h0_decoder:
            h0_per_label[lab].append(score)

    if not h0_per_label:
        raise SystemExit(
            f"No rows found for H0 decoder '{args.h0_decoder}'. "
            f"Available: {sorted(decoders_seen)}"
        )

    # H0-merged: pool all conditions for a single threshold
    h0_pooled = [s for ss in h0_per_label.values() for s in ss]
    tau_pooled = float(np.quantile(h0_pooled, 1.0 - args.fpr))
    # Per-condition thresholds (alternative)
    tau_per_label = {
        lab: float(np.quantile(scores, 1.0 - args.fpr))
        for lab, scores in h0_per_label.items()
    }

    print(f"# TPR @ FPR={args.fpr:.0%} across drafter-T conditions")
    print(f"# Score: {args.score_key}")
    print(f"# H0 decoder: {args.h0_decoder}")
    print(f"# H0 pool size: {len(h0_pooled)} (across {len(labels)} conditions)")
    print(f"# Pooled threshold τ_pooled = {tau_pooled:.4f}")
    for lab, t in tau_per_label.items():
        print(f"#   per-condition τ_{lab} = {t:.4f}")
    print()

    # Compute TPR per (decoder, condition) using pooled threshold
    print(f"{'decoder':<26}" + "".join(f"  TPR@{lab}".rjust(12) for lab in labels)
          + f"  {'spread (max-min)':>18}")
    print("-" * (26 + 12 * len(labels) + 20))
    for dec in sorted(decoders_seen):
        if dec == args.h0_decoder:
            continue
        tprs = {}
        for lab in labels:
            scores = [
                s for (d, l, _), s in indexed.items()
                if d == dec and l == lab
            ]
            if not scores:
                tprs[lab] = float("nan")
                continue
            tprs[lab] = float(np.mean(np.asarray(scores) > tau_pooled))
        vals = [tprs[lab] for lab in labels if tprs[lab] == tprs[lab]]
        spread = (max(vals) - min(vals)) if vals else float("nan")
        row = f"{dec:<26}" + "".join(f"  {tprs[lab]:>10.1%}" for lab in labels)
        row += f"  {spread:>17.1%}"
        print(row)

    # Also print mean / std-error of detection score per (decoder, condition)
    print()
    print(f"# Mean {args.score_key} per (decoder, condition) ± SEM:")
    print(f"{'decoder':<26}" + "".join(f"  {lab}".rjust(15) for lab in labels))
    print("-" * (26 + 15 * len(labels)))
    for dec in sorted(decoders_seen):
        line = f"{dec:<26}"
        for lab in labels:
            scores = [s for (d, l, _), s in indexed.items() if d == dec and l == lab]
            if scores:
                m = statistics.mean(scores)
                sem = statistics.stdev(scores) / (len(scores) ** 0.5) if len(scores) >= 2 else 0
                line += f"  {m:>7.4f}±{sem:.4f}"
            else:
                line += "         nan    "
        print(line)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate multi-seed Qwen shard outputs into paper-style Tables 5-6.

Seed 1 = outputs/multi/shards/shard_qwen_*_n1000_B*.json  (seed_offset 0)
Seeds 2,3 = outputs/multi/shards_seeds/shard_qwen_*_n1000_B*_s{2,3}.json

For each (cell, decoder, B) the per-seed cell mean is taken from the shard's
"summary" block; the table reports mean +- std over the seed-level means
(ddof=1).  Emits a plain-text table and a LaTeX fragment per cell.

Usage: python scripts/aggregate_3seed_tables.py [--latex]
"""
import argparse
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRICS = ["AATPS", "token_rate", "ANLPPT_U", "ANLPPT_Li", "ANLPPT_PL", "LPPL"]
HEADS = ["AATPS", "TR", "ANLPPT-U", "ANLPPT-Li", "ANLPPT-PL", "LPPL"]
DECS = [("mpfr_torchgen_cached", "MPFR"), ("invariant_multi", "Invariant")]


def load_seed(pattern: str) -> dict:
    """-> {(decoder, B): {metric: value}} from shard summary blocks."""
    out = {}
    for f in glob.glob(pattern):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for key, agg in d["summary"].items():
            dec, rest = key.rsplit("_L", 1)
            B = int(rest.split("_B")[1])
            out[(dec, B)] = {m: agg.get(m, float("nan")) for m in METRICS}
    return out


def mean_std(vals):
    vals = [v for v in vals if not math.isnan(v)]
    n = len(vals)
    mu = sum(vals) / n
    sd = (sum((v - mu) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mu, sd


PAIRS = {
    "qwen": {
        "title": "Qwen2.5-7B-Instruct",
        "patterns": lambda cell: [
            f"outputs/multi/shards/shard_qwen_{cell}_n1000_B*.json",
            f"outputs/multi/shards_seeds/shard_qwen_{cell}_n1000_B*_s2.json",
            f"outputs/multi/shards_seeds/shard_qwen_{cell}_n1000_B*_s3.json",
        ],
    },
    "llama": {
        "title": "Llama-3.1-8B-Instruct",
        "patterns": lambda cell: [
            f"outputs/multi/shards_llama/shard_llama_{cell}_n1000_B*_s{s}.json"
            for s in (1, 2, 3)
        ],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--pair", choices=sorted(PAIRS), default="qwen")
    args = ap.parse_args()

    spec = PAIRS[args.pair]
    for cell, ds_name in (("cnn", "CNN/DailyMail"), ("eli5", "ELI5")):
        title = f"{spec['title']} x {ds_name}"
        seeds = [load_seed(str(ROOT / p)) for p in spec["patterns"](cell)]
        seeds = [s for s in seeds if s]
        print(f"\n=== {title}  (L=4, n=1000, {len(seeds)} seeds) ===")
        print(f"{'Decoder':<10}{'B':>2}  " + "  ".join(f"{h:>16}" for h in HEADS))
        latex_rows = []
        for dec_key, dec_name in DECS:
            for B in (2, 4, 6, 8):
                per_metric = []
                for m in METRICS:
                    mu, sd = mean_std([s[(dec_key, B)][m] for s in seeds
                                       if (dec_key, B) in s])
                    per_metric.append((mu, sd))
                cells_txt = []
                cells_tex = []
                for (mu, sd), m in zip(per_metric, METRICS):
                    prec = 2 if m == "token_rate" else 3
                    cells_txt.append(f"{mu:>9.{prec}f}±{sd:<6.{prec}f}")
                    cells_tex.append(f"${mu:.{prec}f} \\pm {sd:.{prec}f}$")
                print(f"{dec_name:<10}{B:>2}  " + "  ".join(cells_txt))
                latex_rows.append(
                    f"\\textsc{{{dec_name}}} & {B} & " + " & ".join(cells_tex) + r" \\")
        if args.latex:
            print(f"\n% ---- LaTeX rows: {title} ----")
            for r in latex_rows:
                print(r)


if __name__ == "__main__":
    main()

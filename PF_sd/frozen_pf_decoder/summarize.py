#!/usr/bin/env python3
"""Summarize correctness and runtime of tree vs tree-free Latin-PF runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    tree = {
        int(row["prompt_idx"]): row
        for row in rows
        if row["method"] == "latin_pf_counter_fused"
    }
    flat = {
        int(row["prompt_idx"]): row
        for row in rows
        if row["method"] == "latin_pf_counter_tree_free"
    }
    common = sorted(set(tree) & set(flat))
    exact_tokens = [
        tree[idx].get("output_token_ids") == flat[idx].get("output_token_ids")
        for idx in common
    ]
    exact_blocks = [tree[idx]["blocks"] == flat[idx]["blocks"] for idx in common]
    exact_signals = [
        all(
            tree[idx].get(metric) == flat[idx].get(metric)
            for metric in ("PF_ANLPPT_original", "PF_Li_score", "PF_PL_score")
        )
        for idx in common
    ]
    tree_elapsed = np.array([tree[idx]["elapsed_sec"] for idx in common])
    flat_elapsed = np.array([flat[idx]["elapsed_sec"] for idx in common])
    tree_tokens = sum(int(tree[idx]["tokens"]) for idx in common)
    flat_tokens = sum(int(flat[idx]["tokens"]) for idx in common)
    return {
        "file": str(path),
        "n_matched_prompts": len(common),
        "exact_token_paths": int(sum(exact_tokens)),
        "exact_block_counts": int(sum(exact_blocks)),
        "exact_watermark_metrics": int(sum(exact_signals)),
        "tree_token_rate": tree_tokens / float(tree_elapsed.sum()),
        "tree_free_token_rate": flat_tokens / float(flat_elapsed.sum()),
        "tree_free_tr_change_pct": 100.0
        * (tree_elapsed.sum() / flat_elapsed.sum() - 1.0),
        "median_paired_latency_change_pct": 100.0
        * float(np.median(flat_elapsed / tree_elapsed - 1.0)),
        "tree_aatps": tree_tokens
        / float(sum(int(tree[idx]["blocks"]) for idx in common)),
        "tree_free_aatps": flat_tokens
        / float(sum(int(flat[idx]["blocks"]) for idx in common)),
        "tree_peak_gib": max(float(tree[idx]["peak_allocated_gib"]) for idx in common),
        "tree_free_peak_gib": max(
            float(flat[idx]["peak_allocated_gib"]) for idx in common
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.results], indent=2))


if __name__ == "__main__":
    main()

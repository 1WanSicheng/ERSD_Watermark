"""Analyze drafter-invariance experiments.

Reads multiple multi_draft output JSONs that share (target, prompt set,
target_T, key) and differ only in ``draft_temperature``. For each
(decoder, num_drafts, prompt_idx) cell it checks whether the output
token sequence is byte-identical across the drafter conditions, and
reports per-cell:
  - identical_rate: fraction of prompts where outputs match exactly
                    across all conditions
  - U_std_across_Td: per-prompt std of ANLPPT_U across the conditions
                     (averaged over prompts)
  - AATPS / token_rate per condition

Usage:
  python -m experiments.analyze_drafter_invariance \\
      outputs/drafter_invariance_qwen_cnn_n200_Td05.json \\
      outputs/drafter_invariance_qwen_cnn_n200_Td10.json \\
      outputs/drafter_invariance_qwen_cnn_n200_Td15.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _label_from_path(p: Path) -> str:
    name = p.stem
    # heuristic: trailing TdXY
    for tag in ("Td05", "Td10", "Td15", "Td20", "Td01"):
        if tag in name:
            return tag
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path,
                    help="multi-draft output JSONs that differ only in draft_temperature")
    args = ap.parse_args()

    if len(args.paths) < 2:
        raise SystemExit("need at least 2 paths to compare")

    runs: List[Tuple[str, dict]] = []
    for p in args.paths:
        with open(p) as f:
            d = json.load(f)
        runs.append((_label_from_path(p), d))

    # Index rows by (decoder, num_drafts, prompt_idx)
    indexed: Dict[Tuple[str, int, int], Dict[str, dict]] = defaultdict(dict)
    for label, d in runs:
        for r in d.get("rows", []):
            key = (r["decoder"], int(r["num_drafts"]), int(r["prompt_idx"]))
            indexed[key][label] = r

    # Verify each cell is populated for every condition
    labels = [lab for lab, _ in runs]
    cells_by_combo: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for (decoder, B, idx), per_label in indexed.items():
        if len(per_label) == len(labels):
            cells_by_combo[(decoder, B)].append(idx)

    print(f"# Drafter-invariance analysis")
    print(f"# Conditions: {labels}")
    print()

    # Per (decoder, B) cell summary
    print(f"{'decoder':<28}{'B':>3} {'n':>5} "
          + " ".join(f"{lab+'_AATPS':>13}" for lab in labels)
          + f" {'identical_rate':>15} {'U_std_avg':>10}")
    print("-" * (40 + 13 * len(labels) + 27))

    for (decoder, B), idx_list in sorted(cells_by_combo.items()):
        n = len(idx_list)
        if n == 0:
            continue
        # AATPS per condition
        aatps_per_cond = {}
        tr_per_cond = {}
        u_per_cond = {}
        for lab in labels:
            vals_a = [indexed[(decoder, B, i)][lab].get("AATPS") for i in idx_list]
            vals_a = [v for v in vals_a if v == v]
            aatps_per_cond[lab] = statistics.mean(vals_a) if vals_a else float("nan")
            vals_tr = [indexed[(decoder, B, i)][lab].get("token_rate") for i in idx_list]
            vals_tr = [v for v in vals_tr if v == v]
            tr_per_cond[lab] = statistics.mean(vals_tr) if vals_tr else float("nan")
            vals_u = [indexed[(decoder, B, i)][lab].get("ANLPPT_U") for i in idx_list]
            vals_u = [v for v in vals_u if v is not None and v == v]
            u_per_cond[lab] = statistics.mean(vals_u) if vals_u else float("nan")

        # output identity rate
        identical = 0
        for i in idx_list:
            ids_per_lab = []
            ok = True
            for lab in labels:
                ids = indexed[(decoder, B, i)][lab].get("output_ids")
                if ids is None:
                    ok = False
                    break
                ids_per_lab.append(tuple(ids))
            if not ok:
                continue
            if all(s == ids_per_lab[0] for s in ids_per_lab):
                identical += 1
        identical_rate = identical / n if n else 0.0

        # U std per prompt across conditions, averaged
        u_stds = []
        for i in idx_list:
            us = []
            for lab in labels:
                u = indexed[(decoder, B, i)][lab].get("ANLPPT_U")
                if u is not None and u == u:
                    us.append(u)
            if len(us) == len(labels) and len(us) >= 2:
                u_stds.append(statistics.stdev(us))
        u_std_avg = statistics.mean(u_stds) if u_stds else float("nan")

        print(f"{decoder:<28}{B:>3} {n:>5} "
              + " ".join(f"{aatps_per_cond[lab]:>13.3f}" for lab in labels)
              + f" {identical_rate:>14.1%} {u_std_avg:>10.4f}")

    # First few divergent cases for sanity
    print()
    print("# Divergence sanity check (first 5 cells with non-identical outputs):")
    found = 0
    for (decoder, B), idx_list in sorted(cells_by_combo.items()):
        for i in idx_list:
            ids_per_lab = {}
            for lab in labels:
                ids = indexed[(decoder, B, i)][lab].get("output_ids")
                if ids is not None:
                    ids_per_lab[lab] = tuple(ids)
            if len(set(ids_per_lab.values())) > 1:
                first = min((len(s) for s in ids_per_lab.values()))
                # find first divergence position
                arrs = list(ids_per_lab.values())
                pos = next((p for p in range(first)
                            if not all(a[p] == arrs[0][p] for a in arrs)),
                           first)
                print(f"  decoder={decoder} B={B} idx={i}: divergence at token pos {pos} "
                      f"(lengths={[len(s) for s in ids_per_lab.values()]})")
                found += 1
                if found >= 5:
                    break
        if found >= 5:
            break
    if found == 0:
        print("  (no divergent cells found — outputs byte-identical across all conditions)")


if __name__ == "__main__":
    main()

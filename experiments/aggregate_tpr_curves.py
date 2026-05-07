"""Aggregate per-row TPR-at-T into per-cell curves for figure 2 plotters.

Reads raw experiment JSONs from
    outputs/single/   (run_single_draft.py outputs)
    outputs/multi/    (run_multi_draft.py outputs)
and writes a single aggregated curves JSON whose schema is consumed by
    experiments/plot_tpr_comparison_4cells.py
    experiments/plot_drafter_invariance_v2.py  (left panel only)

Curves emitted per cell (model x dataset):

    "Basic UWM"          : decoder=basic_uwm                   (single-draft)
    "MC-UWM (speed)"     : decoder=mc_uwm_speed                (single-draft)
    "MC-UWM (strength)"  : decoder=mc_uwm_strength             (single-draft)
    "MC-UWM (pseudo-r)"  : decoder=mc_uwm_pseudo_r             (single-draft)
                           If a rerun_pseudo_r_*_n1000.json file with
                           dual_Us_pk / dual_Us_mc / dual_r row fields is
                           available, we instead run the dual-key
                           trained-threshold detector (50/50 split, 10
                           trials, 101-step grid over t in [0,1]) and emit
                           that curve.
    "PFR (ours)"         : decoder=pfr                          (single-draft)
    "MPFR (ours)"        : decoder=mpfr_torchgen_cached, mean across B
                                                                (multi-draft)
    "No watermark (H_0)" : pooled rows from decoder in
                           {mc, pfr_no_watermark}              (single-draft)

Cell key (qwen_cnn / qwen_eli5 / vicuna_cnn / vicuna_eli5) is inferred
from `config.target_model` + `config.dataset` per file.

Usage:
    python -m experiments.aggregate_tpr_curves \\
        --single-dir outputs/single \\
        --multi-dir  outputs/multi \\
        --out outputs/single/tpr_curves.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import gammainccinv

T_GRID = [8, 16, 24, 32, 48, 64, 96, 128]
ALPHA_DEFAULT = 0.01


def cell_key(cfg):
    """Map (target_model, dataset) -> short cell key."""
    target = (cfg.get("target_model") or "").lower()
    if "qwen" in target:
        model_short = "qwen"
        model_full = "Qwen2.5-7B-Instruct" if "qwen2.5-7b" in target else (
                     "Qwen2.5-32B-AWQ" if "32b" in target else "Qwen")
    elif "vicuna" in target:
        model_short = "vicuna"
        model_full = "Vicuna-7b-v1.5"
    else:
        model_short = "unk"
        model_full = cfg.get("target_model", "?")
    ds_raw = (cfg.get("dataset") or "").lower()
    ds_short = {"cnn_dailymail": "cnn", "eli5": "eli5"}.get(ds_raw, ds_raw or "unk")
    ds_full = {"cnn_dailymail": "CNN/DailyMail", "eli5": "ELi5"}.get(
        ds_raw, ds_raw or "?")
    return f"{model_short}_{ds_short}", model_full, ds_full


def mean_det_at_T(rows, T):
    """Mean of det_aaronson_at_<T> across rows (skipping missing values)."""
    vals = [r[f"det_aaronson_at_{T}"]
            for r in rows if f"det_aaronson_at_{T}" in r and
            r[f"det_aaronson_at_{T}"] is not None]
    return float(np.mean(vals)) if vals else None


def curve_for_decoder(rows, decoder):
    sub = [r for r in rows if r.get("decoder") == decoder]
    if not sub:
        return None
    return [mean_det_at_T(sub, T) for T in T_GRID]


def pool_h0_curve(rows):
    sub = [r for r in rows
           if r.get("decoder") in ("mc", "pfr_no_watermark")]
    if not sub:
        return None
    return [mean_det_at_T(sub, T) for T in T_GRID]


def mpfr_curve_mean_over_B(rows):
    """multi-draft: average MPFR's det_at_T across all B values present."""
    sub = [r for r in rows if r.get("decoder") == "mpfr_torchgen_cached"]
    if not sub:
        return None
    return [mean_det_at_T(sub, T) for T in T_GRID]


# --- pseudo-r dual-key trained-threshold post-hoc detector ----------------

def _detect_per_prompt(Us_list, lens, T, fpr):
    n = len(Us_list)
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        L = int(min(lens[i], T))
        if L <= 0:
            continue
        u = np.clip(Us_list[i][:L], 1e-10, 1 - 1e-10)
        score = float(-np.log1p(-u).sum())
        thr = float(gammainccinv(L, fpr))
        out[i] = 1 if score >= thr else 0
    return out


def _mix(pk, mc, r, t):
    return np.where(r > t, mc, pk)


def pseudo_r_trained_curve(rows, n_trials=10, grid_size=101, alpha=0.01,
                            seed=42):
    rs = [r for r in rows if r.get("decoder") == "mc_uwm_pseudo_r"
          and r.get("dual_Us_pk") and r.get("dual_Us_mc") and r.get("dual_r")]
    if not rs:
        return None
    pk_list = [np.asarray(r["dual_Us_pk"], np.float64) for r in rs]
    mc_list = [np.asarray(r["dual_Us_mc"], np.float64) for r in rs]
    r_list  = [np.asarray(r["dual_r"],     np.float64) for r in rs]
    lens = np.asarray([len(u) for u in pk_list])
    n = len(rs)
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, grid_size)
    out = []
    for T in T_GRID:
        trial_tpr = []
        for _ in range(n_trials):
            idx = np.arange(n); rng.shuffle(idx)
            half = n // 2
            tr_idx, te_idx = idx[:half], idx[half:]
            best_t, best_train = grid[0], -1.0
            for t in grid:
                mix_tr = [_mix(pk_list[i], mc_list[i], r_list[i], t) for i in tr_idx]
                d_tr = _detect_per_prompt(mix_tr, lens[tr_idx], T, alpha)
                tpr = float(d_tr.mean())
                if tpr > best_train:
                    best_train, best_t = tpr, float(t)
            mix_te = [_mix(pk_list[i], mc_list[i], r_list[i], best_t) for i in te_idx]
            d_te = _detect_per_prompt(mix_te, lens[te_idx], T, alpha)
            trial_tpr.append(float(d_te.mean()))
        out.append(float(np.mean(trial_tpr)))
    return out


# --- main aggregation -----------------------------------------------------

def aggregate(single_dir: Path, multi_dir: Path, alpha: float):
    cells = {}

    # Single-draft files (one or more per cell)
    by_cell_single = defaultdict(list)   # cell_key -> [json_dict, ...]
    if single_dir.is_dir():
        for p in sorted(single_dir.glob("*.json")):
            j = json.loads(p.read_text(encoding="utf-8"))
            cfg = j.get("config", {})
            ck, model_full, ds_full = cell_key(cfg)
            by_cell_single[ck].append((j, model_full, ds_full, p))

    # Multi-draft files (one or more per cell)
    by_cell_multi = defaultdict(list)
    if multi_dir.is_dir():
        for p in sorted(multi_dir.glob("*.json")):
            j = json.loads(p.read_text(encoding="utf-8"))
            cfg = j.get("config", {})
            ck, model_full, ds_full = cell_key(cfg)
            by_cell_multi[ck].append((j, model_full, ds_full, p))

    all_keys = set(by_cell_single) | set(by_cell_multi)

    for ck in sorted(all_keys):
        # Pool rows from all single-draft files for this cell.
        single_rows = []
        single_pseudo_r_rows = []   # subset with dual-key fields
        model_full = ds_full = "?"
        for j, m, d, _ in by_cell_single.get(ck, []):
            single_rows.extend(j.get("rows", []))
            single_pseudo_r_rows.extend([
                r for r in j.get("rows", [])
                if r.get("decoder") == "mc_uwm_pseudo_r"
                and r.get("dual_Us_pk") and r.get("dual_Us_mc") and r.get("dual_r")
            ])
            model_full = m if model_full == "?" else model_full
            ds_full = d if ds_full == "?" else ds_full

        multi_rows = []
        for j, m, d, _ in by_cell_multi.get(ck, []):
            multi_rows.extend(j.get("rows", []))
            model_full = m if model_full == "?" else model_full
            ds_full = d if ds_full == "?" else ds_full

        curves = {}

        # decoders from single-draft
        named = [
            ("Basic UWM",         "basic_uwm"),
            ("MC-UWM (speed)",    "mc_uwm_speed"),
            ("MC-UWM (strength)", "mc_uwm_strength"),
            ("PFR (ours)",        "pfr"),
        ]
        for label, dec in named:
            c = curve_for_decoder(single_rows, dec)
            if c is not None:
                curves[label] = c

        # MC-UWM (pseudo-r): prefer dual-key trained-threshold if rows
        # carry dual_Us_pk/dual_Us_mc/dual_r; otherwise fall back to
        # single-key Aaronson on the regular pseudo_r row.
        if single_pseudo_r_rows:
            c = pseudo_r_trained_curve(single_pseudo_r_rows, alpha=alpha)
            if c is not None:
                curves["MC-UWM (pseudo-r)"] = c
        else:
            c = curve_for_decoder(single_rows, "mc_uwm_pseudo_r")
            if c is not None:
                curves["MC-UWM (pseudo-r)"] = c

        # MPFR from multi-draft
        c = mpfr_curve_mean_over_B(multi_rows)
        if c is not None:
            curves["MPFR (ours)"] = c

        # H0 pooled
        c = pool_h0_curve(single_rows)
        if c is not None:
            curves["No watermark (H_0)"] = c

        cells[ck] = {
            "model": model_full,
            "dataset": ds_full,
            "curves": curves,
        }

    return {
        "alpha": alpha,
        "T_grid": T_GRID,
        "detector": "Aaronson Gamma-tail",
        "pseudo_r_methodology": (
            "dual-key trained threshold (50/50 split, 10 trials, 101-step "
            "grid over t in [0,1]); falls back to single-key Aaronson if "
            "no rerun_pseudo_r row data is present."),
        "h0_pooling": (
            "mc + pfr_no_watermark rows pooled into single 'No watermark "
            "(H_0)' curve."),
        "cells": cells,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-dir", type=Path, default=Path("outputs/single"))
    ap.add_argument("--multi-dir",  type=Path, default=Path("outputs/multi"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    args = ap.parse_args()

    out = aggregate(args.single_dir, args.multi_dir, args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    n_cells = len(out["cells"])
    print(f"wrote {args.out}  ({n_cells} cell{'s' if n_cells != 1 else ''})")
    for ck, cell in out["cells"].items():
        labels = list(cell["curves"].keys())
        print(f"  {ck:<14s}  {cell['model']} x {cell['dataset']}  "
              f"({len(labels)} curves: {labels})")


if __name__ == "__main__":
    main()

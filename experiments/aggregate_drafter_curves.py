"""Aggregate per-row TPR-at-T per drafter condition into the JSON consumed
by experiments/plot_drafter_invariance_v2.py (right-side composite panel).

Inputs are 4 single-draft experiment JSONs, one per drafter condition
D0/D1/D2/D3, all on the same target/dataset.  Convention used by the
ablation_n1000/exp2_drafter_inv_*.json configs:

    D0 :  default drafter   (e.g.  Qwen2.5-0.5B-Instruct, T=1.0)
    D1 :  model swap        (e.g.  Qwen2.5-1.5B-Instruct, T=1.0)
    D2 :  sharper drafter   (drafter T=0.5)
    D3 :  more diffuse      (drafter T=1.5)

Per drafter, we collect TPR vs detection-window T_eval for the four
decoders shown in the figure --- pfr, mc_uwm_strength, mc_uwm_speed,
mc_uwm_pseudo_r --- using the per-row `det_aaronson_at_<T>` fields
emitted by run_single_draft.py.  Mean across rows.

Output schema (matches data/0506 ablation_n500_qwen_cnn/tpr_vs_T_per_drafter.json):

    {
      "alpha":   0.01,
      "T_grid":  [<sorted T values present in row data>],
      "detector": "Aaronson Gamma-tail",
      "n_samples_per_cell": <int>,
      "target_model": <str>,
      "dataset": <str>,
      "drafters": {
        "D0": {"label": "default", "drafter_model": ..., "drafter_temperature": 1.0,
               "curves": {"PFR (ours)": [...], "MWS": [...], "MSE": [...], "mse_pseudo": [...]}},
        "D1": {...}, "D2": {...}, "D3": {...}
      }
    }

Usage:

    python -m experiments.aggregate_drafter_curves \\
        --inputs path/D0.json path/D1.json path/D2.json path/D3.json \\
        --out outputs/ablation_n1000/tpr_vs_T_per_drafter.json

or, with auto-detect by filename pattern (`*_D{0,1,2,3}.json`):

    python -m experiments.aggregate_drafter_curves \\
        --input-dir outputs/ablation_n1000 \\
        --out outputs/ablation_n1000/tpr_vs_T_per_drafter.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Map from internal decoder names to the labels used in the figure JSON.
DECODER_LABELS = {
    "pfr":              "PFR (ours)",
    "mc_uwm_strength":  "MWS",
    "mc_uwm_speed":     "MSE",
    "mc_uwm_pseudo_r":  "mse_pseudo",
}

# Drafter slot labels (kept in sync with the existing ablation_n1000 configs
# and with the DRAFTERS list in plot_drafter_invariance_v2.py).
SLOT_LABEL = {
    "D0": "default",
    "D1": "model swap",
    "D2": "T=0.5",
    "D3": "T=1.5",
}

T_FIELD_RE = re.compile(r"^det_aaronson_at_(\d+)$")
SLOT_RE    = re.compile(r"_D([0-3])(?:[._]|$)")


def detect_slot(path: Path) -> str:
    """Return D0/D1/D2/D3 by matching `_D<n>` somewhere in the filename."""
    m = SLOT_RE.search(path.stem)
    if m is None:
        raise ValueError(
            f"Cannot infer drafter slot (D0/D1/D2/D3) from filename: {path.name}")
    return f"D{m.group(1)}"


def detect_T_grid(rows) -> list[int]:
    """Scan one row for `det_aaronson_at_<T>` fields and return sorted T list."""
    Ts = set()
    for r in rows[:1]:
        for k in r:
            m = T_FIELD_RE.match(k)
            if m:
                Ts.add(int(m.group(1)))
    return sorted(Ts)


def per_decoder_curve(rows, decoder, T_grid):
    sub = [r for r in rows if r.get("decoder") == decoder]
    if not sub:
        return None
    out = []
    for T in T_grid:
        key = f"det_aaronson_at_{T}"
        vals = [r[key] for r in sub if key in r and r[key] is not None]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def short_model_name(target):
    name = str(target or "")
    # Strip path components down to the basename.
    return Path(name).name or name


def aggregate(input_paths, alpha=0.01):
    by_slot = {}
    target_model = dataset = None
    n_per_cell = None
    T_grid = None

    for p in input_paths:
        slot = detect_slot(p)
        if slot in by_slot:
            raise ValueError(f"Duplicate drafter slot {slot} in inputs: {p}")
        j = json.loads(p.read_text(encoding="utf-8"))
        cfg = j.get("config", {})
        rows = j.get("rows", [])
        if not rows:
            raise ValueError(f"{p}: no rows")

        if T_grid is None:
            T_grid = detect_T_grid(rows)
        if not T_grid:
            raise ValueError(f"{p}: no det_aaronson_at_<T> fields found in rows")

        # Cell-level metadata (same across drafters; pulled from D0 first).
        target_model = target_model or short_model_name(cfg.get("target_model"))
        dataset      = dataset      or {"cnn_dailymail": "CNN/DailyMail",
                                        "eli5":          "ELi5"}.get(
                                            cfg.get("dataset"), cfg.get("dataset"))
        n_per_cell   = n_per_cell   or cfg.get("samples")

        # Drafter-level metadata.  Drafter temperature is `draft_temperature`
        # if present (D2/D3 use a sharper / more diffuse drafter), else falls
        # back to the shared `temperature`.
        drafter_model = short_model_name(cfg.get("draft_model"))
        plk = cfg.get("process_logits") or {}
        drafter_temperature = float(
            plk.get("draft_temperature", plk.get("temperature", 1.0)))

        curves = {}
        for dec_name, label in DECODER_LABELS.items():
            c = per_decoder_curve(rows, dec_name, T_grid)
            if c is not None:
                curves[label] = c

        by_slot[slot] = {
            "label": SLOT_LABEL.get(slot, slot),
            "drafter_model": drafter_model,
            "drafter_temperature": drafter_temperature,
            "curves": curves,
        }

    return {
        "alpha": alpha,
        "T_grid": T_grid,
        "detector": "Aaronson Gamma-tail",
        "n_samples_per_cell": n_per_cell,
        "target_model": target_model,
        "dataset": dataset,
        "drafters": {k: by_slot[k] for k in sorted(by_slot)},
    }


def collect_inputs_from_dir(d: Path) -> list[Path]:
    paths = []
    for p in sorted(d.glob("*.json")):
        if SLOT_RE.search(p.stem):
            paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inputs", type=Path, nargs="+",
                   help="Explicit list of D0..D3 JSON paths.")
    g.add_argument("--input-dir", type=Path,
                   help="Directory containing 4 ablation JSONs whose "
                        "filenames contain _D0/_D1/_D2/_D3.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Path to write the aggregated tpr_vs_T_per_drafter.json")
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()

    if args.inputs:
        inputs = list(args.inputs)
    else:
        inputs = collect_inputs_from_dir(args.input_dir)
    if len(inputs) < 1:
        raise SystemExit("no inputs found")
    if len(inputs) != 4:
        print(f"[warn] expected 4 inputs (D0..D3), got {len(inputs)}; "
              "some panels will be missing")

    out = aggregate(inputs, alpha=args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    n_drafters = len(out["drafters"])
    print(f"wrote {args.out}  ({n_drafters} drafter slot(s))")
    for slot, d in out["drafters"].items():
        labels = list(d["curves"].keys())
        print(f"  {slot:<3s}  {d['label']:<14s}  "
              f"drafter={d['drafter_model']:<25s}  "
              f"T_eval={d['drafter_temperature']:.2f}  "
              f"({len(labels)} curves)")


if __name__ == "__main__":
    main()

"""Ablation smoke results parser.

Three sections:
  (1) Exp 1 quality table (single + multi)
  (2) Exp 2 drafter-invariance TPR table (D0..D3, spread per decoder)
  (3) Pairwise ROUGE-L across drafter conditions
"""
from __future__ import annotations
import json, math, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("outputs/ablation_smoke")


def load(p):
    return json.load(open(p))


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    return float(statistics.mean(xs)) if xs else float("nan")


def collect(rows, key):
    return [r.get(key) for r in rows]


def emp_quantile(xs, q):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    if not xs:
        return float("nan")
    xs = sorted(xs)
    n = len(xs)
    k = max(0, min(n - 1, int(round(q * (n - 1)))))
    return xs[k]


def tpr_at_fpr(h0_scores, h1_scores, fpr=0.01):
    """Empirical TPR at given FPR using H0 quantile threshold."""
    h0 = [x for x in h0_scores if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    h1 = [x for x in h1_scores if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    if not h0 or not h1:
        return float("nan"), float("nan")
    tau = emp_quantile(h0, 1 - fpr)
    tpr = sum(1 for x in h1 if x > tau) / len(h1)
    return tpr, tau


def section_exp1():
    print("=" * 110)
    print("EXP 1 QUALITY  (n=100, target=Qwen-7B, drafter=Qwen-0.5B, T=1.0, L=4)")
    print("=" * 110)
    hdr = (f"{'decoder':<24} {'LPPL':>8} {'ROUGE_L':>8} {'AATPS':>8} {'tok/s':>7} "
           f"{'ANL_U':>8} {'DG_64_U':>9} {'Aar_64':>9}")
    print(hdr)
    print("-" * len(hdr))
    files = [
        ("exp1_quality_qwen_cnn_n100.json",
         ["mc", "pfr_no_watermark", "basic_uwm", "mc_uwm_speed",
          "mc_uwm_strength", "mc_uwm_pseudo_r", "pfr"]),
        ("exp1_quality_qwen_cnn_n100_multi.json",
         ["mpfr_torchgen_cached", "invariant_multi"]),
    ]
    for fn, order in files:
        d = load(ROOT / fn)
        by_dec = defaultdict(list)
        for r in d["rows"]:
            by_dec[r["decoder"]].append(r)
        for dec in order:
            rs = by_dec.get(dec, [])
            if not rs:
                print(f"{dec:<24} (missing)")
                continue
            ln = (f"{dec:<24}"
                  f" {mean(collect(rs, 'LPPL')):>8.4f}"
                  f" {mean(collect(rs, 'ROUGE_L_vs_ref')):>8.4f}"
                  f" {mean(collect(rs, 'AATPS')):>8.4f}"
                  f" {mean(collect(rs, 'token_rate')):>7.2f}"
                  f" {mean(collect(rs, 'ANLPPT_U')):>8.5f}"
                  f" {mean(collect(rs, 'det_at_64_U')):>9.4f}"
                  f" {mean(collect(rs, 'det_aaronson_at_64')):>9.4f}")
            print(ln)
    print()


def section_exp2():
    print("=" * 110)
    print("EXP 2 DRAFTER-INVARIANCE  (TPR @ FPR=1%, T_eval=64 tokens)")
    print("D0=0.5B@1.0  D1=1.5B@1.0  D2=0.5B@T0.5  D3=0.5B@T1.5")
    print("=" * 110)
    files = {
        "D0": "exp2_drafter_inv_qwen_cnn_n100_D0.json",
        "D1": "exp2_drafter_inv_qwen_cnn_n100_D1.json",
        "D2": "exp2_drafter_inv_qwen_cnn_n100_D2.json",
        "D3": "exp2_drafter_inv_qwen_cnn_n100_D3.json",
    }
    drafters = ["D0", "D1", "D2", "D3"]
    rows_by_drafter = {}
    for k, fn in files.items():
        d = load(ROOT / fn)
        by_dec = defaultdict(list)
        for r in d["rows"]:
            by_dec[r["decoder"]].append(r)
        rows_by_drafter[k] = by_dec

    decoders = ["mc", "pfr_no_watermark", "basic_uwm", "mc_uwm_speed",
                "mc_uwm_strength", "mc_uwm_pseudo_r", "pfr"]

    # Two reports: mean(det_aaronson_at_64) and TPR @ FPR=1%
    # First report: mean Aaronson Gamma-tail score (raw watermark signal).
    # Higher means stronger watermark presence.
    print("\n[A] mean(det_aaronson_at_64) per (decoder, drafter):")
    hdr = (f"{'decoder':<24}" + "".join(f"{d:>10}" for d in drafters)
           + f"  {'spread (max-min)':>18}")
    print(hdr); print("-" * len(hdr))
    for dec in decoders:
        vals = []
        for k in drafters:
            rs = rows_by_drafter[k].get(dec, [])
            v = mean(collect(rs, "det_aaronson_at_64")) if rs else float("nan")
            vals.append(v)
        finite = [v for v in vals if not math.isnan(v)]
        spread = (max(finite) - min(finite)) if len(finite) >= 2 else float("nan")
        ln = f"{dec:<24}" + "".join(f"{v:>10.4f}" for v in vals)
        ln += f"  {spread:>18.4f}"
        print(ln)

    # Second report: empirical TPR @ FPR=1% per (decoder, drafter)
    # H0 = pfr_no_watermark scores from each drafter cell
    # H1 = the decoder's scores in that same drafter cell
    print("\n[B] TPR @ FPR=1% (H0=pfr_no_watermark same-cell, H1=this decoder same-cell):")
    print("    Aaronson Gamma-tail variant (det_aaronson_at_64)")
    hdr = (f"{'decoder':<24}" + "".join(f"{d:>10}" for d in drafters)
           + f"  {'spread (pp)':>14}")
    print(hdr); print("-" * len(hdr))
    for dec in decoders:
        vals = []
        for k in drafters:
            h0_rs = rows_by_drafter[k].get("pfr_no_watermark", [])
            h1_rs = rows_by_drafter[k].get(dec, [])
            h0 = collect(h0_rs, "det_aaronson_at_64")
            h1 = collect(h1_rs, "det_aaronson_at_64")
            tpr, _ = tpr_at_fpr(h0, h1, fpr=0.01)
            vals.append(tpr)
        finite = [v for v in vals if not math.isnan(v)]
        spread = (max(finite) - min(finite)) * 100 if len(finite) >= 2 else float("nan")
        ln = f"{dec:<24}" + "".join(f"{v:>10.4f}" for v in vals)
        ln += f"  {spread:>14.2f}"
        print(ln)

    print("\n[C] TPR @ FPR=1% via DG U-pivot (det_at_64_U):")
    hdr = (f"{'decoder':<24}" + "".join(f"{d:>10}" for d in drafters)
           + f"  {'spread (pp)':>14}")
    print(hdr); print("-" * len(hdr))
    for dec in decoders:
        vals = []
        for k in drafters:
            h0_rs = rows_by_drafter[k].get("pfr_no_watermark", [])
            h1_rs = rows_by_drafter[k].get(dec, [])
            h0 = collect(h0_rs, "det_at_64_U")
            h1 = collect(h1_rs, "det_at_64_U")
            tpr, _ = tpr_at_fpr(h0, h1, fpr=0.01)
            vals.append(tpr)
        finite = [v for v in vals if not math.isnan(v)]
        spread = (max(finite) - min(finite)) * 100 if len(finite) >= 2 else float("nan")
        ln = f"{dec:<24}" + "".join(f"{v:>10.4f}" for v in vals)
        ln += f"  {spread:>14.2f}"
        print(ln)
    print()


if __name__ == "__main__":
    section_exp1()
    section_exp2()

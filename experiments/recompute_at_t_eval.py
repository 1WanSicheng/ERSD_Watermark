"""Recompute Aaronson Gamma-tail TPR @ FPR=1% at T_eval=128 from u_per_token.

The smoke configs evaluated only at T_eval=64. Each row has u_per_token /
skipped_per_token saved; we recompute the score and detection at T_eval=128
using the same Erlang gamma-tail formula run_single_draft uses at runtime.
"""
from __future__ import annotations
import json, math, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("outputs/ablation_smoke")


def gamma_tail_log_p(n: int, score: float) -> float:
    """log P(Gamma(n, 1) >= score) via scipy gammaincc."""
    if n <= 0 or score <= 0 or not math.isfinite(score):
        return 0.0
    from scipy.special import gammaincc
    p = gammaincc(n, score)
    return math.log(p) if p > 0 else float("-inf")


def aaronson_at_T(u_per_token, skipped_per_token, T):
    if u_per_token is None:
        return float("nan"), float("nan"), 0
    u = np.asarray(u_per_token[:T], dtype=float)
    if skipped_per_token is not None:
        sk = np.asarray(skipped_per_token[:T], dtype=bool)
        u = u[~sk]
    n_eff = int(u.size)
    if n_eff <= 0:
        return float("nan"), float("nan"), 0
    u_clip = np.clip(u, 0.0, 1.0 - 1e-10)
    score = float(np.sum(-np.log1p(-u_clip)))
    log_p = gamma_tail_log_p(n_eff, score)
    return score, log_p, n_eff


def emp_quantile(xs, q):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def tpr_at_fpr(h0, h1, fpr=0.01):
    h0 = [x for x in h0 if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    h1 = [x for x in h1 if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    if not h0 or not h1:
        return float("nan")
    tau = emp_quantile(h0, 1 - fpr)
    return sum(1 for x in h1 if x > tau) / len(h1)


def main():
    files = {
        "D0": "exp2_drafter_inv_qwen_cnn_n100_D0.json",
        "D1": "exp2_drafter_inv_qwen_cnn_n100_D1.json",
        "D2": "exp2_drafter_inv_qwen_cnn_n100_D2.json",
        "D3": "exp2_drafter_inv_qwen_cnn_n100_D3.json",
    }
    drafters = ["D0", "D1", "D2", "D3"]
    decoders = ["mc", "pfr_no_watermark", "basic_uwm", "mc_uwm_speed",
                "mc_uwm_strength", "mc_uwm_pseudo_r", "pfr"]

    log_p_thresh = math.log(0.01)  # FPR=1% via gamma-tail p-value

    # Build {drafter: {decoder: [rows...]}} with recomputed scores
    rows_by_drafter = {}
    for k, fn in files.items():
        d = json.load(open(ROOT / fn))
        by_dec = defaultdict(list)
        for r in d["rows"]:
            score, log_p, n_eff = aaronson_at_T(
                r.get("u_per_token"), r.get("skipped_per_token"), 128,
            )
            r["_score_aaronson_at_128"] = score
            r["_log_p_aaronson_at_128"] = log_p
            r["_det_aaronson_at_128"] = int(log_p <= log_p_thresh) if math.isfinite(log_p) else 0
            r["_n_eff_at_128"] = n_eff
            by_dec[r["decoder"]].append(r)
        rows_by_drafter[k] = by_dec

    print("=" * 110)
    print("EXP 2  TPR @ FPR=1%  AT  T_eval = 128  (recomputed from u_per_token)")
    print("D0=Qwen-0.5B@T1.0  D1=Qwen-1.5B@T1.0  D2=Qwen-0.5B@T0.5  D3=Qwen-0.5B@T1.5")
    print("=" * 110)

    print("\n[A]  mean(det_aaronson_at_128)  (gamma-tail threshold; FPR=1% by p-value)")
    hdr = (f"{'decoder':<24}" + "".join(f"{d:>10}" for d in drafters)
           + f"  {'spread (pp)':>14}")
    print(hdr); print("-" * len(hdr))
    for dec in decoders:
        vals = []
        for k in drafters:
            rs = rows_by_drafter[k].get(dec, [])
            scores = [r.get("_det_aaronson_at_128") for r in rs]
            v = float(statistics.mean(scores)) if scores else float("nan")
            vals.append(v)
        finite = [v for v in vals if not math.isnan(v)]
        spread = (max(finite) - min(finite)) * 100 if len(finite) >= 2 else float("nan")
        ln = f"{dec:<24}" + "".join(f"{v:>10.4f}" for v in vals)
        ln += f"  {spread:>14.2f}"
        print(ln)

    print("\n[B]  TPR @ FPR=1% via empirical H0 quantile  (H0 = pfr_no_watermark same-cell, score=score_aaronson_at_128)")
    hdr = (f"{'decoder':<24}" + "".join(f"{d:>10}" for d in drafters)
           + f"  {'spread (pp)':>14}")
    print(hdr); print("-" * len(hdr))
    for dec in decoders:
        vals = []
        for k in drafters:
            h0_rs = rows_by_drafter[k].get("pfr_no_watermark", [])
            h1_rs = rows_by_drafter[k].get(dec, [])
            h0 = [r.get("_score_aaronson_at_128") for r in h0_rs]
            h1 = [r.get("_score_aaronson_at_128") for r in h1_rs]
            tpr = tpr_at_fpr(h0, h1, fpr=0.01)
            vals.append(tpr)
        finite = [v for v in vals if not math.isnan(v)]
        spread = (max(finite) - min(finite)) * 100 if len(finite) >= 2 else float("nan")
        ln = f"{dec:<24}" + "".join(f"{v:>10.4f}" for v in vals)
        ln += f"  {spread:>14.2f}"
        print(ln)

    # Sanity: report mean n_eff_at_128 per decoder (so we know how many tokens
    # actually got included after skipping repeated-context tokens)
    print("\n[C]  mean(n_eff_at_128) per (decoder, drafter)  (effective tokens scored)")
    hdr = (f"{'decoder':<24}" + "".join(f"{d:>10}" for d in drafters))
    print(hdr); print("-" * len(hdr))
    for dec in decoders:
        ln = f"{dec:<24}"
        for k in drafters:
            rs = rows_by_drafter[k].get(dec, [])
            vs = [r.get("_n_eff_at_128", 0) for r in rs]
            v = float(statistics.mean(vs)) if vs else float("nan")
            ln += f"{v:>10.2f}"
        print(ln)


if __name__ == "__main__":
    main()

"""Side-by-side comparison: WITHIN_BLOCK_PATCH baseline vs B1B2B3 patch."""
import json
from pathlib import Path

OUT = Path("/root/PFR/ERSD_Watermark/outputs")
new = json.load(open(OUT / "mpfr_bench_cnn_n100_L4_B2-4-6-8_B1B2B3.json"))
old = json.load(open(OUT / "mpfr_bench_cnn_n100_L4_B2-4-6-8_WITHIN_BLOCK_PATCH.json"))

methods = ["ms_pfr_batched_cached", "mpfr_batched_torchgen_cached", "invariant_multi"]
Bs = [2, 4, 6, 8]

print(f"{'method':<32}{'B':>3} {'AATPS_old':>10} {'AATPS_new':>10} {'dA':>7} "
      f"{'TR_old':>8} {'TR_new':>8} {'dTR_pct':>9} {'vs_inv_pct':>11}")
print("-" * 110)
for m in methods:
    for B in Bs:
        k = f"{m}_B{B}"
        a_old = old["summary"][k]["AATPS"]
        a_new = new["summary"][k]["AATPS"]
        t_old = old["summary"][k]["token_rate"]
        t_new = new["summary"][k]["token_rate"]
        d_aatps = a_new - a_old
        d_tr_pct = (t_new - t_old) / t_old * 100
        t_inv = new["summary"][f"invariant_multi_B{B}"]["token_rate"]
        gap = (t_new - t_inv) / t_inv * 100
        print(f"{m:<32}{B:>3} {a_old:>10.3f} {a_new:>10.3f} {d_aatps:>+7.3f} "
              f"{t_old:>8.2f} {t_new:>8.2f} {d_tr_pct:>+8.2f}% {gap:>+10.2f}%")

"""Summarize the 1000-sample multi_draft Qwen run."""
import json
from pathlib import Path
from collections import defaultdict

PATH = Path("/root/PFR/ERSD_Watermark/outputs/multi_draft_qwen_cnn_n1000_L4_B2-4-6-8_B1B2B3.json")
d = json.load(open(PATH))

# The file structure is typically {"args":..., "rows":[{decoder, lookahead, num_drafts, ...}]}
# Let's discover.
print("top-level keys:", list(d.keys()))
print("args:", json.dumps(d.get("args", {}), indent=2)[:400])

rows = d.get("rows") or d.get("results") or d
if isinstance(rows, dict) and "summary" in d:
    print("summary keys:", list(d["summary"].keys())[:5])
    summary = d["summary"]
    print(f"\n{'method × B':<35} {'AATPS':>8} {'TR':>8} {'U':>10} {'Li':>10} {'PL':>10} {'n':>5}")
    print("-" * 100)
    for k, v in summary.items():
        a = v.get("AATPS", float("nan"))
        tr = v.get("token_rate", v.get("TR", float("nan")))
        u = v.get("anlppt_U", v.get("U", float("nan")))
        li = v.get("anlppt_Li", v.get("Li", float("nan")))
        pl = v.get("anlppt_PL", v.get("PL", float("nan")))
        n = v.get("n", v.get("count", "-"))
        print(f"{k:<35} {a:>8.3f} {tr:>8.2f} {u:>10.4f} {li:>10.4f} {pl:>10.4f} {str(n):>5}")
elif isinstance(rows, list):
    # Aggregate per (decoder, B)
    agg = defaultdict(list)
    keys_seen = set()
    for r in rows:
        if isinstance(r, dict):
            keys_seen.update(r.keys())
            B = r.get("num_drafts") or r.get("B") or r.get("batch")
            decoder = r.get("decoder") or r.get("method")
            if B is None or decoder is None:
                continue
            agg[(decoder, B)].append(r)
    print("row keys:", sorted(keys_seen)[:15])
    print(f"\n{'decoder':<25} {'B':>3} {'AATPS':>8} {'TR':>8} {'U':>10} {'Li':>10} {'PL':>10} {'n':>5}")
    print("-" * 95)
    def m(rs, key):
        vs = [r.get(key) for r in rs if r.get(key) is not None]
        vs = [v for v in vs if isinstance(v, (int, float))]
        return sum(vs) / len(vs) if vs else float("nan")
    for (decoder, B) in sorted(agg.keys(), key=lambda x: (x[0], x[1])):
        rs = agg[(decoder, B)]
        print(f"{decoder:<25} {B:>3} {m(rs,'AATPS'):>8.3f} {m(rs,'token_rate'):>8.2f} "
              f"{m(rs,'anlppt_U'):>10.4f} {m(rs,'anlppt_Li'):>10.4f} {m(rs,'anlppt_PL'):>10.4f} "
              f"{len(rs):>5}")

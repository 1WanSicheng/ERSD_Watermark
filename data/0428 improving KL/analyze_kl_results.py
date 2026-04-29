#!/usr/bin/env python3
"""
Analyze KL watermark-strength experiments produced by
my_experiment/kl_watermark_strength_experiment_with_pfr.py.

Typical usage on the server:

  cd ~/MPFR/improving_KL
  python analyze_kl_results.py \
    outputs/qwen_summarization450_len200_K2_pfr.json \
    outputs/qwen_summarization450_len200_K3_pfr.json \
    --outdir analysis_kl_K2_K3

or recursively scan a directory:

  python analyze_kl_results.py outputs --pattern '*qwen_summarization450_len200_K*_pfr.json'

The script writes:
  - kl_summary_long.csv
  - kl_comparison_vs_basic.csv
  - kl_takeaways.md
  - kl_tradeoff_AATPS_vs_KL_ratio.png
  - kl_tradeoff_AATPS_vs_KL_mean.png
  - kl_metrics_by_K.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics as stats
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def is_bad_number(x: Any) -> bool:
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))


def mean(vals: Iterable[float]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and not is_bad_number(v)]
    return sum(xs) / len(xs) if xs else None


def stdev(vals: Iterable[float]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and not is_bad_number(v)]
    if len(xs) < 2:
        return 0.0 if len(xs) == 1 else None
    return stats.stdev(xs)


def infer_K(path: Path, data: dict) -> Optional[int]:
    # Prefer explicit args/config.
    for key_path in [
        ("args", "lookahead"),
        ("config", "lookahead"),
        ("config", "lookaheads"),
    ]:
        cur: Any = data
        ok = True
        for k in key_path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            if isinstance(cur, list) and len(cur) == 1:
                return int(cur[0])
            if isinstance(cur, int):
                return int(cur)
            try:
                return int(cur)
            except Exception:
                pass
    # Fall back to filename patterns.
    name = path.name
    for pat in [r"(?:^|[_-])K(\d+)(?:[_-]|\.)", r"(?:^|[_-])L(\d+)(?:[_-]|\.)", r"lookahead(\d+)"]:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def aggregate_rows(rows: List[dict], method: str) -> Dict[str, Any]:
    sub = [r for r in rows if r.get("method") == method]
    if not sub:
        return {}

    sample_aatps = []
    total_tokens = 0
    total_steps = 0
    elapsed = 0.0

    for r in sub:
        chunks = r.get("chunk_lengths") or []
        if chunks:
            tok = int(sum(chunks))
            steps = len(chunks)
        else:
            tok = int(r.get("num_tokens", r.get("tokens", 0)) or 0)
            # If no chunks, one generation step is not reliable, so leave steps 0.
            steps = int(r.get("blocks", 0) or 0)
        total_tokens += tok
        total_steps += steps
        elapsed += float(r.get("generation_elapsed_sec", r.get("elapsed_sec", 0.0)) or 0.0)
        if steps > 0:
            sample_aatps.append(tok / steps)

    kl_sum = sum(float(r.get("KL_WS_sum", 0.0) or 0.0) for r in sub)
    kl_count = sum(int(r.get("KL_WS_count", 0) or 0) for r in sub)
    entropy_sum = sum(float(r.get("KL_WS_entropy_sum", 0.0) or 0.0) for r in sub)

    vals_kl = [r.get("KL_WS_mean") for r in sub if r.get("KL_WS_mean") is not None]
    vals_ratio = [r.get("KL_WS_ratio") for r in sub if r.get("KL_WS_ratio") is not None]

    out = {
        "num_samples": len(sub),
        "num_tokens": total_tokens,
        "num_steps": total_steps,
        "AATPS": total_tokens / total_steps if total_steps > 0 else mean(r.get("AATPS") for r in sub),
        "AATPS_sample_mean": mean(sample_aatps),
        "AATPS_sample_std": stdev(sample_aatps),
        "token_rate": total_tokens / elapsed if elapsed > 0 else mean(r.get("token_rate") for r in sub),
        "generation_elapsed_sec": elapsed,
        "KL_WS_sum": kl_sum,
        "KL_WS_count": kl_count,
        "KL_WS_entropy_sum": entropy_sum,
        "KL_WS_mean": kl_sum / kl_count if kl_count > 0 else mean(vals_kl),
        "KL_WS_sample_mean": mean(vals_kl),
        "KL_WS_sample_std": stdev(vals_kl),
        "KL_WS_ratio": kl_sum / entropy_sum if entropy_sum > 0 else mean(vals_ratio),
        "KL_WS_sample_ratio_mean": mean(vals_ratio),
        "KL_WS_sample_ratio_std": stdev(vals_ratio),
    }
    # Copy useful sums if present.
    for key in [
        "accepted_count_sum", "attempted_draft_tokens_sum", "target_forward_calls_sum",
        "draft_forward_calls_sum", "draft_tree_size_sum", "target_context_count_sum",
        "KL_WS_kind", "KL_WS_pfr_top_k", "KL_WS_pfr_baseline", "KL_WS_num_keys",
    ]:
        vals = [r.get(key) for r in sub if key in r]
        if vals:
            if isinstance(vals[0], (int, float)):
                out[key] = sum(v for v in vals if isinstance(v, (int, float)))
            else:
                out[key] = vals[0]
    return out


def normalize_result(path: Path, data: dict) -> List[Dict[str, Any]]:
    """Return one summary row per method."""
    K = infer_K(path, data)
    rows = data.get("rows") or []
    summary = data.get("summary") or {}

    methods = set(summary.keys())
    methods.update(r.get("method") for r in rows if "method" in r)
    methods = {m for m in methods if m}

    out_rows: List[Dict[str, Any]] = []
    for method in sorted(methods):
        s = dict(summary.get(method, {})) if isinstance(summary.get(method, {}), dict) else {}
        # If rows exist, recompute robust aggregate and let explicit summary override only if not available.
        if rows:
            agg = aggregate_rows(rows, method)
            for k, v in agg.items():
                if k not in s or s[k] is None or is_bad_number(s[k]):
                    s[k] = v
            # Prefer recomputed core metrics from raw rows because it handles partial oddities.
            for k in [
                "num_samples", "num_tokens", "num_steps", "AATPS", "token_rate",
                "KL_WS_mean", "KL_WS_ratio", "KL_WS_sum", "KL_WS_count", "KL_WS_entropy_sum",
                "AATPS_sample_mean", "AATPS_sample_std", "KL_WS_sample_mean",
                "KL_WS_sample_std", "KL_WS_sample_ratio_mean", "KL_WS_sample_ratio_std",
                "generation_elapsed_sec",
            ]:
                if k in agg:
                    s[k] = agg[k]
        out = {"K": K, "method": method, "source_file": str(path)}
        out.update(s)
        out_rows.append(out)
    return out_rows


def collect_json_files(inputs: List[str], pattern: str) -> Tuple[List[Path], tempfile.TemporaryDirectory | None]:
    files: List[Path] = []
    tmp: tempfile.TemporaryDirectory | None = None

    for raw in inputs:
        p = Path(raw).expanduser()
        if not p.exists():
            continue
        if p.is_dir():
            files.extend(sorted(p.rglob(pattern)))
        elif p.suffix.lower() == ".zip":
            if tmp is None:
                tmp = tempfile.TemporaryDirectory(prefix="kl_results_extract_")
            with zipfile.ZipFile(p) as z:
                z.extractall(tmp.name)
            root = Path(tmp.name)
            files.extend(sorted(root.rglob(pattern)))
        elif p.suffix.lower() == ".json":
            files.append(p)

    # Remove macOS metadata files and duplicates.
    dedup = []
    seen = set()
    for f in files:
        if "__MACOSX" in f.parts or f.name.startswith("._"):
            continue
        key = str(f.resolve())
        if key not in seen:
            dedup.append(f)
            seen.add(key)
    return dedup, tmp


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Stable field order.
    preferred = [
        "K", "method", "num_samples", "num_tokens", "num_steps", "AATPS", "AATPS_sample_std",
        "token_rate", "KL_WS_mean", "KL_WS_sample_std", "KL_WS_ratio", "KL_WS_sample_ratio_std",
        "KL_WS_count", "KL_WS_entropy_sum", "generation_elapsed_sec", "source_file",
    ]
    keys = []
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_comparisons(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_k = defaultdict(dict)
    for r in rows:
        by_k[r.get("K")][r.get("method")] = r

    comps = []
    for K, d in sorted(by_k.items(), key=lambda kv: (kv[0] is None, kv[0])):
        basic = d.get("basic_uwm")
        if not basic:
            continue
        b_aatps = float(basic.get("AATPS") or 0.0)
        b_kl = float(basic.get("KL_WS_mean") or 0.0)
        b_ratio = float(basic.get("KL_WS_ratio") or 0.0)
        for method, r in sorted(d.items()):
            a = r.get("AATPS")
            kl = r.get("KL_WS_mean")
            ratio = r.get("KL_WS_ratio")
            comps.append({
                "K": K,
                "method": method,
                "AATPS": a,
                "KL_WS_mean": kl,
                "KL_WS_ratio": ratio,
                "AATPS_vs_basic_x": (float(a) / b_aatps) if a is not None and b_aatps > 0 else None,
                "KL_mean_vs_basic_x": (float(kl) / b_kl) if kl is not None and b_kl > 0 else None,
                "KL_ratio_vs_basic_x": (float(ratio) / b_ratio) if ratio is not None and b_ratio > 0 else None,
                "token_rate": r.get("token_rate"),
            })
    return comps


def fmt(x: Any, nd: int = 3) -> str:
    if x is None or is_bad_number(x):
        return "NA"
    if isinstance(x, int):
        return str(x)
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def make_markdown(rows: List[Dict[str, Any]], comps: List[Dict[str, Any]], found_files: List[Path]) -> str:
    lines = []
    lines.append("# KL watermark-strength analysis")
    lines.append("")
    lines.append(f"Analyzed {len(found_files)} JSON file(s).")
    if found_files:
        lines.append("")
        lines.append("Files:")
        for f in found_files:
            lines.append(f"- `{f}`")
    lines.append("")

    if not rows:
        lines.append("No compatible KL result JSON files were found.")
        lines.append("")
        lines.append("Expected files look like:")
        lines.append("- `qwen_summarization450_len200_K2_pfr.json`")
        lines.append("- `qwen_summarization450_len200_K3_pfr.json`")
        lines.append("")
        return "\n".join(lines)

    # Summary table.
    lines.append("## Summary table")
    lines.append("")
    lines.append("| K | method | samples | AATPS | token rate | KL mean | KL ratio |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (x.get("K") is None, x.get("K"), x.get("method"))):
        lines.append(
            f"| {fmt(r.get('K'), 0)} | `{r.get('method')}` | {fmt(r.get('num_samples'), 0)} "
            f"| {fmt(r.get('AATPS'))} | {fmt(r.get('token_rate'))} "
            f"| {fmt(r.get('KL_WS_mean'))} | {fmt(r.get('KL_WS_ratio'))} |"
        )

    lines.append("")
    lines.append("## Main takeaways")
    lines.append("")
    by_k = defaultdict(list)
    for r in rows:
        by_k[r.get("K")].append(r)
    for K in sorted(by_k, key=lambda x: (x is None, x)):
        sub = by_k[K]
        valid_kl = [r for r in sub if r.get("KL_WS_ratio") is not None]
        valid_a = [r for r in sub if r.get("AATPS") is not None]
        if valid_kl:
            strongest = max(valid_kl, key=lambda r: float(r.get("KL_WS_ratio")))
            lines.append(f"- For K={K}, strongest by KL ratio: `{strongest['method']}` with KL ratio {fmt(strongest.get('KL_WS_ratio'))}.")
        if valid_a:
            fastest_steps = max(valid_a, key=lambda r: float(r.get("AATPS")))
            lines.append(f"- For K={K}, best AATPS: `{fastest_steps['method']}` with AATPS {fmt(fastest_steps.get('AATPS'))}.")
        d = {r["method"]: r for r in sub}
        if "pfr_uwm" in d and "basic_uwm" in d:
            p, b = d["pfr_uwm"], d["basic_uwm"]
            p_a, b_a = float(p.get("AATPS") or 0), float(b.get("AATPS") or 0)
            p_r, b_r = float(p.get("KL_WS_ratio") or 0), float(b.get("KL_WS_ratio") or 0)
            if b_a > 0 and b_r > 0:
                lines.append(
                    f"- For K={K}, `pfr_uwm` gives {p_a/b_a:.2f}x AATPS of `basic_uwm` "
                    f"and retains {100*p_r/b_r:.1f}% of its KL ratio."
                )
    lines.append("")
    lines.append("Interpretation: higher KL mean/ratio means a stronger white-box watermark signal, while higher AATPS/token rate means better sampling efficiency.")
    return "\n".join(lines)


def make_plots(rows: List[Dict[str, Any]], outdir: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    outdir.mkdir(parents=True, exist_ok=True)
    methods = sorted({r.get("method") for r in rows if r.get("method")})

    # Tradeoff: AATPS vs KL ratio.
    plt.figure(figsize=(7, 5))
    for m in methods:
        sub = sorted([r for r in rows if r.get("method") == m], key=lambda r: r.get("K") or -1)
        xs = [r.get("AATPS") for r in sub]
        ys = [r.get("KL_WS_ratio") for r in sub]
        labels = [r.get("K") for r in sub]
        if any(x is None for x in xs) or any(y is None for y in ys):
            continue
        plt.plot(xs, ys, marker="o", label=m)
        for x, y, k in zip(xs, ys, labels):
            plt.annotate(f"K={k}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("AATPS")
    plt.ylabel("KL_WS_ratio")
    plt.title("Sampling efficiency vs. normalized KL watermark strength")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "kl_tradeoff_AATPS_vs_KL_ratio.png", dpi=200)
    plt.close()

    # Tradeoff: AATPS vs KL mean.
    plt.figure(figsize=(7, 5))
    for m in methods:
        sub = sorted([r for r in rows if r.get("method") == m], key=lambda r: r.get("K") or -1)
        xs = [r.get("AATPS") for r in sub]
        ys = [r.get("KL_WS_mean") for r in sub]
        labels = [r.get("K") for r in sub]
        if any(x is None for x in xs) or any(y is None for y in ys):
            continue
        plt.plot(xs, ys, marker="o", label=m)
        for x, y, k in zip(xs, ys, labels):
            plt.annotate(f"K={k}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("AATPS")
    plt.ylabel("KL_WS_mean")
    plt.title("Sampling efficiency vs. KL watermark strength")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "kl_tradeoff_AATPS_vs_KL_mean.png", dpi=200)
    plt.close()

    # Metrics by K: four simple charts.
    for metric, ylabel, fname in [
        ("AATPS", "AATPS", "AATPS_by_K.png"),
        ("token_rate", "token rate", "token_rate_by_K.png"),
        ("KL_WS_mean", "KL_WS_mean", "KL_mean_by_K.png"),
        ("KL_WS_ratio", "KL_WS_ratio", "KL_ratio_by_K.png"),
    ]:
        plt.figure(figsize=(7, 5))
        for m in methods:
            sub = sorted([r for r in rows if r.get("method") == m and r.get(metric) is not None], key=lambda r: r.get("K") or -1)
            if not sub:
                continue
            plt.plot([r.get("K") for r in sub], [r.get(metric) for r in sub], marker="o", label=m)
        plt.xlabel("lookahead K")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by K")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=200)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="JSON files, directories, or zip archives")
    ap.add_argument("--pattern", default="*.json", help="glob pattern when scanning directories/zips")
    ap.add_argument("--outdir", default="analysis_kl_results", help="output directory")
    ap.add_argument("--require-method", default=None, help="optional method that must appear in a result, e.g. pfr_uwm")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files, tmp = collect_json_files(args.inputs, args.pattern)
    result_rows: List[Dict[str, Any]] = []
    compatible_files = []
    rejected_files = []

    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            rejected_files.append((str(f), f"json load error: {e}"))
            continue
        if not isinstance(data, dict) or "summary" not in data:
            rejected_files.append((str(f), "missing top-level summary"))
            continue
        rows = normalize_result(f, data)
        if args.require_method and args.require_method not in {r.get("method") for r in rows}:
            rejected_files.append((str(f), f"missing method {args.require_method}"))
            continue
        if not rows:
            rejected_files.append((str(f), "no method rows"))
            continue
        compatible_files.append(f)
        result_rows.extend(rows)

    # Keep only likely KL experiment results if many unrelated summary JSONs are found.
    if args.require_method is None:
        kl_like = [r for r in result_rows if r.get("method") in {"basic_uwm", "mc_uwm_strength", "mc_uwm_speed", "pfr_uwm"}]
        if kl_like:
            result_rows = kl_like

    comps = build_comparisons(result_rows)

    write_csv(outdir / "kl_summary_long.csv", result_rows)
    write_csv(outdir / "kl_comparison_vs_basic.csv", comps)
    md = make_markdown(result_rows, comps, compatible_files)
    (outdir / "kl_takeaways.md").write_text(md, encoding="utf-8")
    # Rejected file list for debugging.
    with (outdir / "scan_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Scanned {len(files)} JSON file(s).\n")
        f.write(f"Compatible {len(compatible_files)} file(s).\n\n")
        for x in compatible_files:
            f.write(f"OK: {x}\n")
        f.write("\nRejected:\n")
        for x, why in rejected_files:
            f.write(f"{x}: {why}\n")

    make_plots(result_rows, outdir)

    print(md)
    print(f"\nWrote outputs to: {outdir}")

    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()

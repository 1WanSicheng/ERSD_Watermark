from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import transformers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kl_watermark_strength_experiment import (  # noqa: E402
    aggregate,
    collect_generation,
    encode_prompt,
    estimate_kl,
    load_dataset,
)
from my_experiment.worker import MaxLengthLogitsProcessor  # noqa: E402


def summarize_pairs(rows: list[dict[str, Any]], prefix_source: str) -> dict[str, Any]:
    source_rows = [r for r in rows if r["prefix_source"] == prefix_source]
    by_sample: dict[int, dict[str, dict[str, Any]]] = {}
    for row in source_rows:
        by_sample.setdefault(row["sample_idx"], {})[row["eval_method"]] = row

    pairs = []
    for sample_idx, methods in by_sample.items():
        if "synthid_basic" not in methods or "mc_mws" not in methods:
            continue
        basic = methods["synthid_basic"]
        mws = methods["mc_mws"]
        pairs.append(
            {
                "sample_idx": sample_idx,
                "entropy_mean_basic": basic["KL_WS_entropy_mean"],
                "entropy_mean_mws": mws["KL_WS_entropy_mean"],
                "ratio_basic": basic["KL_WS_ratio"],
                "ratio_mws": mws["KL_WS_ratio"],
                "ratio_gap_mws_minus_basic": mws["KL_WS_ratio"] - basic["KL_WS_ratio"],
                "kl_mean_basic": basic["KL_WS_mean"],
                "kl_mean_mws": mws["KL_WS_mean"],
            }
        )
    if not pairs:
        return {"num_pairs": 0}

    gaps = np.array([p["ratio_gap_mws_minus_basic"] for p in pairs], dtype=np.float64)
    ent_basic = np.array([p["entropy_mean_basic"] for p in pairs], dtype=np.float64)
    ratio_basic = np.array([p["ratio_basic"] for p in pairs], dtype=np.float64)
    ratio_mws = np.array([p["ratio_mws"] for p in pairs], dtype=np.float64)
    corr_basic = float(np.corrcoef(ent_basic, ratio_basic)[0, 1]) if len(pairs) > 1 else 0.0
    corr_mws = float(np.corrcoef(ent_basic, ratio_mws)[0, 1]) if len(pairs) > 1 else 0.0
    return {
        "num_pairs": len(pairs),
        "ratio_gap_mean": float(gaps.mean()),
        "ratio_gap_std": float(gaps.std()),
        "ratio_basic_mean": float(ratio_basic.mean()),
        "ratio_mws_mean": float(ratio_mws.mean()),
        "entropy_basic_mean": float(ent_basic.mean()),
        "corr_entropy_basic_ratio_basic": corr_basic,
        "corr_entropy_basic_ratio_mws": corr_mws,
        "pairs": pairs,
    }


def entropy_bins(rows: list[dict[str, Any]], prefix_source: str, bins: int) -> list[dict[str, Any]]:
    source_rows = [
        r
        for r in rows
        if r["prefix_source"] == prefix_source and r["eval_method"] == "synthid_basic"
    ]
    if not source_rows:
        return []
    ent = np.array([r["KL_WS_entropy_mean"] for r in source_rows], dtype=np.float64)
    edges = np.quantile(ent, np.linspace(0.0, 1.0, bins + 1))
    out = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            selected = [r for r in source_rows if lo <= r["KL_WS_entropy_mean"] <= hi]
        else:
            selected = [r for r in source_rows if lo <= r["KL_WS_entropy_mean"] < hi]
        if not selected:
            continue
        sample_ids = {r["sample_idx"] for r in selected}
        paired = [
            r
            for r in rows
            if r["prefix_source"] == prefix_source and r["sample_idx"] in sample_ids
        ]
        basic = aggregate(
            [{**r, "method": r["eval_method"]} for r in paired],
            "synthid_basic",
        )
        mws = aggregate(
            [{**r, "method": r["eval_method"]} for r in paired],
            "mc_mws",
        )
        out.append(
            {
                "bin": i,
                "entropy_range": [float(lo), float(hi)],
                "num_samples": len(sample_ids),
                "synthid_basic_ratio": basic["KL_WS_ratio"],
                "mc_mws_ratio": mws["KL_WS_ratio"],
                "gap_mws_minus_basic": mws["KL_WS_ratio"] - basic["KL_WS_ratio"],
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="code/real/model/huggyllama__llama-7b")
    parser.add_argument("--ref-model", default="code/real/model/JackFram__llama-68m")
    parser.add_argument("--dataset", default="summarization", choices=["manual", "summarization", "oeg", "eli5"])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--prefix-methods", nargs="+", default=["synthid_basic", "mc_mws"])
    parser.add_argument("--eval-methods", nargs="+", default=["synthid_basic", "mc_mws"])
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kl-num-keys", type=int, default=1)
    parser.add_argument("--kl-key-offset", type=int, default=0)
    parser.add_argument("--private-key", default="1234")
    parser.add_argument("--mc-private-key", default="4321")
    parser.add_argument("--reweight", default="deltagumbel", choices=["deltagumbel", "gamma", "synthid"])
    parser.add_argument("--synthid-private-key", type=int, default=0)
    parser.add_argument("--synthid-mc-private-key", type=int, default=1)
    parser.add_argument("--synthid-seed", type=int, default=42)
    parser.add_argument("--synthid-depth", type=int, default=30)
    parser.add_argument("--entropy-bins", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    load_kwargs = {"device_map": args.device, "low_cpu_mem_usage": True}
    if args.device.startswith("cuda"):
        load_kwargs["torch_dtype"] = torch.float16
    target = transformers.AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    draft = transformers.AutoModelForCausalLM.from_pretrained(args.ref_model, **load_kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    target.eval()
    draft.eval()

    max_length_lp = MaxLengthLogitsProcessor(args.max_length, tokenizer.eos_token_id)
    process_logits_kwargs = {
        "logits_processor": transformers.LogitsProcessorList([max_length_lp]),
    }
    prompts = load_dataset(args.dataset, args.samples, tokenizer=tokenizer)

    rows = []
    start = time.perf_counter()
    for sample_idx, prompt in enumerate(prompts):
        for prefix_source in args.prefix_methods:
            input_ids = encode_prompt(tokenizer, args.model, prompt, target.device)
            max_length_lp.input_length = input_ids.shape[-1]
            input_ids, out_ids, generation_meta = collect_generation(
                prefix_source,
                target,
                draft,
                tokenizer,
                prompt,
                args,
                process_logits_kwargs,
            )
            for eval_method in args.eval_methods:
                kl = estimate_kl(
                    eval_method,
                    target,
                    draft,
                    input_ids,
                    out_ids,
                    args,
                    process_logits_kwargs,
                    generation_meta=generation_meta,
                    kl_key_index=0,
                )
                row = {
                    "sample_idx": sample_idx,
                    "prefix_source": prefix_source,
                    "eval_method": eval_method,
                    "num_tokens": int(out_ids.numel()),
                    **generation_meta,
                    **{
                        k: v
                        for k, v in kl.items()
                        if k not in {"per_token_kl", "per_token_entropy"}
                    },
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))

    summary = {}
    for prefix_source in args.prefix_methods:
        source_rows = [
            {**r, "method": r["eval_method"]}
            for r in rows
            if r["prefix_source"] == prefix_source
        ]
        summary[prefix_source] = {
            method: aggregate(source_rows, method) for method in args.eval_methods
        }
        summary[prefix_source]["paired"] = summarize_pairs(rows, prefix_source)
        summary[prefix_source]["entropy_bins"] = entropy_bins(
            rows, prefix_source, args.entropy_bins
        )

    result = {
        "elapsed_sec": time.perf_counter() - start,
        "args": vars(args),
        "summary": summary,
        "rows": rows,
    }
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

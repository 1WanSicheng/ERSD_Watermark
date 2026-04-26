from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from functools import partial
from typing import Any

import numpy as np
import torch
import transformers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import accuwm
import accuwm.basic_synthid
import accuwm.mc_synthid
import unbiased_watermark as uwm
from experiments import tasks
from my_experiment.worker import MaxLengthLogitsProcessor, StopWordsLogitsProcessor
from unbiased_watermark.scores.kl_watermark_strength import (
    compute_basic_uwm_kl_from_sequence,
    compute_mc_uwm_speed_kl_from_sequence,
    next_token_logits_from_full_sequence,
    synthid_mc_kl_for_contexts,
    synthid_topk_kl_for_contexts,
)
from unbiased_watermark.synthid import SynthID_Reweight_fast


def to_display_model_name(s: str) -> str:
    return s.split("/")[-1] if "/" in s else s


def build_reweight(name: str):
    if name == "deltagumbel":
        return uwm.DeltaGumbel_Reweight()
    if name == "gamma":
        return uwm.Gamma_Reweight()
    if name == "synthid":
        return uwm.SynthID_Reweight()
    raise ValueError(f"unknown reweight: {name}")


def encode_prompt(tokenizer, model_name: str, prompt: str, device) -> torch.LongTensor:
    if "-it" in model_name and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.strip('"')}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


def load_dataset(name: str, samples: int, tokenizer=None) -> list[str]:
    if name == "manual":
        return [
            "Explain why speculative decoding can accelerate language model generation.",
            "Summarize the difference between a forest and a wood.",
        ][:samples]
    if name == "summarization":
        return [row["prompt"] for row in tasks.get_summarization_ds(samples)]
    if name == "oeg":
        return [row["prompt"] for row in tasks.get_oeg_ds(samples)]
    if name == "eli5":
        return list(tasks.get_eli5_ds_dataset(samples))
    raise ValueError(f"unknown dataset: {name}")


@torch.no_grad()
def collect_generation(
    method: str,
    target,
    draft,
    tokenizer,
    prompt: str,
    args,
    process_logits_kwargs: dict[str, Any],
) -> tuple[torch.LongTensor, torch.LongTensor, dict[str, Any]]:
    input_ids = encode_prompt(tokenizer, args.model, prompt, target.device)
    private_key = bytes(args.seed) + args.private_key.encode("utf-8")
    mc_private_key = bytes(args.seed) + args.mc_private_key.encode("utf-8")
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=args.context_width)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    if method == "basic_uwm":
        gen = accuwm.basic_watermark.basic_uwm_generator(
            reweight=build_reweight(args.reweight),
            cc_extractor=cc_extractor,
            cch=cch,
            private_key=private_key,
            model=target,
            input_ids=input_ids,
            n=1,
            temperature=args.temperature,
            process_logits_kwargs=process_logits_kwargs,
        )
    elif method in {"mc_uwm_strength", "mc_uwm_speed", "mc_uwm_synthid_psedo_r"}:
        gen = accuwm.mc_watermark.mc_uwm_sample_generator(
            reweight=build_reweight(args.reweight),
            cc_extractor=cc_extractor,
            cch=cch,
            private_key=private_key,
            reweight_in_mc=(method == "mc_uwm_strength"),
            model=target,
            ref_model=draft,
            input_ids=input_ids,
            n=args.lookahead,
            mc_synthid=(method == "mc_uwm_synthid_psedo_r"),
            mc_private_key=mc_private_key if method == "mc_uwm_synthid_psedo_r" else None,
            psedo_r=(method == "mc_uwm_synthid_psedo_r"),
            temperature=args.temperature,
            process_logits_kwargs=process_logits_kwargs,
        )
    elif method == "synthid_basic":
        synthid = SynthID_Reweight_fast(
            sampling_table_size=2**16,
            sampling_table_seed=args.synthid_seed,
            device=target.device,
            ngram_len=args.context_width,
            private_key=args.synthid_private_key,
            watermarking_depth=args.synthid_depth,
        )
        gen = accuwm.basic_synthid.basic_synthid_generator(
            reweight=synthid,
            cc_extractor=cc_extractor,
            cch=cch,
            model=target,
            input_ids=input_ids,
            n=1,
            temperature=args.temperature,
            top_k=args.top_k,
            process_logits_kwargs=process_logits_kwargs,
        )
    elif method in {"mc_mse", "mc_mws", "mc_2keys"}:
        synthid = SynthID_Reweight_fast(
            sampling_table_size=2**16,
            sampling_table_seed=args.synthid_seed,
            device=target.device,
            ngram_len=args.context_width,
            private_key=args.synthid_private_key,
            watermarking_depth=args.synthid_depth,
        )
        gen = accuwm.mc_synthid.mc_synthid_sample_generator(
            method=method,
            reweight=synthid,
            cc_extractor=cc_extractor,
            cch=cch,
            model=target,
            ref_model=draft,
            input_ids=input_ids,
            n=args.lookahead,
            mc_private_key=args.synthid_mc_private_key,
            psedo_r=True,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.synthid_seed + 1,
            process_logits_kwargs=process_logits_kwargs,
        )
    else:
        raise ValueError(f"unknown method: {method}")

    chunks = []
    chunk_lengths = []
    generated = 0
    for step in gen:
        output_ids = step[0]
        remaining = args.max_length - generated
        if remaining <= 0:
            break
        if output_ids.shape[1] > remaining:
            output_ids = output_ids[:, :remaining]
        chunks.append(output_ids.detach())
        chunk_lengths.append(int(output_ids.shape[1]))
        generated += output_ids.shape[1]
        if generated >= args.max_length:
            break
    if chunks:
        out_ids = torch.cat(chunks, dim=1)
    else:
        out_ids = torch.empty((1, 0), dtype=torch.long, device=input_ids.device)
    return input_ids, out_ids, {"chunk_lengths": chunk_lengths}


@torch.no_grad()
def estimate_kl(
    method: str,
    target,
    draft,
    input_ids: torch.LongTensor,
    out_ids: torch.LongTensor,
    args,
    process_logits_kwargs: dict[str, Any],
    generation_meta: dict[str, Any] | None = None,
    kl_key_index: int = 0,
) -> dict[str, Any]:
    if out_ids.numel() == 0:
        return {"KL_WS_mean": 0.0, "KL_WS_sum": 0.0, "KL_WS_count": 0, "KL_WS_kind": "empty"}
    full_ids = torch.cat([input_ids, out_ids], dim=1)
    prompt_length = input_ids.shape[1]
    p_logits = next_token_logits_from_full_sequence(
        target, full_ids, prompt_length, process_logits_kwargs=process_logits_kwargs
    )
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=args.context_width)
    kl_seed = args.seed + args.kl_key_offset + kl_key_index
    private_key = bytes(kl_seed) + args.private_key.encode("utf-8")
    mc_private_key = bytes(kl_seed) + args.mc_private_key.encode("utf-8")
    if method in {"basic_uwm", "mc_uwm_strength"}:
        baseline_logits = p_logits / args.temperature
        return compute_basic_uwm_kl_from_sequence(
            baseline_logits,
            out_ids,
            input_ids,
            build_reweight(args.reweight),
            cc_extractor,
            private_key,
        )
    if method in {"mc_uwm_speed", "mc_uwm_synthid_psedo_r"}:
        q_logits = next_token_logits_from_full_sequence(
            draft, full_ids.to(draft.device), prompt_length, process_logits_kwargs=process_logits_kwargs
        ).to(p_logits.device)
        return compute_mc_uwm_speed_kl_from_sequence(
            p_logits,
            q_logits,
            out_ids,
            input_ids,
            build_reweight(args.reweight),
            cc_extractor,
            private_key,
            pseudo_r_private_key=mc_private_key if method == "mc_uwm_synthid_psedo_r" else None,
            residual_private_key=private_key if method == "mc_uwm_synthid_psedo_r" else None,
            watermark_temperature=args.temperature,
            baseline_temperature=args.temperature,
        )
    if method == "synthid_basic":
        synthid = SynthID_Reweight_fast(
            sampling_table_size=2**16,
            sampling_table_seed=args.synthid_seed,
            device=target.device,
            ngram_len=args.context_width,
            private_key=args.synthid_private_key + kl_key_index,
            watermarking_depth=args.synthid_depth,
        )
        return synthid_topk_kl_for_contexts(
            p_logits,
            full_ids,
            prompt_length,
            synthid,
            cc_extractor,
            args.temperature,
            args.top_k,
        )
    if method in {"mc_mse", "mc_mws", "mc_2keys"}:
        q_logits = next_token_logits_from_full_sequence(
            draft, full_ids.to(draft.device), prompt_length, process_logits_kwargs=process_logits_kwargs
        ).to(p_logits.device)
        synthid = SynthID_Reweight_fast(
            sampling_table_size=2**16,
            sampling_table_seed=args.synthid_seed,
            device=target.device,
            ngram_len=args.context_width,
            private_key=args.synthid_private_key + kl_key_index,
            watermarking_depth=args.synthid_depth,
        )
        residual_synthid = SynthID_Reweight_fast(
            sampling_table_size=2**16,
            sampling_table_seed=args.synthid_seed,
            device=target.device,
            ngram_len=args.context_width,
            private_key=args.synthid_private_key + kl_key_index,
            watermarking_depth=args.synthid_depth,
        )
        return synthid_mc_kl_for_contexts(
            method,
            p_logits,
            q_logits,
            full_ids,
            prompt_length,
            synthid,
            residual_synthid,
            cc_extractor,
            args.temperature,
            args.top_k,
        )
    raise ValueError(f"unknown method: {method}")


def average_kl_over_keys(estimates: list[dict[str, Any]]) -> dict[str, Any]:
    """Average KL numerators over keys while keeping the shared entropy denominator."""
    if not estimates:
        return {"KL_WS_mean": 0.0, "KL_WS_sum": 0.0, "KL_WS_count": 0, "KL_WS_kind": "empty"}
    first = estimates[0]
    count = int(first["KL_WS_count"])
    entropy_sum = float(first.get("KL_WS_entropy_sum", 0.0))
    kl_sums = np.array([float(est["KL_WS_sum"]) for est in estimates], dtype=np.float64)
    kl_means = np.array([float(est["KL_WS_mean"]) for est in estimates], dtype=np.float64)
    ratios = np.array([float(est.get("KL_WS_ratio", 0.0)) for est in estimates], dtype=np.float64)
    kl_sum = float(kl_sums.mean())
    result = {
        **first,
        "KL_WS_sum": kl_sum,
        "KL_WS_mean": kl_sum / max(count, 1),
        "KL_WS_entropy_sum": entropy_sum,
        "KL_WS_entropy_mean": float(first.get("KL_WS_entropy_mean", 0.0)),
        "KL_WS_ratio": kl_sum / max(entropy_sum, 1e-20),
        "KL_WS_key_mean": float(kl_means.mean()),
        "KL_WS_key_std": float(kl_means.std()),
        "KL_WS_key_ratio_mean": float(ratios.mean()),
        "KL_WS_key_ratio_std": float(ratios.std()),
        "KL_WS_num_keys": len(estimates),
    }
    result["KL_WS_kind"] = f"{first.get('KL_WS_kind', 'unknown')}_key_avg"
    return result


def aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    method_rows = [row for row in rows if row["method"] == method]
    vals = [row["KL_WS_mean"] for row in method_rows]
    ratios = [row.get("KL_WS_ratio", 0.0) for row in method_rows]
    counts = [row["KL_WS_count"] for row in method_rows]
    sums = [row["KL_WS_sum"] for row in method_rows]
    entropy_sums = [row.get("KL_WS_entropy_sum", 0.0) for row in method_rows]
    total_count = int(np.sum(counts)) if counts else 0
    total_sum = float(np.sum(sums)) if sums else 0.0
    total_entropy = float(np.sum(entropy_sums)) if entropy_sums else 0.0
    chunk_lengths = [row.get("chunk_lengths", []) for row in method_rows]
    step_counts = [len(chunks) for chunks in chunk_lengths]
    token_counts = [int(np.sum(chunks)) for chunks in chunk_lengths]
    total_steps = int(np.sum(step_counts)) if step_counts else 0
    total_tokens = int(np.sum(token_counts)) if token_counts else 0
    sample_atps = [
        token_count / step_count
        for token_count, step_count in zip(token_counts, step_counts)
        if step_count > 0
    ]
    return {
        "num_samples": len(vals),
        "num_tokens": total_tokens,
        "num_steps": total_steps,
        "AATPS": total_tokens / max(total_steps, 1),
        "AATPS_sample_mean": float(np.mean(sample_atps)) if sample_atps else 0.0,
        "AATPS_sample_std": float(np.std(sample_atps)) if sample_atps else 0.0,
        "KL_WS_mean": total_sum / max(total_count, 1),
        "KL_WS_sample_mean": float(np.mean(vals)) if vals else 0.0,
        "KL_WS_sample_std": float(np.std(vals)) if vals else 0.0,
        "KL_WS_entropy_mean": total_entropy / max(total_count, 1),
        "KL_WS_entropy_sum": total_entropy,
        "KL_WS_ratio": total_sum / max(total_entropy, 1e-20),
        "KL_WS_sample_ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
        "KL_WS_sample_ratio_std": float(np.std(ratios)) if ratios else 0.0,
        "KL_WS_sum": total_sum,
        "KL_WS_count": total_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="JSON config file. Values in the config update CLI/default args.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ref-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", default="manual", choices=["manual", "summarization", "oeg", "eli5"])
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=["basic_uwm", "mc_uwm_strength", "mc_uwm_speed"])
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
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    return parser


def normalize_config_keys(config: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in config.items():
        normalized[key.replace("-", "_")] = value
    return normalized


def args_from_dict(base: argparse.Namespace, updates: dict[str, Any]) -> argparse.Namespace:
    values = vars(copy.deepcopy(base))
    values.update(normalize_config_keys(updates))
    values.pop("config", None)
    return argparse.Namespace(**values)


def load_configured_runs(parser: argparse.ArgumentParser, cli_args: argparse.Namespace) -> list[argparse.Namespace]:
    if cli_args.config is None:
        return [cli_args]

    with open(cli_args.config, "r", encoding="utf-8") as f:
        config = normalize_config_keys(json.load(f))

    base = parser.parse_args([])
    shared = {k: v for k, v in config.items() if k != "experiments"}
    if "experiments" not in config:
        return [args_from_dict(base, shared)]

    runs = []
    for experiment in config["experiments"]:
        merged = {**shared, **normalize_config_keys(experiment)}
        runs.append(args_from_dict(base, merged))
    return runs


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    load_kwargs = {
        "device_map": args.device,
        "low_cpu_mem_usage": True,
    }
    if args.device.startswith("cuda"):
        load_kwargs["torch_dtype"] = torch.float16
    target = transformers.AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    draft = transformers.AutoModelForCausalLM.from_pretrained(args.ref_model, **load_kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    target.eval()
    draft.eval()

    max_length_lp = MaxLengthLogitsProcessor(args.max_length, tokenizer.eos_token_id)
    stop_words_lp = StopWordsLogitsProcessor([], tokenizer.eos_token_id)
    process_logits_kwargs = {
        "logits_processor": transformers.LogitsProcessorList([max_length_lp]),
    }

    prompts = load_dataset(args.dataset, args.samples, tokenizer=tokenizer)
    rows = []
    start = time.perf_counter()
    for sample_idx, prompt in enumerate(prompts):
        for method in args.methods:
            input_ids = encode_prompt(tokenizer, args.model, prompt, target.device)
            max_length_lp.input_length = input_ids.shape[-1]
            input_ids, out_ids, generation_meta = collect_generation(
                method,
                target,
                draft,
                tokenizer,
                prompt,
                args,
                process_logits_kwargs,
            )
            kl_estimates = [
                estimate_kl(
                    method,
                    target,
                    draft,
                    input_ids,
                    out_ids,
                    args,
                    process_logits_kwargs,
                    generation_meta=generation_meta,
                    kl_key_index=key_idx,
                )
                for key_idx in range(args.kl_num_keys)
            ]
            kl = average_kl_over_keys(kl_estimates)
            row = {
                "sample_idx": sample_idx,
                "method": method,
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
    summary = {method: aggregate(rows, method) for method in args.methods}
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
    return result


def main():
    parser = build_parser()
    cli_args = parser.parse_args()
    runs = load_configured_runs(parser, cli_args)
    results = []
    for index, args in enumerate(runs):
        if len(runs) > 1:
            print(json.dumps({"run_index": index, "output": args.output, "methods": args.methods}, ensure_ascii=False))
        results.append(run_experiment(args))
    if len(results) > 1:
        print(json.dumps({"num_runs": len(results), "outputs": [run["args"].get("output") for run in results]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

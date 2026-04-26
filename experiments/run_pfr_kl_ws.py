import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuwm.pfr import build_default_labeler, pfr_sample_generator
from experiments.configured_eval_utils import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_TARGET_MODEL,
    encode_prompt,
    load_eval_rows,
    load_model,
    private_key_from_config,
    resolve_path,
)
from unbiased_watermark.scores.pfr_watermark_strength import (
    compute_pfr_watermark_strength_from_sequence,
)


def load_source_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data.get("args", data))


def build_process_logits_kwargs(args: dict) -> dict:
    warpers = []
    temperature = float(args.get("temperature", 1.0))
    top_k = int(args.get("top_k", 0) or 0)
    top_p = float(args.get("top_p", 0.0) or 0.0)
    if temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        warpers.append(TopKLogitsWarper(top_k))
    if top_p > 0.0:
        warpers.append(TopPLogitsWarper(top_p))
    return {
        "logits_warper": LogitsProcessorList(warpers) if warpers else None,
    }


def run_one(
    target,
    draft,
    tokenizer,
    prompt,
    lookahead: int,
    max_length: int,
    private_key: bytes,
    process_logits_kwargs: dict,
) -> dict:
    input_ids = encode_prompt(tokenizer, prompt, target.device)
    gen_labeler = build_default_labeler(mode="prefix")
    gen = pfr_sample_generator(
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        max_length=max_length,
        private_key=private_key,
        labeler=gen_labeler,
        process_logits_kwargs=process_logits_kwargs,
    )

    chunks = []
    chunk_lengths = []
    generated = 0
    for step_ids, _step_logprobs, _meta in gen:
        block_len = min(step_ids.shape[1], max_length - generated)
        if block_len <= 0:
            break
        chunks.append(step_ids[:, :block_len].detach())
        chunk_lengths.append(int(block_len))
        generated += block_len
        if generated >= max_length:
            break

    output_ids = (
        torch.cat(chunks, dim=1)
        if chunks
        else torch.empty((1, 0), dtype=torch.long, device=target.device)
    )
    full_ids = torch.cat([input_ids, output_ids], dim=1)
    score_labeler = build_default_labeler(mode="prefix")
    score = compute_pfr_watermark_strength_from_sequence(
        full_ids=full_ids,
        prompt_length=input_ids.shape[1],
        target_model=target,
        labeler=score_labeler,
        private_key=private_key,
        process_logits_kwargs=process_logits_kwargs,
    )
    return {
        "num_tokens": int(output_ids.shape[1]),
        "chunk_lengths": chunk_lengths,
        "KL_WS_sum": score["ws_sum"],
        "KL_WS_mean": score["WS_PFR_hat"],
        "KL_WS_entropy_sum": score["entropy_sum"],
        "KL_WS_entropy_mean": score["H_P_hat"],
        "KL_WS_ratio": score["ratio"],
        "KL_WS_count": score["num_scored"],
        "KL_WS_skipped_ratio": score["masked_ratio"],
        "KL_WS_kind": "pfr_target_side_full_vocab_keyed_winner",
    }


def summarize(rows: list[dict]) -> dict:
    total_ws = float(sum(r["KL_WS_sum"] for r in rows))
    total_h = float(sum(r["KL_WS_entropy_sum"] for r in rows))
    total_count = int(sum(r["KL_WS_count"] for r in rows))
    return {
        "num_samples": len(rows),
        "KL_WS_mean": float(total_ws / total_count) if total_count else 0.0,
        "KL_WS_sample_mean": float(np.mean([r["KL_WS_mean"] for r in rows])) if rows else 0.0,
        "KL_WS_sample_std": float(np.std([r["KL_WS_mean"] for r in rows])) if rows else 0.0,
        "KL_WS_entropy_mean": float(total_h / total_count) if total_count else 0.0,
        "KL_WS_entropy_sum": total_h,
        "KL_WS_ratio": float(total_ws / total_h) if total_h > 0.0 else 0.0,
        "KL_WS_sample_ratio_mean": float(np.mean([r["KL_WS_ratio"] for r in rows])) if rows else 0.0,
        "KL_WS_sample_ratio_std": float(np.std([r["KL_WS_ratio"] for r in rows])) if rows else 0.0,
        "KL_WS_sum": total_ws,
        "KL_WS_count": total_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pfr_kl_ws.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    source_args = load_source_config(args.source_config)
    samples = int(args.samples if args.samples is not None else source_args.get("samples", 100))
    dataset = source_args.get("dataset", "summarization")
    max_length = int(source_args.get("max_length", 32))
    lookahead = int(source_args.get("lookahead", 4))
    seed = int(source_args.get("seed", 1))
    private_key = private_key_from_config({"private_key": source_args.get("private_key", "1234")})
    device = args.device or source_args.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")

    target_model = resolve_path(source_args.get("target_model") or source_args.get("model"), DEFAULT_TARGET_MODEL)
    draft_model = resolve_path(source_args.get("draft_model") or source_args.get("ref_model"), DEFAULT_DRAFT_MODEL)
    if target_model is None or not target_model.exists():
        target_model = DEFAULT_TARGET_MODEL
    if draft_model is None or not draft_model.exists():
        draft_model = DEFAULT_DRAFT_MODEL

    torch.manual_seed(seed)
    target = load_model(target_model, device)
    draft = load_model(draft_model, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(target_model), local_files_only=True)
    eval_rows = load_eval_rows(dataset, samples)
    process_logits_kwargs = build_process_logits_kwargs(source_args)

    rows = []
    t0 = time.perf_counter()
    for idx, eval_row in enumerate(eval_rows):
        row = run_one(
            target=target,
            draft=draft,
            tokenizer=tokenizer,
            prompt=eval_row["prompt"],
            lookahead=lookahead,
            max_length=max_length,
            private_key=private_key,
            process_logits_kwargs=process_logits_kwargs,
        )
        row = {"sample_idx": idx, "method": "pfr", **row}
        rows.append(row)
        if (idx + 1) % 10 == 0 or idx + 1 == samples:
            print(json.dumps({"progress": idx + 1, "summary": summarize(rows)}, ensure_ascii=False), flush=True)

    out = {
        "elapsed_sec": time.perf_counter() - t0,
        "source_config": str(args.source_config),
        "args": {
            "model": str(target_model),
            "ref_model": str(draft_model),
            "dataset": dataset,
            "samples": samples,
            "method": "pfr",
            "max_length": max_length,
            "lookahead": lookahead,
            "temperature": source_args.get("temperature", 1.0),
            "top_k": source_args.get("top_k", 0),
            "top_p": source_args.get("top_p", 0.0),
            "seed": seed,
            "private_key": source_args.get("private_key", "1234"),
            "device": device,
        },
        "summary": {"pfr": summarize(rows)},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

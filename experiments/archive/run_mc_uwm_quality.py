import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unbiased_watermark as uwm

from accuwm.mc_watermark import mc_uwm_sample_generator
from experiments.run_multi_draft_pfr_aatps import load_model
from experiments.run_pfr_quality import encode_prompt, load_eval_rows, text_quality_metrics


DEFAULT_TARGET_MODEL = ROOT / "model" / "Qwen2.5-7B-Instruct"
DEFAULT_DRAFT_MODEL = ROOT / "model" / "Qwen2.5-0.5B-Instruct"


def build_reweight(name: str):
    if name == "deltagumbel":
        return uwm.DeltaGumbel_Reweight()
    if name == "gamma":
        return uwm.Gamma_Reweight()
    raise ValueError(f"unknown reweight: {name}")


@torch.no_grad()
def run_prompt(
    target,
    draft,
    tokenizer,
    prompt,
    reference,
    method,
    reweight_name,
    lookahead,
    max_length,
    seed,
):
    torch.manual_seed(seed)
    input_ids = encode_prompt(tokenizer, prompt, target.device)
    reweight = build_reweight(reweight_name)
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    gen = mc_uwm_sample_generator(
        reweight=reweight,
        cc_extractor=cc_extractor,
        cch=cch,
        private_key=seed.to_bytes(8, "big", signed=False) + b"1234",
        reweight_in_mc=(method == "mc_uwm_strength"),
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
    )

    output_chunks = []
    token_logprobs = []
    block_lens = []
    generated = 0
    start = time.perf_counter()
    for step_output_ids, step_output_logprobs in gen:
        block_len = step_output_ids.shape[1]
        if generated + block_len > max_length:
            block_len = max_length - generated
        if block_len <= 0:
            break

        ids = step_output_ids[:, :block_len]
        logprobs = step_output_logprobs[:, :block_len, :]
        gathered = torch.gather(logprobs, dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
        output_chunks.append(ids.detach().cpu())
        token_logprobs.extend(float(x) for x in gathered[0].detach().cpu().tolist())
        block_lens.append(int(block_len))

        generated += block_len
        if generated >= max_length:
            break
    elapsed = time.perf_counter() - start

    output_ids = (
        torch.cat(output_chunks, dim=1)
        if output_chunks
        else torch.empty((1, 0), dtype=torch.long)
    )
    generation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    quality = text_quality_metrics(generation, reference)

    num_steps = len(block_lens)
    num_tokens = int(output_ids.shape[1])
    block_efficiency = float(num_tokens / num_steps) if num_steps else 0.0
    token_rate = float(num_tokens / elapsed) if elapsed > 0 else 0.0
    log_perplexity = -float(np.mean(token_logprobs)) if token_logprobs else 0.0

    return {
        "method": method,
        "reweight": reweight_name,
        "seed": seed,
        "num_steps": num_steps,
        "num_invocations": num_steps,
        "num_tokens": num_tokens,
        "AATPS": block_efficiency,
        "BE": block_efficiency,
        "log_perplexity": log_perplexity,
        "perplexity": float(np.exp(min(log_perplexity, 50.0))) if token_logprobs else 0.0,
        **quality,
        "elapsed_sec": float(elapsed),
        "total_time": float(elapsed),
        "token_rate": token_rate,
        "TR": token_rate,
        "prediction": generation,
        "reference": reference,
    }


def aggregate(rows):
    numeric_keys = [
        "BE",
        "log_perplexity",
        "perplexity",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "bleu",
        "token_rate",
    ]
    total_tokens = int(sum(row["num_tokens"] for row in rows))
    total_time = float(sum(row["total_time"] for row in rows))
    summary = {
        "num_samples": len(rows),
        "num_tokens_total": total_tokens,
        "num_invocations_total": int(sum(row["num_invocations"] for row in rows)),
        "TR_global": float(total_tokens / total_time) if total_time > 0 else 0.0,
        "total_time_total": total_time,
    }
    for key in numeric_keys:
        values = [row[key] for row in rows]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    summary["TR_mean"] = summary["token_rate_mean"]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--dataset", choices=["gsm8k", "summarization"], default="gsm8k")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["mc_uwm_speed", "mc_uwm_strength"],
        choices=["mc_uwm_speed", "mc_uwm_strength"],
    )
    parser.add_argument("--reweight", choices=["deltagumbel", "gamma"], default="deltagumbel")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--draft-device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rows-output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    draft_device = args.draft_device or args.target_device
    target = load_model(args.target_model, args.target_device)
    draft = load_model(args.draft_model, draft_device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(args.target_model),
        local_files_only=True,
    )
    rows_in = load_eval_rows(args.dataset, args.samples)

    if args.warmup > 0:
        for row in rows_in[: args.warmup]:
            run_prompt(
                target,
                draft,
                tokenizer,
                row["prompt"],
                row["reference"],
                args.methods[0],
                args.reweight,
                args.lookahead,
                args.max_length,
                args.seed,
            )

    all_rows = []
    results = {}
    if args.rows_output is not None:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        args.rows_output.write_text("", encoding="utf-8")

    for method in args.methods:
        rows = []
        for idx, row_in in enumerate(rows_in):
            sample_start = time.perf_counter()
            metrics = run_prompt(
                target,
                draft,
                tokenizer,
                row_in["prompt"],
                row_in["reference"],
                method,
                args.reweight,
                args.lookahead,
                args.max_length,
                args.seed,
            )
            row = {
                "sample_idx": idx,
                "dataset_idx": row_in["idx"],
                "lookahead": args.lookahead,
                **metrics,
            }
            rows.append(row)
            all_rows.append(row)
            if args.rows_output is not None:
                with args.rows_output.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if args.progress_every > 0 and (
                (idx + 1) % args.progress_every == 0 or idx + 1 == len(rows_in)
            ):
                partial = aggregate(rows)
                print(
                    json.dumps(
                        {
                            "progress": {
                                "method": method,
                                "done": idx + 1,
                                "total": len(rows_in),
                                "last_sample_sec": time.perf_counter() - sample_start,
                                "partial_summary": partial,
                            }
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        results[method] = {
            "summary": aggregate(rows),
            "rows": rows,
        }

    payload = {
        "samples": args.samples,
        "dataset": args.dataset,
        "warmup": args.warmup,
        "lookahead": args.lookahead,
        "max_length": args.max_length,
        "methods": args.methods,
        "reweight": args.reweight,
        "seed": args.seed,
        "target_model": str(args.target_model),
        "draft_model": str(args.draft_model),
        "target_device": args.target_device,
        "draft_device": draft_device,
        "results": results,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuwm.pfr import pfr_sample_generator
from experiments.run_multi_draft_pfr_aatps import load_model


DEFAULT_TARGET_MODEL = ROOT / "model" / "Qwen2.5-7B-Instruct"
DEFAULT_DRAFT_MODEL = ROOT / "model" / "Qwen2.5-0.5B-Instruct"


def encode_prompt(tokenizer, prompt, device):
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    return tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def f1_score(overlap: int, pred_count: int, ref_count: int) -> float:
    if overlap <= 0 or pred_count <= 0 or ref_count <= 0:
        return 0.0
    precision = overlap / pred_count
    recall = overlap / ref_count
    return 2 * precision * recall / (precision + recall)


def ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_f1(pred_tokens: list[str], ref_tokens: list[str], n: int) -> float:
    pred = ngrams(pred_tokens, n)
    ref = ngrams(ref_tokens, n)
    overlap = sum((pred & ref).values())
    return f1_score(overlap, sum(pred.values()), sum(ref.values()))


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def text_quality_metrics(prediction: str, reference: str) -> dict:
    pred_tokens = normalize_tokens(prediction)
    ref_tokens = normalize_tokens(reference)
    smoothie = SmoothingFunction().method1
    bleu = (
        float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie))
        if pred_tokens and ref_tokens
        else 0.0
    )
    return {
        "rouge1_f1": rouge_n_f1(pred_tokens, ref_tokens, 1),
        "rouge2_f1": rouge_n_f1(pred_tokens, ref_tokens, 2),
        "rougeL_f1": f1_score(lcs_len(pred_tokens, ref_tokens), len(pred_tokens), len(ref_tokens)),
        "bleu": bleu,
    }


def load_eval_rows(dataset: str, samples: int):
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")["train"]
        ds = ds.select(range(samples))
        rows = []
        for idx, row in enumerate(ds):
            rows.append(
                {
                    "idx": idx,
                    "prompt": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": row["question"]},
                    ],
                    "reference": row["answer"],
                }
            )
        return rows

    if dataset == "summarization":
        ds = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)["test"]
        ds = ds.filter(lambda x: len(x["article"]) < 3000)
        ds = ds.select(range(samples))
        rows = []
        for idx, row in enumerate(ds):
            rows.append(
                {
                    "idx": idx,
                    "prompt": (
                        "System:Summarize the following article.\n"
                        f"INPUT:{row['article'][:1000]}\nOUTPUT:"
                    ),
                    "reference": row["highlights"],
                }
            )
        return rows

    raise ValueError(f"unknown dataset: {dataset}")


@torch.no_grad()
def run_prompt(target, draft, tokenizer, prompt, reference, lookahead, max_length):
    input_ids = encode_prompt(tokenizer, prompt, target.device)
    gen = pfr_sample_generator(
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        max_length=max_length,
        private_key=b"1234",
        labeler_mode="prefix",
    )

    output_chunks = []
    token_logprobs = []
    accepted_counts = []
    draft_lens = []
    generated = 0
    start = time.perf_counter()
    for step_output_ids, step_output_logprobs, meta in gen:
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
        accepted_counts.append(int(meta["accepted_count"]))
        draft_lens.append(int(meta["draft_len"]))

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

    num_tokens = int(output_ids.shape[1])
    nll = -float(np.mean(token_logprobs)) if token_logprobs else 0.0
    perplexity = float(math.exp(min(nll, 50.0))) if token_logprobs else 0.0
    num_steps = len(accepted_counts)
    block_efficiency = float(num_tokens / num_steps) if num_steps else 0.0
    token_rate = float(num_tokens / elapsed) if elapsed > 0 else 0.0

    return {
        "num_steps": num_steps,
        "num_invocations": num_steps,
        "num_tokens": num_tokens,
        "AATPS": block_efficiency,
        "BE": block_efficiency,
        "accepted_draft_mean": float(np.mean(accepted_counts)) if accepted_counts else 0.0,
        "draft_len_mean": float(np.mean(draft_lens)) if draft_lens else 0.0,
        "log_perplexity": nll,
        "perplexity": perplexity,
        **quality,
        "elapsed_sec": float(elapsed),
        "total_time": float(elapsed),
        "token_rate": token_rate,
        "prediction": generation,
        "reference": reference,
    }


def aggregate(rows):
    numeric_keys = [
        "BE",
        "accepted_draft_mean",
        "draft_len_mean",
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
    summary["BE_mean"] = summary["BE_mean"]
    summary["TR_mean"] = summary["token_rate_mean"]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--dataset", choices=["gsm8k", "summarization"], default="gsm8k")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--draft-device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rows-output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
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
                args.lookahead,
                args.max_length,
            )

    rows = []
    if args.rows_output is not None:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        args.rows_output.write_text("", encoding="utf-8")

    for idx, row_in in enumerate(rows_in):
        sample_start = time.perf_counter()
        metrics = run_prompt(
            target,
            draft,
            tokenizer,
            row_in["prompt"],
            row_in["reference"],
            args.lookahead,
            args.max_length,
        )
        row = {
            "sample_idx": idx,
            "dataset_idx": row_in["idx"],
            "lookahead": args.lookahead,
            **metrics,
        }
        rows.append(row)
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
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "samples": args.samples,
                            "dataset": args.dataset,
                            "warmup": args.warmup,
                            "lookahead": args.lookahead,
                            "max_length": args.max_length,
                            "target_model": str(args.target_model),
                            "draft_model": str(args.draft_model),
                            "target_device": args.target_device,
                            "draft_device": draft_device,
                            "partial": True,
                            "summary": partial,
                            "rows": rows,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    payload = {
        "samples": args.samples,
        "dataset": args.dataset,
        "warmup": args.warmup,
        "lookahead": args.lookahead,
        "max_length": args.max_length,
        "target_model": str(args.target_model),
        "draft_model": str(args.draft_model),
        "target_device": args.target_device,
        "draft_device": draft_device,
        "summary": aggregate(rows),
        "rows": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

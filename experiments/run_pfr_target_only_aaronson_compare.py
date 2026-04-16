import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import transformers


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import accuwm.pfr as pfr
import unbiased_watermark as uwm
from accuwm.utils import process_logits
from experiments.tasks import get_summarization_ds
from unbiased_watermark.scores.pfr_aaronson import compute_pfr_aaronson_from_sequence


TARGET_MODEL = ROOT / "model" / "Qwen2.5-7B-Instruct"
DRAFT_MODEL = ROOT / "model" / "Qwen2.5-0.5B-Instruct"


def load_model(model_path: Path, device: str):
    kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "device_map": device,
        "low_cpu_mem_usage": True,
    }
    if device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16
    return transformers.AutoModelForCausalLM.from_pretrained(**kwargs)


@torch.no_grad()
def target_only_pfr_generate(
    model,
    input_ids: torch.LongTensor,
    max_length: int,
    private_key: bytes,
    labeler_mode: str,
    cc_extractor=None,
    process_logits_kwargs=None,
):
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    if cc_extractor is None:
        cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    labeler = pfr.build_default_labeler(mode=labeler_mode, cc_extractor=cc_extractor)
    source_factory = pfr.PFRSourceFactory(private_key=private_key)

    running_ids = input_ids
    past_key_values = None
    input_tokens = input_ids
    output_ids = []
    step_lens = []

    while len(output_ids) < max_length:
        output = model(input_tokens, past_key_values=past_key_values)
        logits = output.logits[:, -1, :]
        logits = process_logits(running_ids, logits, **process_logits_kwargs)
        label_info = labeler.label_info(running_ids)
        source = source_factory.build(label_info.source_label)
        winner, _, _ = pfr.pfr_win_from_logits(logits, source, model.device)
        new_token = winner.unsqueeze(1)

        output_ids.extend(new_token[0].tolist())
        step_lens.append(1)
        running_ids = torch.cat([running_ids, new_token], dim=1)
        input_tokens = new_token
        past_key_values = output.past_key_values
        if (new_token == model.config.eos_token_id).all():
            break

    return running_ids, step_lens


def run_method(target, tokenizer, prompt, method, n, max_length):
    del n
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(target.device)
    labeler_mode = "prefix" if method.endswith("prefix") else "context_code"
    t0 = time.perf_counter()
    full_ids, step_lens = target_only_pfr_generate(
        model=target,
        input_ids=input_ids,
        max_length=max_length,
        private_key=b"1234",
        labeler_mode=labeler_mode,
    )
    elapsed = time.perf_counter() - t0
    detect_labeler = pfr.build_default_labeler(
        mode=labeler_mode,
        cc_extractor=uwm.lm.PrevN_ContextCodeExtractor(n=3),
    )
    score = compute_pfr_aaronson_from_sequence(
        full_ids=full_ids,
        prompt_length=input_ids.shape[1],
        labeler=detect_labeler,
        private_key=b"1234",
        vocab_size=target.config.vocab_size,
    )
    num_tokens = full_ids.shape[1] - input_ids.shape[1]
    return {
        "method": method,
        "num_tokens": int(num_tokens),
        "num_steps": len(step_lens),
        "AATPS": float(np.mean(step_lens)) if step_lens else 0.0,
        "tokens_per_sec": float(num_tokens / elapsed) if elapsed > 0 else 0.0,
        **score,
    }


def aggregate(rows, method):
    method_rows = [row for row in rows if row["method"] == method]
    keys = [
        "AATPS",
        "ANLPPT_Aaronson",
        "score_mean",
        "masked_ratio",
        "tokens_per_sec",
    ]
    summary = {"num_samples": len(method_rows)}
    for key in keys:
        vals = [row[key] for row in method_rows]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["target_pfr_prefix", "target_pfr_cc"],
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target = load_model(TARGET_MODEL, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(TARGET_MODEL))
    prompts = [row["prompt"] for row in get_summarization_ds(ds_cut_len=args.samples)]

    rows = []
    t0 = time.perf_counter()
    for sample_idx, prompt in enumerate(prompts):
        for method in args.methods:
            row = run_method(
                target=target,
                tokenizer=tokenizer,
                prompt=prompt,
                method=method,
                n=args.n,
                max_length=args.max_length,
            )
            row["sample_idx"] = sample_idx
            rows.append(row)
    elapsed = time.perf_counter() - t0

    summary = {method: aggregate(rows, method) for method in args.methods}
    print(
        json.dumps(
            {
                "samples": args.samples,
                "n": args.n,
                "max_length": args.max_length,
                "elapsed_sec": elapsed,
                "detector": "PFR_Aaronson_sequence_no_tag",
                "summary": summary,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

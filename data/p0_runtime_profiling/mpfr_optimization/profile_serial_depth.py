"""Profile the serial structure of the cached MPFR draft tree.

Measures synchronized model-forward time at each draft depth and the work
between consecutive forwards.  This is a diagnostic only; generation uses the
normal MPFR implementation and its outputs are saved for regression checks.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MPFR_spec"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mpfr_batched_torchgen_cached as mpfr  # noqa: E402
from validate_mpfr_optimization import run_one  # noqa: E402
from experiments._shared import (  # noqa: E402
    _maybe_inject_chat_template,
    encode_prompt,
    load_prompts,
)


def sync() -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


class SerialDepthProfiler:
    def __init__(self, draft_model):
        self.draft_model = draft_model
        self.original = mpfr._model_forward
        self.values = defaultdict(list)
        self.depth = 0
        self.last_kind = None
        self.last_end = None

    def reset_sequence(self) -> None:
        self.depth = 0
        self.last_kind = None
        self.last_end = None

    def install(self) -> None:
        def wrapped(model, *, num_logits_to_keep, **kwargs):
            sync()
            start = time.perf_counter()
            is_draft = model is self.draft_model
            if is_draft:
                self.depth += 1
                if self.last_end is not None:
                    gap_name = (
                        "block_boundary_gap_ms"
                        if self.last_kind == "target"
                        else f"gap_before_depth_{self.depth}_ms"
                    )
                    self.values[gap_name].append(1000.0 * (start - self.last_end))
            else:
                if self.last_end is not None:
                    self.values["pre_target_gap_ms"].append(
                        1000.0 * (start - self.last_end)
                    )

            output = self.original(
                model, num_logits_to_keep=num_logits_to_keep, **kwargs
            )
            sync()
            end = time.perf_counter()
            if is_draft:
                depth = self.depth
                ids = kwargs["input_ids"]
                self.values[f"draft_depth_{depth}_forward_ms"].append(
                    1000.0 * (end - start)
                )
                self.values[f"draft_depth_{depth}_batch_rows"].append(
                    int(ids.shape[0])
                )
                self.values[f"draft_depth_{depth}_input_tokens"].append(
                    int(ids.shape[1])
                )
                self.last_kind = "draft"
            else:
                self.values["target_forward_ms"].append(1000.0 * (end - start))
                self.last_kind = "target"
                self.depth = 0
            self.last_end = end
            return output

        mpfr._model_forward = wrapped

    def restore(self) -> None:
        mpfr._model_forward = self.original

    def summary(self):
        return {
            key: {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
            for key, values in sorted(self.values.items())
            if values
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    _maybe_inject_chat_template(tokenizer, args.target)
    model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float16, device_map=args.device,
        low_cpu_mem_usage=True,
    ).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, torch_dtype=torch.float16, device_map=args.device,
        low_cpu_mem_usage=True,
    ).eval()
    prompts = load_prompts("cnn_dailymail", args.samples)
    profiler = SerialDepthProfiler(draft)
    results = {}

    # Warm model kernels and allocator before installing synchronization hooks.
    warm_ids = encode_prompt(tokenizer, prompts[0], args.device)
    run_one(model, draft, warm_ids, prompt_idx=10_000, width=args.widths[0], args=args)

    profiler.install()
    try:
        for width in args.widths:
            profiler.values.clear()
            rows = []
            for prompt_idx, prompt in enumerate(prompts):
                profiler.reset_sequence()
                ids = encode_prompt(tokenizer, prompt, args.device)
                rows.append(run_one(
                    model, draft, ids, prompt_idx=prompt_idx,
                    width=width, args=args,
                ))
            results[str(width)] = {
                "rows": rows,
                "serial_profile": profiler.summary(),
            }
    finally:
        profiler.restore()

    payload = {
        "config": vars(args) | {"output": str(args.output)},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({b: v["serial_profile"] for b, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()

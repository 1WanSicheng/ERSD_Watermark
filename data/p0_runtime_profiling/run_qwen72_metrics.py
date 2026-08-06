#!/usr/bin/env python3
"""Run the paper-style watermark and quality metrics on sharded Qwen-72B."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import _shared as S  # noqa: E402
from experiments.run_multi_draft import run_experiment as run_multi  # noqa: E402
from experiments.run_single_draft import run_experiment as run_single  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "multi"), required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def install_sharded_loader():
    def load_models_and_tokenizer(config: dict, device: str | None = None):
        dtype_name = str(config.get("dtype", "bfloat16")).lower()
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        target_path = str(S.resolve_model_id(config["target_model"], None))
        draft_path = str(S.resolve_model_id(config["draft_model"], None))

        target = AutoModelForCausalLM.from_pretrained(
            target_path,
            device_map=config.get("target_device_map", "balanced"),
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).eval()
        draft_device = str(config.get("draft_device", "cuda:0"))
        draft = AutoModelForCausalLM.from_pretrained(
            draft_path,
            device_map=draft_device,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).eval()
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                target_path, local_files_only=True
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                target_path, local_files_only=True, use_fast=False
            )
        S._maybe_inject_chat_template(tokenizer, target_path)
        return target, draft, tokenizer, target.device

    S.load_models_and_tokenizer = load_models_and_tokenizer


def common_config(args):
    return {
        "experiment": f"qwen72_{args.mode}_metrics_n{args.samples}",
        "samples": args.samples,
        "dataset": "cnn_dailymail",
        "max_new_tokens": args.max_new_tokens,
        "lookaheads": [4],
        "private_key": "1234",
        "target_model": str(args.target),
        "draft_model": str(args.draft),
        "target_device_map": "balanced",
        "draft_device": "cuda:0",
        "dtype": "bfloat16",
        "use_chat_template": True,
        "process_logits": {
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 1.0,
        },
        "metrics": {
            "save_output_ids": True,
            "aatps": True,
            "token_rate": True,
            "anlppt": {
                "variants": ["U", "Li", "PL"],
                "li_delta": 0.5,
                "pl_eps": 0.1,
            },
            "tpr_at_n": {
                "tokens": [32, 64, 96, 128],
                "fpr": 0.01,
                "variants": ["U", "Li", "PL"],
            },
        },
    }


def main():
    args = parse_args()
    install_sharded_loader()
    config = common_config(args)

    if args.mode == "single":
        decoders = ["mc", "pfr_no_watermark", "pfr"]
        config["decoders"] = decoders
        config["metrics"]["lppl"] = {"applies_to": decoders}
        config["metrics"]["rouge_vs_reference"] = {"applies_to": decoders}
        config["metrics"]["kl_ratio"] = {"applies_to": ["pfr"]}
        config["metrics"]["tpr_at_n"]["applies_to"] = decoders
        config["metrics"]["tpr_at_n"]["detector_per_decoder"] = {
            "mc": {"kind": "DeltaGumbel"},
            "pfr_no_watermark": {
                "kind": "PFR",
                "labeler": "context_code",
            },
            "pfr": {"kind": "PFR", "labeler": "context_code"},
        }
        result = run_single(config)
    else:
        decoders = ["mpfr_torchgen_cached", "invariant_multi"]
        config["num_drafts"] = [4, 8]
        config["decoders"] = decoders
        config["metrics"]["anlppt"]["applies_to"] = decoders
        config["metrics"]["lppl"] = {"applies_to": decoders}
        config["metrics"]["rouge_vs_reference"] = {"applies_to": decoders}
        config["metrics"]["kl_ratio"] = {
            "applies_to": ["mpfr_torchgen_cached"]
        }
        config["metrics"]["tpr_at_n"]["applies_to"] = decoders
        config["metrics"]["tpr_at_n"]["detector_per_decoder"] = {
            "mpfr_torchgen_cached": {
                "kind": "PFR",
                "labeler": "mpfr_direct",
            },
            "invariant_multi": {
                "kind": "PFR",
                "labeler": "mpfr_direct",
            },
        }
        result = run_multi(config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

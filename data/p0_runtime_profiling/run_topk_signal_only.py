#!/usr/bin/env python3
"""Generate and detect top-k watermark signal without runtime profiling."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_multi_draft import run_experiment as run_multi  # noqa: E402
from experiments.run_single_draft import run_experiment as run_single  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "multi"), required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def common_config(args):
    return {
        "experiment": f"topk_signal_only_{args.mode}",
        "signal_only": True,
        "samples": args.samples,
        "dataset": "cnn_dailymail",
        "max_new_tokens": 128,
        "lookaheads": [4],
        "private_key": "1234",
        "target_model": "Qwen/Qwen2.5-7B-Instruct",
        "draft_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "process_logits": {
            "top_k": args.top_k,
            "top_p": 1.0,
            "temperature": 1.0,
        },
        "metrics": {
            "save_output_ids": True,
            "aatps": True,
            "token_rate": False,
            "anlppt": {
                "variants": ["U"],
                "li_delta": 0.5,
                "pl_eps": 0.1,
            },
            "tpr_at_n": {
                "tokens": [32, 64, 96, 128],
                "fpr": 0.01,
                "variants": ["U"],
            },
        },
    }


def main():
    args = parse_args()
    config = common_config(args)
    if args.mode == "single":
        config["decoders"] = [
            "pfr",
            "pfr_no_watermark",
            "mc_uwm_speed",
            "mc_uwm_strength",
        ]
        config["metrics"]["tpr_at_n"]["applies_to"] = config["decoders"]
        config["metrics"]["tpr_at_n"]["detector_per_decoder"] = {
            "pfr_no_watermark": {
                "kind": "PFR",
                "labeler": "context_code",
            },
        }
        result = run_single(config)
    else:
        config["num_drafts"] = [4, 8]
        config["decoders"] = [
            "mpfr_torchgen_cached",
            "invariant_multi",
        ]
        config["metrics"]["anlppt"]["applies_to"] = config["decoders"]
        config["metrics"]["tpr_at_n"]["applies_to"] = config["decoders"]
        config["metrics"]["tpr_at_n"]["detector_per_decoder"] = {
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

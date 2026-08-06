#!/usr/bin/env python3
"""Evaluate PDF detectors on saved PF generations without loading a model.

This script loads only the tokenizer and prompt dataset needed to reconstruct
the exact context labels.  It regenerates the emitted tokens' keyed pivots,
then calibrates every detector against i.i.d. uniform pivots.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import _shared as shared  # noqa: E402
from PF_sd.frozen_pf_decoder import benchmark as frozen_benchmark  # noqa: E402
from PF_sd.frozen_pf_decoder.core.max_order_pf import (  # noqa: E402
    max_order_context_label,
    recover_max_order_pivots,
)
from PF_sd.task12_model_free_pf_detectors.detectors import (  # noqa: E402
    DetectorSpec,
    NullCalibrator,
)


def _specs() -> list[DetectorSpec]:
    specs = [DetectorSpec("Original", "original")]
    specs.extend(
        DetectorSpec(f"Li-rho={rho:g}", "li", rho)
        for rho in (0.05, 0.1, 0.2, 0.5)
    )
    specs.extend(
        DetectorSpec(f"PowerLaw-eps={eps:g}", "power_law", eps)
        for eps in (0.01, 0.02, 0.05, 0.1)
    )
    return specs


def _load_tokenizer(config: dict):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["target_model"],
        local_files_only=True,
        trust_remote_code=True,
    )
    # Match the benchmark's tokenizer-only part of model loading exactly.
    # Vicuna tokenizers do not necessarily ship a chat_template, so the
    # shared runner injects the repository's stable template at runtime.
    shared._maybe_inject_chat_template(tokenizer, config["target_model"])
    return tokenizer


def _recover_rows(payload: dict, method: str, device: str) -> list[np.ndarray]:
    config = payload["config"]
    tokenizer = _load_tokenizer(config)
    prompts = frozen_benchmark._load_prompts(config)
    use_chat = bool(config.get("use_chat_template", True))
    base_key = shared.private_key_from_str(config.get("private_key", "max-order-pf"))
    vocab_size = int(tokenizer.vocab_size)
    rows = [row for row in payload["rows"] if row["method"] == method]
    recovered = []
    for row in rows:
        prompt_idx = int(row["prompt_idx"])
        width = int(row["width"])
        prompt_ids = shared.encode_prompt(
            tokenizer, prompts[prompt_idx], "cpu", use_chat_template=use_chat
        ).reshape(-1).tolist()
        output_ids = [int(token) for token in row["output_token_ids"]]
        context = list(prompt_ids)
        labels = []
        for token in output_ids:
            labels.append(max_order_context_label(tuple(context), width))
            context.append(token)
        private_key = shared.seeded_private_key(
            int(config.get("seed", 7)) + prompt_idx, base_key
        )
        pivots = recover_max_order_pivots(
            out_ids=torch.tensor(output_ids, dtype=torch.long).reshape(1, -1),
            context_labels=labels,
            width=width,
            private_key=private_key,
            vocab_size=vocab_size,
            device=device,
            target_coupling="latin_hypercube",
            rng_backend="counter_philox",
        )
        recovered.append(pivots.detach().cpu().numpy().astype(np.float64))
    return recovered


def evaluate(payload: dict, method: str, device: str, n_null: int) -> dict:
    sequences = _recover_rows(payload, method, device)
    lengths = sorted({len(pivots) for pivots in sequences})
    calibrators = {
        (spec.name, length): NullCalibrator(spec, length, n_null=n_null)
        for spec in _specs()
        for length in lengths
    }
    results = {}
    for spec in _specs():
        evaluated = [
            calibrators[(spec.name, len(pivots))].evaluate(pivots)
            for pivots in sequences
        ]
        scores_by_length = defaultdict(list)
        for pivots, row in zip(sequences, evaluated):
            scores_by_length[len(pivots)].append(row["score"])
        detection = {}
        for alpha in (0.01, 0.001):
            hits = 0
            total = 0
            for length, scores in scores_by_length.items():
                threshold = calibrators[(spec.name, length)].threshold(alpha)
                hits += int(np.count_nonzero(np.asarray(scores) >= threshold))
                total += len(scores)
            detection[f"TPR@FPR={alpha:g}"] = hits / max(total, 1)
        results[spec.name] = {
            "n_sequences": len(evaluated),
            "mean_score_per_token": float(
                np.mean([row["score_per_token"] for row in evaluated])
            ),
            "mean_ANLPPT": float(np.mean([row["ANLPPT"] for row in evaluated])),
            "median_ANLPPT": float(np.median([row["ANLPPT"] for row in evaluated])),
            **detection,
        }
    return {
        "source_config": payload["config"],
        "evaluated_method": method,
        "n_null": n_null,
        "model_access": False,
        "detectors": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", default="latin_pf_counter_tree_free")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-null", type=int, default=200_000)
    args = parser.parse_args()
    with args.input.open() as handle:
        payload = json.load(handle)
    result = evaluate(payload, args.method, args.device, args.n_null)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["detectors"], indent=2))


if __name__ == "__main__":
    main()

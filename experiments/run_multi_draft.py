"""
Multi-draft experiment runner: JSON-config driven.

Compares B>=1 multi-draft decoders (ms_pfr_cached, mpfr_torchgen_cached,
invariant_multi, strong_multi) at one or more (B, lookahead) settings.
Reports AATPS and token_rate per (decoder, B, lookahead).  ANLPPT-{U, Li, PL}
is reported only for the PFR-family decoders configured in
``metrics.anlppt.applies_to``.

Config schema (see experiments/configs/multi_draft_default.json):

    {
      "experiment": "multi_draft",
      "samples": 100,
      "dataset": "cnn_dailymail",
      "max_new_tokens": 128,
      "lookaheads": [4],
      "num_drafts": [1, 2, 4, 8],
      "private_key": "1234",
      "process_logits": {"top_k": 50, "top_p": 1.0, "temperature": 1.0},
      "decoders": ["ms_pfr_cached", "mpfr_torchgen_cached",
                   "invariant_multi", "strong_multi"],
      "metrics": {
          "aatps": true,
          "token_rate": true,
          "anlppt": {
              "variants": ["U", "Li", "PL"],
              "li_delta": 0.5, "pl_eps": 0.1,
              "applies_to": ["ms_pfr_cached", "mpfr_torchgen_cached"]
          }
      },
      "output": "outputs/multi_draft.json"
    }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from . import _shared as S


def _decoder_uniforms(
    *, target, in_ids, out_ids, wm_kind, source_labels, masked_flags,
    private_key, vocab_size,
):
    if wm_kind == "PFR":
        if not source_labels or len(source_labels) != int(out_ids.shape[-1]):
            return None
        Us, skipped = S.uniforms_from_pfr(
            out_ids=out_ids, source_labels=source_labels,
            masked_flags=masked_flags or [False] * int(out_ids.shape[-1]),
            private_key=private_key, vocab_size=vocab_size,
        )
        return Us, skipped
    if wm_kind == "DeltaGumbel":
        Us, skipped = S.uniforms_from_dg(
            target=target, out_ids=out_ids, in_ids=in_ids,
            private_key=private_key,
        )
        return Us, skipped
    return None


def run_one_prompt(
    *, decoder_name, decoder_fn, target, draft, tokenizer, prompt,
    lookahead, num_drafts, max_length, seed, base_key, plk, vocab_size,
    metrics_cfg: dict, anlppt_applies: bool,
) -> dict:
    input_ids = S.encode_prompt(tokenizer, prompt, target.device)
    torch.manual_seed(seed)
    S._sync()
    t0 = time.perf_counter()
    out_ids, block_lens, pk, wm_kind, src_labels, masked_flags = decoder_fn(
        target, draft, input_ids,
        lookahead=lookahead, max_length=max_length,
        seed=seed, base_key=base_key, plk=plk,
        num_drafts=num_drafts,
    )
    S._sync()
    elapsed = time.perf_counter() - t0

    n_tokens = int(out_ids.shape[-1])
    n_steps = max(len(block_lens), 1)
    aatps = n_tokens / n_steps
    token_rate = n_tokens / elapsed if elapsed > 0 else 0.0

    row: Dict[str, float | int | str] = {
        "decoder": decoder_name,
        "lookahead": int(lookahead),
        "num_drafts": int(num_drafts),
        "tokens": n_tokens,
        "blocks": n_steps,
        "AATPS": float(aatps),
        "token_rate": float(token_rate),
        "elapsed_sec": float(elapsed),
        "wm_kind": wm_kind,
    }

    if anlppt_applies:
        anlppt_cfg = (metrics_cfg or {}).get("anlppt") or {}
        urec = _decoder_uniforms(
            target=target, in_ids=input_ids,
            out_ids=out_ids.to(target.device),
            wm_kind=wm_kind, source_labels=src_labels,
            masked_flags=masked_flags, private_key=pk,
            vocab_size=vocab_size,
        )
        if urec is None:
            for v in anlppt_cfg.get("variants", ["U", "Li", "PL"]):
                row[f"ANLPPT_{v}"] = float("nan")
        else:
            Us, skipped = urec
            metrics = S.anlppt_metrics(
                Us,
                li_delta=float(anlppt_cfg.get("li_delta", 0.5)),
                pl_eps=float(anlppt_cfg.get("pl_eps", 0.1)),
                skipped=skipped,
                variants=list(anlppt_cfg.get("variants", ["U", "Li", "PL"])),
            )
            row.update(metrics)
    return row


def run_experiment(config: dict) -> dict:
    samples = int(config.get("samples", 100))
    dataset = str(config.get("dataset", "cnn_dailymail"))
    max_new_tokens = int(config.get("max_new_tokens", 128))
    lookaheads: List[int] = list(config.get("lookaheads", [4]))
    num_drafts: List[int] = list(config.get("num_drafts", [4]))
    base_key = S.private_key_from_str(config.get("private_key", "1234"))
    plk = S.build_process_logits_kwargs(config.get("process_logits"))
    decoders: List[str] = list(
        config.get("decoders", list(S.MULTI_DRAFT_DECODERS.keys()))
    )
    metrics_cfg: dict = config.get("metrics", {}) or {}
    anlppt_applies_to = set(
        ((metrics_cfg.get("anlppt") or {}).get("applies_to") or [])
    )

    target, draft, tokenizer, device = S.load_models_and_tokenizer(config)
    vocab_size = int(target.config.vocab_size)
    prompts = S.load_prompts(dataset, samples)

    print(
        f"[multi_draft] dataset={dataset}  samples={samples}  "
        f"max_new={max_new_tokens}  L={lookaheads}  B={num_drafts}  "
        f"decoders={decoders}"
    )

    rows: List[dict] = []
    for d_name in decoders:
        if d_name not in S.MULTI_DRAFT_DECODERS:
            print(f"  [warn] unknown decoder: {d_name}, skipped")
            continue
        decoder_fn = S.MULTI_DRAFT_DECODERS[d_name]
        anlppt_applies = (d_name in anlppt_applies_to) if anlppt_applies_to else False
        for L in lookaheads:
            for B in num_drafts:
                for idx, prompt in enumerate(prompts):
                    row = run_one_prompt(
                        decoder_name=d_name, decoder_fn=decoder_fn,
                        target=target, draft=draft, tokenizer=tokenizer,
                        prompt=prompt, lookahead=L, num_drafts=B,
                        max_length=max_new_tokens, seed=idx + 7,
                        base_key=base_key, plk=plk, vocab_size=vocab_size,
                        metrics_cfg=metrics_cfg,
                        anlppt_applies=anlppt_applies,
                    )
                    row["prompt_idx"] = idx
                    rows.append(row)
                    print(
                        f"  [{d_name} L={L} B={B}] p={idx} "
                        f"tok={row['tokens']} AATPS={row['AATPS']:.3f} "
                        f"TR={row['token_rate']:.2f}"
                        + (
                            f" U={row.get('ANLPPT_U', float('nan')):.3f}"
                            if "ANLPPT_U" in row else ""
                        )
                    )

    summary = _summarize(
        rows, decoders=decoders, lookaheads=lookaheads, num_drafts=num_drafts,
    )
    return {
        "config": config,
        "summary": summary,
        "rows": rows,
    }


def _summarize(rows, *, decoders, lookaheads, num_drafts):
    summary = {}
    for d in decoders:
        for L in lookaheads:
            for B in num_drafts:
                sub = [
                    r for r in rows
                    if r["decoder"] == d
                    and int(r["lookahead"]) == int(L)
                    and int(r["num_drafts"]) == int(B)
                ]
                if not sub:
                    continue
                agg = {}
                for k in (
                    "AATPS", "token_rate",
                    "ANLPPT_U", "ANLPPT_Li", "ANLPPT_PL", "ANLPPT_A",
                ):
                    vals = [r[k] for r in sub if k in r and not (
                        isinstance(r[k], float) and (np.isnan(r[k]) or np.isinf(r[k]))
                    )]
                    if vals:
                        agg[k] = float(np.mean(vals))
                summary[f"{d}_L{L}_B{B}"] = agg
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    config = S.load_config(args.config)
    out_path = (
        args.output if args.output is not None
        else S.resolve_path(config.get("output"))
    )
    if out_path is None:
        out_path = (
            S.ROOT / "outputs"
            / f"multi_draft_{int(time.time())}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_experiment(config)

    print("\n=== Summary ===")
    print(f"{'decoder_L_B':<36s} {'AATPS':>8s} {'TR':>10s} "
          f"{'U':>8s} {'Li':>8s} {'PL':>8s}")
    for key, agg in result["summary"].items():
        print(
            f"{key:<36s} {agg.get('AATPS', float('nan')):>8.3f} "
            f"{agg.get('token_rate', float('nan')):>10.2f} "
            f"{agg.get('ANLPPT_U', float('nan')):>8.3f} "
            f"{agg.get('ANLPPT_Li', float('nan')):>8.3f} "
            f"{agg.get('ANLPPT_PL', float('nan')):>8.3f}"
        )

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

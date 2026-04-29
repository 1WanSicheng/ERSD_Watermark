"""
Single-draft experiment runner: JSON-config driven.

Compares B=1 decoders (mc, basic_uwm, mc_uwm_speed, mc_uwm_strength, pfr,
pfr_no_watermark) at one or more lookahead lengths.  Reports AATPS,
token_rate, and ANLPPT-{U, Li, PL} per (decoder, lookahead).  KL/WS-ratio
is reported only for ``pfr`` (delta-conditional case under Definition 3.1).

Config schema (see experiments/configs/single_draft_default.json):

    {
      "experiment": "single_draft",
      "samples": 100,
      "dataset": "cnn_dailymail",
      "max_new_tokens": 128,
      "lookaheads": [2, 4, 6, 8],
      "private_key": "1234",
      "process_logits": {"top_k": 50, "top_p": 1.0, "temperature": 1.0},
      "decoders": ["mc", "basic_uwm", "mc_uwm_speed", "mc_uwm_strength",
                   "pfr", "pfr_no_watermark"],
      "metrics": {
          "aatps": true,
          "token_rate": true,
          "anlppt": {"variants": ["U", "Li", "PL"],
                     "li_delta": 0.5, "pl_eps": 0.1},
          "kl_ratio_pfr_only": true
      },
      "output": "outputs/single_draft.json"
    }

Decoders not subject to lookahead (basic_uwm) are run once per prompt and
copied to every lookahead row in the output table.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from . import _shared as S


def _decoder_uniforms(
    *, target, in_ids, out_ids, wm_kind, source_labels, masked_flags,
    private_key, vocab_size,
):
    """Return (Us, skipped) for ANLPPT scoring; None if no detection path."""
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
    if wm_kind == "none":
        # No watermark recovery path available; ANLPPT is not meaningful.
        return None
    raise ValueError(f"unknown wm_kind: {wm_kind}")


def run_one_prompt(
    *, decoder_name, decoder_fn, target, draft, tokenizer, prompt,
    lookahead, max_length, seed, base_key, plk, vocab_size,
    metrics_cfg: dict,
) -> dict:
    input_ids = S.encode_prompt(tokenizer, prompt, target.device)
    torch.manual_seed(seed)
    S._sync()
    t0 = time.perf_counter()
    out_ids, block_lens, pk, wm_kind, src_labels, masked_flags = decoder_fn(
        target, draft, input_ids,
        lookahead=lookahead, max_length=max_length,
        seed=seed, base_key=base_key, plk=plk,
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
        "tokens": n_tokens,
        "blocks": n_steps,
        "AATPS": float(aatps),
        "token_rate": float(token_rate),
        "elapsed_sec": float(elapsed),
        "wm_kind": wm_kind,
    }

    anlppt_cfg = (metrics_cfg or {}).get("anlppt") or {}
    if anlppt_cfg:
        urec = _decoder_uniforms(
            target=target, in_ids=input_ids, out_ids=out_ids.to(target.device),
            wm_kind=wm_kind, source_labels=src_labels,
            masked_flags=masked_flags, private_key=pk,
            vocab_size=vocab_size,
        )
        if urec is None:
            for v in anlppt_cfg.get("variants", ["U", "Li", "PL"]):
                row[f"ANLPPT_{v}"] = float("nan")
            row["n_skipped"] = 0
            row["n_added"] = 0
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

    if metrics_cfg.get("kl_ratio_pfr_only") and decoder_name == "pfr":
        full = torch.cat(
            [input_ids, out_ids.to(input_ids.device)], dim=1
        )
        kl = S.kl_ws_ratio_pfr(
            target_model=target, full_ids=full,
            prompt_length=int(input_ids.shape[1]),
            private_key=pk,
            process_logits_kwargs=plk,
        )
        row.update(kl)
    return row


def run_experiment(config: dict) -> dict:
    samples = int(config.get("samples", 100))
    dataset = str(config.get("dataset", "cnn_dailymail"))
    max_new_tokens = int(config.get("max_new_tokens", 128))
    lookaheads: List[int] = list(config.get("lookaheads", [4]))
    base_key = S.private_key_from_str(config.get("private_key", "1234"))
    plk = S.build_process_logits_kwargs(config.get("process_logits"))
    decoders: List[str] = list(
        config.get("decoders", list(S.SINGLE_DRAFT_DECODERS.keys()))
    )
    metrics_cfg: dict = config.get("metrics", {}) or {}

    target, draft, tokenizer, device = S.load_models_and_tokenizer(config)
    vocab_size = int(target.config.vocab_size)
    prompts = S.load_prompts(dataset, samples)

    print(
        f"[single_draft] dataset={dataset}  samples={samples}  "
        f"max_new={max_new_tokens}  lookaheads={lookaheads}  "
        f"decoders={decoders}"
    )

    rows: List[dict] = []
    # Plan: for each decoder, for each lookahead.  basic_uwm runs once per
    # prompt (lookahead-invariant) and is duplicated in the output table at
    # each requested lookahead so downstream aggregations are uniform.
    for d_name in decoders:
        if d_name not in S.SINGLE_DRAFT_DECODERS:
            print(f"  [warn] unknown decoder: {d_name}, skipped")
            continue
        decoder_fn = S.SINGLE_DRAFT_DECODERS[d_name]
        is_invariant = d_name in S.LOOKAHEAD_INVARIANT
        sweep = [lookaheads[0]] if is_invariant else lookaheads
        for L in sweep:
            for idx, prompt in enumerate(prompts):
                row = run_one_prompt(
                    decoder_name=d_name, decoder_fn=decoder_fn,
                    target=target, draft=draft, tokenizer=tokenizer,
                    prompt=prompt, lookahead=L,
                    max_length=max_new_tokens, seed=idx + 7,
                    base_key=base_key, plk=plk, vocab_size=vocab_size,
                    metrics_cfg=metrics_cfg,
                )
                row["prompt_idx"] = idx
                if is_invariant:
                    # Duplicate the row for each requested lookahead so
                    # plotting is uniform.
                    for L_actual in lookaheads:
                        rows.append({**row, "lookahead": int(L_actual)})
                else:
                    rows.append(row)
                print(
                    f"  [{d_name} L={L}] p={idx} tok={row['tokens']} "
                    f"AATPS={row['AATPS']:.3f} TR={row['token_rate']:.2f}"
                    + (
                        f" U={row.get('ANLPPT_U', float('nan')):.3f}"
                        if "ANLPPT_U" in row else ""
                    )
                    + (
                        f" KL={row.get('kl_ws_ratio', float('nan')):.3f}"
                        if "kl_ws_ratio" in row else ""
                    )
                )
            if is_invariant:
                # Ran once for the first lookahead; rows already duplicated.
                break

    summary = _summarize(rows, decoders=decoders, lookaheads=lookaheads)
    return {
        "config": config,
        "summary": summary,
        "rows": rows,
    }


def _summarize(rows: List[dict], *, decoders: List[str], lookaheads: List[int]):
    summary: Dict[str, Dict] = {}
    for d in decoders:
        for L in lookaheads:
            sub = [
                r for r in rows
                if r["decoder"] == d and int(r["lookahead"]) == int(L)
            ]
            if not sub:
                continue
            agg: Dict[str, float] = {}
            for k in (
                "AATPS", "token_rate",
                "ANLPPT_U", "ANLPPT_Li", "ANLPPT_PL", "ANLPPT_A",
                "kl_ws_ratio",
            ):
                vals = [r[k] for r in sub if k in r and not (
                    isinstance(r[k], float) and (np.isnan(r[k]) or np.isinf(r[k]))
                )]
                if vals:
                    agg[k] = float(np.mean(vals))
            summary[f"{d}_L{L}"] = agg
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None,
                    help="override output path from config")
    args = ap.parse_args()

    config = S.load_config(args.config)
    out_path = (
        args.output if args.output is not None
        else S.resolve_path(config.get("output"))
    )
    if out_path is None:
        out_path = (
            S.ROOT / "outputs"
            / f"single_draft_{int(time.time())}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_experiment(config)

    print("\n=== Summary ===")
    print(f"{'decoder_L':<32s} {'AATPS':>8s} {'TR':>10s} "
          f"{'U':>8s} {'Li':>8s} {'PL':>8s} {'KL':>8s}")
    for key, agg in result["summary"].items():
        print(
            f"{key:<32s} {agg.get('AATPS', float('nan')):>8.3f} "
            f"{agg.get('token_rate', float('nan')):>10.2f} "
            f"{agg.get('ANLPPT_U', float('nan')):>8.3f} "
            f"{agg.get('ANLPPT_Li', float('nan')):>8.3f} "
            f"{agg.get('ANLPPT_PL', float('nan')):>8.3f} "
            f"{agg.get('kl_ws_ratio', float('nan')):>8.3f}"
        )

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

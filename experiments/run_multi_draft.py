"""
Multi-draft experiment runner: JSON-config driven.

Compares B>=1 multi-draft decoders (mpfr_torchgen_cached, invariant_multi)
at one or more (B, lookahead) settings.
Reports AATPS and token_rate per (decoder, B, lookahead).  ANLPPT-{U, Li, PL}
is reported only for the PFR-family decoders configured in
``metrics.anlppt.applies_to``.

Config schema:

    {
      "experiment": "multi_draft",
      "samples": 100,
      "dataset": "cnn_dailymail",
      "max_new_tokens": 128,
      "lookaheads": [4],
      "num_drafts": [1, 2, 4, 8],
      "private_key": "1234",
      "process_logits": {"top_k": 50, "top_p": 1.0, "temperature": 1.0},
      "decoders": ["mpfr_torchgen_cached", "invariant_multi"],
      "metrics": {
          "aatps": true,
          "token_rate": true,
          "anlppt": {
              "variants": ["U", "Li", "PL"],
              "li_delta": 0.5, "pl_eps": 0.1,
              "applies_to": ["mpfr_torchgen_cached"]
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


# Map labeler mode strings to a *factory* that returns a per-prefix label
# callable.  The factory pattern is required for ``"context_code"`` (stateful
# repeated-context masking).  Stateless labelers ignore the per-prompt fresh
# closure but use the same factory shape for uniformity.
_LABEL_FN_FACTORY_BY_MODE = {
    "mpfr_direct":  lambda: S._mpfr_direct_label,
    "context_code": lambda: S.make_context_code_label_fn(n=3),
}


def _decoder_uniforms(
    *, target, in_ids, out_ids, wm_kind, source_labels, masked_flags,
    private_key, vocab_size,
    force_detector_kind: Optional[str] = None,
    force_labeler_mode: Optional[str] = None,
):
    """Recover per-token uniforms u_t for the watermark detector.

    By default dispatches by ``wm_kind`` (the generator's label).  For
    empirical-FPR experiments we need to score H0 (no-watermark) outputs with
    the H1 method's detector — pass ``force_detector_kind='PFR'`` together
    with ``force_labeler_mode`` (e.g. ``'mpfr_direct'``) to override.

    When labels were not emitted at gen time (mc / invariant_multi /
    pfr_no_watermark), they are reconstructed from the realized prefix using
    the chosen labeler — this is exactly what the gen-side decoder would have
    done, so detector u_t is identical to "what u_t would have been under PFR
    sampling with the same labeler".
    """
    detector_kind = force_detector_kind or wm_kind

    if detector_kind == "PFR":
        labels = source_labels
        n_out = int(out_ids.shape[-1])
        if (not labels or len(labels) != n_out) and force_labeler_mode is not None:
            factory = _LABEL_FN_FACTORY_BY_MODE.get(force_labeler_mode)
            if factory is None:
                return None
            label_fn = factory()  # fresh per-prompt closure (stateful for context_code)
            labels = S._rebuild_labels_for_prefix_scheme(in_ids, out_ids, label_fn)
            masked_flags = [False] * n_out
        if not labels or len(labels) != n_out:
            return None
        Us, skipped = S.uniforms_from_pfr(
            out_ids=out_ids, source_labels=labels,
            masked_flags=masked_flags or [False] * n_out,
            private_key=private_key, vocab_size=vocab_size,
        )
        return Us, skipped
    if detector_kind == "DeltaGumbel":
        Us, skipped = S.uniforms_from_dg(
            target=target, out_ids=out_ids, in_ids=in_ids,
            private_key=private_key,
        )
        return Us, skipped
    return None


_KL_LABELER_MODE_BY_DECODER = {
    # Multi-draft PFR variants — each uses a different per-context label scheme
    # at gen time, so post-hoc KL scoring must reconstruct the same per-context
    # PFR source via a matching labeler mode.
    "mpfr_torchgen_cached": "mpfr_direct",
}


def run_one_prompt(
    *, decoder_name, decoder_fn, target, draft, tokenizer, prompt,
    lookahead, num_drafts, max_length, seed, base_key, plk, vocab_size,
    metrics_cfg: dict, anlppt_applies: bool, use_chat_template: bool = True,
    lppl_applies: bool = False,
    tpr_applies: bool = False,
    kl_applies: bool = False,
    rouge_applies: bool = False,
    reference_text: Optional[str] = None,
    detector_spec: Optional[dict] = None,
) -> dict:
    input_ids = S.encode_prompt(
        tokenizer, prompt, target.device,
        use_chat_template=use_chat_template,
    )
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

    # Resolve any deferred label-rebuild sentinel from the decoder OUTSIDE
    # the timing window — this O(T) Python loop would otherwise inflate
    # the measured token-rate for PFR-family decoders.
    src_labels, masked_flags = S.finalize_labels(src_labels, masked_flags, out_ids)

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
    if (metrics_cfg or {}).get("save_output_ids"):
        # Used by drafter-invariance / cross-condition pairwise ROUGE-L.
        row["output_ids"] = [int(t) for t in out_ids[0].tolist()]

    Us = None
    skipped = None
    force_detector_kind = None
    force_labeler_mode = None
    if detector_spec:
        force_detector_kind = detector_spec.get("kind")
        force_labeler_mode = detector_spec.get("labeler")
        # If detector_spec is set, it overrides the natural wm_kind dispatch.
        # Annotate row so downstream analysis knows what scoring was applied.
        row["detector_kind"] = force_detector_kind or wm_kind
        if force_labeler_mode:
            row["detector_labeler"] = force_labeler_mode
    if anlppt_applies or tpr_applies:
        urec = _decoder_uniforms(
            target=target, in_ids=input_ids,
            out_ids=out_ids.to(target.device),
            wm_kind=wm_kind, source_labels=src_labels,
            masked_flags=masked_flags, private_key=pk,
            vocab_size=vocab_size,
            force_detector_kind=force_detector_kind,
            force_labeler_mode=force_labeler_mode,
        )
        if urec is not None:
            Us, skipped = urec

    if anlppt_applies:
        anlppt_cfg = (metrics_cfg or {}).get("anlppt") or {}
        if Us is None:
            for v in anlppt_cfg.get("variants", ["U", "Li", "PL"]):
                row[f"ANLPPT_{v}"] = float("nan")
        else:
            metrics = S.anlppt_metrics(
                Us,
                li_delta=float(anlppt_cfg.get("li_delta", 0.5)),
                pl_eps=float(anlppt_cfg.get("pl_eps", 0.1)),
                skipped=skipped,
                variants=list(anlppt_cfg.get("variants", ["U", "Li", "PL"])),
            )
            row.update(metrics)

    if tpr_applies and Us is not None:
        tpr_cfg = (metrics_cfg or {}).get("tpr_at_n") or {}
        anlppt_inner = (metrics_cfg or {}).get("anlppt") or {}
        t_raw = tpr_cfg.get("tokens", 64)
        T_list = ([int(t_raw)] if isinstance(t_raw, (int, float))
                  else [int(t) for t in t_raw])
        for T_n in T_list:
            det = S.detector_at_first_n(
                Us, skipped,
                n_tokens=T_n,
                fpr=float(tpr_cfg.get("fpr", 0.01)),
                li_delta=float(tpr_cfg.get("li_delta",
                    anlppt_inner.get("li_delta", 0.5))),
                pl_eps=float(tpr_cfg.get("pl_eps",
                    anlppt_inner.get("pl_eps", 0.1))),
                variants=list(tpr_cfg.get("variants", ["U", "Li", "PL"])),
            )
            row.update(det)

    if lppl_applies:
        full = torch.cat([input_ids, out_ids.to(input_ids.device)], dim=1)
        row.update(S.lppl_under_target(
            target_model=target, full_ids=full,
            prompt_length=int(input_ids.shape[1]),
            process_logits_kwargs=plk,
        ))

    if kl_applies and decoder_name in _KL_LABELER_MODE_BY_DECODER:
        full = torch.cat([input_ids, out_ids.to(input_ids.device)], dim=1)
        row.update(S.kl_ws_ratio_pfr(
            target_model=target, full_ids=full,
            prompt_length=int(input_ids.shape[1]),
            private_key=pk,
            process_logits_kwargs=plk,
            labeler_mode=_KL_LABELER_MODE_BY_DECODER[decoder_name],
        ))

    if rouge_applies and reference_text:
        try:
            row["ROUGE_L_vs_ref"] = float(
                S.rouge_l_against_reference(
                    tokenizer=tokenizer,
                    output_ids=[int(t) for t in out_ids[0].tolist()],
                    reference_text=str(reference_text),
                )
            )
        except Exception:
            row["ROUGE_L_vs_ref"] = float("nan")
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
    lppl_applies_to = set(
        ((metrics_cfg.get("lppl") or {}).get("applies_to") or [])
    )
    tpr_applies_to = set(
        ((metrics_cfg.get("tpr_at_n") or {}).get("applies_to") or [])
    )
    kl_applies_to = set(
        ((metrics_cfg.get("kl_ratio") or {}).get("applies_to") or [])
    )
    # detector_per_decoder: optional override telling the post-hoc detector
    # which (kind, labeler) to use to recover u_t from a generator's output.
    # Required for empirical-FPR experiments where, e.g., we generate with
    # `invariant_multi` (no watermark, our H0) but want to score the output
    # with the SAME detector used for `mpfr_torchgen_cached` H1.
    detector_per_decoder: dict = (
        (metrics_cfg.get("tpr_at_n") or {}).get("detector_per_decoder") or {}
    )
    rouge_applies_to = set(
        ((metrics_cfg.get("rouge_vs_reference") or {}).get("applies_to") or [])
    )

    target, draft, tokenizer, device = S.load_models_and_tokenizer(config)
    vocab_size = int(target.config.vocab_size)
    prompts = S.load_prompts(dataset, samples)
    references = S.load_references(dataset, samples) if rouge_applies_to else None
    use_chat_template = bool(config.get("use_chat_template", True))

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
        lppl_applies = d_name in lppl_applies_to
        tpr_applies = d_name in tpr_applies_to
        kl_applies = d_name in kl_applies_to
        rouge_applies = d_name in rouge_applies_to
        detector_spec = detector_per_decoder.get(d_name)
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
                        rouge_applies=rouge_applies,
                        reference_text=(references[idx] if references else None),
                        anlppt_applies=anlppt_applies,
                        lppl_applies=lppl_applies,
                        tpr_applies=tpr_applies,
                        kl_applies=kl_applies,
                        detector_spec=detector_spec,
                        use_chat_template=use_chat_template,
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
                # Mean over numeric scalar fields, ignoring NaN/Inf.  We collect
                # the union of metric keys present in any row of the cell so
                # newly added metrics (LPPL, det_at_*, log_p_at_*, kl_ws_*) get
                # aggregated automatically without listing them explicitly.
                metric_keys: set = set()
                for r in sub:
                    metric_keys.update(r.keys())
                metric_keys -= {
                    "decoder", "lookahead", "num_drafts", "wm_kind", "prompt_idx",
                }
                for k in sorted(metric_keys):
                    vals: list = []
                    for r in sub:
                        v = r.get(k)
                        if v is None:
                            continue
                        if isinstance(v, bool):
                            vals.append(int(v))
                            continue
                        if isinstance(v, (int, float)) and not (
                            isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                        ):
                            vals.append(float(v))
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
    print(f"{'decoder_L_B':<36s} {'AATPS':>7s} {'TR':>7s} "
          f"{'U':>7s} {'Li':>7s} {'PL':>7s} "
          f"{'LPPL':>7s} {'TPR_U':>7s} {'KLrat':>7s}")
    for key, agg in result["summary"].items():
        det_keys = sorted(k for k in agg if k.startswith("det_at_") and k.endswith("_U"))
        tpr_u = agg.get(det_keys[0]) if det_keys else float("nan")
        if not isinstance(tpr_u, float):
            tpr_u = float("nan")
        print(
            f"{key:<36s} {agg.get('AATPS', float('nan')):>7.3f} "
            f"{agg.get('token_rate', float('nan')):>7.2f} "
            f"{agg.get('ANLPPT_U', float('nan')):>7.3f} "
            f"{agg.get('ANLPPT_Li', float('nan')):>7.3f} "
            f"{agg.get('ANLPPT_PL', float('nan')):>7.3f} "
            f"{agg.get('LPPL', float('nan')):>7.3f} "
            f"{tpr_u:>7.3f} "
            f"{agg.get('kl_ws_ratio', float('nan')):>7.3f}"
        )

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

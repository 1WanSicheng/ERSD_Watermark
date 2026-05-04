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


# Factory pattern for label functions (stateful for context_code repeated-
# context masking; stateless labelers fit the same shape).
_LABEL_FN_FACTORY_BY_MODE = {
    "prefix":       lambda: S._prefix_labeler_label,
    "mpfr_direct":  lambda: S._mpfr_direct_label,
    "context_code": lambda: S.make_context_code_label_fn(n=3),
}


def _decoder_uniforms(
    *, target, in_ids, out_ids, wm_kind, source_labels, masked_flags,
    private_key, vocab_size,
    force_detector_kind: Optional[str] = None,
    force_labeler_mode: Optional[str] = None,
):
    """Return (Us, skipped) for ANLPPT scoring; None if no detection path.

    For empirical-FPR experiments, set ``force_detector_kind`` (and
    ``force_labeler_mode`` for PFR) to score H0 outputs (e.g. ``mc``,
    ``pfr_no_watermark``) with the H1 method's detector.  When labels were
    not emitted by the gen-side decoder, they are reconstructed from the
    realized prefix using the configured labeler.
    """
    detector_kind = force_detector_kind or wm_kind

    if detector_kind == "PFR":
        labels = source_labels
        n_out = int(out_ids.shape[-1])
        if (not labels or len(labels) != n_out) and force_labeler_mode is not None:
            factory = _LABEL_FN_FACTORY_BY_MODE.get(force_labeler_mode)
            if factory is None:
                return None
            label_fn = factory()  # fresh per-prompt closure
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
    if detector_kind == "none":
        return None
    raise ValueError(f"unknown detector_kind: {detector_kind}")


# Per-decoder labeler mode used by post-hoc KL/WS scoring (must match the
# gen-side scheme so the detector reconstructs the same per-context PFR
# source).  Single-draft PFR was switched from "context_code" to "prefix"
# (see _run_pfr docstring), so KL scoring follows.
_KL_LABELER_MODE_BY_DECODER = {
    "pfr": "prefix",
}


def run_one_prompt(
    *, decoder_name, decoder_fn, target, draft, tokenizer, prompt,
    lookahead, max_length, seed, base_key, plk, vocab_size,
    metrics_cfg: dict, use_chat_template: bool = True,
    lppl_applies: bool = False,
    tpr_applies: bool = False,
    kl_applies: bool = False,
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
    )
    S._sync()
    elapsed = time.perf_counter() - t0

    # Resolve any deferred label-rebuild sentinel from the decoder OUTSIDE
    # the timing window (only multi-draft PFR decoders defer; single-draft
    # decoders already return real labels and pass through unchanged).
    src_labels, masked_flags = S.finalize_labels(src_labels, masked_flags, out_ids)

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
    Us = None
    skipped = None
    force_detector_kind = None
    force_labeler_mode = None
    if detector_spec:
        force_detector_kind = detector_spec.get("kind")
        force_labeler_mode = detector_spec.get("labeler")
        row["detector_kind"] = force_detector_kind or wm_kind
        if force_labeler_mode:
            row["detector_labeler"] = force_labeler_mode
    if anlppt_cfg or tpr_applies:
        urec = _decoder_uniforms(
            target=target, in_ids=input_ids, out_ids=out_ids.to(target.device),
            wm_kind=wm_kind, source_labels=src_labels,
            masked_flags=masked_flags, private_key=pk,
            vocab_size=vocab_size,
            force_detector_kind=force_detector_kind,
            force_labeler_mode=force_labeler_mode,
        )
        if urec is not None:
            Us, skipped = urec

    if anlppt_cfg:
        if Us is None:
            for v in anlppt_cfg.get("variants", ["U", "Li", "PL"]):
                row[f"ANLPPT_{v}"] = float("nan")
            row["n_skipped"] = 0
            row["n_added"] = 0
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
        det = S.detector_at_first_n(
            Us, skipped,
            n_tokens=int(tpr_cfg.get("tokens", 64)),
            fpr=float(tpr_cfg.get("fpr", 0.01)),
            li_delta=float(tpr_cfg.get("li_delta",
                anlppt_cfg.get("li_delta", 0.5))),
            pl_eps=float(tpr_cfg.get("pl_eps",
                anlppt_cfg.get("pl_eps", 0.1))),
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

    # Legacy hook: still honour kl_ratio_pfr_only for pfr decoder.
    if metrics_cfg.get("kl_ratio_pfr_only") and decoder_name == "pfr":
        full = torch.cat([input_ids, out_ids.to(input_ids.device)], dim=1)
        row.update(S.kl_ws_ratio_pfr(
            target_model=target, full_ids=full,
            prompt_length=int(input_ids.shape[1]),
            private_key=pk,
            process_logits_kwargs=plk,
        ))
    elif kl_applies and decoder_name in _KL_LABELER_MODE_BY_DECODER:
        full = torch.cat([input_ids, out_ids.to(input_ids.device)], dim=1)
        row.update(S.kl_ws_ratio_pfr(
            target_model=target, full_ids=full,
            prompt_length=int(input_ids.shape[1]),
            private_key=pk,
            process_logits_kwargs=plk,
            labeler_mode=_KL_LABELER_MODE_BY_DECODER[decoder_name],
        ))
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
    lppl_applies_to = set(
        ((metrics_cfg.get("lppl") or {}).get("applies_to") or [])
    )
    tpr_applies_to = set(
        ((metrics_cfg.get("tpr_at_n") or {}).get("applies_to") or [])
    )
    kl_applies_to = set(
        ((metrics_cfg.get("kl_ratio") or {}).get("applies_to") or [])
    )
    # detector_per_decoder: optional override mapping decoder name to a
    # detector spec ``{"kind": "PFR"|"DeltaGumbel", "labeler": <mode>}``,
    # used for empirical-FPR experiments where H0 outputs (mc /
    # pfr_no_watermark) get scored with the H1 method's detector.
    detector_per_decoder: dict = (
        (metrics_cfg.get("tpr_at_n") or {}).get("detector_per_decoder") or {}
    )

    target, draft, tokenizer, device = S.load_models_and_tokenizer(config)
    vocab_size = int(target.config.vocab_size)
    prompts = S.load_prompts(dataset, samples)
    use_chat_template = bool(config.get("use_chat_template", True))

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
        lppl_applies = d_name in lppl_applies_to
        tpr_applies = d_name in tpr_applies_to
        kl_applies = d_name in kl_applies_to
        detector_spec = detector_per_decoder.get(d_name)
        for L in sweep:
            for idx, prompt in enumerate(prompts):
                row = run_one_prompt(
                    decoder_name=d_name, decoder_fn=decoder_fn,
                    target=target, draft=draft, tokenizer=tokenizer,
                    prompt=prompt, lookahead=L,
                    max_length=max_new_tokens, seed=idx + 7,
                    base_key=base_key, plk=plk, vocab_size=vocab_size,
                    metrics_cfg=metrics_cfg,
                    lppl_applies=lppl_applies,
                    tpr_applies=tpr_applies,
                    kl_applies=kl_applies,
                    detector_spec=detector_spec,
                    use_chat_template=use_chat_template,
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
            metric_keys: set = set()
            for r in sub:
                metric_keys.update(r.keys())
            metric_keys -= {
                "decoder", "lookahead", "wm_kind", "prompt_idx",
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
    print(f"{'decoder_L':<32s} {'AATPS':>7s} {'TR':>7s} "
          f"{'U':>7s} {'Li':>7s} {'PL':>7s} "
          f"{'LPPL':>7s} {'TPR_U':>7s} {'KLrat':>7s}")
    for key, agg in result["summary"].items():
        det_keys = sorted(k for k in agg if k.startswith("det_at_") and k.endswith("_U"))
        tpr_u = agg.get(det_keys[0]) if det_keys else float("nan")
        if not isinstance(tpr_u, float):
            tpr_u = float("nan")
        print(
            f"{key:<32s} {agg.get('AATPS', float('nan')):>7.3f} "
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

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

import unbiased_watermark as uwm

from accuwm.mc import mc_sample_generator
from accuwm.mc_watermark import mc_uwm_sample_generator
from accuwm.multi_draft_pfr import multi_draft_pfr_sample_generator
from accuwm.pfr import PrefixLabeler, build_default_labeler, pfr_sample_generator
from accuwm.pfr_no_watermark import pfr_no_watermark_generator
from unbiased_watermark.scores.pfr_aaronson import (
    PFR_Aaronson_U_Score,
    _uniform_for_token,
)


DEFAULT_TARGET_MODEL = ROOT / "model" / "Qwen2.5-7B-Instruct"
DEFAULT_DRAFT_MODEL = ROOT / "model" / "Qwen2.5-0.5B-Instruct"
DEFAULT_PRIVATE_KEY = b"1234"


def load_json_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def private_key_from_config(config: dict) -> bytes:
    value = config.get("private_key", "1234")
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, list):
        return bytes(value)
    return DEFAULT_PRIVATE_KEY


def seeded_private_key(seed: int, private_key: bytes) -> bytes:
    return int(seed).to_bytes(8, "big", signed=False) + private_key


def build_reweight(name: str):
    if name == "deltagumbel":
        return uwm.DeltaGumbel_Reweight()
    if name == "gamma":
        return uwm.Gamma_Reweight()
    raise ValueError(f"unknown reweight: {name}")


def load_model(model_path: Path, device: str):
    kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "device_map": device,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16
    return transformers.AutoModelForCausalLM.from_pretrained(**kwargs)


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
        return [
            {
                "idx": idx,
                "prompt": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": row["question"]},
                ],
                "reference": row["answer"],
            }
            for idx, row in enumerate(ds)
        ]
    if dataset == "summarization":
        ds = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)["test"]
        ds = ds.filter(lambda x: len(x["article"]) < 3000)
        ds = ds.select(range(samples))
        return [
            {
                "idx": idx,
                "prompt": (
                    "System:Summarize the following article.\n"
                    f"INPUT:{row['article'][:1000]}\nOUTPUT:"
                ),
                "reference": row["highlights"],
            }
            for idx, row in enumerate(ds)
        ]
    raise ValueError(f"unknown dataset: {dataset}")


def canonical_algorithm_type(spec: dict) -> str:
    algo_type = spec.get("type", spec.get("algorithm", spec.get("name")))
    aliases = {
        "pfr_no_watermark": "pfr_nowatermark",
        "pfr-nowatermark": "pfr_nowatermark",
        "multi_draft_pfr": "mspfr",
        "mc_uwm_strength": "uwm_strength",
        "mc_uwm_speed": "uwm_speed",
        "basic_speulative_decoding": "basic_speculative_decoding",
        "basic_specdec": "basic_speculative_decoding",
        "speculative_decoding": "basic_speculative_decoding",
    }
    return aliases.get(algo_type, algo_type)


def build_models_and_tokenizer(config: dict):
    target_model = resolve_path(config.get("target_model"), DEFAULT_TARGET_MODEL)
    draft_model = resolve_path(config.get("draft_model"), DEFAULT_DRAFT_MODEL)
    target_device = config.get("target_device", "cuda:0" if torch.cuda.is_available() else "cpu")
    draft_device = config.get("draft_device") or target_device
    target = load_model(target_model, target_device)
    draft = load_model(draft_model, draft_device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(target_model),
        local_files_only=True,
    )
    return target, draft, tokenizer


def iter_algorithm_specs(config: dict) -> list[dict]:
    specs = config.get("algorithms")
    if not specs:
        raise ValueError("config must define a non-empty 'algorithms' list")
    normalized = []
    for spec in specs:
        if isinstance(spec, str):
            spec = {"name": spec, "type": spec}
        spec = dict(spec)
        spec.setdefault("name", spec.get("type", spec.get("algorithm")))
        spec["type"] = canonical_algorithm_type(spec)
        normalized.append(spec)
    return normalized


def build_generator(
    spec: dict,
    target,
    draft,
    input_ids,
    lookahead: int,
    max_length: int,
    seed: int,
    private_key: bytes,
    process_logits_kwargs: dict | None = None,
):
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    algo_type = spec["type"]
    if algo_type == "pfr_nowatermark":
        return pfr_no_watermark_generator(
            model=target,
            ref_model=draft,
            input_ids=input_ids,
            n=lookahead,
            max_length=max_length,
            process_logits_kwargs=process_logits_kwargs,
        )
    if algo_type == "pfr":
        labeler_mode = spec.get("labeler_mode", "prefix")
        return pfr_sample_generator(
            model=target,
            ref_model=draft,
            input_ids=input_ids,
            n=lookahead,
            max_length=max_length,
            private_key=private_key,
            labeler_mode=labeler_mode,
            process_logits_kwargs=process_logits_kwargs,
        )
    if algo_type == "mspfr":
        return multi_draft_pfr_sample_generator(
            model=target,
            ref_model=draft,
            input_ids=input_ids,
            n=lookahead,
            B=int(spec.get("num_drafts", spec.get("B", 4))),
            max_length=max_length,
            private_key=private_key,
            labeler=PrefixLabeler(),
            return_meta=True,
            process_logits_kwargs=process_logits_kwargs,
        )
    if algo_type == "basic_speculative_decoding":
        return mc_sample_generator(
            model=target,
            ref_model=draft,
            input_ids=input_ids,
            n=lookahead,
            process_logits_kwargs=process_logits_kwargs,
        )
    if algo_type in {"uwm_strength", "uwm_speed"}:
        reweight = build_reweight(spec.get("reweight", "deltagumbel"))
        cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=int(spec.get("context_width", 3)))
        cch = uwm.lm.ContextCodeHistory(batch_shape=(1,))
        return mc_uwm_sample_generator(
            reweight=reweight,
            cc_extractor=cc_extractor,
            cch=cch,
            private_key=seeded_private_key(seed, private_key),
            reweight_in_mc=(algo_type == "uwm_strength"),
            model=target,
            ref_model=draft,
            input_ids=input_ids,
            n=lookahead,
            **process_logits_kwargs,
        )
    raise ValueError(f"unsupported algorithm type: {algo_type}")


def pfr_u_from_source_labels(
    output_ids: torch.LongTensor,
    source_labels: list[bytes],
    private_key: bytes,
    vocab_size: int,
) -> dict:
    if output_ids.numel() == 0:
        return {"ANLPPT_U": 0.0, "log_p_value_u": 0.0, "detector_u": "PFR_Aaronson_U"}
    skipped = np.zeros((1, len(source_labels)), dtype=bool)
    labels = np.array(source_labels, dtype=object)[None, :]
    score = PFR_Aaronson_U_Score.from_watermarkcode(
        None,
        output_ids,
        skipped=skipped,
        source_labels=labels,
        private_key=private_key,
        vocab_size=vocab_size,
    )
    log_p_value = float(score.get_log_p_value())
    return {
        "ANLPPT_U": float(-log_p_value / max(int(output_ids.numel()), 1)),
        "log_p_value_u": log_p_value,
        "detector_u": "PFR_Aaronson_U",
    }


def pfr_u_from_sequence_prefix(
    full_ids: torch.LongTensor,
    prompt_length: int,
    private_key: bytes,
    vocab_size: int,
) -> dict:
    labeler = PrefixLabeler()
    scores = []
    for pos in range(prompt_length, full_ids.shape[1]):
        context = full_ids[:, :pos]
        token_id = int(full_ids[0, pos].item())
        label = labeler.label(context)
        r_t = _uniform_for_token(label, private_key, token_id, vocab_size)
        scores.append(-np.log(1.0 - r_t))
    if not scores:
        return {"ANLPPT_U": 0.0, "log_p_value_u": 0.0, "detector_u": "PFR_Aaronson_U"}
    score = PFR_Aaronson_U_Score(
        scores=np.array(scores, dtype=np.float32)[None, :],
        skipped=np.zeros((1, len(scores)), dtype=bool),
    )
    log_p_value = float(score.get_log_p_value())
    return {
        "ANLPPT_U": float(-log_p_value / max(len(scores), 1)),
        "log_p_value_u": log_p_value,
        "detector_u": "PFR_Aaronson_U",
    }


def uwm_u_score(
    spec: dict,
    input_ids: torch.LongTensor,
    output_ids: torch.LongTensor,
    output_logprobs: torch.FloatTensor,
    seed: int,
    private_key: bytes,
) -> dict:
    if output_ids.numel() == 0:
        return {"ANLPPT_U": 0.0, "log_p_value_u": 0.0, "detector_u": "DeltaGumbel_U"}
    reweight = build_reweight(spec.get("reweight", "deltagumbel"))
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=int(spec.get("context_width", 3)))
    cch_detect = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    score = uwm.lm.detect(
        vocab_size=output_logprobs.shape[-1],
        score_type=uwm.scores.DeltaGumbel_U,
        reweight=reweight,
        cc_extractor=cc_extractor,
        cch=cch_detect,
        private_key=seeded_private_key(seed, private_key),
        out_ids=output_ids,
        in_ids=input_ids,
        p_logits=output_logprobs,
    )
    log_p_value = float(score.get_log_p_value())
    return {
        "ANLPPT_U": float(-log_p_value / max(int(output_ids.numel()), 1)),
        "log_p_value_u": log_p_value,
        "detector_u": "DeltaGumbel_U",
    }


@torch.no_grad()
def run_generation(
    spec: dict,
    target,
    draft,
    tokenizer,
    prompt,
    reference,
    lookahead: int,
    max_length: int,
    seed: int,
    private_key: bytes,
    include_quality: bool,
    include_u_score: bool,
):
    torch.manual_seed(seed)
    input_ids = encode_prompt(tokenizer, prompt, target.device)
    gen = build_generator(
        spec=spec,
        target=target,
        draft=draft,
        input_ids=input_ids,
        lookahead=lookahead,
        max_length=max_length,
        seed=seed,
        private_key=private_key,
    )

    output_chunks = []
    logprob_chunks = []
    token_logprobs = []
    block_lens = []
    accepted_counts = []
    draft_tree_sizes = []
    target_context_counts = []
    source_labels = []
    generated = 0
    start = time.perf_counter()
    for step in gen:
        if len(step) == 3:
            step_output_ids, step_output_logprobs, meta = step
        else:
            step_output_ids, step_output_logprobs = step
            meta = {}
        block_len = step_output_ids.shape[1]
        if generated + block_len > max_length:
            block_len = max_length - generated
        if block_len <= 0:
            break

        ids = step_output_ids[:, :block_len]
        logprobs = step_output_logprobs[:, :block_len, :]
        gathered = torch.gather(logprobs, dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
        output_chunks.append(ids.detach().cpu())
        logprob_chunks.append(logprobs.detach().cpu())
        token_logprobs.extend(float(x) for x in gathered[0].detach().cpu().tolist())
        block_lens.append(int(block_len))
        if "accepted_count" in meta:
            accepted_counts.append(int(meta["accepted_count"]))
        if "draft_tree_size" in meta:
            draft_tree_sizes.append(float(meta["draft_tree_size"]))
        if "target_context_count" in meta:
            target_context_counts.append(float(meta["target_context_count"]))
        if "labels" in meta:
            source_labels.extend([label.source_label for label in meta["labels"][:block_len]])

        generated += block_len
        if generated >= max_length:
            break
    elapsed = time.perf_counter() - start

    output_ids_cpu = (
        torch.cat(output_chunks, dim=1)
        if output_chunks
        else torch.empty((1, 0), dtype=torch.long)
    )
    output_logprobs_cpu = (
        torch.cat(logprob_chunks, dim=1)
        if logprob_chunks
        else torch.empty((1, 0, target.config.vocab_size), dtype=torch.float32)
    )
    output_ids = output_ids_cpu.to(target.device)
    output_logprobs = output_logprobs_cpu.to(target.device)
    num_tokens = int(output_ids_cpu.shape[1])
    num_steps = len(block_lens)
    aatps = float(num_tokens / num_steps) if num_steps else 0.0
    token_rate = float(num_tokens / elapsed) if elapsed > 0 else 0.0
    log_perplexity = -float(np.mean(token_logprobs)) if token_logprobs else 0.0

    row = {
        "algorithm": spec["name"],
        "algorithm_type": spec["type"],
        "num_steps": num_steps,
        "num_invocations": num_steps,
        "num_tokens": num_tokens,
        "AATPS": aatps,
        "BE": aatps,
        "TR": token_rate,
        "token_rate": token_rate,
        "elapsed_sec": float(elapsed),
        "total_time": float(elapsed),
        "accepted_draft_mean": float(np.mean(accepted_counts)) if accepted_counts else 0.0,
        "draft_tree_size_mean": float(np.mean(draft_tree_sizes)) if draft_tree_sizes else 0.0,
        "target_context_count_mean": (
            float(np.mean(target_context_counts)) if target_context_counts else 0.0
        ),
    }

    if include_u_score:
        if spec["type"] in {"uwm_strength", "uwm_speed"}:
            row.update(
                uwm_u_score(
                    spec=spec,
                    input_ids=input_ids,
                    output_ids=output_ids,
                    output_logprobs=output_logprobs,
                    seed=seed,
                    private_key=private_key,
                )
            )
        elif source_labels and len(source_labels) == num_tokens:
            row.update(
                pfr_u_from_source_labels(
                    output_ids=output_ids,
                    source_labels=source_labels,
                    private_key=private_key,
                    vocab_size=target.config.vocab_size,
                )
            )
        else:
            full_ids = torch.cat([input_ids.to(output_ids.device), output_ids], dim=1)
            row.update(
                pfr_u_from_sequence_prefix(
                    full_ids=full_ids,
                    prompt_length=input_ids.shape[1],
                    private_key=private_key,
                    vocab_size=target.config.vocab_size,
                )
            )

    if include_quality:
        generation = tokenizer.decode(output_ids_cpu[0], skip_special_tokens=True)
        row.update(
            {
                "log_perplexity": log_perplexity,
                "perplexity": float(math.exp(min(log_perplexity, 50.0)))
                if token_logprobs
                else 0.0,
                **text_quality_metrics(generation, reference),
                "prediction": generation,
                "reference": reference,
            }
        )

    return row


def aggregate_rows(rows: list[dict], metric_keys: list[str]) -> dict:
    total_tokens = int(sum(row.get("num_tokens", 0) for row in rows))
    total_time = float(sum(row.get("total_time", 0.0) for row in rows))
    summary = {
        "num_samples": len(rows),
        "num_tokens_total": total_tokens,
        "num_invocations_total": int(sum(row.get("num_invocations", 0) for row in rows)),
        "TR_global": float(total_tokens / total_time) if total_time > 0 else 0.0,
        "elapsed_sec_total": total_time,
        "total_time_total": total_time,
    }
    for key in metric_keys:
        values = [row[key] for row in rows if key in row and row[key] is not None]
        if not values:
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def run_configured_experiment(config: dict, *, include_quality: bool, include_u_score: bool):
    private_key = private_key_from_config(config)
    seed = int(config.get("seed", 1))
    if "lookaheads" in config:
        lookaheads = [int(x) for x in config["lookaheads"]]
    elif "draft_lengths" in config:
        lookaheads = [int(x) for x in config["draft_lengths"]]
    else:
        lookaheads = [int(config.get("lookahead", config.get("draft_length", 4)))]
    max_length = int(config.get("max_length", 128))
    samples = int(config.get("samples", 20))
    dataset = config.get("dataset", "gsm8k")
    progress_every = int(config.get("progress_every", 5))
    warmup = int(config.get("warmup", 0))

    target, draft, tokenizer = build_models_and_tokenizer(config)
    eval_rows = load_eval_rows(dataset, samples)
    algorithms = iter_algorithm_specs(config)

    if warmup > 0:
        for row in eval_rows[:warmup]:
            run_generation(
                algorithms[0],
                target,
                draft,
                tokenizer,
                row["prompt"],
                row["reference"],
                lookaheads[0],
                max_length,
                seed,
                private_key,
                include_quality=False,
                include_u_score=False,
            )

    rows_output = resolve_path(config.get("rows_output"))
    output = resolve_path(config.get("output"))
    if rows_output is not None:
        rows_output.parent.mkdir(parents=True, exist_ok=True)
        rows_output.write_text("", encoding="utf-8")

    metric_keys = ["AATPS", "BE", "TR", "token_rate", "accepted_draft_mean"]
    if include_u_score:
        metric_keys.extend(["ANLPPT_U", "log_p_value_u"])
    if include_quality:
        metric_keys.extend(
            [
                "log_perplexity",
                "perplexity",
                "rouge1_f1",
                "rouge2_f1",
                "rougeL_f1",
                "bleu",
            ]
        )

    results = {}
    all_rows = []
    for lookahead in lookaheads:
        for spec in algorithms:
            rows = []
            result_key = f"{spec['name']}_L{lookahead}"
            for idx, eval_row in enumerate(eval_rows):
                sample_start = time.perf_counter()
                row = run_generation(
                    spec,
                    target,
                    draft,
                    tokenizer,
                    eval_row["prompt"],
                    eval_row["reference"],
                    lookahead,
                    max_length,
                    seed + idx,
                    private_key,
                    include_quality=include_quality,
                    include_u_score=include_u_score,
                )
                row = {
                    "sample_idx": idx,
                    "dataset_idx": eval_row["idx"],
                    "lookahead": lookahead,
                    "max_length": max_length,
                    **row,
                }
                rows.append(row)
                all_rows.append(row)
                if rows_output is not None:
                    with rows_output.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if progress_every > 0 and (
                    (idx + 1) % progress_every == 0 or idx + 1 == len(eval_rows)
                ):
                    print(
                        json.dumps(
                            {
                                "progress": {
                                    "algorithm": spec["name"],
                                    "lookahead": lookahead,
                                    "done": idx + 1,
                                    "total": len(eval_rows),
                                    "last_sample_sec": time.perf_counter() - sample_start,
                                    "partial_summary": aggregate_rows(rows, metric_keys),
                                }
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            results[result_key] = {
                "spec": spec,
                "lookahead": lookahead,
                "summary": aggregate_rows(rows, metric_keys),
                "rows": rows,
            }

    payload = {
        "dataset": dataset,
        "samples": samples,
        "lookaheads": lookaheads,
        "max_length": max_length,
        "seed": seed,
        "target_model": str(resolve_path(config.get("target_model"), DEFAULT_TARGET_MODEL)),
        "draft_model": str(resolve_path(config.get("draft_model"), DEFAULT_DRAFT_MODEL)),
        "target_device": config.get("target_device", "cuda:0" if torch.cuda.is_available() else "cpu"),
        "draft_device": config.get("draft_device")
        or config.get("target_device", "cuda:0" if torch.cuda.is_available() else "cpu"),
        "include_quality": include_quality,
        "include_u_score": include_u_score,
        "results": results,
        "rows": all_rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return payload

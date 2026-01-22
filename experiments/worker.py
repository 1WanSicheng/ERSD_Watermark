import ray

import torch
import numpy as np
import transformers
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from functools import partial
import time

import accuwm
import unbiased_watermark as uwm


class MaxLengthLogitsProcessor(transformers.LogitsProcessor):
    def __init__(self, max_length, eos_token_id):
        self.input_length = 0
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] > self.max_length + self.input_length:
            scores = torch.full_like(scores, -float("inf"))
            scores[:, self.eos_token_id] = 0.0
        return scores


class StopWordsLogitsProcessor(transformers.LogitsProcessor):
    def __init__(self, stop_words_ids: list[any], eos_token_id: int):
        # stop_words_ids: list of input_ids (shape: (seq_len,))
        self.stop_words_ids = stop_words_ids
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores) -> torch.FloatTensor:
        stopped = False
        for stop_token_seq in self.stop_words_ids:
            if torch.equal(input_ids[0, -len(stop_token_seq) :], stop_token_seq):
                stopped = True
                break
        if stopped:
            scores = torch.full_like(scores, -float("inf"))
            scores[:, self.eos_token_id] = 0.0
        return scores

    def _to_ids(self, stop_word: str, tokenizer):
        ids = tokenizer(stop_word, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0, :]
        if "llama" in tokenizer.name_or_path:
            # Remove SPIECE_UNDERLINE
            while ids.shape[-1] > 0 and ids[0] == 29871:
                ids = ids[1:]
        return ids

    def set_stop_words(self, stop_words: list[str], tokenizer, device="cpu"):
        self.stop_words_ids = [
            self._to_ids(stop_word, tokenizer).to(device) for stop_word in stop_words
        ]


class DummyWorker:
    def __init__(self, param):
        self.param = param

    @ray.method(concurrency_group="model")
    def __call__(self, d):
        return d


def safe_ln(x):
    x = np.array(x)
    with np.errstate(divide="ignore"):
        return np.where(x <= 0, -np.inf, np.log(x))


scores = {
    "deltagumbel": [
        uwm.scores.DeltaGumbel_C,
        uwm.scores.DeltaGumbel_U,
        uwm.scores.LLRScore,
        uwm.scores.RobustLLRScore.builder(
            safe_ln([(0, a) for a in np.linspace(0, 0.9, 10)]).tolist()
        ),
    ],
    "gamma": [
        uwm.scores.Gamma_U,
        uwm.scores.LLRScore,
        uwm.scores.RobustLLRScore.builder(
            safe_ln([(0, a) for a in np.linspace(0, 0.9, 10)]).tolist()
        ),
    ],
    "ersd": [
        uwm.scores.ERSD_Aaronson_Gamma_Score,
        uwm.scores.ERSD_Aaronson_U_Score,
    ],
}
score_strs = {
    "deltagumbel": [
        "DeltaGumbel_C",
        "DeltaGumbel_U",
        "LLR",
        "RobustLLR",
    ],
    "gamma": [
        "Gamma_U",
        "LLR",
        "RobustLLR",
    ],
    "ersd": [
        "ERSD_Aaronson_Gamma",
        "ERSD_Aaronson_U",
    ],
}


class Worker:
    def __init__(self, param):
        """
        param: {
            "model_str": str,
            "ref_model_str": str,
            "title": str,
            "device": str, # "cuda:0", "cpu", ...
        }
        """
        self.param = param
        load_model_kwargs = {
            "device_map": str(self.param["device"]),
            "pretrained_model_name_or_path": self.param["model_str"],
            "low_cpu_mem_usage": True,
        }
        if self.param["device"].startswith("cuda"):
            load_model_kwargs["torch_dtype"] = torch.float16
        load_ref_model_kwargs = {
            **load_model_kwargs,
            "pretrained_model_name_or_path": self.param["ref_model_str"],
        }
        transformers.utils.logging.disable_progress_bar()
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            **load_model_kwargs
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.param["model_str"]
        )
        self.ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            **load_ref_model_kwargs
        )
        self.max_length_lp = MaxLengthLogitsProcessor(1, self.tokenizer.eos_token_id)
        self.stop_words_lp = StopWordsLogitsProcessor([], self.tokenizer.eos_token_id)

    def process(self, d):
        """
        d: {
            "prompt": str,
            "seed": int,
            "method": str, # "basic", "basic_uwm", "mc", "mc_uwm_strength", "mc_uwm_speed", "ersd"
            "reweight": str,# "deltagumbel", "gamma"
            "private_key": bytes,
            "n": int,
            "max_length": int, # optional
            "stop_words": list[str], # optional
        }
        return: {
            # all input fields in d, except prompt
            "output_ids": list[int],
            "gen_seq_lens": list[int],
            "output": str,
            # timestamps
            "t_got_input": float,
            "t_got_first_output": float,
            "t_got_last_output": float,
        }
        """
        torch.manual_seed(d["seed"])
        if "max_length" in d:
            self.max_length_lp.max_length = d["max_length"]
        else:
            self.max_length_lp.max_length = 9999999
        if "stop_words" in d:
            self.stop_words_lp.set_stop_words(
                d["stop_words"], self.tokenizer, self.param["device"]
            )

        input_ids = self.tokenizer(d["prompt"], return_tensors="pt")["input_ids"].to(
            self.param["device"]
        )
        self.max_length_lp.input_length = input_ids.shape[-1]
        if d["method"] == "basic":
            generator = accuwm.basic.basic_generator
        elif d["method"] == "basic_uwm":
            generator = accuwm.basic_watermark.basic_uwm_generator
        elif d["method"] == "mc":
            generator = accuwm.mc.mc_sample_generator
        elif d["method"] == "mc_uwm_strength":
            generator = partial(
                accuwm.mc_watermark.mc_uwm_sample_generator, reweight_in_mc=True
            )
        elif d["method"] == "mc_uwm_speed":
            generator = partial(
                accuwm.mc_watermark.mc_uwm_sample_generator, reweight_in_mc=False
            )
        elif d["method"] == "ersd":
            generator = accuwm.ersd.ersd_sample_generator
        elif d["method"] == "ersd_wm":
            generator = accuwm.ersd_wm.ersd_wm_sample_generator
        else:
            raise ValueError(f"unknown sampling method {d['method']}")
        if "mc" in d["method"] or d["method"] == "ersd" or d["method"] == "ersd_wm":
            generator = partial(generator, ref_model=self.ref_model)
        if "uwm" in d["method"] or d["method"] == "ersd_wm":
            if "deltagumbel" == d["reweight"]:
                reweight = uwm.DeltaGumbel_Reweight()
            elif "gamma" == d["reweight"]:
                reweight = uwm.Gamma_Reweight()
            elif "ersd" == d["reweight"]:
                reweight = uwm.DeltaGumbel_Reweight()
            else:
                raise ValueError(f"unknown reweight {d['reweight']}")
            cch_add = uwm.lm.ContextCodeHistory(batch_shape=(1,))
            cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
            generator = partial(
                generator,
                reweight=reweight,
                cc_extractor=cc_extractor,
                cch=cch_add,
                #  private_key=d["private_key"],
                # combine private_key with seed to simulate the process of random private_key
                private_key=bytes(d["seed"]) + d["private_key"],
            )

        warpers = []
        temperature = self.param.get("temperature", 1.0)
        top_k = self.param.get("top_k", 0)
        top_p = self.param.get("top_p", 0.0)
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k > 0:
            warpers.append(TopKLogitsWarper(top_k))
        if top_p is not None and top_p > 0.0:
            warpers.append(TopPLogitsWarper(top_p))
        logits_warper = LogitsProcessorList(warpers) if warpers else None

        gen = generator(
            model=self.model,
            input_ids=input_ids,
            n=d["n"],
            process_logits_kwargs={
                "logits_processor": transformers.LogitsProcessorList(
                    ([self.max_length_lp] if "max_length" in d else [])
                    + ([self.stop_words_lp] if "stop_words" in d else [])
                ),
                "logits_warper": logits_warper,
            },
        )
        output_ids = []
        output_logprobs = []
        logperplexities = []
        entropies = []
        gen_seq_lens = []
        t_got_input = time.time()
        t_got_first_output = None
        ersd_wm_meta = {
            "context_codes": [],
            "time_steps": [],
            "tags": [],
            "skipped": [],
        }
        for step in gen:
            if d["method"] == "ersd_wm":
                step_output_ids, step_output_logprobs, meta = step
                ersd_wm_meta["context_codes"].append(meta["context_codes"])
                ersd_wm_meta["time_steps"].append(meta["time_steps"])
                ersd_wm_meta["tags"].append(meta["tags"])
                ersd_wm_meta["skipped"].append(meta["skipped"])
            elif d["method"] == "ersd":
                step_output_ids, step_output_logprobs = step
            else:
                step_output_ids, step_output_logprobs = step
            if t_got_first_output is None:
                t_got_first_output = time.time()
            output_ids.extend(step_output_ids[0].cpu().tolist())
            # step_output_logprobs: (batch_size, seq_len, vocab_size)
            output_logprobs.append(step_output_logprobs)
            assert step_output_logprobs.shape[:-1] == step_output_ids.shape
            assert step_output_logprobs.shape[0] == 1
            assert np.allclose(
                step_output_logprobs.exp().sum(-1).cpu().numpy(),
                np.ones(step_output_ids.shape),
                atol=1e-2,
            ), {
                "method": d["method"],
                "sum_probs": step_output_logprobs.exp().sum(-1).cpu().numpy(),
                "entropies": -torch.sum(
                    step_output_logprobs * step_output_logprobs.exp(), dim=-1
                )
                .cpu()
                .numpy(),
                "output": self.tokenizer.decode(output_ids),
            }
            logperplexities.extend(
                (
                    -torch.gather(
                        step_output_logprobs[0],
                        -1,
                        step_output_ids[0].unsqueeze(-1),
                    )
                )
                .squeeze(-1)
                .cpu()
                .tolist()
            )
            entropies.extend(
                (
                    -torch.sum(
                        torch.clamp(
                            step_output_logprobs[0],
                            min=torch.finfo(step_output_logprobs.dtype).min,
                        )
                        * step_output_logprobs[0].exp(),
                        dim=-1,
                    )
                )
                .cpu()
                .tolist()
            )
            gen_seq_lens.append(step_output_ids.shape[-1])
        t_got_last_output = time.time()
        output = self.tokenizer.decode(output_ids)
        assert len(output_ids) == len(logperplexities) == sum(gen_seq_lens)
        output_logprobs = torch.cat(output_logprobs, dim=1)
        r = {
            **{
                k: v
                for k, v in d.items()
                if k not in ["prompt", "stop_words", "max_length"]
            },
            #  "output": output,
            #  "output_ids": output_ids,
            "gen_seq_lens": gen_seq_lens,
            "t_got_input": t_got_input,
            "t_got_first_output": t_got_first_output,
            "t_got_last_output": t_got_last_output,
            "logperplexity": np.mean(logperplexities),
            "entropy": np.mean(entropies),
            "u_score_mean": None,
            "u_score_std": None,
            "u_skipped_ratio": None,
            "u_score_count": None,
        }
        log_p_values = []
        if d["method"] == "ersd_wm":
            out_ids = torch.tensor(output_ids).unsqueeze(0).to(input_ids.device)
            context_codes = np.concatenate(ersd_wm_meta["context_codes"], axis=0)[
                None, :
            ]
            time_steps = np.concatenate(ersd_wm_meta["time_steps"], axis=0)[None, :]
            tags = np.concatenate(ersd_wm_meta["tags"], axis=0)[None, :]
            skipped = np.concatenate(ersd_wm_meta["skipped"], axis=0)[None, :]
            base_score = uwm.scores.ERSD_Aaronson_Score.from_watermarkcode(
                None,
                out_ids,
                skipped=skipped,
                context_codes=context_codes,
                time_steps=time_steps,
                tags=tags,
                private_key=bytes(d["seed"]) + d["private_key"],
                vocab_size=output_logprobs.shape[-1],
            )
            gamma_score = uwm.scores.ERSD_Aaronson_Gamma_Score(
                scores=base_score.scores,
                skipped=base_score.skipped,
            )
            u_score = uwm.scores.ERSD_Aaronson_U_Score(
                scores=base_score.scores,
                skipped=base_score.skipped,
            )
            log_p_values.append(gamma_score.get_log_p_value())
            log_p_values.append(u_score.get_log_p_value())
        elif "uwm" in d["method"]:
            out_ids = torch.tensor(output_ids).unsqueeze(0).to(input_ids.device)
            # out_ids: (batch_size, seq_len)
            # verify score
            for score_type in scores[d["reweight"]]:
                cch_detect = uwm.lm.ContextCodeHistory(batch_shape=(1,))
                score = uwm.lm.detect(
                    vocab_size=output_logprobs.shape[-1],
                    score_type=score_type,
                    reweight=reweight,
                    cc_extractor=cc_extractor,
                    cch=cch_detect,
                    #  private_key=d["private_key"],
                    private_key=bytes(d["seed"]) + d["private_key"],
                    out_ids=out_ids,
                    in_ids=input_ids,
                    p_logits=output_logprobs,
                )
                if self.param.get("assert_cch", False):
                    assert np.all(cch_add.data == cch_detect.data), {
                        **r,
                        "cch_add": cch_add.data,
                        "cch_detect": cch_detect.data,
                    }
                log_p_values.append(score.get_log_p_value())
                if getattr(score_type, "__name__", "") == "DeltaGumbel_U":
                    score_vals = np.array(score.scores)
                    skipped_vals = np.array(score.skipped).astype(bool)
                    if score_vals.shape == skipped_vals.shape:
                        mask = ~skipped_vals
                        if mask.any():
                            r["u_score_mean"] = float(score_vals[mask].mean())
                            r["u_score_std"] = float(score_vals[mask].std())
                            r["u_skipped_ratio"] = float(skipped_vals.mean())
                            r["u_score_count"] = int(mask.sum())
                        else:
                            r["u_score_mean"] = float("nan")
                            r["u_score_std"] = float("nan")
                            r["u_skipped_ratio"] = float(skipped_vals.mean())
                            r["u_score_count"] = 0
                if self.param.get("assert_log_p_values", False):
                    if d["method"] != "mc_uwm_speed" and "LLR" not in str(score_type):
                        assert score.get_log_p_value() <= 0, {
                            **r,
                            "score_type": str(score_type),
                            "prompt": d["prompt"],
                            "output": output,
                            "scores": str(score.scores),
                            "skipped": str(score.skipped),
                        }
        if len(log_p_values) == 0:
            log_p_values.append(0.0)  # placeholder for ray to infer right data type
        r["log_p_values"] = log_p_values
        if self.param.get("print_output", False):
            print({**r, "output": output, "prompt": d["prompt"]})

        return r

    @ray.method(concurrency_group="model")
    def __call__(self, d):
        """
        only batch_size=1 is supported
        """
        rs = []
        anykey = next(iter(d.keys()))
        batch_size = len(d[anykey])
        for i in range(batch_size):
            r = self.process({k: v[i] for k, v in d.items()})
            rs.append(r)
        return {k: [r[k] for r in rs] for k in rs[0]}

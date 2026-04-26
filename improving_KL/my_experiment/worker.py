import torch
import numpy as np
import transformers
from functools import partial
import time
from typing import Dict, List, Any, Optional

import accuwm
import unbiased_watermark as uwm
from experiments import tasks

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
            while ids.shape[-1] > 0 and ids[0] == 29871:
                ids = ids[1:]
        return ids

    def set_stop_words(self, stop_words: list[str], tokenizer, device="cpu"):
        self.stop_words_ids = [
            self._to_ids(stop_word, tokenizer).to(device) for stop_word in stop_words
        ]


def safe_ln(x):
    x = np.array(x)
    with np.errstate(divide="ignore"):
        return np.where(x <= 0, -np.inf, np.log(x))

def _process_raw_prompt(tokenizer, prompt) -> str:
  """Add chat template to the raw prompt."""
  return tokenizer.apply_chat_template(
      [{'role': 'user', 'content': prompt.decode().strip('"')}],
      tokenize=False,
      add_generation_prompt=True,
  )


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
}


class Worker:
    def __init__(self, param: Dict[str, Any]):
        """
        param: {
            "model_str": str,
            "ref_model_str": str,
            "title": str,
            "device": str, # "cuda:0", "cpu", ...
            "hf_token": str, # Hugging Face access token
        }
        """
        self.param = param
        load_model_kwargs = {
            "device_map": str(self.param["device"]),
            "pretrained_model_name_or_path": self.param["model_str"],
            "low_cpu_mem_usage": True,
            "token": self.param.get("hf_token"),
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
            self.param["model_str"],
            token=self.param.get("hf_token"),
        )
        self.ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            **load_ref_model_kwargs
        )
        self.max_length_lp = MaxLengthLogitsProcessor(1, self.tokenizer.eos_token_id)
        self.stop_words_lp = StopWordsLogitsProcessor([], self.tokenizer.eos_token_id)

    def process(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single experiment run.
        
        Args:
            d: Dictionary containing experiment parameters:
                - prompt: str
                - seed: int
                - method: str (basic, basic_uwm, mc, mc_uwm_strength, mc_uwm_speed)
                - reweight: str (deltagumbel, gamma, synthid)
                - private_key: bytes
                - n: int
                - max_length: int (optional)
                - stop_words: list[str] (optional)
        
        Returns:
            Dictionary containing results and metrics
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
        # add chat template to the raw prompt for instruction tuning models
        if '-it' in self.param["model_str"]:
            d["prompt"] = _process_raw_prompt(self.tokenizer, d["prompt"].encode())
            
        input_ids = self.tokenizer(d["prompt"], return_tensors="pt")["input_ids"].to(
            self.param["device"]
        )
        self.max_length_lp.input_length = input_ids.shape[-1]
        
        # Setup generator based on method
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
        elif d["method"] == "mc_uwm_synthid":
            generator = partial(
                accuwm.mc_watermark.mc_uwm_sample_generator, reweight_in_mc=False, mc_synthid=True, mc_private_key=bytes(d["seed"]) + d["mc_private_key"]
            )
        elif d["method"] == "mc_uwm_synthid_psedo_r":
            generator = partial(
                accuwm.mc_watermark.mc_uwm_sample_generator, reweight_in_mc=False, mc_synthid=True, mc_private_key=bytes(d["seed"]) + d["mc_private_key"], psedo_r=True
            )
        else:
            raise ValueError(f"unknown sampling method {d['method']}")

        if "mc" in d["method"]:
            generator = partial(generator, ref_model=self.ref_model)
        if "uwm" in d["method"]:
            if "deltagumbel" == d["reweight"]:
                reweight = uwm.DeltaGumbel_Reweight()
            elif "gamma" == d["reweight"]:
                reweight = uwm.Gamma_Reweight()
            elif "synthid" == d["reweight"]:
                reweight = uwm.SynthID_Reweight()
            else:
                raise ValueError(f"unknown reweight {d['reweight']}")
            cch_add = uwm.lm.ContextCodeHistory(batch_shape=(1,))
            cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=4)
            generator = partial(
                generator,
                reweight=reweight,
                cc_extractor=cc_extractor,
                cch=cch_add,
                private_key=bytes(d["seed"]) + d["private_key"],
            )

            gen = generator(
                model=self.model,
                input_ids=input_ids,
                n=d["n"],
                temperature=d["temperature"],
                process_logits_kwargs={
                    "logits_processor": transformers.LogitsProcessorList(
                        ([self.max_length_lp] if "max_length" in d else [])
                        + ([self.stop_words_lp] if "stop_words" in d else [])
                    )
                },
            )
        else:   # if not involve watermark, do not set temperature.
            gen = generator(
            model=self.model,
            input_ids=input_ids,
            n=d["n"],
            process_logits_kwargs={
                "logits_processor": transformers.LogitsProcessorList(
                    ([self.max_length_lp] if "max_length" in d else [])
                    + ([self.stop_words_lp] if "stop_words" in d else [])
                    )
                },
            )

        output_ids = []
        output_logprobs = []
        logperplexities = []
        entropies = []
        gen_seq_lens = []
        t_got_input = time.time()
        t_got_first_output = None

        for step_output_ids, step_output_logprobs in gen:
            if t_got_first_output is None:
                t_got_first_output = time.time()
            output_ids.extend(step_output_ids[0].cpu().tolist())
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
            "output": output,
            "output_ids": output_ids,
            "output_len": len(output_ids),
            "gen_seq_lens": gen_seq_lens,
            "logperplexity": np.mean(logperplexities),
            "entropy": np.mean(entropies),
        }

        if "uwm" in d["method"] or d["method"] == "basic":
            out_ids = torch.tensor(output_ids).unsqueeze(0).to(input_ids.device)
            cch_detect = uwm.lm.ContextCodeHistory(batch_shape=(1,))

            if "synthid" == d["reweight"]:
                # get the watermark code (g_values)
                # when get the g_value for unwatermarked sequence, we need first set the reweight and cc_extractor
                if d["method"] == "basic":
                    reweight = uwm.SynthID_Reweight()
                    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=4)
                _, _, _, watermark_code, skipped = uwm.lm.detect_pre(
                    vocab_size=self.model.config.vocab_size,
                        reweight=reweight,
                        cc_extractor=cc_extractor,
                        cch=cch_detect,
                        private_key=bytes(d["seed"]) + d["private_key"],
                        out_ids=out_ids,
                        in_ids=input_ids,
                        p_logits=output_logprobs,
                    )
                all_g_values = watermark_code.binary_matrix
                assert all_g_values.shape[1] == out_ids.shape[-1], f"the watermark code should have the same length as the output, but got {all_g_values.shape[1]} and {out_ids.shape[-1]}"
                # Get g_values for each token by selecting from binary_matrix using output_ids
                # For each position, select the matrix corresponding to the output token at that position
                g_values = torch.stack([all_g_values[0, i, out_ids[0][i], :] for i in range(len(out_ids[0]))])
                r["g_values"] = g_values.cpu().numpy()
                r["g_values_mean"] = g_values.mean().cpu().numpy()
                r["skipped"] = skipped

            elif "deltagumbel" == d["reweight"]:
                _, _, _, watermark_code, skipped = uwm.lm.detect_pre(
                    vocab_size=self.model.config.vocab_size,
                        reweight=reweight,
                        cc_extractor=cc_extractor,
                        cch=cch_detect,
                        private_key=bytes(d["seed"]) + d["private_key"],
                        out_ids=out_ids,
                        in_ids=input_ids,
                        p_logits=output_logprobs,
                    )
                all_g_values = watermark_code.g
                assert all_g_values.shape[1] == out_ids.shape[-1], f"the watermark code should have the same length as the output, but got {all_g_values.shape[1]} and {out_ids.shape[-1]}"
                g_values = torch.stack([all_g_values[0, i, out_ids[0][i]] for i in range(len(out_ids[0]))])
                y_values = g_values.cpu().numpy()
                y_values = np.exp(-np.exp(-y_values))
                r["y_values"] = y_values
                r["y_values_mean"] = y_values.mean()
                r['skipped'] = skipped
            
        if "psedo_r" in d["method"] or "mc_" in d["method"]:
            _, _, _, watermark_code_mc, _ = uwm.lm.detect_pre(
            vocab_size=self.model.config.vocab_size,
                reweight=reweight,
                cc_extractor=cc_extractor,
                cch=cch_detect,
                private_key=bytes(d["seed"]) + d["mc_private_key"], # use another private key for reject sampling
                out_ids=out_ids,
                in_ids=input_ids,
                p_logits=output_logprobs,
            )
            if "synthid" == d["reweight"]:
                g_values_mc = watermark_code_mc.binary_matrix
                g_values_mc = torch.stack([g_values_mc[0, i, out_ids[0][i], :] for i in range(len(out_ids[0]))])
                r["g_values_mc"] = g_values_mc.cpu().numpy()
                r["g_values_mc_mean"] = g_values_mc.mean().cpu().numpy()
            elif "deltagumbel" == d["reweight"]:
                g_values_mc = watermark_code_mc.g
                g_values_mc = torch.stack([g_values_mc[0, i, out_ids[0][i]] for i in range(len(out_ids[0]))])
                y_values_mc = g_values_mc.cpu().numpy()
                y_values_mc = np.exp(-np.exp(-y_values_mc))
                r["mc_y_values"] = y_values_mc
                r["mc_y_values_mean"] = y_values_mc.mean()

            if "psedo_r" in d["method"]:
                cch_r_detect = uwm.lm.ContextCodeHistory(batch_shape=(1,))
                r["r_values"] = uwm.lm.get_r_values(cc_extractor, cch_r_detect, bytes(d["seed"]) + d["mc_private_key"], out_ids, input_ids).squeeze(0)

        if self.param.get("print_output", False):
            print({**r, "prompt": d["prompt"]})

        return r

    def process_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Process a batch of experiments.
        
        Args:
            batch: Dictionary where each value is a list of parameters for each experiment
            
        Returns:
            Dictionary where each value is a list of results for each experiment
        """
        results = []
        batch_size = len(next(iter(batch.values())))
        
        for i in range(batch_size):
            experiment_params = {k: v[i] for k, v in batch.items()}
            result = self.process(experiment_params)
            results.append(result)
            
        return {k: [r[k] for r in results] for k in results[0]} 
    

def process_human_data(output_ids: List[int], vocab_size: int, seed: int, private_key: bytes, mc_private_key: bytes) -> Dict[str, Any]:
    """
    Process human data. Get the g_values and r_values for the human data.
    """
    r = {}
    out_ids = torch.tensor(output_ids).unsqueeze(0)
    cch_detect = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    reweight = uwm.SynthID_Reweight()
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=4)
    _, _, _, watermark_code, skipped = uwm.lm.detect_pre(
        vocab_size=vocab_size,
        reweight=reweight,
        cc_extractor=cc_extractor,
        cch=cch_detect,
        private_key=bytes(seed) + private_key,
        out_ids=out_ids,
    )
    all_g_values = watermark_code.binary_matrix
    assert all_g_values.shape[1] == out_ids.shape[-1], f"the watermark code should have the same length as the output, but got {all_g_values.shape[1]} and {out_ids.shape[-1]}"
    g_values = torch.stack([all_g_values[0, i, out_ids[0][i], :] for i in range(len(out_ids[0]))])
    r["g_values"] = g_values.cpu().numpy()
    r["g_values_mean"] = g_values.mean().cpu().numpy()
    r["skipped"] = skipped
    
    _, _, _, watermark_code_mc, _ = uwm.lm.detect_pre(
        vocab_size=vocab_size,
        reweight=reweight,
        cc_extractor=cc_extractor,
        cch=cch_detect,
        private_key=bytes(seed) + mc_private_key,
        out_ids=out_ids,
    )
    g_values_mc = watermark_code_mc.binary_matrix
    g_values_mc = torch.stack([g_values_mc[0, i, out_ids[0][i], :] for i in range(len(out_ids[0]))])
    r["g_values_mc"] = g_values_mc.cpu().numpy()
    r["g_values_mc_mean"] = g_values_mc.mean().cpu().numpy()
    
    cch_r_detect = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    r["r_values"] = uwm.lm.get_r_values(cc_extractor, cch_r_detect, bytes(seed) + mc_private_key, out_ids).squeeze(0)
    
    return r
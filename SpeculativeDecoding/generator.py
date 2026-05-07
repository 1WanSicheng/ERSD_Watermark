from transformers import DynamicCache
from utils import GeneratorOutput, LogitsProcessor, gumbel_sample
from time import perf_counter
import torch
import numpy as np

class BasicGenerator:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input_ids, eos_token_id, temperature=1.0, 
                 top_p=1.0, top_k=50, max_new_tokens=512):
        past_key_values = None
        input_len = input_ids.size(-1)
        logits_processor = LogitsProcessor(temperature, top_p, top_k)

        num_invocations = 0
        avg_generation_time = 0

        t_0 = perf_counter()
        while input_ids.size(-1) < input_len + max_new_tokens:
            if past_key_values is not None:
                pruned_input_ids = input_ids[:, past_key_values[0][0].size(2):]
            else:
                pruned_input_ids = input_ids
                past_key_values = DynamicCache()

            # Run the model and collect outputs
            tt_0 = perf_counter()
            outputs = self.model(input_ids=pruned_input_ids, use_cache=True, 
                                 past_key_values=past_key_values, return_dict=True, 
                                 output_attentions=False, output_hidden_states=False)
            tt_1 = perf_counter()
            avg_generation_time += tt_1 - tt_0
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # Sample from the model
            logits = logits_processor(logits[:, -1, :])
            new_ids = gumbel_sample(logits)
            input_ids = torch.cat((input_ids, new_ids.unsqueeze(0)), dim=-1)
            num_invocations += 1

            if new_ids[0] == eos_token_id:
                break
        t_1 = perf_counter()

        num_tokens = input_ids.size(-1) - input_len
        token_rate = num_tokens / (t_1 - t_0)
        del past_key_values

        return GeneratorOutput(
            sequences=input_ids,
            acceptance_rate=1.0,
            token_rate=token_rate,
            avg_generation_time=avg_generation_time / num_invocations,
            avg_verification_time=0.0,
            num_invocations=num_invocations,
            total_time=t_1 - t_0
        )


###log version
class SpeculativeGenerator:
    def __init__(self, strategy):
        self.strategy = strategy 
        self.max_draft_len = strategy.max_draft_len
        self.max_num_drafts = strategy.max_num_drafts

    def __call__(self, input_ids, eos_token_id, temperature=1.0,
                 top_p=0.0, top_k=50, max_new_tokens=512):

        # ---- Debug flags ----
        DEBUG = os.getenv("PPR_DEBUG", "0") == "1"
        DEBUG_STEPS = int(os.getenv("PPR_DEBUG_STEPS", "5"))
        SYNC_CUDA = (os.getenv("PPR_DEBUG_SYNC", "1") == "1") and torch.cuda.is_available()

        def cache_len(pkv):
            if pkv is None:
                return 0
            if hasattr(pkv, "key_cache") and pkv.key_cache:
                return pkv.key_cache[0].size(2)
            try:
                return pkv[0][0].size(2)
            except Exception:
                return 0

        target_past_key_values = None
        draft_past_key_values = None
        input_len = input_ids.size(-1)

        # Build logits processors
        if hasattr(temperature, '__len__'):
            target_temp = torch.tensor(temperature[0], device=self.strategy.target.device).reshape((1, 1, 1))
            draft_temp = torch.tensor(temperature[1:], device=self.strategy.target.device).reshape((-1, 1, 1))
            target_logits_processor = LogitsProcessor(target_temp, top_p, top_k)
            draft_logits_processor = LogitsProcessor(draft_temp, top_p, top_k)
        else:
            temp = torch.tensor(temperature, device=self.strategy.target.device).reshape((1, 1, 1))
            target_logits_processor = LogitsProcessor(temp, top_p, top_k)
            draft_logits_processor = LogitsProcessor(temp, top_p, top_k)

        num_invocations = 0
        num_accept = 0
        avg_verification_time = 0.0
        avg_generation_time = 0.0

        if DEBUG:
            print(f"[GEN] DEBUG enabled. max_draft_len={self.max_draft_len}, "
                  f"strategy={type(self.strategy).__name__}, top_k={top_k}, top_p={top_p}")

        if SYNC_CUDA:
            torch.cuda.synchronize()
        t_0 = perf_counter()

        while input_ids.size(-1) < input_len + max_new_tokens:

            if DEBUG and num_invocations < DEBUG_STEPS:
                print(f"\n[STEP {num_invocations}] seq_len={input_ids.size(-1)} "
                      f"draft_cache_len={cache_len(draft_past_key_values)} "
                      f"target_cache_len={cache_len(target_past_key_values)}")

            # ---- Draft ----
            if SYNC_CUDA:
                torch.cuda.synchronize()
            tt_0 = perf_counter()

            draft_outputs = self.strategy.generate_draft(
                input_ids=input_ids,
                past_key_values=draft_past_key_values,
                logits_processor=draft_logits_processor
            )

            if SYNC_CUDA:
                torch.cuda.synchronize()
            tt_1 = perf_counter()

            avg_generation_time += (tt_1 - tt_0)
            draft_past_key_values = draft_outputs.draft_past_key_values

            # ---- Verify ----
            if SYNC_CUDA:
                torch.cuda.synchronize()
            tt_0 = perf_counter()

            verify_outputs = self.strategy.verify_draft(
                input_ids=draft_outputs.sequences,
                target_past_key_values=target_past_key_values,
                draft_past_key_values=draft_past_key_values,
                draft_probs=draft_outputs.draft_probs,
                logits_processor=target_logits_processor
            )

            if SYNC_CUDA:
                torch.cuda.synchronize()
            tt_1 = perf_counter()

            avg_verification_time += (tt_1 - tt_0)

            # Update states
            input_ids = verify_outputs.sequences
            draft_past_key_values = verify_outputs.draft_past_key_values
            target_past_key_values = verify_outputs.target_past_key_values

            num_invocations += 1
            num_accept += verify_outputs.accept_count

            if DEBUG and num_invocations <= DEBUG_STEPS:
                print(f"[STEP {num_invocations-1}] accept_count={verify_outputs.accept_count}, "
                      f"new_seq_len={input_ids.size(-1)}")

            # EOS check (keep your original logic)
            if eos_token_id in input_ids[0, -self.max_draft_len:]:
                if DEBUG and num_invocations <= DEBUG_STEPS:
                    print("[GEN] EOS detected, stopping.")
                break

        if SYNC_CUDA:
            torch.cuda.synchronize()
        t_1 = perf_counter()

        num_tokens = input_ids.size(-1) - input_len
        token_rate = num_tokens / (t_1 - t_0)

        del target_past_key_values
        del draft_past_key_values

        return GeneratorOutput(
            sequences=input_ids,
            acceptance_rate=num_accept / max(1, num_invocations),
            token_rate=token_rate,
            avg_generation_time=avg_generation_time / max(1, num_invocations),
            avg_verification_time=avg_verification_time / max(1, num_invocations),
            num_invocations=num_invocations,
            total_time=t_1 - t_0
        )

# class SpeculativeGenerator:
    def __init__(self, strategy):
        self.strategy = strategy 
        self.max_draft_len = strategy.max_draft_len
        self.max_num_drafts = strategy.max_num_drafts

    def __call__(self, input_ids, eos_token_id, temperature=1.0,
                 top_p=0.0, top_k=50, max_new_tokens=512):
        target_past_key_values = None
        draft_past_key_values = None
        input_len = input_ids.size(-1)

        if hasattr(temperature, '__len__'):
            target_temp = torch.tensor(temperature[0], device=self.strategy.target.device).reshape((1, 1, 1))
            draft_temp = torch.tensor(temperature[1:], device=self.strategy.target.device).reshape((-1, 1, 1))
            target_logits_processor = LogitsProcessor(target_temp, top_p, top_k)
            draft_logits_processor = LogitsProcessor(draft_temp, top_p, top_k)
        else:
            temp = torch.tensor(temperature, device=self.strategy.target.device).reshape((1, 1, 1))
            target_logits_processor = LogitsProcessor(temp, top_p, top_k)
            draft_logits_processor = LogitsProcessor(temp, top_p, top_k)

        num_invocations = 0
        num_accept = 0
        avg_verification_time = 0
        avg_generation_time = 0
        
        t_0 = perf_counter()
        while input_ids.size(-1) < input_len + max_new_tokens:
            tt_0 = perf_counter()
            draft_outputs = self.strategy.generate_draft(
                input_ids=input_ids, 
                past_key_values=draft_past_key_values, 
                logits_processor=draft_logits_processor
            )
            tt_1 = perf_counter()
            avg_generation_time += tt_1 - tt_0
            draft_past_key_values = draft_outputs.draft_past_key_values

            tt_0 = perf_counter()
            verify_outputs = self.strategy.verify_draft(
                input_ids=draft_outputs.sequences,
                target_past_key_values=target_past_key_values,
                draft_past_key_values=draft_past_key_values,
                draft_probs=draft_outputs.draft_probs,
                logits_processor=target_logits_processor
            )
            tt_1 = perf_counter()
            avg_verification_time += tt_1 - tt_0
            input_ids = verify_outputs.sequences
            draft_past_key_values = verify_outputs.draft_past_key_values
            target_past_key_values = verify_outputs.target_past_key_values

            num_invocations += 1
            num_accept += verify_outputs.accept_count

            new_token_count = input_ids.size(-1) - input_len
            window = min(new_token_count, self.max_draft_len)
            if window > 0 and eos_token_id in input_ids[0, -window:]:
                break
        t_1 = perf_counter()

        num_tokens = input_ids.size(-1) - input_len
        token_rate = num_tokens / (t_1 - t_0)
        del target_past_key_values
        del draft_past_key_values

        return GeneratorOutput(
            sequences=input_ids,
            acceptance_rate=num_accept / num_invocations,
            token_rate=token_rate,
            avg_generation_time=avg_generation_time / num_invocations,
            avg_verification_time=avg_verification_time / num_invocations,
            num_invocations=num_invocations,
            total_time=t_1 - t_0
        )
    
class InvariantGenerator:
    def __init__(self, strategy):
        self.strategy = strategy 
        self.max_draft_len = strategy.max_draft_len
        self.max_num_drafts = strategy.max_num_drafts
        self.vocab_size = strategy.vocab_size

    def __call__(self, input_ids, eos_token_id, temperature=1.0,
                 top_p=0.0, top_k=50, max_new_tokens=128):
        target_past_key_values = None
        draft_past_key_values = None
        input_len = input_ids.size(-1)

        if hasattr(temperature, '__len__'):
            target_temp = torch.tensor(temperature[0], device=self.strategy.target.device).reshape((1, 1, 1))
            draft_temp = torch.tensor(temperature[1:], device=self.strategy.target.device).reshape((-1, 1, 1))
            target_logits_processor = LogitsProcessor(target_temp, top_p, top_k)
            draft_logits_processor = LogitsProcessor(draft_temp, top_p, top_k)
        else:
            temp = torch.tensor(temperature, device=self.strategy.target.device).reshape((1, 1, 1))
            target_logits_processor = LogitsProcessor(temp, top_p, top_k)
            draft_logits_processor = LogitsProcessor(temp, top_p, top_k)

        num_invocations = 0
        num_accept = 0
        avg_verification_time = 0
        avg_generation_time = 0

        t_0 = perf_counter()
        # Generate common randomness ahead of time
        randomness = torch.empty((max_new_tokens + self.max_draft_len + 1,
                                  self.max_num_drafts,
                                  self.vocab_size), device=self.strategy.target.device)
        randomness.uniform_()

        while input_ids.size(-1) < input_len + max_new_tokens:
            position = input_ids.size(-1) - input_len

            tt_0 = perf_counter()
            draft_outputs = self.strategy.generate_draft(
                input_ids=input_ids, 
                past_key_values=draft_past_key_values, 
                logits_processor=draft_logits_processor,
                position=position,
                randomness=randomness
            )
            tt_1 = perf_counter()
            avg_generation_time += tt_1 - tt_0
            draft_past_key_values = draft_outputs.draft_past_key_values

            tt_0 = perf_counter()
            verify_outputs = self.strategy.verify_draft(
                input_ids=draft_outputs.sequences,
                target_past_key_values=target_past_key_values,
                draft_past_key_values=draft_past_key_values,
                draft_probs=draft_outputs.draft_probs,
                logits_processor=target_logits_processor,
                position=position,
                randomness=randomness
            )
            tt_1 = perf_counter()
            avg_verification_time += tt_1 - tt_0
            input_ids = verify_outputs.sequences
            draft_past_key_values = verify_outputs.draft_past_key_values
            target_past_key_values = verify_outputs.target_past_key_values

            num_invocations += 1
            num_accept += verify_outputs.accept_count

            new_token_count = input_ids.size(-1) - input_len
            window = min(new_token_count, self.max_draft_len)
            if window > 0 and eos_token_id in input_ids[0, -window:]:
                break
        t_1 = perf_counter()

        num_tokens = input_ids.size(-1) - input_len
        token_rate = num_tokens / (t_1 - t_0)
        del target_past_key_values
        del draft_past_key_values

        return GeneratorOutput(
            sequences=input_ids,
            acceptance_rate=num_accept / num_invocations,
            token_rate=token_rate,
            avg_generation_time=avg_generation_time / num_invocations,
            avg_verification_time=avg_verification_time / num_invocations,
            num_invocations=num_invocations,
            total_time=t_1 - t_0
        )
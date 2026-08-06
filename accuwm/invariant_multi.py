"""Invariant multi-draft speculative-decoding strategy + generator.

Self-contained port of the only two classes from the legacy
``SpeculativeDecoding/`` folder that the active codebase still uses
(InvariantMultiDraftStrategy + InvariantGenerator).  Algorithm semantics are
unchanged from the original; this file only consolidates the strategy, the
generator, and the small set of helpers they depend on into one module under
``accuwm/``.

Both classes implement Ashish-style multi-draft speculative decoding with
unkeyed, pre-sampled common randomness shared across drafts, hence the name
"invariant".  No watermark.
"""
from __future__ import annotations

from collections import namedtuple
from time import perf_counter

import torch
from transformers import DynamicCache


DraftOutput = namedtuple("DraftOutput", [
    "sequences",
    "draft_probs",
    "draft_past_key_values",
])

VerifyOutput = namedtuple("VerifyOutput", [
    "sequences",
    "target_past_key_values",
    "draft_past_key_values",
    "accept_count",
])

GeneratorOutput = namedtuple("SpeculativeGeneratorOutput", [
    "sequences",
    "acceptance_rate",
    "token_rate",
    "avg_generation_time",
    "avg_verification_time",
    "num_invocations",
    "total_time",
])


def _log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def _gumbel_noise(noise):
    return -_log(-_log(noise))


def gumbel_sample(logits, noise=None, dim=-1):
    if noise is None:
        noise = torch.zeros_like(logits).uniform_(0, 1)
    return (logits + _gumbel_noise(noise)).argmax(dim=dim)


class LogitsProcessor:
    def __init__(self, temperature, top_p, top_k):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def __call__(self, logits):
        if self.top_k is None or self.top_k <= 0 or self.top_k >= logits.shape[-1]:
            return logits / self.temperature
        val, ind = torch.topk(logits, self.top_k, sorted=False, dim=-1)
        probs = torch.full_like(logits, float("-inf"))
        probs.scatter_(-1, ind, val)
        return probs / self.temperature


class InvariantMultiDraftStrategy:
    def __init__(self, target, drafter, tokenizer, max_draft_len, max_num_drafts):
        self.target = target
        self.drafter = drafter
        self.max_draft_len = max_draft_len
        self.max_num_drafts = max_num_drafts
        self.vocab_size = tokenizer.vocab_size
        self.randomness = torch.zeros(
            (self.max_num_drafts, self.max_draft_len, self.vocab_size),
            device=target.device,
        )

    @torch.no_grad()
    def generate_draft(self, input_ids, past_key_values, logits_processor, position, randomness):
        batch_size = self.max_num_drafts
        rn = randomness[position:(position + self.max_draft_len + 1), :, :]

        # Duplicate the input ids for batching
        input_ids = input_ids.repeat((batch_size, 1))

        if past_key_values is not None:
            for i in range(len(past_key_values)):
                past_key_values.key_cache[i] = past_key_values.key_cache[i].repeat(
                    (batch_size, 1, 1, 1))
                past_key_values.value_cache[i] = past_key_values.value_cache[i].repeat(
                    (batch_size, 1, 1, 1))

        for i in range(self.max_draft_len):
            if past_key_values is not None:
                pruned_input_ids = input_ids[:, past_key_values[0][0].size(2):]
            else:
                pruned_input_ids = input_ids
                past_key_values = DynamicCache()

            outputs = self.drafter(input_ids=pruned_input_ids, use_cache=True,
                                   past_key_values=past_key_values, return_dict=True,
                                   output_attentions=False, output_hidden_states=False)
            logits = outputs.logits[..., :self.vocab_size]
            past_key_values = outputs.past_key_values

            logits = logits_processor(logits[:, -1:])
            draft_ids = gumbel_sample(logits, rn[i, :, :].unsqueeze(0).permute(1, 0, 2))
            input_ids = torch.cat((input_ids, draft_ids), dim=-1)

        return DraftOutput(
            sequences=input_ids,
            draft_probs=None,
            draft_past_key_values=past_key_values,
        )

    @torch.no_grad()
    def verify_draft(self, input_ids, target_past_key_values, draft_past_key_values,
                     draft_probs, logits_processor, position, randomness):
        assert draft_past_key_values is not None
        batch_size, input_len = input_ids.shape

        if target_past_key_values is not None:
            pruned_input_ids = input_ids[:, target_past_key_values.key_cache[0].size(2):]

            # Duplicate the past key values for batch verification
            for i in range(len(target_past_key_values)):
                target_past_key_values.key_cache[i] = target_past_key_values.key_cache[i].repeat(
                    (batch_size, 1, 1, 1))
                target_past_key_values.value_cache[i] = target_past_key_values.value_cache[i].repeat(
                    (batch_size, 1, 1, 1))
        else:
            pruned_input_ids = input_ids
            target_past_key_values = DynamicCache()

        rn = randomness[position:(position + self.max_draft_len + 1), :, :]

        outputs = self.target(input_ids=pruned_input_ids, use_cache=True,
                              past_key_values=target_past_key_values, return_dict=True,
                              output_attentions=False, output_hidden_states=False)
        logits = outputs.logits[..., :self.vocab_size]
        target_past_key_values = outputs.past_key_values

        logits = logits_processor(logits[:, (-self.max_draft_len - 1):])
        draft_ids = input_ids[:, -self.max_draft_len:]
        assert logits.size(1) == draft_ids.size(1) + 1

        rejected_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.target.device)
        alive_group_id = 0
        for depth in range(self.max_draft_len):
            # Take the maximum along active drafts and sample from the target model
            target_noise, _ = torch.max(rn[depth, ~rejected_mask, :], dim=0)
            target_ids = gumbel_sample(logits[:, depth], target_noise)

            for k in range(batch_size):
                # Check if this draft has already been rejected
                if rejected_mask[k]:
                    continue

                if draft_ids[k, depth] == target_ids[k]:
                    # Accept the draft token
                    alive_group_id = k
                    rejected_mask |= torch.ne(draft_ids[:, depth], draft_ids[k, depth])
                    break

                # Reject the draft token
                rejected_mask[k] = True

            if torch.all(rejected_mask):
                break

        # See if we have accepted all the drafts
        if not torch.all(rejected_mask):
            depth = self.max_draft_len

        last_token = gumbel_sample(logits[alive_group_id, depth]).unsqueeze(0)
        input_ids = input_ids[alive_group_id, :(input_len - self.max_draft_len + depth)]
        input_ids = torch.cat((input_ids, last_token)).unsqueeze(0)

        # Adjust cache
        for i in range(len(target_past_key_values)):
            target_past_key_values.key_cache[i] = target_past_key_values.key_cache[i][
                alive_group_id, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)
            target_past_key_values.value_cache[i] = target_past_key_values.value_cache[i][
                alive_group_id, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)
        for i in range(len(draft_past_key_values)):
            draft_past_key_values.key_cache[i] = draft_past_key_values.key_cache[i][
                alive_group_id, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)
            draft_past_key_values.value_cache[i] = draft_past_key_values.value_cache[i][
                alive_group_id, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)

        return VerifyOutput(
            sequences=input_ids,
            target_past_key_values=target_past_key_values,
            draft_past_key_values=draft_past_key_values,
            accept_count=depth + 1,
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

        if hasattr(temperature, "__len__"):
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
        randomness = torch.empty(
            (max_new_tokens + self.max_draft_len + 1,
             self.max_num_drafts,
             self.vocab_size),
            device=self.strategy.target.device,
        )
        randomness.uniform_()

        while input_ids.size(-1) < input_len + max_new_tokens:
            position = input_ids.size(-1) - input_len

            tt_0 = perf_counter()
            draft_outputs = self.strategy.generate_draft(
                input_ids=input_ids,
                past_key_values=draft_past_key_values,
                logits_processor=draft_logits_processor,
                position=position,
                randomness=randomness,
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
                randomness=randomness,
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
            total_time=t_1 - t_0,
        )

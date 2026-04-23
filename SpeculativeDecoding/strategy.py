from transformers import DynamicCache
from utils import DraftOutput, VerifyOutput, gumbel_sample, poisson_race_sample,poisson_race_sample_from_probs
from scipy.optimize import brentq
import torch
import time
import random
import numpy as np

class SingleDraftStrategy:
    def __init__(self, target, drafter, tokenizer, max_draft_len, max_num_drafts):
        self.target = target
        self.drafter = drafter
        self.max_draft_len = max_draft_len
        self.max_num_drafts = 1
        self.vocab_size = tokenizer.vocab_size

    @torch.no_grad()
    def generate_draft(self, input_ids, past_key_values, logits_processor):
        draft_probs = torch.zeros((self.max_draft_len, self.vocab_size), 
                                  device=self.drafter.device)

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
            draft_probs[i, :] = torch.nn.functional.softmax(logits, dim=-1).squeeze()
            draft_ids = gumbel_sample(logits, None)
            input_ids = torch.cat((input_ids, draft_ids), dim=-1)
            
        return DraftOutput(
            sequences=input_ids,
            draft_probs=draft_probs,
            draft_past_key_values=past_key_values
        )

    @torch.no_grad()
    def verify_draft(self, input_ids, target_past_key_values, draft_past_key_values, 
                     draft_probs, logits_processor):
        assert draft_past_key_values is not None
        batch_size, input_len = input_ids.shape
        assert batch_size == 1

        if target_past_key_values is not None:
            pruned_input_ids = input_ids[:, target_past_key_values.key_cache[0].size(2):]
        else:
            pruned_input_ids = input_ids
            target_past_key_values = DynamicCache()

        outputs = self.target(input_ids=pruned_input_ids, use_cache=True,
                              past_key_values=target_past_key_values, return_dict=True,
                              output_attentions=False, output_hidden_states=False)
        logits = outputs.logits[..., :self.vocab_size]
        target_past_key_values = outputs.past_key_values

        logits = logits_processor(logits[:, (-self.max_draft_len - 1):])
        target_probs = torch.nn.functional.softmax(logits, dim=-1)

        draft_ids = input_ids[:, -self.max_draft_len:]
        assert target_probs.size(1) == draft_ids.size(1) + 1

        reject_flag = False
        for depth in range(self.max_draft_len):
            target_dist = target_probs[0, depth, :]
            draft_dist = draft_probs[depth, :]

            residual = target_dist - draft_dist
            residual = torch.maximum(residual, torch.zeros_like(residual))
            residual = residual / torch.sum(residual, dim=-1, keepdim=True)

            r = random.random()
            x = draft_ids[0, depth]
            h = target_dist[x] / draft_dist[x]
            if r >= h:
                reject_flag = True
                break

        if not reject_flag:
            depth = self.max_draft_len
            residual = target_probs[0, -1, :]
            
        last_token = torch.multinomial(residual, num_samples=1)
        input_ids = input_ids[0, :(input_len - self.max_draft_len + depth)]
        input_ids = torch.cat((input_ids, last_token)).unsqueeze(0)

        # Adjust cache
        for i in range(len(target_past_key_values)):
            target_past_key_values.key_cache[i] = target_past_key_values.key_cache[i][
                0, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)
            target_past_key_values.value_cache[i] = target_past_key_values.value_cache[i][
                0, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)
        for i in range(len(draft_past_key_values)):
            draft_past_key_values.key_cache[i] = draft_past_key_values.key_cache[i][
                0, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)
            draft_past_key_values.value_cache[i] = draft_past_key_values.value_cache[i][
                0, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)

        return VerifyOutput(
            sequences=input_ids,
            target_past_key_values=target_past_key_values,
            draft_past_key_values=draft_past_key_values,
            accept_count=depth + 1
        )
    
class SpecInferStrategy:
    def __init__(self, target, drafter, tokenizer, max_draft_len, max_num_drafts):
        self.target = target
        self.drafter = drafter
        self.max_draft_len = max_draft_len
        self.max_num_drafts = max_num_drafts
        self.vocab_size = tokenizer.vocab_size

    @torch.no_grad()
    def generate_draft(self, input_ids, past_key_values, logits_processor):
        batch_size = self.max_num_drafts

        # Duplicate the input ids for batching
        input_ids = input_ids.repeat((batch_size, 1))
        draft_probs = torch.zeros((batch_size, self.max_draft_len, self.vocab_size), 
                                  device=self.drafter.device)

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
            draft_probs[:, i, :] = torch.nn.functional.softmax(logits, dim=-1).squeeze()
            draft_ids = gumbel_sample(logits, None)
            input_ids = torch.cat((input_ids, draft_ids), dim=-1)
            
        return DraftOutput(
            sequences=input_ids,
            draft_probs=draft_probs,
            draft_past_key_values=past_key_values
        )
        
    @torch.no_grad()
    def verify_draft(self, input_ids, target_past_key_values, draft_past_key_values, 
                     draft_probs, logits_processor):
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

        outputs = self.target(input_ids=pruned_input_ids, use_cache=True,
                              past_key_values=target_past_key_values, return_dict=True,
                              output_attentions=False, output_hidden_states=False)
        logits = outputs.logits[..., :self.vocab_size]
        target_past_key_values = outputs.past_key_values

        logits = logits_processor(logits[:, (-self.max_draft_len - 1):])
        target_probs = torch.nn.functional.softmax(logits, dim=-1)

        draft_ids = input_ids[:, -self.max_draft_len:]
        assert target_probs.size(1) == draft_ids.size(1) + 1

        rejected_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.target.device)
        alive_group_id = 0
        for depth in range(self.max_draft_len):
            idx = torch.nonzero(~rejected_mask)[0][0]
            target_dist = target_probs[idx, depth, :]

            for k in range(batch_size):
                # Check if this draft has already been rejected
                if rejected_mask[k]:
                    continue

                r = random.random()
                x = draft_ids[k, depth]
                h = target_dist[x] / draft_probs[k, depth, x]
                if r < h:
                    # Accept the draft token
                    alive_group_id = k
                    rejected_mask |= torch.ne(draft_ids[:, depth], x)
                    break

                # Reject the draft token and adjust probability distribution
                rejected_mask[k] = True
                target_dist = target_dist - draft_probs[k, depth, :]
                target_dist = torch.maximum(target_dist, torch.zeros_like(target_dist))
                target_dist = target_dist / torch.sum(target_dist, dim=-1, keepdim=True)

            if torch.all(rejected_mask):
                # We did not find an acceptable draft token
                break

        # See if we have accepted all the drafts
        if not torch.all(rejected_mask):
            depth = self.max_draft_len
            idx = torch.nonzero(~rejected_mask)[0][0]
            target_dist = target_probs[idx, -1, :]
        
        last_token = torch.multinomial(target_dist, num_samples=1)
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
            accept_count=depth + 1
        )
    
def _func(rho, p, q, k):
    beta = torch.sum(torch.minimum(p, q / rho.unsqueeze(1)), dim=1)
    ret = 1 - rho * beta - (1 - beta) ** k
    return ret

def _get_division_factor(target_dist, draft_dist, num_drafts):
    rho = torch.linspace(1, num_drafts, 100, device=target_dist.device, dtype=target_dist.dtype)
    res = _func(rho, draft_dist, target_dist, num_drafts)
    
    nonzero_idx = torch.nonzero(res <= 0)
    if nonzero_idx.nelement() > 0:
        best_rho = rho[nonzero_idx[0, 0]]
    else:
        best_rho = num_drafts

    best_beta = torch.sum(torch.minimum(draft_dist, target_dist / best_rho))
    return best_rho, best_beta

class SpecTrStrategy:
    def __init__(self, target, drafter, tokenizer, max_draft_len, max_num_drafts):
        self.target = target
        self.drafter = drafter
        self.max_draft_len = max_draft_len
        self.max_num_drafts = max_num_drafts
        self.vocab_size = tokenizer.vocab_size

    @torch.no_grad()
    def generate_draft(self, input_ids, past_key_values, logits_processor):
        batch_size = self.max_num_drafts

        # Duplicate the input ids for batching
        input_ids = input_ids.repeat((batch_size, 1))
        draft_probs = torch.zeros((batch_size, self.max_draft_len, self.vocab_size), 
                                   device=self.drafter.device)

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
            draft_probs[:, i, :] = torch.nn.functional.softmax(logits, dim=-1).squeeze()
            draft_ids = gumbel_sample(logits, None)
            input_ids = torch.cat((input_ids, draft_ids), dim=-1)

        return DraftOutput(
            sequences=input_ids,
            draft_probs=draft_probs,
            draft_past_key_values=past_key_values
        )

    @torch.no_grad()
    def verify_draft(self, input_ids, target_past_key_values, draft_past_key_values, 
                     draft_probs, logits_processor):
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

        outputs = self.target(input_ids=pruned_input_ids, use_cache=True,
                              past_key_values=target_past_key_values, return_dict=True,
                              output_attentions=False, output_hidden_states=False)
        logits = outputs.logits[..., :self.vocab_size]
        target_past_key_values = outputs.past_key_values

        logits = logits_processor(logits[:, (-self.max_draft_len - 1):])
        target_probs = torch.nn.functional.softmax(logits, dim=-1)

        draft_ids = input_ids[:, -self.max_draft_len:]
        assert target_probs.size(1) == draft_ids.size(1) + 1

        rejected_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.target.device)
        alive_group_id = 0
        for depth in range(self.max_draft_len):
            target_dist = target_probs[alive_group_id, depth, :]
            draft_dist = draft_probs[alive_group_id, depth, :]
            num_alive = batch_size - torch.count_nonzero(rejected_mask).item()
            rho, beta = _get_division_factor(target_dist, draft_dist, num_alive)

            for k in range(batch_size):
                # Check if this draft has already been rejected
                if rejected_mask[k]:
                    continue

                r = random.random()
                x = draft_ids[k, depth]
                h = target_dist[x] / (rho * draft_probs[k, depth, x])
                if r < h:
                    # Accept the draft token
                    alive_group_id = k
                    rejected_mask |= torch.ne(draft_ids[:, depth], x)
                    break

                # Reject the draft token
                rejected_mask[k] = True

            if torch.all(rejected_mask):
                # We did not find an acceptable draft token, so we find the residual distribution
                p_acc = 1 - (1 - beta) ** num_alive
                if p_acc > 0.0:
                    residual = ((target_dist - p_acc * torch.minimum(draft_dist, target_dist / rho) / beta)
                                / (1 - p_acc))
                else:
                    residual = target_dist
                
                # This is required because of possible rounding errors, to make sure it is a valid
                # probability distribution for sampling
                residual = torch.clamp(residual, 0.0, 1.0)
                residual = residual / torch.sum(residual)

                break

        # See if we have accepted all the drafts
        if not torch.all(rejected_mask):
            depth = self.max_draft_len
            residual = target_probs[alive_group_id, -1, :]

        last_token = torch.multinomial(residual, num_samples=1)
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
            accept_count=depth + 1
        )
    
class InvariantSingleDraftStrategy:
    def __init__(self, target, drafter, tokenizer, max_draft_len, max_num_drafts):
        self.target = target
        self.drafter = drafter
        self.max_draft_len = max_draft_len
        self.max_num_drafts = 1
        self.vocab_size = tokenizer.vocab_size
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate_draft(self, input_ids, past_key_values, logits_processor, position, randomness):
        rn = randomness[position:(position + self.max_draft_len + 1), 0, :].squeeze(1)

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
            draft_ids = gumbel_sample(logits, rn[i, :])
            input_ids = torch.cat((input_ids, draft_ids), dim=-1)
            
        return DraftOutput(
            sequences=input_ids,
            draft_probs=None,
            draft_past_key_values=past_key_values
        )

    @torch.no_grad()
    def verify_draft(self, input_ids, target_past_key_values, draft_past_key_values, 
                     draft_probs, logits_processor, position, randomness):
        assert draft_past_key_values is not None
        batch_size, input_len = input_ids.shape
        assert batch_size == 1

        if target_past_key_values is not None:
            pruned_input_ids = input_ids[:, target_past_key_values.key_cache[0].size(2):]
        else:
            pruned_input_ids = input_ids
            target_past_key_values = DynamicCache()

        rn = randomness[position:(position + self.max_draft_len + 1), 0, :]

        outputs = self.target(input_ids=pruned_input_ids, use_cache=True,
                              past_key_values=target_past_key_values, return_dict=True,
                              output_attentions=False, output_hidden_states=False)
        logits = outputs.logits[..., :self.vocab_size]
        target_past_key_values = outputs.past_key_values
        
        logits = logits_processor(logits[:, (-self.max_draft_len - 1):])
        target_ids = gumbel_sample(logits, rn)
        draft_ids = input_ids[:, -self.max_draft_len:]
        assert target_ids.size(1) == draft_ids.size(1) + 1

        reject_flag = False
        for depth in range(self.max_draft_len):
            if draft_ids[0, depth] != target_ids[0, depth]:
                reject_flag = True
                break

        if not reject_flag:
            depth = self.max_draft_len
        
        last_token = target_ids[0, depth].unsqueeze(0)
        input_ids = input_ids[0, :(input_len - self.max_draft_len + depth)]
        input_ids = torch.cat((input_ids, last_token)).unsqueeze(0)

        # Adjust cache
        for i in range(len(target_past_key_values)):
            target_past_key_values.key_cache[i] = target_past_key_values.key_cache[i][
                0, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)
            target_past_key_values.value_cache[i] = target_past_key_values.value_cache[i][
                0, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)
        for i in range(len(draft_past_key_values)):
            draft_past_key_values.key_cache[i] = draft_past_key_values.key_cache[i][
                0, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)
            draft_past_key_values.value_cache[i] = draft_past_key_values.value_cache[i][
                0, :, :(input_len - self.max_draft_len + depth), :
            ].unsqueeze(0)

        return VerifyOutput(
            sequences=input_ids,
            target_past_key_values=target_past_key_values,
            draft_past_key_values=draft_past_key_values,
            accept_count=depth + 1
        )

class InvariantMultiDraftStrategy:
    def __init__(self, target, drafter, tokenizer, max_draft_len, max_num_drafts):
        self.target = target
        self.drafter = drafter
        self.max_draft_len = max_draft_len
        self.max_num_drafts = max_num_drafts
        self.vocab_size = tokenizer.vocab_size
        self.randomness = torch.zeros((self.max_num_drafts, self.max_draft_len, self.vocab_size), device=target.device)
    
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
            draft_past_key_values=past_key_values
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

        #target_noise, _ = torch.max(rn[depth, :, :], dim=0)
        #last_token = gumbel_sample(logits[alive_group_id, depth], target_noise).unsqueeze(0)
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
            accept_count=depth + 1
        )

class StrongMultiDraftStrategy:
    def __init__(self, target, drafter, tokenizer, max_draft_len, max_num_drafts):
        self.target = target
        self.drafter = drafter
        self.max_draft_len = max_draft_len
        self.max_num_drafts = max_num_drafts
        self.vocab_size = tokenizer.vocab_size
        self.randomness = torch.zeros((self.max_num_drafts, self.max_draft_len, self.vocab_size), device=target.device)
    
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
            draft_past_key_values=past_key_values
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
            target_noise, _ = torch.max(rn[depth, :, :], dim=0)
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

        target_noise, _ = torch.max(rn[depth, :, :], dim=0)
        last_token = gumbel_sample(logits[alive_group_id, depth], target_noise).unsqueeze(0)
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
            accept_count=depth + 1
        )


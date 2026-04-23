from collections import namedtuple
import torch

DraftOutput = namedtuple('DraftOutput', [
    'sequences', 
    'draft_probs', 
    'draft_past_key_values'
    ])
VerifyOutput = namedtuple('VerifyOutput', [
    'sequences', 
    'target_past_key_values', 
    'draft_past_key_values', 
    'accept_count'
    ])
GeneratorOutput = namedtuple('SpeculativeGeneratorOutput', [
    'sequences',
    'acceptance_rate',
    'token_rate',
    'avg_generation_time',
    'avg_verification_time',
    'num_invocations',
    'total_time'
])

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(noise):
    return -log(-log(noise))

def gumbel_sample(logits, noise=None, dim=-1):
    if noise == None:
        noise = torch.zeros_like(logits).uniform_(0, 1)
        print(noise)
        print(logits)
    return (logits + gumbel_noise(noise)).argmax(dim=dim)

def poisson_race_sample(logits, noise=None, alpha=1.0, top_k=None):

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs = torch.softmax(logits, dim=-1)
    if top_k is not None and top_k < probs.size(-1):
        vals, idx = torch.topk(probs, top_k, dim=-1)
        probs = vals
    else:
        idx = None

    if noise is None:
        noise = -torch.log(torch.rand_like(probs))
    tilde = noise / probs.clamp_min(1e-20)   # E_i / p(i)
    weights = torch.pow(tilde, -alpha)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    sample = torch.multinomial(weights, 1).squeeze(-1)
    if idx is not None:
        sample = idx.gather(-1, sample.unsqueeze(-1)).squeeze(-1)
    return sample, noise

# def poisson_race_sample_from_probs(probs, noise=None, alpha=1.0):
#     # probs: [B, M]  (M=V 或 M=topk)
#     if noise is None:
#         noise = -torch.log(torch.rand_like(probs))
#     tilde = noise / probs.clamp_min(1e-20)
#     w = tilde.pow(-alpha)
#     w = w / w.sum(dim=-1, keepdim=True)
#     sample = torch.multinomial(w, 1).squeeze(-1)  # [B]
#     return sample, noise

def poisson_race_sample_from_probs(probs, noise=None, eps=1e-20):
    probs = probs.clamp_min(eps)
    if noise is None:
        noise = -torch.log(torch.rand_like(probs).clamp_min(eps))  # Exp(1)
    tilde = noise / probs
    sample = tilde.argmin(dim=-1)
    return sample, noise


def normalized_poisson_race_sample_from_probs(probs, noise=None, alpha=1.0, eps=1e-20):
    """
    probs: [B, M] probabilities (must be >=0)
    noise: [B, M] Exp(1) noise; if None, sample Exp(1)
    alpha: float
    """
    probs = probs.clamp_min(eps)

    if noise is None:
        noise = -torch.log(torch.rand_like(probs).clamp_min(eps))

    # tilde = noise / probs  (positive)
    tilde = noise / probs

    # Stable weights: log_w = -alpha * log(tilde)
    log_w = -float(alpha) * torch.log(tilde.clamp_min(eps))

    # Normalize in log-space
    log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)

    # Back to prob space for multinomial
    w = torch.exp(log_w).clamp_min(0.0)

    # Safety: avoid all-zero row (can happen if everything underflows)
    row_sum = w.sum(dim=-1, keepdim=True)
    w = torch.where(row_sum > 0, w / row_sum, torch.full_like(w, 1.0 / w.size(-1)))

    sample = torch.multinomial(w, 1).squeeze(-1)
    return sample, noise


class LogitsProcessor:
    def __init__(self, temperature, top_p, top_k):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def __call__(self, logits):
        val, ind = torch.topk(logits, self.top_k, sorted=False, dim=-1)
        probs = torch.full_like(logits, float('-inf'))
        probs.scatter_(-1, ind, val)
        return probs / self.temperature

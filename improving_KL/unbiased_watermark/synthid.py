from collections.abc import Callable
import numpy as np
import torch
from torch import FloatTensor, Tensor

from . import AbstractWatermarkCode, AbstractReweight

# keys = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990, 966, 226, 324, 585, 118, 504, 421, 521, 129, 669, 732, 225, 90, 960]

def get_synthid_variables(rng, vocab_size, d):
    # Generate d random integers between 0 and 1000 as seeds
    seeds = rng.integers(0, 1000, size=(d,))
    
    # Create d random generators using the seeds
    generators = [np.random.Generator(np.random.PCG64(seed)) for seed in seeds]
    
    # Generate random binary numbers for each generator
    binary_matrix = np.zeros((vocab_size, d), dtype=np.float32)
    for i, gen in enumerate(generators):
        binary_matrix[:, i] = gen.integers(0, 2, size=(vocab_size,))
    
    return binary_matrix


def accumulate_hash(
    current_hash: torch.LongTensor,
    data: torch.LongTensor,
    multiplier: int = 6364136223846793005,
    increment: int = 1,
) -> torch.LongTensor:
  """Accumulate hash of data on current hash.

  Method uses adapted linear congruential generator with newlib/musl parameters.

  This function has following property -
  f(x, data[T]) = f(f(x, data[:T - 1]), data[T])

  This function expects current_hash.shape and data.shape[:-1] to
  match/broadcastable.

  Args:
    current_hash: (shape,)
    data: (shape, tensor_len)
    multiplier: (int) multiplier of linear congruential generator
    increment: (int) increment of linear congruential generator

  Returns:
    updated hash (shape,)
  """
  for i in range(data.shape[-1]):
    current_hash = torch.add(current_hash, data[..., i])
    current_hash = torch.mul(current_hash, multiplier)
    current_hash = torch.add(current_hash, increment)
  return current_hash


class SynthID_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, binary_matrix: FloatTensor):
        self.binary_matrix = binary_matrix

    @classmethod
    def from_random_(
        cls,
        rng: np.ndarray,  # dtype=object, a nprandom.Generator
        vocab_size: int,
        d: int = 30,  # dimension of the watermark, set default=30
    ):
        binary_matrix = np.empty(rng.shape + (vocab_size, d), dtype=np.float32)
        for index in np.ndindex(rng.shape):
            binary_matrix[index] = get_synthid_variables(rng[index], vocab_size, d)
        return cls(torch.tensor(binary_matrix))

    def tensor_shape_map(
        self,
        shape_map: Callable[[Tensor], Tensor],
    ):
        shape_map = torch.func.vmap(shape_map, in_dims=-1, out_dims=-1)
        return self.__class__(shape_map(self.binary_matrix))

    @classmethod
    def stack(
        cls,
        codes: list["SynthID_WatermarkCode"],
        dim: int,
    ) -> "SynthID_WatermarkCode":
        if dim < 0:
            dim -= 2
        #  when dim = -2, the shape of the result is (batch_size, n, vocab_size, d), where n is the generated token length
        return cls(torch.stack([code.binary_matrix for code in codes], dim=dim))

    @classmethod
    def concat(
        cls,
        codes: list["SynthID_WatermarkCode"],
        dim: int,
    ) -> "SynthID_WatermarkCode":
        if dim < 0:
            dim -= 2
        #  when dim = -2, the shape of the result is (batch_size, n+1, vocab_size, d), where n is the generated token length
        return cls(torch.concat([code.binary_matrix for code in codes], dim=dim))


class SynthID_Reweight(AbstractReweight):
    watermark_code_type = SynthID_WatermarkCode

    def __repr__(self):
        return f"SynthID_Reweight()"

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor, input_is_probs: bool = False
    ) -> FloatTensor:
        assert isinstance(code, self.watermark_code_type)
        assert p_logits.shape == code.binary_matrix.shape[:-1]
        
        # Get the depth (d) from the watermark code shape
        depth = code.binary_matrix.shape[-1]
        device = p_logits.device

        # Convert logits to probabilities
        if input_is_probs:
            probs = p_logits
        else:
            probs = torch.softmax(p_logits, dim=-1)

        # Apply watermarking for each dimension
        for i in range(depth):
            g_values_at_depth = code.binary_matrix[..., i]
            g_mass_at_depth = (g_values_at_depth * probs).sum(dim=-1, keepdim=True)
            probs = probs * (1 + g_values_at_depth - g_mass_at_depth)
        # Convert back to log space
        log_probs = torch.log(probs)
        log_probs = torch.where(
            torch.isfinite(log_probs), log_probs, torch.tensor(-1e4, device=device)
        )
        return log_probs    # shape (batch_size, n, vocab_size)


class SynthID_Reweight_fast():
    def __init__(self,
                 sampling_table_size: int,
                 sampling_table_seed: int,
                 device: torch.device,
                 ngram_len: int,
                 private_key: int,
                 watermarking_depth: int = 30,
                 ):
        base_keys = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990, 966, 226, 324, 585, 118, 504, 421, 521, 129, 669, 732, 225, 90, 960]
        if watermarking_depth <= len(base_keys):
            self.keys = base_keys[:watermarking_depth]
        else:
            extra_keys = []
            state = 104729
            while len(base_keys) + len(extra_keys) < watermarking_depth:
                state = (1103515245 * state + 12345) % 1000003
                extra_keys.append(int(state))
            self.keys = base_keys + extra_keys
        generator = torch.Generator(device=device).manual_seed(sampling_table_seed)
        self.sampling_table = torch.randint(
        low=0,
        high=2,
        size=(sampling_table_size,),
        generator=generator,
        device=device,
    )
        self.sampling_table_seed = sampling_table_seed
        self.ngram_len = ngram_len
        self.private_key = private_key
    def reweight_logits(
        self, g_values: torch.LongTensor, scores: torch.FloatTensor
    ) -> FloatTensor:
        """
        :param g_values: (batch_size, top_k, depth)
        :param scores: (batch_size, top_k)
        :return: (batch_size, top_k)
        """
        _, _, depth = g_values.shape
        device = scores.device

        probs = torch.softmax(scores, dim=1)

        for i in range(depth):
            g_values_at_depth = g_values[:, :, i]
            g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
            probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

        log_probs = torch.log(probs)
        log_probs = torch.where(
            torch.isfinite(log_probs), log_probs, torch.tensor(-1e4, device=device)
        )
        return log_probs
        
    
    def _compute_keys(self, context: torch.LongTensor, top_k_indices: torch.LongTensor) -> torch.LongTensor:
        """
        Use ngram context and the token indices to compute the hash of the ngram, so the actual ngram length is ngram_len + 1

        :param context: (batch_size, n_grams)
        :param top_k_indices: (batch_size, top_k)
        :return: (batch_size, top_k, depth)
        """
        device = top_k_indices.device
        batch_size, _ = top_k_indices.shape
        hash_result = torch.ones(batch_size, device=device, dtype=torch.long)
        hash_result_with_just_context = accumulate_hash(
            hash_result, context
        )

        hash_result = torch.vmap(
            accumulate_hash, in_dims=(None, 1), out_dims=1
        )(hash_result_with_just_context, top_k_indices[:, :, None]) # shape (batch_size, top_k)
        # use global keys and private key to generate wm_keys
        wm_keys = [key + self.private_key for key in self.keys]
        wm_keys = torch.tensor(wm_keys, dtype=torch.long, device=device)[None, None, :, None]
        hash_result = torch.vmap(
            accumulate_hash, in_dims=(None, 2), out_dims=2
        )(hash_result, wm_keys)
        # hash_result shape [batch_size, top_k, depth]
        return hash_result
    
    def sample_g_values(self, ngram_keys: torch.LongTensor) -> torch.LongTensor:
        """Samples g values from Bernoulli distribution.

        It is not possible to pass random keys in a vectorized way in torch. Instead
        we pre-compute a random sampling table, and use apply modulo table size to
        map from ngram keys (int64) to g values.
        """
        (sampling_table_size,) = self.sampling_table.shape
        sampling_table = self.sampling_table.reshape((1, 1, sampling_table_size))
        ngram_keys = ngram_keys % sampling_table_size
        return torch.take_along_dim(sampling_table, indices=ngram_keys, dim=2)
    
    def compute_g_values(self, input_ids: torch.LongTensor, prompt_ids: torch.LongTensor = None) -> torch.LongTensor:
        """
        :param input_ids: (batch_size, input_len)
        :param prompt_ids: (batch_size, prompt_len)
        :return: (batch_size, input_len, depth)
        """
        batch_size, _ = input_ids.shape
        if prompt_ids is None or prompt_ids.shape[1] < self.ngram_len:
            # pad input_ids with zeros on the left with length ngram_len
            inputs = torch.cat([torch.zeros(batch_size, self.ngram_len, device=input_ids.device, dtype=torch.long), input_ids], dim=1)
        else:
            inputs = torch.cat([prompt_ids[:, -self.ngram_len:], input_ids], dim=1)
        assert inputs.shape[1] == self.ngram_len + input_ids.shape[1], f"get inputs shape {inputs.shape[1]} != {self.ngram_len + input_ids.shape[1]}"
        ngrams = inputs.unfold(dimension=1, size=self.ngram_len + 1, step=1)
        assert ngrams.shape[1] == input_ids.shape[1], f"get ngrams shape {ngrams.shape[1]} != {input_ids.shape[1]}"
        ngram_keys = self.compute_ngram_keys(ngrams)
        return self.sample_g_values(ngram_keys)
    
    def compute_ngram_keys(self, ngrams: torch.LongTensor) -> torch.LongTensor:
        """
        Note that the actual ngram length is ngram_len + 1
        :param ngrams: (batch_size, input_len, ngram_len+1)
        :return: (batch_size, input_len, depth)
        """
        batch_size, input_len, ngram_len = ngrams.shape
        hash_result = torch.ones(batch_size, device=ngrams.device, dtype=torch.long)
        hash_result = torch.vmap(
            accumulate_hash, in_dims=(None, 1), out_dims=1
        )(hash_result, ngrams)
        # hash_result shape [batch_size, input_len]
        wm_keys = [key + self.private_key for key in self.keys]
        wm_keys = torch.tensor(wm_keys, dtype=torch.long, device=ngrams.device)[None, None, :, None]
        hash_result = torch.vmap(
            accumulate_hash, in_dims=(None, 2), out_dims=2
        )(hash_result, wm_keys)
        # hash_result shape [batch_size, input_len, depth]
        return hash_result

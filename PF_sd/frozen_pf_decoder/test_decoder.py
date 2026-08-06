from __future__ import annotations

import unittest

import torch

from PF_sd.frozen_pf_decoder.core.max_order_pf import (
    recover_max_order_pivots,
    speculative_max_order_pf_generator,
)
from PF_sd.frozen_pf_decoder.core.test_core import ToyLM, _drain
from PF_sd.frozen_pf_decoder.decoder import (
    speculative_tree_free_latin_pf_generator,
    verify_draft_matrix,
)


class TreeFreeLatinPFTests(unittest.TestCase):
    def test_alive_mask_is_equivalent_to_prefix_membership(self):
        drafts = [[1, 2, 3], [1, 4, 5], [8, 9, 0]]
        self.assertEqual(verify_draft_matrix(drafts, [1, 4, 7]), (2, 1))
        self.assertEqual(verify_draft_matrix(drafts, [8, 9, 0, 6]), (3, 2))
        self.assertEqual(verify_draft_matrix(drafts, [7]), (0, 0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/Triton required")
    def test_toy_tree_and_list_paths_match(self):
        device = torch.device("cuda")
        target = ToyLM(vocab_size=17, shift=0, scale=0.7).to(device)
        draft = ToyLM(vocab_size=17, shift=1, scale=0.8).to(device)
        input_ids = torch.tensor([[2, 5, 1]], device=device)
        common = dict(
            lookahead=3,
            width=4,
            max_length=24,
            private_key=b"tree-free-test",
            process_logits_kwargs={"temperature": 1.0, "top_k": 8, "top_p": 1.0},
            return_meta=True,
            return_logprobs=False,
            record_pivots=True,
        )
        tree = _drain(
            speculative_max_order_pf_generator(
                target,
                draft,
                input_ids.clone(),
                target_coupling="latin_hypercube",
                rng_backend="counter_philox",
                fuse_latin_sampling=True,
                **common,
            ),
        )
        tree_free = _drain(
            speculative_tree_free_latin_pf_generator(
                target, draft, input_ids.clone(), **common
            ),
        )
        self.assertEqual(tree_free[0], tree[0])
        self.assertEqual(tree_free[1], tree[1])
        self.assertTrue(torch.equal(tree_free[2], tree[2]))
        tree_labels = tree[1]
        list_labels = tree_free[1]
        tree_pivots = recover_max_order_pivots(
            out_ids=torch.tensor([tree[0]], dtype=torch.long),
            context_labels=tree_labels,
            width=common["width"],
            private_key=common["private_key"],
            vocab_size=17,
            device=device,
            target_coupling="latin_hypercube",
            rng_backend="counter_philox",
        )
        list_pivots = recover_max_order_pivots(
            out_ids=torch.tensor([tree_free[0]], dtype=torch.long),
            context_labels=list_labels,
            width=common["width"],
            private_key=common["private_key"],
            vocab_size=17,
            device=device,
            target_coupling="latin_hypercube",
            rng_backend="counter_philox",
        )
        self.assertTrue(torch.equal(list_pivots.cpu(), tree_pivots.cpu()))


if __name__ == "__main__":
    unittest.main()

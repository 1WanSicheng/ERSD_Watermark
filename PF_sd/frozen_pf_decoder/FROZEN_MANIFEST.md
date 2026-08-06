# Frozen manifest

- Freeze date: 2026-08-06
- Backend: optimized independent Latin-PF, exact context-seed reuse, prefix
  compaction, and KV-cache view selection
- Correctness: fixed-key output, AATPS, and watermark preserving

## SHA-256

| File | SHA-256 |
|---|---|
| `decoder.py` | `8fcfeecef790c75badec244f8e8d4e75d7758c3de17f1f6fab7be9b5d5331a51` |
| `core/max_order_pf.py` | `197f43ca897796312de84984341dda3c377d4878c33f55b7d1e3b8aaf6fb6f99` |
| `benchmark.py` | `6353fd12a4be3b3ea97f69b8dcf9923956326c25ba0461652428213f8efa0890` |
| `configs/large_scale_sweep.json` | `3233ce2511048262fd9396501e5d6cf92bf200d74fccc7e7e6754c5f1da9d592` |
| `configs/vicuna_b2.json` | `00190a5647c73942d6a0a68d04c409f2c50f1b662a836772dee9025e2a8f09fd` |
| `configs/vicuna_b4.json` | `fc9280803869b2b415d3e373352e0b821ee6f497b6a4fb29e921e03cfd4e3588` |
| `results/vicuna_b2_n100_kvview.json` | `c6d783becfc7714aa798f6185f598a82fdcf29a8541cc9a23d8b5accd495b048` |
| `results/vicuna_b2_n100_seed107_kvview.json` | `c464aab66de06c4aba53884347e7d803a35755cb83bae0bb91c65f9a6ec33fbe` |
| `results/vicuna_b4_n100_kvview.json` | `a7c3d964ee971df55246b6f2c1b12a74e5af5fbf935518dfe6bd17496055a554` |
| `results/vicuna_b4_n100_seed107_kvview.json` | `62a8f6e0fc1e07d3af350212c01a95e0e8abfc3f79940f2647e259c61a49095f` |

## External runtime dependencies

| File | SHA-256 |
|---|---|
| `MPFR_spec/mpfr_batched_torchgen_cached.py` | `7878050509ac401e0d5e06f3ba3e4d46708afdceccd175ece1b42b50690c955e` |
| `MPFR_spec/mpfr_direct_optimized.py` | `2d75fc061a5f5750445627bbbac99125e02b41f3a90d0e16214927062a691c56` |
| `accuwm/multi_draft_utils.py` | `1f3f605a98f0d6cce0b5f4f718d1333b1c4a47e26f4836b4bd161f6189869ea6` |
| `accuwm/pfr.py` | `b8e90cc860330d1479c5170a9a694a87f108d847228d5c25088450281ffcbfbc` |
| `accuwm/utils.py` | `ad23ae0c473055ed5013bcc5d6248acdced37f1ded41ef13542ef193bc0a360d` |
| `experiments/_shared.py` | `e10832be824cb41697e9c34873567393cb653a69f22c5445a1b657acb9591aef` |

Changing any hashed file creates a new backend version and requires rerunning
the exactness and runtime validation.

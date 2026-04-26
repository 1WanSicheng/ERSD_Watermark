# 对话关键总结（截至 2026-04-24）

## 1. 目标与工作主线

本轮对话主要围绕以下几条主线展开：

1. 在 `AcceleratedUnbiasedWatermark-main` 中部署 LLaMA / Qwen 模型，并基于现有代码实现、验证和对比多种 watermarkable speculative decoding 方法。
2. 在 `accuwm` 中实现并迭代 `multi_draft_pfr`，重点考察：
   - AATPS / BE / TR / token rate
   - single draft PFR 与 multi-draft PFR 的关系
   - 与 `SpeculativeDecoding` 目录下 strategy 实现的对比
3. 增加文本质量评估指标：
   - `log perplexity`
   - `ROUGE-1/2/L`
   - `BLEU`
4. 跑 GSM8K 上的大规模实验，并整理结论。

---

## 2. 代码与脚本新增/修改

### 2.1 `accuwm` 侧

- 新增 active-set 版本实现：
  - [accuwm/active_set_multi_draft_pfr.py](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/accuwm/active_set_multi_draft_pfr.py)
- 调整包导出：
  - [accuwm/__init__.py](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/accuwm/__init__.py)

说明：
- `active_set_multi_draft_pfr.py` 是按“draft-indexed keyed source + active set”版本算法实现的。
- `accuwm/__init__.py` 后来做过一次修正：去掉了对当前目录中不存在模块的 import，避免 `import accuwm` 直接失败。

### 2.2 `experiments` 侧

- 新增质量评测脚本：
  - [experiments/run_pfr_quality.py](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/experiments/run_pfr_quality.py)
  - [experiments/run_mc_uwm_quality.py](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/experiments/run_mc_uwm_quality.py)
- 新增 active-set 实验脚本：
  - [experiments/run_active_set_multi_draft_pfr_aatps.py](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/experiments/run_active_set_multi_draft_pfr_aatps.py)

### 2.3 运行脚本

- 新增串行跑 GSM8K PFR 质量实验脚本：
  - [scripts/run_gsm8k_pfr_quality_seq.sh](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/scripts/run_gsm8k_pfr_quality_seq.sh)

作用：
- 顺序跑 `L=1,2,4,6` 四组 single PFR 质量实验；
- 每组 `GSM8K + 1000 samples`；
- 避免多卡/多进程同时 load 大模型导致的不稳定。

---

## 3. Multi-draft PFR 的关键实现与理解

### 3.1 当前 `multi_draft_pfr` 的语义

当前主版本 `multi_draft_pfr` 使用的是：

- 对每个 context 生成 keyed shared source
- 在 draft model `Q(.|c)` 上运行 `MS-PFR`
- 取 merged Poisson race 的前 `B` 个 arrivals
- 对重复 token 做 collapse，形成 branch multiplicity

也就是说：

- `B` 个 draft 不是先生成 `B` 条完整序列再比较；
- 而是在每个 context 上，通过 `MS-PFR` 取前 `B` 个“到达”；
- 重复 token 会被合并成 multiplicity，后续树扩展时继续传播。

### 3.2 为什么会有重复 token

这个问题在对话中专门澄清过。

如果实现的是：

- 每个 token 只有一个 arrival time；
- 然后在整个词表上做一次 race，取 top-B，

那么 token 不会重复。

但当前主版本实现的是：

- 每个 token 对应一个 Poisson process；
- 每个 token 可以有多次 arrival；
- 再从所有 token 的所有 arrivals 中取全局前 `B` 个；

因此同一个 token 可能重复出现。

### 3.3 one-race top-B 版本的尝试与结论

曾尝试将 `MS-PFR` 改为：

- 每个 token 只采一个 arrival time；
- 按 `Exp(1) / p_v` 排序；
- 取 top-B distinct tokens。

这个版本的动机是：

- 更贴近“one race top-B”的直觉；
- 不出现重复 token。

但 200 条 GSM8K 结果表明，这个方向不适合当前的 tree + multiplicity 实现：

- tree 变得很宽；
- `draft_tree_size_mean` 明显上升；
- BE 下降；
- TR 下降。

因此最后已回退到原来的 arrival-matrix 版本，并确认回退后的 200 sample 结果回到之前水平。

结论：

- 当前项目中，`multi_draft_pfr` 继续保留“merged arrivals + multiplicity collapse”的版本。

---

## 4. 为什么 `multi_draft_pfr (B=1)` 的 TR 明显低于 `pfr`

这个问题是本轮对话中的一个核心分析点。

### 4.1 现象

在 `outputs/gsm8k_l4_1000_pfr_strategy_summary.json` 一类实验中，观察到：

- `multi_draft_pfr, B=1` 的 BE 与 `pfr` 基本一致；
- 但 `TR` 明显低于 `pfr`。

### 4.2 结论

原因不是算法语义错，而是实现没有退化成 single-draft 的高效路径。

虽然 `B=1` 时，proposal tree 理论上退化成单路径，但实现仍走通用 tree 逻辑，带来额外开销：

1. 仍在构建 tree / `ContextKey` / dict / set / multiplicity；
2. target verify 没有像 `pfr.py` 那样“一次 target forward 验整段”；
3. 仍走通用 `MS-PFR` 路径，而不是 single PFR 的最简 winner 逻辑。

### 4.3 单路径为何理论上可以退化

如果 tree 宽度为 1，那么语义上应退化为 single-draft PFR：

- accept rule 一致；
- output distribution 一致；
- BE 一致。

这也和实验吻合。

### 4.4 为什么实现上没有退化

原因在于 target verify 的执行方式不同：

- `pfr.py`：一次 target forward 即可获得一整段 logits，用于验证整个 speculative block；
- `multi_draft_pfr B=1`：即使 tree 已退化成链，仍然是按 context 逐个 evaluate target context。

因此：

- 语义退化了；
- 计算图没有退化。

### 4.5 优化方向

对话中的结论是：

- 最直接可行的优化，是给 `multi_draft_pfr` 增加 `B=1` fast path；
- 或者在“tree 退化为单路径”时，走 batched target verify。

---

## 5. Active-set Multi-draft PFR 的实现与结论

按用户给出的伪代码，新实现了 active-set 版本：

- 不再构 proposal tree；
- 而是生成 `B` 条 draft trajectories；
- target 侧为每个 `(s, b)` 评估 mapped sample 和 mapped time；
- 每一步在 active set 中选最小 mapped time 的流；
- 仅保留 token 匹配的 draft streams。

### 5.1 20 sample 结果（GSM8K, Qwen, `B=8, L=4`）

结果文件：

- `outputs/gsm8k_active_set_multi_draft_pfr_B8_l4_20.json`
- `outputs/gsm8k_active_set_multi_draft_pfr_B8_l4_20.jsonl`

主要结果：

- `BE_mean ≈ 4.7058`
- `TR_global ≈ 6.55 tok/s`

### 5.2 结论

这个版本的 BE 和原 multi-draft PFR 接近，但速度很差。

根因：

- 每个 block 需要评估约 `B * (L+1)` 个 target contexts；
- 而原 multi-draft PFR 只沿 realized path 做 lazy verification；
- target 侧开销被放大很多。

因此该方向暂时搁置。

---

## 6. 与 `SpeculativeDecoding` 中 strategy 的对比

### 6.1 统计口径对齐

对话中先明确了需要对齐的统计量：

- `BE`
- `RE`
- `TR`

其中后来明确了 `TR` 的定义应改为：

```text
TR_improvement = (tr - tr_single) / tr_single * 100%
```

也就是：

- multi-draft 相对 single-draft baseline 的 token rate 提升百分比；
- 这样能减少实现细节差异对结论的影响。

### 6.2 Baseline 的澄清

后来又明确了：

- 对 `InvariantMultiDraftStrategy` 和 `StrongMultiDraftStrategy` 的比较；
- baseline 应使用 `SingleDraftStrategy`；
- 而不是把 `B=1` 的 multidraft 实现直接拿来当 baseline。

### 6.3 关于旧结果和新结果不一致的分析

用户指出过：

- 旧实验中 `TR` 多为 `40+`
- 新实验中绝对 `TR` 到了 `60+`

分析结论：

- 不是生成长度变化导致；
- 主要是运行耗时变快了；
- BE 基本一致，说明算法行为没有明显变化；
- 差异更像是 runner / timing 口径、warmup、GPU 负载或运行环境差异。

---

## 7. 文本质量指标扩展

### 7.1 Single PFR 新增指标

在 [experiments/run_pfr_quality.py](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/experiments/run_pfr_quality.py) 中为 single PFR 增加了：

- `log_perplexity`
- `perplexity`
- `ROUGE-1 F1`
- `ROUGE-2 F1`
- `ROUGE-L F1`
- `BLEU`

说明：

- ROUGE 在脚本内自行实现；
- BLEU 采用 `nltk`；
- 输出同时保留 `prediction` 和 `reference`。

### 7.2 `mc_uwm_speed / mc_uwm_strength` 新增指标

在 [experiments/run_mc_uwm_quality.py](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/experiments/run_mc_uwm_quality.py) 中为：

- `mc_uwm_speed`
- `mc_uwm_strength`

增加了同样的质量指标。

注意：

- 当前 `mc_uwm` 质量脚本要求 target / draft 在同一张 GPU 上。

---

## 8. 代表性实验结果

### 8.1 PFR 与 `mc_uwm_*` 的 20-sample 质量对比（GSM8K, L=4）

#### `pfr`

结果文件：

- [outputs/gsm8k_pfr_quality_l4_20.json](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/outputs/gsm8k_pfr_quality_l4_20.json)

摘要：

- `TR_global = 44.8153`
- `BE_mean = 3.9811`
- `log_perplexity_mean = 0.1322`
- `ROUGE-1 = 0.3392`
- `ROUGE-2 = 0.1337`
- `ROUGE-L = 0.2162`
- `BLEU = 0.0682`

#### `mc_uwm_speed`

结果文件：

- [outputs/gsm8k_mc_uwm_quality_l4_20.json](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/outputs/gsm8k_mc_uwm_quality_l4_20.json)

摘要：

- `TR_global = 44.9008`
- `BE_mean = 4.1288`
- `log_perplexity_mean = 0.1377`
- `ROUGE-1 = 0.3379`
- `ROUGE-2 = 0.1273`
- `ROUGE-L = 0.2097`
- `BLEU = 0.0652`

#### `mc_uwm_strength`

同文件摘要：

- `TR_global = 43.4372`
- `BE_mean = 4.1143`
- `log_perplexity_mean = 0.1147`
- `ROUGE-1 = 0.3434`
- `ROUGE-2 = 0.1362`
- `ROUGE-L = 0.2104`
- `BLEU = 0.0707`

#### 结论

在这 20 条上：

- `pfr` 与 `mc_uwm_speed` 的速度几乎一致；
- `mc_uwm_strength` 略慢；
- `mc_uwm_strength` 的文本质量指标略好一些；
- 三者差异不大，20 条只足够看趋势。

---

## 9. Single PFR 的 1000-sample 质量实验（GSM8K）

按用户要求，后来又跑了 1000 sample，并对比 `L=1,2,4,6`。

结果文件：

- [outputs/gsm8k_pfr_quality_l1_1000.json](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/outputs/gsm8k_pfr_quality_l1_1000.json)
- [outputs/gsm8k_pfr_quality_l2_1000.json](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/outputs/gsm8k_pfr_quality_l2_1000.json)
- [outputs/gsm8k_pfr_quality_l4_1000.json](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/outputs/gsm8k_pfr_quality_l4_1000.json)
- [outputs/gsm8k_pfr_quality_l6_1000.json](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/outputs/gsm8k_pfr_quality_l6_1000.json)

### 9.1 结果汇总

| L | TR_global | BE_mean | log_ppl | ppl | R1 | R2 | RL | BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35.7545 | 1.9013 | 0.10781 | 1.11566 | 0.36008 | 0.14052 | 0.22622 | 0.07291 |
| 2 | 41.4882 | 2.7195 | 0.10781 | 1.11566 | 0.36008 | 0.14052 | 0.22622 | 0.07291 |
| 4 | 45.8878 | 4.1270 | 0.10781 | 1.11566 | 0.36010 | 0.14045 | 0.22630 | 0.07284 |
| 6 | 46.9305 | 5.2861 | 0.10778 | 1.11563 | 0.36013 | 0.14052 | 0.22628 | 0.07288 |

### 9.2 结论

结论比较明确：

1. `L` 增大时，`TR_global` 提升明显，但在 `L=4 -> 6` 时已经开始趋缓。
2. `BE_mean` 随 `L` 增大而稳定上升。
3. 文本质量指标几乎不变：
   - `log perplexity`
   - `ROUGE`
   - `BLEU`

这说明：

- 在当前 single PFR 配置下，增大 lookahead 主要影响的是推理效率和块接受行为；
- 对最终文本质量没有明显负面影响。

---

## 10. 进程与实验调度上的经验

对话中还踩到了一个实际运行层面的坑：

- 同时起多个大模型实验进程时，容易在模型加载阶段失败或表现不稳定；
- 尤其是多张卡并发 load checkpoint 时，容易出现只写了日志开头但没有稳定跑完的情况。

因此后面改为：

- 停掉旧后台任务；
- 用单个串行脚本按顺序跑；
- 避免一夜之间出现半成品结果。

对应脚本：

- [scripts/run_gsm8k_pfr_quality_seq.sh](/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/scripts/run_gsm8k_pfr_quality_seq.sh)

---

## 11. 当前明确的结论

截至目前，可以认为已经比较明确的结论有：

1. `multi_draft_pfr` 当前主版本应保留“merged arrivals + multiplicity collapse”的实现。
2. one-race top-B distinct token 版本不适合现有 tree 结构，200 sample 已验证。
3. `multi_draft_pfr B=1` 与 `pfr` 的 BE 一致，说明语义一致；TR 较低是实现未走 single-path fast path，而不是算法错误。
4. active-set multi-draft PFR 虽然 BE 不差，但 target 评估过多，TR 太低，暂不继续。
5. 对 `SpeculativeDecoding` strategy 的比较中，TR 统计应使用相对 `SingleDraftStrategy` 的提升百分比。
6. 对 single PFR 而言，增大 lookahead 能显著提升 TR / BE，但对 `log perplexity / ROUGE / BLEU` 几乎没有影响。

---

## 12. 后续建议

如果继续推进，优先级建议如下：

1. 给 `multi_draft_pfr` 加 `B=1` fast path，直接复用 `pfr.py` 的 batched verify 思路。
2. 在 `L=4` 固定下，把质量评估从 single PFR 扩展到：
   - `multi_draft_pfr (B=1,4,8)`
   - `InvariantMultiDraftStrategy`
   - `StrongMultiDraftStrategy`
3. 如果要继续追 `multi_draft_pfr` 的 TR，重点应放在：
   - target verify batching
   - 减少 context / tuple / dict 转换开销
   - 降低 `MS-PFR` 核心算子的实现成本


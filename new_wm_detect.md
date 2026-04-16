# Poisson Race 水印检测统计量：可行性验证实验方案

## 1. 研究假设

**核心假设：** 在 Poisson race 框架下，到达时间 $T_{(1)}, T_{(2)}$ 携带的检测信号独立于经典的 Aaronson 统计量 $Z_t$，联合使用可以显著提升水印检测功效。

**具体子假设：**

- H-A：到达时间比值 $R_t = T_{(1)}/T_{(2)}$ 在 $H_0$ 和 $H_1$ 下的分布存在可观测的偏移
- H-B：$R_t$ 与经典统计量 $Z_t$ 的相关性较低（$|\rho| < 0.5$），即两者携带近似正交的信息
- H-C：联合统计量 $S_{\text{joint}}$ 的 TPR 严格优于单独的 $S_Z$，尤其在高熵 regime 下

---

## 2. 实验前置：Poisson Race 回顾

从分布 $P = (P_w)_{w \in \mathcal{W}}$ 采样，等价于：

```
对每个 token w:
    E_w ~ Exp(1)          # 指数随机变量
    T_w = E_w / P_w       # 到达时间
输出 w* = argmin_w T_w
```

等价变换：令 $U_w \sim \text{Uniform}(0,1)$，则 $E_w = -\log U_w$。

水印化：将 $U_w$ 替换为伪随机数 $U_w = G(\zeta_w)$，整个过程变为 $\zeta$ 的确定性函数。

---

## 3. 假设检验框架

### 3.1 两个假设

| | 描述 | token 与 seed 的关系 |
|---|---|---|
| $H_0$ | 无水印 | token 从 $P$ 独立采样，与 seed $\zeta$ 无关 |
| $H_1$ | 有水印 | token 是 Poisson race 的赢家，由 seed $\zeta$ 确定 |

### 3.2 检测者的能力

检测者知道 seed，因此可以复原每个 token 位置的 $\{U_w\}_{w \in \mathcal{W}}$，进而计算所有到达时间 $\{T_w\}$。
检测者观测到生成的 token 序列 $w_1, w_2, \ldots, w_n$。

---

## 4. 候选统计量定义

对每个 token 位置 $t$，检测者可以计算以下量：

### 统计量 S1：经典 Aaronson 统计量

```
y_t = U_{w_t}^{1/P_{w_t}}
Z_t = -log(1 - y_t)
```

- $H_0$ 下：$y_t \sim \text{Uniform}(0,1)$，$Z_t \sim \text{Exp}(1)$
- $H_1$ 下：$y_t$ 偏向 1，$Z_t$ 偏大

序列级统计量：$S_Z = \sum_{t=1}^n Z_t$

### 统计量 S2：到达时间比值

```
给定 seed，计算所有到达时间 T_w = -log(U_w) / P_w
T_{(1)} = min_w T_w （最小到达时间）
T_{(2)} = second_min_w T_w （次小到达时间）
R_t = T_{(1)} / T_{(2)}
```

- $R_t \in [0, 1]$，越小表示赢家赢得越轻松
- $H_0$ 下：$w_t$ 与 seed 无关，$T_{w_t}$ 不一定是最小的，$R_t$ 的分布由 $P$ 决定
- $H_1$ 下：$w_t$ 就是 argmin，$T_{w_t} = T_{(1)}$，$R_t$ 的分布会偏移

序列级统计量：$S_R = \sum_{t=1}^n (-\log R_t)$

### 统计量 S3：到达时间间距

```
G_t = T_{(2)} - T_{(1)}
```

- $H_0$ 下：间距分布由 $P$ 决定
- $H_1$ 下：赢家的到达时间被 seed 锁定，间距分布会改变

序列级统计量：$S_G = \sum_{t=1}^n G_t$

### 统计量 S4：联合统计量

```
S_joint = S_Z + lambda * S_R
```

其中 $\lambda$ 是可调权重。

---

## 5. 实验设置

### 5.1 分布配置

使用三种 entropy regime，词表大小 $|\mathcal{W}| = 10$：

**低熵（Low Entropy）：**
```python
P = [0.60, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03]
# Ent(P) ≈ 1.48
```

**中熵（Medium Entropy）：** 使用 He et al. (2026) 的设置
```python
P = [0.10, 0.13, 0.155, 0.115, 0.235, 0.065, 0.055, 0.05, 0.06, 0.035]
# Ent(P) ≈ 2.10
```

**高熵（High Entropy）：**
```python
P = Dirichlet(alpha=[10,10,...,10]).sample()  # 接近均匀
# Ent(P) ≈ 2.25
```

### 5.2 实验参数

| 参数 | 值 | 说明 |
|---|---|---|
| 词表大小 $\|\mathcal{W}\|$ | 10 | 与 He et al. 一致 |
| 序列长度 $n$ | 20, 50, 80, 100, 150, 200 | 覆盖短文本到中长文本 |
| 试验次数 $M$ | 2000 | 每个 $(H_0, H_1)$ 各 2000 条序列 |
| 目标 FPR | 1% | 与 He et al. 一致 |

---

## 6. 实验流程

### 实验 0：相关性诊断（第零步，最先做）

**目的：** 判断 $Z_t$ 和 $R_t$ 是否携带独立信息。如果高度相关，后续实验价值有限。

**步骤：**

```
对每种 entropy regime:
    对 trial = 1, ..., 10000:
        seed_t = deterministic_seed(trial)
        用 seed_t 生成 {U_w}，跑 Poisson race
        记录赢家 w*
        计算 Z_t = -log(1 - U_{w*}^{1/P_{w*}})
        计算 R_t = T_{(1)} / T_{(2)}
        计算 G_t = T_{(2)} - T_{(1)}
    
    计算 Pearson 相关系数:
        corr(Z, R), corr(Z, G), corr(R, G)
```

**判断标准：**

- $|\rho(Z, R)| < 0.3$：两者近似独立，联合使用预期有显著增益 → 继续后续实验
- $0.3 \le |\rho(Z, R)| < 0.7$：中等相关，联合使用仍有增益但不会很大
- $|\rho(Z, R)| > 0.7$：高度相关，需要寻找其他统计量（见第 9 节备选方案）

**输出：** 3×3 相关性矩阵表（三种 regime × 三对统计量）

---

### 实验 1：单 token 分布分离度

**目的：** 可视化每个统计量在 $H_0$ 和 $H_1$ 下的分布差异。

**$H_1$ 样本生成：**
```
seed_t = PRG(key, t)               # 确定性 seed
用 seed_t 生成 {U_w}
跑 Poisson race: w_t = argmin_w T_w  # token 由 seed 决定
计算 Z_t, R_t, G_t
```

**$H_0$ 样本生成：**
```
w_t ~ Multinomial(P)               # 从 P 真随机采样，不经过 Poisson race
seed_t = PRG(key, t)               # 用同一个 seed 生成 {U_w}
计算 T_w = -log(U_w) / P_w
Z_t = -log(1 - U_{w_t}^{1/P_{w_t}})
R_t = min_w(T_w) / second_min_w(T_w)   # 注意：这里 w_t 不一定是 argmin
G_t = second_min_w(T_w) - min_w(T_w)
```

**关键说明：** $H_0$ 的生成方式模拟了"检测者拿着正确的 seed，但文本其实不是水印生成的"。
此时 $w_t$ 是从 $P$ 独立采样的，和 seed 无关。

**分析：**

对每种 regime，生成 10000 个 $(H_0, H_1)$ 样本对，画出：
1. $Z_t$ 的 $H_0$ vs $H_1$ 直方图
2. $R_t$ 的 $H_0$ vs $H_1$ 直方图
3. $G_t$ 的 $H_0$ vs $H_1$ 直方图

**量化指标：** 对每个统计量计算 Cohen's d
```
d = (mean_H1 - mean_H0) / sqrt(0.5 * (var_H0 + var_H1))
```

**输出：** 每种 regime 下三个统计量的直方图 + Cohen's d 表

---

### 实验 2：序列级检测功效（TPR vs 序列长度）

**目的：** 验证联合统计量是否在序列级别提升检测功效。

**步骤：**

```
对每种 entropy regime:
    对每个序列长度 n in [20, 50, 80, 100, 150, 200]:
        
        # 生成 H0 序列
        对 trial = 1, ..., 2000:
            对 t = 1, ..., n:
                w_t ~ Multinomial(P)
                seed_t = PRG(key, trial, t)
                用 seed_t 生成 {U_w}，计算 T_w
                记录 Z_t, R_t, G_t
            计算序列得分:
                S_Z = sum(Z_t)
                S_R = sum(-log(R_t))
                S_G = sum(G_t)
                S_joint = S_Z + lambda * S_R
        
        # 生成 H1 序列
        对 trial = 1, ..., 2000:
            对 t = 1, ..., n:
                seed_t = PRG(key, trial, t)
                用 seed_t 生成 {U_w}，跑 Poisson race
                w_t = argmin_w T_w
                记录 Z_t, R_t, G_t
            计算序列得分（同上）
        
        # 计算 TPR @ FPR=1%
        对每个统计量:
            threshold = H0 得分的 99% 分位数
            TPR = H1 中超过 threshold 的比例
```

**输出：** 四条曲线（$S_Z$, $S_R$, $S_G$, $S_{\text{joint}}$）在每种 regime 下的 TPR vs $n$ 图

**核心判断：** $S_{\text{joint}}$ 的曲线是否严格在 $S_Z$ 之上

---

### 实验 3：联合统计量权重优化

**目的：** 找到 $\lambda^*$ 使得 $S_{\text{joint}} = S_Z + \lambda \cdot S_R$ 的检测功效最大。

**步骤：**

```
固定 n = 100，使用 medium entropy 分布

将 2000 条 H0 和 2000 条 H1 序列拆分:
    训练集: 各 1000 条
    测试集: 各 1000 条

对 lambda in [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
    在训练集上:
        threshold = H0_train 的 S_joint 的 99% 分位数
    在测试集上:
        TPR = H1_test 中 S_joint > threshold 的比例

选择使测试集 TPR 最大的 lambda*
```

**输出：** $\lambda$ vs TPR 的曲线 + 最优 $\lambda^*$

---

### 实验 4：ROC 曲线对比

**目的：** 在完整 FPR 范围内比较检测功效，而不仅在 FPR=1% 处。

**步骤：**

```
固定 n = 100（或选择 TPR 差异最大的序列长度）
使用实验 3 找到的 lambda*

生成 2000 条 H0 和 2000 条 H1 序列

对每个统计量 (S_Z, S_R, S_joint):
    对 FPR 阈值从 0 到 0.15（步长 0.001）:
        threshold = H0 的 (1-FPR) 分位数
        TPR = H1 中超过 threshold 的比例
    画 ROC 曲线
    计算 AUC（仅在 FPR ∈ [0, 0.10] 范围内）
```

**输出：**

- ROC 曲线图（放大到 FPR ∈ [0, 0.15] 区域）
- 部分 AUC 表

---

### 实验 5（扩展）：Entropy 依赖性分析

**目的：** 验证到达时间统计量在高熵分布下的相对优势是否更大。

**步骤：**

```
生成一系列分布 P_1, P_2, ..., P_K，entropy 从低到高
    方法: P_k = Dirichlet(alpha_k * ones(10))
    alpha_k 从 0.5 到 50，对应 entropy 从 ~1.0 到 ~2.30

对每个 P_k:
    固定 n = 100
    计算 S_Z 的 TPR@FPR=1%
    计算 S_R 的 TPR@FPR=1%
    计算 S_joint 的 TPR@FPR=1%

画: x 轴 = Ent(P), y 轴 = TPR，三条曲线
```

**预期结果：** 在高熵端，$S_R$ 和 $S_Z$ 的差距缩小甚至 $S_R$ 反超；$S_{\text{joint}}$ 的增益在高熵端更显著。

**输出：** TPR vs Entropy 图

---

## 7. $H_0$ 下统计量理论分布的推导提示

为了设计更 powerful 的检测，理解 $H_0$ 下各统计量的精确分布会有帮助。

### $Z_t$ 在 $H_0$ 下

$y_t = U_{w_t}^{1/P_{w_t}}$。当 $w_t$ 从 $P$ 独立采样、$U_{w_t}$ 独立 uniform 时，$y_t \sim \text{Uniform}(0,1)$。
因此 $Z_t = -\log(1-y_t) \sim \text{Exp}(1)$。这是已知结果。

### $R_t$ 在 $H_0$ 下（需要模拟）

$R_t = T_{(1)}/T_{(2)}$ 的分布取决于 $P$ 的结构，没有简洁的闭式。
但可以注意到：$T_{(1)} \sim \text{Exp}(1)$（因为 $\sum P_w = 1$），而 $T_{(2)} | T_{(1)}$ 的条件分布
可以通过 order statistics of exponentials 来推导。

对于均匀分布 $P_w = 1/V$，$T_w = V \cdot E_w$，此时 $R = T_{(1)}/T_{(2)}$ 的分布可以精确计算：
$R \sim \text{Beta}(1, V-1)$ 的某种变换。这是一个可以验证的 sanity check。

### $R_t$ 在 $H_1$ 下

$H_1$ 下 $w_t$ 就是 argmin，所以 $T_{w_t} = T_{(1)}$。此时 $R_t$ 就是 Poisson race 中第一名和第二名
到达时间的比值。这个量的分布和 $H_0$ 下的分布不同，因为 $H_0$ 下 $w_t$ 不一定是第一名。

---

## 8. 预期结果与判断标准

| 结果 | 含义 | 后续行动 |
|---|---|---|
| $S_{\text{joint}}$ TPR > $S_Z$ TPR 超过 5 个百分点 | 到达时间提供显著增量信息 | 有足够的 delta 支撑论文贡献 |
| $S_R$ 单独 TPR 接近 $S_Z$ | 到达时间本身就是强信号 | 可以主打"替代性统计量"的故事 |
| 高熵下增益 > 低熵下增益 | 到达时间在困难 regime 下更有价值 | 强化论文贡献，因为高熵是实际部署的瓶颈 |
| $\|\rho(Z, R)\| > 0.7$ | 两者高度相关 | 需要转向备选统计量（见第 9 节） |
| $S_{\text{joint}}$ TPR ≈ $S_Z$ TPR | 到达时间不提供增量 | 到达时间优势不在检测功效，转向其他维度（鲁棒性、自适应性） |

---

## 9. 备选统计量（如果 $R_t$ 与 $Z_t$ 高度相关）

如果实验 0 发现 $|\rho(Z, R)| > 0.7$，考虑以下替代方案：

### 9.1 条件排名统计量

```
Rank_t = w_t 对应的 T_{w_t} 在所有 {T_w} 中的排名
```

- $H_1$ 下：$\text{Rank}_t = 1$（始终第一名）
- $H_0$ 下：$\text{Rank}_t$ 的分布由 $P$ 和 $P_{w_t}$ 决定，不一定是第一名

这个统计量和 $Z_t$ 完全不同——$Z_t$ 看的是 $U_{w_t}$ 的绝对值，Rank 看的是 $T_{w_t}$ 在竞赛中的相对位置。

### 9.2 次优 token 统计量

```
D_t = 观测到的 w_t 是否等于 Poisson race 的赢家
```

这是一个 binary 统计量：$H_1$ 下 $D_t = 1$ 必然成立，$H_0$ 下 $D_t = 1$ 的概率等于 $P_{w_t}$（因为
随机采到的 token 恰好也是 Poisson race 赢家的概率不高）。

序列级：$S_D = \sum_t D_t$。$H_0$ 下 $\mathbb{E}[S_D] = \sum_t P_{w_t}$（较小），$H_1$ 下 $S_D = n$（确定）。

### 9.3 加权到达时间统计量

```
W_t = log(P_{w_t}) * R_t
```

用 $\log P_{w_t}$ 加权，让低概率 token 的到达时间信号被放大——因为低概率 token 赢得 Poisson race
本身就更"可疑"。

---

## 10. 代码实现提示

### 关键函数签名

```python
def poisson_race(P: np.ndarray, seed: int) -> Tuple[int, np.ndarray, float, float]:
    """
    输入: 分布 P, 伪随机 seed
    输出: (winner_index, all_arrival_times, T1, T2)
    """
    rng = np.random.RandomState(seed)
    U = rng.uniform(0, 1, size=len(P))
    E = -np.log(U)
    T = E / P
    sorted_idx = np.argsort(T)
    return sorted_idx[0], T, T[sorted_idx[0]], T[sorted_idx[1]]


def compute_statistics(w_t: int, P: np.ndarray, U: np.ndarray, T: np.ndarray,
                       T1: float, T2: float) -> dict:
    """
    输入: 观测 token w_t, 分布 P, race 的 U 值和到达时间
    输出: dict of {Z_t, R_t, G_t, Rank_t, D_t}
    """
    y_t = U[w_t] ** (1.0 / P[w_t])
    Z_t = -np.log(1.0 - y_t + 1e-15)
    R_t = T1 / T2
    G_t = T2 - T1
    Rank_t = np.sum(T <= T[w_t])          # w_t 的排名
    D_t = int(w_t == np.argmin(T))        # 是否是赢家
    return {'Z': Z_t, 'R': R_t, 'G': G_t, 'Rank': Rank_t, 'D': D_t}


def generate_H0_sequence(P, n, key):
    """H0: token 从 P 采样，与 seed 无关"""
    stats = []
    for t in range(n):
        w_t = np.random.choice(len(P), p=P)           # 真随机采样
        seed_t = deterministic_seed(key, t)
        winner, T, T1, T2 = poisson_race(P, seed_t)   # 跑 race（但 w_t 不受 race 控制）
        U = np.exp(-T * P)                             # 从 T 反算 U
        s = compute_statistics(w_t, P, U, T, T1, T2)
        stats.append(s)
    return stats


def generate_H1_sequence(P, n, key):
    """H1: token 就是 race 的赢家"""
    stats = []
    for t in range(n):
        seed_t = deterministic_seed(key, t)
        w_t, T, T1, T2 = poisson_race(P, seed_t)      # w_t = race 赢家
        U = np.exp(-T * P)
        s = compute_statistics(w_t, P, U, T, T1, T2)
        stats.append(s)
    return stats
```

### 注意事项

1. **数值稳定性：** $-\log(1 - y_t)$ 在 $y_t \to 1$ 时会溢出，加 `1e-15` 的 clamp
2. **seed 管理：** $H_0$ 中 token 的采样和 seed 的生成必须使用不同的随机数生成器，否则会引入虚假相关
3. **可复现性：** 所有实验固定全局随机种子，结果应当完全可复现

---

## 11. 输出清单

完成实验后，应当产出以下结果：

| 编号 | 内容 | 格式 |
|---|---|---|
| T0 | 相关性矩阵（3 regime × 3 对统计量） | 表格 |
| T1 | 单 token 分布直方图 + Cohen's d | 3×3 图 + 表格 |
| T2 | TPR vs 序列长度曲线（3 regime × 4 统计量） | 3 张图 |
| T3 | $\lambda$ vs TPR 曲线 + 最优 $\lambda^*$ | 1 张图 + 数值 |
| T4 | ROC 曲线（放大到 FPR ∈ [0, 0.15]）+ 部分 AUC | 3 张图 + 表格 |
| T5 | TPR vs Entropy 曲线 | 1 张图 |

---

## 12. 实验优先级

如果时间有限，按以下顺序执行：

1. **实验 0**（相关性诊断）— 30 分钟可出结果，决定后续实验是否值得做
2. **实验 1**（单 token 直方图）— 直觉验证，快速判断信号强度
3. **实验 2**（TPR vs 序列长度）— 核心结果，直接回答"有没有增益"
4. **实验 3**（权重优化）— 如果实验 2 显示增益，优化联合统计量
5. **实验 4**（ROC 曲线）— 更完整的评估
6. **实验 5**（Entropy 依赖性）— 锦上添花，支撑论文故事
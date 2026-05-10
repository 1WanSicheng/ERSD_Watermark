# Figure 2 修改说明：Detection and Drafter Substitution

## 0. 当前 Figure 2 的核心问题

当前 Figure 2 的结果本身是有价值的，但图面呈现略显“别扭”，主要原因是它同时承载了两个不同层次的信息：

1. **左图**：比较不同方法在默认 drafter 下的 fixed-FPR detection power；
2. **右图**：比较不同方法在 drafter substitution 下的 detection stability。

这两个问题都应该保留，但需要重新组织视觉层级，让 reviewer 一眼看懂：

- **PFR/MPFR 的 detection 接近 strong watermark reference；**
- **PFR 的 detection 对 drafter substitution 更稳定；**
- **MSE 和 MSE-Pseudo 在 fixed-FPR detection 和 drafter robustness 上弱于 PFR/MPFR。**

当前版本更像 appendix dashboard，主文图需要更简洁、更结论导向。

---

## 1. 推荐的最终 Figure 2 结构

建议保留当前的“左大图 + 右 2×2 小图”结构，但要明确拆成两个 panel：

### (a) Detection on the default drafter

左侧大图保留为 TPR curve：

- x 轴：`T_eval (# generated tokens)`
- y 轴：`TPR @ 1% FPR`
- 方法：
  - `PFR (ours)`
  - `MPFR (ours)`
  - `Basic-UWM`
  - `MWS`
  - `MSE`
  - `MSE-Pseudo`
  - `No watermark (H0)`

这个 panel 回答：

> 在默认 drafter 下，PFR/MPFR 的 detection power 是否接近 strong watermark baseline？

### (b) Robustness to drafter substitution

右侧保留 2×2 小图，每个小图一个方法：

- `PFR`
- `MWS`
- `MSE`
- `MSE-Pseudo`

每个小图中画四条线：

- `D0`: default drafter
- `D1`: model swap
- `D2`: sharper drafter, `T_drafter = 0.5`
- `D3`: diffuse drafter, `T_drafter = 1.5`

这个 panel 回答：

> 当 drafter 改变时，PFR 的 detection curve 是否保持稳定？

---

## 2. 必须修改的地方

### 2.1 给两个 panel 加明确标题

当前图的结构需要从视觉上明确区分两个问题。

建议在图中加：

```text
(a) Detection on the default drafter
(b) Robustness to drafter substitution
```

这两个标题应放在各自 panel 上方，而不是只写在 caption 中。

---

### 2.2 删除右侧小图里的 annotation box

当前右侧每个小图里有类似：

```text
Δ_64 = ...
Δ_128 = ...
```

这种小白框不建议放在主文图里。原因是：

- 字太小；
- 信息密度高但收益不大；
- 让图看起来像工程调试图；
- reviewer 大概率不会仔细读。

建议删除这些小框。

如果一定要保留 drift 信息，可以在 caption 或正文中写：

```latex
\textsc{PFR} has at most one percentage point of TPR drift at
$T_{\mathrm{eval}}\in\{64,128\}$, while \textsc{MSE} drifts by
$12$--$22$ percentage points and \textsc{MSE-Pseudo} by $6$--$11$ percentage points.
```

这样比在图中塞小框更清楚。

---

### 2.3 统一方法命名

当前图里有一些方法名偏代码风格，例如：

```text
mse_pseudo
no watermark H0
```

建议统一成论文风格：

| 当前写法 | 建议写法 |
|---|---|
| `mse_pseudo` | `MSE-Pseudo` |
| `Basic UWM` | `Basic-UWM` |
| `no watermark H_0` | `No watermark ($H_0$)` |
| `PFR (ours)` | `PFR (ours)` |
| `MPFR (ours)` | `MPFR (ours)` |
| `MSE` | `MSE` |
| `MWS` | `MWS` |

如果 LaTeX 中要统一命令，建议定义：

```latex
\newcommand{\pfr}{\textsc{PFR}}
\newcommand{\mpfr}{\textsc{MPFR}}
\newcommand{\mse}{\textsc{MSE}}
\newcommand{\mws}{\textsc{MWS}}
\newcommand{\msepseudo}{\textsc{MSE-Pseudo}}
\newcommand{\basicuwm}{\textsc{Basic-UWM}}
```

---

### 2.4 统一 y 轴范围

左右两部分都在画 `TPR @ 1% FPR`，建议统一 y 轴范围：

```text
0.0 to 1.0
```

这有两个好处：

1. 左右图可以直观比较 detection strength；
2. 不会因为局部放大造成视觉误导。

右侧 2×2 小图也应共享 y 轴范围。

---

### 2.5 减轻 legend 负担

当前图中有两个 legend 系统：

1. 左图的方法 legend；
2. 右图的 drafter condition legend。

建议：

- 左图 legend 放在左图内部左上角或图外上方；
- 右图只保留一个统一 legend，放在右侧 2×2 小图下方或上方；
- 不要在每个小图重复 legend；
- 方法数量不要再增加。

右图 legend 建议写成：

```text
D0 default
D1 model swap
D2 T=0.5
D3 T=1.5
```

---

## 3. 颜色与线型建议

### 3.1 方法颜色

建议全文固定方法颜色，避免 Figure 1 和 Figure 2 中颜色含义不同。

一个可行方案：

| 方法 | 颜色建议 | 线型 |
|---|---|---|
| `PFR (ours)` | orange / red-orange | solid |
| `MPFR (ours)` | green | solid |
| `Basic-UWM` | blue | solid |
| `MWS` | gold | solid |
| `MSE` | pink / magenta | dashed |
| `MSE-Pseudo` | light blue | dotted / dash-dot |
| `No watermark (H0)` | gray / black | dotted |

原则：

- ours 的方法要突出；
- no-watermark baseline 要弱化；
- MSE / MSE-Pseudo 可以用虚线，表示 prior efficiency-oriented baselines；
- Basic-UWM / MWS 作为 strong watermark references，可以用实线但不要比 PFR/MPFR 更抢眼。

### 3.2 右图 drafter condition 颜色

右图中颜色表示 drafter condition，而不是 method。建议：

| Drafter condition | 颜色/线型 |
|---|---|
| `D0 default` | blue solid |
| `D1 model swap` | orange solid |
| `D2 T=0.5` | green dashed |
| `D3 T=1.5` | yellow / brown dotted |

注意：右图每个小图的 method 已由 subplot title 表示，所以线条颜色只需要区分 drafter condition。

---

## 4. 布局建议

### 4.1 推荐比例

当前左图大、右图小，比例可以保留，但右侧小图需要放大一点。

推荐布局：

```text
width_ratios = [1.25, 1.0]
right panel = 2 x 2 subplots
```

也就是说：

- 左图占 55%左右宽度；
- 右图占 45%左右宽度；
- 右侧四个小图不要太挤。

### 4.2 图整体尺寸

如果是 NeurIPS 双栏主文，建议使用：

```python
figsize = (7.0, 2.8)
```

或者：

```python
figsize = (7.2, 3.0)
```

不要太高，否则 caption 会挤；不要太宽，否则缩进后字体太小。

### 4.3 字号建议

```python
axis_label_fontsize = 8
tick_fontsize = 7
legend_fontsize = 6.5
subplot_title_fontsize = 8
panel_title_fontsize = 9
```

---

## 5. Caption 建议

当前 caption 太长。建议压缩为“总述 + panel (a) + panel (b)”三句话。

推荐 caption：

```latex
\caption{
\textbf{Detection and robustness to drafter substitution.}
\textbf{(a)} Detection at a fixed false-positive rate on the default drafter.
\textsc{PFR} and \textsc{MPFR} achieve TPR@1\%FPR close to the strong-watermark
references \textsc{Basic-UWM} and \textsc{MWS}, and outperform the
efficiency-oriented baselines \textsc{MSE} and \textsc{MSE-Pseudo}.
\textbf{(b)} Robustness under drafter substitution. We vary the drafter across
four conditions while keeping the target model, watermark key, and prompts fixed.
\textsc{PFR} remains nearly unchanged, whereas \textsc{MSE} and
\textsc{MSE-Pseudo} exhibit substantially larger drift.
}
```

不要在 caption 里写完整的 D0/D1/D2/D3 定义。可以在正文中写一句，或者放 appendix。

正文中可以写：

```latex
The four drafter conditions are the default drafter, a model-scale swap, a sharper
drafter temperature, and a more diffuse drafter temperature; details are given in
Appendix~\ref{app:ablation}.
```

---

## 6. 正文引用 Figure 2 的推荐写法

正文可以这样写：

```latex
\paragraph{Detection at fixed false-positive rate.}
ANLPPT-U measures average evidence per token, while TPR@1\%FPR measures operational
detectability under a calibrated false-positive budget. Figure~\ref{fig:drafter_invariance}(a)
shows that \textsc{PFR} and \textsc{MPFR} closely track the strong-watermark
references \textsc{Basic-UWM} and \textsc{MWS}. The pseudo-randomized
\textsc{MSE-Pseudo} baseline improves over \textsc{MSE}, but remains below the
Poisson-race methods in the evaluated setting.

\paragraph{Drafter-substitution ablation.}
Figure~\ref{fig:drafter_invariance}(b) evaluates whether the watermark signal remains
stable when the proposal mechanism changes. We keep the target model, watermark key,
and prompts fixed, and vary only the drafter. \textsc{PFR} has at most one percentage
point of TPR drift at $T_{\mathrm{eval}}\in\{64,128\}$, while \textsc{MSE} and
\textsc{MSE-Pseudo} exhibit substantially larger drift. This supports the intended
role of target-side keyed Poisson races: the drafter can affect how many tokens are
verified, but the watermark statistic of the emitted tokens remains tied to the
realized prefix and the target-side keyed randomness.
```

---

## 7. 如果只做最小修改

如果时间不够，至少做以下 5 件事：

1. 删除右侧四个小图里的 `Δ` annotation box；
2. 给图加 `(a)` 和 `(b)` 两个 panel title；
3. 把 `mse_pseudo` 改成 `MSE-Pseudo`；
4. 把 caption 缩短成上面的版本；
5. 右侧 2×2 小图共享 y 轴范围，并把 legend 合并成一个。

这 5 个改完，Figure 2 会明显更像主文图。

---

## 8. 如果可以稍微重画

更理想的 Figure 2 版本是：

```text
Figure 2:
  Left: one large TPR curve plot for default drafter.
  Right: 2x2 robustness plots, each method one subplot.
  Remove all annotation boxes.
  Use unified panel labels.
  Use a single global legend for methods on the left and a single global legend for drafter conditions on the right.
```

完整数据仍然可以放 appendix，例如：

```text
Appendix Figure:
  Full detection curves across all model-dataset cells.
  Full drafter-substitution curves for all methods.
```

主文 Figure 2 只展示 anchor setting，不展示所有 cells。

---

## 9. 最终目标

修改后的 Figure 2 应该让 reviewer 在 5 秒内读出：

1. PFR/MPFR 的 TPR@1%FPR 接近 Basic-UWM 和 MWS；
2. MSE-Pseudo 比 MSE 好，但仍低于 PFR/MPFR；
3. PFR 在 drafter substitution 下几乎不漂移；
4. MSE/MSE-Pseudo 明显更受 drafter 变化影响。

如果图能做到这四点，就足够支撑正文 claim。

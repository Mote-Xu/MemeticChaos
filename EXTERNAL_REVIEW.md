# MemeticChaos 外部审查请求

> 完整代码：https://github.com/Mote-Xu/MemeticChaos
> 项目蓝图：PROJECT_BLUEPRINT.md | 上次自审：CODE_REVIEW_FINDINGS.md

---

## 现状

21 项功能，5945 行 Python，10 模块，24 测试全部通过，14 次 commit。

GPT 上次评价：「从研究原型升级为正在形成独立理论框架的研究计划」。核心产出是模因相图（29 个热梗在 R₀ × Chaos Axis 空间的相态分布，5 相区 + 2 吸引子盆地 + 情绪状态机）。

上次自审发现 1 个真实 bug（`_compute_loss` 中 `params` 未定义，已修复）+ 1 次代码简化（R₀ 解析解替代二分搜索）+ 1 次去重。

**我们希望外部 AI 重点审查：已实现功能的算法正确性和功能完备性。** 以下是每个模块的核心实现细节和具体问题。

---

## 模块 1：SIR 模因传播模型

`src/models/sir_meme.py`，731 行，24 测试覆盖。

**ODE 系统**：标准 SIR + SIRS（变异复燃）+ 双群体（核心圈→大众扩散），`solve_ivp` RK45 求解。

**参数估算**（第 394-448 行）：
```python
def estimate_params_from_lifecycle(peak_day, total_infected, duration_days):
    gamma_est = 2.0 / max(duration_days, 1.0)
    # SIR最终规模: R∞ = 1 - exp(-R₀ × R∞)
    # 解析解: R₀ = -ln(1 - R∞) / R∞
    R0_est = -np.log(1.0 - target) / target
    beta_est = R0_est * gamma_est
```

这个解析解对 S₀≈1 假设下的 SIR 模型是正确的。但实际中热梗的传播起点 S₀ 往往远小于 1（很多人一开始就不关注）。**问题：这个近似在 S₀ 偏离 1 时会引入多大误差？**

**参数拟合**（第 363-401 行）：`fit_sir_to_curve()` 用 `curve_fit` 从 I(t) 反推 β,γ。内部反复调用 `solve_sir` + `np.interp` 插值匹配观测点。**问题：这个拟合流程的数值稳定性如何？curve_fit 对初始猜测敏感吗？**

**熵轨迹**（第 498-517 行）：
```python
def compute_entropy_curve(result):
    total = S + I + R
    p_S, p_I, p_R = S/total, I/total, R/total
    entropy = -(p_S*log(p_S+ε) + p_I*log(p_I+ε) + p_R*log(p_R+ε))
```
三个概率分量的香农熵。峰值出现在 I 接近 max 时（系统混沌度最大）。**问题：ε=1e-12 的保护是否足够应对 p=0 的边界？**

---

## 模块 2：ABM 多智能体仿真

`src/models/abm_simulation.py`，667 行，无 Mesa 依赖独立实现。

**Agent 步进逻辑**（第 90-130 行）：
```python
def step(self):
    self._entropy_drift()       # 1. 不加维护的秩序自发衰退
    self._emotional_contagion() # 2. 邻居情感传染（回音壁效应）
    self._meme_step()           # 3. SIR 状态转换
    self._update_vitality()     # 4. 生命力更新
    if role == CHAOS_INJECTOR:
        self._chaos_injection() # 5. 主动向邻居注入混沌
```

回音壁效应（第 133-153 行）：邻居影响力按 `similarity = 1 - |chaos_i - chaos_j|/2` 加权，越相似影响越大。

稳态锚点（第 80 行）：`chaos += 0.02 * resilience * (intrinsic_chaos - chaos)`，防止 chaos_position 被 contagion 冲散。

**问题**：
- 这 5 步的顺序会影响结果吗？先 entroy_drift 再 contagion vs 反过来？
- 稳态锚点的 0.02 系数是调参得来的——有更严格的推导吗？
- chaos_injector 的 `push *= 1.3`（对已是负 chaos 的邻居）——这个非对称性合理吗？

---

## 模块 3：吸引子检测

`src/models/attractor.py`，586 行。

**Takens 嵌入**（第 61-83 行）：标准延迟坐标，`τ` 用自相关 1/e 穿越估计（第 88-117 行），`d` 用简化 FNN 估计（第 120-174 行）。

FNN 实现（自审已标注"非标准"）：
```python
# 在m维中找最近邻，然后检查第m个坐标上的差异
dist_m = norm(embedded[i] - embedded[nn])
d_extra = abs(ts[i + (m-1)*delay] - ts[nn + (m-1)*delay])
if dist_m > 1e-10 and d_extra / dist_m > 10.0:
    fnn_count += 1
```
标准 FNN 应该在 m-1 维中找最近邻，再到 m 维中检查。当前实现在 m 维中同时做搜索和比较。**这个简化在什么情况下会给出错误结果？**

**Lyapunov 指数**（第 304-376 行），Rosenstein 算法：
```python
divergences = zeros(n_steps)
for i in sample_idx:
    nn_idx = argmin(distances to other points)
    d0 = norm(embedded[i] - embedded[nn_idx])
    for k in range(1, n_steps):
        dk = norm(embedded[i+k] - embedded[nn_idx+k])
        divergences[k] += log(dk / d0)
# slope of <log divergence> vs time → λ_max
```
**问题**：
- Theiler window 设为 `2*delay`——这是否足够排除时间相关的伪近邻？
- 当 `n - max(i, nn_idx) < n_steps` 时循环提前 break——被截断的轨迹对斜率拟合有什么影响？
- `np.polyfit` 做线性拟合是否比 Theil-Sen 或 RANSAC 更不稳定？

---

## 模块 4：个体混沌校准器

`src/models/individual_calibrator.py`，957 行。

**直接启发式**（第 444-578 行）：从 5 类观测信号（自我报告、有序表达率、矛盾频率、trolling 频率、模因参与模式）加权投票 → chaos_position 估计。

**贝叶斯后验构建**（第 605-630 行）：
```python
for vote, weight in zip(chaos_votes, chaos_weights):
    for i, (lo, hi) in enumerate(bins):
        center = (lo + hi) / 2
        posterior_raw[i] += weight * exp(-((vote - center)**2) / (2 * 0.15**2))
```
5 个 bin：极端混沌(-1~-0.6)、偏混沌(-0.6~-0.2)、中性(-0.2~+0.2)、偏秩序(+0.2~+0.6)、极端秩序(+0.6~+1.0)。

**问题**：
- 高斯核的 σ=0.15 是怎么确定的？不同 σ 对后验形状有多敏感？
- 加权投票的权重（self_report=3.0, orderly=1.5, trolling=1.5, narrative=1.0）有校准依据吗？
- 4 种场景验证都正确分类——但如果输入矛盾信号（如 trolling 高但自评 chaos 为正），后验分布是否合理？

---

## 模块 5：模因相图（核心产出）

`src/analysis/phase_diagram.py`，452 行。

29 个热梗 → `estimate_params_from_lifecycle()` 估算 SIR 参数 → 求解 → 在 R₀ × Chaos Axis 空间聚类。

**核心发现**：
- 2 个吸引子盆地（回测验证 100% 鲁棒）
- 2021 年混沌轴 -0.88 漂移 + 预测力 0.722→0.389
- 5 状态情绪状态机 + 4 条历史转移路径

**问题**：
- R₀ 估算依赖 `total_infected`（从 `circle_count` 映射为 0.25/0.50/0.75 三档）。这个映射本身就是最粗糙的一步。在真实数据缺位的情况下，相图的结构多大程度上是这个映射的 artifact？
- 5 相区的聚类用的是类别标签（category）作为先验——如果不用类别标签，纯从 R₀ × Chaos 空间做无监督聚类，还能不能得到同样的 5 个区？

---

## 模块 6：验证层

`src/analysis/backtest.py`，487 行。

**留一验证**：chaos MAE = 0.186，类别准确率 = 58.6%（随机基线 20%）。

**鲁棒性**：删 30% 热梗后盆地 93% 存活，加 20% 噪声后 100% 存活。

**问题**：
- 留一验证的"预测"用的是同一类别内其他热梗的 chaos 均值——这个 baseline 合理吗？有没有更强的 baseline 应该对比（如全局均值、年份均值、SIR 参数预测值）？
- 58.6% 的类别准确率——如果打乱类别标签重跑，准确率会降到多少？

---

## 模块 7：B站字幕管道

`src/data/bilibili_pipeline.py`，Gemini 贡献，261 行。

字幕 JSON → regex 关键词扫描 → rolling window 平滑 → `minimize(loss, L-BFGS-B)` 拟合 SIRS-M 参数。

**问题**：
- `loss_function` 内部每次调用都重新 `solve_ivp`——对 L-BFGS-B 的收敛速度影响多大？
- 参数边界 `[(0.01, 2.0), (0.01, 1.0), (0.0, 0.5), (0.0, 0.5)]` 是否合理？真实热梗的 β 会不会超过 2.0？

---

## 优先审查顺序

如果你时间有限，请重点审查 **模块 1（SIR ODE + 参数估算）、模块 3（Lyapunov + FNN）、模块 5（相图聚类）**。这三个的正确性直接影响所有下游结论。

# MemeticChaos 代码审查报告

> 自我审查 + 供外部 AI（GPT/Gemini）独立验证。
> 审查日期：2026-06-26
> 代码量：5945 行 Python，10 模块，24 测试

---

## 🤖 发送给外部 AI 的提示词

> 复制下面内容发给 GPT 或 Gemini：

```
你正在审查一个计算模因学 × 混沌动力学项目（5945行Python，10模块，24测试）。
项目用2020-2025年中国互联网热梗数据为人类集体情感混沌属性建模。

附件 CODE_REVIEW_FINDINGS.md 是我们自己的审查报告，发现了5个问题（2个bug + 3个issue）。

请：
1. 独立验证这5个发现的正确性
2. 指出我们漏掉的任何问题（算法逻辑错误、数值稳定性、架构反模式、测试缺口）
3. 对Bug 2（R₀公式）给出数学上严格正确的修复方案
4. 诚实评估：这个代码库如果要用于学术研究，最大的可信度缺口在哪

完整代码在：https://github.com/Mote-Xu/MemeticChaos
项目蓝图见仓库内 PROJECT_BLUEPRINT.md
```

---

## 外部 AI 审查反馈（待填写）

### GPT 的审查意见

> （待填入）

### Gemini 的审查意见

> （待填入）

---

---

## 一、总体评估

| 维度 | 评级 | 说明 |
|------|:--:|------|
| 运行状态 | 🟢 | 9/10 模块可独立运行（curator 是类库，正常） |
| 测试覆盖 | 🟡 | SIR 层 24/24，其余模块仅有 demo 脚本 |
| 算法正确性 | 🟡 | 核心算法正确，发现 2 个数值/逻辑漏洞 |
| 架构设计 | 🟢 | 模块划分清晰，低耦合，无循环依赖 |
| 代码质量 | 🟢 | 类型标注完整，docstring 规范，命名一致 |
| 哲学一致性 | 🟢 | 混沌≠随机、后验非点估计、黑箱约束贯穿始终 |

---

## 二、发现的实现问题

### 🔴 Bug 1：`_compute_loss` 中的未定义变量

**文件**：`src/models/individual_calibrator.py`，第 393 行附近

**问题**：
```python
def _compute_loss(sim_result: dict, observation: BehavioralObservation) -> float:
    ...
    if observation.self_reported_resilience is not None:
        loss += (observation.self_reported_resilience - params.get("resilience", 0.5)) ** 2
```

`params` 变量在 `_compute_loss` 的作用域中未定义。它只存在于调用方 `_evaluate_individual` 中。当前未触发是因为所有场景模板都没有设置 `self_reported_resilience`，但如果有人传入带此字段的观测，会抛 `NameError`。

**修复**：将 `params` 作为第三个参数传入 `_compute_loss`，或在 `_evaluate_individual` 中完成 resilience 的 loss 计算。

---

### 🟢 ~~Bug 2~~ → 代码简化：`estimate_params_from_lifecycle` 二分搜索 → 解析解

**文件**：`src/models/sir_meme.py`

**原判断**：二分搜索公式 `1 - exp(-mid * target)` 疑似将 target 误代入 R∞ 的位置。

**更正**：经代数验证，二分搜索收敛到 `R₀ = -ln(1-target)/target`，这与 SIR 最终规模方程 `R∞ = 1 - exp(-R₀ × R∞)`（令 R∞ = target）**代数等价**。原公式在数学上正确——二分搜索自动求解了同一个隐式方程。

**实际修复**：用解析解 `R₀ = -ln(1-target)/target` 替代 O(log n) 二分搜索 → O(1)。这是代码简化，不是 bug 修复。

**根本问题**：R₀ 区间窄（1.39-1.85）的真正原因不是公式错误，而是输入 `total_infected` 只有 3 个离散值（0.25/0.50/0.75），来自 qualitative 的 `circle_count` 到 quantitative 的粗糙映射。要解决 R₀ 分辨率问题，需要**真实时间序列数据**——这正是 B站字幕接入的意义。

---

### 🟡 Issue 3：FNN 假近邻法实现不符合标准算法

**文件**：`src/models/attractor.py`，`estimate_embedding_dim()` 第 141-174 行

**问题**：标准的假近邻法流程是：
1. 在 m-1 维嵌入中找到最近邻
2. 在 m 维嵌入中检查该近邻是否变成"假近邻"（距离比值超过阈值）

但当前实现在 m 维中同时做搜索和比较——没有真正比较 m-1 维和 m 维之间的近邻关系变化。实际上它检查的是"m 维空间中，最近邻对在第 m 个坐标上的差异是否过大"——这更接近一种粗糙的替代判据。

**影响**：`estimate_embedding_dim` 倾向于返回偏低的维数（通常就是 2），可能不足以充分展开相空间轨迹。但考虑到输入序列较短（100-200 点），低嵌入维数实际上是一种实用主义的保护措施。

**建议**：文档中明确标注这是"简化版 FNN"，不声称符合标准实现。

---

### 🟡 Issue 4：Lyapunov 估算对短序列敏感

**文件**：`src/models/attractor.py`，`estimate_lyapunov()` 第 304-376 行

**问题**：
1. `sample_idx = np.linspace(0, n - n_steps - 1, sample_size, dtype=int)` ——当 `n - n_steps - 1 < 0` 时（序列极短），`linspace` 产生负数索引
2. 邻居排除窗口 `2*delay` 可能不足以排除时间上相邻的点（Takens 嵌入中时间相关的点是伪近邻的主要来源）
3. Theiler window 应该是固定的时间窗口而非 `2*delay`

**影响**：对 SIR 曲线（通常 1000 点），这些不是问题。但对于 ABM 中采样的短混沌轨迹（150 步），Lyapunov 估算可能不稳定。

---

### 🟡 Issue 5：代码重复——`build_state_points` 两份实现

**文件**：`src/analysis/phase_diagram.py` 和 `src/analysis/backtest.py`

两者各自实现了从热梗数据到 MemeStatePoint 的转换逻辑，代码几乎相同（约 40 行重复）。`backtest.py` 的版本叫 `build_state_points_from_memes`。

**建议**：统一到 `phase_diagram.py` 的 `build_state_points`，`backtest.py` 直接 import。

---

## 三、架构评估

### 优点
- **模块边界清晰**：models / analysis / data / viz 四层分离，职责明确
- **无循环依赖**：依赖方向是单向的（analysis → models → data）
- **参数与结果分离**：dataclass 用于数据传递，函数保持纯计算
- **哲学约束编码**：个体校准器的后验输出、黑箱警示都被写入代码而非注释

### 可改进
- `abm_simulation.py` 667 行单文件偏大。Agent 类、网络构建、仿真循环可拆分为独立文件
- 缺少统一的配置管理（各模块 config 各自 dataclass，没有全局 settings）
- 缺少日志系统——所有输出都靠 `print()`

---

## 四、测试覆盖审计

| 模块 | 测试数 | 覆盖类型 | 缺口 |
|------|:--:|------|------|
| sir_meme | 24 | 守恒律/R₀阈值/收敛性/拟合精度/熵 | ✅ 充分 |
| abm_simulation | 0 | — | 网络构建/参数扫描/涌现行为 |
| attractor | 0 | — | 嵌入维数/Lyapunov符号/RQA统计 |
| individual_calibrator | 0 | — | 4场景回归/后验归一化/边界输入 |
| lifecycle | 0 | — | 极端I值/单峰/平线 |
| sentiment | 0 | — | 空phases/单phase |
| phase_detect | 0 | — | 无相变/多重相变 |
| phase_diagram | 0 | — | 单类别/空数据集 |
| backtest | 0 | — | 单样本测试集/年边界 |

**优先级**：attractor > individual_calibrator > abm_simulation

---

## 五、数值稳定性审计

| 检查项 | 状态 | 说明 |
|--------|:--:|------|
| log(0) 保护 | ✅ | 熵计算有 epsilon 保护 |
| 除零保护 | ✅ | gamma/max(1.0)/1e-10 保护 |
| 溢出保护 | 🟡 | Lyapunov 发散未 clamp |
| 收敛判据 | 🟡 | GA 无早停机制，跑满迭代 |
| 浮点精度 | ✅ | pytest.approx 用于比较 |
| 随机种子 | 🟡 | 部分模块 seed=42，部分未设 |

---

## 六、完备性检查

### 文档承诺 vs 实际实现

| CLAUDE.md 声明 | 实际 | 状态 |
|----------------|------|:--:|
| SIR/SIRS/双群体 | 全部实现 | ✅ |
| 参数拟合 (curve_fit) | 实现但未用真实数据验证 | ⚠️ |
| 300 Agent ABM | 实现，参数过度趋同 | ⚠️ |
| Takens/RQA/Lyapunov | 全部实现 | ✅ |
| 个体校准器 贝叶斯后验 | ✅ | ✅ |
| 模因相图 5相区 | ✅ | ✅ |
| 历史回测 | ✅ | ✅ |
| 鲁棒性验证 | ✅ | ✅ |
| B站字幕接入 | 视频已下载，字幕待接入 | ❌ |
| 百度指数爬虫 | 未开始 | ❌ |
| 跨平台验证 | 未开始 | ❌ |

---

## 七、优先修复建议

1. **立即修**：Bug 2（R₀ 公式）→ 影响所有模块的定量结果
2. **立即修**：Bug 1（params 作用域）→ 静默 bug，触发即 crash
3. **尽快修**：Issue 5（代码重复）→ 防止未来分叉
4. **计划修**：Issue 3（FNN 标注）→ 加文档说明
5. **可延后**：Issue 4（Lyapunov 短序列）→ 当前场景不触发

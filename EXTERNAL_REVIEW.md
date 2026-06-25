# MemeticChaos 外部 AI 审查请求（v2）

> 请以代码审查者视角，重点评估**已实现功能的正确性和完备性**。
> 完整代码：https://github.com/Mote-Xu/MemeticChaos
> 项目蓝图：PROJECT_BLUEPRINT.md | 上次审查：CODE_REVIEW_FINDINGS.md

---

## 项目现状

21项功能，11/14 FR完成。5945行Python，10模块，24测试全部通过。

上次自审发现1个真实bug（已修复）+ 1次代码简化 + 1次去重。
GPT评价：从"研究原型"升级为"正在形成独立理论框架的研究计划"。
核心产出：模因相图（Meme Phase Diagram）。

## 审查重点

### 1. 功能正确性：每个模块的核心算法对吗？

**SIR模型** (`src/models/sir_meme.py`，731行)
- 标准SIR/SIRS/双群体三种ODE，`scipy.integrate.solve_ivp` RK45求解
- `fit_sir_to_curve()` 用 `curve_fit` 从I(t)反推β,γ
- `estimate_params_from_lifecycle()` 从定性数据(peak_day, total_infected, duration)估算参数，R₀用解析解 `-ln(1-R∞)/R∞`
- `compute_entropy_curve()` 跟踪SIR轨迹的香农熵
- 审查问题：ODE参数化是否正确？curve_fit的bounds设置是否合理？熵归一化用log(n_bins)对吗？

**ABM仿真** (`src/models/abm_simulation.py`，667行)
- 无Mesa依赖独立实现，Barabási-Albert无标度网络
- 5种Agent角色(含chaos_injector)，情感传染+回音壁+熵增漂移
- 稳态锚点：`chaos += homeostatic_strength * (intrinsic_chaos - chaos)` 防止漂移
- 审查问题：Agent step()的更新顺序是否合理？回音壁实现是否正确？vitality会不会在边界条件下发散？

**吸引子检测** (`src/models/attractor.py`，586行)
- `estimate_embedding_delay()` 用自相关1/e首次穿越
- `estimate_embedding_dim()` 用简化版FNN假近邻法（自审标注：非标准实现）
- `recurrence_matrix()` + `recurrence_quantification()` 完整RQA
- `estimate_lyapunov()` Rosenstein算法
- 审查问题：简化FNN在什么情况下会给出错误的嵌入维数？RQA的对角线/垂直线提取逻辑有遗漏吗？Lyapunov的Theiler window (2*delay)是否太小？

**个体校准器** (`src/models/individual_calibrator.py`，957行)
- 双模式：直接启发式（规则映射）+ 遗传算法（可选）
- 核心输出：贝叶斯后验分布（5个bin × 概率），非点估计
- 审查问题：后验构建的高斯核宽度(σ=0.15)是否合理？角色分类的权重分配有理论依据吗？GA的交叉/变异算子是否正确？

### 2. 功能完备性：文档承诺的都实现了吗？

对照 CLAUDE.md 列出的21项功能：
- 集体层14项全部可运行✅
- 个体层4项场景模板正确分类✅
- B站管道有代码但缺真实字幕数据⚠️
- 跨平台验证/自动采集未开始❌

审查问题：有没有文档写了但代码里找不到的功能？有没有代码里有但文档没列的功能（隐藏功能或死代码）？

### 3. 模块间一致性

- `build_state_points()` 在 `phase_diagram.py` 和 `backtest.py` 间已去重，但两者对 `MemeStatePoint` 的使用方式是否一致？
- curator.py 的 `to_sir_estimation()` 和 bilibili_pipeline.py 的 `fit_sirs_m_model()` 是否对SIR参数的定义一致（β/γ vs β/σ）？
- phase_diagram.py 的 `estimate_params_from_lifecycle` 调用和 sir_meme.py 的原函数参数是否匹配？

### 4. 边界与错误处理

- `attractor.py` 的 `takens_embedding()` 在序列短于 `embedding_dim * delay` 时抛ValueError——调用方都有保护吗？
- `individual_calibrator.py` 的 `calibrate_from_observation()` 当观测信号为空时会返回全0剖面——这是预期行为还是应该报错？
- `backtest.py` 的 `_make_noisy_meme()` 用 `copy.deepcopy` 修改 chaos_vector 的副作用——会对原 curator 数据造成污染吗？

### 5. 数值稳定性

- 所有 `np.log()` 调用是否有 epsilon 保护？
- `solve_ivp` 失败时各模块的处理方式是否一致（有的返回None，有的raise）？
- 熵计算当所有hist值相等（完全均匀分布）时是否确实返回最大值1.0？

## 审查输出格式

请按以下结构回复：

```
### 正确性问题
- [模块: 函数] 问题描述 + 严重程度(🔴/🟡/🟢) + 修复建议

### 完备性问题
- 文档写了但代码缺失的功能
- 代码有但文档未列的死代码

### 一致性问题
- 模块间参数命名/定义不一致

### 边界问题
- 空输入/极端输入下的行为

### 总体评价
- 代码可信度评级 (A-F)
- 最需要重写的模块
- 最大优势
```

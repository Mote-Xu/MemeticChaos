# 外部 AI 代码审查请求

> 将此文件完整发送给外部 AI（GPT/Gemini/Claude）进行代码审查。
> 请以**代码审查者**而非崇拜者的视角，诚实评估实现质量。

---

你正在审查一个计算模因学 × 混沌动力学研究项目。项目用 2020-2025 年中国互联网热梗数据为人类集体情感混沌属性建模。

## 项目规模

```
5945 行 Python，18 个源文件，10 个模块
24/24 单元测试通过
10 次 Git commit，已推送 GitHub
```

文件分布：
```
sir_meme.py                 731 行  SIR/SIRS/双群体 + 参数拟合 + 熵分析
abm_simulation.py           667 行  300-Agent 无标度网络仿真
attractor.py                586 行  Takens/RQA/Lyapunov/盆地检测
individual_calibrator.py    957 行  直接启发式+GA, 贝叶斯后验
curator.py                  252 行  策展数据管理
lifecycle.py                263 行  生命周期剖面 + 分类
sentiment.py                228 行  情感弧线分类
phase_detect.py             272 行  相变检测
phase_diagram.py            452 行  模因相图 ★核心产出
backtest.py                 487 行  历史回测+鲁棒性验证
plots.py                    318 行  可视化
test_sir_model.py           277 行  24个测试
```

## 关键实现细节

### SIR 模型 (sir_meme.py)
- 标准 SIR / SIRS（变异复燃 μ）/ 双群体 三种 ODE 系统
- `scipy.integrate.solve_ivp` RK45 求解
- `scipy.optimize.curve_fit` 从时间序列反推 β, γ, R₀
- 香农熵轨迹追踪系统混沌度
- 四类模因分类器（脉冲/爆发/长尾/流产）

### ABM 仿真 (abm_simulation.py)
- 无 Mesa 依赖的独立实现
- Barabási-Albert 无标度网络
- 5 种 Agent 角色（normal / builder / injector / lurker）
- 情感传染 + 回音壁效应 + 熵增漂移 + 混沌投放
- 稳态锚点防止 chaos_position 无限漂移

### 吸引子检测 (attractor.py)
- Takens 延迟坐标嵌入（互信息法 τ + 假近邻法 d）
- 递归矩阵构建 + RQA 量化（REC / DET / LAM / ENTR / L_max）
- Rosenstein 算法估算最大 Lyapunov 指数
- 吸引子盆地自动检测

### 个体校准器 (individual_calibrator.py)
- 双模式：直接启发式（规则映射，秒出）+ 遗传算法（可选）
- **贝叶斯后验分布**输出（5 bins × 概率），非单点估计
- 4 种场景模板验证（builder / injector / follower / lurker）
- 显式声明"小真实不可穿透"约束

### 验证层 (backtest.py)
- 时序切分回测（逐年推进训练窗口）
- 留一交叉验证
- 吸引子盆地鲁棒性（删 30% 梗 / 20% 噪声扰动）
- 结果：chaos MAE = 0.186 (2.7x 随机基线)，盆地 100% 稳定

## 已知问题和自我批评

1. **R₀ 估算偏保守**：`estimate_params_from_lifecycle()` 从 qualitative 数据估算的参数分辨率低，需要真实时间序列验证
2. **ABM 参数过度趋同**：传染率稍高时 agents 全部收敛到同一 chaos 位置，需要更多随机噪声
3. **GA 校准区分度不足**：遗传算法模式在简化仿真中无法可靠区分角色，目前主力是直接启发式
4. **仅有 SIR 层有测试**：ABM / Attractor / Calibrator 只有 demo 脚本，缺 formal tests
5. **无真实数据验证**：所有 SIR 参数来自 qualitative 估算，B站 字幕数据尚未接入

## 审查问题

请针对以下方面给出诚实评估：

1. **代码架构**：模块划分是否合理？耦合度如何？有没有明显的反模式？
2. **算法正确性**：SIR、Takens、RQA、GA 的实现是否有逻辑错误或数值问题？
3. **测试覆盖**：当前只有 SIR 层有 24 个测试。ABM / Attractor / Calibrator 最需要补哪些测试？
4. **数值稳定性**：Lyapunov 估算、遗传算法收敛、熵计算有没有数值陷阱？
5. **可复现性**：随机种子管理是否一致？仿真结果是否可复现？
6. **最薄弱环节**：如果要挑一个最需要重写的模块，是哪个？为什么？
7. **代码质量**：命名、注释、类型标注、错误处理有什么改进空间？

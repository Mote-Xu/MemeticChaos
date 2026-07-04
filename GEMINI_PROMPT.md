# MemeticChaos 项目现状（给外部 AI 的求助）

> 最后更新：2026-07-04 (v4.1 — regime transition map 完成)

## 一句话

中国互联网集体情感的混沌属性建模。当前阶段不再是"预测器"——是**刻画互联网集体意识相变结构的动力系统模型**。H1b 的否定结果（VARX R²=-0.32 < lag-1 R²=+0.44）是两年来最具科学含金量的发现：月度尺度上，点预测是逆物理学的。系统本质是**流形上的密度演化 + 相变边界**，而非轨迹外推。

## 数据资产

```
外部场:      51 关键词 × 132 月 (Google Trends 2015-2025)
叙事:        57 条 (22 B站 + 36 曲线, 含 spread_phases/mutations/semantic_drift)
Level 1:     127 月 × 4 硬事实特征 (Stage/Mutation/Inst/Drift)
Level 2:     127 月 × 10 维 Narrative State x(t) (PCA d90=10)
Regime Map:  4 个叙事气候相区 + 转移矩阵 + 驻留时间 (GMM, BIC 最优)
实时采集:    微博50 + 百度50 + 知乎30 = 130条/小时 (mote-home 24/7)
Dashboard:   chaos.mote-pal.xyz (Flask + ECharts)
Agent:       Stella ⭐ (OpenClaw + 企业微信 + DeepSeek V4 Flash)
```

## H1 假说验证结果

| 假说 | 结论 | 证据 |
|------|:--:|------|
| **H1a** (低维表示) | ✅ SUPPORTED | 18 维 → d90=10, d95=12 |
| **H1b** (动力学连续性) | ❌ REJECTED | VARX test R²=-0.32 < lag-1 R²=+0.44 |
| **H1c** (当前状态主导) | ⚠️ MIXED | State-only R²=+0.14, 外生变量不帮正向忙 |

**核心发现**: 月度尺度集体叙事动态由随机漂移主导。不是"建模失败"——是系统物理属性。

## v4.1 新增: Regime Transition Map

GMM 对 10 维 x(t) 聚类, BIC 自动选出 **4 个叙事气候相区**:

| Regime | 月数 | 占比 | 主导阶段 | 时期 | 自持概率 |
|--------|:--:|:--:|------|------|:--:|
| **R0** Peak Burst | 4 | 3.1% | peak | 罕见爆发 | 50.0% |
| **R1** Origin-Active | 39 | 30.7% | origin | 2020-2022 活跃期 | 87.2% |
| **R2** Fixation Lock | 38 | 29.9% | fixation | 2022.12→至今 | **97.3%** |
| **R3** Origin-Stable | 46 | 36.2% | origin | 2015-2019 基线 | 87.0% |

### 转移矩阵 (几乎单向不可逆)

```
R3 (基线) ──8.7%──→ R1 (活跃) ──5.1%──→ R2 (僵化锁死)
   ↑                  ↑                     │
   └──4.3%── R0 (爆发) ──────────────────────┘ (罕见)
                      
R2 → R1: 仅 2.7%          R1 → R3: 0%
R2 → R3: 0%               R2 → R0: 0%
```

### 关键发现

- **R2 自持 97.3%** — 一旦进入 fixation lock，几乎不可能自行逃脱
- **当前 R2 驻留 37 个月** — 中位驻留时间 (19 月) 的 2 倍。极端值
- **转移矩阵趋近三角** — R3→R1→R2, 有明确的时间箭头
- **不可逆漂移**: 叙事生态系统天然向 fixation 演化，且越陷越深
- 切换率仅 0.111/月, 但归一化熵 0.932 — 系统在少数几次切换中探索了大部分可能的转移路径

## FR31 四指标当前值 (2025-12)

| 指标 | 值 | 物理含义 |
|------|:--:|------|
| **Inertia** | 0.77 | 势阱深度 — fixation 占 52%, origin+emergence=0% |
| **Resilience** | 0.37 | 盆地回归时间 — 2025H2 恢复力归零, 偏离均衡 3.9σ |
| **Sensitivity** | 0.56 ↑ | Jacobian 谱半径 — Critical Slowing=0.76, PC4/PC5 >2x 基线方差 |
| **Position** | R2 边缘 | 吸引子盆地边缘位置, 近 12 月 0 次阶段转换 |

**四指标联合诊断**: 高惯性 + 零恢复力 + 敏感性加速上升 = **亚稳态 (Metastable)**。系统同时"动不了"和"越来越容易炸"。Critical Slowing 0.76 是经典的分岔前预警信号。R2 的 37 个月锁定期是历史极值。

## 当前已建成 (v4.1)

```
src/
├── data/
│   ├── narrative_hard_facts.py         ✅ Level 1: 4 硬事实 × 127 月
│   └── scraper.py / live_pipeline.py / signal_pipeline.py
├── models/
│   └── representation_learning.py      ✅ Level 2: PCA d90=10 + H1 验证
├── analysis/
│   └── regime_detector.py              ✅ v4.1: GMM 4 相区 + 转移矩阵
├── advisor/
│   └── metrics.py                      ✅ FR31 四指标: I/R/P/S
└── dashboard/
    └── app.py                          ✅ Flask :8931, 9 端点
```

## 上次外部 AI 反馈总结 (Gemini + GPT, 2026-07-04)

### 双方共识
1. H1b 否定是科学资产，不是失败 — 点预测在物理上是大错对象
2. 从点预测转向分布/结构/相变 — 转移概率 > 点估计
3. 图的价值在动力学 (Laplacian, diffusion)，不在 embedding
4. 缺第四个指标 Sensitivity — 已补上 ✅
5. u(t) 调制方差(GARCH)，不调制均值(Ridge) — 接入方式错了
6. 多尺度解耦: 宏观慢变量(月度 Level 1) + 微观快变量(小时级 scraper → 闪洪预警)

### GPT 单独判断: "你现在建模的不是轨迹，是流形切换"
> "You're not modeling trajectories. You're modeling a switching manifold where attention acts as an order parameter of latent phase transitions."

- 不是在预测 x(t+1)，是在判断 x(t) ∈ M_k（当前在哪个流形上）
- Graph ≠ data structure, graph = phase space geometry
- Attention ≠ observation, attention = order parameter of symmetry breaking
- 随机游走 ≠ 无结构, 结构在 diffusion kernel 里

## 待建 (v4.1 路线图)

| 优先级 | 任务 | 说明 |
|:--:|------|------|
| P0 | `src/advisor/persona.py` | 双层编码器: 用户文本 → sentence-transformers → 叙事图节点投影 |
| P1 | `src/data/micro_burst_detector.py` | 小时级 scraper → 日度波动率熵 → Sensitivity 微观修正 |
| P2 | `src/agent/stella_plugin.py` | 控制论推理链 Prompt → DeepSeek → 企微 `/chaos` 交互 |
| P2 | 图谱拓扑指标 | λ₁(惯性) + λ₂(连通度) → 替换 metrics.py 的纯向量指标 |

## 核心困惑（请你帮忙想的）

### 1. 不可逆漂移是真实的还是采样偏误？

转移矩阵显示 R3→R1→R2 的单向箭头。R2 的自持概率 97.3%。
但这只覆盖了 10 年。更长时间尺度上，fixation lock 是否真的不可逆？
还是说存在更慢的恢复机制（代际更替、平台迁移、重大政策转向）？
如果不可逆是真的，这对互联网文化意味着什么？

### 2. R2 → ?  系统会怎么离开当前的 fixation lock？

37 个月是历史极值。四个指标显示亚稳态。
什么类型的冲击最可能触发逃逸？外部政策？代际新平台？
还是系统会在 fixation 中无限期锁死，直到数据覆盖的尺度不够展示恢复？

### 3. 两个 Origin 相区 (R1 vs R3) 的本质区别是什么？

BIC 把它们分成两个相区，都是 origin 主导，但时期不同（2015-2019 vs 2020-2022）。
区分它们的是疫情前后叙事生态的结构性变化，还是仅仅时间上的漂移？
如果重跑数据到 2030 年，会出现 R4 (Origin-Neo) 吗？

### 4. 概率密度预测怎么做？

已经放弃了点预测。下一步是 GARCH 建模方差、还是 Fokker-Planck 方程数值解、
还是直接 MS-AR (Markov-Switching AR)？哪个最适配当前的 10 维小样本数据？

### 5. Graph Laplacian 怎么和 GMM 相区合龙？

图拓扑指标（λ₁, λ₂）和向量指标（PCA + GMM）目前是两套独立的描述。
怎么把它们统一？λ₂ → 0 是否恰好对应 R2 相区？
图结构分裂是否可以作为 regime switch 的早期预警？

### 6. 个体→集体映射怎么避免"伪精密"？

sentence-transformers 余弦相似度可以做用户文本→梗节点的投影。
但两个短文本的余弦相似度 0.7 vs 0.8 在现实中意味着什么？
会不会把一个粗糙的数学操作伪装成有物理意义的"个体坐标"？
FR31 的底线是"不知道就是不知道"——怎么守住这条线？

## 技术栈

Python 3.12 + numpy/scipy/pandas/scikit-learn/sentence-transformers + Flask + ECharts
LLM: DeepSeek API
Agent: OpenClaw + 企业微信 (mote-home, Ubuntu 24.04)
Dashboard: chaos.mote-pal.xyz (cloudflared tunnel + token auth)

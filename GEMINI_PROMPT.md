# MemeticChaos 项目现状（给外部 AI 的求助）

> 最后更新：2026-07-04 (v4.1 — regime map + irreversibility tests 完成)

## 一句话

中国互联网集体情感的混沌属性建模。当前阶段不再是"预测器"——是**刻画互联网集体意识相变结构的动力系统模型**。H1b 的否定结果（VARX R²=-0.32 < lag-1 R²=+0.44）是两年来最具科学含金量的发现：月度尺度上，点预测是逆物理学的。系统本质是**流形上的密度演化 + 相变边界**，而非轨迹外推。RQA 已确认 Fixation Lock (R2) 为真实的结构分离, 非表示 artifact; Time-Reversal 表明不可逆性来自外部场慢漂移而非内禀动力学。

## 数据资产

```
外部场:      51 关键词 × 132 月 (Google Trends 2015-2025)
叙事:        57 条 (22 B站 + 36 曲线, 含 spread_phases/mutations/semantic_drift)
Level 1:     127 月 × 4 硬事实特征 (Stage/Mutation/Inst/Drift)
Level 2:     127 月 × 10 维 Narrative State x(t) (PCA d90=10)
Regime Map:  4 叙事气候相区 + 转移矩阵 + 驻留时间 (GMM, BIC 最优)
Irreversibility: RQA 确认 R2 真实分离 + Time-Reversal 确认无内禀时间箭头
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

### Irreversibility Tests (GPT 质疑后的验证, 2026-07-04)

针对"R2 是否 GMM 投影 artifact"的质疑, 做了两个独立测试:

**Recurrence Analysis (RQA)**:
- R2 与 R1/R3 **零跨相区复发** (ε=1.84, ≥12月间隔)
- R1↔R3 有 10 对复发 — 两个 Origin 相区在状态空间中是连通的
- 相区间最小距离: R1↔R3=1.18 ≪ R2↔R1=2.24 < R2↔R3=3.38
- **结论: R2 是真实的结构分离, 非 GMM artifact**

**Time-Reversal Test**:
- 漂移量 0.08, 0/10 维有显著方向性
- 置换测试: 原始步长 2.46 ≪ 随机重排 5.37 (时序平滑, 非方向性)
- 步间余弦均值 -0.29 (弱均值回归, 不显著)
- 前后半段漂移对齐 -0.13 (漂移方向反转过)
- **结论: 逐月动力学没有内禀时间箭头**

**综合: WEAK_IRREVERSIBILITY**
- R2 分离是真实的物理结构 (RQA 确认)
- 不可逆性来自外部场 u(t) 的慢漂移, 而非系统内禀动力学
- 如果 u(t) 逆转 (重大政策/平台变迁), 系统理论上可能回归
- 这是最干净的科学立场: 证实结构, 不宣称不可逆

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
│   ├── regime_detector.py              ✅ v4.1: GMM 4 相区 + 转移矩阵
│   └── irreversibility_test.py         ✅ v4.1: RQA + Time-Reversal
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

### 1. 不可逆漂移 → 已部分验证 ✅

转移矩阵 R3→R1→R2。Irreversibility tests 确认:
- R2 是真实的结构分离 (RQA: 零跨相区复发)
- 但不可逆性来自 u(t) 慢漂移, 非内禀动力学 (Time-reversal: 对称)
- 如果 u(t) 逆转, 系统理论上可能回归

**新问题**: u(t) 中哪些分量驱动了向 R2 的漂移? 经济压力? 平台算法? 政策收紧?
能否从 51 维外部场中识别出导致 fixate 的关键变量?

### 2. R2 → ? 逃逸机制

37 个月是历史极值。但 time-reversal 说动力学本身没有固化——
这意味着如果 u(t) 发生足够大的变化, R2 理论上可逃逸。
需要监测 u(t) 的变化幅度 (∥Δu∥) 作为逃逸可能性的代理指标。

### 3. R1 vs R3 → GPT 的判断: 可能不是两个 regime, 是一个 regime 的时间切片

R1↔R3 在 RQA 中有复发 (10对), 最小距离仅 1.18。
这两个"origin-dominated"相区可能本质是同一结构,
被疫情前后的不同 u(t) 条件形变。
问题转化为: p(x|u(t)) 的条件分布结构, 而非 p(x) 的静态聚类。

### 4. 概率密度预测 → 方案待定

GARCH (方差建模) vs Fokker-Planck (密度演化) vs MS-AR (regime-switching)。
目前 127 月样本对 GARCH/Fokker-Planck 偏少。
MS-AR 最直接——已有 4 个 regime, 可以直接估计 regime-dependent 参数。

### 5. Graph + GMM → 转为 spectral phase transition 问题

GPT 的建议: 不要合龙, 而是把 λ₂ (代数连通度) 直接作为 regime 切换的
早期预警信号。λ₂ → 0 应对应 R2 的进入点。
如果月度图拓扑数据可用, 可以直接验证这个关系。

### 6. 个体→集体 → FR31 必须改为概率分布接口

GPT 的警告: 0.7 vs 0.8 的余弦相似度在语义空间中没有绝对意义。
正确做法: 输出 P(meme_i | user_text) 分布 + 熵 + top-k 浓度,
而非"你在 X 节点上"的点定位。
必须显式标注不确定性。FR31 是测量系统, 不是判断系统。

## 技术栈

Python 3.12 + numpy/scipy/pandas/scikit-learn/sentence-transformers + Flask + ECharts
LLM: DeepSeek API
Agent: OpenClaw + 企业微信 (mote-home, Ubuntu 24.04)
Dashboard: chaos.mote-pal.xyz (cloudflared tunnel + token auth)

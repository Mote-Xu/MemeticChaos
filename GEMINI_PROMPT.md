# MemeticChaos 项目现状（给外部 AI 的求助）

> 最后更新：2026-07-05 (v4.1 收敛 — 外部 AI 共识: 停建新模块, 做四层理论统一)

## 一句话

中国互联网集体情感的混沌属性建模。当前阶段不再是"预测器"——是**刻画互联网集体意识相变结构的动力系统模型**。H1b 的否定结果（VARX R²=-0.32 < lag-1 R²=+0.44）是两年来最具科学含金量的发现：月度尺度上，点预测是逆物理学的。系统本质是**流形上的密度演化 + 相变边界**，而非轨迹外推。RQA确认R2为真实结构分离; Time-Reversal表明无内禀时间箭头; Control Manifold推翻"u(t)极端→R2"假说, 修正为"u(t)沿AI/Tech轴单调漂移→系统滑入R2盆地被困". persona.py已实现P(meme|text)概率投影, 双熔断.

## 数据资产

```
外部场:      51 关键词 × 132 月 (Google Trends 2015-2025)
叙事:        57 条 (22 B站 + 36 曲线, 含 spread_phases/mutations/semantic_drift)
Level 1:     127 月 × 4 硬事实特征 (Stage/Mutation/Inst/Drift)
Level 2:     127 月 × 10 维 Narrative State x(t) (PCA d90=10)
Regime Map:  4 叙事气候相区 + 转移矩阵 + 驻留时间 (GMM, BIC 最优)
Irreversibility: RQA 确认 R2 真实分离 + Time-Reversal 确认无内禀时间箭头
Control Manifold: 51维外部场 → 3维控制轴 z(t). R2 不在极端区, AI/Tech轴主导漂移
Persona:      57节点叙事图 → P(meme|text) 概率分布, 96.5%召回, 双熔断
实时采集:    微博50 + 百度50 + 知乎30 = 130条/小时 (mote-home 24/7)
Dashboard:   chaos.mote-pal.xyz (Flask + ECharts)
Agent:       Stella ⭐ (OpenClaw + 企业微信 + DeepSeek V4 Flash)
              企微管道已通 — 每小时刷新 STATE.md → Stella 读文件 → 白话诊断
              DM agent 专属, 不污染群聊和其他项目
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

### Control Manifold: u(t) 逼迫假说修正 (2026-07-04)

将 51 维外部场 PCA→8维→Diffusion Map→3维控制轴 z(t), 检测 R2 是否在 z(t) 极端区间.

**推翻原假说**: R2 不在 z-space 极端位置 (d=0.11 < 其他相区均值 0.15).
但在 z₂ 轴上 R2 处于 99.2% 分位数 — 单轴极端.

**z₁ 轴由 AI/Tech 话语主导**: AI (r=-0.68), ChatGPT (r=+0.54).
外部场十年趋势由技术话语的单调上升驱动. z₁ 范围 [-0.62, 0.003] — 几乎单极.

**R2 进入点**: 2020-11, 在 z-space 原点 (0.001, 0.000) — 中性过渡点.

**修正物理图景**: 不是 "u(t) 极端 → R2", 而是 "u(t) 沿 AI/Tech 轴单调漂移 →
系统经过 R2-favoring 盆地 → 被困". u(t) 的**方向**决定系统去向,
该方向十年来未逆转. R2 逃逸需要 u(t) 漂移方向反转.

### Persona Encoder: P(meme|text) 概率分布 (2026-07-04)

`src/advisor/persona.py` — 个体文本→叙事图投影器.

**设计原则** (来自 GPT 的 epistemic humility 要求):
- 输出分布, 非点定位. P(meme_i | user_text) 完整概率向量
- 双熔断: FREE_NOISE (sim<0.30) + AMBIGUOUS (gap<0.03)
- 校准后阈值适配中文跨域文本 (自检 96.5% top-1)
- "不知道"是默认, "确定"需要多个信号同时通过

**已验证语义**:
- "奋斗鸡汤+老板点名" → 打工人 (sim=0.51)
- "抠字眼+阴阳怪气" → 普信男 (sim=0.57)
- "不想努力了" → 躺平/轻松绷住 歧义 (gap=0.02) → MELTDOWN
- 纯噪声 → FREE_NOISE (sim=0.25) → MELTDOWN

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
│   ├── irreversibility_test.py         ✅ v4.1: RQA + Time-Reversal
│   └── control_manifold.py             ✅ v4.1: u(t)→z(t) 控制轴分析
├── advisor/
│   ├── metrics.py                      ✅ FR31 四指标: I/R/P/S
│   └── persona.py                      ✅ P(meme|text) 概率编码器, 双熔断
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
| **P0** | **`src/advisor/persona.py`** | ✅ DONE. 概率编码器 + 双熔断, 96.5%召回 |
| P1 | `src/data/micro_burst_detector.py` | 小时级 scraper → 日度波动率熵 → Sensitivity 微观修正 |
| P2 | `src/agent/stella_plugin.py` | 控制论推理链 Prompt → DeepSeek → 企微 `/chaos` 交互 |
| P2 | 图谱拓扑指标 | λ₁(惯性) + λ₂(连通度) → 替换 metrics.py 的纯向量指标 |

### Stella 企微集成 (2026-07-04 晚)

管道已通, 实测可用:
- Stella 读取 `STATE.md` (每小时 cron 刷新) → 白话翻译
- 用户问 "现在互联网叙事什么状态" → 返回 FR31 诊断
- 实测回复质量: "表面热闹、底层固化"、"AI/Tech轴垄断十年"、"绷紧的弦太久没换过音高"

集成方式: DM agent 专属 TOOLS.md + STATE.md, 不污染群聊和其他项目.
失败尝试: OpenClaw plugin skill (需要 MCP, 不支持), 全局 TOOLS.md (污染风险).

### 外部 AI 第三轮反馈共识 (2026-07-05)

**双方共识**: 不要再堆新模块。系统已经进入理论收敛阶段。下一步不是写代码,
是定义 State/Observation/Control/Dynamics 四层形式化框架。

**关键判断分歧与收束**:
- R1/R3 合并: 双方 + RQA 都支持 → **确定合并为单一 Origin manifold**
- AI/Tech 因果性: Gemini 确定因果, GPT 质疑可能是 time proxy → **需要 counterfactual test**
- Graph λ₂: Gemini 推到立即做, GPT 反对 (边是人工的, λ₂ 测到的是建图规则不是互联网动力学) → **先不做, 等 scraper 累计真实动态边**
- FR31 的 Measurement vs Policy 分层: GPT 强调必须分开, 否则从"测量"滑向"爹味建议" → **采纳, 写进 skill**

**GPT 的核心建议 (被采纳为下一步)**:
定义四层形式化框架——State / Observation / Control / Dynamics.
这一步不做, 后面所有新模块都会让系统越来越散.

## 核心困惑（请你帮忙想的）

### 1. 不可逆漂移 → u(t) 驱动机制已初步定位 ✅

Control manifold 分析: 51维外部场 → 3维控制轴.
- z₁ 由 AI/Tech 话语主导 (AI r=-0.68, ChatGPT r=+0.54)
- z₁ 范围 [-0.62, 0.003] — 十年来几乎单向漂移
- R2 进入于 2020-11, z-space 原点 — 中性过渡点, 非极端条件
- **修正**: "u(t) 极端 → R2" 被推翻. 正确机制是 "u(t) 方向决定盆地, 系统沿方向滑入并被困"

**新问题**: AI/Tech 话语是否能作为 R2-favoring 的因果变量,
还是仅仅与时间共线? 如果 AI 搜索热度逆转 (技术衰退周期),
系统能否沿原路返回?

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

### 6. 个体→集体 → persona.py 已实现, 待实战校准 ✅

已建: `src/advisor/persona.py` — 57节点概率投影器.
- 自检 top-1=96.5%, 双熔断运行正常
- 语义映射已验证: 职场抱怨→打工人, 关系博弈→普信男, 倦怠→躺平/轻松绷住歧义
- 阈值按跨域中文校准 (FREE_NOISE=0.30, AMBIGUITY=0.03)

**新问题**: 模型在真实聊天记录上的表现如何?
虹姐的 17529 行私聊 + 6527 行群聊数据可以用来做外部验证.
另外: 余弦相似度 0.51 vs 0.57 的差异在现实中有意义吗?
需要积累 case study 来校准"多高算高".

### 7. persona 跨域校准 → 鸿沟本身是信号

同域自检 max_sim=0.95, 跨域用户文本仅 0.42-0.57.
外部 AI 一致反对 query rewriting — 鸿沟本身就是信息量:
口语化程度反映了"个体与主流叙事话语的距离".
建议保留双投影 (raw + canonical) 作为额外特征.
另: 虹姐 2.4 万行聊天记录可作为外部验证集, 待评估.

## 技术栈

Python 3.12 + numpy/scipy/pandas/scikit-learn/sentence-transformers + Flask + ECharts
LLM: DeepSeek API
Agent: OpenClaw + 企业微信 (mote-home, Ubuntu 24.04)
Dashboard: chaos.mote-pal.xyz (cloudflared tunnel + token auth)

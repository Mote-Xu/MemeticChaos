# MemeticChaos 项目现状（给外部 AI 的求助）

> 最后更新：2026-07-06 (Q9: regime 的物理含义 — 结构崩塌还是共识收敛?)

## ⚠ 最重要的元注记

**本项目的架构已被多次推翻。**

- v1-v3: "混沌轴 + 约束场 + 秩序形态预测" → 2026-07-04 审计发现混沌轴是人工权重、约束场是静态标签、内部叙事层从未激活 → **推翻**。
- v4.0: "H1 假说检验 — 点预测" → H1b REJECTED, 月度不可预测 → **推翻预测范式**。
- v4.0: "AI/Tech 轴主导控制流形" → Counterfactual 发现平台 16.5× > AI 1.9× → **推翻驱动因假设**。
- v4.1: "两层架构" → 外部 AI 指出层次混淆 → **重建为四层形式化**。
- v4.1: "MS-AR 建模 P(S'|S,z)" → 仅 14 次切换, z1 是 regime discriminator 不是 modulator → **推翻建模方向**。
- 2026-07-05: "三数据源实时采集" → 审计发现 scraper 从未进入分析链 → **推翻数据假设**。
- 2026-07-06: "测量的是集体情感状态" → 发现测量的从来是叙事化反应, 不是现实结构 → **推翻测量对象的定义**。
- 2026-07-06: "R2 = fixation lock, 系统被外力困住" → Q9 发现 R2 可能是共识收敛而非结构崩塌 → **推翻 regime 的物理含义**。

**当前架构中几乎肯定仍有大量未被识别的预设。** 没有任何理由相信这一版不会被推翻。
请外部 AI 不要接受任何"给定的"前提——包括四层形式化、regime 定义、甚至"叙事"这个概念本身。
如果觉得某个前提需要被质疑，直接指出来。

### 方法论原则

**未被意识的预设 = 绝对秩序。** 它不觉得自己是预设，就变成"显然""自然""大家都这么认为"。

每引入一个概念，标注它是三类之一：
- **数学工具** — PCA、GMM。使用但不声称物理解释。
- **观测描述** — "R2 内部方差放大 2.31×"。数据在说什么。
- **物理声称** — "势阱""分岔""hysteresis"。必须独立证据，随时可被推翻。

如果分不清某概念属于哪一类，它很可能就是未被意识的预设。

## 一句话

中国互联网集体情感的混沌属性建模。当前阶段不是"预测器"——是**刻画互联网集体意识相变结构的动力系统模型**。H1b REJECTED（VARX R²=-0.32）：月度尺度上点预测是逆物理学的。系统本质是流形上的密度演化 + 相变边界。RQA确认R2为真实结构分离; Counterfactual发现平台(16.5x)>AI(1.9x) — 平台生态是primary control driver。MS-AR第一刀: R2内部方差放大2.31×, PC4/PC5与z1强耦合(r=±0.85), z(t)调控方式不是触发切换而是在regime内部形变状态。数据审计: scraper从未进入分析链(v2.0已修复采集端, 下游待接)。信号质量危机: 所有数据测量的不是现实结构, 是人群的叙事化反应 — 这个叙事化倾向本身就是要测量的对象。

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
实时采集 v2.0: 微博50 + 百度50 + 知乎30 = 130条/小时 (mote-home 24/7)
              → 全量 384 维 headline embedding (paraphrase-multilingual-MiniLM-L12-v2)
              → 日聚合: 梗语义相似度 + 注意力集中度 + 新颖度
              ⚠ 数据审计: scraper 历史上从未进入分析链 (仅 Dashboard widget).
                v2.0 已修复采集端, 下游月度分析待接入 (Q7).
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

**⚠ 2026-07-05 Counterfactual 修正**: 平台生态(16.5×) > AI(1.9×).
z₁ 轴标签保留"AI/Tech"但 primary control driver 已修正为平台此消彼长。
AI/Tech 话语是平台变化的副产品, 非独立 driver. 详见 Q3-4.

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

## MS-AR 第一刀: z(t) 条件 regime 转移 (2026-07-05)

**约束**: 127 月仅 14 次 regime 切换。z1 在 R0/R1/R3 内方差近乎零 — 是 regime discriminator 而非连续 modulator。

**三项测试**:
1. 时间切分 → 2018-01 切分显著 (p=0.029)
2. z1 置换 → 切换不聚集在特定 z1 (p=0.713, 统计效力不足)
3. **R2 内部漂移** → PC4 r=+0.85, PC5 r=-0.87 (p≈0), 方差放大 2.31×

**判断**: `PROCEED_TO_PHASE_2`。z(t) 调控系统但不是在触发切换 — 是在 regime 内部形变状态分布。Phase 2 方向修正: 放弃 P(S'|S,z), 做 p(x|R2, z1)。详见 Q6。

## 当前已建成 (v4.1)

```
src/
├── data/
│   ├── narrative_hard_facts.py         ✅ Level 1: 4 硬事实 × 127 月
│   ├── scraper.py (v2.0: full headline embedding) / live_pipeline.py / signal_pipeline.py
├── models/
│   └── representation_learning.py      ✅ Level 2: PCA d90=10 + H1 验证
├── analysis/
│   ├── regime_detector.py              ✅ v4.1: GMM 4 相区 + 转移矩阵
│   ├── irreversibility_test.py         ✅ v4.1: RQA + Time-Reversal
│   ├── control_manifold.py             ✅ v4.1: u(t)→z(t) 控制轴分析
│   └── ms_ar_first_cut.py              ✅ v4.1: MS-AR 第一刀 (time-split+bootstrap+permutation+R2 drift)
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
| **P0** | **数据管道接入: 日级 embedding → 月度语义状态** | Q7 结论 — scraper 数据从未进入分析链, 需在断开处接上 |
| **P0** | **MS-AR Phase 2: `p(x\|R2, z1)`** | Q6 结论 — 建模 fixation 内 z(t) 调制效应 |
| P0 | `src/advisor/persona.py` | ✅ DONE. 概率编码器 + 双熔断, 96.5%召回 |
| P1 | `src/data/micro_burst_detector.py` | 换输入: 关键词命中 → 日级 embedding novelty+concentration |
| P2 | `src/agent/stella_plugin.py` | 控制论推理链 Prompt → DeepSeek → 企微 `/chaos` 交互 |
| P2 | 图谱拓扑指标 | λ₁(惯性) + λ₂(连通度) → 替换 metrics.py 的纯向量指标 |

### Stella 企微集成 (2026-07-04 晚)

管道已通, 实测可用:
- Stella 读取 `STATE.md` (每小时 cron 刷新) → 白话翻译
- 用户问 "现在互联网叙事什么状态" → 返回 FR31 诊断
- 实测回复质量: "表面热闹、底层固化"、"AI/Tech轴垄断十年"、"绷紧的弦太久没换过音高"

集成方式: DM agent 专属 TOOLS.md + STATE.md, 不污染群聊和其他项目.
失败尝试: OpenClaw plugin skill (需要 MCP, 不支持), 全局 TOOLS.md (污染风险).

### 外部 AI 第三轮反馈后的行动 (2026-07-05)

**四层形式化已完成** → `FORMALISM.md` (完整项目逻辑, 12章).
定义了 State / Observation / Control / Dynamics 的本体论层次,
所有已有模块归入对应层次。新模块不再需要新概念。

**关键的边界定义**: FR19 → FR31 有适用范围边界。
集体数据不可能覆盖个人现实中所有的结构规律。
有一整类个体层面的结构性知识在集体叙事数据中完全没有信号
(如"表白无益处"——没有梗、没有热搜、违背主流道德叙事、
但在个人现实中可能是经过反复验证的结构性事实)。
FR31 的正确行为是说"不知道", 不是硬推导。
这不是系统缺陷——是科学诚实性的基石。

**请求外部 AI 审查**:
1. 四层形式化的层次划分是否合理? 有没有跨层混淆?
2. FR19→FR31 的边界定义是否完整? 有没有遗漏的情况?
3. Dynamics 层: MS-AR vs GARCH vs Fokker-Planck 的优先级?
4. Control 层: AI/Tech 轴的 counterfactual test 是否值得做?

完整的形式化文档见 `FORMALISM.md`。

### Counterfactual Test (2026-07-05)

Leave-one-group-out: 抹平每个关键词组在 2023 后的流量, 重跑 Diffusion Map.

| Group | 漂移比 | 判定 |
|------|:--:|------|
| **平台** (B站/知乎/小红书/抖音/微博/快手) | **16.5x** | Primary driver |
| 文化 (二次元) | 11.6x | 仅1关键词, 待扩展 |
| 政策 (体制内) | 5.8x | 仅1关键词 |
| AI (AI/ChatGPT) | 1.9x | Weak causal |
| 经济 (房价/就业/消费等7词) | 1.4x | Time proxy |
| 国际 (美国/日本/韩国/俄罗斯) | 1.3x | Time proxy |

**关键修正**: 之前认为"AI/Tech轴主导控制流形"是错误的.
真正的 primary control driver 是**平台生态系统** (六大平台的此消彼长).
AI/ChatGPT 话语是平台变化的副产品, 非独立 driver.

### Persona 五态模型 (2026-07-05)

`persona.py` 新增 `assess()` 方法, 输出五种认知状态:
- **KNOWN**: Type A, 集体数据充分覆盖 → 直接回答
- **PARTIAL**: Type B, 宏观有信号但个体变量缺 → 提供地形, 标注边界
- **UNKNOWN**: Type C, 集体数据无信号 → 说不知道
- **AMBIGUOUS**: 多节点竞争, gap<0.03 → 拒绝选边
- **OOD**: ≥2个指标极端 → 模型可能失效, 历史规律不可外推

分界方式: 不用硬阈值 (0.45 vs 0.50 vs 0.xx), 用 project() 的 confidence 三级
(HIGH/MEDIUM/LOW) 映射到五态. 避免假装有精确分界线. 阈值需外部验证数据校准.
**请外部 AI 评审**: 这种映射方式是否合理? confidence 的判定条件
(max_sim>0.55+gap>0.06→HIGH, max_sim>0.40+gap>0.03→MEDIUM) 是否需要调整?

### Stella 集成修复

Skill 放到了正确位置 (`workspace/skills/fr31-advisor/`, 项目专属).
重启 gateway 后验证通过: 用户问 "现在互联网叙事什么状态" →
Stella 读 STATE.md → 白话回复.

### 第五轮外部 AI 审查共识 (2026-07-05)

**双方一致**:
1. MS-AR 是 Dynamics 层最高优先级 (已有 regime, N=127 刚好够)
2. Counterfactual 结论有效, 但平台需拆为三类算子 (快扩散/慢沉淀/模因复制)
3. Graph λ₂ 需等边稳定后再做, 当前人工边不可靠
4. 五态模型的 confidence 映射方式合理, 避免硬阈值

**GPT 新增建议**:
- 新增第六态: LATENT STRUCTURAL DOMAIN (弱集体信号但个体结构强, 聚合会破坏信号)
- R2 应视为 hysteresis basin (滞回势垒), 非普通 cluster
- 平台 = mixture of dynamical operators, 不是 covariate
- 逃逸需三条件同时满足: z(t)方向反转 + 平台权重重排 + 语义记忆失活

**Gemini 新增建议**:
- platform 内部拆解: 短视频(势阱加深剂) vs 图文问答(流形边界定义者)
- 二阶 counterfactual: 分别抹平 [抖音,快手] 和 [知乎,小红书], 看谁维持锁定 vs 谁定义边界
- 伪 KNOWN 态风险: 宏观高饱和度叙事可能是虚假意识, 需加生存周期校验

## 核心困惑 (更新后)

### Q1: 平台内部结构 → 二阶 counterfactual 待做

16.5x 不可能均匀分摊。需拆为三类算子:
- 短视频(抖音/快手): 快扩散, 势阱加深剂 → 维持 R2 锁定
- 图文问答(知乎/B站): 慢沉淀, 流形边界定义者 → 定义相区边界
- 社区模因(小红书/微博): 结构记忆, 影响不可逆性
二阶 counterfactual: 分别抹平 [抖音,快手] 和 [知乎,小红书], 看效应分离.

### Q2: R2 逃逸条件 → 升级为三条件

GPT: R2 是 hysteresis basin (滞回势垒), 非普通 cluster.
逃逸需同时满足: z(t)方向反转 + 平台权重重排 + 语义记忆失活.
单靠 u(t) 逆转不够.

### Q3: Dynamics → MS-AR 优先 (双方共识)

MS-AR → GARCH(残差) → Fokker-Planck(远期).
有了 regime + transition matrix, 直接升级为 Markov operator with z(t) modulation.

### Q4: 边界 → 新增 LATENT STRUCTURAL DOMAIN? (待定)

GPT 建议第六态: 集体信号极弱但个体结构强且稳定,
聚合会破坏信号 (如"表白无效性"). 不是 UNKNOWN — 是 aggregation-anti-invariant.
是否需要加入五态模型?

### Q5: Graph λ₂ → 先冻结 edge rule v1.0

等边定义稳定 ≥3 regime cycles + platform dominance corrected 后再做.
当前人工边不可靠.

### Q9: regime 的物理含义 — 结构崩塌还是共识收敛? (NEW, 2026-07-06)

**当前预设**: 系统有"相区" — R2 是 fixation lock，系统被外部约束力困在势阱中。
这个预设藏在 "regime""势阱""逃逸""分岔" 这些词里。

**替代假说**: Q8 说测量对象是人群的叙事化反应，也就是一种**集体应对策略**。
如果测量对象本身就是应对策略，那"fixation lock"完全可能是另一种东西：
**不是系统被锁死了，是百万个体各自独立抵达了同一个应对策略**。
不是被困住了，是都选择了这个解。

**这两者看起来完全一样，但物理含义相反**:

| | 结构崩塌 (当前预设) | 共识收敛 (替代假说) |
|------|------|------|
| 机制 | 外部约束力把系统按在盆地里 | 大量个体独立收敛到同一应对策略 |
| R2 自持 97.3% | 系统被卡住了 | 当前共识是最优解，没人想走 |
| 方差放大 2.31× | 势阱底部变平，越抖越厉害 | 共识内部的表达方式在多样化 — "躺平"有 20 种说法，但都是躺平 |
| 逃逸条件 | 需要外部冲击打破约束场 | 需要现实结构变化让旧共识失效 |
| 预测 | 临界点 → 灾变性相变 | 不会炸，会慢慢淡出，被新共识替代 |
| 物理语言 | hysteresis, potential well, critical slowing | self-organized consensus, strategy convergence |

**目前数据分辨不了这两者。** 需要什么:

1. **跨平台叙事姿态差异** (scraper 已有): 结构崩塌 → 不同平台趋同。共识收敛 → 不同平台用各自模式表达同一共识。
2. **共识内部创新率** (scraper 已有): 结构崩塌 → 语义空间冻结。共识收敛 → 持续创新但不改变底层结构。2.31× 方差放大可能已经是共识收敛的信号。
3. **冲击响应** (依赖历史事件): 冲击后系统弹回原状（崩塌）vs 共识被事件本身改变（收敛）。
4. **微观逃逸尝试** (目前无数据): 有没有个体/社群在尝试打破主流叙事？需要 tracking 小社群数据。

**请外部 AI 判断**:

1. "结构崩塌"vs"共识收敛"这个二分是否完整？有没有第三种物理图景？还是说它们是一个连续谱的两端？

2. 这两种物理在统计上是否可区分？已知数据（10 维 x(t), regime labels, 日级 embedding）是否够？
   还是需要本质上不同类型的数据？

3. 如果区分不了，整个项目的物理语言（势阱/分岔/逃逸/hysteresis）是否应该被替换为
   更中性的描述（"叙事生态的元稳定性""集体策略协调"）？
   还是像 GPT 之前警告的那样——物理概念需要先有可观测的统计量，再给物理解释，
   不是先给物理名字再用数据合理化？

4. 这个"共识收敛"视角是否串联了之前所有矛盾？
   - 叙事化反应不是噪声，是策略
   - R1 和 R3 连通 — 同一策略空间内的位移
   - R2 锁死 — 策略空间收敛到少数解
   - 方差放大 — 同一策略内部的表达创新
   - H1b REJECTED — 策略选择的随机性主导

5. 架构应该如何重建？如果"共识收敛"是对的，那整个项目的核心对象
   从"集体情感的物理状态"变成了"集体认知的策略分布"。
   这不是修修补补——是重新定义项目在测量什么。

### Q8: 信号质量危机 + 认知螺旋 (2026-07-06)

**第一层: 信号质量危机 — 我们采集的到底是什么?**

整个 FR19 框架的隐含前提是:

```
互联网叙事活动 —映射→ 集体情感状态 —反映→ 现实结构
```

但这个前提的第一个箭头就是错的:

```
互联网叙事活动 = 对现实结构的叙事化反应
                 + 社会环境预设
                 + 情绪化发泄
                 + 道德直觉判断
                 + 犬儒化消解
                 + 转移注意力
                 + ...
                 ≠ 对现实结构的观测
```

关键不在于"有噪声"。关键在于是**系统性的**: 
人群在面对现实结构时, 几乎总是产生叙事化反应而非结构分析。

举例: 一个人在现实中反复经历"表白 → 负收益", 但他在网上的输出不是
"我发现了表白不对称负收益的结构规律", 而是情绪发泄、造梗玩自嘲、
转发段子、站队互骂、转移注意力。一百万人都是这样。

不是"聚合破坏了信号"——是**原材料就已经不是信号了**。

**第二层: 这个"问题"恰恰就是要测量的对象**

上述分析把"叙事化污染"框定为一个需要在提取特征之前被清洗的问题。
但这个框可能是错的。

那个让一百万人面对同一堵墙却不分析墙、而是发泄、造梗、站队的东西 —
那种**集体层面信息处理的内在倾向** — 恰恰就是项目试图建模的
"集体层面小真实内在的混沌属性"。

叙事化反应不是观测噪声。
它是**观测目标**。
你不能在观测之前把它清洗掉, 因为你在试图观测的就是它。

**第三层: 认知螺旋 — 结论必须反向注入数据提取**

区分"接近结构分析的讨论"和"纯叙事消费"的标准,
不能也不应该被预设。这个标准本身正是要从数据中发现的。

这意味着这不是一个"先清洗数据再建模"的线性流程。它是一个螺旋:

```
粗糙数据 → 粗糙分析 → 初步理解"这类讨论偏向X, 那类偏向Y"
    → 这种理解反向注入数据提取层
        → 更精细的数据切分 → 更精确的分析
            → ...
```

项目最终要产出的结论之一 — "集体叙事活动在什么条件下更接近/更远离现实结构" — 
会同时成为项目自身数据管道的**输入参数**。

**请外部 AI 判断**:

1. 这个"认知螺旋"在方法论上是否成立? 有没有类似学科中的先例
   (比如天体物理中 instrumental calibration 和目标天体物理参数
   的迭代解耦)?

2. 把"叙事化污染"从"要清洗的噪声"重新框定为"要测量的对象",
   是否改变了 Q7 中数据管道接入的设计? 
   日级 embedding → 月级聚合时, 是否需要保留能区分
   "叙事消费型讨论"和"结构观察型讨论"的分布特征?

3. 认知螺旋的启动条件: 第一轮粗糙分析的输入应该是什么?
   在没有人工标注的前提下, 什么信号可以作为"叙事化程度"的
   初始 proxy?

4. 如果这个螺旋收敛, 最终产出的不只是"集体情感相图",
   而是**两样东西**: (a) 叙事活动的动力学规律 +
   (b) 叙事活动和现实结构之间映射关系的时变估计。
   这两者是否应该作为项目的正式双重产出?

### Q7: 数据审计 — scraper 从未进入分析链 (NEW, 2026-07-05)

**发现的事实**:

```
Scraper (每小时, 2026-06至今):
  → 226 次采集, ~22,000 条原始 headline
  → 关键词匹配: 命中 29 条 (0.13%)
  → 99.87% 的数据被丢弃
  → 知乎数据源: 一直返回 0 条 (cookie 过期, 未被发现)

这些数据进入了什么分析?
  → 没有。
  → Level 1/2 特征工程: 没用
  → Regime 检测: 没用
  → RQA: 没用
  → Control Manifold: 没用
  → FR31 指标: 没用
  → MS-AR: 没用
```

Scraper 的唯一 "下游消费" 是 Dashboard 上一个 widget — "当前月混沌轴 +0.229" —
基于 2 个梗 8 条关键词信号计算。这条数据本身就是垃圾，而它是 scraper 数据的唯一出口。

**这意味着**: 所有 127 个月的 "科学结论" (H1/Regime/RQA/ControlManifold/FR31/MS-AR)
都是基于两块静态历史数据做的 — Google Trends (2015-2025) + Narrative JSON (57 条)。
Scraper 这整条管线 — 226 个文件, 22,000 条 headline, 每小时运行 — 在分析层面等于不存在。

这是一个项目级的盲点，不是 bug。我几个月来一直在说"我们有三数据源、实时采集、每小时 130 条"，
但从来没追问过"这些数据进了哪个分析模块"。

**已完成的修复** (v2.0):
- 全量 headline embedding (384维 paraphrase-multilingual-MiniLM-L12-v2)
- 日级聚合: 梗相似度分布 + 注意力集中度 + 语义新颖度 + 日平均 embedding
- 知乎数据源修复 (cookie 更新)
- 存储: 日 ~4MB, 月 ~120MB, hourly 7天自动清理

**待解决的真正问题**: 日级语义状态如何接入月度分析层? 三条路径:

1. **日→月聚合 (最低成本)**: 每日 embedding 按月取均值 → `monthly_semantic_state.json`,
   作为 Level 1 的新增特征维度。直接接续 Google Trends 的时间序列 (目前断在 2025-12)。

2. **替换 Google Trends 作为实时 proxy**: Google Trends 有 3-7 天滞后且只能查已知关键词。
   Scraper embedding 没有滞后、不需要预设关键词。月级聚合后可以同时提供"量"
   (headline 中梗的语义密度) 和"质" (梗在说什么, 语义是否漂移)。

3. **Feed into FR31 Sensitivity 微观修正**: `micro_burst_detector.py` 已设计框架但输入
   是 scraper 关键词命中 (≈零)。换成日级 embedding 的 `attention_concentration` + `novelty`。

**请外部 AI 判断**:
1. 三条路径的优先级? 是否应该先做 1 (日→月聚合, 低成本验证), 再做 2/3?
2. 日级 embedding (384维) 聚合到月级, 信息损失多大? 是否需要保留分布信息 (不只均值, 加方差/分位数)?
3. 当前 127 个月的历史分析用的是 Google Trends 月搜索量作为"注意力"的 proxy。
   用 headline 语义密度替代它, 是补充还是替换? 两者是否应该共存?
4. 现在 scraper 有 130条/小时 × 24小时 = ~3000 headline/天。这么大的量,
   除了月级聚合, 还有什么分析是只有高频数据才能做的?

### Q6: MS-AR 第一刀结果 → Phase 2 方向确认

**做了什么**: 准备做 `P(S'|S, z)` — z(t) 条件 regime 转移。但数据现实暴露了问题:
- z1 (AI/Tech 轴) 在 R0/R1/R3 内方差近乎为零 — 只在 R2 内部有实质性变化
- 127 月仅 14 次切换 — 拟合连续 MS-AR 统计效力严重不足

**调整后的第一刀**: 不做连续拟合，做四件事:
1. **时间切分转移矩阵** (pre/post 2018-01, pre/post 2020-01, pre/post R2 entry)
2. **Bootstrap 置信区间** — 每个转移概率的 95% CI
3. **z1 置换检验** — 切换事件是否在特定 z1 聚集
4. **R2 内部漂移分析** — x(t) 是否随 z1 在 R2 内共变 (利用全部 38 个 R2 月, 不需切换)

**结果**:
- 时间切分: 2018-01 切分显著 (p=0.029), 但可能仅是 regime 分布时间差异, 非转移结构本身不同
- z1 置换: 切换不聚集在特定 z1 (p=0.713) — 预期之中, 14 次切换效力太低
- **R2 内部**: PC4 r=+0.85, PC5 r=-0.87 (p≈0.000), Bonferroni 校正后仍显著.
  状态方差从 early R2 到 late R2 放大 **2.31 倍**.
  z1 强烈调制 R2 内部状态配置.
- 总漂移与随机游走一致 (ratio=0.76) — 不是定向走, 是在同一个坑里越抖越厉害

**综合判断**: 检测到 2/3 信号 → `PROCEED_TO_PHASE_2`.
**但 Phase 2 方向需修正**: 放弃 `P(S'|S, z)` (没切换可拟合).
改为 **`p(x | R2, z1)`** — R2 内部 Narrative State 的条件分布, 建模
"在 fixation 势阱中, 系统在哪些状态配置间震荡, 以及这如何随 AI/Tech 话语密度变化".

**与 FR31 指标的呼应**: R2 内部方差放大 2.31× 与 Sensitivity=0.56 (Critical Slowing=0.76)
来自独立计算路径 — 一个是 regime 内部方差比, 一个是全历史方差×自相关.
两者都指向"系统在势阱内越来越动荡". 这是最强的信号.

**请外部 AI 判断**:
1. `p(x|R2, z1)` 方向是否合理? 统计上 (R2 内 N=38, 10 维, z1 1 维) 是否有更好的建模方式?
2. 2.31× 方差放大 + Critical Slowing 0.76 — 这两个信号的物理含义是否一致? 有没有可能是同一现象的两种投影?
3. 14 次切换是否真的不够做 MS-AR? 有没有我们遗漏的低样本方法 (如 Bayesian MS-AR with informative priors)?
4. 如果接受"当前无法做 regime 切换建模", 是否应该转成纯描述性方向: 把 R2 内部的 z1 调制效应可视化, 作为"势阱形变图"而非预测模型?

## 技术栈

Python 3.12 + numpy/scipy/pandas/scikit-learn/sentence-transformers + Flask + ECharts
LLM: DeepSeek API
Agent: OpenClaw + 企业微信 (mote-home, Ubuntu 24.04)
Dashboard: chaos.mote-pal.xyz (cloudflared tunnel + token auth)

# MemeticChaos — 人类集体情感混沌属性建模

> 最后更新：2026-07-09

## ⚠ 架构推翻史

本项目的核心架构已被**多次推翻**：
- 混沌轴 + 约束场 + 秩序形态预测 (v1-v3) → 2026-07-04 审计推翻
- H1 点预测范式 → H1b REJECTED 推翻
- AI/Tech 轴主导 → Counterfactual (平台 16.5×) 推翻
- 两层架构 → 四层形式化推翻
- MS-AR P(S'|S,z) → 14 次切换不足推翻
- "三数据源实时采集" → scraper 从未进入分析链推翻
- "测量集体情感状态" → 测量的从来是叙事化反应推翻
- "R2 = 结构崩塌" → 可能是共识收敛推翻 (2026-07-06)

**当前架构中几乎肯定仍有未被识别的预设。** 每个新会话应该用怀疑的眼光审视当前架构，
而非把它当作给定的前提。

### 方法论原则：猎杀未标注的预设

**未被意识的预设 = 绝对秩序。** 它不觉得自己是预设，就变成"显然""自然""大家都这么认为"。
所有后续推理建在它上面，没人再回头看它。

每引入一个概念，必须标注它是哪一类：
- **数学工具** — PCA、GMM、cosine similarity。使用但不声称物理解释。
- **观测描述** — "R2 内部方差放大 2.31×"。数据在说什么，不加解释。
- **物理声称** — "R2 是 hysteresis basin""系统接近分岔点"。必须有独立证据，且随时可被推翻。

前两类可以留。第三类必须等到独立证据——不能同一组数据既用来提出声称又用来验证声称。
如果分不清某概念属于哪一类，它很可能就是未被意识的预设。

**新增原则：本体离散化才是危险预设 (2026-07-09, 第八轮外部 AI 修正)。**
最初表述"离散化即预设"太宽 —— 会把所有统计工作打成预设污染。GPT 切三层, 只有第三层危险:
- **操作离散化** (月/日/PCA10/GMM4): 只为计算, 换算法就变, **无本体承诺** → 安全。
- **分析离散化** (pre2020/2020-2022/2023+ 时段切分): 是**实验设计** ("若真有变化这样切可能敏感"),
  不是世界模型 → 安全, 但必须写进 Experimental Design 而非 Theory。
- **本体离散化** (五阶段 / "情绪 vs 策略 vs 协议"): 开始宣称"世界本来就这样组成" → **危险**。
引入任何切分/分类时问: "这个边界是**数据给的**(前两层, 标注归属即可), 还是我**替世界宣称了本体**(第三层)?"

**最刺的推论仍成立**: 用来猎杀预设的实验工具, 自身通过设计选择带着**分析离散化** (时段审计硬编码"三段")。
只要诚实归类为实验设计、不偷渡成理论, 就不算污染。**没有绝对的内部出路** (审计要找边界就得先假设边界范式) ——
出路是让预设**显式化 + 独立通道交叉验证** (Bayesian prior sensitivity / 多 DAG 求 invariant / 气候 CMIP 多模型
一致性 / 天体物理 multi-messenger)。现代科学不消灭预设, 只让它显式。

**切当前架构的结果 (2026-07-06)**:

**五个冒充观测的物理声称** (已修正为描述性语言):
- "势阱" → high persistence region
- "分岔" → autocorrelation increase + variance inflation
- "逃逸" → transition (从未观测到)
- "吸引子" → persistent occupancy region
- "metastable" → long residence with rising fluctuation

**七个概念层预设** (待检验):
- "五阶段"分类 — 人工阈值划分, 和数据驱动原则矛盾
- "叙事"本体论 — Scraper/Trends/Narrative JSON/B站从未被区分
- u(t)/x(t)边界 — 建模选择, 不是经验发现
- 月度聚合 — Google Trends 粒度决定, 不是系统的自然时间尺度
- Markov 假设 — 非线性情况从未被检验
- "梗"作为同质单元 — "躺平"(3年)和"鸡你太美"(2个月)被同等对待
- 单梗阶段划分 — 阈值法, 和已杀掉的混沌轴同构

**三个架构级预设** (最深层的未被意识的前提):
1. "叙事活动是一个可被降维描述的物理系统" — PCA 发现的结构可能是叙事反应模式的结构, 不是现实结构
2. "存在时不变的生成机制" — 如果互联网底层逻辑变了, 2015和2025是两个不同系统
   → **已审计 (2026-07-09): VERDICT=UNDERPOWERED。措辞升级 (GPT): 不是"不可检验", 是
   `resolution-dependent identifiability` —— "at monthly resolution: not identifiable"。
   以后日级/小时级若给不同答案, 不是理论变了, 是观测算子变了。**
3. "宏观可由微观聚合" — 聚合反因性可能系统性缺失了最重要的结构
4. **"存在唯一正确的物理图景" (2026-07-09, GPT 发现, 我漏掉的更深预设)** —
   一直在问 R2 到底是 Structure-Collapse / Consensus-Convergence / Communication-Protocol / Narrative-Damper
   "哪一个"。隐含"只有一个是真的"。但它们可能是**不同统计量对应的不同投影, 未必互斥**:
   RQA→geometry, embedding→communication protocol, Trends→attention ecology, Persona→strategy。
   → **对策: Competing Explanatory Layer** —— 观察不立即升级成唯一解释; 多个候选解释并行存活,
   直到新统计量杀掉其中几个。⚠ Gemini 的"彻底纠正为 ESS 分布"正是此预设的现场犯案 (未验证就抢占唯一答案位)。

**元预设 (2026-07-09)**: "本体离散化才是危险预设" (见方法论原则)。第三层离散化偷渡本体; 前两层安全。

**研究节奏风险 (2026-07-09, GPT 警告 — 非技术)**: 项目持续产出有解释力的新概念
(Constraint Field / Hysteresis Basin / Narrative Damper / Communication Protocol / Consensus Convergence /
Semantic Smuggling / 三层离散化), 而**验证工作被持续后移**。必须分账:
- **方法论资产** (长期保留, 除非内部矛盾): Evidence Ledger、Semantic Smuggling、三层离散化、Competing Explanatory Layer。
- **解释性假说** (竞争并存, 待证据支持或淘汰): ESS/策略分布、通信协议收敛、叙事阻尼器、结构崩塌、共识收敛。
下一步优先做**验证/基础设施**, 不再产新概念。

## Evidence Ledger — 证据分级账本 (2026-07-09, 第八轮外部 AI 共识, 建设中)

GPT+Gemini 一致收敛的方法论基础设施。把"四类标注"升级为**带等级 + 假设依赖**的账本:
当某个底层假设倒了, **只有挂靠它的高等级条目自动失效, 低等级观测继续保留**——项目不会整盘崩。

**等级** (E0→E4, 越高越依赖解释):
- **E0 原始观测** — headline 文本、Trends 数值。不加任何加工。
- **E1 统计描述** — "R2 自持 97.3%""方差放大 2.31×""子空间夹角 40.9°"。数据在说什么。
- **E2 操作结果** — "PCA d90=10""平台 16.5×""GMM 4 簇"。依赖算法/参数选择, 换工具可变。
- **E3 条件解释** — "R2 是真实 cluster (RQA 零复发)"。有条件的推断, 依赖具体假设。
- **E4 机制假说** — "R2 是 hysteresis basin""R2 是 consensus convergence"。竞争并存, 待证据淘汰。

**Assumption 列** (每条目挂靠的底层假设): Time-Invariance / Aggregation / Sampling / Stationarity / ...
→ 예: 所有 E3/E4 若挂 `Time-Invariance`, 则该假设一旦被证伪, 这些条目自动降级/失效; E0/E1/E2 不受影响。

**产出**: `data/processed/evidence_ledger.json` + 生成脚本。每个结论一行, 带 grade + assumptions + source。
这同时是**假设生命周期**的载体 (一个假说何时出生/升级/降级/退休 —— 目前全靠脑子记, GPT 指出的空缺)。

## 项目目标

利用 2015-2025 年中国互联网热梗数据，刻画互联网集体叙事系统的相变结构。

- **Hassabis 猜想 × AlphaGo 范式**：自然界中存在的模式可被经典学习算法发现。不灌输人类总结的套路，让数据自己说话。
- **四层形式化**：State（叙事状态）/ Observation（观测）/ Control（控制流形）/ Dynamics（动力学）。详见 `PROJECT_STATUS.md`（现状 + 承重路径 + 审计方法论）。
- **H1b REJECTED**：月度尺度叙事状态是近似随机游走。从"预测值"转为**刻画相区结构 + 检测分岔临界**。
- **核心产出**：集体叙事相图 + 四指标实时诊断 (Inertia/Resilience/Sensitivity/Position)。

## 技术栈

Python 3.12 + conda `MemeticChaos` + numpy/scipy/pandas/scikit-learn/sentence-transformers + Flask + ECharts
仓库：git@github.com:Mote-Xu/MemeticChaos.git

## 数据资产（截至 2026-07-06）

```
外部层:      51 关键词 × 132 月 Google Trends (2015-2025)
内部层:      57 条叙事档案 (22 B站 + 36 曲线, 含 spread_phases/mutations/semantic_drift)
LLM 概念分数: 57 条叙事 × 35 维
实时采集 v2.0: 微博50 + 百度50 + 知乎30 = 130条/小时 → 全量 384 维 embedding
日聚合:       每天 23:57 → 梗语义相似度 + 注意力集中度 + 新颖度
```

数据文件：
- `data/collector/external_field_2015_2025.json` — 51 关键词外部场 (Google Trends)
- `data/collector/google_trends_2015_2025.json` — 43 梗注意力曲线
- `data/processed/narratives/` — 22 个 B站 LLM 叙事抽取
- `data/processed/narratives_from_trends/` — 36 个 Trends 曲线生成叙事
- `data/processed/llm_concept_scores.json` — 57 条 LLM 概念分数
- `data/processed/level1_hard_facts.json` — ★ Level 1: 127 月 × 4 硬事实 (Stage/Mutation/Inst/Drift)
- `data/processed/representation_state.json` — ★ Level 2: 127 月 × 10 维 Narrative State x(t)
- `data/processed/regime_map.json` — ★ v4.1: 4 相区 + 转移矩阵 + 驻留时间
- `data/processed/control_manifold.json` — ★ v4.1: 3 维控制轴 z(t)
- `data/processed/irreversibility_results.json` — ★ v4.1: RQA + Time-Reversal 结果
- `data/scraped/hourly/` — 每小时 headline + 384 维 embedding (永久保留, 不可再生高分辨率资产)
- `data/scraped/daily/` — 日级语义聚合: 梗相似度 + 注意力集中度 + 新颖度 (永久)

## 项目结构

```
src/
├── data/
│   ├── scraper.py                     # 实时采集 v2.0 (微博+百度+知乎, 每小时, 全量 embedding)
│   ├── live_pipeline.py               # 采集→更新→报告 (每天)
│   ├── signal_pipeline.py             # 新梗发现→LLM叙事→概念打分 (每天)
│   ├── monthly_narrative.py           # LLM 月度叙事摘要 (每周)
│   ├── narrative_extractor.py         # B站视频叙事抽取 (22/22)
│   ├── narrative_from_trends.py       # Google Trends 曲线→叙事
│   ├── narrative_hard_facts.py         # ★ Level 1 硬事实提取
│   └── trends_loader.py / trends_weekly.py / curator.py / rebuild_from_trends.py
├── constraint/
│   ├── llm_concept_scorer.py          # LLM 35概念打分
│   ├── concept_bottleneck.py          # 软匹配概念瓶颈 (弃用)
│   └── delta_transition.py            # ΔTransition + 3Validator
├── trajectory/
│   ├── meme_trajectory.py             # 29条轨迹 (Schema 2.1)
│   └── trajectory_viz.py              # 轨迹相图
├── models/
│   ├── representation_learning.py     # ★ Level 2: PCA 表示学习 + H1 验证
│   ├── order_form_predictor.py        # FR19 v0.2 (v4.0 已退役, 保留参考)
│   ├── collective_predictor.py        # 旧版预测器 (保留参考)
│   └── sir_meme.py / abm_simulation.py / attractor.py / individual_calibrator.py
├── analysis/
│   ├── regime_detector.py             # ★ v4.1: GMM 4 相区 + 转移矩阵
│   ├── irreversibility_test.py        # ★ v4.1: RQA + Time-Reversal
│   ├── control_manifold.py            # ★ v4.1: u(t)→z(t) 控制轴分析
│   ├── ms_ar_first_cut.py             # ★ MS-AR 第一刀: z(t) 调控 regime 转移
│   ├── collective_dynamics.py         # 集体情感系统动力学 (133月)
│   ├── platform_flow.py               # 跨平台注意力流动
│   ├── narrative_clustering.py        # 叙事无监督聚类 (ARI=0.27)
│   └── phase_diagram.py / backtest.py / lifecycle.py / phase_detect.py
├── dashboard/
│   ├── app.py                         # Flask API (9端点, token auth)
│   ├── analyzer.py                    # 精细建模查询
│   └── templates/index.html           # ECharts 前端 (回放/拖拽/倍速/移动端)
├── advisor/
│   ├── engine.py                      # ★ FR31 统一引擎 (macro/persona/full 查询)
│   ├── persona.py                     # ★ 个体画像: P(meme|text) 五态输出
│   └── metrics.py                     # ★ FR31 四指标: Inertia/Resilience/Sensitivity/Position
└── meme_inspector.py
tests/
├── test_sir_model.py                  # 24/24
└── test_system_integrity.py           # 38/38
```

## FR19: Narrative Dynamics (v4.1)

### 核心转变

FR19 不再是预测器。它是关于互联网集体叙事动力学的**可检验科学假说**。

### H1 假说链 — 验证结果

| 假说 | 结论 | 证据 |
|------|:--:|------|
| **H1a** (低维表示) | ✅ SUPPORTED | 18 维 → d90=10, d95=12 |
| **H1b** (动力学连续性) | ❌ REJECTED | VARX test R²=-0.32 < lag-1 R²=+0.44 |
| **H1c** (当前状态主导) | ⚠️ MIXED | State-only R²=+0.14, 外生变量不加帮助 |

**核心发现**: 月度尺度集体叙事动态由随机漂移主导。预测值不是正确的目标——应预测转移概率、结构属性、相变。

### Regime Map (v4.1)

GMM 对 10 维 x(t) 聚类, BIC 选出 4 个观测簇, 合并为 3 个物理相区:

| 物理相区 | 包含 | 月数 | 自持概率 | 时期 |
|------|:--:|:--:|:--:|------|
| **Origin** | R1 + R3 | 85 | 87% | 2015-2021 健康涌现 |
| **Fixation** | R2 | 38 | **97.3%** | 2022.12→今 僵化锁死 |
| **Peak** | R0 | 4 | 50% | 罕见全网爆发 |

- **R2 当前驻留 37 月** (中位 19 月, 已超 2×)
- RQA 确认 R2 为真实结构分离 (零跨相区复发), 非 GMM artifact
- Time-Reversal 对称: 不可逆性来自 Control 层漂移, 非系统内禀
- 转移矩阵趋近三角: R3→R1→R2, 有明确时间箭头

### 动力学方程

```
x(t+1) = F(x(t), u(t), y(t))
```
x = Narrative State（10 维），u = External Field（51 维），y = Attention（反馈）

### Control Manifold

51 维外部场 → PCA → Diffusion Map → 3 维控制轴 z(t)
- z₁: AI/Tech 话语轴, 十年单向漂移
- Counterfactual test: 平台 (16.5×) > AI (1.9×) — 平台生态是 primary control driver

### 模型性能

| 指标 | 值 | 注意 |
|------|-----|------|
| Structure 预测 R² | 0.79 | HHI+叙事熵可从外部场预测 |
| 叙事聚类 ARI | 0.27 | 对 35 维概念向量聚类：机器见 3 类，非人工 5 类 |
| 内部叙事层贡献 | 0 | 叙事被偷换成约束向量，从未真正参与预测 |

### MS-AR 第一刀 (2026-07-05)

**约束**: 127 月仅 14 次 regime 切换, z1 在 R0/R1/R3 内方差近乎零。

**结果**:
- R2 内部 PC4 r=+0.85, PC5 r=-0.87 (p≈0), 方差放大 2.31×
- z(t) 调控系统的方式不是触发切换，是在 regime 内部形变状态分布
- **判断**: `PROCEED_TO_PHASE_2` — 放弃 P(S'|S,z), 做 p(x|R2, z1)

### 已知局限 (2026-07-04 全面审计)

**结构性问题**:
- 约束场是静态标签: 每个梗只有一个向量, 同一梗爆发期 vs 消退期用同一约束值
- 内部层被偷换概念: 本应使用叙事结构信息, 实际压缩成了约束向量
- 类别应允许多标签: 当前互斥分类有信息损失

**实证结论 (非 bug)**:
- 注意力结构可预测（R²=0.79）
- 叙事聚类：机器见 3 类，非人工 5 类（ARI=0.27）
- 五类别是脚手架，归类基于叙事特征有逻辑支撑问题

## FR31: 情感约束场顾问

### 定位

不是替用户写回复的 AI。是**战略对弈伙伴**——用户在具体情境中给出自己的判断，
系统给出独立判断，两者对质，直到找到比独自判断更好的方案。

FR19（集体混沌属性建模）是 FR31 的推理引擎。

### 已建成

- `engine.py`: 统一查询引擎, macro/persona/full 三种模式
- `persona.py`: 个体文本→叙事图投影, P(meme|text) 概率分布, 五态输出
- `metrics.py`: 四指标 — Inertia(惯性)/Resilience(恢复力)/Sensitivity(敏感性)/Position(图位置)
- Stella 企微管道: 每小时刷新 STATE.md → Stella 读文件 → 白话诊断

### 四指标当前值 (2025-12)

| 指标 | 值 | 物理含义 |
|------|:--:|------|
| **Inertia** | 0.77 | 势阱深度 — fixation 占 52% |
| **Resilience** | 0.37 | 恢复力 — 2025H2 归零, 偏离均衡 3.9σ |
| **Sensitivity** | 0.56 ↑ | Critical Slowing=0.76, 分岔前预警 |
| **Position** | R2 边缘 | 近 12 月 0 次阶段转换 |

### 五态输出模型

| 状态 | 含义 |
|------|------|
| **KNOWN** | Type A, 集体数据充分覆盖 |
| **PARTIAL** | Type B, 宏观有信号但个体变量缺 |
| **UNKNOWN** | Type C, 完全不可触达 |
| **AMBIGUOUS** | 多节点竞争, gap<0.03 |
| **OOD** | 模型可能失效, 历史规律不可外推 |

### 知识底座

> ⚠️ AlphaGo 原则：体系知识写入 .md 作为背景参考，但**不硬编码进模型**。
> 代码中唯一可引用的结构划分来自真实主权——小真实的内部结构
> （感知→认知→体验），用于建模对方偏好。其余不进代码。

## 服务器部署 (mote-home)

> **服务器是项目主运行环境。本地仅用于查看结果和精细建模。**

- **位置**: `/mnt/data/MemeticChaos/` → symlink `~/MemeticChaos/`
- **Python**: `~/miniconda3/envs/MemeticChaos/bin/python` (PYTHONNOUSERSITE=1)
- **cron 任务**:
  - 每小时 17 分: 采集 + 全量 embedding (`scraper.py` v2.0)
  - 每小时 :20: Stella 状态刷新 (`refresh_stella_state.py`)
  - 每天 23:57: 日级语义聚合 (`scraper.py --aggregate`)
  - 每天 2:37: B站时间序列 (`bilibili_timeseries.py`)
  - 每天 4:47: 实时管线更新 (`live_pipeline.py` + `signal_pipeline.py`)
  - 每周日 4:50: LLM 月度叙事摘要 (`monthly_narrative.py`)
  - 每周日 5:00: 12月预测报告 + Dashboard 状态刷新 (`order_form_predictor.py`)
  - 每月 1 号 5:32: 月度语义聚合 + 前向序列 (`monthly_aggregator.py` + `semantic_state_series.py`) → `cron_monthly.log`
- **hourly 保留**: 永久 (`HOURLY_RETENTION_DAYS=0` 哨兵, `_cleanup_hourly` 跳过)。逐条 embedding 不可再生, 且月度聚合从 hourly 读, 故不清理。
- **Dashboard**: https://chaos.mote-pal.xyz/?token=<TOKEN> (Flask :8931 + cloudflared tunnel)
- **systemd**: `memeticchaos-dashboard.service` (enabled, auto-restart)
- **数据同步到本地**: `bash sync_from_server.sh`
- **数据同步到服务器**: `bash sync_to_server.sh`
- **本地任务**: Google Trends 通过飞鸟代理每日拉取 + 叙事生成

> **⚠ 服务器运维要点 (2026-07-09 踩坑)**:
> - **代码走 scp/rsync 不走 git**: 服务器 `~/MemeticChaos` 的 git 是 unborn master (零提交、无 remote)。
>   `git@github.com:Mote-Xu/MemeticChaos.git` 只是本地→GitHub, **push 到 GitHub 到不了服务器**。
>   部署新脚本用 `scp src/xxx.py mote:~/MemeticChaos/src/xxx/` 或 `sync_to_server.sh`。
> - **非交互 ssh 无 conda**: 用完整路径 `PYTHONNOUSERSITE=1 ~/miniconda3/envs/MemeticChaos/bin/python`。
> - **embedding 任务必须 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`**: 否则 sentence-transformers
>   联网查 HF 更新会卡死 (模型已缓存)。加了之后 6060 条 <45s; 不加会 400s+ 超时。
> - **长任务用 `nohup ... > /tmp/x.log 2>&1 &`** 服务器端脱离, 避免 ssh 超时中断。

## 已完成

- 24/24 测试 + 38/38 系统完整性验证
- FNN 标准 Kennel / 连续 R₀ / S₀ 参数 / R₀ 扰动鲁棒性 / 盲聚类
- 数据自动补充：服务器 24/7 + 本地 Google Trends
- Dashboard: ECharts + Flask + token auth + 历史回放 + 精细建模查询
- 实时信号管线: 新梗自动发现 → LLM叙事 → 概念打分 → 入库
- 跨平台流动框架 + 叙事无监督聚类
- 三数据源: 微博+百度+知乎
- Level 1 硬事实提取 — Stage/Mutation/Institutionalized/SemanticDrift 月度序列 (2026-07-04)
- Level 2 表示学习 — PCA d90=10, H1b REJECTED (随机游走), H1a SUPPORTED (低维) (2026-07-04)
- FR31 四指标接口 — Inertia/Resilience/Position/Sensitivity 可查询 (2026-07-04)
- v4.1 四层形式化 — State/Observation/Control/Dynamics (2026-07-05)
- Regime Map — GMM 4 观测簇 → 3 物理相区 + 转移矩阵 (2026-07-05)
- RQA + Time-Reversal — R2 真实分离确认 (2026-07-05)
- Counterfactual Test — 平台(16.5×) > AI(1.9×) (2026-07-05)
- Persona 五态模型 — KNOWN/PARTIAL/UNKNOWN/AMBIGUOUS/OOD (2026-07-05)
- MS-AR 第一刀 — z(t) 调控 regime 转移分析 (2026-07-05)
- Scraper v2.0 — 全量 headline embedding + 日级语义聚合 + 知乎恢复 (2026-07-05)
- 时段剪切审计 — 三段独立拟合 + 连续区块零假设 → 预设2 VERDICT=UNDERPOWERED (2026-07-09, `temporal_slice_audit.py`)
- TVP-AR 参数漂移估计 — Nyblom 的**估计版对应物** (非独立通道): PC2 σ̂²_η=0.00155 (boot_p=0.009, β(t) 持久性轨迹 AR(2) 主根 |ρ| 0.42→0.86, Σφ 0.39→0.83), PC1 无漂移。合成正/负对照已验估计器不假阳; H0 堆积 51.9%(PC2) + ~1/6 漏检 → σ̂²_η≈0 不坐实时不变 (UNDERPOWERED); 不 resolve P2 归因简并 (2026-07-11, `tvp_ar_stationarity.py`)
- Regime 离散性审计 — 拆三层 (镜像 P2a/b/c): **RD-a** 低维 label-free 结构=**holds** (PC1 Silverman 多峰 p=0, 非循环); **RD-b** K 相区计数=**suspect** (gap→**k=1**, GMM(4) 全簇 Jaccard 0.20-0.56, R2 仅 0.48); **RD-c** R2 分离盆地=**not-supported**。★不 crown 连续★: R2 盆地 vs 连续漂移驻留 UNDERPOWERED 竞争 (RQA零复发/weak-irrev/PC2持久升 三路都在 regime 框架内算, 打不破平局) → Competing Explanatory Layer 并存; ★两头(4-clean-regimes/pure-continuum)都不获支持★。R2_axis 多峰因轴由标签选=循环已剔。承重: ledger regime-discretization→**split**; raw E1(rqa/switch/方差)作数存活挂 regime-count; 触第0层"相变"框定, 已同步终审 (2026-07-12, `regime_discreteness.py`)
- Embedding 回溯 — 227 个 v1.0 scraper 文件 → 6060 headline×384维 → `monthly_semantic_state.json` (2026-06/07) 重建 (2026-07-09, `backfill_v1_embeddings.py`)
- 修复 `monthly_aggregator.py` 单平台月份 KeyError (mean_cosine_distance 防御性 .get)

## 关键未解决问题

### 信号质量危机 + 认知螺旋 (2026-07-06)

系统采集的所有数据（Google Trends、Scraper headline、叙事 JSON、LLM 概念分数）
测量的不是"现实结构"，而是**人群对现实结构的叙事化反应**。

**关键翻转**: 这个"叙事化倾向"不是需要清洗的噪声——它恰恰是要测量的对象。
那个让一百万人面对同一堵墙却不分析墙、而是发泄、造梗、站队的东西，
就是"集体层面小真实内在的混沌属性"。

**认知螺旋**: 区分"接近结构分析的讨论"和"纯叙事消费"的标准无法预设，
必须从数据中迭代发现。结论会反向注入数据提取层。

### 数据管道断裂 (2026-07-05, 部分修复 2026-07-09)

Scraper 数据从未进入任何历史分析结论。所有 127 个月的科学产出
(H1/Regime/RQA/ControlManifold/FR31) 输入只有 Google Trends + Narrative JSON。

**2026-07-09 进展**: 227 个 v1.0 scraper 文件已回溯 embedding (6060 headline×384维),
`monthly_semantic_state.json` 重建, 2026-06/07 两月真实语义状态已生成 (协方差迹/POS熵/跨平台JSD)。
**管道打通前半段 (采集→月度语义聚合), 后半段仍未接 (月度语义→月度状态 x(t)→分析链)** ——
`monthly_semantic_state.json` 目前无任何下游读取。从"完全断裂"推进到"断在下游接口"。

## 时段剪切审计 (2026-07-09 执行完成)

> 注: 一个前序会话 (2026-07-06, 已因上下文换出) 已把本审计的两个方法缺陷+修法推理并写入本节,
> 但脚本未留存。2026-07-09 从零重建 `src/analysis/temporal_slice_audit.py` 独立命中同样两个坑
> —— 两次独立推理的收敛, 是对方法的强验证。

**目的**: 检验预设 2（时不变生成机制）—— 三段 (55/36/36 月) 是否同一个系统。
每段独立从零拟合全套 (scaler/ext-PCA/state-PCA/GMM), 复用全局基=预设时不变=作弊。

**方法自检暴露两个缺陷, 在发布结论前修复**:
1. 零假设基线采样错误 — 随机散点 vs 连续区块, PCA 稳定性不可比。null 的 max 夹角 p95=89.7° 顶到天花板, 测试被设计废了。改为连续非重叠区块 block-bootstrap + 每窗独立标准化。
2. KS (10/10 移位) 混淆了两件事 — KS 测**状态分布漂移**不是**机制改变** (时不变系统状态游走也会 KS 显著)。剔出判据。此即 GPT 说的 Semantic Smuggling 实例。

**VERDICT: `UNDERPOWERED_CANNOT_REJECT_P2`** —— 既证伪不了也坐实不了。
- 三段两两子空间均值夹角 40.9/48.3/52.7°, **全部低于**连续区块噪声 p95=70.7° (真实段比随机窗口更相似)。
- 交叉重构 gap 0.19~0.34, 全部低于噪声 p95=0.54。
- **月度分辨率下协方差探针检验力≈0**: 任意两个连续 36 月窗口本身就差 61°、gap 0.46。
- **结论: 不是"验过没问题", 是"仪器看不见"。历史段 (当前手头的 Google Trends 月度粒度) 现有数据不可检验。**
  - ⚠ **2026-07-12 修正一个过绝对声称**: "历史**永久**不可检验" **错** —— 只对"当前月度数据"成立, 不对"历史本身"成立。高分辨率历史痕迹 (带时间戳的微博/B站帖、百度/微博指数日级、日级 Trends 窗口拼接、Wayback 快照) **大量存在、可工程回填**。→ 见「当前待办」新主轴: 历史高分辨率重建。

**★衍生的新预设**: 审计硬编码了"三段"边界 (疫情/ChatGPT 当时代分界) —— **段边界本身是未被意识的
预设**, 与关键词过滤器逻辑同构 (人替数据定边界)。这是"离散化即预设"元原则的实例, 且撞回同一堵
分辨率墙 (边界敏感性扫描/连续 changepoint 都需要窗口, 月度窗口无功率)。详见方法论原则 + Q11。

## 当前待办

| 优先级 | 任务 |
|:--:|------|
| **★主轴** | **历史高分辨率重建 (人类终审 2026-07-12): 往回挖高分辨率、广覆盖历史 (2015-2025), 破月度分辨率墙。广爬(不按51关键词) → 同时攻时间墙+覆盖墙; 用同一 F 侧 embedding 管道 → H/F 观测算子统一。诚实边界: 幸存者/审查偏差永久; 真瓶颈是外部获取非我们硬件。worker 正做可行性侦察 → auditor 复核 → 定工程规模** |
| **P0** | **① Evidence Ledger: 建 `evidence_ledger.json` (E0-E4 + Assumption 列), 现有结论全部过一遍分级归档** |
| **P0** | **② 数据管道接入下游: `monthly_semantic_state.json` 的 cov_trace/各向异性/漂移 三标量 → 缝进 2025-12 后分析链 (aggregator 已算好)** |
| P0 | MS-AR Phase 2: R2 内部 p(x\|R2, z1) — 势阱形变建模 (条件 KDE 势能面) |
| P0 | FR31: engine.py 对接 Stella 自动回复 |
| P1 | ~~低维 TVP-AR 时不变性检验 (不切段)~~ **✅ 2026-07-11** `tvp_ar_stationarity.py`: PC2 σ̂²_η=0.00155 (boot_p=0.009, 与 Nyblom 0.0065 一致), AR(2) 主根 |ρ| 0.42→0.86; PC1 无漂移。**估计版对应物 (非独立通道); 不 resolve P2; H0 堆积 51.9% → UNDERPOWERED** |
| P1 | 第二观察者 (验预设1): 原始文本 embedding KDE 簇 vs GMM R0-R3 拓扑同构。**限制: 原始 embedding 仅 2026-06/07 有, 不能追溯历史段, 只能验新 regime** |
| P1 | micro_burst_detector: 换输入为日级 embedding (替换关键词命中) |
| P1 | 信号质量: 叙事化程度的初始 proxy 设计 |
| P2 | Schema 3.0: 图动力学前置支持 (边定义 + 邻接矩阵) |
| P2 | Level 3: 后验解释 — 特征载荷 → 叙事语义 |
| P3 | Dashboard 增强 — 信号报警, 约束场突变预警 |

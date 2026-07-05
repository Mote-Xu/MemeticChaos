# MemeticChaos — 人类集体情感混沌属性建模

> 最后更新：2026-07-06

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

## 项目目标

利用 2015-2025 年中国互联网热梗数据，刻画互联网集体叙事系统的相变结构。

- **Hassabis 猜想 × AlphaGo 范式**：自然界中存在的模式可被经典学习算法发现。不灌输人类总结的套路，让数据自己说话。
- **四层形式化**：State（叙事状态）/ Observation（观测）/ Control（控制流形）/ Dynamics（动力学）。详见 `FORMALISM.md`。
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
- `data/scraped/hourly/` — 每小时 headline + 384 维 embedding (保留 7 天)
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
- **Dashboard**: https://chaos.mote-pal.xyz/?token=<TOKEN> (Flask :8931 + cloudflared tunnel)
- **systemd**: `memeticchaos-dashboard.service` (enabled, auto-restart)
- **数据同步到本地**: `bash sync_from_server.sh`
- **数据同步到服务器**: `bash sync_to_server.sh`
- **本地任务**: Google Trends 通过飞鸟代理每日拉取 + 叙事生成

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

## 关键未解决问题

### 信号质量危机 + 认知螺旋 (2026-07-06)

系统采集的所有数据（Google Trends、Scraper headline、叙事 JSON、LLM 概念分数）
测量的不是"现实结构"，而是**人群对现实结构的叙事化反应**。

**关键翻转**: 这个"叙事化倾向"不是需要清洗的噪声——它恰恰是要测量的对象。
那个让一百万人面对同一堵墙却不分析墙、而是发泄、造梗、站队的东西，
就是"集体层面小真实内在的混沌属性"。

**认知螺旋**: 区分"接近结构分析的讨论"和"纯叙事消费"的标准无法预设，
必须从数据中迭代发现。结论会反向注入数据提取层。

### 数据管道断裂 (2026-07-05)

Scraper 数据从未进入任何历史分析结论。所有 127 个月的科学产出
(H1/Regime/RQA/ControlManifold/FR31) 输入只有 Google Trends + Narrative JSON。
v2.0 已修复采集端, 但下游仍未接入。日级语义状态待接入月度分析层。

## 当前待办

| 优先级 | 任务 |
|:--:|------|
| **P0** | **数据管道接入: 日级 embedding → 月度语义状态, 续接 2025-12 后的分析链** |
| P0 | MS-AR Phase 2: R2 内部 p(x\|R2, z1) — 势阱形变建模 |
| P0 | FR31: engine.py 对接 Stella 自动回复 |
| P1 | micro_burst_detector: 换输入为日级 embedding (替换关键词命中) |
| P1 | 信号质量: 叙事化程度的初始 proxy 设计 |
| P2 | Schema 3.0: 图动力学前置支持 (边定义 + 邻接矩阵) |
| P2 | Level 3: 后验解释 — 特征载荷 → 叙事语义 |
| P3 | Dashboard 增强 — 信号报警, 约束场突变预警 |

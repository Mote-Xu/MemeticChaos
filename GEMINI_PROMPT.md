# MemeticChaos 项目现状（给外部 AI 的求助）

> 最后更新：2026-07-04

## 一句话

中国互联网集体情感的混沌属性建模。从 2015-2025 年热梗数据中提取 Narrative Dynamics（叙事动力学）规律。不是预测"下一个爆款梗"，是理解集体叙事系统的结构属性和相变行为。

## 数据资产

```
外部场:      51 关键词 × 132 月 (Google Trends 2015-2025)
叙事:        57 条 (22 B站视频 + 36 曲线生成, 含 spread_phases/mutations/semantic_drift)
Level 1:     127 月 × 4 特征 (Stage/Mutation/Institutionalized/SemanticDrift) — 硬事实
Level 2:     127 月 × 10 维 Narrative State x(t) — PCA 降维表示
实时采集:    微博50 + 百度50 + 知乎30 = 130条/小时 (mote-home 24/7)
Dashboard:   Flask + ECharts, chaos.mote-pal.xyz
```

## 当前架构 (v4.0, 2026-07-04 重构完成)

### H1 假说链

**H1**: 存在一个能够描述互联网叙事演化动力学规律的低维状态表示。

| 假说 | 结论 | 证据 |
|------|:--:|------|
| **H1a** (低维性) | ✅ SUPPORTED | 18 维 → d90=10, d95=12 |
| **H1b** (动力学连续性) | ❌ REJECTED | VARX test R²=-0.32 < lag-1 R²=+0.44 |
| **H1c** (当前状态主导) | ⚠️ MIXED | State-only R²=+0.14, 外生变量不帮忙 |

**核心发现**: 月度尺度上，集体叙事动态由随机漂移主导。预测值不可行——应预测转移概率/结构属性/相变。

### 三层特征工程 (AlphaGo 原则: 数据→表示→解释)

**Level 1** ✅ — 硬事实提取 (`src/data/narrative_hard_facts.py`):
- Stage Occupancy (5 维, 复用 Trends 锚定)
- Mutation_Occurred: 是否发生变异 → 月度比率
- Institutionalized: 是否被官方/主流引用 → 月度比率
- Semantic_Drift_Distance: 原义→现义的 embedding 余弦距离 (sentence-transformers)

**Level 2** ✅ — 表示学习 (`src/models/representation_learning.py`):
- PCA 降维: 18 维 → 10 维 Narrative State x(t)
- PC1: entropy(+), hhi(-), stage_fixation(+), stage_origin(-)
  即"注意力多样性 vs 集中度" + "叙事僵化 vs 新生"
- H1b REJECTED: 系统是近似随机游走

**Level 3** 🔜 — 后验解释: 特征载荷 → 叙事语义

### Narrative 是动态图

节点（梗）+ 有向边（变异/竞争/替代/融合）。Schema 支持边定义。
工程上用邻接矩阵主特征向量衔接。图神经网络后续可升级。

### 动力学方程

```
x(t+1) = F(x(t), u(t), y(t))
```
u(t) = External Field (51 关键词 → PCA 8 维)
y(t) = Attention (HHI + entropy, 反馈闭环)
F 的具体形式: Ridge 基线已否定，非线性待评估

### FR31 四指标接口 ✅

`src/advisor/metrics.py` — Inertia / Resilience / Position / Sensitivity:

| 指标 | 当前值 | 含义 |
|------|:--:|------|
| **Inertia** | 0.77 | 叙事生态高度僵化, fixation 占 52%, origin+emergence=0% |
| **Resilience** | 0.37 | 2025H2 恢复力归零, 偏离均衡 3.9σ |
| **Position** | fixation主导 | 连续 24 个月 fixation 主导, 零阶段转换 |
| **Sensitivity** | 0.56 ↑ | 上升中. Critical Slowing=0.76, PC4/PC5 >2x 基线方差 |

四个指标拼出当前系统状态：**高惯性 + 零恢复力 + 敏感性加速上升 = 亚稳态**。
系统同时"动不了"和"越来越容易炸"。Critical Slowing 0.76 是经典的分岔前预警信号。

## 已退役的旧架构 (v0.2)

- 混沌轴: 五类别人肉权重 → 杀掉了
- 约束场 5D: 每梗一个静态向量 → 已替换为 Level 1 动态特征
- 类别互斥: "predict chaos value" → 已转为 predict structure

## 当前困惑（外部 AI 已给方向，待消化执行）

### 1. 随机游走之后怎么建模？（→ 概率密度 + 相变检测）
外部 AI 共识：放弃点估计，转向转移概率矩阵 / Fokker-Planck 范式 /
CUSUM 结构断裂检测。已有 Sensitivity 指标的 Critical Slowing 是第一步。

### 2. 外部层的角色？（→ 多尺度解耦）
外部 AI 指出：u(t) 是边界条件（慢变量，调制方差），不应与快变量混在
同一时间轴。建议用 GARCH 建模 u(t)→方差，而非 u(t)→均值。

### 3. 图结构怎么用？（→ 拓扑弹性 + 图扩散）
外部 AI 分歧：Graph Embedding vs Graph Diffusion (Laplacian)。
共识：图拓扑变化是相变的先导信号。代数连通度(次小特征值)
可直接反映系统的级联传导能力。

### 4. 个体→集体映射？（→ 双层语义编码器）
GPT 建议的 Two-stage Encoder: LLM 抽取语义三元组 →
与叙事图节点做语义距离投影 → 个体坐标 p_i(t)。
需要 sentence-transformers 计算用户文本与梗描述的余弦相似度。

### 5. 时间分辨率？（→ 快变量做闪洪预警）
外部 AI 共识：Scraper 小时级数据用于微观"闪洪预警"——
注意力波动率、情绪点火率。宏观慢变量(Level 1)保持月度。
快慢分离，各司其职。

### 6. 新增：敏感性指标的价值
Sensitivity = ∂state/∂input。GPT 指出这比 Position 更重要——
FR31 顾问的核心是告诉用户"现在不要碰这个话题"，
而不是"你位于哪个节点"。已实现。

## FR31 顾问方向（可讨论）

Stella ⭐ — 运行在 mote-home 上的 OpenClaw AI agent，通过企业微信接入。
已具备：24/7 运行、企微收发、DeepSeek V4 Flash 驱动、个性/记忆系统。

FR31 是情感约束场顾问——不是替用户写回复，是战略对弈伙伴。
用户给出情境和判断，系统给出独立判断，两者交叉验证。

下一步：把三指标接口挂到 Stella 上，让企微上能查询 Narrative State。
然后建 persona.py 做个体定位。

需要思考：
- 个体行为（聊天文本）→ 叙事图坐标的映射方式
- 顾问的推理链：集体规律 → 个体约束场 → 策略建议
- 如何避免变成爹味教练？如何不假装能读心？

## 技术栈

Python 3.12 + numpy/scipy/pandas/scikit-learn/sentence-transformers + Flask + ECharts
LLM: DeepSeek API
Agent: OpenClaw + 企业微信 (mote-home, Ubuntu 24.04)
Dashboard: chaos.mote-pal.xyz (cloudflared tunnel + token auth)

## 相关文档

- `DESIGN_DISCUSSION.md` — 双 AI 审议结论 + H1 假说 + 三层工程
- `CLAUDE.md` — 项目全貌 + 服务器配置
- `REQUIREMENTS.md` — 功能需求 + FR19 审计
- `FRAMEWORK_DESIGN.md` — v4.0 架构概要

# MemeticChaos — 人类集体情感混沌属性建模

> 最后更新：2026-06-30

## 项目目标

利用 2020-2025 年中国互联网热梗数据，对人类集体情感的混沌属性进行建模。

- **Hassabis 猜想 × AlphaGo 范式**：自然界中存在的模式可被经典学习算法发现。不灌输人类总结的套路，让数据自己说话。
- **两层架构**：外部环境场（平台注意力数据）+ 内部叙事混沌层（叙事结构）。两层耦合驱动集体情感状态演化。
- **数学框架退到验证层**：SIR/相图/吸引子验证生成轨迹的动力学合理性，不做生成。
- **核心产出**：集体情感相图 + 秩序形态预测模型（FR19 v0.2）。

## 技术栈

Python 3.12 + conda `MemeticChaos` + numpy/scipy/pandas/scikit-learn/pytrends + Flask + ECharts
仓库：git@github.com:Mote-Xu/MemeticChaos.git

## 数据资产（截至 2026-06-30）

```
外部层:      51 关键词, 5217 月数据点 (Google Trends 2015-2025)
内部层:      43 梗, 1048 月数据点
叙事:        87 条 (22 B站 + 36 曲线生成 + 29 策展)
LLM 概念分数: 57 条叙事 × 35 维
集体混沌轴:  133 个月 (2015-01 → 2026-06, 含 scraper 当前月)
实时采集:     微博50 + 百度50 + 知乎30 = 130条/小时
```

数据文件：
- `data/collector/external_field_2015_2025.json` — 51 关键词外部场
- `data/collector/google_trends_2015_2025.json` — 43 梗注意力曲线
- `data/processed/narratives/` — 22 个 LLM 抽取
- `data/processed/narratives_from_trends/` — 36 个曲线生成
- `data/processed/llm_concept_scores.json` — 57 条 LLM 概念分数
- `data/processed/dashboard_state.json` — 当前状态 (预计算)
- `data/processed/dashboard_history.json` — 133 月历史 (预计算)
- `data/curated/memes_2020_2025.json` — 29 人工策展

## 项目结构

```
src/
├── data/
│   ├── scraper.py                     # 实时采集 (微博+百度+知乎, 每小时)
│   ├── live_pipeline.py               # 采集→更新→报告 (每天)
│   ├── signal_pipeline.py             # 新梗发现→LLM叙事→概念打分 (每天)
│   ├── monthly_narrative.py           # LLM 月度叙事摘要 (每周)
│   ├── narrative_extractor.py         # B站视频叙事抽取 (22/22)
│   ├── narrative_from_trends.py       # Google Trends 曲线→叙事
│   └── trends_loader.py / trends_weekly.py / curator.py / rebuild_from_trends.py
├── constraint/
│   ├── llm_concept_scorer.py          # LLM 35概念打分 (48/48) — Step 2 核心
│   ├── concept_bottleneck.py          # 软匹配概念瓶颈 (弃用, Step 1)
│   └── delta_transition.py            # ΔTransition + 3Validator
├── trajectory/
│   ├── meme_trajectory.py             # 29条轨迹 (Schema 2.1)
│   └── trajectory_viz.py              # 轨迹相图
├── models/
│   ├── order_form_predictor.py        # ★ 两层预测模型 (FR19 v0.2)
│   ├── collective_predictor.py        # 旧版预测器 (保留参考)
│   └── sir_meme.py / abm_simulation.py / attractor.py / individual_calibrator.py
├── analysis/
│   ├── collective_dynamics.py         # 集体情感系统动力学 (133月)
│   ├── platform_flow.py               # 跨平台注意力流动
│   ├── narrative_clustering.py        # 叙事无监督聚类 (ARI=0.27)
│   └── phase_diagram.py / backtest.py / lifecycle.py / phase_detect.py
├── dashboard/
│   ├── app.py                         # Flask API (9端点, token auth)
│   ├── analyzer.py                    # 精细建模查询
│   └── templates/index.html           # ECharts 前端 (回放/拖拽/倍速/移动端)
├── advisor/
│   ├── persona.py                     # 个体画像模型 (FR31 Layer 2)
│   └── (engine.py — 待建)
└── meme_inspector.py
tests/
├── test_sir_model.py                  # 24/24
└── test_system_integrity.py           # 38/38
```

## FR19: 秩序形态预测模型 (v0.2) — ⚠️ 已知结构性缺陷

### 架构

```
外部场 (51维 PCA→8维) ──→ RidgeCV ──→ 混沌轴 + 约束场(5D) + 类别分布 + 注意力结构
                                          │
LLM约束历史 (5维×6月) ──→ RidgeCV 残差 ──→│
                                          ▼
                                   秩序形态 (8聚类, 含约束维度)
                                          │
                            LLM 月度叙事摘要 ──→ NL 报告
```

### 模型性能

| 指标 | 值 | 注意 |
|------|-----|------|
| 混沌轴月间预测 MAE | 比 lag-1 差 ~22% | 实证：混沌轴是随机游走 |
| Structure 预测 R² | 0.79 | HHI+叙事熵可从外部场预测 |
| 叙事聚类 ARI | 0.27 | 对 35 维概念向量聚类：机器见 3 类，非人工 5 类 |
| 内部叙事层贡献 | 0 | 叙事被偷换成约束向量，从未真正参与预测 |

### 当前局限与待解决 (2026-07-04 全面审计)

**结构性问题**：
- **约束场是静态标签**：每个梗只有一个向量，不随时间变化。同一梗在爆发期和消退期用同一约束值。
- **内部层被偷换概念**：本应使用 87 条叙事的结构信息（传播阶段、变异类型、语义漂移），
  实际压缩成了 5 维约束向量。从未真正参与预测。
- **类别应允许多标签**：一个梗可以同时属于多个类别（普信男既解构自嘲又攻击发泄），
  当前互斥分类有信息损失。

**数据质量问题**：
- **概念打分零值过高**：平均 16/35 概念为零分。64 条中 24% 超过 20 个零。
  约束场是分组均值，零分把约束压向零，人为降低方差。
- **Dashboard 当前月基于小样本**：混沌轴 +0.229 仅来自 2 个梗 8 条 scraper 信号。

**实证结论（非 bug）**：
- 混沌轴是随机游走（预测比 lag-1 差 ~22%）——符合"与混沌共存"预期
- 注意力结构可预测（R²=0.79）
- 叙事聚类：机器见 3 类，非人工 5 类（ARI=0.27）
- 五类别是脚手架，归类基于叙事特征有逻辑支撑问题

## FR31: 情感约束场顾问 (待建)

### 定位

不是替用户写回复的 AI。是**战略对弈伙伴**——用户在具体情境中给出自己的判断，
系统给出独立判断，两者对质，直到找到比独自判断更好的方案。

FR19（集体混沌属性建模）是 FR31 的推理引擎。没有集体层的结构发现，
FR31 就是一个更聪明的聊天机器人。市面上 AI 在情感领域的两个死胡同：
1. 只会教人维护边界、做好好先生——违背现实结构
2. 依赖道德直觉理解情感——判断完全偏离现实

FR31 必须避开这两者，从集体层的结构性规律推导个体层决策。

### 架构

三层：
- Layer 1（参考层）：FR19 集体混沌规律——信号-动机分离、约束场形变、秩序形态
- Layer 2（数据层）：对方画像 + 互动历史——纯数据，不掺杂理论框架
- Layer 3（情境层）：当前聊天上下文 + 用户的情感温度 + 时机约束

### 虹姐参考数据

虹姐（情感咨询机构老师）一个月服务的完整聊天记录已导出：
- `E:\Desktop\杂🐟项\与虹姐的聊天记录.txt`（17529 行，私聊）
- `E:\Desktop\杂🐟项\徐子浩服务指导1.29-2.28-群聊聊天记录.txt`（6527 行，群聊）

虹姐的有效价值：情境化的即时战术判断——"怎么回""等一等""先想好方案再约"。
她的局限：不理解用户的认知框架、部分判断带有爹味、无法从底层原理推导策略。

FR31 应该达到：虹姐的战术精度 + 用户的认知深度。

知识底座参考 `E:\Desktop\weichen.txt`（体系纯理论版）和 `wctxjx.txt`（含个人经历的完整版）。

> ⚠️ AlphaGo 原则：体系知识写入 .md 作为背景参考，但**不硬编码进模型**。
> 代码中唯一可引用的结构划分来自真实主权——小真实的内部结构
> （感知→认知→体验），用于建模对方偏好。其余不进代码。

## 服务器部署 (mote-home)

> **服务器是项目主运行环境。本地仅用于查看结果和精细建模。**

- **位置**: `/mnt/data/MemeticChaos/` → symlink `~/MemeticChaos/`
- **Python**: `~/miniconda3/envs/MemeticChaos/bin/python` (PYTHONNOUSERSITE=1)
- **cron 任务**:
  - 每小时 17 分: 采集 (`scraper.py`)
  - 每天 2:37: B站时间序列 (`bilibili_timeseries.py`)
  - 每天 4:47: 实时管线更新 (`live_pipeline.py` + `signal_pipeline.py`)
  - 每周日 4:50: LLM 月度叙事摘要 (`monthly_narrative.py`)
  - 每周日 5:00: 12月预测报告 + Dashboard 状态刷新 (`order_form_predictor.py`)
- **Dashboard**: https://chaos.mote-pal.xyz/?token=DASHBOARD_TOKEN_REMOVED (Flask :8931 + cloudflared tunnel)
- **systemd**: `memeticchaos-dashboard.service` (enabled, auto-restart)
- **数据同步到本地**: `bash sync_from_server.sh`
- **数据同步到服务器**: `bash sync_to_server.sh`
- **本地任务**: Google Trends 通过飞鸟代理每日拉取 + 叙事生成

## 已完成

- 24/24 测试 + 38/38 系统完整性验证
- FNN 标准 Kennel / 连续 R₀ / S₀ 参数 / R₀ 扰动鲁棒性 / 盲聚类
- 集体混沌轴 133 月数据，10 次系统级相变
- 数据自动补充：服务器 24/7 + 本地 Google Trends
- FR19 v0.2: 秩序形态预测模型 + LLM 概念打分 + 月度叙事摘要
- Dashboard: ECharts + Flask + token auth + 历史回放 + 精细建模查询
- 实时信号管线: 新梗自动发现 → LLM叙事 → 概念打分 → 入库
- 跨平台流动框架 + 叙事无监督聚类
- 知乎 API 修复 (cookie 认证)
- 三数据源: 微博+百度+知乎

## 当前待办

| 优先级 | 任务 |
|:--:|------|
| P0 | FR31: 情感约束场顾问 (需求已定, 待建) |
| P1 | 内部叙事层激活 — 约束场信息→预测能力 |
| P2 | 周度/日度分辨率 — scraper 数据聚合 |
| P3 | 叙事类型作为预测特征 |
| P4 | Dashboard 增强 — 信号报警, 约束场突变预警 |

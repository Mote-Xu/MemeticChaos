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

| 指标 | 值 | 解读 |
|------|-----|------|
| 混沌轴月间预测 MAE | 比 lag-1 差 ~22% | 不如"用上个月的值" |
| Constraint 预测 R² | 0.52 | 约束场可从外部特征预测 |
| Structure 预测 R² | 0.79 | 注意力结构可预测 |
| 内部层增益 | 0 | 叙事层对预测无贡献 |
| 叙事聚类 ARI | 0.27 | 机器见3类, 非人工5类 |

### ⚠️ 已知结构性缺陷 (2026-07-04 审计)

1. **混沌轴权重是编的**：五类别映射值 (解构自嘲=-0.33, 攻击发泄=-0.62…)
   是人工拍的，无数据支撑。混沌轴 = 类别固定权重 × 活跃梗注意力加权平均。

2. **约束场是静态的**：LLM 对每个梗只打了一次分，约束向量不随时间变化。
   同一个梗在 2021 年爆发期和 2024 年消退期用同一个约束值。
   月度聚合的变化仅来自"哪些梗活跃"的名单变化，变化幅度极小。

3. **五类别是手工分的**：五类别（解构自嘲/攻击发泄/虚无退却/身份认同/纯粹娱乐）
   是人工标注的，没有聚类验证。HHI/熵的计算依赖正确分类。

4. **内部层不是叙事**：内部层用 5 维约束向量做 Ridge，而不是用真正的叙事结构
   （传播阶段数、变异类型、语义漂移）。87 条叙事的结构信息被压缩成 5 个数字。

5. **月度粒度可能太粗**：混沌轴的 "随机游走" 可能不是随机——可能是周度/日度有结构，
   月间采样频率太低。

6. **36/43 覆盖**：64 条 LLM 概念分数中只有 36 条有 Google Trends 时间序列。
   其余 28 条无法参与时间维度建模。问题

## FR31: 情感约束场顾问 (新需求, 待建)

三层架构：
- Layer 1 (参考层): MemeticChaos 集体规律
- Layer 2 (个体模型): 对方画像 + 互动历史
- Layer 3 (情境层): 当前聊天上下文 + 情感温度

知识底座参考 `wctxjx.txt`（用户体系文档），详见 `REQUIREMENTS.md`。

> ⚠️ AlphaGo 原则：体系知识写入 .md 作为背景参考，但**不硬编码进模型**。
> 机器从数据中自己发现结构，不被人类总结的套路污染。

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

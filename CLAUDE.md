# MemeticChaos — 人类集体情感混沌属性建模

> 最后更新：2026-06-27

## 项目目标

利用 2020-2025 年中国互联网热梗数据，对人类集体情感的混沌属性进行建模。

- **Hassabis 猜想驱动**：自然界中任何存在的模式都可以被经典学习算法高效地发现。模因是被集体情感/社会压力/语言演化筛选过的系统——有结构，就能被学习。
- **两层架构**：外部环境场（平台注意力数据）+ 内部混沌层（叙事结构）。两层耦合驱动集体情感状态演化。
- **数学框架退到验证层**：SIR/相图/吸引子验证生成轨迹的动力学合理性，不做生成。
- **核心产出**：集体情感相图（已有）+ 秩序形态预测模型（FR19, v0.1 已上线）。

## 技术栈

Python 3.12 + conda `MemeticChaos` + numpy/scipy/pandas/scikit-learn/pytrends
仓库：git@github.com:Mote-Xu/MemeticChaos.git

## 数据资产（截至 2026-06-27）

```
外部层 (环境场):  51 关键词 × 132 月 = 5217 数据点 (Google Trends 2015-2025)
内部层 (梗):      43 关键词 × ~100 月 = 1048 数据点
叙事层:           87 条 (22 B站视频 + 36 曲线生成 + 29 人工策展)
集体混沌轴:       127 个月 (2015-01 至 2025-07)
```

数据文件：
- `data/collector/external_field_2015_2025.json` — 51 关键词外部场
- `data/collector/google_trends_2015_2025.json` — 43 梗注意力曲线
- `data/processed/narratives/` — 22 个 LLM 抽取
- `data/processed/narratives_from_trends/` — 36 个曲线生成
- `data/curated/memes_2020_2025.json` — 29 人工策展（含真实注意力数据）

## 项目结构

```
src/
├── data/
│   ├── curator.py                     # 策展数据管理
│   ├── narrative_extractor.py         # LLM叙事抽取 (22/22)
│   ├── narrative_from_trends.py       # 曲线→叙事生成 (36条)
│   ├── trends_loader.py               # Google Trends 加载器
│   ├── rebuild_from_trends.py         # 真实数据重构项目
│   ├── scraper.py                     # 微博实时采集
│   ├── auto_collector.py              # 自动采集器
│   ├── bilibili_timeseries.py         # B站时间序列
│   └── live_pipeline.py               # 采集→更新→报告
├── constraint/
│   ├── concept_bottleneck.py          # 35概念→5D约束 (软匹配, Step 1用)
│   └── delta_transition.py            # ΔTransition + 3Validator + Ridge
├── trajectory/
│   ├── meme_trajectory.py             # 29条轨迹 (Schema 2.1)
│   └── trajectory_viz.py              # 轨迹相图
├── models/
│   ├── order_form_predictor.py        # ★ 两层预测模型 (FR19 v0.1)
│   ├── collective_predictor.py        # 旧版预测器 (保留参考)
│   ├── sir_meme.py / abm_simulation.py / attractor.py / individual_calibrator.py
├── analysis/
│   ├── collective_dynamics.py         # 集体情感系统动力学 (127月)
│   ├── phase_diagram.py / backtest.py / lifecycle.py / phase_detect.py
└── meme_inspector.py                  # 梗分析工具
tests/
├── test_sir_model.py                  # 24/24
└── test_system_integrity.py           # 38/38
```

## FR19: 秩序形态预测模型 (v0.2 — Step 2 完成)

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

### Step 2 关键升级 (2026-06-27)

1. **LLM 概念打分** 替换软匹配：DeepSeek API 对 48 条 narrative 打分 35 概念
   - 约束场方差提升 15-22x (Humor std 0.01→0.22, Conflict std 0.01→0.20)
   - Constraint R² = 0.52（约束场可从外部特征预测）
2. **LLM 月度叙事摘要**：微博热搜 + 活跃梗 + Trends 异常 → NL 描述
   - 周报现在包含 LLM 生成的完整叙事分析
3. **8 种秩序形态**含约束维度："混沌释放·虚无退却·Identity·分散"(40%)
4. **混沌轴仍是随机游走**：确认"与混沌共存"哲学——个别月不可预测，但秩序形态有结构

### 预测输出

- 6-12 个月秩序形态预测（含约束场 profile + 置信度）
- 混沌轴趋势区间
- 类别分布 + HHI + 叙事熵
- **LLM 生成的月度集体叙事摘要**（NL，3-5 句深度分析）

### 文件

| 文件 | 用途 |
|------|------|
| `src/models/order_form_predictor.py` | 主预测器 (v0.2) |
| `src/constraint/llm_concept_scorer.py` | LLM 概念打分器 (48/48 成功) |
| `src/data/monthly_narrative.py` | LLM 月度叙事摘要生成器 |
| `data/processed/llm_concept_scores.json` | 已打分的概念矩阵 |
| `data/processed/monthly_narratives.jsonl` | 历史月度叙事摘要 |
| `data/processed/order_form_report.txt` | 最新预测报告 |

## 服务器部署 (mote-home) — 项目主阵地

> **服务器是项目的主运行环境。本地仅用于查看结果和精细建模。**

- **位置**: SSD `~/MemeticChaos/`
- **Python**: `~/miniconda3/envs/MemeticChaos/bin/python`
- **cron 任务**:
  - 每小时 17 分: 微博热搜采集 (`scraper.py`)
  - 每天 2:37: B站时间序列采集 (`bilibili_timeseries.py`)
  - 每天 4:47: 实时管线更新 (`live_pipeline.py`)
  - **每周日 4:50**: LLM 月度叙事摘要 (`monthly_narrative.py`)
  - **每周日 5:00**: 秩序形态预测报告 (`order_form_predictor.py --forecast 12`)
- **数据同步到本地**: `bash sync_from_server.sh`（拉取 scraper 数据 + 预测报告）
- **数据同步到服务器**: `bash sync_to_server.sh`（推送 Google Trends 数据 + 叙事）
- **本地任务**: Google Trends 通过飞鸟代理每日拉取 (`trends_loader.py`) + 叙事生成 (`narrative_from_trends.py`)

## 已完成

- 24/24 测试 + 38/38 系统完整性验证
- FNN 标准 Kennel / 连续 R₀ / S₀ 参数 / R₀ 扰动鲁棒性 / 盲聚类
- 集体混沌轴 127 月数据，10 次检测到系统级相变
- 数据自动补充：服务器 24/7 + 本地 Google Trends
- **FR19 v0.1**: 秩序形态预测模型上线，每周自动预测

## 下一步 (Step 3 — Dashboard)

1. **Dashboard**：交互式 Web 界面（相图 + 时间轴 + 活跃梗列表 + NL 摘要 + 精细建模查询入口）
2. **精细建模 API**：服务器端查询接口，支持"对某个具体话题做深度分析"
3. **知乎 API 修复**：实时采集数据源补充

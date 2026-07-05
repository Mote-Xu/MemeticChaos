# 需求文档

> 方向：Hassabis 猜想驱动的集体情感混沌属性建模。
> 四层形式化：State / Observation / Control / Dynamics。
> 不是论文，是功能系统。AlphaGo 范式：不灌输人类套路。
> 最后更新：2026-07-06

## 功能需求

| ID | 功能 | 角色 | 状态 |
|----|------|:--:|:--:|
| FR1 | 29热梗策展数据集 | 训练数据 | ✅ |
| FR2 | SIR/SIRS/双群体 + 参数拟合 | 验证层 | ✅ |
| FR3 | 生命周期分析 | 验证层 | ✅ |
| FR4 | ABM 300-Agent仿真 | 验证层 | ✅ |
| FR5 | 吸引子检测 (FNN/RQA/Lyapunov) | 验证层 | ✅ |
| FR6 | 相变检测 | 验证层 | ✅ |
| FR7 | 情感弧线分类 | 训练数据 | ✅ |
| FR8 | 个体校准器 (贝叶斯后验) | 验证层 | ✅ |
| FR9 | 模因相图 | 验证层 | ✅ |
| FR10 | 历史回测 + R₀扰动 + 盲聚类 | 验证层 | ✅ |
| FR11 | B站拼音+SIRS-M拟合 | 已跑通 | ⚠️ |
| FR12 | LLM叙事抽取 (22 B站视频) | 数据源 | ✅ |
| FR13 | MemeTrajectory 构建 (Schema 2.1) | 数据结构 | ✅ |
| FR14 | 实时爬虫 v2.0 — 微博+百度+知乎, 全量 embedding | 实时采集 | ✅ |
| FR15 | 采集→更新→报告管线 (live_pipeline) | 自动化 | ✅ |
| FR16 | Google Trends 十年数据 (51外场+43梗) | 数据源 | ✅ |
| FR17 | 叙事档案 (57条: 22 B站+36 曲线) | 数据源 | ✅ |
| FR18 | 集体动力学分析 (127月) | 分析 | ✅ |
| FR19 | Narrative Dynamics — 相变结构刻画 + H1 验证 | ★核心 | ✅ v4.1 |
| FR20 | LLM 概念打分 (57条 narrative × 35维) | 内部层 | ✅ |
| FR21 | LLM 月度叙事摘要 | 报告 | ✅ |
| FR22 | 新梗自动发现 (signal_pipeline) | 自动化 | ✅ |
| FR23 | Dashboard — ECharts 前端 + Flask API | 展示 | ✅ |
| FR24 | 精细建模查询 (/api/analyze) | 交互 | ✅ |
| FR25 | Dashboard 密码保护 (token + cookie) | 安全 | ✅ |
| FR26 | Dashboard 历史回放 (播放/拖拽/倍速) | 交互 | ✅ |
| FR27 | 知乎 API 修复 (cookie 认证) | 实时采集 | ✅ |
| FR28 | 跨平台注意力流动框架 | 分析 | ✅ |
| FR29 | 叙事无监督聚类 (ARI=0.27) | 分析 | ✅ |
| FR30 | Regime Map — GMM 4 相区 + 转移矩阵 | Dynamics | ✅ |
| FR31 | FR31 统一引擎 + 四指标 + 五态输出 | 顾问层 | ✅ |
| FR32 | RQA 不可逆性检验 + Time-Reversal | Dynamics | ✅ |
| FR33 | Control Manifold — u(t)→z(t) 控制轴 | Control | ✅ |
| FR34 | MS-AR 第一刀 — z(t) 调控 regime 转移 | Dynamics | ✅ |
| FR35 | Scraper v2.0 — 全量 headline embedding + 日语义聚合 | 实时采集 | ✅ |

## 数据资产 (截至 2026-07-06)

```
外部场:       51 关键词 × 132 月 Google Trends (2015-2025)
叙事档案:     57 条 (22 B站 + 36 曲线, 含 spread_phases/mutations/semantic_drift)
Level 1 硬事实: 127 月 × 4 特征 (Stage/Mutation/Inst/Drift)
Level 2 表示:   127 月 × 10 维 Narrative State x(t) (PCA d90=10)
Regime Map:     4 观测簇 → 3 物理相区 + 转移矩阵
Control:        3 维控制轴 z(t) (Diffusion Map)
Persona:        57节点叙事图 → P(meme|text) 概率分布, 五态输出
实时采集 v2.0:  微博50 + 百度50 + 知乎30 → 130条/小时 → 全量 384 维 embedding
日级聚合:       梗语义相似度 + 注意力集中度 + 语义新颖度
```

## 核心发现

1. **H1a ✅, H1b ❌**：叙事状态存在低维表示 (d90=10), 但月度点预测在物理上不可行 (VARX R²=-0.32)。确认"与混沌共存"不是妥协，是实证。
2. **R2 Fixation 是真实结构分离**：RQA 零跨相区复发, Time-Reversal 对称。当前锁死 37 月 (中位 ×2)。
3. **注意力结构可预测 (R²=0.79)**：可预测的不是每月状态值，是系统的结构配置。
4. **机器见 3 类叙事，非人类 5 类**：ARI=0.27。数据自己说了不同的话。
5. **内部叙事层贡献仍是 0**：叙事被偷换成约束向量，从未真正参与预测。
6. **z(t) 在 regime 内部形变状态**：R2 内 PC4/PC5 与 z1 强相关 (r=±0.85), 方差放大 2.31×。切换不聚集在特定 z1 (仅 14 次切换, 统计效力不足)。
7. **Scraper v2.0 已建但未接入**：日级语义管道产出了 embedding 聚合, 但下游分析层未接。2026 年数据悬空。
8. **信号源是叙事化反应，非现实结构**：采集的数据测量的是人群对现实结构的叙事化反应。叙事化倾向本身是要测量的对象，不能预设。

## FR19 全面审计 (2026-07-04, 仍适用)

**结构性问题**：

| # | 当前状态 | 本应是什么 |
|---|---------|----------|
| 1 | 约束场是静态标签 | 应随时间变化（同一梗爆发期 vs 消退期不同） |
| 2 | 内部层用的是约束向量，不是叙事 | 应用叙事结构（传播阶段/变异/漂移） |
| 3 | 五类别互斥，不能多标签 | 一个梗可同时属于多个类别 |

**数据质量问题**：

| # | 问题 |
|---|------|
| 4 | 概念打分平均 16/35 为零，24% 梗超过 20 个零 |
| 5 | Dashboard 当前月基于 2 个梗 8 条 signal — v2.0 scraper 已修复此问题 |

**v4.1 重构方向 (已完成)**：
1. 杀混沌轴 → 用数据驱动的 State Space
2. 预测转移概率/相变，不预测值
3. 四层形式化：State / Observation / Control / Dynamics

## 服务器 (mote-home)

```
mote-home cron:
  每小时 :17    scraper.py v2.0       → 微博50 + 百度50 + 知乎30 + 全量 embedding
  每小时 :20    refresh_stella_state  → Stella 状态刷新
  每天 23:57    scraper --aggregate   → 日级语义聚合
  每天 2:37     bilibili_timeseries   → B站时间序列
  每天 4:47     live_pipeline + signal_pipeline → 轨迹更新 + 新梗发现
  周日 4:50     monthly_narrative     → LLM 月度叙事摘要
  周日 5:00     order_form_predictor  → 12月预测 + Dashboard 刷新

systemd:
  memeticchaos-dashboard  → Flask :8931 (always on)
  cloudflared             → chaos.mote-pal.xyz (token auth)

数据同步:
  sync_to_server.sh    → 本地 → 服务器 (Google Trends + 叙事)
  sync_from_server.sh  → 服务器 → 本地 (报告 + 采集数据)
```

- **服务器**: `ssh mote@100.118.10.0`, 项目 `/mnt/data/MemeticChaos/` → symlink `~/MemeticChaos/`
- **Python**: `~/miniconda3/envs/MemeticChaos/bin/python` (PYTHONNOUSERSITE=1)
- **Dashboard**: https://chaos.mote-pal.xyz/?token=<TOKEN>
- **本地**: Google Trends 通过飞鸟代理拉取, conda 环境 `MemeticChaos`

## 项目结构

```
src/
├── data/
│   ├── scraper.py                     # v2.0: 全量 headline embedding
│   ├── live_pipeline.py               # 采集→更新→报告
│   ├── signal_pipeline.py             # 新梗发现→LLM叙事→概念打分
│   ├── monthly_narrative.py           # LLM 月度叙事摘要
│   ├── narrative_extractor.py         # B站视频叙事抽取
│   ├── narrative_from_trends.py       # Trends 曲线→叙事
│   ├── narrative_hard_facts.py         # ★ Level 1 硬事实提取
│   └── trends_loader.py / trends_weekly.py / curator.py
├── constraint/
│   ├── llm_concept_scorer.py          # LLM 35概念打分
│   ├── concept_bottleneck.py          # 软匹配概念瓶颈 (弃用)
│   └── delta_transition.py            # ΔTransition + 3Validator
├── models/
│   ├── representation_learning.py     # ★ Level 2: PCA + H1 验证
│   ├── order_form_predictor.py        # FR19 v0.2 (已退役, 保留参考)
│   └── sir_meme.py / abm_simulation.py / attractor.py
├── analysis/
│   ├── regime_detector.py             # ★ GMM 4 相区 + 转移矩阵
│   ├── irreversibility_test.py        # ★ RQA + Time-Reversal
│   ├── control_manifold.py            # ★ u(t)→z(t) 控制轴
│   ├── ms_ar_first_cut.py             # ★ MS-AR 第一刀
│   ├── collective_dynamics.py         # 集体系统动力学
│   ├── narrative_clustering.py        # 叙事无监督聚类
│   └── platform_flow.py / phase_diagram.py / backtest.py
├── dashboard/
│   ├── app.py                         # Flask API
│   ├── analyzer.py                    # 精细建模查询
│   └── templates/index.html           # ECharts 前端
├── advisor/
│   ├── engine.py                      # ★ FR31 统一引擎
│   ├── persona.py                     # ★ 个体画像, 五态模型
│   └── metrics.py                     # ★ 四指标
└── meme_inspector.py
```

## 非功能需求

| ID | 描述 | 状态 |
|----|------|:--:|
| NFR1 | 可复现 | ✅ |
| NFR2 | 哲学一致性 | ✅ |
| NFR3 | 模块化 | ✅ |
| NFR4 | 外部数据驱动 (非纯数学建模) | ✅ |
| NFR5 | 服务器 24/7 主阵地, 本地查看结果 | ✅ |
| NFR6 | 安全 (Dashboard token auth) | ✅ |

## 下一步

| 优先级 | 任务 | 说明 |
|:--:|------|------|
| **P0** | **数据管道闭环** | 日级 embedding → 月度语义状态, 续接 2026 分析链 |
| **P0** | **MS-AR Phase 2** | R2 内部 p(x\|R2, z1) 势阱形变建模 |
| P0 | FR31 对接 Stella | engine.py → 企微自动回复 |
| P1 | micro_burst_detector | 输入替换: 关键词 → 日级 embedding |
| P1 | 信号质量初始 proxy | 叙事化程度的第一轮近似度量 |
| P2 | Schema 3.0: 图动力学 | 边定义 + 邻接矩阵 |
| P2 | Level 3: 后验解释 | 特征载荷 → 叙事语义 |
| P3 | Dashboard 增强 | 信号报警, 约束场突变预警 |

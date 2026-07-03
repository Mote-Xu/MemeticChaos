# 新会话启动提示词

## 先读这些文件（按顺序）

1. `DESIGN_DISCUSSION.md` — 2026-07-04 双 AI 联合审议结论，H1 假说、反馈闭环、三层工程、图动力学
2. `CLAUDE.md` — 项目全貌、数据结构、服务状态
3. `REQUIREMENTS.md` — FR19 审计 + FR31 需求 + 虹姐数据
4. `FRAMEWORK_DESIGN.md` — v4.0 架构概要

## 我是谁

项目作者。不是你的客户——是你的合作者。我对项目有自己的完整认知框架。
不要用"你可以这样做"的口吻跟我说话。直接讨论技术问题。

## 项目当前状态

- Dashboard 在服务器上跑着：chaos.mote-pal.xyz (token: DASHBOARD_TOKEN_REMOVED)
- 数据管线正常：每小时采集 130 条热搜，每天自动更新
- **FR19 v0.2 已退役** —— 混沌轴是人工编的、约束场是静态的、内部层从未激活
- **v4.0 目标**：将 FR19 从一个预测器转变为一个关于 Narrative Dynamics 的可检验科学假说

## P0 任务

从 87 条叙事 JSON 提取 Level 1 硬事实特征：
- Stage（已有 Stage Occupancy，可复用）
- Mutation_Occurred（是否发生变异）
- Institutionalized（是否被主流引用）
- Semantic_Drift_Distance（语义漂移幅度）

然后按月流量权重聚合为时间序列。

## 关键约束

- **AlphaGo 原则**：不硬编码人类理论。数据→表示→解释，不反过来
- 杀混沌轴。不预测值，预测转移
- 不在代码里写死维度数或算法——由数据决定
- Narrative 是动态图，不是向量。Schema 现支持边定义
- 服务器是项目主阵地 (ssh mote@100.118.10.0)，本地仅开发和查看结果
- 文档同步：CLAUDE.md / REQUIREMENTS.md / FRAMEWORK_DESIGN.md 保持一致

## 数据位置

- 叙事 JSON：`data/processed/narratives/` + `data/processed/narratives_from_trends/`
- Stage Occupancy：`data/processed/stage_occupancy.json`
- Google Trends：`data/collector/google_trends_2015_2025.json`
- 外部场：`data/collector/external_field_2015_2025.json`
- 虹姐聊天记录：`E:\Desktop\杂🐟项\` (用于 FR31 参考)
- 用户体系文档：`E:\Desktop\weichen.txt` (纯理论版，背景参考，不进代码)

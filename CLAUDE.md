# MemeticChaos — 人类情感混沌属性建模

> 项目上下文，供 Claude 新会话快速理解。最后更新：2026-06-25

## 项目目标

利用 2020-2025 年中国互联网热梗的由来与演变数据，对人类集体情感的「混沌属性」进行建模。

- **目标不是**预测下一个具体的热梗
- **目标是**识别系统级别的吸引子（Attractors）和相变点（Phase Transitions），对大众在日常生活/网络语境下的普遍喜恶倾向进行结构性预测

## 哲学基础

对齐「微尘哲学」核心元定律：
- 集体情感默认处于熵增背景，混沌是底色，无法被消灭
- 系统在「绝对混沌」（虚无、意义消散）和「绝对秩序」（僵化意识形态）之间振荡
- 热梗可被视为集体情感系统在混沌中「建立局部秩序」的尝试

## 技术栈

- **语言**: Python 3.12
- **环境**: conda `MemeticChaos`
- **核心库**: numpy, scipy, pandas, matplotlib, plotly, networkx, jupyter, jieba, scikit-learn, mesa
- **仓库**: git@github.com:Mote-Xu/MemeticChaos.git

## 项目结构

```
src/
├── data/       # 数据策展 + 轻量爬虫
├── models/     # SIR / ABM / 吸引子检测
├── analysis/   # 生命周期 / 情感 / 相变分析
└── viz/        # 可视化
data/
├── curated/    # 手动策展数据集（git 跟踪）
└── scraped/    # 爬虫补充数据（gitignore）
notebooks/      # Jupyter 探索笔记本
outputs/        # 图表和报告（gitignore）
```

## 当前阶段

Phase 1: 环境搭建 + 数据策展

## 已知问题

- 尚无自动数据采集管道
- 百度指数爬虫需处理反爬

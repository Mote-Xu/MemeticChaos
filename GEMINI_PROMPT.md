# MemeticChaos 项目总结（给外部 AI）

## 项目是什么

计算模因学 × 混沌动力学。用 2020-2025 年中国互联网热梗为集体情感混沌属性建模。

## 当前状态（2026-06-26）

21项功能，11/14 FR完成，24/24测试，13次commit。
GitHub: git@github.com:Mote-Xu/MemeticChaos.git

## 核心产出

**模因相图**（GPT认证的核心IP）：
- 29热梗映射到 R₀ × Chaos Axis 空间
- 5相区 + 2吸引子盆地（100%鲁棒验证）
- 5状态集体情绪状态机 + 4条历史转移路径
- 2021年确认结构性相变（混沌轴 -0.88 漂移）

## 技术架构

```
src/
├── data/    curator.py / bilibili_pipeline.py (Gemini贡献)
├── models/  sir_meme / abm / attractor / individual_calibrator
├── analysis/ lifecycle / sentiment / phase_detect / phase_diagram / backtest
└── viz/     plots.py
```

## 外部审查共识

- GPT：模因相图=不可替代的理论资产；2021相变可能是吸引子重构
- Gemini：贡献了B站字幕拟合管道
- 自审发现：1个真实bug已修复 + R₀公式简化 + 代码去重

## 当前卡点

1. B站22个视频字幕转文字中→接入得第一个真实R₀
2. 仅SIR层有formal tests
3. 跨平台验证+自动采集未开始

## 设计约束

- 混沌≠随机，追求识别确定性结构
- 个体校准器输出后验分布，永不做点断言
- 小真实不可穿透——所有个体推断标置信度+警示

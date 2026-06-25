# 新会话启动提示词

> 复制全文，贴到 Claude Code 新会话窗口。

---

你正在接手 MemeticChaos 项目。在开始任何工作之前，请先阅读以下文件以建立上下文：

1. `CLAUDE.md` — 项目全貌（必读）
2. `PROJECT_BLUEPRINT.md` — 数学框架 + 架构地图
3. `CODE_REVIEW_FINDINGS.md` — 上次审查发现的 1 个真实 bug + GPT/Gemini 反馈

然后确认你能访问 GitHub 仓库：https://github.com/Mote-Xu/MemeticChaos

## 项目一句话

用 2020-2025 年中国互联网热梗数据，为人类集体情感的混沌属性建模。核心产出是模因相图——29 个热梗在 R₀ × Chaos Axis 空间的相态分布。

## 当前状态

21 项功能，24/24 测试通过，15 次 commit。9 个模块可独立运行。

```
已验证的关键发现：
- 2 个吸引子盆地 100% 鲁棒（删 30% 数据仍存在）
- 2021 年混沌轴 -0.88 漂移 → 结构性相变（GPT 认为可能是吸引子重构）
- 留一验证 chaos MAE=0.186，类别准确率 58.6%
- 5 状态集体情绪状态机：建构性解构→攻击宣泄→虚无退却→建构性解构（循环）
```

## 环境

```bash
conda activate MemeticChaos
cd e:\Desktop\MemeticChaos
pytest tests/ -v                           # 24/24 通过
python -m src.models.sir_meme              # SIR 演示
python -m src.analysis.phase_diagram       # 模因相图
python -m src.analysis.backtest            # 回测验证
```

## 当前卡点

**唯一阻塞项**：22 个 B站梗指南视频已下载，正在通过 Video_to_Text 转文字。字幕就绪后跑 `python src/data/bilibili_pipeline.py` 得到第一个真实 R₀。

**已知但可延后**：仅 SIR 层有 formal tests；跨平台验证未开始；百度指数采集未开始。

## 上次会话的待办

1. ⏳ B站字幕转完后接入 bilibili_pipeline.py
2. ABM/Attractor/Calibrator 补测试
3. 个体校准器用真实行为数据验证
4. 跨平台验证（B站 vs 微博 vs 知乎）

## 行为约束

- 混沌 ≠ 随机，追求识别确定性结构
- 个体校准器永不做点断言，输出贝叶斯后验分布
- 小真实不可穿透——所有个体推断标置信度 + 警示
- 讨论中暴露预设，不偷渡价值判断

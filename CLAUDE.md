# MemeticChaos — 人类情感混沌属性建模

> 项目上下文，供 Claude 新会话快速理解。最后更新：2026-06-26

## 项目目标

利用 2020-2025 年中国互联网热梗的由来与演变数据，对人类集体情感的「混沌属性」进行建模。

- **目标不是**预测下一个具体的热梗
- **目标是**识别系统级别的吸引子（Attractors）和相变点（Phase Transitions）
- **扩展目标**：从外部行为信号反推个体混沌属性剖面（贝叶斯后验）
- **核心产出**：模因相图（Meme Phase Diagram）——中国互联网集体情绪相图

## 哲学基础

对齐「微尘哲学」核心元定律：熵增常态 / 与混沌共存 / 小真实不可穿透

## 技术栈

Python 3.12 + conda `MemeticChaos` + numpy/scipy/pandas/networkx/matplotlib
仓库：git@github.com:Mote-Xu/MemeticChaos.git

## 项目结构

```
src/
├── data/
│   ├── curator.py                 # 策展数据管理 + SIR参数估算
│   └── bilibili_pipeline.py       # B站字幕→SIRS-M真实拟合 (Gemini贡献)
├── models/
│   ├── sir_meme.py                # SIR/SIRS/双群体 + 参数拟合 + 混沌熵
│   ├── abm_simulation.py          # 300 Agent 无标度网络仿真
│   ├── attractor.py               # Takens/RQA/Lyapunov/盆地检测
│   └── individual_calibrator.py   # 直接启发式+GA, 贝叶斯后验输出
├── analysis/
│   ├── lifecycle.py               # 生命周期剖面 + 跨类别对比
│   ├── sentiment.py               # 情感弧线分类 (8种类型)
│   ├── phase_detect.py            # 相变检测 (R₀/混沌轴/熵 三维)
│   ├── phase_diagram.py          # ★模因相图 — 核心研究产出
│   └── backtest.py                # 历史回测 + 鲁棒性验证
└── viz/
    └── plots.py                   # 可复用可视化函数库
data/curated/memes_2020_2025.json  # 29个热梗策展
tests/test_sir_model.py            # 24/24 通过
```

## 功能清单（21项）

### 集体层
1. SIR/SIRS/双群体 模因传播模拟
2. 从时间序列反推 β, γ, R₀ (curve_fit)
3. 定性描述 → 参数估算
4. 300-Agent ABM 网络仿真
5. 模因四分类 (脉冲/爆发/长尾/流产)
6. Lyapunov指数 + RQA混沌检测
7. 吸引子盆地识别
8. 生命周期提取 (萌芽/高峰/衰退)
9. 8种情感弧线分类 + 情感熵
10. R₀/混沌轴/熵 三维相变检测
11. ★模因相图 (5相区+2盆地+情绪状态机)
12. 历史回测验证 (时序切分+留一)
13. 吸引子鲁棒性测试
14. B站字幕→SIRS-M拟合管道

### 个体层
15. 行为信号→混沌后验分布
16. 角色类型推断 (builder/injector/normal/lurker)
17. 模因行为预测 (early_adopter/follower/resister)
18. 4种场景模板一键校准

### 数据与验证
19. 29热梗策展数据管理 (筛选/排序/统计)
20. 24单元测试
21. 跨类别统计对比

## 关键发现

- 两个吸引子盆地 100% 鲁棒验证 → 真吸引子
- 2021年混沌轴 -0.88 漂移 + 预测力 0.722→0.389 → 结构性相变（GPT: 可能是吸引子重构）
- 留一验证 chaos MAE=0.186 (2.7x随机基线)
- 解构自嘲类情感熵最高(0.713)，攻击发泄最靠近绝对混沌(-0.62)

## 外部审查

- GPT：模因相图=核心IP，2021相变=论文主线
- Gemini：贡献 bilibili_pipeline.py
- 审查记录：CODE_REVIEW_FINDINGS.md

## 当前卡点

- B站22个视频字幕转文字中 → 接入可得到第一个真实R₀
- ABM/Attractor/Calibrator 缺 formal tests
- 跨平台验证未开始

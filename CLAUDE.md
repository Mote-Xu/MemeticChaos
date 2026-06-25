# MemeticChaos — 人类情感混沌属性建模

> 项目上下文，供 Claude 新会话快速理解。最后更新：2026-06-26

## 项目目标

利用 2020-2025 年中国互联网热梗的由来与演变数据，对人类集体情感的「混沌属性」进行建模。

- **目标不是**预测下一个具体的热梗
- **目标是**识别系统级别的吸引子（Attractors）和相变点（Phase Transitions），对大众在日常生活/网络语境下的普遍喜恶倾向进行结构性预测
- **扩展目标**：从外部行为信号反推个体混沌属性剖面（概率性黑箱映射）

## 哲学基础

对齐「微尘哲学」核心元定律：
- 集体情感默认处于熵增背景，混沌是底色，无法被消灭
- 系统在「绝对混沌」（虚无、意义消散）和「绝对秩序」（僵化意识形态）之间振荡
- 热梗可被视为集体情感系统在混沌中「建立局部秩序」的尝试
- 其他主体的「小真实」是不可穿透的黑箱，但可通过外部行为信号进行概率性结构推断

## 技术栈

- **语言**: Python 3.12
- **环境**: conda `MemeticChaos`
- **核心库**: numpy, scipy, pandas, matplotlib, plotly, networkx, jupyter, jieba, scikit-learn, mesa, pytest
- **仓库**: git@github.com:Mote-Xu/MemeticChaos.git

## 项目结构

```
src/
├── data/
│   └── curator.py                    # 策展数据管理 + SIR参数估算
├── models/
│   ├── sir_meme.py                   # SIR/SIRS/双群体 模因传播模型
│   ├── abm_simulation.py             # 多智能体情感混沌仿真 (300 agents)
│   ├── attractor.py                  # 吸引子检测 (Takens/RQA/Lyapunov)
│   └── individual_calibrator.py      # 个体混沌属性校准器 (黑箱映射)
├── analysis/
│   ├── lifecycle.py                  # 生命周期剖面 + 跨类别对比
│   ├── sentiment.py                  # 情感弧线分类 (5种类型)
│   └── phase_detect.py              # 相变检测 (R₀/混沌轴/熵突变)
└── viz/
    └── plots.py                      # 可复用可视化函数库
data/
├── curated/memes_2020_2025.json     # 29个热梗完整策展 (git 跟踪)
└── scraped/                          # 爬虫补充数据 (gitignore)
notebooks/
├── 01_data_exploration.py            # 数据探索
└── 02_sir_modeling.py               # SIR建模分析
tests/test_sir_model.py              # 24/24 测试通过
outputs/figures/                      # 7张分析图表 (gitignore)
```

## 当前阶段

Phase 2 完成 → 进入 Phase 3 (个体校准 + B站视频数据接入)

## 已实现能力

| 模块 | 功能 | 状态 |
|------|------|:--:|
| SIR 模型 | 标准SIR/SIRS/双群体 + 参数拟合 + 混沌熵分析 | ✅ |
| ABM 仿真 | 300 Agent 无标度网络: 情感传染/回音壁/混沌投放 | ✅ |
| 吸引子检测 | Takens相空间重构/递归图RQA/Lyapunov指数/盆地检测 | ✅ |
| 个体校准器 | 遗传算法从外部行为反推个体混沌参数 (概率性) | ✅ |
| 生命周期 | 29热梗剖面对比 + 异常检测 | ✅ |
| 情感分析 | 5种弧线分类 + 情感熵 + 混沌关联 | ✅ |
| 相变检测 | R₀临界/混沌轴漂移/熵突变 3种相变 | ✅ |
| 策展数据 | 29热梗, 5类别, 2020-2025 | ✅ |

## 关键发现

- R₀=1 是模因传播的临界相变点
- 2021年混沌轴剧烈向负漂移 (-0.88) → 躺平/普信男时代
- 解构自嘲类情感熵最高 (0.713) → 多层反讽结构
- 攻击发泄类最靠近绝对混沌 (均值 -0.62)
- 两个清晰的吸引子盆地: R₀<1 (stable=0.007) + R₀>1.9 (stable=0.952)

## 待办

- [ ] B站视频字幕数据接入 SIR 真实参数拟合
- [ ] 百度指数爬虫
- [ ] 个体校准器真实数据验证
- [ ] 实际用户行为序列输入 个体校准器

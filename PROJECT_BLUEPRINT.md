# MemeticChaos: 项目蓝图

> 最后更新：2026-07-06

## 项目是什么

**刻画 2015-2025 年中国互联网集体叙事系统的相变结构。**

不是预测下一个爆款梗。不是分析某个梗为什么火。
是回答：整个叙事生态系统在十年的时间尺度上，经历了什么样的**结构转变**，以及**为什么会转变**。

核心发现：月度尺度上，叙事状态是近似随机游走（H1b REJECTED）——
点预测在物理上不可行。方向从"预测值"转为**"刻画相区结构 + 检测分岔临界"**。

## 当前在哪

**v4.1 — 四层形式化（State / Observation / Control / Dynamics），系统处于亚稳态。**

127 个月数据已验证：
- 叙事状态存在低维表示（d90=10, H1a ✅）
- 三个物理相区：Origin / Fixation / Peak（GMM + RQA 验证）
- R2 Fixation 是真实结构分离，非聚类 artifact（RQA 零跨相区复发）
- 当前 R2 锁死 37 个月（中位驻留的 2 倍），Inertia=0.77, Sensitivity=0.56 ↑
- 不可逆性来自外部场漂移（Control Manifold），非系统内禀（Time-Reversal 对称）
- R2 内部方差放大 2.31×，PC4/PC5 与 AI/Tech 轴强耦合（r=±0.85）

## 数据

```
历史层 (127 月, 2015-01 → 2025-12):
  51 关键词 Google Trends + 57 条叙事 JSON
  → Level 1 硬事实 + Level 2 10 维 Narrative State

实时层 (每小时, 2026-06 至今):
  微博 50 + 百度 50 + 知乎 30 = 130 条/小时
  → v2.0 全量 384 维 embedding → 日级语义聚合
  ⚠ 日级管道已建，尚未接入月度分析链（2026 年数据悬空）
```

## 四层本体论

```
State      — x(t) ∈ M(z(t)), 叙事状态活在控制流形上
Observation — y(t) = Proj[x(t)] + noise, 从不声称是真相
Control    — z(t) 定义可达集合边界, 不直接推动状态
Dynamics   — 状态演化 + 控制形变 (MS-AR, within-regime)
```

详见 `FORMALISM.md`。

## FR31: 个体层顾问

战略对弈伙伴 — 用户在情境中给出判断，系统给出独立判断，两者对质。
从集体层结构性规律推导个体层决策。
`engine.py` + `persona.py` (五态输出) + `metrics.py` (四指标: Inertia/Resilience/Sensitivity/Position)。
已通过 Stella 企微管道验证。

## 部署

- **服务器** (mote-home): 7×24 采集 + Dashboard (chaos.mote-pal.xyz)
- **本地**: Google Trends 通过飞鸟代理 + 精细建模

## 关键未解决问题

1. **数据管道闭环** (P0): 日级 embedding → 月级语义状态 → 续接 2026 年分析链
2. **信号质量** (开放性): 采集的数据测量的是"人群对现实结构的叙事化反应"而非现实结构本身。叙事化倾向是要测量的对象，不是要清洗的噪声。
3. **R2 内部动力学** (P0): 势阱形变建模 p(x|R2, z1)

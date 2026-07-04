# MemeticChaos 形式化框架 v4.1

> 四层统一: State / Observation / Control / Dynamics
> 最后更新: 2026-07-05

---

## 概述

MemeticChaos 是一个**非平衡统计物理系统**。它描述互联网集体叙事在外部控制场驱动下,
在一个低维流形上的密度演化与相变行为。

本章定义系统的四个本体论层次。每一个层次有明确的数学对象、测量方式和更新规则。
新模块不再需要新概念——只需归入某一层。

```
                    ┌──────────────────────────┐
                    │   Control z(t)           │  ← 外部场 → 控制流形
                    │   定义 M(z) 的几何形态     │
                    └──────────┬───────────────┘
                               │ 形变
                    ┌──────────▼───────────────┐
                    │   State x(t), S(t)       │  ← 叙事状态 + 相区身份
                    │   在 M(z) 上演化          │
                    └──────────┬───────────────┘
                               │ 投影
                    ┌──────────▼───────────────┐
                    │   Observation y(t)       │  ← 可测量量
                    │   状态的不完全投影         │
                    └──────────────────────────┘

                    Dynamics: 层间演化规则
```

---

## 1. State (状态)

### 1.1 定义

系统的**不可直接观测的内部状态**, 是动力学的载体。

| 对象 | 符号 | 维度 | 含义 |
|------|------|:--:|------|
| Narrative State | **x(t)** | 10 | PCA 降维后的叙事隐状态 |
| Regime Identity | **S(t)** | 1 | 相区标签 ∈ {Origin, Peak, Fixation} |
| Stage Occupancy | **σ(t)** | 5 | 五阶段占比 (origin/emergence/peak/controversy/fixation) |

### 1.2 相区结构

GMM + BIC 识别出 4 个观测簇, RQA 将其合并为 3 个物理相区:

| 物理相区 | GMM 标签 | 主导阶段 | 时期 | 自持概率 |
|----------|:--:|------|------|:--:|
| **Origin** | R1 + R3 | origin | 2015-2022 | ~87% |
| **Fixation** | R2 | fixation | 2022.12→ | **97.3%** |
| **Peak** | R0 | peak | 罕见 | 50% |

R1 和 R3 是同一 Origin manifold 在不同 u(t) 条件下的几何形变
(RQA 确认: 10 对跨期复发, 最小距离 1.18). Fixation (R2) 是真实的结构分离
(RQA: 零跨相区复发, 最小距离 2.24-3.38).

### 1.3 关键性质

- **H1a ✅**: 低维表示存在 (18→10, 90% 方差)
- **H1b ❌**: x(t+1) 不可点预测 (VARX R²=-0.32 < lag-1 R²=+0.44)
- **H1c ⚠️**: 状态有惯性 (lag-1 R²=0.44), 但外生变量不帮正向忙
- **弱不可逆**: RQA 确认 R2 真实分离, Time-Reversal 确认无内禀时间箭头
- **不可逆性来源**: Control 层 z(t) 的单调漂移, 非 State 层内禀动力学

### 1.4 更新方式

x(t) 每月由 Level 1 硬事实 + 外部场 PCA + 注意力结构 → PCA 降维生成。
S(t) 由 GMM 对 x(t) 聚类分配。不预测 x(t+1) 的点值。

---

## 2. Observation (观测)

### 2.1 定义

State 在测量空间中的**不完全投影**。观测不改变 State, 但观测的时序结构
揭示了 State 的统计属性。

| 对象 | 符号 | 维度 | 来源 |
|------|------|:--:|------|
| Level 1 硬事实 | — | 4 | Stage/Mutation/Inst/Drift 月度聚合 |
| 注意力集中度 | HHI(t) | 1 | Google Trends 关键词份额 |
| 注意力多样性 | Ent(t) | 1 | 同上 |
| 外部场原始值 | u_raw(t) | 51 | Google Trends 关键词月度值 |
| FR31 四指标 | I,R,P,S | 4 | 从 x(t) 和 σ(t) 计算 |
| 个体投影 | P(meme\|text) | 57 | persona.py embedding 余弦分布 |
| 微观闪洪 | burst(t) | 1 | 小时级 scraper → 日度波动率 |

### 2.2 关键性质

- 观测是 State 的**有噪投影**, 不是 State 本身
- Attention 是**响应变量**, 不是序参量 (order parameter)。
  它反映系统的注意力分配, 而非决定相区归属
- Persona 输出是**概率分布**, 不是点定位。不确定性是合法输出
- 微观闪洪是 Control→State 传导链上的快变量前兆

### 2.3 更新方式

- Level 1 + HHI + Ent: 月度 (Google Trends)
- FR31 四指标: 从 x(t) 实时计算, 每小时 cron 刷新
- Persona: 按需 (用户输入时触发)
- 微观闪洪: 每小时 scraper → 日度聚合

---

## 3. Control (控制)

### 3.1 定义

**不进入 State 演化方程右侧**, 但**定义 State 可达集合的几何边界**。
Control 改变的是势能景观 V(x; z), 而非直接推动 x。

```
正确:  x(t) ∈ M(z(t))
错误:  x(t+1) = F(x(t), u(t))
```

### 3.2 控制流形

| 对象 | 符号 | 维度 | 含义 |
|------|------|:--:|------|
| 外部场 | **u(t)** | 51 | Google Trends 关键词原始值 |
| 控制流形 | **z(t)** | 3 | u(t) → PCA(8) → Diffusion Map(3) |

### 3.3 关键发现

- **z₁ 轴**: AI/Tech 话语主导 (AI r=-0.68, ChatGPT r=+0.54)
  - 范围 [-0.62, 0.003] — 十年来几乎单向漂移
  - **因果性待验证**: 可能与时间高度共线 (corr(z₁, time) 待测)
- **z₂ 轴**: B站/二次元/俄罗斯等多元话题
- **R2 不在 z(t) 极端区间** — 推翻 "u(t) 极端→R2" 假说
  - 正确机制: u(t) 沿 z₁ 单调漂移 → 系统经过 R2-favoring 盆地 → 被困
- **逃逸条件**: z(t) 方向反转 OR 新控制轴出现 (非 AI/Tech 共线)

### 3.4 待验证

- Counterfactual test: 固定 time=2024, 人为令 AI trend 下降,
  观察 z(t) 是否移动 (区分因果 driver vs time proxy)

---

## 4. Dynamics (动力学)

### 4.1 定义

系统**状态如何随时间演化**——以及控制如何形变这种演化。
Dynamics 是唯一可以包含时间导数的层次。

### 4.2 已知的动力学规律

**宏观 — Regime 转移**:
```
P(S(t+1) = j | S(t) = i, z(t)) = f(z(t))
```
- Origin → Fixation: P ≈ 5.1% (受 z₁ 调制)
- Fixation → Origin: P ≈ 2.7% (极难, 需 z₁ 大幅逆转)
- Fixation 自持: P = 97.3% (当前 z 配置下的吸收态)

**介观 — State 漂移**:
```
x(t+1) = x(t) + ε(t),  ε(t) ~ D(z(t))
```
- 月度尺度近似随机游走 (H1b 否定)
- 扩散核 D(z) 由 Control 调制 (方差, 非均值)
- 下一步: GARCH / MS-AR 建模方差

**微观 — 闪洪**:
- 小时级 scraper → 日度波动率 → Sensitivity 微观修正因子
- 当宏观 Sensitivity > 0.5 且微观 burst 连续上升 → 相变可能性上调

### 4.3 动力学层次对应

| 尺度 | 时间分辨率 | 建模对象 | 方法 |
|------|:--:|------|------|
| 宏观 | 月-年 | Regime 转移概率 | MS-AR (待建) |
| 介观 | 月 | State 漂移 + 扩散 | GARCH/随机游走 |
| 微观 | 时-日 | 注意力波动 | 闪洪检测 (已建) |

### 4.4 待建

- MS-AR: Regime-dependent 转移概率, z(t) 作为外生调制
- GARCH: u(t) → 方差, 非均值
- Fokker-Planck: 密度演化的连续极限 (需更多样本)

---

## 5. 层次间交互规则

```
┌─────────────────────────────────────────────────────┐
│                                                      │
│  Control z(t)                                        │
│    │                                                 │
│    │ 形变 M 的几何                                     │
│    ▼                                                 │
│  State x(t), S(t)                                    │
│    │                                                 │
│    │ 投影                                             │
│    ▼                                                 │
│  Observation y(t)                                    │
│    │                                                 │
│    │ 经验反馈 (y 影响 z 的更新, 但非闭环)               │
│    └──────────────────────────────────→  Control z   │
│                                                      │
│  Dynamics: S(t)→S(t+1), x(t)→x(t+1)                 │
│    受 Control 调制, 由 Observation 校准               │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 5.1 不该跨越的边界

- ❌ Observation 直接驱动 State (Persona 输出不是心理学诊断)
- ❌ State 精确预测 (H1b 已否定)
- ❌ Control 被视为因果变量 (可能是 time proxy)
- ✅ Observation 校准 Dynamics 参数
- ✅ Control 定义 Dynamics 的可行区域
- ✅ State 是 Observation 和 Dynamics 之间的中介

---

## 6. 工程映射

每个已有模块归入对应层次:

| 层次 | 模块 |
|------|------|
| **State** | `representation_state.json`, `regime_map.json`, `irreversibility_results.json` |
| **Observation** | `level1_hard_facts.json`, `metrics.py` (四指标), `persona.py` (个体投影), `micro_burst.json` |
| **Control** | `external_field_2015_2025.json`, `control_manifold.json` |
| **Dynamics** | `regime_detector.py` (转移矩阵), `representation_learning.py` (VARX), `irreversibility_test.py` |

API 端点:
- `/api/fr31` → Observation 层 (宏观)
- `/api/fr31/persona` → Observation 层 (个体投影)
- `/api/fr31/stella` → Observation → NL (Stella 消纳)

---

## 7. 开放问题

1. **AI/Tech 是 Control driver 还是 time proxy?** Counterfactual test 待做.
2. **R2 逃逸的充分条件?** z(t) 方向反转的量级阈值待定.
3. **λ₂ 谱预警?** 需等 scraper 积累真实动态边后再上, 当前人工边不可靠.
4. **D(z) 的具体形式?** MS-AR 或 GARCH 待选.
5. **个体 ↔ 集体映射的统计校准?** 虹姐数据可作为外部验证集.

---

*本框架定义了 MemeticChaos 的本体论。新模块不应引入新层次——只需归入 State / Observation / Control / Dynamics 之一.*

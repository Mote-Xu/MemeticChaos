# FR19 v4.0 — 2026-07-04 双 AI 联合审议最终结论

## 核心科学假说

FR19 不再是一个预测器。它是关于互联网集体叙事动力学的**可检验科学假说**。

### 逐级可检验的假说链

**H1**：存在一个能够较好描述互联网叙事演化动力学规律的低维状态表示。

**H1a**：该表示具有低维性（降维后信息损失可控）。
**H1b**：该表示具有动力学连续性（t 期状态对 t+1 期有预测力）。
**H1c**：未来的 Narrative 演化主要由当前 Narrative State 决定，而非历史或外部随机扰动。

逐级验证——H1a 被否定不影响 H1 成立（可能不是低维但仍是状态表示）。
H1b 被否定说明 Narrative 本身可能没有自组织惯性。
这样每步失败都有明确归因，不会"全盘否定但不知道为什么"。

## 修正后的世界观

### Attention 是观测 + 反馈闭环，不是纯摄像头

Google Trends / 微博热搜一旦飙升，就会成为显性环境，直接改变网民感知，
反向加速或阻断叙事的制度化收编或变异。

```
 u(t) [外部环境 Shock] ────> ⚡ 扰动
                              │
                              ▼
 ╭───────────────────> x(t) [叙事生态隐状态/Graph]
 │                            │
 │                            ▼
 │                     y(t) [观测注意力/Attention]
 └──── 🔄 反馈 ─────────┘
```

### Narrative 是动态图（Dynamic Graph），不是向量

"躺平 → 摆烂 → 发疯文学"这条链路在向量里无法表达——
它们是节点之间的有向边。

叙事 JSON 天然支持边定义，现在就可以写进 Schema：
```json
{
  "id": "躺平",
  "stage": "Fixation",
  "edges": [
    {"target": "摆烂", "type": "derivation", "strength": 0.68},
    {"target": "发疯文学", "type": "derivation", "strength": 0.42},
    {"target": "Citywalk", "type": "competition", "strength": -0.30}
  ]
}
```

当前工程替代：用邻接矩阵主特征向量作为状态表示——
既保留拓扑关系又能向量注入预测器。以后可以直接升级到图神经网络。

## 三层渐进式特征工程（AlphaGo 原则）

### Level 1：无损事实统计（仅提取硬数据，不在代码里写理论）
- `Stage`：5 维阶段计数
- `Mutation_Occurred`：是否发生变异，布尔值
- `Institutionalized`：是否被主流引用，布尔值
- `Semantic_Drift_Distance`：原义→现义的语义向量欧氏距离

### Level 2：机器自发表示学习（Representation Learning）
将 Level 1 的基础事实 + 月度流量截面，用 Representation Learning 降维。
具体算法（PCA/UMAP/Autoencoder/VAE/Graph Embedding）由数据决定，不提前指定。
机器盲聚类的 3 类叙事（ARI=0.27）直接作为特征参与。

### Level 3：后验结构阐释
训练完成后分析特征载荷，看看哪一维对应"性别冲突"、哪一维对应"虚无退却"。
**这才是科学——以前因果和顺序颠倒了。**

## Narrative Dynamics（叙事动力系统）

```
x(t+1) = F(x(t), u(t), y(t))
```

不是 Ax+Bu。F 是待学习的动力系统——内部自组织 + 外部场扰动 + 注意力反馈。
第一阶段以 VARX/Ridge 线性模型作为基线；若不能解释叙事演化，
再逐步尝试非线性模型（Gradient Boosting / MLP / Neural ODE）。
F 的具体形式和 x(t) 的维度都不提前写死——由 Level 2 表示学习结果决定。

## FR31 对接

FR19 的核心产出不是预测值——是对 Narrative Dynamics 的理解。
FR31 只需要向 FR19 查询三个指标：
1. **Inertia（惯性）**：当前大盘叙事在多大程度上具有自持性（"踩刹车也停不下来"）
2. **Resilience（恢复力）**：外部冲击后系统回到原吸引子的速度
3. **Position（图位置）**：对方个体状态在叙事演化图中的拓扑位置

## 当前已建成（可重用）

- Stage Occupancy 127 月 × 5 维
- 87 条叙事 JSON（含 spread_phases、mutations、semantic_drift、social_context）
- 叙事聚类 3 类（ARI=0.27）→ 可直接作为 Level 2 特征
- 外部场 PCA 8 维
- 注意力结构（HHI、Entropy、类别分布）127 月

## 待建（按优先级）

| 优先级 | 任务 |
|:--:|------|
| P0 | Level 1：从 87 条 JSON 提取 4 类硬事实 |
| P0 | 月度截面投影：按流量权重聚合为时间序列 |
| P1 | Level 2：Autoencoder/UMAP 学习隐状态 |
| P1 | H1 验证：用 VARX 框架检验 x(t+1) = F(x(t), y(t), u(t)) |
| P2 | Schema 3.0：图动力学前置支持（边定义 + 邻接矩阵） |
| P2 | FR31 接口：Inertia / Resilience / Position 三指标 |

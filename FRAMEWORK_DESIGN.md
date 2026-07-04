# MemeticChaos 框架设计 v4.0

> Narrative Dynamics — 可检验的科学假说。
> 最后更新：2026-07-04（双 AI 联合审议 + GPT 修正）

## 核心科学假说 H1

FR19 不再是预测器。它是关于互联网集体叙事动力学的可检验科学假说。

**H1**：存在一个能够描述互联网叙事演化动力学规律的低维状态表示。
- H1a：低维性。降维后信息损失可控。
- H1b：动力学连续性。t 期对 t+1 期有预测力。
- H1c：当前状态主导。未来演化主要由当前状态而非历史或外部噪声决定。

## 世界观

### 三类变量

```
External Field u(t) — 经济/政策/平台/明星（扰动项）
      ↓                ↓
Narrative State x(t) — 动态图，系统核心
      ↓                ↑
Attention y(t) — 观测 + 反馈闭环（不是纯摄像头）
```

### 动力学方程

```
x(t+1) = F(x(t), u(t), y(t))
```
F 的具体形式由数据决定。第一阶段 VARX/Ridge 作基线，不成立再试非线性。

## 三层渐进式特征工程（AlphaGo 原则）

- **Level 1** ✅ (2026-07-04)：已提取硬事实 → `data/processed/level1_hard_facts.json`。127 月 × 4 特征序列 (Stage/Mutation/Inst/Drift)。脚本: `src/data/narrative_hard_facts.py`。
- **Level 2** 🔜：Representation Learning。算法（PCA/UMAP/Autoencoder/VAE/Graph Embedding）由数据决定。
- **Level 3**：后验解释。训练完成后分析特征载荷。

## Narrative 是动态图

节点（梗）+ 有向边（变异/竞争/替代/融合）。Schema 现支持边定义：
```json
{"edges": [{"target": "摆烂", "type": "derivation", "strength": 0.68}]}
```
工程上用邻接矩阵主特征向量衔接向量预测器。后续可升级到图神经网络。

## 验证层 (不变)

SIR / 相图 / 吸引子盆地 / 状态机 —— 验证动力学合理性。

## FR19 → FR31

FR19 产出 Narrative Dynamics 理解 → FR31 查询三个控制论指标：
Inertia / Resilience / Position。不是预测值——是对系统的结构性认知。

详见 `DESIGN_DISCUSSION.md`。

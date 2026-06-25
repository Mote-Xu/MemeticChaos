"""
Notebook 1: 策展数据集探索
作为 Jupyter Notebook 使用：# %%
对应: notebooks/01_curated_dataset.ipynb

目标:
1. 加载并验证策展数据集
2. 统计概览：类别分布、年份分布、平台分布
3. 混沌轴分析：集体情感在绝对混沌与绝对秩序之间的分布
4. R₀ 分布与类别关联
"""

# %% [markdown]
# # MemeticChaos — 策展数据集探索
#
# 本节目标：理解 2020-2025 年中国互联网热梗数据集的整体结构。
# 核心问题：
# - 热梗在「绝对混沌 ↔ 绝对秩序」轴上是如何分布的？
# - 不同类别的热梗具有怎样不同的传播特征？
# - 集体情感系统的熵是如何随时间变化的？

# %% imports
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from src.data.curator import MemeCurator

# %% 加载数据
curator = MemeCurator()
print(curator.stats_report())

# %% 构建 DataFrame
records = []
for m in curator.memes:
    records.append({
        "name": m.name,
        "year": m.year,
        "peak_year": m.peak_year,
        "category": m.category,
        "chaos_position": m.chaos_position,
        "R0_est": m.estimated_R0,
        "platforms": ", ".join(m.source_platforms[:3]),
        "duration_months": m.lifecycle.get("duration_months", "?"),
        "status": m.lifecycle.get("status", "?"),
    })

df = pd.DataFrame(records)
print(f"Dataset: {len(df)} memes")
print(f"Year range: {df['year'].min()} – {df['year'].max()}")
print(f"\nCategory distribution:")
print(df["category"].value_counts())
print(f"\nChaos position stats:")
print(df["chaos_position"].describe())
print(f"\nR₀ stats:")
print(df["R0_est"].describe())

# %% 类别分布可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 类别柱状图
cat_counts = df["category"].value_counts()
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
axes[0].barh(cat_counts.index, cat_counts.values, color=colors[:len(cat_counts)])
axes[0].set_xlabel("count")
axes[0].set_title("Meme Categories Distribution")

# 年份分布
year_counts = df["year"].value_counts().sort_index()
axes[1].bar(year_counts.index.astype(str), year_counts.values, color="#3498db", alpha=0.7)
axes[1].set_xlabel("year")
axes[1].set_ylabel("count")
axes[1].set_title("Memes by Year")
plt.tight_layout()
plt.savefig("outputs/figures/category_year_distribution.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/category_year_distribution.png")

# %% 混沌轴分析
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 混沌轴直方图
axes[0].hist(df["chaos_position"], bins=12, color="#8e44ad", alpha=0.7, edgecolor="white")
axes[0].axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="neutral")
axes[0].axvline(x=-1, color="#e74c3c", linestyle=":", alpha=0.3, label="absolute chaos")
axes[0].axvline(x=1, color="#2ecc71", linestyle=":", alpha=0.3, label="absolute order")
axes[0].set_xlabel("Chaos ← → Order")
axes[0].set_ylabel("count")
axes[0].set_title("Meme Distribution on Chaos-Order Axis")
axes[0].legend(fontsize=8)

# 按类别的混沌轴分布
cat_chaos = df.groupby("category")["chaos_position"].agg(["mean", "std", "count"])
print("\nChaos position by category:")
print(cat_chaos.sort_values("mean"))

cat_order = cat_chaos.sort_values("mean").index
y_pos = range(len(cat_order))
axes[1].barh(y_pos, cat_chaos.loc[cat_order, "mean"],
             xerr=cat_chaos.loc[cat_order, "std"],
             color=[colors[i] for i in range(len(cat_order))],
             alpha=0.7, capsize=3)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(cat_order)
axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
axes[1].set_xlabel("Chaos ← → Order")
axes[1].set_title("Mean Chaos Position by Category (±1σ)")
plt.tight_layout()
plt.savefig("outputs/figures/chaos_axis_analysis.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/chaos_axis_analysis.png")

# %% R₀ vs Chaos 散点图
fig, ax = plt.subplots(figsize=(10, 7))

category_colors = {"解构自嘲": "#3498db", "攻击发泄": "#e74c3c",
                   "虚无退却": "#95a5a6", "身份认同": "#2ecc71",
                   "纯粹娱乐": "#f39c12"}

for cat in df["category"].unique():
    subset = df[df["category"] == cat]
    ax.scatter(subset["chaos_position"], subset["R0_est"],
               c=category_colors.get(cat, "gray"), label=cat,
               s=100, alpha=0.7, edgecolors="white", linewidth=0.5)

# Annotate selected memes
for _, row in df.iterrows():
    if row["R0_est"] >= 4.5 or abs(row["chaos_position"]) >= 0.6:
        ax.annotate(row["name"], (row["chaos_position"], row["R0_est"]),
                   fontsize=7, alpha=0.8,
                   xytext=(5, 5), textcoords="offset points")

ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.3, label="R₀=1 (threshold)")
ax.set_xlabel("Chaos ← → Order")
ax.set_ylabel("Estimated R₀")
ax.set_title("Meme Landscape: R₀ vs Chaos-Order Position")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("outputs/figures/R0_vs_chaos.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/R0_vs_chaos.png")

# %% 关键洞察
print("""
=== 关键洞察 ===

1. 混沌轴均值 ≈ 0（中性），但方差大：集体情感系统在绝对混沌和绝对秩序
   之间持续振荡，没有稳定的平衡点。

2. 攻击发泄类（普信男、XX刺客、建议专家不要建议）最靠近绝对混沌：
   这类热梗本质上是「混沌投放」——降低了系统建立健康秩序的可能性。

3. 身份认同类（i人/e人、小镇做题家、内卷）偏秩序：
   它们为模糊的感受提供了命名和分析框架——典型的负熵行为。

4. 纯粹娱乐类（科目三、鸡你太美）接近中性：
   它们几乎不携带社会情绪负载——是系统的「无害休息」。

5. 「后浪」案例独特：它是一个从强秩序（官方叙事）被解构为混沌的
   经典案例——背离真实 → 扩散性混沌投放。
""")

# %% 备选：用 SIR 模型估算参数
print("\n--- SIR Parameter Estimation from Lifecycle ---")
estimations = curator.to_sir_estimation()
est_df = pd.DataFrame(estimations)
print(est_df[["name", "category", "chaos_position", "R0_qualitative", "R0_estimated"]].head(10).to_string())

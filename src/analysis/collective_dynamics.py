"""
集体情感动力学 — 从 24 梗真实注意力数据重构系统级相空间轨迹

这才是 MemeticChaos 最初设计的核心产出：
不是分析单个梗，是观测整个集体情感系统如何随时间演化。

输入: Google Trends 月度数据 (732 点, 24 梗, 2015-2025)
输出: 系统级相图 + 集体情绪轨迹

用法:
    python src/analysis/collective_dynamics.py
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path("data")

# 5 类别的混沌轴位置（人工标注，来自策展数据）
CATEGORY_CHAOS = {
    "解构自嘲": -0.33,
    "攻击发泄": -0.62,
    "虚无退却": -0.59,
    "身份认同": +0.34,
    "纯粹娱乐": +0.19,
}

# Google Trends keyword → curated meme (name, category)
TREND_TO_MEME = {
    "打工人": ("打工人", "解构自嘲"),
    "内卷": ("内卷 / 卷", "身份认同"),
    "躺平": ("躺平", "虚无退却"),
    "普信男": ("普信男", "攻击发泄"),
    "小镇做题家": ("小镇做题家", "身份认同"),
    "摆烂": ("摆烂", "虚无退却"),
    "润": ("润", "虚无退却"),
    "吗喽": ("吗喽", "解构自嘲"),
    "鼠鼠": ("鼠鼠", "解构自嘲"),
    "牛马": ("牛马", "解构自嘲"),
    "i人 e人": ("i人/e人", "身份认同"),
    "遥遥领先 华为": ("遥遥领先", "纯粹娱乐"),
    "孔乙己的长衫": ("孔乙己的长衫", "身份认同"),
    "精神状态": ("精神状态良好", "解构自嘲"),
    "雪糕刺客": ("XX刺客", "攻击发泄"),
    "科目三": ("科目三", "纯粹娱乐"),
    "鸡你太美": ("鸡你太美", "纯粹娱乐"),
    "后浪": ("后浪", "身份认同"),
    "情绪价值": ("情绪价值", "身份认同"),
    "原生家庭": ("原生家庭", "身份认同"),
}


def load_trends() -> dict:
    with open(DATA_DIR / "collector/google_trends_2015_2025.json", "r", encoding="utf-8") as f:
        return json.load(f).get("memes", {})


def compute_collective_trajectory(trends: dict):
    """从所有梗的注意力曲线计算系统级集体情绪轨迹。

    对每个月，计算:
    - 集体混沌轴位置 (注意力加权平均)
    - 集体 R₀ (总注意力)
    - 主导类别 (哪个类别注意力最高)
    """
    monthly = defaultdict(lambda: {"chaos_sum": 0.0, "weight_sum": 0.0,
                                     "cat_weights": defaultdict(float)})

    for trend_name, (meme_name, category) in TREND_TO_MEME.items():
        if trend_name not in trends:
            continue
        chaos = CATEGORY_CHAOS.get(category, 0.0)
        for month_str, value in trends[trend_name].items():
            month = month_str[:7]  # YYYY-MM
            w = float(value)
            monthly[month]["chaos_sum"] += chaos * w
            monthly[month]["weight_sum"] += w
            monthly[month]["cat_weights"][category] += w

    # 构建时间序列
    sorted_months = sorted(monthly.keys())
    chaos_axis = []
    total_attention = []
    dominant_cat = []

    for m in sorted_months:
        d = monthly[m]
        if d["weight_sum"] > 0:
            chaos_axis.append(d["chaos_sum"] / d["weight_sum"])
        else:
            chaos_axis.append(0.0)
        total_attention.append(d["weight_sum"])
        # 主导类别
        if d["cat_weights"]:
            dominant_cat.append(max(d["cat_weights"], key=d["cat_weights"].get))
        else:
            dominant_cat.append("none")

    return {
        "months": sorted_months,
        "chaos_axis": np.array(chaos_axis),
        "total_attention": np.array(total_attention),
        "dominant_category": dominant_cat,
    }


def detect_phase_transitions(traj: dict) -> list[dict]:
    """检测系统级相变点。"""
    chaos = traj["chaos_axis"]
    months = traj["months"]
    transitions = []

    # 检测混沌轴显著漂移 (连续6个月窗口的均值变化 > 0.15)
    window = 6
    for i in range(window, len(chaos) - window):
        before = np.mean(chaos[i - window:i])
        after = np.mean(chaos[i:i + window])
        shift = after - before
        if abs(shift) > 0.12:
            # 去重: 不与上一个相变点重叠
            if not transitions or i - transitions[-1]["index"] > window:
                transitions.append({
                    "month": months[i],
                    "index": i,
                    "shift": float(shift),
                    "direction": "向混沌" if shift < 0 else "向秩序",
                    "magnitude": abs(float(shift)),
                })

    return transitions


def plot_collective_phase_diagram(traj: dict, transitions: list[dict],
                                   save_path: str = None):
    """绘制集体情感相图 — 这才是核心产出。"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    months = traj["months"]
    chaos = traj["chaos_axis"]
    attention = traj["total_attention"]
    n = len(months)

    # ── 1. 集体混沌轴时间序列 (主图) ──
    ax1 = axes[0, 0]
    ax1.plot(range(n), chaos, linewidth=1.5, color="#333333")
    ax1.fill_between(range(n), chaos, 0, where=(chaos > 0),
                     color="#2196F3", alpha=0.15, label="偏秩序")
    ax1.fill_between(range(n), chaos, 0, where=(chaos <= 0),
                     color="#F44336", alpha=0.15, label="偏混沌")
    ax1.axhline(y=0, color="#999999", linestyle=":", alpha=0.5)

    # 标记相变点
    for t in transitions:
        color = "#F44336" if t["direction"] == "向混沌" else "#2196F3"
        ax1.axvline(x=t["index"], color=color, alpha=0.3, linewidth=1)
        ax1.annotate(f"{t['shift']:+.2f}", xy=(t["index"], chaos[t["index"]]),
                    fontsize=7, color=color, fontweight="bold")

    # 标注关键年份
    for year in range(2015, 2026):
        for i, m in enumerate(months):
            if m.startswith(f"{year}-01"):
                ax1.axvline(x=i, color="#CCCCCC", alpha=0.3, linewidth=0.5)
                ax1.text(i, ax1.get_ylim()[1] * 0.95, str(year), fontsize=7, color="#999")
                break

    ax1.set_title("Collective Chaos Axis  (attention-weighted)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Chaos Axis\n(— order, + chaos)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # ── 2. 总注意力 ──
    ax2 = axes[0, 1]
    ax2.fill_between(range(n), attention, alpha=0.5, color="#FF9800")
    ax2.plot(range(n), attention, linewidth=1, color="#FF9800")
    ax2.set_title("Total Meme Attention (Google Trends)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Sum of Interest", fontsize=10)
    ax2.grid(True, alpha=0.2)

    # ── 3. 相空间轨迹 (Chaos × Attention) ──
    ax3 = axes[1, 0]
    # 按年份着色
    year_starts = []
    for year in range(2015, 2026):
        for i, m in enumerate(months):
            if m.startswith(f"{year}-01"):
                year_starts.append((year, i))
                break
    for yi, (year, start) in enumerate(year_starts[:-1]):
        end = year_starts[yi + 1][1]
        color = plt.cm.viridis(yi / len(year_starts))
        ax3.plot(chaos[start:end], attention[start:end] / max(1, attention[start:end].max()),
                linewidth=1.5, color=color, alpha=0.8, label=str(year))
    ax3.set_xlabel("Collective Chaos Axis", fontsize=10)
    ax3.set_ylabel("Normalized Attention", fontsize=10)
    ax3.set_title("Phase Space Trajectory  2015-2025", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=7, ncol=3, loc="upper left")
    ax3.grid(True, alpha=0.2)

    # ── 4. 类别主导时间线 ──
    ax4 = axes[1, 1]
    cat_colors = {
        "解构自嘲": "#4CAF50", "攻击发泄": "#F44336",
        "虚无退却": "#9C27B0", "身份认同": "#2196F3",
        "纯粹娱乐": "#FF9800",
    }
    # 计算每月主导类别
    cat_sequence = traj["dominant_category"]
    unique_cats = list(set(cat_sequence))
    y_positions = {cat: i for i, cat in enumerate(unique_cats)}
    colors = [cat_colors.get(c, "#999999") for c in cat_sequence]
    ax4.scatter(range(n), [y_positions[c] for c in cat_sequence],
               c=colors, s=3, alpha=0.6)
    ax4.set_yticks(list(y_positions.values()))
    ax4.set_yticklabels(list(y_positions.keys()), fontsize=8)
    ax4.set_title("Dominant Meme Category Over Time", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Month (2015-2025)", fontsize=10)
    ax4.grid(True, alpha=0.2, axis="x")

    fig.suptitle("Chinese Internet Collective Emotional Dynamics  2015-2025",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path is None:
        save_path = str(DATA_DIR / "processed/collective_phase_diagram.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[保存] 集体情感相图 → {save_path}")
    plt.close()


def print_report(traj: dict, transitions: list[dict]):
    """打印集体情感系统报告。"""
    print(f"\n{'='*70}")
    print(f"  中国互联网集体情感系统 — 2015-2025 年度报告")
    print(f"{'='*70}")

    chaos = traj["chaos_axis"]
    attention = traj["total_attention"]
    months = traj["months"]

    # 整体趋势
    first_half = chaos[:len(chaos)//2].mean()
    second_half = chaos[len(chaos)//2:].mean()
    trend = "向秩序" if second_half > first_half else "向混沌"

    print(f"\n  📊 整体趋势:")
    print(f"     2015-2020 平均混沌轴: {first_half:+.3f}")
    print(f"     2020-2025 平均混沌轴: {second_half:+.3f}")
    print(f"     十年趋势: {trend} (漂移 {second_half - first_half:+.3f})")

    # 峰值事件
    peak_idx = int(np.argmax(attention))
    print(f"\n  🔥 注意力峰值:")
    print(f"     {months[peak_idx]} (总关注度 {attention[peak_idx]:.0f})")

    # 混沌极端
    min_idx = int(np.argmin(chaos))
    max_idx = int(np.argmax(chaos))
    print(f"\n  🎯 混沌极端:")
    print(f"     最秩序: {months[max_idx]} ({chaos[max_idx]:+.3f})")
    print(f"     最混沌: {months[min_idx]} ({chaos[min_idx]:+.3f})")

    # 相变
    print(f"\n  ⚡ 检测到 {len(transitions)} 次系统级相变:")
    for t in transitions:
        direction_emoji = "🔴→" if t["direction"] == "向混沌" else "🔵→"
        print(f"     {t['month']} {direction_emoji} {t['direction']} "
              f"(漂移 {t['shift']:+.3f})")

    # 各年份主导类别
    print(f"\n  📅 年度主导类别:")
    yearly = defaultdict(lambda: defaultdict(float))
    for m, cat in zip(months, traj["dominant_category"]):
        yearly[m[:4]][cat] += 1
    for year in sorted(yearly.keys()):
        dom = max(yearly[year], key=yearly[year].get)
        months_dom = yearly[year][dom]
        print(f"     {year}: {dom} ({months_dom}/12 个月)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("MemeticChaos — 集体情感动力学分析")
    print("=" * 70)

    # 加载数据
    trends = load_trends()
    print(f"\n[数据] {len(trends)} 梗, 2015-2025")

    # 计算系统级轨迹
    traj = compute_collective_trajectory(trends)
    print(f"[轨迹] {len(traj['months'])} 个月")

    # 检测相变
    transitions = detect_phase_transitions(traj)
    print(f"[相变] {len(transitions)} 次检测到")

    # 报告
    print_report(traj, transitions)

    # 可视化
    plot_collective_phase_diagram(traj, transitions)
    print("Done.")

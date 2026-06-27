"""
轨迹可视化 — 用 MemeTrajectory 替代静态散点绘制模因相图

将每条轨迹绘制为 R₀ × Chaos Axis 空间中的有向折线，
颜色按类别，节点大小按阶段。

用法:
    python src/trajectory/trajectory_viz.py
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from pathlib import Path

# 中文字体
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════════
# 颜色方案
# ═══════════════════════════════════════════════

CATEGORY_COLORS = {
    "解构自嘲": "#4CAF50",     # 绿 — 健康秩序
    "攻击发泄": "#F44336",     # 红 — 混沌宣泄
    "虚无退却": "#9C27B0",     # 紫 — 退出现实
    "身份认同": "#2196F3",     # 蓝 — 建构秩序
    "纯粹娱乐": "#FF9800",     # 橙 — 中性休息
}

PHASE_MARKERS = {
    "origin": "s",         # 方形 — 起点
    "emergence": "D",      # 菱形 — 萌芽
    "peak": "o",           # 圆 — 爆发顶点
    "controversy": "X",    # X — 争议
    "fixation": "^",       # 三角 — 固化
}

PHASE_COLORS = {
    "origin": "#666666",
    "emergence": "#FF9800",
    "peak": "#F44336",
    "controversy": "#9C27B0",
    "fixation": "#4CAF50",
}


def load_trajectories(path: str = None) -> list[dict]:
    if path is None:
        path = "data/processed/trajectories.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("trajectories", [])


def plot_trajectory_phase_diagram(trajectories: list[dict],
                                   save_path: str = None,
                                   show_labels: bool = True):
    """在 R₀ × Chaos Axis 空间中绘制所有模因轨迹。

    替代 phase_diagram.py 的散点图 —— 每条梗是一条有向路径。
    """
    fig, ax = plt.subplots(figsize=(16, 12))

    # ── 背景相区分区 ──
    # Viral Chaos basin (R₀ > 1.6, chaos < -0.2)
    ax.axvspan(1.6, 3.0, alpha=0.06, color="#F44336", label="Viral Chaos Basin")
    # Viral Order basin (R₀ > 1.6, chaos > +0.1)
    ax.axvspan(1.6, 3.0, ymin=0.55, alpha=0.06, color="#2196F3", label="Viral Order Basin")
    # R₀ = 1 相变线
    ax.axvline(x=1.0, color="#333333", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(1.02, 0.95, "R₀=1\n(phase transition)", transform=ax.get_xaxis_transform(),
            fontsize=8, alpha=0.5)
    # Chaos = 0 中性线
    ax.axhline(y=0, color="#333333", linestyle=":", alpha=0.3, linewidth=1)

    # ── 绘制每条轨迹 ──
    for traj in trajectories:
        nodes = traj.get("nodes", [])
        if len(nodes) < 2:
            # 单节点画点
            if nodes:
                n = nodes[0]
                ds = n.get("dynamic_state", {})
                r0 = ds.get("R0", 0)
                chaos = ds.get("chaos_axis", 0)
                cat = traj.get("category", "")
                color = CATEGORY_COLORS.get(cat, "#999999")
                ax.plot(r0, chaos, "o", color=color, markersize=6, alpha=0.7)
            continue

        cat = traj.get("category", "")
        color = CATEGORY_COLORS.get(cat, "#999999")
        name = traj.get("name", "")

        # 提取坐标
        r0s = []
        chaoses = []
        for n in nodes:
            ds = n.get("dynamic_state", {})
            r0s.append(ds.get("R0", 0))
            chaoses.append(ds.get("chaos_axis", 0))

        # 绘制连线（轨迹）
        ax.plot(r0s, chaoses, "-", color=color, alpha=0.5, linewidth=1.5)

        # 绘制每个节点
        for i, n in enumerate(nodes):
            phase = n.get("phase", "peak")
            marker = PHASE_MARKERS.get(phase, "o")
            size = 80 if phase == "peak" else 50 if phase == "fixation" else 35
            edge_color = PHASE_COLORS.get(phase, "#999999")
            ax.scatter(r0s[i], chaoses[i], marker=marker, s=size,
                      facecolor=color, edgecolors=edge_color, linewidth=1.5,
                      alpha=0.85, zorder=5)

        # 标注梗名
        if show_labels and len(nodes) > 0:
            # 在终点标注
            end_node = nodes[-1]
            ds = end_node.get("dynamic_state", {})
            ax.annotate(name,
                       xy=(ds.get("R0", 0), ds.get("chaos_axis", 0)),
                       xytext=(5, 5), textcoords="offset points",
                       fontsize=7, color=color, alpha=0.9,
                       fontfamily="sans-serif")

    # ── 图例 ──
    legend_elements = []
    for cat, color in CATEGORY_COLORS.items():
        legend_elements.append(mpatches.Patch(color=color, alpha=0.7, label=cat))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              title="Category", title_fontsize=10)

    # 阶段标记图例
    phase_elements = []
    for phase, marker in PHASE_MARKERS.items():
        phase_elements.append(plt.Line2D([0], [0], marker=marker, color="w",
                                          markerfacecolor="#666666",
                                          markersize=8, label=phase))
    ax.legend(handles=phase_elements, loc="lower left", fontsize=8,
              title="Phase", title_fontsize=9)

    ax.set_xlabel("R₀ (Basic Reproduction Number)", fontsize=12)
    ax.set_ylabel("Chaos Axis (-1 = absolute chaos, +1 = absolute order)", fontsize=12)
    ax.set_title("Meme Phase Diagram — Trajectory View (2020-2025)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(2.5, max(
        max([n.get("dynamic_state", {}).get("R0", 0) for n in t.get("nodes", [])])
        for t in trajectories if t.get("nodes")) + 0.3))
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[保存] 轨迹相图 → {save_path}")

    plt.show()


def plot_constraint_evolution(trajectories: list[dict], save_path: str = None):
    """绘制约束场向量在时间上的演化。

    每条轨迹的 constraint_state.pressures 在 phases 维度上的变化。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 对每条轨迹绘制约束场变化
    for traj in trajectories:
        nodes = traj.get("nodes", [])
        name = traj.get("name", "")
        cat = traj.get("category", "")
        color = CATEGORY_COLORS.get(cat, "#999999")

        pressures = []
        for n in nodes:
            cs = n.get("constraint_state", {}).get("pressures", [0.5]*5)
            pressures.append(cs[:5])

        if len(pressures) < 2:
            continue
        pressures = np.array(pressures)
        phases = list(range(len(pressures)))

        for dim in range(5):
            ax = axes[dim]
            ax.plot(phases, pressures[:, dim], "-o", color=color,
                   alpha=0.6, linewidth=1, markersize=3, label=name if dim == 0 else "")

    for dim in range(5):
        ax = axes[dim]
        ax.set_title(f"Constraint p{dim+1}", fontsize=11)
        ax.set_xlabel("Phase")
        ax.set_ylabel("Pressure")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)

    axes[5].set_visible(False)
    fig.suptitle("Constraint Field Evolution — All Trajectories", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[保存] 约束场演化 → {save_path}")

    plt.show()


# ═══════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    trajectories = load_trajectories()
    print(f"[加载] {len(trajectories)} 条轨迹")

    # 统计有丰富叙事数据的轨迹
    rich = [t for t in trajectories if len(t.get("nodes", [])) >= 3]
    print(f"[丰富] {len(rich)} 条轨迹有 3+ 阶段节点 (有 LLM 叙事)")

    # 仅绘制有 3+ 阶段的轨迹（更干净）
    plot_trajectory_phase_diagram(
        rich if len(rich) > 5 else trajectories,
        save_path="data/processed/trajectory_phase_diagram.png",
        show_labels=True,
    )

    plot_constraint_evolution(
        rich if len(rich) > 5 else trajectories,
        save_path="data/processed/constraint_evolution.png",
    )

"""
可视化模块 — MemeticChaos 可视化层。

提供可复用的绘图函数，用于：
1. SIR 曲线对比
2. 混沌轴散点图
3. 情感弧线可视化
4. 相变检测图
5. 参数空间热力图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Font setup
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from src.models.sir_meme import (
    SIRParams, SIRResult, TwoPopParams,
    solve_sir, solve_two_population,
    compute_entropy_curve, sweep_R0, sweep_beta_gamma,
    classify_meme_type, extract_lifecycle,
)


# ── Color palette ──────────────────────────────

CATEGORY_COLORS = {
    "解构自嘲": "#3498db",
    "攻击发泄": "#e74c3c",
    "虚无退却": "#95a5a6",
    "身份认同": "#2ecc71",
    "纯粹娱乐": "#f39c12",
}

CHAOS_CMAP = plt.cm.RdYlBu_r  # red (chaos) → blue (order)


def plot_sir_curve(result: SIRResult, ax: plt.Axes = None, title: str = None,
                   show_entropy: bool = True) -> plt.Axes:
    """绘制单个 SIR 曲线，可选熵叠加。

    Args:
        result: SIRResult
        ax: matplotlib Axes (created if None)
        title: 图表标题
        show_entropy: 是否叠加熵曲线

    Returns:
        matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    classification = classify_meme_type(result)
    lifecycle = extract_lifecycle(result)

    ax.plot(result.t, result.S, "b-", alpha=0.5, label="S (Susceptible)", linewidth=1)
    ax.plot(result.t, result.I, "r-", alpha=0.9, label="I (Infected)", linewidth=2)
    ax.plot(result.t, result.R, "g-", alpha=0.5, label="R (Recovered)", linewidth=1)
    ax.fill_between(result.t, 0, result.I, color="red", alpha=0.08)
    ax.axvline(x=result.peak_day, color="red", linestyle="--", alpha=0.3, label="peak")

    if show_entropy:
        H = compute_entropy_curve(result)
        ax2 = ax.twinx()
        ax2.plot(result.t, H, "purple", linewidth=1, alpha=0.6, label="H(t)")
        ax2.set_ylabel("Entropy", color="purple", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="purple", labelsize=7)

    title_str = title or f"{classification['type']} | R₀={result.params.R0:.1f}"
    ax.set_title(title_str, fontsize=10)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Proportion")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(-0.02, 0.55)

    return ax


def plot_sir_comparison(results: dict[str, SIRResult],
                        ncols: int = 2, figsize: tuple = None) -> plt.Figure:
    """并排比较多个 SIR 结果。

    Args:
        results: {label: SIRResult}
        ncols: 列数
        figsize: 图大小 (auto if None)

    Returns:
        matplotlib Figure
    """
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (6 * ncols, 4.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, (label, result) in enumerate(results.items()):
        row, col = i // ncols, i % ncols
        plot_sir_curve(result, ax=axes[row, col], title=label)

    # Hide unused axes
    for j in range(i + 1, nrows * ncols):
        row, col = j // ncols, j % ncols
        axes[row, col].set_visible(False)

    fig.suptitle("SIR Meme Dynamics Comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_chaos_landscape(memes: list, ax: plt.Axes = None) -> plt.Axes:
    """绘制热梗混沌景观图：R₀ × 混沌位置 散点图。

    Args:
        memes: MemeEntry 对象列表
        ax: matplotlib Axes

    Returns:
        matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    for cat, color in CATEGORY_COLORS.items():
        subset = [m for m in memes if m.category == cat]
        if not subset:
            continue
        xs = [m.chaos_position for m in subset]
        ys = [m.estimated_R0 for m in subset]
        ax.scatter(xs, ys, c=color, label=cat, s=100, alpha=0.7,
                   edgecolors="white", linewidth=0.5, zorder=3)

    # Annotate key memes
    for m in memes:
        if m.estimated_R0 >= 4.5 or abs(m.chaos_position) >= 0.55:
            ax.annotate(m.name, (m.chaos_position, m.estimated_R0),
                       fontsize=7, alpha=0.8,
                       xytext=(5, 5), textcoords="offset points")

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3, zorder=1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3,
               label="R₀=1 (threshold)", zorder=1)

    # Shade quadrants
    ax.axhspan(0, 1, xmin=0, xmax=0.5, alpha=0.03, color="gray")
    ax.axhspan(0, 1, xmin=0.5, xmax=1, alpha=0.03, color="gray")
    ax.axhspan(1, ax.get_ylim()[1], xmin=0, xmax=0.5, alpha=0.03, color="red")
    ax.axhspan(1, ax.get_ylim()[1], xmin=0.5, xmax=1, alpha=0.03, color="green")

    ax.set_xlabel("Chaos ← → Order")
    ax.set_ylabel("Estimated R₀")
    ax.set_title("Meme Landscape: Propagation Power vs Chaos-Order Position")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)

    return ax


def plot_R0_sweep(R0_range: np.ndarray, gamma: float = 0.1,
                  ax: plt.Axes = None) -> plt.Axes:
    """绘制 R₀ 扫描图：峰值和最终感染率 vs R₀。

    Args:
        R0_range: R₀ 值数组
        gamma: 固定恢复率
        ax: matplotlib Axes

    Returns:
        matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    sweeps = sweep_R0(R0_range, gamma=gamma)
    peaks = [s["result"].peak_infected for s in sweeps]
    finals = [s["result"].total_infected for s in sweeps]

    ax.plot(R0_range, peaks, "r-o", label="Peak Infected (I_max)", markersize=5)
    ax.plot(R0_range, finals, "g-s", label="Final Recovered (R_∞)", markersize=5)
    ax.axvline(x=1.0, color="k", linestyle="--", alpha=0.5, label="R₀ = 1")
    ax.set_xlabel("R₀")
    ax.set_ylabel("Proportion")
    ax.set_title("Outbreak Size vs Basic Reproduction Number")
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


def plot_entropy_trajectory(result: SIRResult, ax: plt.Axes = None,
                            title: str = None) -> plt.Axes:
    """绘制熵轨迹与感染曲线的叠加图。

    Args:
        result: SIRResult
        ax: matplotlib Axes
        title: 标题

    Returns:
        matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    H = compute_entropy_curve(result)
    ax2 = ax.twinx()

    ax.fill_between(result.t, 0, result.I, color="red", alpha=0.15)
    ax.plot(result.t, result.I, "r-", linewidth=2, alpha=0.8, label="Infected I(t)")
    ax2.plot(result.t, H, "purple", linewidth=1.5, label="Entropy H(t)")

    H_max = np.log(3)
    ax2.axhline(y=H_max, color="gray", linestyle=":", alpha=0.3,
                label=f"H_max = ln(3) ≈ {H_max:.2f}")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected Proportion", color="red")
    ax2.set_ylabel("Shannon Entropy", color="purple")

    classification = classify_meme_type(result)
    title_str = title or f"Chaos Dynamics: {classification['type']} Meme"
    ax.set_title(title_str)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)

    return ax


def plot_two_population(tp_result: dict, ax: plt.Axes = None) -> plt.Axes:
    """绘制双群体模型的核心 vs 大众对比。

    Args:
        tp_result: solve_two_population 的返回结果
        ax: matplotlib Axes

    Returns:
        matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    t = tp_result["t"]
    ax.plot(t, tp_result["I_core"], "r-", linewidth=2, alpha=0.8, label="I_core (5%)")
    ax.plot(t, tp_result["I_mass"], "orange", linewidth=2, alpha=0.8, label="I_mass (95%)")

    core_peak_t = t[np.argmax(tp_result["I_core"])]
    mass_peak_t = t[np.argmax(tp_result["I_mass"])]
    ax.axvline(x=core_peak_t, color="red", linestyle="--", alpha=0.3)
    ax.axvline(x=mass_peak_t, color="orange", linestyle="--", alpha=0.3)

    params = tp_result["params"]
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected Proportion (within group)")
    ax.set_title(f"Two-Population SIR: Core → Mass Diffusion\n"
                 f"β_core={params.beta_core}, β_mass={params.beta_mass}, "
                 f"β_cross={params.beta_cross_c2m}")
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


def plot_sentiment_arc(meme, ax: plt.Axes = None) -> plt.Axes:
    """绘制单个热梗的情感弧线。

    Args:
        meme: MemeEntry 对象
        ax: matplotlib Axes

    Returns:
        matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    phases = meme.sentiment_arc
    if not phases:
        ax.text(0.5, 0.5, "No sentiment data", ha="center", va="center",
                transform=ax.transAxes)
        return ax

    labels = [p["phase"] for p in phases]
    intensities = [p["intensity"] for p in phases]
    sentiments = [p["sentiment"] for p in phases]

    x = np.arange(len(phases))
    ax.plot(x, intensities, "o-", linewidth=2, markersize=8, color="#e74c3c")
    ax.fill_between(x, 0, intensities, alpha=0.1, color="#e74c3c")

    for i, (label, s) in enumerate(zip(labels, sentiments)):
        ax.annotate(f"{label}\n{s}", (x[i], intensities[i]),
                   fontsize=7, ha="center",
                   xytext=(0, 12), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Sentiment Intensity")
    ax.set_title(f"Sentiment Arc: {meme.name} [{meme.category}]")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")

    return ax


def savefig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """保存图表到项目 outputs/figures/ 目录。"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")

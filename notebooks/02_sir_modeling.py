"""
Notebook 2: SIR 模因动力学建模
作为 Jupyter Notebook 使用：# %%
对应: notebooks/02_sir_modeling.ipynb

目标:
1. 演示四种模因类型的 SIR 曲线
2. 参数扫描：R₀-γ 空间的传播行为
3. 双群体模型：核心圈层 → 大众扩散
4. 混沌动力学分析：熵轨迹、相变检测
5. 用策展数据估算的 SIR 参数模拟实际热梗
"""

# %% [markdown]
# # MemeticChaos — SIR 模因动力学建模
#
# 本节将 SIR 传染病模型应用于热梗传播，
# 并用混沌动力学工具分析集体情感系统的演化。

# %% imports
import sys
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from src.models.sir_meme import (
    SIRParams, TwoPopParams,
    solve_sir, solve_two_population,
    sweep_R0, sweep_beta_gamma,
    classify_meme_type, extract_lifecycle,
    compute_entropy_curve, detect_phase_transition,
    demo_all_types,
)
from src.data.curator import MemeCurator

# %% 1. 四种模因类型的 SIR 曲线对比
demos = demo_all_types()
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
labels = {
    "脉冲型_雪糕刺客": "Pulse Type (xuegao cike)",
    "爆发型_打工人": "Outbreak Type (dagongren)",
    "长尾型_内卷": "Long-tail Type (neijuan)",
    "流产型_小众梗": "Abortive Type (niche meme)",
}

for ax, (name, result) in zip(axes.flat, demos.items()):
    ax.plot(result.t, result.S, "b-", alpha=0.7, label="S (Susceptible)", linewidth=1.5)
    ax.plot(result.t, result.I, "r-", alpha=0.9, label="I (Infected)", linewidth=2)
    ax.plot(result.t, result.R, "g-", alpha=0.7, label="R (Recovered)", linewidth=1.5)

    classification = classify_meme_type(result)
    lifecycle = extract_lifecycle(result)

    ax.axvline(x=result.peak_day, color="red", linestyle="--", alpha=0.3)
    ax.fill_between(result.t, 0, result.I, color="red", alpha=0.1)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Proportion")
    ax.set_title(f"{name}\n{result.params} | {classification['type']}")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.02, 0.55)

plt.suptitle("SIR Meme Dynamics: Four Archetypes", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("outputs/figures/sir_four_archetypes.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/sir_four_archetypes.png")

# %% [markdown]
# ### 解读
# - **脉冲型**: 高 R₀ + 高 γ → 爆发迅猛但消退也快。典型：XX刺客、娱乐梗。
# - **爆发型**: 中高 R₀ → 经典传播曲线。典型：打工人、鸡你太美。
# - **长尾型**: 低 R₀ ≈ 1-2 → 缓慢渗透、持续存在。典型：内卷、原生家庭。
# - **流产型**: R₀ < 1 → 未能建立秩序，被混沌吞没。

# %% 2. R₀ 参数扫描
R0_values = np.linspace(0.2, 8.0, 20)
sweeps = sweep_R0(R0_values, gamma=0.1)
peaks = [s["result"].peak_infected for s in sweeps]
finals = [s["result"].total_infected for s in sweeps]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(R0_values, peaks, "r-o", label="Peak Infected (Imax)", markersize=6)
ax.plot(R0_values, finals, "g-s", label="Final Recovered (R∞)", markersize=6)
ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="R₀ = 1 (critical threshold)")
ax.set_xlabel("R₀ (Basic Reproduction Number)")
ax.set_ylabel("Proportion")
ax.set_title("Meme Outbreak Size vs R₀\nPhase Transition at R₀ = 1")
ax.legend()
ax.grid(alpha=0.3)

# Annotate meme categories on the curve
cat_regions = {
    "Abortive": (0.2, 0.8),
    "Long-tail": (0.9, 2.5),
    "Outbreak": (2.5, 5.0),
    "Pulse/Viral": (5.0, 8.0),
}
for label, (lo, hi) in cat_regions.items():
    ax.axvspan(lo, hi, alpha=0.08, color=["gray", "blue", "orange", "red"][list(cat_regions.keys()).index(label)])
    ax.text((lo+hi)/2, 0.02, label, ha="center", fontsize=8, alpha=0.6)

plt.tight_layout()
plt.savefig("outputs/figures/R0_sweep.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/R0_sweep.png")

# %% 3. β-γ 参数空间热力图
beta_range = np.linspace(0.05, 1.0, 30)
gamma_range = np.linspace(0.02, 0.5, 30)
peak_matrix = sweep_beta_gamma(beta_range, gamma_range)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.contourf(beta_range, gamma_range, peak_matrix, levels=20, cmap="RdYlBu_r")
ax.contour(beta_range, gamma_range, peak_matrix, levels=[0.01, 0.1, 0.3, 0.5],
           colors="black", linewidths=0.5, alpha=0.5)

# R₀ = 1 line
R0_line = beta_range / 1.0  # gamma values where R₀ = 1
ax.plot(beta_range, beta_range, "k--", alpha=0.5, label="R₀ = 1")
ax.fill_between(beta_range, 0, beta_range, alpha=0.05, color="green")
ax.fill_between(beta_range, beta_range, gamma_range.max(), alpha=0.05, color="red")

ax.set_xlabel("β (Infection Rate)")
ax.set_ylabel("γ (Recovery Rate)")
ax.set_title("Peak Infected in β-γ Parameter Space\nGreen region: R₀ < 1 (no outbreak)\nRed region: R₀ > 1 (outbreak)")
ax.legend()
plt.colorbar(im, ax=ax, label="Peak Infected Proportion")
plt.tight_layout()
plt.savefig("outputs/figures/beta_gamma_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/beta_gamma_heatmap.png")

# %% 4. 双群体模型：核心圈层 → 大众扩散
tp = TwoPopParams(
    beta_core=0.5, beta_mass=0.15,
    beta_cross_c2m=0.12, beta_cross_m2c=0.01,
    gamma=0.1, N_core=0.05, N_mass=0.95,
)
result = solve_two_population(tp, t_span=(0, 200))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Core population
ax1.plot(result["t"], result["I_core"], "r-", linewidth=2, label="I_core")
ax1.plot(result["t"], result["S_core"], "b--", alpha=0.5, label="S_core")
ax1.plot(result["t"], result["R_core"], "g--", alpha=0.5, label="R_core")
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Proportion (within core)")
ax1.set_title(f"Core Circle (5% of population)\nR₀_core = {tp.R0_core:.1f}")
ax1.legend()
ax1.grid(alpha=0.3)

# Mass population
ax2.plot(result["t"], result["I_mass"], "orange", linewidth=2, label="I_mass")
ax2.plot(result["t"], result["S_mass"], "b--", alpha=0.5, label="S_mass")
ax2.plot(result["t"], result["R_mass"], "g--", alpha=0.5, label="R_mass")
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Proportion (within mass)")
ax2.set_title(f"Mass Population (95%)\nCross-infection from core: β_c2m={tp.beta_cross_c2m}")
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle("Two-Population SIR Model: Core → Mass Diffusion", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/two_population_model.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/two_population_model.png")

# %% [markdown]
# ### 解读
# 双群体模型捕捉了热梗传播的关键结构：
# 1. **核心圈层**（B站/贴吧/豆瓣小组）先感染，R₀ 高（内部链接紧密）
# 2. **跨层传播**触达大众，但感染率降低（大众链接稀疏）
# 3. 核心圈的峰值早于大众峰值 → 解释了为什么「老用户」总感觉梗「出圈就变味」

# %% 5. 混沌动力学：熵轨迹分析
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (name, result) in zip(axes.flat, demos.items()):
    H = compute_entropy_curve(result)
    ax2_twin = ax.twinx()

    ax.plot(result.t, result.I, "r-", alpha=0.4, linewidth=3, label="Infected I(t)")
    ax2_twin.plot(result.t, H, "purple", linewidth=1.5, label="Entropy H(t)")
    ax2_twin.axhline(y=np.log(3), color="gray", linestyle=":", alpha=0.3, label="H_max = ln(3)")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected Proportion", color="red")
    ax2_twin.set_ylabel("Shannon Entropy", color="purple")
    ax.set_title(f"{name}\nEntropy Trajectory During Meme Lifecycle")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

plt.suptitle("Chaos Dynamics: Entropy Trajectory Across Meme Archetypes", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/figures/entropy_trajectory.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/entropy_trajectory.png")

# %% [markdown]
# ### 解读
# 熵在模因生命周期中的典型轨迹：
# 1. **萌芽期**（低熵）：梗在小圈子传播，系统有序
# 2. **爆发期**（熵增）：梗扩散至大众，参与者的情感状态多样化，系统混沌度上升
# 3. **消退期**（熵减）：人群趋于免疫/厌倦，系统收敛到新稳态
#
# 熵峰值出现在爆发与消退的过渡期 → 这是系统的「混沌最大化」时刻。

# %% 6. 相变检测
R0_values = np.linspace(0.5, 3.0, 15)
sweeps = sweep_R0(R0_values, gamma=0.1)
transition = detect_phase_transition(sweeps)

print(f"Phase transition detected at sweep index {transition['transition_at']}")
print(f"  Pre-transition R₀: {transition['pre_R0']:.2f}")
print(f"  Post-transition R₀: {transition['post_R0']:.2f}")
print(f"  This is where meme goes from 'no outbreak' to 'outbreak' regime.")

# %% 7. 用策展数据估算参数，模拟实际热梗
curator = MemeCurator()
estimations = curator.to_sir_estimation()

# Pick key memes for simulation
key_memes = ["打工人", "躺平", "后浪", "科目三", "普信男"]
fig, axes = plt.subplots(len(key_memes), 1, figsize=(12, 3 * len(key_memes)))

for ax, name in zip(axes, key_memes):
    est = next((e for e in estimations if e["name"] == name), None)
    if est is None:
        continue

    params = SIRParams(beta=est["beta_estimated"], gamma=est["gamma_estimated"], N=1.0)
    result = solve_sir(params)
    H = compute_entropy_curve(result)
    classification = classify_meme_type(result)

    ax.plot(result.t, result.I, "r-", linewidth=2, alpha=0.8)
    ax.fill_between(result.t, 0, result.I, color="red", alpha=0.1)
    ax2 = ax.twinx()
    ax2.plot(result.t, H, "purple", linewidth=1, alpha=0.5)
    ax2.set_ylabel("H(t)", color="purple", fontsize=8)

    ax.set_title(f"{name} [{est['category']}] | R₀={est['R0_estimated']:.1f} | "
                 f"chaos={est['chaos_position']:+.1f} | {classification['type']} | "
                 f"peak={result.peak_day:.0f}d, dur={result.duration:.0f}d",
                 fontsize=9)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected", color="red")

plt.suptitle("Simulated SIR Curves for Key Memes (Lifecycle-Estimated Parameters)", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/key_memes_simulation.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/key_memes_simulation.png")

# %% [markdown]
# # 总结

# 从 SIR 建模中得到的核心洞察：

# 1. **R₀ = 1 是临界点**
#    模因在 R₀ = 1 处经历相变：低于此阈值，梗无法建立秩序；
#    高于此阈值，梗成功在混沌中建立了短暂的局部秩序。

# 2. **熵轨迹呈拱形**
#    模因生命周期对应一个完整的熵循环：
#    低熵（小圈传播）→ 高熵（大众扩散）→ 低熵（免疫/遗忘）。
#    这完美对应了核心元定律中「与混沌共存 → 建立秩序 → 秩序消亡或固化」的循环。

# 3. **核心圈层 → 大众的跨层传播**
#    双群体模型解释了为什么「老用户」总感觉梗「出圈就变味」：
#    核心圈层的 R₀ 高（内部紧密链接），大众 R₀ 低。
#    跨层传播的 β 值决定了梗能否「破圈」。

# 4. **混沌轴位置与 SIR 参数的相关性**
#    靠近「绝对混沌」的梗（攻击发泄类）→ 高 R₀ + 短 duration（脉冲型）
#    靠近「绝对秩序」的梗（身份认同类）→ 中 R₀ + 长 duration（长尾型）
#    这一模式将在 Phase 3 的吸引子分析中进一步验证。

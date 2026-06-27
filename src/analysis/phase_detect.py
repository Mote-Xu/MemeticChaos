"""
相变检测模块 — MemeticChaos 分析层。

核心问题：在什么条件下，集体情感系统会从一种状态模式
切换到另一种？（如：解构自嘲 → 攻击发泄 → 虚无退却）

方法：
1. R₀ 临界跨越检测（模因传播的 on/off 相变）
2. 混沌轴漂移检测（集体情感方向变化的早期预警）
3. 熵突变检测（系统混沌度的 sudden change）

对齐「微尘哲学」：相变点 = 集体情感系统在绝对混沌与绝对秩序
之间的振荡转折点。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.models.sir_meme import SIRParams, SIRResult, solve_sir, sweep_R0, compute_entropy_curve
from src.data.curator import MemeCurator


@dataclass
class PhaseTransition:
    """一次检测到的相变。"""
    name: str  # descriptive label
    transition_type: str  # "R0_crossing" | "chaos_shift" | "entropy_surge"
    time_index: int  # position in sequence
    pre_state: dict  # state before transition
    post_state: dict  # state after transition
    significance: float  # 0-1 how significant the transition is
    description: str


def detect_R0_crossings(R0_values: np.ndarray, gamma: float = 0.1) -> list[PhaseTransition]:
    """检测 R₀ 跨越 1.0 的临界点。

    R₀ = 1 是模因传播的相变点：
    - R₀ < 1: 梗无法建立秩序（流产型）
    - R₀ > 1: 梗成功建立局部秩序（爆发型/长尾型）

    Args:
        R0_values: R₀ 扫描值数组
        gamma: 固定恢复率

    Returns:
        检测到的相变列表
    """
    sweeps = sweep_R0(R0_values, gamma=gamma)
    transitions = []

    for i in range(len(R0_values) - 1):
        R0_pre, R0_post = R0_values[i], R0_values[i + 1]
        if (R0_pre - 1.0) * (R0_post - 1.0) < 0:
            pre_raw = sweeps[i]
            post_raw = sweeps[i + 1]

            pre_I = pre_raw["result"].peak_infected
            post_I = post_raw["result"].peak_infected

            # Significance: how dramatic is the change?
            significance = min(1.0, abs(post_I - pre_I) * 5)

            transitions.append(PhaseTransition(
                name=f"R₀ crossing at ~{R0_post:.2f}",
                transition_type="R0_crossing",
                time_index=i,
                pre_state={"R0": float(R0_pre), "peak_I": float(pre_I)},
                post_state={"R0": float(R0_post), "peak_I": float(post_I)},
                significance=round(significance, 3),
                description=(
                    f"模因传播能力从 R₀={R0_pre:.2f} (peak={pre_I:.1%}) "
                    f"跨越至 R₀={R0_post:.2f} (peak={post_I:.1%})。"
                    f"这标志着从'流产型'到'爆发型'的质变。"
                ),
            ))

    return transitions


def detect_chaos_shift(yearly_chaos: dict[int, list[float]],
                        threshold: float = 0.3) -> list[PhaseTransition]:
    """检测集体情感混沌轴的年份间显著漂移。

    如果相邻年份的平均混沌位置变化超过阈值，
    说明系统的整体情感倾向发生了结构性偏移。

    Args:
        yearly_chaos: {year: [chaos_positions]} 按年份的混沌位置数据
        threshold: 漂移检测阈值

    Returns:
        检测到的相变列表
    """
    years = sorted(yearly_chaos.keys())
    transitions = []

    for i in range(len(years) - 1):
        y_pre, y_post = years[i], years[i + 1]
        mean_pre = np.mean(yearly_chaos[y_pre])
        mean_post = np.mean(yearly_chaos[y_post])
        shift = mean_post - mean_pre

        if abs(shift) > threshold:
            significance = min(1.0, abs(shift) / 0.5)

            direction = "向绝对混沌" if shift < 0 else "向绝对秩序"
            transitions.append(PhaseTransition(
                name=f"Chaos shift {y_pre}→{y_post}",
                transition_type="chaos_shift",
                time_index=i,
                pre_state={"year": y_pre, "mean_chaos": round(float(mean_pre), 3)},
                post_state={"year": y_post, "mean_chaos": round(float(mean_post), 3)},
                significance=round(significance, 3),
                description=(
                    f"{y_pre}→{y_post} 年间，集体情感混沌轴从 "
                    f"{mean_pre:+.2f} 漂移至 {mean_post:+.2f} ({direction})。"
                    f"漂移幅度 {shift:+.2f} > 阈值 {threshold}。"
                ),
            ))

    return transitions


def detect_entropy_surge(result: SIRResult,
                         z_threshold: float = 2.5) -> list[PhaseTransition]:
    """检测 SIR 轨迹中的熵突变点。

    熵的 sudden increase → 系统进入高度混沌状态
    熵的 sudden decrease → 系统快速建立秩序

    Args:
        result: SIRResult
        z_threshold: Z-score 阈值

    Returns:
        检测到的熵突变列表
    """
    H = compute_entropy_curve(result)
    t = result.t

    # Compute entropy rate of change
    dH = np.gradient(H, t)
    mean_dH = np.mean(dH)
    std_dH = np.std(dH)

    # Find surges (significantly above mean rate)
    z_scores = (dH - mean_dH) / (std_dH + 1e-10)
    surge_indices = np.where(np.abs(z_scores) > z_threshold)[0]

    transitions = []
    for idx in surge_indices:
        significance = min(1.0, abs(z_scores[idx]) / 5)
        direction = "熵增（混沌加剧）" if dH[idx] > 0 else "熵减（秩序建立）"

        transitions.append(PhaseTransition(
            name=f"Entropy surge at t≈{t[idx]:.1f}",
            transition_type="entropy_surge",
            time_index=int(idx),
            pre_state={"t": float(t[idx]), "H": float(H[idx]), "dH": float(dH[idx])},
            post_state={"z_score": float(z_scores[idx])},
            significance=round(significance, 3),
            description=(
                f"在 t≈{t[idx]:.1f} 出现熵的显著{direction}，"
                f"z-score={z_scores[idx]:.1f}，"
                f"这是系统行为模式的突变点。"
            ),
        ))

    return transitions


def phase_transition_summary(curator: Optional[MemeCurator] = None) -> dict:
    """对整个数据集进行相变检测的综合分析。

    Returns:
        {
            "R0_crossings": [...],
            "chaos_shifts": [...],
            "key_meme_entropy_surges": {...},
            "interpretation": str
        }
    """
    if curator is None:
        curator = MemeCurator()

    # 1. R0 critical crossings
    R0_range = np.linspace(0.3, 5.0, 30)
    r0_transitions = detect_R0_crossings(R0_range)

    # 2. Chaos shifts across years
    yearly_chaos = curator.chaos_by_year()
    chaos_transitions = detect_chaos_shift(yearly_chaos)

    # 3. Entropy surges for key memes
    key_memes = ["打工人", "躺平", "后浪", "普信男", "小镇做题家"]
    entropy_surges = {}
    for name in key_memes:
        meme = curator.get(name)
        if not meme:
            continue
        # Estimate SIR params
        from src.models.sir_meme import estimate_params_from_lifecycle, estimate_total_infected
        lc = meme.lifecycle
        dur = lc.get("duration_months", 6)
        if dur >= 999:
            dur = 18
        pm = meme.propagation_model
        circle_count = len(pm.get("circle_layers", []))
        sa = meme.sentiment_arc
        peak_intensity = max(p.get("intensity", 0.5) for p in sa) if sa else 0.5
        ti = estimate_total_infected(circle_count, peak_intensity, dur)
        params = estimate_params_from_lifecycle(
            peak_day=dur * 30 * 0.3,
            total_infected=ti,
            duration_days=dur * 30,
        )
        result = solve_sir(params)
        surges = detect_entropy_surge(result)
        entropy_surges[name] = {
            "n_surges": len(surges),
            "surges": [{"t": s.time_index, "sig": s.significance, "desc": s.description}
                       for s in surges[:3]],
        }

    # Interpretation
    surge_summary = ", ".join(
        f"{k}({v['n_surges']})" for k, v in entropy_surges.items()
    )
    interpretation_lines = [
        f"检测到 {len(r0_transitions)} 次 R0 临界跨越（传播相变）",
        f"检测到 {len(chaos_transitions)} 次混沌轴年份间显著漂移",
        f"关键热梗熵突变：{surge_summary}",
    ]

    return {
        "R0_crossings": [t.__dict__ for t in r0_transitions],
        "chaos_shifts": [t.__dict__ for t in chaos_transitions],
        "key_meme_entropy_surges": entropy_surges,
        "interpretation": "\n".join(interpretation_lines),
    }


# ═══════════════════════════════════════════════
# Script entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("MemeticChaos — Phase Transition Detection")
    print("=" * 60)

    summary = phase_transition_summary()

    print("\n--- Interpretation ---")
    print(summary["interpretation"])

    print("\n--- R₀ Critical Crossings ---")
    for t in summary["R0_crossings"]:
        print(f"  {t['name']}: significance={t['significance']:.3f}")
        pre, post = t["pre_state"]["R0"], t["post_state"]["R0"]
        print(f"    R₀: {pre:.2f} → {post:.2f}")

    print("\n--- Chaos Axis Shifts ---")
    for t in summary["chaos_shifts"]:
        print(f"  {t['name']}: significance={t['significance']:.3f}")
        print(f"    {t['description']}")

    print("\n--- Entropy Surges in Key Memes ---")
    for name, info in summary["key_meme_entropy_surges"].items():
        print(f"  {name}: {info['n_surges']} surge(s)")
        for s in info["surges"]:
            print(f"    t={s['t']}, sig={s['sig']:.3f}")

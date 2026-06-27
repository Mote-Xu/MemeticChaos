"""
热梗生命周期分析 — MemeticChaos 分析层。

功能：
1. 从 SIR 曲线提取并分类生命周期形态
2. 跨类别生命周期对比
3. 生命周期异常检测（过早消亡 / 异常长寿）
4. 与策展数据中的 qualitative lifecycle 交叉验证
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.models.sir_meme import (
    SIRParams, SIRResult, MemeLifecycle,
    solve_sir, extract_lifecycle, classify_meme_type,
    estimate_params_from_lifecycle, estimate_total_infected,
)
from src.data.curator import MemeCurator, MemeEntry


@dataclass
class LifecycleProfile:
    """完整的热梗生命周期剖面：定量 SIR + 定性策展。"""
    name: str
    category: str
    year: int
    chaos_position: float
    # Quantitative (SIR)
    sir_params: SIRParams
    sir_result: SIRResult
    meme_type: str
    lifecycle: MemeLifecycle
    # Qualitative (curated)
    qualitative_status: str
    qualitative_duration_months: int

    def summary(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "type": self.meme_type,
            "R0": round(self.sir_params.R0, 2),
            "peak_day": round(self.sir_result.peak_day, 1),
            "peak_infected_pct": round(self.sir_result.peak_infected * 100, 1),
            "duration_days": round(self.sir_result.duration, 1),
            "sir_status": self.lifecycle.status,
            "curated_status": self.qualitative_status,
        }


def build_lifecycle_profiles(curator: Optional[MemeCurator] = None) -> list[LifecycleProfile]:
    """为策展数据集中的每个热梗构建生命周期剖面。

    使用 estimate_params_from_lifecycle 从 qualitative 数据估算 SIR 参数，
    然后求解 SIR 模型，提取 quantitative 生命周期。

    Args:
        curator: MemeCurator 实例，默认自动创建

    Returns:
        LifecycleProfile 列表
    """
    if curator is None:
        curator = MemeCurator()

    profiles = []
    for meme in curator.memes:
        lc_data = meme.lifecycle
        pm = meme.propagation_model

        # Estimate duration
        dur_months = lc_data.get("duration_months", 3)
        if dur_months >= 999:
            dur_months = 18
        dur_days = dur_months * 30

        # Estimate total infected: continuous from lifecycle features
        circle_count = len(pm.get("circle_layers", []))
        sa = meme.sentiment_arc
        if sa and len(sa) > 0:
            peak_intensity = max(p.get("intensity", 0.5) for p in sa)
        else:
            peak_intensity = 0.5
        total_infected = estimate_total_infected(
            circle_count, peak_intensity, dur_months
        )

        # Estimate SIR params
        params = estimate_params_from_lifecycle(
            peak_day=dur_days * 0.3,
            total_infected=total_infected,
            duration_days=dur_days,
        )

        # Solve SIR
        result = solve_sir(params)
        classification = classify_meme_type(result)
        lifecycle = extract_lifecycle(result)

        profiles.append(LifecycleProfile(
            name=meme.name,
            category=meme.category,
            year=meme.year,
            chaos_position=meme.chaos_position,
            sir_params=params,
            sir_result=result,
            meme_type=classification["type"],
            lifecycle=lifecycle,
            qualitative_status=lc_data.get("status", "未知"),
            qualitative_duration_months=dur_months,
        ))

    return profiles


def compare_categories(profiles: list[LifecycleProfile]) -> dict:
    """按类别聚合生命周期统计。

    Returns:
        dict: {category: {mean_R0, mean_peak, mean_duration, count, ...}}
    """
    from collections import defaultdict
    agg = defaultdict(lambda: {"R0": [], "peak": [], "duration": [], "chaos": []})

    for p in profiles:
        cat = p.category
        agg[cat]["R0"].append(p.sir_params.R0)
        agg[cat]["peak"].append(p.sir_result.peak_infected)
        agg[cat]["duration"].append(p.sir_result.duration)
        agg[cat]["chaos"].append(p.chaos_position)

    result = {}
    for cat, vals in agg.items():
        result[cat] = {
            "count": len(vals["R0"]),
            "mean_R0": round(float(np.mean(vals["R0"])), 2),
            "std_R0": round(float(np.std(vals["R0"])), 2),
            "mean_peak_pct": round(float(np.mean(vals["peak"])) * 100, 1),
            "mean_duration_days": round(float(np.mean(vals["duration"])), 1),
            "mean_chaos": round(float(np.mean(vals["chaos"])), 2),
        }
    return result


def detect_outliers(profiles: list[LifecycleProfile],
                    z_threshold: float = 2.0) -> dict[str, list[dict]]:
    """检测生命周期异常的热梗。

    两类异常：
    1. 过早消亡：定性预期长久但定量 duration 异常短
    2. 异常长寿：定性预期短但定量 duration 异常长

    Args:
        profiles: LifecycleProfile 列表
        z_threshold: Z-score 阈值

    Returns:
        {"premature_death": [...], "abnormally_long_lived": [...]}
    """
    durations = np.array([p.sir_result.duration for p in profiles])
    mean_dur = np.mean(durations)
    std_dur = np.std(durations)
    z_scores = (durations - mean_dur) / (std_dur + 1e-10)

    premature = []
    long_lived = []

    for p, z in zip(profiles, z_scores):
        if z < -z_threshold and p.qualitative_duration_months >= 12:
            premature.append({
                "name": p.name,
                "category": p.category,
                "duration_days": round(p.sir_result.duration, 1),
                "qualitative_dur_months": p.qualitative_duration_months,
                "z_score": round(float(z), 2),
            })
        elif z > z_threshold and p.qualitative_duration_months <= 3:
            long_lived.append({
                "name": p.name,
                "category": p.category,
                "duration_days": round(p.sir_result.duration, 1),
                "qualitative_dur_months": p.qualitative_duration_months,
                "z_score": round(float(z), 2),
            })

    return {"premature_death": premature, "abnormally_long_lived": long_lived}


def lifecycle_stage_analysis(result: SIRResult) -> dict[str, float]:
    """将 SIR 曲线分解为生命周期阶段的时间占比。

    Returns:
        {"emergence_pct": 0-100, "growth_pct": 0-100, "decay_pct": 0-100,
         "equilibrium_pct": 0-100}
    """
    I = result.I
    t = result.t
    peak_idx = np.argmax(I)
    T = t[-1]

    # Emergence: I from 0 to 10% of peak
    thresh_10 = 0.1 * I[peak_idx]
    e_idx = np.where(I[:peak_idx] >= thresh_10)[0]
    emergence_end = t[e_idx[0]] if len(e_idx) > 0 else 0

    # Growth: from 10% to peak
    growth_dur = t[peak_idx] - emergence_end

    # Decay: from peak to 10% of peak
    post_peak = I[peak_idx:]
    d_idx = np.where(post_peak <= thresh_10)[0]
    decay_end = t[peak_idx + d_idx[0]] if len(d_idx) > 0 else T
    decay_dur = decay_end - t[peak_idx]

    # Equilibrium: rest
    eq_dur = T - decay_end

    return {
        "emergence_pct": round(emergence_end / T * 100, 1),
        "growth_pct": round(growth_dur / T * 100, 1),
        "decay_pct": round(decay_dur / T * 100, 1),
        "equilibrium_pct": round(eq_dur / T * 100, 1),
    }


# ═══════════════════════════════════════════════
# Script entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("MemeticChaos — Lifecycle Analysis")
    print("=" * 60)

    profiles = build_lifecycle_profiles()
    print(f"\nTotal profiles: {len(profiles)}")

    print("\n--- Category Comparison ---")
    cat_stats = compare_categories(profiles)
    for cat, stats in sorted(cat_stats.items()):
        print(f"  {cat}: R₀={stats['mean_R0']:.2f}±{stats['std_R0']:.2f}, "
              f"peak={stats['mean_peak_pct']:.1f}%, "
              f"duration={stats['mean_duration_days']:.0f}d, "
              f"chaos={stats['mean_chaos']:+.2f}")

    print("\n--- Outliers ---")
    outliers = detect_outliers(profiles)
    if outliers["premature_death"]:
        print("  Premature death:")
        for o in outliers["premature_death"]:
            print(f"    {o['name']}: SIR={o['duration_days']}d vs curated={o['qualitative_dur_months']}mo")

    print("\n--- Top 5 by Duration ---")
    by_dur = sorted(profiles, key=lambda p: p.sir_result.duration, reverse=True)
    for p in by_dur[:5]:
        print(f"  {p.name} [{p.category}]: {p.sir_result.duration:.0f}d, R₀={p.sir_params.R0:.1f}")

    print("\n--- Top 5 by Peak ---")
    by_peak = sorted(profiles, key=lambda p: p.sir_result.peak_infected, reverse=True)
    for p in by_peak[:5]:
        print(f"  {p.name} [{p.category}]: peak={p.sir_result.peak_infected:.1%}, R₀={p.sir_params.R0:.1f}")

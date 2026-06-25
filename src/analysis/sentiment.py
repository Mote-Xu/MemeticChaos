"""
情感分析模块 — MemeticChaos 分析层。

功能：
1. 从策展数据提取情感弧线（sentiment arc）
2. 计算情感极性和强度时间序列
3. 识别情感弧线类型（rise-fall / fall-rise / U-shape / inverted-U）
4. 与混沌轴交叉分析
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.data.curator import MemeCurator, MemeEntry

# ── Sentiment arc classification ─────────────────

ArcType = Literal[
    "rise_fall",       # 情感先升后降 (兴奋→厌倦)
    "fall_rise",       # 情感先降后升 (争议→接受)
    "inverted_U",      # 倒U型 (共鸣→衰减)
    "U_shape",         # U型 (负面→转正)
    "monotonic_up",    # 单调上升 (持续正反馈)
    "monotonic_down",  # 单调下降 (持续恶化)
    "stable",          # 稳定中性
    "oscillating",     # 振荡 (反复横跳)
]


@dataclass
class SentimentArc:
    """情感弧线分析结果。"""
    name: str
    category: str
    arc_type: ArcType
    phases: list[dict]
    sentiment_range: tuple[float, float]  # (min, max) of intensity
    mean_intensity: float
    entropy: float  # sentiment entropy across phases
    chaos_correlation: float  # correlation with chaos_vector position


def classify_sentiment_arc(phases: list[dict]) -> ArcType:
    """基于情感阶段序列分类弧线类型。

    Args:
        phases: [{phase, sentiment, intensity}, ...] 按时间排序

    Returns:
        ArcType 标签
    """
    if len(phases) < 2:
        return "stable"

    intensities = [p.get("intensity", 0.5) for p in phases]
    n = len(intensities)

    # Simple pattern matching
    first_half = np.mean(intensities[:n // 2])
    second_half = np.mean(intensities[n // 2:])
    mid_peak = intensities[n // 2] if n >= 3 else intensities[0]
    diff = second_half - first_half
    mid_diff = mid_peak - (first_half + second_half) / 2

    # Detect oscillations
    diffs = np.diff(intensities)
    sign_changes = np.sum(np.abs(np.diff(np.sign(diffs)))) / 2
    if sign_changes >= 2:
        return "oscillating"

    # Stable: very little change across all phases
    if abs(diff) < 0.03 and abs(mid_diff) < 0.03:
        return "stable"

    # Monotonic
    if all(d >= -0.01 for d in diffs):
        return "monotonic_up"
    if all(d <= 0.01 for d in diffs):
        return "monotonic_down"

    # Shape-based
    if mid_diff > 0.05:
        return "inverted_U" if first_half > second_half else "rise_fall"
    elif mid_diff < -0.05:
        return "U_shape" if second_half > first_half else "fall_rise"

    return "rise_fall" if diff < 0 else "fall_rise"


def compute_sentiment_entropy(phases: list[dict]) -> float:
    """计算情感相的香农熵。

    熵越高 → 热梗的情感演变越复杂/不可预测
    熵越低 → 情感轨迹越简单/可预测

    Args:
        phases: 情感阶段列表，每个含 intensity

    Returns:
        归一化熵值 [0, 1]
    """
    intensities = np.array([p.get("intensity", 0.5) for p in phases])
    if len(intensities) < 2:
        return 0.0

    # Discretize intensity into bins
    bins = 5
    hist, _ = np.histogram(intensities, bins=bins, range=(0, 1))
    hist = hist / hist.sum()
    hist = hist[hist > 0]

    # Normalize by log(bins)
    max_entropy = np.log(bins)
    if max_entropy == 0:
        return 0.0
    return float(-np.sum(hist * np.log(hist)) / max_entropy)


def build_sentiment_arcs(curator: MemeCurator) -> list[SentimentArc]:
    """为策展数据中的每个热梗构建情感弧线分析。

    Args:
        curator: MemeCurator 实例

    Returns:
        SentimentArc 列表
    """
    arcs = []
    for meme in curator.memes:
        phases = meme.sentiment_arc
        if not phases:
            continue

        arc_type = classify_sentiment_arc(phases)
        intensities = [p.get("intensity", 0.5) for p in phases]
        entropy = compute_sentiment_entropy(phases)

        arcs.append(SentimentArc(
            name=meme.name,
            category=meme.category,
            arc_type=arc_type,
            phases=phases,
            sentiment_range=(min(intensities), max(intensities)),
            mean_intensity=round(float(np.mean(intensities)), 3),
            entropy=round(entropy, 3),
            chaos_correlation=round(meme.chaos_position, 3),
        ))

    return arcs


def arc_type_distribution(arcs: list[SentimentArc]) -> dict[str, int]:
    """情感弧线类型分布。"""
    dist = {}
    for a in arcs:
        dist[a.arc_type] = dist.get(a.arc_type, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))


def entropy_by_category(arcs: list[SentimentArc]) -> dict[str, list[float]]:
    """按类别统计情感熵分布。"""
    from collections import defaultdict
    result = defaultdict(list)
    for a in arcs:
        result[a.category].append(a.entropy)
    return dict(result)


def chaos_vs_sentiment(arcs: list[SentimentArc]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取混沌位置与情感特征的关联。

    Returns:
        (chaos_positions, mean_intensities, entropies)
    """
    chaos = np.array([a.chaos_correlation for a in arcs])
    intensity = np.array([a.mean_intensity for a in arcs])
    entropy = np.array([a.entropy for a in arcs])
    return chaos, intensity, entropy


# ═══════════════════════════════════════════════
# Script entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("MemeticChaos — Sentiment Analysis")
    print("=" * 60)

    curator = MemeCurator()
    arcs = build_sentiment_arcs(curator)

    print(f"\nTotal arcs: {len(arcs)}")

    # Arc type distribution
    print("\n--- Arc Type Distribution ---")
    dist = arc_type_distribution(arcs)
    for t, count in dist.items():
        bar = "█" * count
        print(f"  {t:16s}: {count:2d} {bar}")

    # Top by sentiment entropy
    print("\n--- Top 5 by Sentiment Entropy (most complex emotional trajectory) ---")
    by_entropy = sorted(arcs, key=lambda a: a.entropy, reverse=True)
    for a in by_entropy[:5]:
        print(f"  {a.name:12s} [{a.category}]: H={a.entropy:.3f}, "
              f"type={a.arc_type}, chaos={a.chaos_correlation:+.2f}")

    # Category-level entropy
    print("\n--- Mean Sentiment Entropy by Category ---")
    ent_by_cat = entropy_by_category(arcs)
    for cat in sorted(ent_by_cat.keys()):
        vals = ent_by_cat[cat]
        print(f"  {cat:12s}: mean={np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Chaos vs sentiment correlation
    c, i, e = chaos_vs_sentiment(arcs)
    corr_intensity = float(np.corrcoef(c, i)[0, 1]) if len(c) > 2 else 0
    corr_entropy = float(np.corrcoef(c, e)[0, 1]) if len(c) > 2 else 0
    print(f"\n--- Correlations ---")
    print(f"  Chaos vs Intensity: r = {corr_intensity:.3f}")
    print(f"  Chaos vs Entropy:   r = {corr_entropy:.3f}")

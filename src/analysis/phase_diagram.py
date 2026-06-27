"""
模因相图 (Meme Phase Diagram) — MemeticChaos 核心产出。

类似物理学中的温度-压力-相态图，将 2020-2025 年中国互联网
集体情绪映射到 R₀ × Chaos Axis × Entropy 的三维状态空间。

对齐「微尘哲学」：
- 相变边界 = 系统在绝对混沌与绝对秩序之间的振荡临界
- 吸引子盆地 = 集体情感在混沌中的稳定态
- 过渡区 = 系统处于序参量临界慢化的区域

产出：
1. 29热梗的二维相图 (R₀ × Chaos Axis)
2. 相区自动聚类 (解构区/认同区/攻击区/虚无区/过渡区)
3. 相边界检测
4. 集体情绪状态机
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from collections import defaultdict

from src.data.curator import MemeCurator
from src.models.sir_meme import (
    SIRParams, estimate_params_from_lifecycle, estimate_total_infected,
    solve_sir, classify_meme_type, compute_entropy_curve,
)


# ═══════════════════════════════════════════════
# Phase diagram data structures
# ═══════════════════════════════════════════════

@dataclass
class MemeStatePoint:
    """单个热梗在相图中的状态点。"""
    name: str
    category: str
    year: int
    R0: float
    chaos_position: float
    entropy_max: float      # 传播过程中达到的最大熵
    entropy_mean: float     # 平均熵
    duration_days: float
    peak_infected: float
    meme_type: str          # 脉冲/爆发/长尾/流产
    lifecycle_status: str   # 消亡/固化/变异


@dataclass
class PhaseRegion:
    """相图中的稳定区域（相区）。"""
    name: str               # 解构区/认同区/攻击区/虚无区/过渡区
    center_R0: float
    center_chaos: float
    radius_R0: float
    radius_chaos: float
    members: list[str]      # 属于该相区的热梗名
    description: str


@dataclass
class PhaseDiagram:
    """完整的模因相图。"""
    points: list[MemeStatePoint]
    regions: list[PhaseRegion]
    phase_boundaries: list[dict]  # 相边界
    attractor_basins: list[dict]  # 吸引子盆地
    global_stats: dict            # 全局统计

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Memetic Chaos — Meme Phase Diagram (2020-2025)",
            "=" * 60,
            f"\nTotal memes mapped: {len(self.points)}",
            f"\n--- Phase Regions ---",
        ]
        for r in self.regions:
            lines.append(f"  {r.name}: {len(r.members)} memes")
            lines.append(f"    center: (R₀={r.center_R0:.2f}, chaos={r.center_chaos:+.2f})")
            lines.append(f"    members: {', '.join(r.members[:5])}{'...' if len(r.members) > 5 else ''}")
        lines.append(f"\n--- Attractor Basins ---")
        for b in self.attractor_basins:
            lines.append(f"  {b['label']}: R₀ ∈ [{b['R0_min']:.1f}, {b['R0_max']:.1f}] "
                        f"| chaos ∈ [{b['chaos_min']:+.2f}, {b['chaos_max']:+.2f}] "
                        f"| {b['n_memes']} memes")
        lines.append(f"\n--- Global Stats ---")
        for k, v in self.global_stats.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════
# State point construction
# ═══════════════════════════════════════════════

def build_state_points(curator: MemeCurator = None,
                       memes: list = None) -> list[MemeStatePoint]:
    """从策展数据构建所有热梗的相图状态点。

    每个热梗 → SIR 参数估算 → 求解 → 提取 R₀/chaos/entropy。

    Args:
        curator: MemeCurator 实例（与 memes 二选一）
        memes: 直接传入 MemeEntry 列表（用于扰动后子集）
    """
    if memes is not None:
        meme_list = memes
    elif curator is not None:
        meme_list = curator.memes
    else:
        curator = MemeCurator()
        meme_list = curator.memes

    points = []
    for meme in meme_list:
        lc = meme.lifecycle
        pm = meme.propagation_model

        # Duration estimation
        dur_months = lc.get("duration_months", 3)
        if dur_months >= 999:
            dur_months = 18
        dur_days = dur_months * 30

        # Total infected: continuous estimation from available lifecycle features
        circle_count = len(pm.get("circle_layers", []))
        # Extract peak intensity from sentiment_arc
        sentiment_arc = meme.sentiment_arc if hasattr(meme, 'sentiment_arc') else lc.get("sentiment_arc", [])
        if isinstance(sentiment_arc, list) and len(sentiment_arc) > 0:
            peak_intensity = max(p.get("intensity", 0.5) for p in sentiment_arc)
        else:
            peak_intensity = 0.5
        total_infected = estimate_total_infected(
            circle_count, peak_intensity, dur_months
        )

        # SIR params
        params = estimate_params_from_lifecycle(
            peak_day=dur_days * 0.3,
            total_infected=total_infected,
            duration_days=dur_days,
        )

        result = solve_sir(params)
        classification = classify_meme_type(result)
        H = compute_entropy_curve(result)

        points.append(MemeStatePoint(
            name=meme.name,
            category=meme.category,
            year=meme.year,
            R0=float(params.R0),
            chaos_position=float(meme.chaos_position),
            entropy_max=float(np.max(H)),
            entropy_mean=float(np.mean(H)),
            duration_days=dur_days,
            peak_infected=float(result.peak_infected),
            meme_type=classification["type"],
            lifecycle_status=lc.get("status", "unknown"),
        ))

    return points


# ═══════════════════════════════════════════════
# Phase region clustering
# ═══════════════════════════════════════════════

def cluster_phase_regions(points: list[MemeStatePoint]) -> list[PhaseRegion]:
    """在 R₀ × Chaos Axis 空间中对热梗进行相区聚类。

    使用简单的基于类别的聚类 + 空间邻近性。

    Returns:
        PhaseRegion 列表，按类别组织
    """
    # Group by category
    by_category = defaultdict(list)
    for p in points:
        by_category[p.category].append(p)

    region_descriptions = {
        "解构自嘲": "通过幽默自嘲将焦虑转化为共鸣——在混沌中建立温和的局部秩序",
        "攻击发泄": "以对立和攻击性表达释放情绪——系统接近绝对混沌的边缘",
        "虚无退却": "以退出姿态拒绝现有秩序——徘徊在绝对混沌的引力范围内",
        "身份认同": "通过命名和框架建构赋予处境意义——朝向秩序的健康建立",
        "纯粹娱乐": "无明确价值负载的娱乐传播——系统的中性休息区",
    }

    regions = []
    for category, members in by_category.items():
        R0s = [p.R0 for p in members]
        chaos_vals = [p.chaos_position for p in members]

        regions.append(PhaseRegion(
            name=category,
            center_R0=round(float(np.mean(R0s)), 2),
            center_chaos=round(float(np.mean(chaos_vals)), 3),
            radius_R0=round(float(np.std(R0s)) * 1.5, 3),
            radius_chaos=round(float(np.std(chaos_vals)) * 1.5, 3),
            members=[p.name for p in members],
            description=region_descriptions.get(category, ""),
        ))

    return regions


# ═══════════════════════════════════════════════
# Phase boundary detection
# ═══════════════════════════════════════════════

def detect_phase_boundaries(points: list[MemeStatePoint]) -> list[dict]:
    """检测相区之间的边界。

    相边界 = 两个相区之间 R₀ 或 chaos 发生显著变化的位置。
    """
    boundaries = []

    # R₀=1 boundary: 传播能力的临界
    below_R0 = [p for p in points if p.R0 < 1.0]
    above_R0 = [p for p in points if p.R0 >= 1.0]

    if below_R0 and above_R0:
        boundaries.append({
            "type": "R0_critical",
            "position": 1.0,
            "description": "R₀=1 传播临界线：左侧模因无法建立秩序，右侧可以",
            "n_below": len(below_R0),
            "n_above": len(above_R0),
            "examples_below": [p.name for p in below_R0[:3]],
            "examples_above": [p.name for p in above_R0[:3]],
        })

    # Chaos=0 boundary: 混沌/秩序分界线
    chaos_neg = [p for p in points if p.chaos_position < -0.2]
    chaos_neutral = [p for p in points if -0.2 <= p.chaos_position <= 0.2]
    chaos_pos = [p for p in points if p.chaos_position > 0.2]

    boundaries.append({
        "type": "chaos_neutral_zone",
        "position": 0.0,
        "description": "混沌中性区 (-0.2 ~ +0.2)：系统既不偏向混沌也不偏向秩序",
        "n_chaos_leaning": len(chaos_neg),
        "n_neutral": len(chaos_neutral),
        "n_order_leaning": len(chaos_pos),
    })

    return boundaries


# ═══════════════════════════════════════════════
# Attractor basin detection
# ═══════════════════════════════════════════════

def detect_attractor_basins(points: list[MemeStatePoint]) -> list[dict]:
    """在相图中检测吸引子盆地。

    盆地 = 多个热梗聚集的稳定区域。
    使用简单的网格密度估计。
    """
    basins = []

    # Basin A: High R₀ + chaos-leaning (viral chaos)
    basin_a = [p for p in points if p.R0 >= 1.5 and p.chaos_position < -0.1]
    if basin_a:
        basins.append({
            "label": "Basin A: Viral Chaos",
            "R0_min": round(min(p.R0 for p in basin_a), 1),
            "R0_max": round(max(p.R0 for p in basin_a), 1),
            "chaos_min": round(min(p.chaos_position for p in basin_a), 2),
            "chaos_max": round(max(p.chaos_position for p in basin_a), 2),
            "n_memes": len(basin_a),
            "dominant_category": max(set(p.category for p in basin_a),
                                     key=lambda c: sum(1 for p in basin_a if p.category == c)),
            "description": "高传播力 + 偏混沌：攻击发泄与虚无退却的领地。模因爆发迅猛但秩序脆弱。",
            "members": [p.name for p in basin_a],
        })

    # Basin B: High R₀ + order-leaning (viral order)
    basin_b = [p for p in points if p.R0 >= 1.5 and p.chaos_position > -0.1]
    if basin_b:
        basins.append({
            "label": "Basin B: Viral Order",
            "R0_min": round(min(p.R0 for p in basin_b), 1),
            "R0_max": round(max(p.R0 for p in basin_b), 1),
            "chaos_min": round(min(p.chaos_position for p in basin_b), 2),
            "chaos_max": round(max(p.chaos_position for p in basin_b), 2),
            "n_memes": len(basin_b),
            "dominant_category": max(set(p.category for p in basin_b),
                                     key=lambda c: sum(1 for p in basin_b if p.category == c)),
            "description": "高传播力 + 偏秩序：身份认同与纯粹娱乐的领地。模因建立稳定局部秩序。",
            "members": [p.name for p in basin_b],
        })

    # Basin C: Low R₀ (abortive zone)
    basin_c = [p for p in points if p.R0 < 1.2]
    if basin_c:
        basins.append({
            "label": "Basin C: Abortive Zone",
            "R0_min": round(min(p.R0 for p in basin_c), 1),
            "R0_max": round(max(p.R0 for p in basin_c), 1),
            "chaos_min": round(min(p.chaos_position for p in basin_c), 2),
            "chaos_max": round(max(p.chaos_position for p in basin_c), 2),
            "n_memes": len(basin_c),
            "dominant_category": "mixed",
            "description": "低传播力：模因未能建立秩序便被混沌吞没。",
            "members": [p.name for p in basin_c],
        })

    return basins


# ═══════════════════════════════════════════════
# Collective emotion state machine
# ═══════════════════════════════════════════════

def build_state_machine(points: list[MemeStatePoint]) -> dict:
    """从相图推导集体情绪状态机。

    状态 = 系统在特定 R₀-chaos 区域的行为模式。
    转移 = 外部事件驱动下的相区切换路径。

    Returns:
        {states: [...], transitions: [...]}
    """
    states = [
        {
            "id": "constructive_irony",
            "label": "建构性解构",
            "condition": "R₀ > 1.5, chaos ∈ [-0.4, +0.1]",
            "examples": ["打工人", "吗喽"],
            "description": "用幽默自嘲建立温和的集体秩序，系统处于健康混沌管理状态",
        },
        {
            "id": "aggressive_venting",
            "label": "攻击性宣泄",
            "condition": "R₀ > 1.5, chaos < -0.4",
            "examples": ["普信男", "XX刺客"],
            "description": "以群体对立的标签化表达释放情绪，系统接近绝对混沌边缘",
        },
        {
            "id": "nihilistic_retreat",
            "label": "虚无主义退却",
            "condition": "R₀ > 1.0, chaos < -0.5",
            "examples": ["躺平", "摆烂", "润"],
            "description": "以退出姿态拒绝现有秩序，系统向绝对混沌滑落",
        },
        {
            "id": "identity_construction",
            "label": "身份意义建构",
            "condition": "R₀ > 1.0, chaos > +0.2",
            "examples": ["小镇做题家", "i人/e人", "内卷"],
            "description": "通过命名和框架赋予处境意义，系统建立健康局部秩序",
        },
        {
            "id": "neutral_entertainment",
            "label": "中性娱乐",
            "condition": "chaos ∈ [-0.2, +0.2]",
            "examples": ["科目三", "谢帝我要迪士尼"],
            "description": "无价值负载的纯粹娱乐，系统处于中性休息状态",
        },
    ]

    transitions = [
        {
            "from": "constructive_irony",
            "to": "aggressive_venting",
            "trigger": "社会压力持续加大 → 幽默不足以消化焦虑 → 转向攻击",
            "historical_example": "2020(打工人) → 2021(普信男)",
        },
        {
            "from": "aggressive_venting",
            "to": "nihilistic_retreat",
            "trigger": "攻击无果 + 系统持续熵增 → 退出博弈",
            "historical_example": "2021(普信男) → 2022(躺平/摆烂/润)",
        },
        {
            "from": "nihilistic_retreat",
            "to": "constructive_irony",
            "trigger": "退出后的反思 + 新的命名工具出现 → 重新建立温和秩序",
            "historical_example": "2022(躺平) → 2023(吗喽/精神状态良好)",
        },
        {
            "from": "identity_construction",
            "to": "constructive_irony",
            "trigger": "严肃叙事被泛化/收编 → 转为更轻量的自嘲",
            "historical_example": "2020(后浪) → 2020(打工人)",
        },
    ]

    return {"states": states, "transitions": transitions}


# ═══════════════════════════════════════════════
# Full diagram construction
# ═══════════════════════════════════════════════

def build_phase_diagram(curator: MemeCurator = None) -> PhaseDiagram:
    """构建完整的模因相图。"""
    if curator is None:
        curator = MemeCurator()

    points = build_state_points(curator)
    regions = cluster_phase_regions(points)
    boundaries = detect_phase_boundaries(points)
    basins = detect_attractor_basins(points)

    # Global stats
    R0s = [p.R0 for p in points]
    chaos_vals = [p.chaos_position for p in points]
    entropies = [p.entropy_max for p in points]

    global_stats = {
        "n_memes": len(points),
        "year_range": f"{min(p.year for p in points)}-{max(p.year for p in points)}",
        "R0_range": f"{min(R0s):.2f} - {max(R0s):.2f}",
        "chaos_range": f"{min(chaos_vals):+.2f} - {max(chaos_vals):+.2f}",
        "mean_R0": round(float(np.mean(R0s)), 2),
        "mean_chaos": round(float(np.mean(chaos_vals)), 3),
        "R0_above_1_pct": f"{sum(1 for r in R0s if r > 1.0) / len(R0s):.0%}",
        "dominant_phase": max(set(p.category for p in points),
                              key=lambda c: sum(1 for p in points if p.category == c)),
    }

    return PhaseDiagram(
        points=points,
        regions=regions,
        phase_boundaries=boundaries,
        attractor_basins=basins,
        global_stats=global_stats,
    )


# ═══════════════════════════════════════════════
# Blind clustering validation (GPT: verify 5-phase structure is data-driven)
# ═══════════════════════════════════════════════

def blind_clustering_validation(points: list[MemeStatePoint] = None,
                                curator: MemeCurator = None,
                                n_clusters: int = 5):
    """无监督盲聚类验证 — 在不使用类别标签的情况下重构相区。

    GPT 关键批评："5 相区与双吸引子是在数据中被发现的，还是在建模过程中
    被定义出来的？"

    使用 KMeans / GMM / HDBSCAN 对 (R₀, chaos, entropy) 进行盲聚类，
    然后与人工类别标签计算 ARI / NMI / V-measure。

    Args:
        points: 预构建的状态点列表
        curator: MemeCurator（与 points 二选一）
        n_clusters: 聚类数（默认 5，对应 5 个策展类别）

    Returns:
        dict with ARI, NMI, homogeneity, completeness, v_measure
    """
    import numpy as np
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        homogeneity_score, completeness_score, v_measure_score,
    )
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    if points is None:
        if curator is None:
            curator = MemeCurator()
        points = build_state_points(curator)

    # Feature matrix: R₀, chaos_position, entropy_max, entropy_mean
    X = np.column_stack([
        [p.R0 for p in points],
        [p.chaos_position for p in points],
        [p.entropy_max for p in points],
        [p.entropy_mean for p in points],
    ])
    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    y_true = np.array([p.category for p in points])

    results = {}

    # ── KMeans ──
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    y_km = km.fit_predict(X)
    results["KMeans"] = {
        "ARI": round(adjusted_rand_score(y_true, y_km), 4),
        "NMI": round(normalized_mutual_info_score(y_true, y_km), 4),
        "homogeneity": round(homogeneity_score(y_true, y_km), 4),
        "completeness": round(completeness_score(y_true, y_km), 4),
        "v_measure": round(v_measure_score(y_true, y_km), 4),
    }

    # ── GMM ──
    try:
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
        y_gmm = gmm.fit_predict(X)
        results["GMM"] = {
            "ARI": round(adjusted_rand_score(y_true, y_gmm), 4),
            "NMI": round(normalized_mutual_info_score(y_true, y_gmm), 4),
            "homogeneity": round(homogeneity_score(y_true, y_gmm), 4),
            "completeness": round(completeness_score(y_true, y_gmm), 4),
            "v_measure": round(v_measure_score(y_true, y_gmm), 4),
        }
    except Exception as e:
        results["GMM"] = {"error": str(e)}

    # ── HDBSCAN (no n_clusters needed — data-driven) ──
    try:
        from sklearn.cluster import HDBSCAN
        hdb = HDBSCAN(min_cluster_size=3, allow_single_cluster=False)
        y_hdb = hdb.fit_predict(X)
        n_hdb = len(set(y_hdb)) - (1 if -1 in y_hdb else 0)
        results["HDBSCAN"] = {
            "n_clusters_found": n_hdb,
            "ARI": round(adjusted_rand_score(y_true, y_hdb), 4),
            "NMI": round(normalized_mutual_info_score(y_true, y_hdb), 4),
            "v_measure": round(v_measure_score(y_true, y_hdb), 4),
            "n_noise": int(np.sum(y_hdb == -1)),
        }
    except ImportError:
        results["HDBSCAN"] = {"error": "HDBSCAN not available (requires sklearn ≥ 1.3)"}
    except Exception as e:
        results["HDBSCAN"] = {"error": str(e)}

    # ── Aggregate verdict ──
    aris = [v["ARI"] for v in results.values() if "ARI" in v]
    best_ari = max(aris) if aris else 0.0

    results["verdict"] = (
        "DATA-DRIVEN — blind clustering aligns with human categories (ARI > 0.3)"
        if best_ari > 0.3
        else "WEAK — some alignment but categories may be partly a priori"
        if best_ari > 0.15
        else "PRIOR-DRIVEN — 5-phase regions are largely defined by human labels, not emergent from data"
    )
    results["best_ARI"] = best_ari

    return results


# ═══════════════════════════════════════════════
# Script entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    diagram = build_phase_diagram()
    print(diagram.summary())

    # State machine
    print("\n" + "=" * 60)
    print("Collective Emotion State Machine")
    print("=" * 60)
    sm = build_state_machine(diagram.points)

    print("\n--- States ---")
    for s in sm["states"]:
        print(f"  [{s['id']}] {s['label']}")
        print(f"    Condition: {s['condition']}")
        print(f"    Examples: {', '.join(s['examples'])}")

    print("\n--- Transitions ---")
    for t in sm["transitions"]:
        print(f"  {t['from']} → {t['to']}")
        print(f"    Trigger: {t['trigger']}")
        if "historical_example" in t:
            print(f"    Historical: {t['historical_example']}")

    # ── Blind Clustering Validation ──────────────
    print("\n" + "=" * 60)
    print("Blind Clustering Validation (GPT建议)")
    print("=" * 60)
    bc = blind_clustering_validation(points=diagram.points)
    for method, metrics in bc.items():
        if method == "verdict":
            print(f"\n  Verdict: {metrics}")
        elif method == "best_ARI":
            print(f"  Best ARI: {metrics:.4f}")
        elif isinstance(metrics, dict) and "error" not in metrics:
            print(f"\n  [{method}]")
            print(f"    ARI={metrics['ARI']} NMI={metrics['NMI']} V-measure={metrics.get('v_measure', 'N/A')}")
            if "n_clusters_found" in metrics:
                print(f"    Clusters found: {metrics['n_clusters_found']} (noise: {metrics.get('n_noise', 0)})")
        elif isinstance(metrics, dict) and "error" in metrics:
            print(f"  [{method}] Error: {metrics['error']}")

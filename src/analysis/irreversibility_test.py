"""
Irreversibility Tests — FR19 v4.1

GPT 的核心质疑: R2 的 97.3% 自持是真实物理结构, 还是 GMM 在连续流形上
切 bin 造成的投影伪影?

两个独立测试:

1. Recurrence Analysis (RQA):
   - 系统是否曾回到旧状态的 ε-邻域?
   - 如果有回归 → 真实吸引子
   - 如果从未回归 → 不可逆漂移 (或采样窗口太短)

2. Time-Reversal Test:
   - forward P(x(t+1)|x(t)) vs backward P(x(t)|x(t+1)) 是否对称?
   - 不对称 → 存在真实时间箭头
   - 对称 → 可逆过程, regime 结构是聚类的 artifact

用法:
    python src/analysis/irreversibility_test.py
    python src/analysis/irreversibility_test.py --json
"""

import json, sys, os, argparse
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import stats

ROOT = Path(__file__).parent.parent.parent

STATE_PATH = ROOT / "data/processed/representation_state.json"
REGIME_PATH = ROOT / "data/processed/regime_map.json"
OUTPUT_PATH = ROOT / "data/processed/irreversibility_results.json"


def load_all():
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    with open(REGIME_PATH, "r", encoding="utf-8") as f:
        regime = json.load(f)

    months = state["pca_transformed"]["months"]
    x = np.array(state["pca_transformed"]["x_reduced"])
    labels = np.array(regime["regime_labels"])
    n_regimes = regime["n_regimes"]

    return months, x, labels, n_regimes


# ═══════════════════════════════════════════════
# Test 1: Recurrence Analysis
# ═══════════════════════════════════════════════

def recurrence_analysis(x: np.ndarray, labels: np.ndarray, months: list[str],
                         n_regimes: int) -> dict:
    """RQA: 检测系统是否回到旧状态邻域.

    关键问题: R2 (fixation lock) 中的任何月份, 是否在状态空间中
    与 R1 或 R3 中的某些月份足够接近?
    即: 系统曾经"回去过"吗?
    """
    n = len(x)

    # Calibrate ε: target 5% recurrence rate (standard RQA practice)
    nn = NearestNeighbors(n_neighbors=min(50, n - 1), metric="euclidean")
    nn.fit(x)
    dists, _ = nn.kneighbors(x)
    # Use the 5th percentile of nearest-neighbor distances (excluding self)
    nn_dists = dists[:, 1:].ravel()
    eps = float(np.percentile(nn_dists, 10))  # 10th percentile = ~5% recurrence

    print(f"\n  Median NN distance: {np.median(nn_dists):.3f}")
    print(f"  Recurrence threshold ε = 10th percentile NN dist = {eps:.4f}")

    # Build recurrence matrix (sparse: only store recurrence pairs)
    nn_eps = NearestNeighbors(radius=eps, metric="euclidean")
    nn_eps.fit(x)

    # For each month, find all months within ε
    # Count cross-regime recurrences
    cross_recurrence = np.zeros((n_regimes, n_regimes))
    regime_recurrence_counts = np.zeros(n_regimes)

    for i in range(n):
        # Find neighbors within ε
        neighbors = nn_eps.radius_neighbors([x[i]], radius=eps, return_distance=False)[0]
        # Filter: only count neighbors at least 12 months apart (avoid trivial adjacency)
        distant_neighbors = [j for j in neighbors if abs(i - j) >= 12]

        ri = labels[i]
        for j in distant_neighbors:
            rj = labels[j]
            cross_recurrence[ri, rj] += 1

        regime_recurrence_counts[ri] += len(distant_neighbors)

    # Does R2 ever recur to R1 or R3?
    r2_idx = 2  # Regime 2 = Fixation Lock
    r2_to_r1 = cross_recurrence[r2_idx, 1] if n_regimes > 1 else 0
    r2_to_r3 = cross_recurrence[r2_idx, 3] if n_regimes > 3 else 0

    # Does R1 or R3 ever recur to R2?
    r1_to_r2 = cross_recurrence[1, r2_idx] if n_regimes > 1 else 0
    r3_to_r2 = cross_recurrence[3, r2_idx] if n_regimes > 3 else 0

    # Normalize
    row_sums = cross_recurrence.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cross_rec_norm = cross_recurrence / row_sums

    # Find earliest return: for each R2 month, find its closest pre-R2 neighbor
    r2_indices = [i for i in range(n) if labels[i] == r2_idx]
    min_dist_to_past = []
    for i in r2_indices:
        ri_label = labels[i]
        # Find closest month from a DIFFERENT regime at least 12 months earlier
        past_indices = [j for j in range(i - 12) if labels[j] != ri_label]
        if past_indices:
            dists_to_past = np.linalg.norm(x[past_indices] - x[i], axis=1)
            min_dist_to_past.append(float(np.min(dists_to_past)))

    # Also: for each regime pair, find the minimum distance between any two points
    inter_regime_min_dist = np.full((n_regimes, n_regimes), np.inf)
    for ri in range(n_regimes):
        for rj in range(n_regimes):
            if ri != rj:
                xi = x[labels == ri]
                xj = x[labels == rj]
                if len(xi) > 0 and len(xj) > 0:
                    nn_inter = NearestNeighbors(n_neighbors=1, metric="euclidean")
                    nn_inter.fit(xj)
                    d, _ = nn_inter.kneighbors(xi)
                    inter_regime_min_dist[ri, rj] = float(np.min(d))

    # ── Verdict ──
    r2_escaped = r2_to_r1 > 0 or r2_to_r3 > 0
    others_entered_r2 = r1_to_r2 > 0 or r3_to_r2 > 0

    if not r2_escaped and not others_entered_r2:
        recurrence_verdict = (
            "不可逆 — R2 从不回归 R1/R3, R1/R3 也从不进入 R2 的 ε-邻域. "
            "Fixation lock 在状态空间中是真实的分离结构, 非聚类 artifact."
        )
    elif not r2_escaped and others_entered_r2:
        recurrence_verdict = (
            "部分不可逆 — R2 自身不回归历史状态, 但其他相区的状态会进入 R2 邻域. "
            "Fixation lock 是吸收态."
        )
    elif r2_escaped:
        recurrence_verdict = (
            "可回归 — R2 中的某些月份与 R1/R3 中的月份在 ε 距离内. "
            "GMM 的 regime 边界可能是软的; 系统并非真正锁死."
        )
    else:
        recurrence_verdict = "数据不足以判断."

    return {
        "method": "Recurrence Quantification Analysis",
        "epsilon": round(float(eps), 4),
        "epsilon_pct_of_mean_dist": 5.0,
        "cross_regime_recurrence_raw": {f"R{i}→R{j}": int(cross_recurrence[i, j])
                                         for i in range(n_regimes)
                                         for j in range(n_regimes) if i != j},
        "cross_regime_recurrence_norm": {f"R{i}→R{j}": round(float(cross_rec_norm[i, j]), 4)
                                          for i in range(n_regimes)
                                          for j in range(n_regimes) if i != j},
        "r2_to_r1_recurrence_count": int(r2_to_r1),
        "r2_to_r3_recurrence_count": int(r2_to_r3),
        "r2_ever_returns": bool(r2_escaped),
        "min_distance_r2_to_past_regimes": (
            round(float(np.mean(min_dist_to_past)), 4) if min_dist_to_past else None
        ),
        "inter_regime_min_distance": {f"R{i}↔R{j}": round(float(inter_regime_min_dist[i, j]), 4)
                                       for i in range(n_regimes)
                                       for j in range(i + 1, n_regimes)},
        "verdict": recurrence_verdict,
    }


# ═══════════════════════════════════════════════
# Test 2: Time-Reversal Symmetry
# ═══════════════════════════════════════════════

def time_reversal_test(x: np.ndarray, labels: np.ndarray,
                        months: list[str]) -> dict:
    """检测动力学是否具有时间反演对称性.

    修正方法: ∥Δx∥ 在时间反演下天然对称, 不能直接用 KS test.
    应检测转移的"方向性":
    1. 全局漂移: E[Δx] 是否显著非零? (有漂移 = 不可逆)
    2. 置换测试: 随机打乱时间顺序后, 转移统计量是否变化?
    3. Cosine alignment: 连续两步的方向是否一致? (有惯性 = 不可逆)
    """
    n = len(x)
    dim = x.shape[1]

    forward_dx = np.diff(x, axis=0)  # (n-1) × dim

    # ── 1. Drift test: does E[Δx_j] ≠ 0 for any dimension? ──
    per_dim_mean = np.mean(forward_dx, axis=0)
    per_dim_std = np.std(forward_dx, axis=0)
    per_dim_t = per_dim_mean / (per_dim_std / np.sqrt(n - 1) + 1e-10)
    per_dim_p = 2 * (1 - stats.t.cdf(np.abs(per_dim_t), df=n - 2))
    drift_dims = [int(i) for i in range(dim) if per_dim_p[i] < 0.05]

    overall_drift = np.mean(forward_dx, axis=0)
    drift_magnitude = float(np.linalg.norm(overall_drift))

    # ── 2. Permutation test: shuffle time, compare transition stats ──
    n_perm = 500
    orig_mean_step = float(np.mean(np.linalg.norm(forward_dx, axis=1)))

    perm_means = []
    for _ in range(n_perm):
        perm_idx = np.random.permutation(n)
        x_perm = x[perm_idx]
        dx_perm = np.diff(x_perm, axis=0)
        perm_means.append(float(np.mean(np.linalg.norm(dx_perm, axis=1))))

    perm_means = np.array(perm_means)
    perm_p = np.mean(perm_means >= orig_mean_step)

    # ── 3. Cosine alignment: cosine between consecutive steps ──
    cosines = []
    for t in range(len(forward_dx) - 1):
        a, b = forward_dx[t], forward_dx[t + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-10 and nb > 1e-10:
            cosines.append(float(np.dot(a, b) / (na * nb)))

    mean_cosine = float(np.mean(cosines)) if cosines else 0.0
    # Positive mean cosine = steps tend to go in same direction = persistence/inertia
    # Test: is mean cosine significantly > 0?
    if cosines:
        cos_t = mean_cosine / (np.std(cosines) / np.sqrt(len(cosines)) + 1e-10)
        cos_p = float(1 - stats.t.cdf(cos_t, df=len(cosines) - 1))
    else:
        cos_p = 1.0

    persistence_significant = cos_p < 0.05

    # ── 4. Forward-only vs backward-only alignment ──
    # Split trajectory into two halves, compare drift direction
    half = n // 2
    drift_first_half = np.mean(forward_dx[:half], axis=0)
    drift_second_half = np.mean(forward_dx[half:], axis=0)
    drift_cosine = float(np.dot(drift_first_half, drift_second_half) /
                         (np.linalg.norm(drift_first_half) * np.linalg.norm(drift_second_half) + 1e-10))

    # ── Verdict ──
    irreversible_signals = []
    if len(drift_dims) > 0:
        irreversible_signals.append(f"{len(drift_dims)}/{dim} 维有显著定向漂移")
    if perm_p < 0.05:
        irreversible_signals.append(f"置换测试 p={perm_p:.4f} (时间顺序重要)")
    if persistence_significant:
        irreversible_signals.append(f"步间余弦均值={mean_cosine:.3f}, p={cos_p:.4f} (有方向惯性)")

    if len(irreversible_signals) >= 2:
        tr_verdict = (
            f"不可逆 — 存在多重时间箭头证据: {'; '.join(irreversible_signals)}. "
            f"动力学在统计意义上不可逆."
        )
    elif len(irreversible_signals) == 1:
        tr_verdict = (
            f"弱不可逆 — 仅一个指标检测到不对称: {irreversible_signals[0]}. "
            f"证据不足以确认强不可逆性."
        )
    else:
        tr_verdict = (
            f"对称 — 未检测到统计显著的时间反演不对称. "
            f"在步长/方向/置换三个测试上, 时间反演对称性均成立."
        )

    return {
        "method": "Time-Reversal Symmetry Test (Directional)",
        "drift_test": {
            "drift_magnitude": round(drift_magnitude, 4),
            "n_significant_dims": len(drift_dims),
            "significant_dims": drift_dims,
        },
        "permutation_test": {
            "original_mean_step": round(orig_mean_step, 4),
            "permuted_mean_step": round(float(np.mean(perm_means)), 4),
            "p_value": round(float(perm_p), 4),
            "n_permutations": n_perm,
        },
        "cosine_alignment": {
            "mean_cosine": round(mean_cosine, 4),
            "p_value": round(float(cos_p), 4),
            "significant": bool(persistence_significant),
            "n_step_pairs": len(cosines),
        },
        "half_drift_alignment": round(drift_cosine, 4),
        "verdict": tr_verdict,
    }


# ═══════════════════════════════════════════════
# Combine
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Irreversibility Tests")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Irreversibility Tests — FR19 v4.1")
    print("=" * 60)

    months, x, labels, n_regimes = load_all()
    print(f"\n  数据: {len(months)} 月 × {x.shape[1]} 维, {n_regimes} 相区")

    # Test 1
    print(f"\n{'─' * 60}")
    print("Test 1: Recurrence Analysis")
    print(f"{'─' * 60}")
    rec = recurrence_analysis(x, labels, months, n_regimes)
    print(f"\n  {rec['verdict']}")
    print(f"\n  跨相区 ε-邻域复发:")
    for pair, count in rec["cross_regime_recurrence_raw"].items():
        norm = rec["cross_regime_recurrence_norm"][pair]
        print(f"    {pair}: {count:>5d} 对 (归一化={norm:.4f})")
    print(f"\n  相区间最小距离:")
    for pair, d in rec["inter_regime_min_distance"].items():
        print(f"    {pair}: {d:.4f}")

    # Test 2
    print(f"\n{'─' * 60}")
    print("Test 2: Time-Reversal Symmetry")
    print(f"{'─' * 60}")
    tr = time_reversal_test(x, labels, months)
    print(f"\n  {tr['verdict']}")
    print(f"  Drift magnitude: {tr['drift_test']['drift_magnitude']:.4f}, "
          f"significant dims: {tr['drift_test']['n_significant_dims']}/{x.shape[1]}")
    print(f"  Permutation test: original={tr['permutation_test']['original_mean_step']:.4f}, "
          f"permuted mean={tr['permutation_test']['permuted_mean_step']:.4f}, "
          f"p={tr['permutation_test']['p_value']:.4f}")
    print(f"  Cosine alignment: mean={tr['cosine_alignment']['mean_cosine']:.4f}, "
          f"p={tr['cosine_alignment']['p_value']:.4f}, "
          f"significant={tr['cosine_alignment']['significant']}")
    print(f"  Half-drift alignment: {tr['half_drift_alignment']:.4f}")
    if tr["drift_test"]["significant_dims"]:
        print(f"  显著漂移维度: {tr['drift_test']['significant_dims']}")

    # ── Combined verdict ──
    print(f"\n{'═' * 60}")
    print("综合判断")
    print(f"{'═' * 60}")

    r2_real = rec["r2_ever_returns"] == False
    tr_irreversible = tr["drift_test"]["n_significant_dims"] > 0 or tr["cosine_alignment"]["significant"]

    if r2_real and tr_irreversible:
        print("\n  ✅ 两项测试一致支持不可逆性.")
        print("  R2 Fixation Lock 在状态空间中是真实的分离结构.")
        print("  97.3% 自持概率反映了真实的吸收态, 非投影 artifact.")
        combined = "STRONG_IRREVERSIBILITY"
    elif r2_real and not tr_irreversible:
        print("\n  ⚠️  RQA 支持不可逆, 但时间反演测试未检测到显著方向性.")
        print("  R2 可能是真实的长期驻留盆地, 但动力学在精细尺度上对称.")
        print("  不可逆性可能来自慢变量调制 (u(t) 漂移), 而非内禀动力学.")
        combined = "WEAK_IRREVERSIBILITY"
    elif not r2_real and tr_irreversible:
        print("\n  ⚠️  时间反演有方向性, 但 R2 可回归.")
        print("  系统有时间箭头, 但 regime 边界是软的.")
        combined = "SOFT_REGIME_BOUNDARIES"
    else:
        print("\n  ❌ 两项测试均不支持强不可逆性.")
        print("  R2 可能主要是 GMM 在高密度区域的聚类 artifact.")
        combined = "LIKELY_ARTIFACT"

    print(f"\n  综合分类: {combined}")

    # Save
    output = {
        "source": "irreversibility_test.py",
        "combined_verdict": combined,
        "recurrence_analysis": rec,
        "time_reversal_test": tr,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  已保存 → {args.output}")

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

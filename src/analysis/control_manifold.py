"""
Control Manifold Analysis — FR19 v4.1

核心假说: x(t) ∈ M(u(t)). u(t) 重塑势能地形, x(t) 在地形中滑动.
R2 (Fixation Lock) 不是内禀吸引子, 而是 u(t) 极端区间逼迫形成的亚稳相.

方法:
1. 51 维外部场 → PCA → 8 维 → Diffusion Map → 2-3 维控制轴 z(t)
2. 将 4 个 GMM regime 映射到 z-space
3. 检测: R2 是否占据 z(t) 的极端区间?
4. 反解: z₁, z₂ 各由哪些外部场关键词主导?

两个外部 AI 独立指向同一 P0. 这是整个物理框架的关键验证.

用法:
    python src/analysis/control_manifold.py
    python src/analysis/control_manifold.py --json
"""

import json, sys, os, argparse
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy import stats

ROOT = Path(__file__).parent.parent.parent

EXTERNAL_PATH = ROOT / "data/collector/external_field_2015_2025.json"
REGIME_PATH = ROOT / "data/processed/regime_map.json"
LEVEL1_PATH = ROOT / "data/processed/level1_hard_facts.json"
OUTPUT_PATH = ROOT / "data/processed/control_manifold.json"


def load_data():
    # External field
    with open(EXTERNAL_PATH, "r", encoding="utf-8") as f:
        ef = json.load(f)
    keywords = sorted(ef["data"].keys())
    raw_data = ef["data"]

    # Align to our canonical 127 months
    with open(LEVEL1_PATH, "r", encoding="utf-8") as f:
        l1 = json.load(f)
    months = l1["months"]

    u_matrix = np.zeros((len(months), len(keywords)))
    for i, m in enumerate(months):
        for j, kw in enumerate(keywords):
            u_matrix[i, j] = raw_data[kw].get(m, 0.0)

    # Regime labels
    with open(REGIME_PATH, "r", encoding="utf-8") as f:
        regime = json.load(f)
    labels = np.array(regime["regime_labels"])
    n_regimes = regime["n_regimes"]

    return months, keywords, u_matrix, labels, n_regimes


def diffusion_map(X: np.ndarray, n_components: int = 3, sigma: float = None) -> np.ndarray:
    """Diffusion map: 非线性流形学习.

    通过谱嵌入 (spectral embedding) 近似扩散映射.
    Steps: 距离 → 亲和矩阵 → 归一化 → 特征分解 → 保留前 k 个非平凡特征向量.
    """
    n = X.shape[0]

    # Pairwise distances
    dists = squareform(pdist(X, metric="euclidean"))

    # Adaptive sigma: median of k-th nearest neighbor distances
    if sigma is None:
        k = min(30, n - 1)
        sigma = float(np.median(np.sort(dists, axis=1)[:, k]))

    # Gaussian affinity kernel
    W = np.exp(-dists ** 2 / (2 * sigma ** 2))

    # Normalize: W → P = D^(-1) W (Markov transition matrix)
    D = np.sum(W, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-10))
    P = D_inv_sqrt @ W @ D_inv_sqrt  # symmetric normalized Laplacian

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(P)

    # Sort descending
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip first eigenvector (trivial, ~constant)
    # Take next n_components
    embedding = eigenvectors[:, 1:n_components + 1]

    # Scale by eigenvalues (diffusion distance)
    for j in range(n_components):
        lam = max(eigenvalues[j + 1], 1e-10)
        embedding[:, j] *= lam

    return embedding


def analyze_control_axes(u_pca: np.ndarray, u_dm: np.ndarray,
                          keywords: list[str], labels: np.ndarray,
                          months: list[str], n_regimes: int) -> dict:
    """分析控制轴: regime 分离度 + 关键词载荷 + R2 极端性."""

    # ── Regime separation in control space ──
    # For each pair of regimes, compute the distance between their centroids
    # relative to their internal spread

    z = u_dm  # use diffusion map as primary control space
    n_dims = z.shape[1]

    regime_centroids = {}
    regime_spread = {}
    for r in range(n_regimes):
        mask = labels == r
        regime_centroids[r] = np.mean(z[mask], axis=0)
        regime_spread[r] = float(np.mean(np.std(z[mask], axis=0)))

    # Pairwise separation
    separation = {}
    for ri in range(n_regimes):
        for rj in range(ri + 1, n_regimes):
            dist = float(np.linalg.norm(regime_centroids[ri] - regime_centroids[rj]))
            pooled_spread = (regime_spread[ri] + regime_spread[rj]) / 2
            separation[f"R{ri}↔R{rj}"] = {
                "centroid_distance": round(dist, 4),
                "separation_ratio": round(dist / max(pooled_spread, 1e-10), 2),
                "well_separated": dist > 2 * pooled_spread,
            }

    # ── R2 extremity test ──
    # Is R2 at an extreme of z-space compared to other regimes?
    r2_centroid = regime_centroids[2]
    all_centroids = np.array([regime_centroids[r] for r in range(n_regimes)])
    global_center = np.mean(all_centroids, axis=0)

    r2_dist_from_center = float(np.linalg.norm(r2_centroid - global_center))
    other_dists = [float(np.linalg.norm(regime_centroids[r] - global_center))
                   for r in range(n_regimes) if r != 2]
    mean_other_dist = np.mean(other_dists)

    r2_is_extreme = r2_dist_from_center > 1.5 * mean_other_dist

    # Per-axis extremity
    axis_extremity = {}
    for d in range(n_dims):
        axis_vals = z[:, d]
        r2_mean = float(np.mean(axis_vals[labels == 2]))
        overall_mean = float(np.mean(axis_vals))
        overall_std = float(np.std(axis_vals))
        z_score = (r2_mean - overall_mean) / max(overall_std, 1e-10)
        percentile = float(stats.percentileofscore(axis_vals, r2_mean))
        axis_extremity[f"z{d+1}"] = {
            "r2_mean": round(r2_mean, 4),
            "overall_mean": round(overall_mean, 4),
            "z_score": round(z_score, 3),
            "percentile": round(percentile, 1),
            "is_extreme": abs(z_score) > 1.0,
        }

    # ── Control axis interpretation: what keywords drive z₁, z₂? ──
    # Regress each z on original keywords to find top contributors
    # Use PCA loadings as proxy
    axis_keywords = {}
    for d in range(min(3, n_dims)):
        # Correlation between z_d and each original keyword
        correlations = []
        for j, kw in enumerate(keywords):
            if j < u_pca.shape[1]:
                corr = float(np.corrcoef(z[:, d], u_pca[:, j])[0, 1])
                if not np.isnan(corr):
                    correlations.append((kw, corr))
        correlations.sort(key=lambda x: -abs(x[1]))
        axis_keywords[f"z{d+1}"] = [
            {"keyword": kw, "correlation": round(c, 4)}
            for kw, c in correlations[:8]
        ]

    # ── Time evolution along control axes ──
    z_timeline = []
    for i, m in enumerate(months):
        z_timeline.append({
            "month": m,
            "regime": int(labels[i]),
            "z1": round(float(z[i, 0]), 4),
            "z2": round(float(z[i, 1]), 4),
            "z3": round(float(z[i, 2]), 4) if n_dims > 2 else 0.0,
        })

    # ── Verdict ──
    if r2_is_extreme:
        r2_verdict = (
            f"R2 是 z-space 的极端区域 — 它的质心到全局中心的距离 "
            f"({r2_dist_from_center:.2f}) 远超其他相区均值 ({mean_other_dist:.2f}). "
            f"这证实了 u(t) 逼迫假说: R2 不是内禀吸引子, 而是外部场极端配置下 "
            f"被'挤'出来的亚稳相."
        )
    else:
        r2_verdict = (
            f"R2 在 z-space 中不处于极端位置 (d={r2_dist_from_center:.2f} ≈ "
            f"其他相区均值 {mean_other_dist:.2f}). "
            f"R2 可能对应 z(t) 的某个中间配置区间, 而非边界."
        )

    return {
        "regime_separation": separation,
        "r2_extremity_test": {
            "r2_distance_from_center": round(r2_dist_from_center, 4),
            "mean_other_regime_distance": round(mean_other_dist, 4),
            "is_extreme": bool(r2_is_extreme),
            "verdict": r2_verdict,
        },
        "axis_extremity": axis_extremity,
        "control_axis_keywords": axis_keywords,
        "timeline": z_timeline,
    }


def main():
    parser = argparse.ArgumentParser(description="Control Manifold Analysis")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Control Manifold Analysis — FR19 v4.1")
    print("=" * 60)

    # Load
    print("\n[1/4] 加载数据...")
    months, keywords, u_matrix, labels, n_regimes = load_data()
    print(f"  外部场: {len(months)} 月 × {len(keywords)} 关键词")

    # PCA → 8 dim
    print("\n[2/4] PCA → Diffusion Map...")
    scaler = StandardScaler()
    u_scaled = scaler.fit_transform(u_matrix)

    pca = PCA(n_components=8)
    u_pca = pca.fit_transform(u_scaled)
    pca_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA 8维: 解释 {pca_var:.1%} 方差")

    # Diffusion map → 3 dim
    u_dm = diffusion_map(u_pca, n_components=3)
    print(f"  Diffusion Map: {u_dm.shape[1]} 维控制轴 z(t)")

    # Analyze
    print(f"\n[3/4] 分析: regime 在 z-space 中的位置...")
    results = analyze_control_axes(u_pca, u_dm, keywords, labels, months, n_regimes)

    # Print
    sep = results["regime_separation"]
    print(f"\n  Regime 分离度 (z-space):")
    for pair, s in sep.items():
        mark = " ✓" if s["well_separated"] else ""
        print(f"    {pair}: d={s['centroid_distance']:.4f}, "
              f"ratio={s['separation_ratio']:.1f}x{mark}")

    ext = results["r2_extremity_test"]
    print(f"\n  R2 极端性测试:")
    print(f"    R2 距全局中心: {ext['r2_distance_from_center']:.4f}")
    print(f"    其他相区均值: {ext['mean_other_regime_distance']:.4f}")
    print(f"    极端: {ext['is_extreme']}")
    print(f"    {ext['verdict']}")

    print(f"\n  各轴 R2 位置:")
    for ax, info in results["axis_extremity"].items():
        ex = " ⬅ 极端" if info["is_extreme"] else ""
        print(f"    {ax}: R2 mean={info['r2_mean']:.4f} "
              f"(全局={info['overall_mean']:.4f}), "
              f"z-score={info['z_score']:+.2f}, "
              f"分位数={info['percentile']:.1f}%{ex}")

    print(f"\n  控制轴关键词 (z₁ 主导因子):")
    for item in results["control_axis_keywords"]["z1"][:5]:
        direction = "↑" if item["correlation"] > 0 else "↓"
        print(f"    {item['keyword']:<20s} r={item['correlation']:+.4f} {direction}")

    print(f"\n  控制轴关键词 (z₂ 主导因子):")
    for item in results["control_axis_keywords"]["z2"][:5]:
        direction = "↑" if item["correlation"] > 0 else "↓"
        print(f"    {item['keyword']:<20s} r={item['correlation']:+.4f} {direction}")

    # Timeline summary
    tl = results["timeline"]
    print(f"\n[4/4] 时间演化...")
    print(f"  z₁ 范围: [{min(p['z1'] for p in tl):.3f}, {max(p['z1'] for p in tl):.3f}]")
    print(f"  z₂ 范围: [{min(p['z2'] for p in tl):.3f}, {max(p['z2'] for p in tl):.3f}]")
    # Find when system entered R2 in z-space
    r2_start = None
    for p in tl:
        if p["regime"] == 2 and r2_start is None:
            r2_start = p["month"]
    if r2_start:
        r2_entry = [p for p in tl if p["month"] == r2_start][0]
        print(f"  R2 进入点: {r2_start}, z=({r2_entry['z1']:.4f}, {r2_entry['z2']:.4f})")

    # Save
    output = {
        "source": "control_manifold.py — FR19 v4.1",
        "method": "PCA(8) → Diffusion Map(3) → z(t) control axes",
        "pca_variance_explained": round(float(pca_var), 4),
        "analysis": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  已保存 → {args.output}")

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("完成。如果 R2 在 z(t) 极端区间 → u(t) 逼迫假说成立。")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Counterfactual Test — FR19 v4.1

验证 AI/Tech 轴是 Control driver 还是 time proxy.

方法:
1. 加载 51 维外部场
2. 将 2023-01 之后的 AI/Tech 关键词流量重置为 2018 年基线
3. 重跑 PCA(8) → Diffusion Map(3) → z_cf(t)
4. 比较 z_cf(t) 与原始 z(t): 是否有显著横向移动?

判定:
- z_cf(t) 留在 R2 盆地 → AI/Tech 只是 time proxy
- z_cf(t) 显著横向漂移 → AI/Tech 是真实的 control driver

用法:
    python src/analysis/counterfactual_test.py
    python src/analysis/counterfactual_test.py --leave-out-group AI
"""

import json, sys, argparse
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).parent.parent.parent
EXTERNAL_PATH = ROOT / "data/collector/external_field_2015_2025.json"
REGIME_PATH = ROOT / "data/processed/regime_map.json"
LEVEL1_PATH = ROOT / "data/processed/level1_hard_facts.json"
OUTPUT_PATH = ROOT / "data/processed/counterfactual_results.json"

# ── Keyword groups ──
AI_TECH_KEYWORDS = [
    "AI", "ChatGPT", "人工智能", "机器学习", "深度学习",
    "大模型", "算力", "算法", "自动驾驶", "机器人",
    "芯片", "5G", "云计算", "大数据", "区块链",
]

# Additional groups for leave-one-group-out
KEYWORD_GROUPS = {
    "AI": AI_TECH_KEYWORDS,
    "经济": ["房价", "就业", "失业", "工资", "消费降级", "通胀", "股市", "经济"],
    "政策": ["体制内", "专家建议", "两会", "政策", "不结婚", "三胎", "双减"],
    "平台": ["B站", "抖音", "小红书", "知乎", "微博", "微信", "快手"],
    "文化": ["二次元", "国潮", "脱口秀", "电竞", "饭圈", "凡尔赛"],
    "国际": ["俄罗斯", "乌克兰", "美国", "日本", "韩国", "印度"],
}


def load_and_prepare():
    with open(EXTERNAL_PATH, "r", encoding="utf-8") as f:
        ef = json.load(f)
    keywords = sorted(ef["data"].keys())
    raw = ef["data"]

    with open(LEVEL1_PATH, "r", encoding="utf-8") as f:
        months = json.load(f)["months"]

    u = np.zeros((len(months), len(keywords)))
    for i, m in enumerate(months):
        for j, kw in enumerate(keywords):
            u[i, j] = raw[kw].get(m, 0.0)

    return months, keywords, u


def find_keyword_indices(keywords: list[str], group_names: list[str]) -> list[int]:
    """找到属于指定 group 的关键词索引."""
    target_kws = set()
    for g in group_names:
        if g in KEYWORD_GROUPS:
            target_kws.update(KEYWORD_GROUPS[g])
    # Fuzzy match: keyword contains the group keyword or vice versa
    indices = []
    for i, kw in enumerate(keywords):
        kw_lower = kw.lower()
        for tk in target_kws:
            if tk.lower() in kw_lower or kw_lower in tk.lower():
                indices.append(i)
                break
    return list(set(indices))


def run_diffusion_map(X: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Diffusion Map 降维 (与 control_manifold.py 一致)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    pca = PCA(n_components=8)
    X_pca = pca.fit_transform(X_s)

    dists = squareform(pdist(X_pca, metric="euclidean"))
    k = min(30, X.shape[0] - 1)
    sigma = float(np.median(np.sort(dists, axis=1)[:, k]))

    W = np.exp(-dists ** 2 / (2 * sigma ** 2))
    D = np.sum(W, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-10))
    P = D_inv_sqrt @ W @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(P)
    idx = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, idx]

    embedding = eigenvectors[:, 1:n_components + 1]
    for j in range(n_components):
        lam = max(eigenvalues[j + 1], 1e-10)
        embedding[:, j] *= lam

    return embedding


def run_counterfactual(months, keywords, u, leave_out_groups, baseline_end="2018-12",
                        reset_start="2023-01"):
    """执行 counterfactual: 抹平 leave_out_groups 在 reset_start 之后的流量."""
    u_cf = u.copy()
    kw_indices = find_keyword_indices(keywords, leave_out_groups)

    if not kw_indices:
        print(f"  ⚠ 未找到匹配关键词: {leave_out_groups}")
        return None, None, []

    # Compute baseline: mean over baseline period
    baseline_mask = np.array([m <= baseline_end for m in months])
    reset_mask = np.array([m >= reset_start for m in months])

    for j in kw_indices:
        baseline_val = np.mean(u[baseline_mask, j])
        u_cf[reset_mask, j] = baseline_val

    # Run diffusion map on both original and counterfactual
    z_orig = run_diffusion_map(u, n_components=3)
    z_cf = run_diffusion_map(u_cf, n_components=3)

    return z_orig, z_cf, kw_indices


def compare_manifolds(z_orig, z_cf, months, labels, leave_out_groups):
    """比较原始和 counterfactual 控制流形."""
    n = len(months)

    # 1. Overall drift (use pre-alignment for raw magnitude, post for R2)
    dz_raw = z_cf - z_orig
    drift_magnitude = np.array([np.linalg.norm(dz_raw[i]) for i in range(n)])

    # Pre- and post-reset drift
    reset_start_idx = None
    for i, m in enumerate(months):
        if m >= "2023-01":
            reset_start_idx = i
            break

    pre_drift = np.mean(drift_magnitude[:reset_start_idx]) if reset_start_idx else 0
    post_drift = np.mean(drift_magnitude[reset_start_idx:]) if reset_start_idx else 0
    drift_ratio = post_drift / max(pre_drift, 1e-10)

    # 2. R2 position: align manifolds via Procrustes, then measure R2 shift
    with open(REGIME_PATH, "r", encoding="utf-8") as f:
        regime = json.load(f)
    labels_arr = np.array(regime["regime_labels"])

    # Align z_cf to z_orig (Diffusion Map defined only up to rotation/reflection)
    # Use orthogonal Procrustes: find rotation R minimizing ||z_orig - z_cf @ R||
    H = z_cf.T @ z_orig
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt  # optimal rotation
    z_cf_aligned = z_cf @ R

    # Now compute R2 shift in aligned space
    r2_mask = labels_arr == 2
    if r2_mask.sum() > 0:
        r2_orig_center = np.mean(z_orig[r2_mask], axis=0)
        r2_cf_center = np.mean(z_cf_aligned[r2_mask], axis=0)
        r2_shift = float(np.linalg.norm(r2_cf_center - r2_orig_center))
    else:
        r2_shift = 0.0

    # 3. Regime separation in aligned z_cf
    all_cf_center = np.mean(z_cf_aligned, axis=0)
    regime_centroids_cf = {}
    for r in range(4):
        mask = labels_arr == r
        if mask.sum() > 0:
            regime_centroids_cf[r] = np.mean(z_cf_aligned[mask], axis=0)
    r2_cf_dist = float(np.linalg.norm(regime_centroids_cf.get(2, all_cf_center) - all_cf_center))

    # 4. Verdict
    # If post_drift >> pre_drift and r2_shift > 10% of mean pairwise distance:
    mean_pairwise = float(np.mean(pdist(z_orig)))
    significant = r2_shift > 0.10 * mean_pairwise

    if significant and drift_ratio > 2.0:
        verdict = (
            f"AI/Tech 是 CAUSAL DRIVER. "
            f"抹平后 z(t) 显著移动 (R2 质心移动 {r2_shift:.4f} > "
            f"10% 平均距离 {0.1*mean_pairwise:.4f}). "
            f"重置后漂移比 = {drift_ratio:.1f}x."
        )
    elif drift_ratio > 1.5:
        verdict = (
            f"AI/Tech 有 WEAK CAUSAL 效应. "
            f"抹平后 z(t) 有移动但不够显著. "
            f"可能 AI 是部分 driver, 但不是唯一."
        )
    else:
        verdict = (
            f"AI/Tech 是 TIME PROXY. "
            f"抹平后 z(t) 几乎不变 (漂移比 = {drift_ratio:.1f}x, "
            f"R2 移动 = {r2_shift:.4f}). "
            f"真正的 control driver 是其他慢变量 (经济/政策/人口)."
        )

    return {
        "leave_out_groups": leave_out_groups,
        "n_keywords_modified": len(find_keyword_indices([], leave_out_groups)),
        "pre_reset_mean_drift": round(float(pre_drift), 4),
        "post_reset_mean_drift": round(float(post_drift), 4),
        "drift_ratio": round(float(drift_ratio), 2),
        "r2_centroid_shift": round(float(r2_shift), 4),
        "mean_pairwise_distance": round(float(mean_pairwise), 4),
        "significant": bool(significant),
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="Counterfactual Test")
    parser.add_argument("--leave-out-group", type=str, default="AI",
                        help="要抹平的关键词组 (AI/经济/政策/平台/文化/国际)")
    parser.add_argument("--all-groups", action="store_true",
                        help="跑全部 Leave-one-group-out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Counterfactual Test — Control Manifold")
    print("=" * 60)

    months, keywords, u = load_and_prepare()
    print(f"\n  外部场: {len(months)} 月 × {len(keywords)} 关键词")

    if args.all_groups:
        groups = list(KEYWORD_GROUPS.keys())
    else:
        groups = [args.leave_out_group]

    all_results = []

    for group in groups:
        print(f"\n{'─' * 60}")
        print(f"  Leave-out: {group}")
        print(f"{'─' * 60}")

        indices = find_keyword_indices(keywords, [group])
        matched_kws = [keywords[i] for i in indices]
        print(f"  匹配关键词 ({len(indices)}): {', '.join(matched_kws[:10])}"
              + ("..." if len(matched_kws) > 10 else ""))

        if not indices:
            print("  ⚠ 无匹配, 跳过")
            continue

        z_orig, z_cf, _ = run_counterfactual(
            months, keywords, u, [group])

        if z_orig is None:
            continue

        result = compare_manifolds(z_orig, z_cf, months, None, [group])
        all_results.append(result)

        print(f"\n  {result['verdict']}")
        print(f"  漂移比 (后/前): {result['drift_ratio']}x")
        print(f"  R2 质心移动: {result['r2_centroid_shift']:.4f}")
        print(f"  显著: {result['significant']}")

    # Summary for all groups
    if len(all_results) > 1:
        print(f"\n{'═' * 60}")
        print("Leave-one-group-out 汇总")
        print(f"{'═' * 60}")
        print(f"\n  {'Group':<10s} {'漂移比':>8s} {'R2移动':>8s} {'显著':>6s} {'判定':>40s}")
        print(f"  {'─'*75}")
        for r in all_results:
            sig = "✓" if r["significant"] else ""
            print(f"  {r['leave_out_groups'][0]:<10s} {r['drift_ratio']:>8.1f}x "
                  f"{r['r2_centroid_shift']:>8.4f} {sig:>6s} "
                  f"{r['verdict'][:40]:>40s}")

        # Which group has the largest effect?
        best = max(all_results, key=lambda r: r["drift_ratio"])
        print(f"\n  最大效应组: {best['leave_out_groups'][0]} "
              f"(漂移比={best['drift_ratio']}x)")
        if best["significant"]:
            print(f"  → {best['leave_out_groups'][0]} 是 primary control driver.")
        else:
            print(f"  → 所有组的单独效应均不显著. "
                  f"Control 可能是多变量协同, 非单一 driver.")

    # Save
    output = {
        "source": "counterfactual_test.py",
        "results": all_results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  已保存 → {OUTPUT_PATH}")

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

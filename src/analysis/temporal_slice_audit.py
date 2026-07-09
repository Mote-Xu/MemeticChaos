"""
Temporal Slice Audit — 证伪/坐实预设 2「存在时不变的生成机制」

第七轮外部 AI 共识: 预设 2 是当前架构最深、最危险的未被意识的前提。
所有 127 月一次性拟合的科学产出 (H1/Regime/RQA/ControlManifold/FR31)
都默认了 2015 年和 2025 年是"同一个系统"。如果互联网底层逻辑变了,
这个默认就是错的, 而所有下游结论都建在它上面。

本审计把 127 月切成三段, 每段独立拟合全套可学习对象
(StandardScaler / 外部场 PCA / 状态 PCA / GMM), 然后跨段比较结构。
复用全局基 = 预设时不变 = 审计作弊。所以一切从零重拟。

═══ 方法论: 三类标注 (见 CLAUDE.md) ═══
- 数学工具:   PCA / GMM / principal angles / KS test。用, 不声称物理解释。
- 统计描述:   "d90 从 8 变到 12""跨段子空间夹角 62°"。数据在说什么。
- 物理假说:   "生成机制变了""2015 和 2025 是两个系统"。★必须等独立证据,
              不能同一组数据既提声称又验声称★。本脚本只输出"证据方向",
              verdict 字段显式标注为 HYPOTHESIS, 永不声称"已证明"。

═══ 小样本诚实处理 ═══
段长 55 / 36 / 36 月。18 维空间跑 full-cov GMM 在 36 月上是过拟合。
- 稳健测试 (可信):   特征分布 KS / 内在维度 / 子空间夹角 / 交叉重构。
- 脆弱测试 (标警告): GMM 相区身份。→ 降维 + diag 协方差 + 显式 reliability 标记。
- 零假设基线 (关键): 跨段夹角必须对比"同分布随机二分"的夹角分布。
  否则分不清"机制变了"和"采样噪声"。这是证伪预设 2 的唯一诚实方式。

用法:
    conda run -n MemeticChaos python src/analysis/temporal_slice_audit.py
    conda run -n MemeticChaos python src/analysis/temporal_slice_audit.py --json
"""

import json, sys, argparse
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.linalg import subspace_angles

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 复用 representation_learning 的加载 + 组装, 保证特征定义完全一致
from models.representation_learning import (
    load_level1, load_external_field, load_trends,
    compute_attention_structure, assemble_features,
)

OUTPUT_PATH = ROOT / "data/processed/temporal_slice_audit.json"

# 三段切分。边界按互联网叙事史的直觉候选 (移动互联网成熟 / 疫情三年 / 后疫情+LLM),
# 但边界本身是一个假设 —— 见 verdict 中的 caveat。
SEGMENTS = [
    ("pre2020",    lambda m: m[:4] <= "2019"),
    ("2020-2022",  lambda m: "2020" <= m[:4] <= "2022"),
    ("2023-2025",  lambda m: m[:4] >= "2023"),
]

# 用于夹角/重构比较的子空间维度。取小而稳: 前 k 个 PC 承载主导结构,
# 高阶 PC 在 n=36 上纯噪声。k=4 是保守选择。
K_SUBSPACE = 4
RNG_SEED = 42  # Date.now/random 在别处禁用; 这里是 numpy 本地 RNG, 允许


# ═══════════════════════════════════════════════
# 特征组装 (全段 + 分段, 外部场 PCA 每段独立)
# ═══════════════════════════════════════════════

def slice_l1(l1: dict, keep_idx: list[int]) -> dict:
    """把 l1 的月度数组切到 keep_idx 指定的月份, 其余字段透传。"""
    arr_keys = ["months", "stage_occupancy", "mutation_rate",
                "institutionalization_rate", "mean_semantic_drift"]
    out = dict(l1)
    for k in arr_keys:
        v = l1[k]
        out[k] = [v[i] for i in keep_idx]
    return out


def build_segment_features(l1, ext_months, ext_matrix, hhi, entropy, total_att,
                           month_filter, ext_pc=8):
    """对单个时段独立组装 18 维特征。

    外部场 PCA 在 assemble_features 内部按传入 l1 的月份拟合 —— 因为我们传的是
    切过的 l1, 所以 ext-PCA 只用该段的月份拟合。这是"每段独立"的关键。
    """
    all_months = l1["months"]
    keep_idx = [i for i, m in enumerate(all_months) if month_filter(m)]
    l1_seg = slice_l1(l1, keep_idx)
    hhi_seg = hhi[keep_idx]
    ent_seg = entropy[keep_idx]
    tot_seg = total_att[keep_idx]

    # 段太短时降低 ext_pc (不能超过样本数-1)
    ext_pc_eff = min(ext_pc, len(keep_idx) - 1)

    X, months, meta = assemble_features(
        l1_seg, ext_months, ext_matrix, hhi_seg, ent_seg, tot_seg, ext_pc=ext_pc_eff)
    return X, months, meta


# ═══════════════════════════════════════════════
# 单段结构拟合 (每段从零)
# ═══════════════════════════════════════════════

def fit_segment_structure(X: np.ndarray, feature_names: list[str], months: list[str]) -> dict:
    """对单段: 标准化 → PCA (内在维度 + 载荷) → GMM (相区, 标低可靠)。"""
    n, p = X.shape
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ── PCA: 内在维度 + 主成分 ──
    pca = PCA()
    pca.fit(Xs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    d90 = int(np.searchsorted(cumvar, 0.90) + 1)
    d95 = int(np.searchsorted(cumvar, 0.95) + 1)

    # PC1/PC2 top 载荷 (统计描述: 该段主导的是哪些特征)
    top_loadings = []
    for pc in range(min(2, p)):
        load = pca.components_[pc]
        idx = np.argsort(np.abs(load))[-5:][::-1]
        top_loadings.append([
            {"feature": feature_names[i], "loading": round(float(load[i]), 4)}
            for i in idx
        ])

    # ── 低维状态 (前 K_SUBSPACE PC) 用于 GMM + 跨段夹角 ──
    k = min(K_SUBSPACE, p)
    x_low = pca.transform(Xs)[:, :k]

    # ── GMM 相区 (脆弱: n 小, 用 diag 协方差 + 限制 k 范围) ──
    gmm_result = fit_regimes_small_n(x_low, months)

    return {
        "n_months": n,
        "n_features": p,
        "intrinsic_dim": {"d90": d90, "d95": d95,
                          "evr_top5": [round(float(v), 4) for v in pca.explained_variance_ratio_[:5]]},
        "pc_loadings": {"PC1": top_loadings[0], "PC2": top_loadings[1] if len(top_loadings) > 1 else []},
        "regime_signature": gmm_result,
        # 内部保留 (不进 JSON): 标准化后矩阵 + PCA 对象供跨段比较
        "_Xs": Xs,
        "_pca": pca,
        "_x_low": x_low,
        "_components_k": pca.components_[:k],
    }


def fit_regimes_small_n(x_low: np.ndarray, months: list[str]) -> dict:
    """小样本 GMM: diag 协方差, k∈{2,3,4}, BIC 选。输出动力学签名 + reliability 警告。

    关注的不是相区身份 (跨段不可比), 而是动力学签名: 切换率 / 主导占比 / 最长锁定。
    这些是能跨段比较的标量。
    """
    n = len(x_low)
    best = None
    for k in [2, 3, 4]:
        if n < k * 3:  # 每组件至少 ~3 点, 否则跳过
            continue
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                              random_state=RNG_SEED, n_init=5)
        labels = gmm.fit_predict(x_low)
        bic = gmm.bic(x_low)
        if best is None or bic < best["bic"]:
            best = {"k": k, "bic": float(bic), "labels": labels}

    if best is None:
        return {"reliability": "insufficient_n", "note": f"n={n} 太小, 跳过 GMM"}

    labels = best["labels"]
    switches = int(sum(1 for t in range(1, n) if labels[t] != labels[t-1]))
    from collections import Counter
    counts = Counter(int(l) for l in labels)
    dominant_pct = max(counts.values()) / n

    # 最长连续同相区锁定
    max_lock, cur = 1, 1
    for t in range(1, n):
        cur = cur + 1 if labels[t] == labels[t-1] else 1
        max_lock = max(max_lock, cur)

    return {
        "reliability": "LOW — n small, diag-cov GMM, 相区身份跨段不可比; 仅签名标量可比",
        "k_selected": best["k"],
        "n_switches": switches,
        "switch_rate_per_month": round(switches / max(n - 1, 1), 4),
        "dominant_regime_pct": round(dominant_pct, 4),
        "max_lock_months": max_lock,
    }


# ═══════════════════════════════════════════════
# 跨段比较
# ═══════════════════════════════════════════════

def feature_distribution_shift(seg_X: dict, feature_names: list[str]) -> dict:
    """KS 2-sample: 每个特征在每对时段间是否显著移位 (统计描述)。

    注意: 各段的 ext_pc_i 是各自 PCA 的产物, 语义不可比 —— 只比 L1(8)+att(2) 这 10
    个语义固定的特征。ext_pc 的比较改由子空间夹角承担。
    """
    # 语义固定的列 (前 8 个 L1 + 后 2 个 att); ext_pc 在中间, 索引随 ext_pc_eff 变
    fixed = [fn for fn in feature_names if not fn.startswith("ext_pc_")]
    seg_names = list(seg_X.keys())
    pairs = [(seg_names[i], seg_names[j])
             for i in range(len(seg_names)) for j in range(i+1, len(seg_names))]

    out = {}
    for a, b in pairs:
        Xa, fna = seg_X[a]
        Xb, fnb = seg_X[b]
        shifted = []
        for fn in fixed:
            if fn in fna and fn in fnb:
                va = Xa[:, fna.index(fn)]
                vb = Xb[:, fnb.index(fn)]
                ks, pval = stats.ks_2samp(va, vb)
                if pval < 0.05:
                    shifted.append({"feature": fn, "ks": round(float(ks), 3),
                                    "p": round(float(pval), 4),
                                    "mean_a": round(float(va.mean()), 3),
                                    "mean_b": round(float(vb.mean()), 3)})
        out[f"{a} vs {b}"] = {
            "n_fixed_features": len(fixed),
            "n_shifted_p05": len(shifted),
            "shifted": sorted(shifted, key=lambda d: d["p"]),
        }
    return out


def subspace_angles_between(comp_a: np.ndarray, comp_b: np.ndarray) -> list[float]:
    """两段前 k 个主成分张成的子空间之间的主夹角 (度)。

    principal angles: 0° = 子空间重合 (同结构), 90° = 正交 (完全不同结构)。
    comp_* 形状 (k, p), 每行一个主成分 (已正交)。
    """
    # subspace_angles 要求列向量矩阵 (p, k)
    angles = subspace_angles(comp_a.T, comp_b.T)
    return [round(float(np.degrees(a)), 2) for a in sorted(angles)]


def cross_reconstruction(seg_struct: dict) -> dict:
    """用 A 段学到的 PCA 基 (前 k) 重构 B 段的标准化数据, 比 B 自重构的损失。

    如果 A 的基几乎无法表示 B (重构 R² 远低于 B 自身) → A、B 结构不同。
    这是"A 段训练的表示能否描述 B 段"的直接检验 —— 正对预设 2。
    """
    names = list(seg_struct.keys())
    k = K_SUBSPACE
    out = {}

    def recon_r2(Xs, comps):
        # 投影到 comps (k, p) 再重构; Xs 已零均值(标准化) → 直接投影
        proj = Xs @ comps.T          # (n, k)
        Xhat = proj @ comps          # (n, p)
        ss_res = np.sum((Xs - Xhat) ** 2)
        ss_tot = np.sum(Xs ** 2)
        return 1 - ss_res / ss_tot

    for a in names:
        for b in names:
            if a == b:
                continue
            Xs_b = seg_struct[b]["_Xs"]
            comp_a = seg_struct[a]["_components_k"]  # A 的前 k 主成分 (在 A 特征空间)
            comp_b = seg_struct[b]["_components_k"]
            # 前提: A、B 特征列一致 (ext_pc_eff 可能不同 → 对齐到公共列)
            pa = comp_a.shape[1]
            pb = comp_b.shape[1]
            p = min(pa, pb)
            self_r2 = recon_r2(Xs_b[:, :p], comp_b[:, :p])
            cross_r2 = recon_r2(Xs_b[:, :p], comp_a[:, :p])
            out[f"{a}-basis on {b}"] = {
                "self_r2": round(float(self_r2), 4),
                "cross_r2": round(float(cross_r2), 4),
                "gap": round(float(self_r2 - cross_r2), 4),
            }
    return out


def _recon_r2(Xs, comps):
    """comps (k,p) 张成子空间对已标准化(零均值) Xs 的重构 R²。"""
    proj = Xs @ comps.T
    Xhat = proj @ comps
    return 1 - np.sum((Xs - Xhat) ** 2) / np.sum(Xs ** 2)


def null_baseline(X_full: np.ndarray, seg_size: int, n_boot: int = 800) -> dict:
    """零假设基线: 若全程是同一个时不变机制, 两个"同长度连续区块"之间
    应产生多大的子空间夹角 / 交叉重构落差?

    ★关键修正★: null 必须匹配真实测试的采样几何。真实时段是连续区块 + 每段独立
    标准化, 所以 null 也用**连续非重叠区块** + 每窗独立 StandardScaler+PCA。
    早先用"随机打散子集"是错的 —— 自相关序列的随机散点子空间几乎正交 (noise floor
    顶到 ~90°), 把测试的检验力废掉。

    对每对区块同时算: 均值主夹角 (未饱和, 主判据) / 最大主夹角 / 交叉重构 gap。
    """
    rng = np.random.default_rng(RNG_SEED)
    n, p = X_full.shape
    k = min(K_SUBSPACE, p)
    L = min(seg_size, (n - 1) // 2)  # 保证能放下两个非重叠窗

    mean_ang, max_ang, xrec_gap = [], [], []
    for _ in range(n_boot):
        # 两个非重叠连续窗: a 在前, b 在 a 之后
        start_a = int(rng.integers(0, n - 2 * L + 1))
        start_b = int(rng.integers(start_a + L, n - L + 1))
        Wa = X_full[start_a:start_a + L]
        Wb = X_full[start_b:start_b + L]
        # 每窗独立标准化 + PCA (匹配真实段处理)
        Za = StandardScaler().fit_transform(Wa)
        Zb = StandardScaler().fit_transform(Wb)
        pa = PCA(n_components=k).fit(Za)
        pb = PCA(n_components=k).fit(Zb)
        ang = np.degrees(subspace_angles(pa.components_.T, pb.components_.T))
        mean_ang.append(float(np.mean(ang)))
        max_ang.append(float(np.max(ang)))
        # 交叉重构 gap: b 自基重构 vs a 基重构 b
        self_r2 = _recon_r2(Zb, pb.components_)
        cross_r2 = _recon_r2(Zb, pa.components_)
        xrec_gap.append(float(self_r2 - cross_r2))

    return {
        "n_bootstrap": n_boot,
        "window_len": L,
        "sampling": "contiguous non-overlapping blocks, per-window standardized (matches real segments)",
        "null_mean_angle_deg": {"median": round(float(np.median(mean_ang)), 2),
                                "p95": round(float(np.percentile(mean_ang, 95)), 2)},
        "null_max_angle_deg": {"median": round(float(np.median(max_ang)), 2),
                               "p95": round(float(np.percentile(max_ang, 95)), 2)},
        "null_xrecon_gap": {"median": round(float(np.median(xrec_gap)), 4),
                            "p95": round(float(np.percentile(xrec_gap, 95)), 4)},
        "_mean_samples": mean_ang,
        "_xrec_samples": xrec_gap,
    }


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Temporal Slice Audit — 预设 2 证伪")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 64)
    print("Temporal Slice Audit — 证伪预设 2「时不变生成机制」")
    print("=" * 64)

    # ── 加载共享原始输入 ──
    l1 = load_level1()
    ext_months, ext_matrix = load_external_field()
    trends = load_trends()
    hhi, entropy, total_att = compute_attention_structure(trends, l1["months"])
    print(f"\n全序列: {len(l1['months'])} 月 ({l1['months'][0]} → {l1['months'][-1]})")

    # ── 全段基准 (提供零假设基线的分布) ──
    X_full, months_full, meta_full = assemble_features(
        l1, ext_months, ext_matrix, hhi, entropy, total_att, ext_pc=8)
    Xs_full = StandardScaler().fit_transform(X_full)

    # ── 每段独立拟合 ──
    print(f"\n[1/5] 每段独立拟合 (scaler/ext-PCA/state-PCA/GMM 全从零)...")
    seg_struct = {}
    seg_X = {}  # 供 KS: (X, feature_names)
    seg_sizes = []
    for name, filt in SEGMENTS:
        X, months, meta = build_segment_features(
            l1, ext_months, ext_matrix, hhi, entropy, total_att, filt, ext_pc=8)
        struct = fit_segment_structure(X, meta["feature_names"], months)
        seg_struct[name] = struct
        seg_X[name] = (X, meta["feature_names"])
        seg_sizes.append(struct["n_months"])
        idim = struct["intrinsic_dim"]
        reg = struct["regime_signature"]
        print(f"  {name:<11s} n={struct['n_months']:>3d}  d90={idim['d90']} d95={idim['d95']}  "
              f"EVR1={idim['evr_top5'][0]:.2f}  "
              f"GMM: k={reg.get('k_selected','-')} switch={reg.get('switch_rate_per_month','-')} "
              f"lock={reg.get('max_lock_months','-')}")

    # ── PC1 载荷跨段对比 (段主导轴是不是同一批特征) ──
    print(f"\n[2/5] 各段 PC1 主导载荷 (是否同一批特征主导):")
    for name in seg_struct:
        top = seg_struct[name]["pc_loadings"]["PC1"]
        s = ", ".join(f"{d['feature']}({d['loading']:+.2f})" for d in top[:3])
        print(f"  {name:<11s} {s}")

    # ── 特征分布移位 KS ──
    print(f"\n[3/5] 特征分布平稳性 (KS 2-sample, 语义固定的 10 个特征):")
    ks = feature_distribution_shift(seg_X, meta_full["feature_names"])
    for pair, r in ks.items():
        print(f"  {pair}: {r['n_shifted_p05']}/{r['n_fixed_features']} 特征显著移位 (p<0.05)")
        for d in r["shifted"][:4]:
            print(f"      {d['feature']:<18s} KS={d['ks']:.2f} p={d['p']:.3f}  "
                  f"mean {d['mean_a']:+.2f}→{d['mean_b']:+.2f}")

    # ── 子空间夹角 + 零假设基线 (连续区块 block-bootstrap) ──
    print(f"\n[4/5] 子空间夹角 (前 {K_SUBSPACE} PC) vs 连续区块噪声基线:")
    null = null_baseline(X_full, int(np.median(seg_sizes)))
    print(f"  零假设 (连续非重叠区块, L={null['window_len']}, {null['n_bootstrap']}x, 每窗独立标准化):")
    print(f"    均值主夹角 (主判据): median={null['null_mean_angle_deg']['median']}° "
          f"p95={null['null_mean_angle_deg']['p95']}°")
    print(f"    最大主夹角 (n 小时饱和, 仅参考): median={null['null_max_angle_deg']['median']}° "
          f"p95={null['null_max_angle_deg']['p95']}°")
    print(f"    交叉重构 gap: median={null['null_xrecon_gap']['median']} "
          f"p95={null['null_xrecon_gap']['p95']}")

    names = list(seg_struct.keys())
    angle_results = {}
    null_mean_p95 = null["null_mean_angle_deg"]["p95"]
    null_mean_samples = np.array(null["_mean_samples"])
    print(f"  跨段真实夹角:")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            ca = seg_struct[a]["_components_k"]
            cb = seg_struct[b]["_components_k"]
            p = min(ca.shape[1], cb.shape[1])
            ang = subspace_angles_between(ca[:, :p], cb[:, :p])
            mean_ang = float(np.mean(ang))
            # 经验 p: 零分布中有多少比例的均值夹角 ≥ 观测均值夹角
            emp_p = float(np.mean(null_mean_samples >= mean_ang))
            exceeds = mean_ang > null_mean_p95
            angle_results[f"{a} vs {b}"] = {
                "principal_angles_deg": ang,
                "mean_angle_deg": round(mean_ang, 2),
                "null_mean_p95_deg": null_mean_p95,
                "empirical_p": round(emp_p, 4),
                "exceeds_null": bool(exceeds),
            }
            flag = "★超出噪声" if exceeds else "在噪声内"
            print(f"    {a} vs {b}: mean={mean_ang:.1f}° angles={ang}  "
                  f"emp_p={emp_p:.3f}  [{flag}]")

    # ── 交叉重构 (A 的基能否表示 B) + gap 是否超出连续区块噪声 ──
    print(f"\n[5/5] 交叉重构 (A 的基能否表示 B; gap 对比噪声 p95={null['null_xrecon_gap']['p95']}):")
    xrec = cross_reconstruction(seg_struct)
    null_gap_p95 = null["null_xrecon_gap"]["p95"]
    null_gap_samples = np.array(null["_xrec_samples"])
    for pair, r in xrec.items():
        emp_p = float(np.mean(null_gap_samples >= r["gap"]))
        r["null_gap_p95"] = null_gap_p95
        r["gap_empirical_p"] = round(emp_p, 4)
        r["exceeds_null"] = bool(r["gap"] > null_gap_p95)
        flag = "★超出噪声" if r["exceeds_null"] else "在噪声内"
        print(f"  {pair}: self_R²={r['self_r2']:.3f} cross_R²={r['cross_r2']:.3f} "
              f"gap={r['gap']:+.3f} emp_p={emp_p:.3f} [{flag}]")

    # ── Verdict (物理假说 — 显式标注, 不声称已证明) ──
    # 主判据: 均值夹角超噪声的时段对 + 重构 gap 超噪声的方向对。
    # KS 移位不进判据 —— 它测状态分布漂移, 不测机制不变性 (混淆 state-movement 与 mechanism-change)。
    n_angle_exceed = sum(1 for r in angle_results.values() if r["exceeds_null"])
    n_gap_exceed = sum(1 for r in xrec.values() if r["exceeds_null"])
    max_ks_shift = max(r["n_shifted_p05"] for r in ks.values())
    mean_xrec_gap = float(np.mean([r["gap"] for r in xrec.values()]))

    if n_angle_exceed >= 2 or n_gap_exceed >= 4:
        direction = "AGAINST_P2"
        verdict = ("证据方向: 反对预设 2。表示的协方差结构 (子空间夹角/交叉重构) 在时段间"
                   "的差异超出连续区块采样噪声 —— 一段学到的结构无法描述另一段。"
                   "指向: 生成机制随时段变化, 2015/2020/2023 可能不是同一系统。")
    elif n_angle_exceed == 0 and n_gap_exceed == 0:
        direction = "UNDERPOWERED_CANNOT_REJECT_P2"
        verdict = ("证据方向: 无法拒绝预设 2, 但主因是检验力不足。36 月段的协方差结构差异"
                   "落在连续区块采样噪声内 —— 在月度分辨率下, 本审计无法把'机制改变'从"
                   "'同机制不同采样'中分离出来。特征分布 KS 大幅移位是真实的, 但它测的是"
                   "状态分布漂移, 不是机制不变性, 故不进判据。P2 仍未被证伪也未被坐实。")
    else:
        direction = "MIXED"
        verdict = ("证据方向: 混合。子空间夹角在噪声内, 但部分交叉重构 gap 超噪声 (或反之)。"
                   "协方差结构可能有段间差异但不稳健。需要更长序列或边界敏感性扫描。")

    print(f"\n{'═' * 64}")
    print(f"VERDICT [HYPOTHESIS — 物理假说, 非已证明]: {direction}")
    print(f"{'═' * 64}")
    print(f"  {verdict}")
    print(f"\n  证据汇总:")
    print(f"    均值子空间夹角超噪声的时段对: {n_angle_exceed}/{len(angle_results)}")
    print(f"    交叉重构 gap 超噪声的方向对:   {n_gap_exceed}/{len(xrec)}")
    print(f"    平均跨段重构 gap: {mean_xrec_gap:+.3f} (噪声 p95={null['null_xrecon_gap']['p95']})")
    print(f"    最大 KS 特征移位: {max_ks_shift}/10 (★不进判据: 测状态漂移非机制)")
    print(f"\n  ⚠ Caveat (未被消除的预设):")
    print(f"    - 段边界 (2020/2023) 本身是假设, 未做敏感性扫描。")
    print(f"    - 36 月段的 GMM 相区身份不可信 (仅签名标量可比)。")
    print(f"    - 特征定义 (18 维) 跨段沿用, 若某段的相关特征根本不同, 本审计看不到。")
    print(f"    - n=36 检验力低: '无法拒绝'≠'成立'。")

    # ── 保存 (剥离内部 _ 字段) ──
    def clean(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}
    output = {
        "source": "temporal_slice_audit.py",
        "target_presupposition": "P2: 存在时不变的生成机制",
        "method_note": ("每段独立拟合全套; null 为连续区块 block-bootstrap; "
                        "主判据=均值子空间夹角+交叉重构gap vs 噪声; "
                        "KS 移位不进判据(测状态漂移非机制); verdict 为假说非证明"),
        "segments": {name: clean(seg_struct[name]) for name in seg_struct},
        "feature_distribution_shift_ks": ks,
        "null_baseline": clean(null),
        "subspace_angles_cross_segment": angle_results,
        "cross_reconstruction": xrec,
        "verdict": {
            "direction": direction,
            "label": "HYPOTHESIS — 物理假说, 不可从本数据声称已证明",
            "text": verdict,
            "evidence": {
                "n_angle_pairs_exceeding_null": n_angle_exceed,
                "n_gap_pairs_exceeding_null": n_gap_exceed,
                "n_angle_pairs_total": len(angle_results),
                "n_gap_pairs_total": len(xrec),
                "max_ks_feature_shift": max_ks_shift,
                "ks_note": "KS 不进判据: 测状态分布漂移, 混淆 state-movement 与 mechanism-change",
                "mean_cross_reconstruction_gap": round(mean_xrec_gap, 4),
            },
            "caveats": [
                "段边界 2020/2023 是假设, 未做敏感性扫描",
                "36 月段 GMM 相区身份不可信",
                "18 维特征定义跨段沿用, 段特异的相关特征不可见",
                "n=36 检验力低: 无法拒绝 P2 不等于 P2 成立",
            ],
        },
    }
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  已保存 → {outp}")

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

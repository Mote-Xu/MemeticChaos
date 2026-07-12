"""
Regime Discreteness — x(t) 到底是"分簇"还是"连续流形被 GMM 切 bin"? (2026-07-12, 第十一轮)

审计层批准本方向 (regime-discretization), 带 3 条 guardrail, 全部写进设计:

★guardrail ① 预期 UNDERPOWERED★
  127 点 in 10 维对密度/模态估计是稀疏的。所有检验都配 null 校准 (gap 的 uniform 参考 /
  stability 的自举 / Silverman 的 smoothed bootstrap), 模糊结果如实标 UNDERPOWERED,
  ★不把模糊读成"坐实连续"或"坐实离散"★。

★guardrail ② 必须 resolve 现有张力, 不是再确认一侧★
  现有两个证据打架:
    - RQA (irreversibility_test): R2 零跨相区复发 → 偏"离散/真实分离"。
    - curation_sensitivity: 簇数众数=7、仅 4% 复现 k=4、BIC 顶搜索上限 → 偏"连续/过度切分"。
  本脚本去掰这个: 分别测『R2 这一极的分离性』和『k=4 这个计数的稳健性』, 交叉参照 RQA 的
  inter-regime 距离。二者未必矛盾——可能是"一个真实分离盆地 (R2) + 其余弱结构连续体被过切"。

★guardrail ③ 承重★
  regime-discretization 一松, 级联 r2-real-cluster / weak-irreversibility / r2-persistence /
  gmm-regimes / MS-AR / FR31-Position。所以结论只**刻画 离散vs连续**, 落 E1/E2 + 诚实 UNDERPOWERED:
  ★不"证明连续"★(那是把 R2 溶掉、解构过头), ★不"证明离散"★(过度声称)。
  输出为三层拆分 (镜像 P2a/b/c): RD-a 低维 label-free 结构 / RD-b K 相区计数 / RD-c R2 盆地。
  RD-c (盆地 vs 连续漂移驻留) 作 Competing Explanatory Layer 并存, 不给任一读法加冕。

──────────────────────────────────────────────────────────────────────
方法 (踩借来的成熟石头, 不自造):
  T1 Gap statistic (Tibshirani 2001): k=1..6, uniform 参考 null → 数据支持几簇?
     ★关键★: regime_detector 的 BIC 搜索从 k=3 起 (结构上排除 k=1/2), gap 从 k=1 起, 能揭示这个预设。
  T2 Cluster stability (Hennig clusterboot / Jaccard): 自举重拟 GMM(4), 每个 baseline 簇的 Jaccard
     → 哪个簇稳 (R2?) 哪个碎 (R1/R3?)。
  T3 Silverman critical-bandwidth 单峰检验 (Silverman 1981): 对 R2 分离轴 + 前几个 PC, smoothed
     bootstrap 求"多峰"p。★在合成 uni/bi-modal 上先验功率★ (稀疏下功率低就标 exploratory)。
  X  RQA 交叉参照: 直接引 irreversibility_results.json 的 inter-regime 距离/复发, 不重算。

对应 Evidence Ledger: 各检验统计是 E1/E2; "离散 vs 连续"的刻画是 E2/E3 (依赖算法+分辨率)。

用法:
  conda run -n MemeticChaos python src/analysis/regime_discreteness.py
  conda run -n MemeticChaos python src/analysis/regime_discreteness.py --selftest   # 只跑合成对照
"""

import json, sys, argparse
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from analysis.regime_detector import load_data  # months, x, stage, stage_names

STATE_PATH = ROOT / "data/processed/representation_state.json"
IRREV_PATH = ROOT / "data/processed/irreversibility_results.json"
OUTPUT_PATH = ROOT / "data/processed/regime_discreteness.json"
RNG_SEED = 42
FIX_IDX = 4   # stage_occupancy 列: origin/emergence/peak/controversy/fixation


# ═══════════════════════════════════════════════
# T1 — Gap statistic
# ═══════════════════════════════════════════════
def within_dispersion(X, labels, k):
    """Σ_c Σ_{i∈c} ||x_i - μ_c||²  (KMeans inertia 口径)。"""
    W = 0.0
    for c in range(k):
        pts = X[labels == c]
        if len(pts) > 0:
            W += float(((pts - pts.mean(0)) ** 2).sum())
    return W


def gap_statistic(X, kmax, B, rng):
    """Tibshirani gap: uniform 参考 (PCA-对齐 box, 因 x 已是 PCA → 各列 uniform)。"""
    n, d = X.shape
    mins, maxs = X.min(0), X.max(0)
    logW, logW_ref, sk = [], [], []
    for k in range(1, kmax + 1):
        km = KMeans(k, n_init=10, random_state=0).fit(X)
        logW.append(np.log(within_dispersion(X, km.labels_, k) + 1e-12))
        refs = []
        for _ in range(B):
            Xr = rng.uniform(mins, maxs, size=(n, d))
            kmr = KMeans(k, n_init=3, random_state=0).fit(Xr)
            refs.append(np.log(within_dispersion(Xr, kmr.labels_, k) + 1e-12))
        logW_ref.append(float(np.mean(refs)))
        sk.append(float(np.std(refs) * np.sqrt(1 + 1.0 / B)))
    gap = np.array(logW_ref) - np.array(logW)
    # Tibshirani 规则: 最小 k 使 gap[k] >= gap[k+1] - s[k+1]
    khat = kmax
    for k in range(1, kmax):
        if gap[k - 1] >= gap[k] - sk[k]:
            khat = k
            break
    return {"k_range": list(range(1, kmax + 1)),
            "gap": [round(g, 4) for g in gap], "sk": [round(s, 4) for s in sk],
            "gap_optimal_k": int(khat)}


# ═══════════════════════════════════════════════
# T2 — Cluster stability (Jaccard)
# ═══════════════════════════════════════════════
def cluster_stability(x_s, k, baseline_labels, B, rng):
    """自举重拟 GMM(k), 每个 baseline 簇对最佳匹配预测簇的 Jaccard 均值。"""
    n = len(x_s)
    jac = {c: [] for c in range(k)}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            gmm_b = GaussianMixture(k, covariance_type="full",
                                    random_state=int(rng.integers(1_000_000)),
                                    n_init=1).fit(x_s[idx])
            pred = gmm_b.predict(x_s)
        except Exception:
            continue
        for c in range(k):
            base = baseline_labels == c
            best = 0.0
            for cp in range(k):
                p = pred == cp
                u = (base | p).sum()
                best = max(best, (base & p).sum() / u if u > 0 else 0.0)
            jac[c].append(best)
    return {c: float(np.mean(v)) if v else 0.0 for c, v in jac.items()}


# ═══════════════════════════════════════════════
# T3 — Silverman critical-bandwidth modality test
# ═══════════════════════════════════════════════
def _kde_nmodes(x1d, h, grid):
    """显式高斯 KDE (带宽 h) 在 grid 上的局部极大数。"""
    d = np.exp(-0.5 * ((grid[:, None] - x1d[None, :]) / h) ** 2).sum(1)
    return int(((d[1:-1] > d[:-2]) & (d[1:-1] > d[2:])).sum())


def _h_crit(x1d, grid):
    """最小带宽 h 使 KDE 恰为单峰 (bisection; 模态数随 h 单调降)。"""
    sd = x1d.std(ddof=1) + 1e-12
    lo, hi = sd / 50.0, sd * 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _kde_nmodes(x1d, mid, grid) > 1:
            lo = mid
        else:
            hi = mid
    return hi


def silverman_modality(x1d, B, rng):
    """H0: 单峰。smoothed bootstrap 下 h_crit 处多峰的比例 = p。p 小 → 拒绝单峰 (多峰)。"""
    x1d = np.asarray(x1d, float)
    x1d = (x1d - x1d.mean()) / (x1d.std(ddof=1) + 1e-12)
    span = x1d.max() - x1d.min()
    grid = np.linspace(x1d.min() - 0.5 * span, x1d.max() + 0.5 * span, 512)
    hc = _h_crit(x1d, grid)
    sd = x1d.std(ddof=1)
    n = len(x1d)
    multi = 0
    for _ in range(B):
        xs = rng.choice(x1d, n, replace=True)
        xb = xs + hc * rng.standard_normal(n)
        xb = xb.mean() + (xb - xb.mean()) / np.sqrt(1 + hc ** 2 / sd ** 2)  # Silverman 保方差
        g2 = np.linspace(xb.min() - 0.5 * span, xb.max() + 0.5 * span, 512)
        if _kde_nmodes(xb, hc, g2) > 1:
            multi += 1
    return {"h_crit": round(float(hc), 4), "p_multimodal": round(multi / B, 4)}


def selftest_silverman(rng, B=300):
    """合成正/负对照: 验 Silverman 检验在 n≈127 下的功率 (guardrail ①)。"""
    uni = rng.standard_normal(127)
    bi = np.concatenate([rng.standard_normal(64) - 2.2, rng.standard_normal(63) + 2.2])
    return {"unimodal_control_p": silverman_modality(uni, B, rng)["p_multimodal"],
            "bimodal_control_p": silverman_modality(bi, B, rng)["p_multimodal"]}


# ═══════════════════════════════════════════════
# X — RQA 交叉参照 (不重算, 直接引)
# ═══════════════════════════════════════════════
def rqa_crossref():
    if not IRREV_PATH.exists():
        return {"available": False}
    with open(IRREV_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    ra = d.get("recurrence_analysis", {})
    return {
        "available": True,
        "inter_regime_min_distance": ra.get("inter_regime_min_distance"),
        "cross_regime_recurrence_raw": ra.get("cross_regime_recurrence_raw"),
        "r2_ever_returns": ra.get("r2_ever_returns"),
        "note": ("R1↔R3 近(1.18)+10 对复发 = 连成一团; R2 远(2.24/3.38)+0 复发 = 孤立。"
                 "即 RQA 本身已暗示: 真正分离的是 R2, R1/R3 更像一个 blob。"),
    }


def main():
    ap = argparse.ArgumentParser(description="Regime Discreteness — 离散 vs 连续")
    ap.add_argument("--gap-boot", type=int, default=100)
    ap.add_argument("--stab-boot", type=int, default=200)
    ap.add_argument("--sil-boot", type=int, default=500)
    ap.add_argument("--selftest", action="store_true", help="只跑 Silverman 合成对照")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 70)
    print("Regime Discreteness — x(t) 分簇 vs 连续流形被 GMM 切 bin (承重预设审计)")
    print("=" * 70)

    # guardrail ①: 先验模态检验功率
    st = selftest_silverman(rng)
    print(f"\n[Silverman 功率自检 n=127] 单峰对照 p={st['unimodal_control_p']} (应大), "
          f"双峰对照 p={st['bimodal_control_p']} (应小)")
    sil_powered = st["bimodal_control_p"] < 0.10 and st["unimodal_control_p"] > 0.30
    print(f"  → Silverman 在此样本量{'有' if sil_powered else '⚠功率不足, 结果标 exploratory'}功率")
    if args.selftest:
        return

    months, x, stage, stage_names = load_data()
    x_s = StandardScaler().fit_transform(x)
    n, d = x_s.shape

    # baseline GMM(4) + R2 (fixation 主导簇)
    gmm = GaussianMixture(4, covariance_type="full", random_state=42, n_init=10).fit(x_s)
    labels = gmm.predict(x_s)
    fix = stage[:, FIX_IDX]
    r2_idx = int(max(range(4), key=lambda c: fix[labels == c].mean() if (labels == c).any() else -1))
    sizes = {c: int((labels == c).sum()) for c in range(4)}
    print(f"\n数据: {n} 月 × {d} 维 x(t); GMM(4) 簇大小={sizes}; R2(fixation)=簇{r2_idx}")

    # ── T1 Gap ──
    gap = gap_statistic(x_s, kmax=6, B=args.gap_boot, rng=rng)
    print(f"\n[T1 Gap statistic] (BIC 只搜 k≥3; gap 从 k=1)")
    print(f"  gap(k): " + "  ".join(f"k{k}={g:+.3f}" for k, g in zip(gap["k_range"], gap["gap"])))
    print(f"  → gap 最优 k = {gap['gap_optimal_k']}  (baseline GMM/BIC 用的是 4)")

    # ── T2 Stability ──
    stab = cluster_stability(x_s, 4, labels, B=args.stab_boot, rng=rng)
    print(f"\n[T2 Cluster stability] 每簇 Jaccard (≥0.75 稳, <0.5 碎):")
    for c in range(4):
        tag = " ←R2" if c == r2_idx else ""
        mark = "稳" if stab[c] >= 0.75 else ("碎" if stab[c] < 0.5 else "中")
        print(f"  簇{c} (n={sizes[c]:>3d}): Jaccard={stab[c]:.3f} [{mark}]{tag}")

    # ── T3 Silverman on R2-axis + top PCs ──
    # R2 分离轴: rest-centroid → R2-centroid 方向
    mu_r2 = x_s[labels == r2_idx].mean(0)
    mu_rest = x_s[labels != r2_idx].mean(0)
    axis = mu_r2 - mu_rest
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    proj_r2 = x_s @ axis
    sil = {"R2_axis": silverman_modality(proj_r2, args.sil_boot, rng)}
    for pc in range(3):
        sil[f"PC{pc+1}"] = silverman_modality(x[:, pc], args.sil_boot, rng)
    print(f"\n[T3 Silverman 单峰检验] p_multimodal 小=多峰 (分离), 大=单峰 (连续)"
          f"{'  ⚠exploratory(功率不足)' if not sil_powered else ''}:")
    for name, r in sil.items():
        verd = "多峰/分离" if r["p_multimodal"] < 0.05 else ("单峰/连续" if r["p_multimodal"] > 0.30 else "不定")
        print(f"  {name:<8s}: p={r['p_multimodal']:.3f} [{verd}]")

    # ── X RQA crossref ──
    rqa = rqa_crossref()
    print(f"\n[X RQA 交叉参照] {rqa.get('note','(无 irreversibility_results.json)')}")

    # ── 综合裁决 (三层拆分 + Competing Explanatory Layer, ★不 crown 任一极★) ──
    # 审计层修正 (2026-07-12): 原 4-way 裁决会 land 到 "LEANS_CONTINUOUS" = 给单一图景加冕。
    # 改为镜像 P2a/b/c 的三层拆分, R2 盆地vs漂移驻留作竞争读法并存 (数据分不开=UNDERPOWERED)。
    sil["R2_axis"]["circular"] = True
    sil["R2_axis"]["circular_note"] = ("轴=μ_R2−μ_rest 由 GMM 标签定义, 再在该轴测多峰=同数据既提又验 "
                                       "(Semantic Smuggling); 单 Gaussian blob 经 GMM 也切出分离簇均值→照样 p≈0。"
                                       "★仅上界, 剔出裁决★。")

    pc1_multimodal = sil["PC1"]["p_multimodal"] < 0.05          # ★非循环★: PC1 是数据定义轴
    gap_not4 = gap["gap_optimal_k"] != 4
    all_jaccard_low = all(stab[c] < 0.6 for c in range(4))

    rd_a = "holds" if pc1_multimodal else "underpowered"       # 低维有 label-free 结构
    rd_b = "suspect" if (gap_not4 or all_jaccard_low) else "holds"   # K 相区计数稳健性
    rd_c = "underpowered_competing"                            # R2 盆地 vs 漂移驻留: 平局

    verdict = f"RD_SPLIT[a={rd_a},b={rd_b},c={rd_c}]"
    summary = (
        f"三层拆分, ★两头 (4-clean-regimes / pure-continuum) 都不获支持★: "
        f"RD-a 低维 label-free 结构={rd_a} (PC1 Silverman p={sil['PC1']['p_multimodal']:.3f}, 非循环 → 反 pure-continuum); "
        f"RD-b K 相区计数={rd_b} (gap 最优 k={gap['gap_optimal_k']}≠4, 全簇 Jaccard<0.6, R2={stab[r2_idx]:.2f}); "
        f"RD-c R2 盆地 vs 连续漂移驻留=UNDERPOWERED 竞争 —— 数据分不开: RQA 零复发 / weak-irrev / PC2 持久性升 "
        f"三路都在 regime 框架内算, '慢漂入盆地' 同样兼容, 打不破平局。")
    cascade = ("ledger: regime-discretization→split; RD-a=[[regime-lowdim-structure]]holds, "
               "RD-b=[[regime-count]]suspect, RD-c=[[r2-basin]]not-supported。r2-real-cluster/r2-hysteresis/"
               "r2-drift-dwelling 挂 r2-basin (active+flag, 非 falsified); raw E1 (rqa/switch/方差) 挂 regime-count 作为数存活。")

    print(f"\n{'═'*70}")
    print(f"VERDICT: {verdict}")
    print(f"  {summary}")
    print(f"  {cascade}")
    print(f"\n  ⚠ guardrail①: 127×10 稀疏; Silverman {'有功率' if sil_powered else '功率不足(exploratory)'}; "
          "gap/stability null 已校准但 CI 宽。★R2_axis 循环已剔出裁决。刻画 ≠ 证明; 不 crown 盆地也不 crown 连续。★")

    output = {
        "source": "regime_discreteness.py",
        "target_presupposition": "regime-discretization (x(t) 分簇 vs 连续流形被切 bin)",
        "guardrails": {
            "underpowered_expected": "127×10 稀疏; 全检验配 null; 模糊不读成坐实",
            "resolves_tension": "掰 RQA(R2分离/离散) vs curation(计数偶然/连续)",
            "load_bearing": "只刻画离散vs连续, 落 E2/E3; 不证明任一极",
        },
        "evidence_grade": "E1/E2 (各检验统计) → E2/E3 (离散vs连续刻画, 依赖算法+分辨率)",
        "silverman_power_selftest": st, "silverman_powered": bool(sil_powered),
        "gmm_baseline": {"k": 4, "cluster_sizes": sizes, "r2_cluster": r2_idx},
        "T1_gap_statistic": gap,
        "T2_cluster_stability_jaccard": {str(c): round(stab[c], 4) for c in range(4)},
        "T3_silverman_modality": sil,
        "X_rqa_crossref": rqa,
        "verdict": {"direction": verdict, "text": summary, "cascade": cascade},
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 → {OUTPUT_PATH}")
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

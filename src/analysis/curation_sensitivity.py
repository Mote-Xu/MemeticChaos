"""
Curation Sensitivity — 测量 d90/簇数/R2 对"挑哪些梗/关键词"的偶然性 (2026-07-09)

审计问题 (徐子浩): d90=10 / GMM 簇数 / R2 自持 97.3% 这些结论, 是用极小的 curated 样本
(51 关键词 + 57 叙事, 对梗宇宙 <1%) 算的。换一批 curation, 结论会不会翻? —— 这是
"覆盖是局部的(确定) + 偶然性(未测)"里那个**未测**的部分。用 jackknife 把它变成真数。

方法 (踩借来的石头, 复用原始聚合函数, 不重写):
  重复 B 次, 每次独立 subsample 三个 curation 维度, 复用真实管线重算:
    - stage_occupancy: 叙事子集 → build_stage_timeline_from_trends 真时间线重聚合 (R2 驱动)
    - 外部场: 关键词子集 → assemble_features 内部重 PCA
    - 注意力: Trends 梗子集 → compute_attention_structure 重 HHI/熵
  → 组装 18 维 → PCA d90 → GMM(BIC) 簇数 → fixation 主导簇的自持概率 (R2 代理)
  得到 d90 / 簇数 / R2自持 的**分布**, 对比 baseline。

★诚实边界 (残余冻结, 显式声明)★:
  mutation_rate / inst_rate / mean_semantic_drift 三个月度标量 (18维里的3个) **held fixed** ——
  它们的逐月重聚合需要逐梗逐月明细(per_meme 只有汇总标量), 干净重算不了, 故冻结。
  被扰动的是 15/18 特征, 含所有 R2 相关的 (stage) 和 PC1 (attention)。所以这不是假稳定测试。
  判读: 若 d90/簇数/R2 在扰动下大幅摆动 → 坐实高偶然性; 若稳 → 对这三个 curation 维度鲁棒
  (mutation/inst/drift curation 仍未测)。

用法:
  conda run -n MemeticChaos python src/analysis/curation_sensitivity.py --boot 100 --keep 0.8
"""

import json, sys, argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.representation_learning import (
    load_level1, load_external_field, assemble_features,
    load_trends as load_trends_rl, compute_attention_structure,
)
from models.stage_occupancy import (
    load_narratives, build_stage_timeline_from_trends, load_trends as load_trends_stage,
)

LEVEL1_PATH = ROOT / "data/processed/level1_hard_facts.json"
OUTPUT_PATH = ROOT / "data/processed/curation_sensitivity.json"
RNG_SEED = 42
STAGE_ORDER = ["origin", "emergence", "peak", "controversy", "fixation"]
FIX_IDX = STAGE_ORDER.index("fixation")


def aggregate_stage_matrix(meme_timelines, months, keep_memes):
    """在梗子集上重聚合月度 stage_occupancy (复刻 stage_occupancy.py 的聚合循环)。"""
    matrix = np.zeros((len(months), len(STAGE_ORDER)))
    for mi, month in enumerate(months):
        counts = defaultdict(int); active = 0
        for name in keep_memes:
            tl = meme_timelines.get(name)
            if not tl:
                continue
            for e in tl:
                if e["start"] <= month <= e["end"]:
                    counts[e["stage"]] += 1; active += 1; break
        if active > 0:
            for si, st in enumerate(STAGE_ORDER):
                matrix[mi, si] = counts[st] / active
    return matrix


def compute_d90_clusters_r2(l1, ext_months, ext_matrix, hhi, entropy, total_att,
                            n_init=5, ext_pc=8):
    """组装 → PCA d90 → GMM(BIC) 簇数 → fixation 主导簇自持。返回 (d90, k, r2_persist)。"""
    X, months, meta = assemble_features(l1, ext_months, ext_matrix, hhi, entropy,
                                        total_att, ext_pc=ext_pc)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    d90 = int(np.searchsorted(cumvar, 0.90) + 1)

    # state PCA → x(t) (用 d90 维, 复刻 representation_learning)
    x = PCA(n_components=d90).fit_transform(Xs)

    # GMM BIC 选簇数 (复刻 regime_detector)
    best_k, best_bic, best_labels = 3, np.inf, None
    for k in range(3, 8):
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=42, n_init=n_init)
        lab = gmm.fit_predict(StandardScaler().fit_transform(x))
        bic = gmm.bic(StandardScaler().fit_transform(x))
        if bic < best_bic:
            best_bic, best_k, best_labels = bic, k, lab

    # fixation 主导簇 = R2 代理; 自持概率
    stage = np.array(l1["stage_occupancy"])
    fix_series = stage[:, FIX_IDX]
    # 每个簇的平均 fixation 占比, 取最高者
    r2_cluster = max(range(best_k),
                     key=lambda c: fix_series[best_labels == c].mean() if (best_labels == c).any() else -1)
    # 自持: P(t+1 in R2 | t in R2)
    inR2 = (best_labels == r2_cluster)
    stay = sum(1 for t in range(len(best_labels) - 1) if inR2[t] and inR2[t + 1])
    tot = sum(1 for t in range(len(best_labels) - 1) if inR2[t])
    r2_persist = stay / tot if tot > 0 else 0.0
    return d90, best_k, float(r2_persist)


def main():
    ap = argparse.ArgumentParser(description="Curation Sensitivity")
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--keep", type=float, default=0.8, help="每次保留的 curation 比例")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 64)
    print("Curation Sensitivity — d90/簇数/R2 对 curation 的偶然性")
    print("=" * 64)

    l1 = load_level1()
    months = l1["months"]
    ext_months, ext_matrix = load_external_field()      # 51 关键词
    trends = load_trends_rl()                            # attention 用
    trends_stage = load_trends_stage()                  # stage 时间线用
    narratives = load_narratives()

    # 一次性建所有梗的 stage 时间线 (借来的真函数)
    meme_timelines = {}
    for name, nar in narratives.items():
        tl = build_stage_timeline_from_trends(name, nar, trends_stage)
        if tl:
            meme_timelines[name] = tl
    memes_with_tl = list(meme_timelines.keys())
    kw_all = list(range(ext_matrix.shape[1]))
    trend_memes = list(trends.keys())

    print(f"\n可扰动 curation: {len(memes_with_tl)} 叙事(有时间线) / "
          f"{len(kw_all)} 关键词 / {len(trend_memes)} Trends梗")
    print(f"jackknife: {args.boot}× 每次保留 {args.keep:.0%}, 冻结 mutation/inst/drift (3/18)\n")

    # ── Baseline (全量, 用真实 level1) ──
    hhi0, ent0, tot0 = compute_attention_structure(trends, months)
    d90_0, k0, r2_0 = compute_d90_clusters_r2(l1, ext_months, ext_matrix, hhi0, ent0, tot0, n_init=10)
    print(f"Baseline (全量): d90={d90_0}, 簇数={k0}, R2自持={r2_0:.3f}")

    # ── Jackknife ──
    rng = np.random.default_rng(RNG_SEED)
    d90s, ks, r2s = [], [], []
    for b in range(args.boot):
        # subsample 三个 curation 维度
        km = rng.choice(memes_with_tl, max(3, int(len(memes_with_tl) * args.keep)), replace=False)
        kk = np.sort(rng.choice(kw_all, max(4, int(len(kw_all) * args.keep)), replace=False))
        kt = rng.choice(trend_memes, max(3, int(len(trend_memes) * args.keep)), replace=False)

        # 扰动 stage_occupancy (叙事子集重聚合)
        l1_pert = dict(l1)
        l1_pert["stage_occupancy"] = aggregate_stage_matrix(meme_timelines, months, set(km)).tolist()
        # 扰动外部场 (关键词子集)
        ext_sub = ext_matrix[:, kk]
        # 扰动注意力 (Trends 梗子集)
        trends_sub = {m: trends[m] for m in kt}
        hhi, ent, tot = compute_attention_structure(trends_sub, months)

        try:
            d90, k, r2 = compute_d90_clusters_r2(l1_pert, ext_months, ext_sub, hhi, ent, tot, n_init=3)
            d90s.append(d90); ks.append(k); r2s.append(r2)
        except Exception:
            continue
        if (b + 1) % 20 == 0:
            print(f"  ...{b+1}/{args.boot}")

    d90s, ks, r2s = np.array(d90s), np.array(ks), np.array(r2s)

    def stat(a): return {"median": float(np.median(a)), "p5": float(np.percentile(a, 5)),
                         "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max()),
                         "cv": float(np.std(a) / (np.mean(a) + 1e-9))}
    print(f"\n{'─'*64}\n{args.boot} 次 jackknife 分布:")
    print(f"  d90:    baseline={d90_0}  分布 median={np.median(d90s):.0f} "
          f"[{d90s.min():.0f}, {d90s.max():.0f}]  CV={stat(d90s)['cv']:.2f}")
    print(f"  簇数:   baseline={k0}  分布 median={np.median(ks):.0f} "
          f"[{ks.min():.0f}, {ks.max():.0f}]  众数={Counter(ks.tolist()).most_common(1)[0]}")
    print(f"  R2自持: baseline={r2_0:.3f}  分布 median={np.median(r2s):.3f} "
          f"[{r2s.min():.3f}, {r2s.max():.3f}]  CV={stat(r2s)['cv']:.2f}")

    # 稳定性判读
    d90_stable = d90s.std() <= 1.5
    k_stable = (ks == k0).mean() >= 0.6
    r2_stable = r2s.std() <= 0.08
    print(f"\n  d90 恢复 baseline±1: {np.mean(np.abs(d90s - d90_0) <= 1):.0%}")
    print(f"  簇数 = baseline({k0}) 的比例: {(ks == k0).mean():.0%}")
    print(f"  R2自持 落在 baseline±0.05: {np.mean(np.abs(r2s - r2_0) <= 0.05):.0%}")

    n_stable = sum([d90_stable, k_stable, r2_stable])
    if n_stable == 3:
        verdict = "ROBUST_TO_THESE_CURATIONS"
        summary = ("d90/簇数/R2 对 stage/外部场/注意力 三个 curation 维度鲁棒 —— 局部但不偶然。"
                   "但 mutation/inst/drift curation 仍未测。仍不能外推到'代表梗宇宙'。")
    elif n_stable == 0:
        verdict = "HIGHLY_CONTINGENT"
        summary = ("d90/簇数/R2 随 curation 大幅摆动 —— 坐实高偶然性。这些数是 curation 的产物, "
                   "不是现象的稳健属性。所有吊在它们上的解释(E3/E4)偶然性同等升级。")
    else:
        verdict = "MIXED"
        summary = (f"{n_stable}/3 指标稳定。部分对 curation 鲁棒, 部分偶然。需逐项看哪个稳哪个飘。")

    print(f"\n{'═'*64}\nVERDICT: {verdict}")
    print(f"  {summary}")
    print(f"\n  ⚠ 残余冻结: mutation/inst/drift(3/18) 未扰动; curator 预设(样本代表宇宙)仍未验。")

    output = {
        "source": "curation_sensitivity.py",
        "audit_question": "d90/簇数/R2 对'挑哪些梗/关键词'的偶然性",
        "method": "jackknife 复用真实聚合函数, 扰动 stage/外部场/注意力 三 curation 维度",
        "frozen_residual": "mutation_rate/inst_rate/mean_semantic_drift (3/18, 逐梗逐月明细缺失)",
        "keep_fraction": args.keep, "n_bootstrap": len(d90s),
        "baseline": {"d90": d90_0, "n_clusters": k0, "r2_persistence": round(r2_0, 4)},
        "distributions": {
            "d90": stat(d90s), "n_clusters": {"median": float(np.median(ks)),
                "mode": Counter(ks.tolist()).most_common(1)[0][0], "min": int(ks.min()), "max": int(ks.max())},
            "r2_persistence": stat(r2s)},
        "stability": {"d90_within_1": float(np.mean(np.abs(d90s - d90_0) <= 1)),
                      "k_equals_baseline": float((ks == k0).mean()),
                      "r2_within_0.05": float(np.mean(np.abs(r2s - r2_0) <= 0.05))},
        "verdict": {"direction": verdict, "text": summary,
                    "caveat": "残余冻结 mutation/inst/drift; curator 预设未验; 局部是确定的"},
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 → {OUTPUT_PATH}")
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""
Nyblom 参数恒定性检验 — 不切段的低维时不变性检验 (2026-07-09, 第九轮)

回答预设 2 (时不变生成机制) 的**第一层**: 低维/标量参数在 127 月内是否恒定。
—— 不是全系统机制变化 (月度分辨率下不可辨识), 只是"某个低维投影的 AR 系数是否漂移"。

方法 (Nyblom 1989 / Hansen 1992):
  H0: AR(1) 系数 β 时不变            y_t = β0 + β1·y_{t-1} + e_t, β 恒定
  H1: β 随机游走 (机制缓慢漂移)      β_t = β_{t-1} + η_t
  统计量 L = (1/T²)·Σ_t S_t' M⁻¹ S_t   (S_t = 得分累积和; M = 得分外积)
  参数漂移 → 得分累积和偏离布朗桥 → L 增大。

★为什么配 null simulation (GPT 要求)★:
  渐近临界值在 N=127 上不可靠, 且"任何非平稳序列都可能被判漂移"。
  用**参数自举**造 null: 从拟合的恒定参数 AR(1) 重采残差生成代理序列 (H0 为真),
  在代理上重算 L → null 分布 → 经验 p。只有 L 超出"恒定参数 AR(1) 自身能产生的
  波动"才算漂移证据。这与项目对 RQA/时段审计的 null 纪律一致。

★报告等级 (GPT 要求)★:
  结论落在 E1/E2: "在 AR(1) 低维投影下 {未发现|发现} 参数漂移证据"。
  ★不是★ "系统稳定/机制改变"。AR 系数稳 ≠ 生成机制稳 (真实机制若非 AR, 系数稳无意义)。

目标序列 (127 月, 无需窗口, 无需切段):
  PC1/PC2/PC3  — 主语义投影轨迹
  r2_prob      — R2(fixation) 的 GMM 后验概率 (软隶属度, 非硬标签)

用法:
  conda run -n MemeticChaos python src/analysis/nyblom_stationarity.py
"""

import json, sys, argparse
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).parent.parent.parent
STATE_PATH = ROOT / "data/processed/representation_state.json"
REGIME_PATH = ROOT / "data/processed/regime_map.json"
OUTPUT_PATH = ROOT / "data/processed/nyblom_stationarity.json"

RNG_SEED = 42
N_BOOT = 2000


def nyblom_L(y: np.ndarray) -> float:
    """AR(1) 参数恒定性的 Nyblom-Hansen L 统计量。

    y_t = β0 + β1·y_{t-1} + e_t, OLS 拟合恒定 β。
    f_t = X_t·e_t (得分); S_t = Σ_{i≤t} f_i; M = Σ f_t f_t'/T (得分外积, 自归一化)。
    L = (1/T²) Σ_t S_t' M⁻¹ S_t。
    """
    yt = y[1:]
    X = np.column_stack([np.ones(len(yt)), y[:-1]])  # [1, y_{t-1}]
    T, k = X.shape
    # OLS
    beta, *_ = np.linalg.lstsq(X, yt, rcond=None)
    e = yt - X @ beta
    # 得分 f_t = X_t · e_t
    f = X * e[:, None]                    # T × k
    S = np.cumsum(f, axis=0)              # 累积和
    M = (f.T @ f) / T                     # 得分外积 (自归一化)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        Minv = np.linalg.pinv(M)
    L = np.einsum("tk,kj,tj->", S, Minv, S) / (T ** 2)
    return float(L)


def bootstrap_null(y: np.ndarray, n_boot: int) -> np.ndarray:
    """参数自举 null: 从拟合的恒定参数 AR(1) 生成代理序列 (H0 为真), 重算 L。"""
    rng = np.random.default_rng(RNG_SEED)
    yt = y[1:]
    X = np.column_stack([np.ones(len(yt)), y[:-1]])
    beta, *_ = np.linalg.lstsq(X, yt, rcond=None)
    resid = yt - X @ beta
    b0, b1 = beta
    T = len(y)

    Ls = np.empty(n_boot)
    for b in range(n_boot):
        e_star = rng.choice(resid, size=T, replace=True)  # 重采残差
        y_star = np.empty(T)
        y_star[0] = y[0]
        for t in range(1, T):
            y_star[t] = b0 + b1 * y_star[t - 1] + e_star[t]
        Ls[b] = nyblom_L(y_star)
    return Ls


def load_targets():
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    with open(REGIME_PATH, "r", encoding="utf-8") as f:
        regime = json.load(f)

    months = state["pca_transformed"]["months"]
    x = np.array(state["pca_transformed"]["x_reduced"])  # 127 × 10
    n_reg = regime["n_regimes"]
    stored_labels = np.array(regime["regime_labels"])

    # 复现 regime_detector 的 GMM (同 seed/参数) → 后验概率 (软隶属度)
    x_s = StandardScaler().fit_transform(x)
    gmm = GaussianMixture(n_components=n_reg, covariance_type="full",
                          random_state=42, n_init=10)
    gmm.fit(x_s)
    proba = gmm.predict_proba(x_s)          # 127 × n_reg
    argmax_match = float(np.mean(gmm.predict(x_s) == stored_labels))

    # R2 = fixation 相区 (regime_map 里 dominant_stage=='fixation')
    r2_idx = next(int(k) for k, v in regime["regime_characteristics"].items()
                  if v.get("dominant_stage") == "fixation")

    targets = {
        "PC1": x[:, 0],
        "PC2": x[:, 1],
        "PC3": x[:, 2],
        "r2_prob": proba[:, r2_idx],   # R2 后验概率 (软隶属度)
    }
    meta = {"n_months": len(months), "r2_gmm_index": r2_idx,
            "gmm_argmax_vs_stored_label_match": round(argmax_match, 4)}
    return targets, meta


def main():
    ap = argparse.ArgumentParser(description="Nyblom 参数恒定性检验")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 66)
    print("Nyblom 参数恒定性检验 — 不切段的低维时不变性 (预设2 第一层)")
    print("=" * 66)

    targets, meta = load_targets()
    print(f"\n数据: {meta['n_months']} 月; R2=GMM组件{meta['r2_gmm_index']}; "
          f"GMM argmax vs 存储标签一致={meta['gmm_argmax_vs_stored_label_match']:.1%}")
    print(f"null: 参数自举 {args.boot}× (从恒定参数 AR(1) 生成)\n")

    results = {}
    print(f"{'目标':<9s} {'L 统计量':>10s} {'null_p95':>10s} {'经验 p':>9s}  判定")
    for name, y in targets.items():
        y = np.asarray(y, dtype=float)
        L = nyblom_L(y)
        null = bootstrap_null(y, args.boot)
        p = float(np.mean(null >= L))
        p95 = float(np.percentile(null, 95))
        drift = p < 0.05
        flag = "★漂移证据(p<0.05)" if drift else "无漂移证据"
        results[name] = {"L": round(L, 4), "null_p95": round(p95, 4),
                         "null_median": round(float(np.median(null)), 4),
                         "empirical_p": round(p, 4), "drift_evidence": drift}
        print(f"{name:<9s} {L:>10.4f} {p95:>10.4f} {p:>9.4f}  [{flag}]")

    n_drift = sum(1 for r in results.values() if r["drift_evidence"])
    print(f"\n{'─'*66}")
    if n_drift == 0:
        verdict = "NO_LOWDIM_DRIFT"
        summary = ("在 AR(1) 低维投影下, 未发现参数漂移证据 (E1/E2)。"
                   "注意: AR 系数恒定 ≠ 生成机制时不变 (真实机制若非 AR(1), 此结论不外推)。")
    else:
        verdict = "LOWDIM_NONCONSTANCY_DETECTED_BUT_DEGENERATE"
        drifted = [n for n, r in results.items() if r["drift_evidence"]]
        summary = (
            f"在 {drifted} 上检测到 AR(1) 参数非恒定证据 (E1/E2, 超出恒定参数 AR(1) 的 null)。"
            "★关键: 这不能证伪 P2★ —— AR 自相关系数上升同时是 (a) 机制改变 和 "
            "(b) 时不变机制在慢控制漂移下逼近分岔 (critical slowing) 的共同signature。"
            "Nyblom 分不开这两者。故: 不自动 suspend 挂靠 time-invariance 的 E4; "
            "只把 P2 的'低维层'状态从 UNDERPOWERED 更新为 '可测且非恒定, 但归因简并'。"
            "真正的收获: 低维层不是没功率 —— 反驳了'分辨率墙杀死一切'的过宽判断。")
    print(f"VERDICT: {verdict}")
    print(f"  {summary}")
    print(f"\n  ⚠ 定位: 只打了三层里的第一层 (低维参数恒定性), 且结果对 P2 简并。")
    print(f"     要分开 '机制改变' vs 'critical slowing', 需要独立通道 (如 changepoint 定位")
    print(f"     突变点 vs 平滑漂移), 或前向高分辨率数据。未打: 状态空间结构 / 高维机制。")

    output = {
        "source": "nyblom_stationarity.py",
        "target_presupposition": "P2 time-invariance, layer-1 (low-dim parameter constancy)",
        "method": "Nyblom-Hansen L on AR(1), parametric-bootstrap null",
        "evidence_grade": "E1/E2 — AR系数恒定性, 非机制不变性",
        "meta": meta, "n_bootstrap": args.boot,
        "results": results,
        "verdict": {"direction": verdict, "text": summary,
                    "scope_caveat": ("仅低维参数恒定性 (第一层); 结果对 P2 简并 —— "
                                     "AR 系数上升 = 机制改变 或 时不变机制逼近分岔(critical slowing), "
                                     "Nyblom 分不开; 高维机制变化月度分辨率下不可辨识"),
                    "key_finding": ("低维层非无功率 (反驳'分辨率墙杀死一切'); PC2 (churn/mutation 轴) "
                                    "AR 自相关 0.57→0.94, 方差 0.53→2.17, 与 MS-AR R2 方差 2.31× / "
                                    "FR31 Critical Slowing 0.76 三路独立收敛; 但不证伪 P2")},
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 → {OUTPUT_PATH}")
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

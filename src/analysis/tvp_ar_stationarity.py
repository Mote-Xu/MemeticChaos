"""
TVP-AR 参数漂移估计 — Nyblom 的估计版对应物 (2026-07-11, 第十轮)

审计层 (d0273e9f) 批准本任务时给了 4 条修正, 全部写进本脚本的框架与统计:

★1. 这不是"独立通道"★
  Nyblom-Hansen L 统计量 = 本状态空间模型在 σ²_η=0 处的 score (LM) 检验。
  两者是**同一假设**的检验版 (Nyblom) vs 估计版 (本脚本), 不独立。
  本脚本的增量价值 = 漂移的**轨迹 β(t)** + **幅度 σ²_η**, 而非"独立确认参数非恒定"。

★2. 不 resolve 预设2 (time-invariance)★
  即使 σ²_η>0 (β 时变), 归因仍简并 (Nyblom 已撞同一堵墙):
    (a) 生成机制真的改变   (b) 固定机制逼近分岔 critical slowing   (c) 观测算子改变
  TVP-AR 只**刻画**漂移的形状与大小, 分不开这三支。产出照此措辞, 落 E1 (统计描述)。

★3. 目标 PC1+PC2, 用历史 x(t), 不用 cov_trace★
  cov_trace 是前向 (2026-06/07 两月) 独立表示, 跑不了 127 月 TVP。
  历史链只有 10 维 x(t) (representation_state.json)。本脚本只碰历史链 (H 侧)。

★4. 预期 UNDERPOWERED★
  N=127 下 σ²_η 的 MLE 有边界堆积 (pile-up-at-zero) + 低功率。
  σ̂²_η≈0 ★不等于★ "P2 成立 / 系统时不变", 只可能是分辨率不足看不见漂移。
  报 profile 似然曲线 + 参数自举 null (H0: q=0) 的经验 p + 堆积比例, 把无漂移明确标成可能的
  UNDERPOWERED, 不读成坐实时不变。

──────────────────────────────────────────────────────────────────────
模型 (匹配 Nyblom 的 AR(p), 同序列、同 AIC 选阶):
  观测:  y_t = z_t'·β_t + e_t,   z_t = [1, y_{t-1},…,y_{t-p}],  e_t ~ N(0, σ²_e)
  状态:  β_t = β_{t-1} + η_t,     η_t ~ N(0, σ²_η·I)   (共享标量漂移方差, Nyblom L 检的就是这个)
  q = σ²_η/σ²_e 信噪比; σ²_e 用集中似然消去; q 网格 profile 求 MLE。
  Kalman 滤波 (精确扩散初始化: 前 k 个观测只做初始化, 不计入似然) + RTS 平滑求 β(t)。

对应 Evidence Ledger: β(t) 轨迹 + σ̂²_η 是 E1 (统计描述); 归因 (a/b/c) 留在 E4 简并。

用法:
  conda run -n MemeticChaos python src/analysis/tvp_ar_stationarity.py
  conda run -n MemeticChaos python src/analysis/tvp_ar_stationarity.py --boot 1000 --json
"""

import json, sys, argparse
from pathlib import Path
import numpy as np

# 复用 Nyblom 的数据加载与 AR 设定 —— 保证 TVP-AR 是**同一** AR 模型的估计版
from nyblom_stationarity import select_ar_order, _fit_ar, ljung_box, load_targets

ROOT = Path(__file__).parent.parent.parent
OUTPUT_PATH = ROOT / "data/processed/tvp_ar_stationarity.json"

RNG_SEED = 42
TARGETS = ["PC1", "PC2"]      # 修正3: 历史 x(t) 的主语义投影 (PC2 是 Nyblom 已标出漂移那个)


def build_design(y: np.ndarray, p: int):
    """y_t = z_t'β_t + e_t 的回归设计, 列构造与 nyblom._fit_ar 完全一致。"""
    yt = y[p:]
    cols = [np.ones(len(yt))] + [y[p - i - 1: len(y) - i - 1] for i in range(p)]
    Z = np.column_stack(cols)          # T_eff × (p+1)
    return Z, yt


def kalman_run(Z, yt, q, P0=1e6, smooth=False):
    """局部水平 TVP 回归的 Kalman 滤波 (σ²_e=1, Q=q·I, 扩散初始化)。

    前 k=p+1 个观测视为扩散初始化, 不计入集中似然 (Durbin-Koopman 扩散似然惯例),
    使结果不依赖 P0 的具体大小。返回集中 loglik; smooth=True 时另返回 RTS 平滑的 β(t)。
    """
    T, k = Z.shape
    Q = np.eye(k) * q
    beta = np.zeros(k)
    P = np.eye(k) * P0

    v = np.empty(T); F = np.empty(T)
    bp = np.empty((T, k)); Pp = np.empty((T, k, k))   # 预测量 (平滑用)
    bf = np.empty((T, k)); Pf = np.empty((T, k, k))   # 滤波量

    for t in range(T):
        b_pred = beta                    # β_{t|t-1}=β_{t-1|t-1} (随机游走)
        P_pred = P + Q
        z = Z[t]
        vt = yt[t] - z @ b_pred
        Ft = z @ P_pred @ z + 1.0        # σ²_e=1 (集中)
        K = P_pred @ z / Ft
        beta = b_pred + K * vt
        P = P_pred - np.outer(K, z) @ P_pred
        v[t] = vt; F[t] = Ft
        bp[t] = b_pred; Pp[t] = P_pred
        bf[t] = beta;   Pf[t] = P

    # 集中似然: 跳过前 k 个 (扩散初始化), 只用稳定段
    use = slice(k, T)
    vu, Fu = v[use], F[use]
    n = len(vu)
    sig2e = float(np.mean(vu ** 2 / Fu))
    ll = -0.5 * (n * np.log(2 * np.pi) + n * np.log(sig2e + 1e-300)
                 + np.sum(np.log(Fu)) + n)

    if not smooth:
        return ll, sig2e, None

    # RTS 平滑 → β_{t|T}
    bs = bf.copy(); Ps = Pf.copy()
    for t in range(T - 2, -1, -1):
        try:
            J = Pf[t] @ np.linalg.inv(Pp[t + 1])
        except np.linalg.LinAlgError:
            J = Pf[t] @ np.linalg.pinv(Pp[t + 1])
        bs[t] = bf[t] + J @ (bs[t + 1] - bp[t + 1])
        Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T
    return ll, sig2e, bs


def q_grid(n=50, qmax=10.0):
    """q=0 (H0) + 对数网格。"""
    return np.concatenate([[0.0], np.logspace(-6, np.log10(qmax), n)])


def fit_tvp(Z, yt, grid):
    """在 q 网格上 profile 集中似然, 返回 MLE。"""
    lls = np.array([kalman_run(Z, yt, q)[0] for q in grid])
    i = int(np.argmax(lls))
    return {"q_hat": float(grid[i]), "ll_hat": float(lls[i]),
            "ll_q0": float(lls[0]), "lr": float(2 * (lls[i] - lls[0])),
            "at_boundary": i == 0, "grid": grid, "lls": lls}


def bootstrap_null(y: np.ndarray, p: int, grid, n_boot: int):
    """参数自举 H0 (q=0, 恒定系数 AR(p)): 造代理序列, 重估 q̂ 与 LR。

    返回 q̂* 与 LR* 的 null 分布 → 经验 p + 边界堆积比例。
    与 nyblom.bootstrap_null 同一造数逻辑 (从拟合的恒定 AR(p) 重采残差)。
    """
    rng = np.random.default_rng(RNG_SEED)
    _, _, beta, resid = _fit_ar(y, p)
    T = len(y)
    q_star = np.empty(n_boot); lr_star = np.empty(n_boot)
    for b in range(n_boot):
        e = rng.choice(resid, size=T, replace=True)
        ys = np.empty(T); ys[:p] = y[:p]
        for t in range(p, T):
            lag = np.array([1.0] + [ys[t - i - 1] for i in range(p)])
            ys[t] = lag @ beta + e[t]
        ys = (ys - ys.mean()) / (ys.std() + 1e-12)     # 与主拟合同样标准化
        Zs, yts = build_design(ys, p)
        fit = fit_tvp(Zs, yts, grid)
        q_star[b] = fit["q_hat"]; lr_star[b] = fit["lr"]
    return q_star, lr_star


def persistence_path(bs: np.ndarray, p: int):
    """AR(p) 持久性轨迹 (前/后半段均值) —— 漂移的可读摘要。

    ★对 AR(p) 持久性不是单个 φ_1★ (审计 nit): 而是 Σφ_i (AR 系数和) 或特征根主模。
    两者都报: sum_phi = Σφ_i 是 AR 系数和 (→1 时持久性→∞; ★注意★ 长程乘子是 1/(1-Σφ), 这里
    报的是 Σφ 本身, 不是乘子); dom_root = |dominant root| 是标准 persistence。
    注意这是"漂移的人读摘要", 漂移证据本身来自 σ²_η>0 (整个 β 向量共享漂移), 与此摘要口径无关。
    """
    ar = bs[:, 1:1 + p]                    # 列 0=截距; 列 1..p = φ_1..φ_p, 逐月
    sum_phi = ar.sum(axis=1)               # Σφ_i(t) = AR 系数和 (非长程乘子)
    dom = np.empty(len(ar))
    for t in range(len(ar)):
        # 特征多项式 z^p - φ_1 z^{p-1} - … - φ_p; persistence = 主根模 |dominant root|
        roots = np.roots(np.concatenate([[1.0], -ar[t]]))
        dom[t] = float(np.max(np.abs(roots)))
    half = len(sum_phi) // 2
    r = lambda v: round(float(v), 4)
    return {
        "measure": "AR(p): sum_phi=Σφ_i (AR coeff sum, NOT long-run multiplier), dom_root=|dominant root|",
        "ar_order": p,
        "sum_phi_early": r(sum_phi[:half].mean()), "sum_phi_late": r(sum_phi[half:].mean()),
        "dom_root_early": r(dom[:half].mean()), "dom_root_late": r(dom[half:].mean()),
        "sum_phi_path": [r(x) for x in sum_phi],
        "dom_root_path": [r(x) for x in dom],
    }


def main():
    ap = argparse.ArgumentParser(description="TVP-AR 参数漂移估计 (Nyblom 估计版对应物)")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--gridn", type=int, default=50)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("TVP-AR 参数漂移估计 — Nyblom 的估计版对应物 (预设2 第一层, 估计视角)")
    print("=" * 70)
    print("★不是独立通道 (= Nyblom 同一假设的估计版) | 不 resolve P2 (归因简并) | 预期 UNDERPOWERED")

    targets, meta = load_targets()
    grid = q_grid(args.gridn)
    print(f"\n数据: {meta['n_months']} 月历史 x(t) (H 侧); q 网格 {len(grid)} 点 [0, 10]; "
          f"null 自举 {args.boot}×\n")

    results = {}
    hdr = f"{'目标':<6s} {'AR阶':>4s} {'LB_p':>6s} {'q̂':>9s} {'σ̂²_η':>9s} {'LR':>7s} {'boot_p':>7s} {'堆积%':>6s}  判定"
    print(hdr)
    for name in TARGETS:
        y = np.asarray(targets[name], float)
        p = select_ar_order(y, max_p=4)                  # 与 Nyblom 同一 AIC 选阶
        _, _, _, resid = _fit_ar(y, p)
        lb_p = ljung_box(resid, lags=10, n_params=p + 1)

        ys = (y - y.mean()) / (y.std() + 1e-12)          # 标准化 → σ²_η 各向同性可解释
        Z, yt = build_design(ys, p)
        fit = fit_tvp(Z, yt, grid)
        _, sig2e, bs = kalman_run(Z, yt, fit["q_hat"], smooth=True)
        sig2_eta = fit["q_hat"] * sig2e

        q_star, lr_star = bootstrap_null(y, p, grid, args.boot)
        boot_p = float(np.mean(lr_star >= fit["lr"]))
        pileup = float(np.mean(q_star == 0.0))           # H0 下 MLE 堆在 0 的比例
        drift = boot_p < 0.05 and not fit["at_boundary"]

        pers = persistence_path(bs, p)
        flag = ("★漂移(估计)" if drift else "无漂移/UNDERPOWERED") + \
               ("" if lb_p > 0.05 else "⚠残差非白噪声")

        results[name] = {
            "ar_order": p, "ljung_box_p": round(lb_p, 4),
            "residual_white_noise": lb_p > 0.05,
            "q_hat": round(fit["q_hat"], 6), "sigma2_eta": round(sig2_eta, 6),
            "sigma2_e": round(sig2e, 6), "lr_stat": round(fit["lr"], 4),
            "mle_at_boundary_zero": fit["at_boundary"],
            "bootstrap_p": round(boot_p, 4),
            "null_pileup_at_zero_frac": round(pileup, 4),
            "drift_estimated": drift,
            "persistence": pers,
            "profile": {"q": [round(float(x), 6) for x in grid],
                        "loglik": [round(float(x), 4) for x in fit["lls"]]},
        }
        print(f"{name:<6s} {p:>4d} {lb_p:>6.3f} {fit['q_hat']:>9.5f} {sig2_eta:>9.5f} "
              f"{fit['lr']:>7.3f} {boot_p:>7.4f} {pileup*100:>5.1f}%  [{flag}]  "
              f"Σφ:{pers['sum_phi_early']}→{pers['sum_phi_late']} ρ主根:{pers['dom_root_early']}→{pers['dom_root_late']}")

    n_drift = sum(1 for r in results.values() if r["drift_estimated"])
    print(f"\n{'─'*70}")
    if n_drift > 0:
        drifted = [n for n, r in results.items() if r["drift_estimated"]]
        verdict = "DRIFT_ESTIMATED_ATTRIBUTION_DEGENERATE"
        summary = (
            f"在 {drifted} 上, TVP-AR 估到非零漂移 (σ̂²_η>0, boot_p<0.05), 并给出 β(t) 轨迹。"
            "★这是 Nyblom 同一假设的估计版, 不是独立确认★ —— 增量是漂移的**幅度+轨迹**。"
            "★不 resolve P2★: β 时变对归因三支 (机制改变 / critical slowing / 观测算子改变) 简并, "
            "TVP-AR 分不开。落 E1 (统计描述); 归因留 E4-pending。")
    else:
        verdict = "UNDERPOWERED_NO_DRIFT_ESTIMATED"
        summary = (
            "未估到显著漂移 (q̂ 堆在边界 0 或 boot_p≥0.05)。★σ̂²_η≈0 ≠ 时不变★ —— "
            "N=127 下 σ²_η 的 MLE 本就 pile-up-at-zero + 低功率 (见 null_pileup_at_zero_frac), "
            "看不见漂移可能只是分辨率不足。这是 UNDERPOWERED, 不是坐实 P2。")
    print(f"VERDICT: {verdict}")
    print(f"  {summary}")

    # 与 Nyblom 交叉参照 (同一假设两视角, 应一致; 不一致则如实报告)
    print(f"\n  参照 Nyblom (nyblom_stationarity.json): 若存在, 对同一 PC 的检验版结论应与本估计版一致。")

    output = {
        "source": "tvp_ar_stationarity.py",
        "target_presupposition": "P2 time-invariance, layer-1 (low-dim parameter drift, ESTIMATION view)",
        "relation_to_nyblom": ("NOT independent — Nyblom L 是本模型 σ²_η=0 的 score test; "
                               "本脚本是同一假设的估计版, 增量=β(t)轨迹+σ²_η幅度"),
        "does_not_resolve": ("time-invariance 归因简并: (a)机制改变 (b)critical slowing "
                             "(c)观测算子改变 —— TVP-AR 分不开, 同 Nyblom"),
        "method": ("TVP-AR local-level coefficients; Kalman filter (exact-diffuse init) + RTS smoother; "
                   "concentrated σ²_e; profile MLE of q=σ²_η/σ²_e; parametric-bootstrap null (H0: q=0)"),
        "evidence_grade": "E1 — 漂移轨迹/幅度是统计描述; 归因是 E4-pending",
        "underpowered_expectation": ("N=127 → σ²_η MLE pile-up-at-zero + 低功率; "
                                     "σ̂²_η≈0 不等于 P2 成立"),
        "hf_fork": "只用历史链 x(t) (2015-01~2025-12, H 侧); 不碰前向 cov_trace (F 侧, 2026+)",
        "meta": meta, "n_bootstrap": args.boot,
        "results": results,
        "verdict": {"direction": verdict, "text": summary},
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 → {OUTPUT_PATH}")
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

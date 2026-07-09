"""
Relaxation Probe — 掰开 PC2 非恒定性的归因简并 (2026-07-09, 第九轮判别探针)

Nyblom 发现 PC2 的 AR 参数非恒定 (p=0.0065, 残差已白噪声)。但这对 P2 简并:
  (a) 生成机制改变 / (b) 固定机制工作点移动(逼近分岔) / (c) 观测算子改变。
GPT+Gemini 收敛的判别探针: PC2 的滚动窗自相关 ρ₁(t) 与方差 var(t) 的**领先-滞后**关系。

判别准则:
  - critical-slowing (分岔): ρ 与 var 同步上升且强相关, 且 ρ 往往**领先** var
    (自相关先升, 方差随后放大 —— 同一势阱变平的两个投影)。
  - 机制切换: var 先跳变, ρ 跟随; 或两者**解耦/负相关**。
  - 纯方差漂移 (非 slowing): 只有 var 有趋势, ρ 平。

★诚实边界 (两个 AI 都低估的)★:
  36 月滚动窗在 127 月上只有 ~92 个**高度重叠**的窗口, 有效独立自由度极少 (~3-4)。
  这与时段审计撞的是同一堵分辨率墙。所以本探针**只能给方向, 不能给可信 p 值** ——
  归类为 exploratory/描述性 (E1 描述 + E3 倾向), 不是 confirmatory。领先-滞后的
  cross-correlation 峰值位置也受窗口平滑主导, 看趋势符号即可, 别信小数点。

用法:
  conda run -n MemeticChaos python src/analysis/relaxation_probe.py
"""

import json, sys, argparse
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
STATE_PATH = ROOT / "data/processed/representation_state.json"
OUTPUT_PATH = ROOT / "data/processed/relaxation_probe.json"

WINDOW = 36
TARGET_PC = 1  # PC2 (0-indexed)


def rolling_ar1_and_var(y: np.ndarray, w: int):
    """每个长度 w 的滚动窗内算 lag-1 自相关 ρ₁ 和方差。返回 (中心索引, ρ₁序列, var序列)。"""
    rhos, vars, centers = [], [], []
    for start in range(len(y) - w + 1):
        seg = y[start:start + w]
        v = float(np.var(seg))
        # lag-1 自相关
        s0 = seg - seg.mean()
        r1 = float(np.sum(s0[1:] * s0[:-1]) / (np.sum(s0 ** 2) + 1e-12))
        rhos.append(r1); vars.append(v); centers.append(start + w // 2)
    return np.array(centers), np.array(rhos), np.array(vars)


def trend(series: np.ndarray) -> dict:
    """线性趋势 (对窗中心索引回归), 返回斜率符号 + 与时间相关。"""
    t = np.arange(len(series))
    r = float(np.corrcoef(t, series)[0, 1])
    slope = float(np.polyfit(t, series, 1)[0])
    return {"slope": round(slope, 5), "corr_with_time": round(r, 3),
            "direction": "↑" if slope > 0 else "↓"}


def lead_lag(rho: np.ndarray, var: np.ndarray, max_lag: int = 12) -> dict:
    """ρ 与 var 的领先-滞后互相关。峰值 lag>0 = ρ 领先 var (支持 slowing)。

    诚实: 窗口平滑使序列高度自相关, 峰值位置受平滑主导, 只看方向。
    """
    a = (rho - rho.mean()) / (rho.std() + 1e-12)
    b = (var - var.mean()) / (var.std() + 1e-12)
    n = len(a)
    best_lag, best_c = 0, -2.0
    corrs = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            c = float(np.corrcoef(a[-lag:], b[:n + lag])[0, 1])
        elif lag > 0:
            c = float(np.corrcoef(a[:n - lag], b[lag:])[0, 1])
        else:
            c = float(np.corrcoef(a, b)[0, 1])
        corrs[lag] = round(c, 3)
        if c > best_c:
            best_c, best_lag = c, lag
    return {"contemporaneous_corr": corrs[0], "peak_lag": best_lag,
            "peak_corr": round(best_c, 3),
            "interpretation": ("ρ 领先 var (支持 slowing)" if best_lag > 0 and best_c > 0.5
                               else "var 领先 ρ (偏机制切换)" if best_lag < 0 and best_c > 0.5
                               else "同步" if abs(best_lag) <= 1 and best_c > 0.5
                               else "解耦/弱相关 (不支持 slowing)")}


def main():
    ap = argparse.ArgumentParser(description="Relaxation Probe — PC2 归因判别")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--window", type=int, default=WINDOW)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    x = np.array(state["pca_transformed"]["x_reduced"])
    y = x[:, TARGET_PC]

    print("=" * 66)
    print(f"Relaxation Probe — PC2 归因判别 (窗={args.window}月)")
    print("=" * 66)

    centers, rho, var = rolling_ar1_and_var(y, args.window)
    n_windows = len(centers)
    eff_df = max(1, n_windows // args.window)  # 粗估有效独立窗数

    rho_tr = trend(rho)
    var_tr = trend(var)
    ll = lead_lag(rho, var)

    print(f"\n滚动窗: {n_windows} 个 (高度重叠, 有效独立 ~{eff_df} → exploratory only)")
    print(f"\n  ρ₁(t)  趋势: {rho_tr['direction']} slope={rho_tr['slope']:+.4f} (与时间 r={rho_tr['corr_with_time']:+.2f})")
    print(f"  var(t) 趋势: {var_tr['direction']} slope={var_tr['slope']:+.4f} (与时间 r={var_tr['corr_with_time']:+.2f})")
    print(f"\n  ρ 与 var 同期相关: {ll['contemporaneous_corr']:+.3f}")
    print(f"  领先-滞后峰值: lag={ll['peak_lag']} (>0=ρ领先), corr={ll['peak_corr']:+.3f}")
    print(f"  → {ll['interpretation']}")

    # 判别 (描述性, 非显著性检验)
    both_up = rho_tr["slope"] > 0 and var_tr["slope"] > 0
    strong_sync = ll["contemporaneous_corr"] > 0.5
    if both_up and strong_sync and ll["peak_lag"] >= 0:
        lean = "LEAN_SLOWING"
        note = ("ρ 与 var 同升且同步(ρ不落后) → 倾向 (b) 固定机制逼近分岔/persistence增强。"
                "但★仍不排除★ (a)机制改变 / (c)观测算子改变: 三者可产生相似同步。")
    elif var_tr["slope"] > 0 and rho_tr["slope"] <= 0:
        lean = "LEAN_VARIANCE_ONLY"
        note = "仅 var 升、ρ 平/降 → 不是 slowing, 更像方差机制单独变化。"
    elif ll["contemporaneous_corr"] < 0.3:
        lean = "LEAN_DECOUPLED"
        note = "ρ 与 var 解耦 → 偏 (a) 机制切换 / (c) 观测算子改变, 不支持 slowing。"
    else:
        lean = "INCONCLUSIVE"
        note = "方向不干净, 分辨率不足以判别。"

    print(f"\n{'─'*66}\n判别 (exploratory): {lean}")
    print(f"  {note}")
    print(f"\n  ⚠ 边界: 36月窗在127月上有效自由度~{eff_df}, 与时段审计同一分辨率墙。")
    print(f"     本结果是 E1描述 + E3倾向, 不是 confirmatory。不写进 verdict 当定论。")

    output = {
        "source": "relaxation_probe.py",
        "role": "掰开 PC2 Nyblom 非恒定性的归因简并 (exploratory)",
        "evidence_grade": "E1 描述 + E3 倾向 (非 confirmatory, 分辨率受限)",
        "window": args.window, "n_windows": n_windows, "eff_independent_windows": eff_df,
        "rho_trend": rho_tr, "var_trend": var_tr, "lead_lag": ll,
        "lean": lean, "note": note,
        "caveat": "36月窗高度重叠, 有效自由度极少; 只看趋势方向, 不信 p 值; 三支简并未完全掰开",
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 → {OUTPUT_PATH}")
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

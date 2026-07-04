"""
Regime Transition Map — FR19 v4.1 相变结构检测

不对 x(t) 做点预测。直接检测叙事状态空间中的"相区" (regimes)，
计算转移概率、驻留时间、切换熵。

方法:
  GMM (Gaussian Mixture Model) 对 10 维 x(t) 聚类
  → BIC 选最优相区数
  → 标记每月相区标签
  → 经验转移矩阵 + 驻留时间分布 + 切换统计

这是 GPT/Gemini 双 AI 独立推荐的下一步——"画出 regime transition map"。
比点预测更接近系统的物理本质: 流形切换, 而非轨迹外推。

用法:
    python src/analysis/regime_detector.py
    python src/analysis/regime_detector.py --n-regimes 5
    python src/analysis/regime_detector.py --json
"""

import json, sys, os, argparse
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict

ROOT = Path(__file__).parent.parent.parent

STATE_PATH = ROOT / "data/processed/representation_state.json"
LEVEL1_PATH = ROOT / "data/processed/level1_hard_facts.json"
OUTPUT_PATH = ROOT / "data/processed/regime_map.json"

# Try adding advisor path for live metrics
sys.path.insert(0, str(ROOT / "src"))


def load_data():
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    with open(LEVEL1_PATH, "r", encoding="utf-8") as f:
        l1 = json.load(f)

    months = state["pca_transformed"]["months"]
    x = np.array(state["pca_transformed"]["x_reduced"])  # n × 10
    stage = np.array(l1["stage_occupancy"])               # n × 5
    stage_names = l1["stages"]

    return months, x, stage, stage_names


def fit_regimes(x: np.ndarray, n_regimes: int | None = None) -> tuple[GaussianMixture, np.ndarray, int]:
    """Fit GMM and return model + labels + optimal n_regimes."""
    scaler = StandardScaler()
    x_s = scaler.fit_transform(x)

    if n_regimes is not None:
        gmm = GaussianMixture(n_components=n_regimes, covariance_type="full",
                               random_state=42, n_init=10)
        labels = gmm.fit_predict(x_s)
        return gmm, labels, n_regimes

    # BIC model selection
    best_bic = np.inf
    best_n = 3
    best_gmm = None
    best_labels = None

    results = []
    for k in range(3, 8):
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                               random_state=42, n_init=10)
        labels = gmm.fit_predict(x_s)
        bic = gmm.bic(x_s)
        aic = gmm.aic(x_s)
        results.append({"k": k, "bic": float(bic), "aic": float(aic)})

        if bic < best_bic:
            best_bic = bic
            best_n = k
            best_gmm = gmm
            best_labels = labels

    print(f"  BIC 模型选择: k={best_n} (BIC={best_bic:.1f})")
    for r in results:
        marker = " ★" if r["k"] == best_n else ""
        print(f"    k={r['k']}: BIC={r['bic']:.1f}, AIC={r['aic']:.1f}{marker}")

    return best_gmm, best_labels, best_n


def compute_transition_matrix(labels: np.ndarray, n_regimes: int) -> np.ndarray:
    """经验转移概率矩阵 P(i→j)."""
    trans = np.zeros((n_regimes, n_regimes))
    for t in range(len(labels) - 1):
        i, j = labels[t], labels[t + 1]
        trans[i, j] += 1

    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return trans / row_sums


def compute_dwell_times(labels: np.ndarray, n_regimes: int) -> dict:
    """每个相区的驻留时间分布 (月)."""
    dwell = defaultdict(list)
    current_regime = labels[0]
    current_start = 0

    for t in range(1, len(labels)):
        if labels[t] != current_regime:
            duration = t - current_start
            dwell[int(current_regime)].append(duration)
            current_regime = labels[t]
            current_start = t

    # Last segment
    duration = len(labels) - current_start
    dwell[int(current_regime)].append(duration)

    result = {}
    for r in range(n_regimes):
        d = dwell.get(r, [0])
        result[r] = {
            "count": len(d),
            "mean_months": round(float(np.mean(d)), 1),
            "median_months": round(float(np.median(d)), 1),
            "max_months": int(np.max(d)),
            "min_months": int(np.min(d)),
            "episodes": [int(x) for x in d],
        }
    return result


def compute_switching_stats(labels: np.ndarray, n_regimes: int, months: list[str]) -> dict:
    """切换统计: 频率, 熵, 稳定性指标."""
    n = len(labels)

    # Count switches
    switches = sum(1 for t in range(1, n) if labels[t] != labels[t - 1])
    switch_rate = switches / max(n - 1, 1)

    # Switching entropy: how evenly distributed are transitions across regimes?
    trans_count = Counter()
    for t in range(1, n):
        if labels[t] != labels[t - 1]:
            trans_count[(int(labels[t - 1]), int(labels[t]))] += 1

    # Entropy of target regimes (given a switch happens)
    target_counts = Counter()
    for t in range(1, n):
        if labels[t] != labels[t - 1]:
            target_counts[int(labels[t])] += 1

    total_switches = sum(target_counts.values())
    if total_switches > 0:
        probs = [c / total_switches for c in target_counts.values()]
        switching_entropy = float(-sum(p * np.log(p) for p in probs if p > 0))
    else:
        switching_entropy = 0.0

    # Max entropy for n_regimes: log(n_regimes)
    max_entropy = np.log(n_regimes)
    normalized_entropy = switching_entropy / max_entropy if max_entropy > 0 else 0.0

    # Regime stability: what % of time is spent in the dominant regime?
    regime_counts = Counter(int(l) for l in labels)
    dominant_pct = max(regime_counts.values()) / n

    # Detect "locked" periods: consecutive months in same regime
    locked_streaks = []
    current_len = 1
    for t in range(1, n):
        if labels[t] == labels[t - 1]:
            current_len += 1
        else:
            if current_len >= 6:
                locked_streaks.append({
                    "regime": int(labels[t - 1]),
                    "duration_months": current_len,
                    "start": months[t - current_len],
                    "end": months[t - 1],
                })
            current_len = 1
    if current_len >= 6:
        locked_streaks.append({
            "regime": int(labels[n - 1]),
            "duration_months": current_len,
            "start": months[n - current_len],
            "end": months[n - 1],
        })

    return {
        "total_months": n,
        "total_switches": switches,
        "switch_rate_per_month": round(switch_rate, 4),
        "switching_entropy": round(switching_entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "dominant_regime_pct": round(dominant_pct, 4),
        "locked_periods_6m_plus": locked_streaks,
        "interpretation": (
            "高切换熵 → 系统在多个相区间频繁跳跃 (活跃/混沌期). "
            "低切换熵 + 长锁定 → 系统卡在单一相区 (僵化/亚稳态). "
            f"当前归一化熵={normalized_entropy:.3f}, "
            f"主导相区占比={dominant_pct:.1%}."
        ),
    }


def characterize_regimes(labels: np.ndarray, n_regimes: int,
                          stage: np.ndarray, stage_names: list[str],
                          x: np.ndarray, months: list[str]) -> dict:
    """描述每个相区的特征."""
    chars = {}
    for r in range(n_regimes):
        mask = labels == r
        n_months = int(np.sum(mask))
        mean_stage = stage[mask].mean(axis=0)
        dominant_stage = stage_names[int(np.argmax(mean_stage))]

        # Find representative months
        center = x[mask].mean(axis=0)
        distances = np.linalg.norm(x[mask] - center, axis=1)
        closest_idx = np.where(mask)[0][int(np.argmin(distances))]
        farthest_idx = np.where(mask)[0][int(np.argmax(distances))]

        # Compute 4 indicators for representative months
        stage_profile = {stage_names[i]: round(float(mean_stage[i]), 3)
                         for i in range(len(stage_names))}

        chars[r] = {
            "n_months": n_months,
            "pct_of_total": round(n_months / len(labels) * 100, 1),
            "dominant_stage": dominant_stage,
            "mean_stage_profile": stage_profile,
            "typical_month": months[closest_idx],
            "extreme_month": months[farthest_idx],
        }
    return chars


def main():
    parser = argparse.ArgumentParser(description="Regime Transition Map")
    parser.add_argument("--n-regimes", type=int, default=None,
                        help="Force number of regimes (default: BIC auto-select)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Regime Transition Map — FR19 v4.1")
    print("=" * 60)

    # Load
    print("\n[1/4] 加载数据...")
    months, x, stage, stage_names = load_data()
    print(f"  状态空间: {len(months)} 月 × {x.shape[1]} 维")

    # Fit
    print(f"\n[2/4] GMM 相区聚类{' (k=' + str(args.n_regimes) + ')' if args.n_regimes else ' (BIC 自动)'}...")
    gmm, labels, n_regimes = fit_regimes(x, args.n_regimes)

    # Characterize
    print(f"\n[3/4] 相区特征...")
    chars = characterize_regimes(labels, n_regimes, stage, stage_names, x, months)
    for r, c in chars.items():
        print(f"  Regime {r}: {c['n_months']:>3d} 月 ({c['pct_of_total']:>5.1f}%)  "
              f"主导阶段={c['dominant_stage']:<14s}  典型月={c['typical_month']}")

    # Transitions
    trans_mat = compute_transition_matrix(labels, n_regimes)
    dwell = compute_dwell_times(labels, n_regimes)
    switch_stats = compute_switching_stats(labels, n_regimes, months)

    print(f"\n[4/4] 转移矩阵 + 驻留时间...")

    # Print transition matrix
    print(f"\n  转移概率矩阵 P(i→j):")
    header = "    " + "".join(f"  →R{j}  " for j in range(n_regimes))
    print(header)
    for i in range(n_regimes):
        row = f"    R{i}→"
        for j in range(n_regimes):
            p = trans_mat[i, j]
            if p >= 0.5:
                row += f"  {p:.3f}*"
            elif p > 0:
                row += f"  {p:.3f} "
            else:
                row += "   .    "
        row += f"  (自持={trans_mat[i,i]:.3f})"
        print(row)

    # Print dwell times
    print(f"\n  驻留时间 (月):")
    for r, d in dwell.items():
        print(f"    Regime {r}: mean={d['mean_months']:.1f}, median={d['median_months']:.1f}, "
              f"max={d['max_months']}, 共{d['count']}次")

    # Print switching stats
    s = switch_stats
    print(f"\n  切换统计:")
    print(f"    总切换次数: {s['total_switches']} / {s['total_months']} 月 "
          f"(率={s['switch_rate_per_month']:.3f}/月)")
    print(f"    切换熵: {s['switching_entropy']:.3f} (归一化={s['normalized_entropy']:.3f})")
    print(f"    主导相区占比: {s['dominant_regime_pct']:.1%}")
    print(f"    {s['interpretation']}")

    if s["locked_periods_6m_plus"]:
        print(f"\n  锁定期 (≥6月连续同相区):")
        for lp in s["locked_periods_6m_plus"]:
            print(f"    Regime {lp['regime']}: {lp['start']} → {lp['end']} "
                  f"({lp['duration_months']} 月)")

    # Save
    output = {
        "source": "regime_detector.py — FR19 v4.1",
        "n_regimes": n_regimes,
        "method": "GMM (full covariance)",
        "regime_labels": [int(l) for l in labels],
        "months": months,
        "regime_characteristics": chars,
        "transition_matrix": trans_mat.tolist(),
        "dwell_times": dwell,
        "switching_stats": switch_stats,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  已保存 → {output_path}")

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("完成。下一步: 用 regime_map.json 更新 metrics.py 的 Position 指标。")
    print("=" * 60)


if __name__ == "__main__":
    main()

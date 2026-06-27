"""
吸引子检测模块 — MemeticChaos 建模层。

在集体情感系统的混沌动力学中识别确定性结构。

对齐「微尘哲学」：
- 吸引子 = 集体情感在混沌中倾向于回归的稳定状态
- 奇异吸引子 = 系统在混沌与秩序之间的非周期振荡模式
- 相变点 = 系统从一个吸引子盆地跃迁到另一个的临界条件

方法：
1. 相空间重构 (Takens embedding) — 从一维时间序列重构高维动力学
2. 递归图分析 (Recurrence Plots) — 检测周期性/混沌性/确定性
3. 吸引子盆地检测 — 识别参数空间中的稳定区域
4. 分岔分析 — 检测系统行为在参数变化时的质变

参考文献：
- Takens, F. (1981). Detecting strange attractors in turbulence.
- Webber, C. L. & Zbilut, J. P. (1994). Recurrence quantification analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree
from scipy.signal import argrelextrema
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════

@dataclass
class AttractorResult:
    """吸引子分析结果。"""
    name: str

    # Recurrence quantification
    recurrence_rate: float       # REC — 递归点比例
    determinism: float           # DET — 确定性（对角线结构的比例）
    laminarity: float            # LAM — 层流性（垂直线结构的比例）
    entropy_rqa: float            # ENTR — 递归图的香农熵
    max_diag_length: float       # L_max — 最长对角线（系统可预测性上限）

    # Lyapunov estimation
    lyapunov_est: float          # 估计的最大 Lyapunov 指数（>0 = 混沌）

    # Phase space
    embedding_dim: int           # 最佳嵌入维数
    embedding_delay: int         # 最佳时间延迟

    # Classification
    regime: str                  # "chaotic" | "periodic" | "quasi_periodic" | "fixed_point"
    description: str


# ═══════════════════════════════════════════════
# Phase space reconstruction (Takens embedding)
# ═══════════════════════════════════════════════

def takens_embedding(time_series: np.ndarray,
                     embedding_dim: int = 3,
                     delay: int = 1) -> np.ndarray:
    """Takens 延迟坐标嵌入 — 从一维时间序列重构相空间。

    x(t) → [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]

    Args:
        time_series: 一维时间序列
        embedding_dim: 嵌入维数 m
        delay: 时间延迟 τ

    Returns:
        重构的相空间轨迹: shape (N - (m-1)*τ, m)
    """
    n = len(time_series)
    n_vectors = n - (embedding_dim - 1) * delay
    if n_vectors <= 0:
        raise ValueError(f"Time series too short: {n} < {embedding_dim}*{delay}")

    embedded = np.zeros((n_vectors, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * delay : i * delay + n_vectors]

    return embedded


def estimate_embedding_delay(time_series: np.ndarray,
                              max_delay: int = 50) -> int:
    """使用互信息 (Mutual Information) 的第一个最小值估计最优时间延迟 τ。

    简化版：使用自相关函数的第一个过零点。

    Args:
        time_series: 一维时间序列
        max_delay: 最大延迟搜索范围

    Returns:
        最优延迟 τ
    """
    n = len(time_series)
    ts = time_series - np.mean(time_series)
    ts_std = np.std(ts)

    if ts_std < 1e-10:
        return 1

    for tau in range(1, min(max_delay, n // 4)):
        # Auto-correlation at lag tau
        ac = np.corrcoef(ts[:n - tau], ts[tau:])[0, 1]
        if ac < 1.0 / np.e:  # First crossing of 1/e
            return tau

    return max(1, max_delay // 4)


def estimate_embedding_dim(time_series: np.ndarray,
                            max_dim: int = 10,
                            delay: int = None,
                            threshold: float = 0.05,
                            R_tol: float = 15.0) -> int:
    """标准假近邻法 (False Nearest Neighbors, Kennel et al. 1992) 估计最优嵌入维数。

    对每个候选维数 d，在 d 维空间中寻找最近邻，然后检查当扩展到 d+1 维时
    这些近邻是否变为"假近邻"（距离急剧增大）：
    - 若 d 维中 FNN > threshold → d 维不足，继续尝试 d+1
    - 若 d 维中 FNN < threshold → d 维足够 → 返回 d

    关键改进（相对之前简化版）：
    - 最近邻搜索和 FNN 检查分离：NN 在 d 维找，FNN 在 d+1 维验证
    - 使用 KD-Tree + Theiler window

    Args:
        time_series: 一维时间序列
        max_dim: 最大嵌入维数
        delay: 时间延迟，None则自动估计
        threshold: FNN 比例阈值，低于此值认为维数足够
        R_tol: 距离比值阈值（假近邻判据，Kennel 1992 用 10-15）

    Returns:
        最优嵌入维数 d
    """
    if delay is None:
        delay = estimate_embedding_delay(time_series)

    n = len(time_series)
    theiler = 2 * delay

    # Degenerate case: flat signal → no dynamics, dimension 1
    if np.std(time_series) < 1e-10:
        return 1

    for d in range(1, max_dim + 1):
        # Need at least d*delay + theiler + 10 points for both d and d+1 embeddings
        min_points = max((d + 1) * delay + 10, 3 * delay + 10)
        if n < min_points:
            return max(1, d)

        # Step 1: Embed in d dimensions, find nearest neighbors
        embedded_d = takens_embedding(time_series, d, delay)
        n_d = len(embedded_d)

        # Build (d+1)-dim embedding for the FNN check.
        # embedded_dp1[i] = [x_i, x_{i+τ}, ..., x_{i+dτ}]
        # embedded_d[i]    = [x_i, x_{i+τ}, ..., x_{i+(d-1)τ}]
        embedded_dp1 = takens_embedding(time_series, d + 1, delay)
        n_dp1 = len(embedded_dp1)  # = n_d - delay

        # Only check points that have both d-dim and (d+1)-dim embeddings
        n_check = min(n_d, n_dp1)

        # KD-Tree on d-dim space (restrict to points with (d+1)-dim counterparts)
        tree = KDTree(embedded_d[:n_check])
        k_needed = min(n_check, theiler + 2)
        distances, indices = tree.query(embedded_d[:n_check], k=k_needed)

        # Step 2: Count false neighbors
        fnn_count = 0
        total_pairs = 0

        for i in range(n_check):
            # Find nearest neighbor outside Theiler window
            nn_idx = None
            for j in range(1, k_needed):
                candidate = int(indices[i, j])
                if candidate >= n_check:
                    continue
                if abs(i - candidate) > theiler:
                    nn_idx = candidate
                    break

            if nn_idx is None:
                continue

            # Distance in d-dim space
            dist_d = np.linalg.norm(embedded_d[i] - embedded_d[nn_idx])
            if dist_d < 1e-10:
                continue

            # Extra distance from the (d+1)-th coordinate
            d_extra = abs(embedded_dp1[i, -1] - embedded_dp1[nn_idx, -1])

            # Kennel criterion
            if d_extra / dist_d > R_tol:
                fnn_count += 1
            total_pairs += 1

        if total_pairs == 0:
            # No valid pairs (e.g., all distances too small, or sequence too short)
            # → cannot assess FNN → assume d is sufficient
            return max(1, d)
        if fnn_count / total_pairs < threshold:
            return d

    return max_dim


# ═══════════════════════════════════════════════
# Recurrence Plot Analysis
# ═══════════════════════════════════════════════

def recurrence_matrix(time_series: np.ndarray,
                      embedding_dim: int = None,
                      delay: int = None,
                      threshold: float = None) -> np.ndarray:
    """构建递归矩阵 (Recurrence Plot)。

    R_{ij} = Θ(ε - ||x_i - x_j||)

    Args:
        time_series: 一维时间序列
        embedding_dim: 嵌入维数，None则自动估计
        delay: 延迟，None则自动估计
        threshold: 距离阈值 ε，None则自动选择（保留10%的递归点）

    Returns:
        二值递归矩阵 R (N x N)
    """
    if embedding_dim is None:
        embedding_dim = min(estimate_embedding_dim(time_series), 5)
    if delay is None:
        delay = estimate_embedding_delay(time_series)

    embedded = takens_embedding(time_series, embedding_dim, delay)
    n = len(embedded)

    # Distance matrix
    dists = squareform(pdist(embedded))

    # Automatic threshold selection: keep ~10% recurrence rate
    if threshold is None:
        # Flatten upper triangle (excluding diagonal)
        upper = dists[np.triu_indices(n, k=1)]
        threshold = np.percentile(upper, 10)

    R = (dists <= threshold).astype(float)
    return R


def recurrence_quantification(R: np.ndarray,
                                min_diag: int = 2,
                                min_vert: int = 2) -> dict:
    """递归量化分析 (RQA)。

    从递归矩阵中提取统计量。

    Args:
        R: 递归矩阵 (N x N)
        min_diag: 最小对角线长度（排除噪声）
        min_vert: 最小垂直线长度

    Returns:
        {REC, DET, LAM, ENTR, L_max, L_mean, TT}
    """
    n = R.shape[0]

    # REC: Recurrence Rate
    rec = np.sum(R) / (n * n)

    # Diagonal line analysis (DET, L_max, L_mean, ENTR)
    diag_lengths = []
    for k in range(-n + 1, n):
        diag = np.diag(R, k)
        # Find runs of 1s
        run_length = 0
        for val in diag:
            if val == 1:
                run_length += 1
            else:
                if run_length >= min_diag:
                    diag_lengths.append(run_length)
                run_length = 0
        if run_length >= min_diag:
            diag_lengths.append(run_length)

    if diag_lengths:
        diag_arr = np.array(diag_lengths)
        det = np.sum(diag_arr) / np.sum(R) if np.sum(R) > 0 else 0.0
        l_max = float(np.max(diag_arr))
        l_mean = float(np.mean(diag_arr))
        # Entropy of diagonal length distribution
        hist, _ = np.histogram(diag_arr, bins=min(20, len(diag_arr)))
        hist = hist[hist > 0] / hist.sum()
        entr = float(-np.sum(hist * np.log(hist))) if len(hist) > 1 else 0.0
    else:
        det, l_max, l_mean, entr = 0.0, 0.0, 0.0, 0.0

    # Vertical line analysis (LAM, TT)
    vert_lengths = []
    for j in range(n):
        col = R[:, j]
        run_length = 0
        for val in col:
            if val == 1:
                run_length += 1
            else:
                if run_length >= min_vert:
                    vert_lengths.append(run_length)
                run_length = 0
        if run_length >= min_vert:
            vert_lengths.append(run_length)

    if vert_lengths:
        vert_arr = np.array(vert_lengths)
        lam = np.sum(vert_arr) / np.sum(R) if np.sum(R) > 0 else 0.0
        tt = float(np.mean(vert_arr))
    else:
        lam, tt = 0.0, 0.0

    return {
        "REC": round(rec, 4),
        "DET": round(det, 4),
        "LAM": round(lam, 4),
        "ENTR": round(entr, 4),
        "L_max": round(l_max, 1),
        "L_mean": round(l_mean, 2),
        "TT": round(tt, 2),
    }


# ═══════════════════════════════════════════════
# Lyapunov exponent estimation
# ═══════════════════════════════════════════════

def estimate_lyapunov(time_series: np.ndarray,
                      embedding_dim: int = None,
                      delay: int = None,
                      n_steps: int = 20) -> float:
    """Rosenstein 算法估算最大 Lyapunov 指数。

    正 Lyapunov 指数 → 混沌（初值敏感）
    零 Lyapunov 指数 → 准周期
    负 Lyapunov 指数 → 稳定不动点/极限环

    Args:
        time_series: 一维时间序列
        embedding_dim: 嵌入维数
        delay: 时间延迟
        n_steps: Lyapunov 谱的追踪步数

    Returns:
        估计的最大 Lyapunov 指数
    """
    if embedding_dim is None:
        embedding_dim = min(estimate_embedding_dim(time_series), 5)
    if delay is None:
        delay = estimate_embedding_delay(time_series)

    embedded = takens_embedding(time_series, embedding_dim, delay)
    n = len(embedded)

    # For each point, find nearest neighbor and track divergence
    divergences = np.zeros(n_steps)
    count = 0

    # Sample points to reduce computation
    sample_size = min(n, 100)
    sample_idx = np.linspace(0, n - n_steps - 1, sample_size, dtype=int)

    for i in sample_idx:
        # Find nearest neighbor (with temporal separation)
        dists = np.linalg.norm(embedded[:n - n_steps] - embedded[i], axis=1)
        # Exclude temporal neighbors (within 2*delay)
        exclude_start = max(0, i - 2 * delay)
        exclude_end = min(n - n_steps, i + 2 * delay)
        dists[exclude_start:exclude_end] = np.inf
        nn_idx = np.argmin(dists)

        if np.isinf(dists[nn_idx]):
            continue

        # Track divergence over n_steps
        d0 = np.linalg.norm(embedded[i] - embedded[nn_idx])
        if d0 < 1e-10:
            continue

        for k in range(1, min(n_steps, n - max(i, nn_idx))):
            if i + k >= n or nn_idx + k >= n:
                break
            dk = np.linalg.norm(embedded[i + k] - embedded[nn_idx + k])
            divergences[k] += np.log(dk / d0)

        count += 1

    if count == 0:
        return 0.0

    divergences /= max(count, 1)

    # Fit slope of <log(divergence)> vs time for first n_steps//2
    steps_arr = np.arange(1, n_steps // 2 + 1)
    valid = np.isfinite(divergences[1:n_steps // 2 + 1])
    if np.sum(valid) < 3:
        return 0.0

    slope, _ = np.polyfit(steps_arr[valid], divergences[1:n_steps // 2 + 1][valid], 1)
    return float(slope)


# ═══════════════════════════════════════════════
# Attractor basin detection
# ═══════════════════════════════════════════════

def detect_attractor_basins(param_results: list[dict],
                            stability_threshold: float = 0.05) -> list[dict]:
    """在参数扫描结果中检测吸引子盆地（稳定区域）。

    吸引子盆地 = 系统在不同初始条件下收敛到的同一区域。

    Args:
        param_results: [{params, result}, ...] 参数扫描结果
        stability_threshold: 稳定性判据（相邻结果的差异阈值）

    Returns:
        [{"basin_start": idx, "basin_end": idx, "stability": float, ...}, ...]
    """
    basins = []
    in_basin = False
    basin_start = 0

    for i in range(len(param_results) - 1):
        current = param_results[i]["result"].total_infected
        next_val = param_results[i + 1]["result"].total_infected
        diff = abs(current - next_val)

        if not in_basin and diff < stability_threshold:
            in_basin = True
            basin_start = i
        elif in_basin and diff >= stability_threshold:
            basin_params = param_results[basin_start:i + 1]
            basin_mean = np.mean([r["result"].total_infected for r in basin_params])
            basins.append({
                "basin_start": basin_start,
                "basin_end": i,
                "size": i - basin_start + 1,
                "mean_state": round(float(basin_mean), 4),
                "param_range": (
                    param_results[basin_start]["params"].R0,
                    param_results[i]["params"].R0,
                ),
            })
            in_basin = False

    # Close last basin
    if in_basin:
        basin_params = param_results[basin_start:]
        basin_mean = np.mean([r["result"].total_infected for r in basin_params])
        basins.append({
            "basin_start": basin_start,
            "basin_end": len(param_results) - 1,
            "size": len(param_results) - basin_start,
            "mean_state": round(float(basin_mean), 4),
            "param_range": (
                param_results[basin_start]["params"].R0,
                param_results[-1]["params"].R0,
            ),
        })

    return basins


# ═══════════════════════════════════════════════
# Full attractor analysis pipeline
# ═══════════════════════════════════════════════

def analyze_attractor(time_series: np.ndarray, name: str = "unnamed") -> AttractorResult:
    """对时间序列执行完整的吸引子分析。

    Args:
        time_series: 一维时间序列（如 SIR 的 I(t) 或 混沌位置均值）
        name: 标识名

    Returns:
        AttractorResult
    """
    # Normalize
    ts = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-10)

    # Embedding parameters
    delay = estimate_embedding_delay(ts)
    dim = min(estimate_embedding_dim(ts, delay=delay), 5)

    # Recurrence plot
    R = recurrence_matrix(ts, embedding_dim=dim, delay=delay)
    rqa = recurrence_quantification(R)

    # Lyapunov
    lyap = estimate_lyapunov(ts, embedding_dim=dim, delay=delay)

    # Classify regime
    if lyap > 0.01:
        regime = "chaotic"
        desc = "正 Lyapunov 指数 → 系统处于混沌态，具有初值敏感性"
    elif rqa["DET"] > 0.8:
        regime = "periodic"
        desc = "高确定性 → 系统行为具有强周期性/可预测性"
    elif rqa["DET"] > 0.4:
        regime = "quasi_periodic"
        desc = "中等确定性 → 准周期行为，多个不可公度的频率叠加"
    else:
        regime = "fixed_point"
        desc = "低确定性 + 零 Lyapunov → 系统收敛到不动点或极限环"

    return AttractorResult(
        name=name,
        recurrence_rate=rqa["REC"],
        determinism=rqa["DET"],
        laminarity=rqa["LAM"],
        entropy_rqa=rqa["ENTR"],
        max_diag_length=rqa["L_max"],
        lyapunov_est=round(lyap, 6),
        embedding_dim=dim,
        embedding_delay=delay,
        regime=regime,
        description=desc,
    )


# ═══════════════════════════════════════════════
# Cross-attractor comparison
# ═══════════════════════════════════════════════

def compare_attractors(meme_series: dict[str, np.ndarray]) -> list[AttractorResult]:
    """比较多个热梗时间序列的吸引子特征。

    Args:
        meme_series: {meme_name: time_series}

    Returns:
        AttractorResult 列表
    """
    results = []
    for name, ts in meme_series.items():
        try:
            result = analyze_attractor(ts, name)
            results.append(result)
        except Exception as e:
            print(f"  Warning: Failed to analyze {name}: {e}")

    return results


# ═══════════════════════════════════════════════
# Script entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from src.models.sir_meme import solve_sir, SIRParams, compute_entropy_curve

    print("=" * 60)
    print("MemeticChaos — Attractor Detection Demo")
    print("=" * 60)

    # Generate SIR trajectories for different meme types
    configs = {
        "脉冲型 (R₀=5.3)": SIRParams(beta=0.8, gamma=0.15, N=1.0),
        "爆发型 (R₀=3.75)": SIRParams(beta=0.3, gamma=0.08, N=1.0),
        "长尾型 (R₀=1.5)": SIRParams(beta=0.12, gamma=0.08, N=1.0),
    }

    print("\n--- Attractor Analysis of Meme Types ---")
    for name, params in configs.items():
        result = solve_sir(params)
        H = compute_entropy_curve(result)

        attractor = analyze_attractor(result.I, name)
        entropy_attractor = analyze_attractor(H, f"{name} [Entropy]")

        print(f"\n  {name}:")
        print(f"    I(t) regime: {attractor.regime:15s} | "
              f"DET={attractor.determinism:.3f} | "
              f"λ_max={attractor.lyapunov_est:+.4f} | "
              f"dim={attractor.embedding_dim}")
        print(f"    H(t) regime: {entropy_attractor.regime:15s} | "
              f"DET={entropy_attractor.determinism:.3f} | "
              f"λ_max={entropy_attractor.lyapunov_est:+.4f}")
        print(f"    I(t) → {attractor.description}")
        print(f"    H(t) → {entropy_attractor.description}")

    # Attractor basins from R₀ sweep
    from src.models.sir_meme import sweep_R0
    print("\n--- Attractor Basins (R₀ sweep) ---")
    sweeps = sweep_R0(np.linspace(0.3, 6.0, 40))
    basins = detect_attractor_basins(sweeps)

    if basins:
        for b in basins:
            print(f"  Basin: R₀ ∈ [{b['param_range'][0]:.1f}, {b['param_range'][1]:.1f}] "
                  f"| stable_state={b['mean_state']:.3f} | size={b['size']}")
    else:
        print("  No distinct basins detected (smooth transition)")

    # Cross-comparison with ABM chaos trajectories
    print("\n--- Cross-Attractor Summary ---")
    print("""
    Key insight: Different meme types occupy different regions of the
    attractor landscape:
    - 脉冲型: High chaos → rapid convergence to fixed point (short-lived order)
    - 爆发型: Intermediate chaos → clear attractor structure
    - 长尾型: Low chaos → quasi-periodic or fixed point (sustained order)

    This maps directly to the 微尘哲学 attractor model:
    绝对混沌 ← [脉冲型] ← [爆发型] → [长尾型] → 绝对秩序
    """)

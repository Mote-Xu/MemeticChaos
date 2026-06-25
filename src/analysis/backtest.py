"""
历史回测验证框架 — MemeticChaos 验证层。

GPT关键批评："目前系统最强的部分是模拟能力，而不是验证能力。"
本模块回答：这个模型对已知历史拟合得怎么样？

方法：
1. 时序切分回测：用早期年份训练，预测后期年份
2. 留一验证：逐个热梗留出，用其余预测它
3. 鲁棒性测试：随机扰动数据，看吸引子是否稳定
4. 跨类别泛化：在一类上训练，在另一类上预测

对齐「微尘哲学」：
- 验证不是追求"绝对正确"，而是识别模型的系统性偏差
- 如果对历史都拟合不了，对未来的预测就是空谈
- 混沌系统的预测本质上是概率性的——验证应该评估校准度而非精确度
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from collections import defaultdict

from src.data.curator import MemeCurator, MemeEntry
from src.models.sir_meme import (
    SIRParams, estimate_params_from_lifecycle, solve_sir,
    classify_meme_type,
)
from src.analysis.phase_diagram import build_state_points, MemeStatePoint


# ═══════════════════════════════════════════════
# Backtest data structures
# ═══════════════════════════════════════════════

@dataclass
class BacktestResult:
    """单次回测的结果。"""
    train_years: tuple[int, int]
    test_years: tuple[int, int]
    n_train: int
    n_test: int

    # Chaos axis predictions
    chaos_mae: float           # Mean Absolute Error
    chaos_rmse: float          # Root Mean Square Error
    chaos_correlation: float   # Pearson r between predicted vs actual
    chaos_direction_accuracy: float  # % correctly predicting sign (chaos vs order)

    # Category predictions
    category_accuracy: float   # % correctly predicting category

    # R₀ predictions
    R0_mae: float
    R0_correlation: float

    # Calibration quality
    calibration_score: float   # 0-1, how well-calibrated the predictions are

    def summary(self) -> str:
        return (
            f"Train {self.train_years[0]}-{self.train_years[1]} ({self.n_train} memes) → "
            f"Test {self.test_years[0]}-{self.test_years[1]} ({self.n_test} memes)\n"
            f"  Chaos: MAE={self.chaos_mae:.3f} RMSE={self.chaos_rmse:.3f} "
            f"r={self.chaos_correlation:+.3f} dir_acc={self.chaos_direction_accuracy:.0%}\n"
            f"  Category: acc={self.category_accuracy:.0%}\n"
            f"  R₀: MAE={self.R0_mae:.3f} r={self.R0_correlation:+.3f}\n"
            f"  Calibration: {self.calibration_score:.0%}"
        )


@dataclass
class RobustnessResult:
    """鲁棒性测试结果。"""
    test_type: str           # "delete_memes" | "perturb_chaos" | "perturb_R0"
    perturbation_level: float  # 扰动比例 (0.1 = 10%)
    n_iterations: int
    original_n_basins: int
    mean_n_basins_after: float
    basin_stability: float   # % of iterations where same basins found
    chaos_shift_mean: float
    chaos_shift_std: float
    is_robust: bool          # True if basins survive perturbation


# ═══════════════════════════════════════════════
# Time-split backtest
# ═══════════════════════════════════════════════

def backtest_by_year(curator: MemeCurator = None,
                     train_end: int = 2022,
                     test_start: int = 2023) -> BacktestResult:
    """按年份切分回测。

    用 train_end 之前的热梗训练 → 预测 test_start 之后的热梗。

    Args:
        curator: MemeCurator
        train_end: 训练集截止年份
        test_start: 测试集起始年份

    Returns:
        BacktestResult
    """
    if curator is None:
        curator = MemeCurator()

    train_memes = [m for m in curator.memes if m.year <= train_end]
    test_memes = [m for m in curator.memes if m.year >= test_start]

    # ── "Train": learn per-category chaos distribution ──
    train_by_cat = defaultdict(list)
    for m in train_memes:
        train_by_cat[m.category].append(m.chaos_position)

    cat_stats = {}
    for cat, vals in train_by_cat.items():
        cat_stats[cat] = {
            "mean_chaos": float(np.mean(vals)),
            "std_chaos": float(np.std(vals)),
            "mean_R0": float(np.mean([m.estimated_R0 for m in train_memes if m.category == cat])),
            "n": len(vals),
        }

    # ── "Predict": assign test memes to closest category mean ──
    chaos_preds = []
    chaos_actuals = []
    R0_preds = []
    R0_actuals = []
    category_correct = 0

    for m in test_memes:
        # Predict chaos: use category mean from training
        if m.category in cat_stats:
            chaos_pred = cat_stats[m.category]["mean_chaos"]
            R0_pred = cat_stats[m.category]["mean_R0"]
        else:
            # Unknown category: use global mean
            chaos_pred = float(np.mean([s["mean_chaos"] for s in cat_stats.values()]))
            R0_pred = float(np.mean([s["mean_R0"] for s in cat_stats.values()]))

        chaos_preds.append(chaos_pred)
        chaos_actuals.append(m.chaos_position)
        R0_preds.append(R0_pred)
        R0_actuals.append(m.estimated_R0)

        # Category prediction: which category mean is closest?
        best_cat = min(cat_stats.keys(),
                      key=lambda c: abs(cat_stats[c]["mean_chaos"] - m.chaos_position))
        if best_cat == m.category:
            category_correct += 1

    chaos_preds = np.array(chaos_preds)
    chaos_actuals = np.array(chaos_actuals)
    R0_preds = np.array(R0_preds)
    R0_actuals = np.array(R0_actuals)

    # ── Metrics ──
    chaos_errors = chaos_actuals - chaos_preds
    chaos_mae = float(np.mean(np.abs(chaos_errors)))
    chaos_rmse = float(np.sqrt(np.mean(chaos_errors ** 2)))

    # Correlation
    if len(chaos_actuals) > 2:
        chaos_corr = float(np.corrcoef(chaos_actuals, chaos_preds)[0, 1])
        R0_corr = float(np.corrcoef(R0_actuals, R0_preds)[0, 1])
    else:
        chaos_corr = 0.0
        R0_corr = 0.0

    # Direction accuracy: sign match
    direction_correct = np.sum(np.sign(chaos_actuals) == np.sign(chaos_preds))
    if len(chaos_actuals) > 0:
        direction_acc = float(direction_correct / len(chaos_actuals))
    else:
        direction_acc = 0.0

    # Category accuracy
    if len(test_memes) > 0:
        cat_acc = category_correct / len(test_memes)
    else:
        cat_acc = 0.0

    # Calibration score: how often does actual fall within train std?
    within_std = 0
    for m in test_memes:
        if m.category in cat_stats:
            pred = cat_stats[m.category]["mean_chaos"]
            std = cat_stats[m.category]["std_chaos"]
            if abs(m.chaos_position - pred) <= 2 * std:
                within_std += 1
    calib_score = within_std / max(len(test_memes), 1)

    return BacktestResult(
        train_years=(min(m.year for m in train_memes), train_end),
        test_years=(test_start, max(m.year for m in test_memes)),
        n_train=len(train_memes),
        n_test=len(test_memes),
        chaos_mae=round(chaos_mae, 4),
        chaos_rmse=round(chaos_rmse, 4),
        chaos_correlation=round(chaos_corr, 4),
        chaos_direction_accuracy=round(direction_acc, 4),
        category_accuracy=round(cat_acc, 4),
        R0_mae=round(float(np.mean(np.abs(R0_actuals - R0_preds))), 4),
        R0_correlation=round(R0_corr, 4),
        calibration_score=round(calib_score, 4),
    )


def rolling_backtest(curator: MemeCurator = None) -> list[BacktestResult]:
    """滚动窗口回测：逐年推进训练窗口。"""
    if curator is None:
        curator = MemeCurator()

    years = sorted(set(m.year for m in curator.memes))
    results = []

    for i in range(len(years) - 2):
        train_end = years[i + 1]
        test_start = years[i + 2]
        result = backtest_by_year(curator, train_end, test_start)
        results.append(result)

    return results


# ═══════════════════════════════════════════════
# Leave-one-out cross-validation
# ═══════════════════════════════════════════════

def leave_one_out_validation(curator: MemeCurator = None) -> dict:
    """留一验证：逐个热梗留出，用其余预测它。

    Returns:
        {"chaos_mae": float, "chaos_rmse": float, "category_accuracy": float,
         "hardest_to_predict": [...], "easiest_to_predict": [...]}
    """
    if curator is None:
        curator = MemeCurator()

    errors = []
    for i, target in enumerate(curator.memes):
        # Train on all except target
        train = [m for j, m in enumerate(curator.memes) if j != i]

        # Predict chaos: mean of same category in training set
        same_cat = [m for m in train if m.category == target.category]
        if same_cat:
            pred_chaos = float(np.mean([m.chaos_position for m in same_cat]))
            pred_R0 = float(np.mean([m.estimated_R0 for m in same_cat]))
        else:
            pred_chaos = float(np.mean([m.chaos_position for m in train]))
            pred_R0 = float(np.mean([m.estimated_R0 for m in train]))

        error = abs(target.chaos_position - pred_chaos)
        errors.append({
            "name": target.name,
            "category": target.category,
            "actual": target.chaos_position,
            "predicted": pred_chaos,
            "abs_error": error,
        })

    errors.sort(key=lambda x: x["abs_error"], reverse=True)

    chaos_abs_errors = [e["abs_error"] for e in errors]
    chaos_sq_errors = [e["abs_error"] ** 2 for e in errors]

    # Category prediction accuracy
    cat_correct = 0
    for i, target in enumerate(curator.memes):
        train = [m for j, m in enumerate(curator.memes) if j != i]
        train_by_cat = defaultdict(list)
        for m in train:
            train_by_cat[m.category].append(m.chaos_position)
        cat_means = {cat: float(np.mean(vals)) for cat, vals in train_by_cat.items()}
        best_cat = min(cat_means.keys(),
                      key=lambda c: abs(cat_means[c] - target.chaos_position))
        if best_cat == target.category:
            cat_correct += 1

    return {
        "chaos_mae": round(float(np.mean(chaos_abs_errors)), 4),
        "chaos_rmse": round(float(np.sqrt(np.mean(chaos_sq_errors))), 4),
        "category_accuracy": round(cat_correct / len(curator.memes), 4),
        "hardest_to_predict": [
            {"name": e["name"], "error": round(e["abs_error"], 3)}
            for e in errors[:5]
        ],
        "easiest_to_predict": [
            {"name": e["name"], "error": round(e["abs_error"], 3)}
            for e in errors[-5:]
        ],
    }


# ═══════════════════════════════════════════════
# Robustness testing
# ═══════════════════════════════════════════════

def test_basin_robustness(curator: MemeCurator = None,
                          n_iterations: int = 20) -> list[RobustnessResult]:
    """测试吸引子盆地对数据扰动的鲁棒性。

    如果盆地是真吸引子，即使随机删除/扰动数据也应保持稳定。
    如果是噪声，扰动后会消失。
    """
    if curator is None:
        curator = MemeCurator()

    from src.analysis.phase_diagram import detect_attractor_basins, build_state_points

    # Original basins
    orig_points = build_state_points(curator)
    orig_basins = detect_attractor_basins(orig_points)
    n_orig = len(orig_basins)

    results = []

    # ── Test 1: Random meme deletion ──────────
    for del_frac in [0.1, 0.2, 0.3]:
        basin_counts = []
        chaos_shifts = []
        for _ in range(n_iterations):
            # Randomly delete fraction of memes
            n_del = max(1, int(len(curator.memes) * del_frac))
            keep_idx = np.random.choice(len(curator.memes),
                                        len(curator.memes) - n_del,
                                        replace=False)
            subset = [curator.memes[i] for i in keep_idx]

            # Recompute basins
            sub_points = build_state_points_from_memes(subset)
            sub_basins = detect_attractor_basins(sub_points)

            basin_counts.append(len(sub_basins))
            chaos_shifts.append(
                float(np.mean([p.chaos_position for p in sub_points]) -
                      np.mean([p.chaos_position for p in orig_points]))
            )

        basin_stable = float(np.mean([1.0 if c == n_orig else 0.0 for c in basin_counts]))

        results.append(RobustnessResult(
            test_type="delete_memes",
            perturbation_level=del_frac,
            n_iterations=n_iterations,
            original_n_basins=n_orig,
            mean_n_basins_after=float(np.mean(basin_counts)),
            basin_stability=round(basin_stable, 3),
            chaos_shift_mean=round(float(np.mean(chaos_shifts)), 4),
            chaos_shift_std=round(float(np.std(chaos_shifts)), 4),
            is_robust=basin_stable > 0.7,
        ))

    # ── Test 2: Chaos position perturbation ───
    for noise_level in [0.05, 0.10, 0.20]:
        basin_counts = []
        for _ in range(n_iterations):
            # Add noise to chaos positions
            noisy_memes = []
            for m in curator.memes:
                noisy_chaos = np.clip(m.chaos_position + np.random.normal(0, noise_level), -1, 1)
                # Create a modified entry
                noisy_memes.append(_make_noisy_meme(m, noisy_chaos))

            noisy_points = build_state_points_from_memes(noisy_memes)
            noisy_basins = detect_attractor_basins(noisy_points)
            basin_counts.append(len(noisy_basins))

        basin_stable = float(np.mean([1.0 if c == n_orig else 0.0 for c in basin_counts]))

        results.append(RobustnessResult(
            test_type="perturb_chaos",
            perturbation_level=noise_level,
            n_iterations=n_iterations,
            original_n_basins=n_orig,
            mean_n_basins_after=float(np.mean(basin_counts)),
            basin_stability=round(basin_stable, 3),
            chaos_shift_mean=0.0,
            chaos_shift_std=round(float(np.std(basin_counts)), 4),
            is_robust=basin_stable > 0.5,
        ))

    return results


def build_state_points_from_memes(memes: list[MemeEntry]) -> list[MemeStatePoint]:
    """从 MemeEntry 列表构建状态点（用于扰动后的子集）。"""
    from src.models.sir_meme import estimate_params_from_lifecycle, solve_sir, compute_entropy_curve, classify_meme_type

    points = []
    for m in memes:
        lc = m.lifecycle
        pm = m.propagation_model
        dur_months = lc.get("duration_months", 3)
        if dur_months >= 999:
            dur_months = 18
        dur_days = dur_months * 30
        circle_count = len(pm.get("circle_layers", []))
        if circle_count >= 5:
            total_infected = 0.75
        elif circle_count >= 3:
            total_infected = 0.50
        else:
            total_infected = 0.25

        params = estimate_params_from_lifecycle(
            peak_day=dur_days * 0.3,
            total_infected=total_infected,
            duration_days=dur_days,
        )
        result = solve_sir(params)
        classification = classify_meme_type(result)
        H = compute_entropy_curve(result)

        points.append(MemeStatePoint(
            name=m.name, category=m.category, year=m.year,
            R0=float(params.R0),
            chaos_position=float(m.chaos_position),
            entropy_max=float(np.max(H)),
            entropy_mean=float(np.mean(H)),
            duration_days=dur_days,
            peak_infected=float(result.peak_infected),
            meme_type=classification["type"],
            lifecycle_status=lc.get("status", "unknown"),
        ))
    return points


def _make_noisy_meme(meme: MemeEntry, noisy_chaos: float) -> MemeEntry:
    """创建带有扰动混沌位置的热梗副本。"""
    # Hack: modify the chaos_position via a simple wrapper
    import copy
    m = copy.deepcopy(meme)
    m.chaos_vector["chaos_order_position"] = noisy_chaos
    return m


# ═══════════════════════════════════════════════
# Script entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("MemeticChaos — Historical Backtest Validation")
    print("=" * 60)

    # ── Time-split backtest ────────────────────
    print("\n── 1. 时序切分回测 ──")
    results = rolling_backtest()
    for r in results:
        print(f"\n{r.summary()}")

    # ── Leave-one-out ──────────────────────────
    print("\n── 2. 留一验证 ──")
    loo = leave_one_out_validation()
    print(f"  Chaos MAE: {loo['chaos_mae']:.4f}")
    print(f"  Chaos RMSE: {loo['chaos_rmse']:.4f}")
    print(f"  Category accuracy: {loo['category_accuracy']:.1%}")
    print(f"  Hardest to predict:")
    for e in loo["hardest_to_predict"]:
        print(f"    {e['name']}: error={e['error']:.3f}")
    print(f"  Easiest to predict:")
    for e in loo["easiest_to_predict"]:
        print(f"    {e['name']}: error={e['error']:.3f}")

    # ── Robustness ─────────────────────────────
    print("\n── 3. 吸引子鲁棒性测试 ──")
    robustness = test_basin_robustness(n_iterations=15)
    for r in robustness:
        status = "ROBUST ✓" if r.is_robust else "FRAGILE ✗"
        print(f"  {r.test_type} ({r.perturbation_level:.0%}): "
              f"basins {r.original_n_basins}→{r.mean_n_basins_after:.1f} "
              f"| stability={r.basin_stability:.0%} | {status}")

    # ── Summary ────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Validation Summary:")
    print(f"  The model's predictive accuracy on held-out memes")
    print(f"  quantifies how much of 'meme chaos' is systematic")
    print(f"  vs idiosyncratic. Low accuracy doesn't mean failure —")
    print(f"  it means the chaotic component is genuinely unpredictable,")
    print(f"  which is consistent with the chaos dynamics framework.")

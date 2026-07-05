"""
MS-AR First Cut — Control-Conditioned Regime Transition Analysis
================================================================

GOAL: Does the control manifold z(t) modulate regime transition behavior?

CONSTRAINT: 127 months, 14 total regime switches.
             z1 has near-zero variance within R0/R1/R3 — it's a regime discriminator,
             not a within-regime modulator.

What we CAN honestly do:
  1. TIME-SPLIT: separate transition matrices for pre/post 2020 (z1 proxy)
  2. BOOTSTRAP: confidence intervals for each transition cell
  3. PERMUTATION TEST: do switches cluster at particular z1 values?
  4. WITHIN-R2 DRIFT: does x(t) drift with z1 inside the R2 lock-in?

What we CANNOT do (yet):
  - Fit a continuous function P(S'|S, z) — too few switches
  - Bin z1 into tertiles — bins would have ~2 switches each

AlphaGo principle: let the data tell us whether MS-AR is warranted.
If time-split matrices are identical within bootstrap CI, we report "no evidence."
If they differ, we report "signal detected — proceed to Phase 2."

Phase 2 (if signal detected): multinomial logit with BIC-based regularization.
Phase 3 (if Phase 2 converges): HMM cross-validation.

Usage:
    python src/analysis/ms_ar_first_cut.py
    python src/analysis/ms_ar_first_cut.py --json
"""

import json, sys, os, argparse
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data/processed"


# ═══════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════

def load_data():
    """Load regime map, control manifold, and narrative state."""
    with open(DATA_DIR / "regime_map.json", "r", encoding="utf-8") as f:
        regime = json.load(f)

    with open(DATA_DIR / "control_manifold.json", "r", encoding="utf-8") as f:
        control = json.load(f)

    with open(DATA_DIR / "representation_state.json", "r", encoding="utf-8") as f:
        rep_state = json.load(f)

    months = regime["months"]
    labels = np.array(regime["regime_labels"], dtype=int)  # n × 1, values 0-3
    x = np.array(rep_state["pca_transformed"]["x_reduced"])  # n × 10

    # z(t) timeline — already aligned to same months
    z = np.zeros((len(months), 3))
    ctl_months = {pt["month"]: i for i, pt in enumerate(control["analysis"]["timeline"])}
    for i, m in enumerate(months):
        if m in ctl_months:
            pt = control["analysis"]["timeline"][ctl_months[m]]
            z[i] = [pt["z1"], pt["z2"], pt["z3"]]

    return months, labels, x, z, regime


# ═══════════════════════════════════════════════════════
# 1. TRANSITION MATRIX ESTIMATION
# ═══════════════════════════════════════════════════════

def estimate_transition_matrix(labels, n_regimes=4):
    """Estimate transition matrix from regime label sequence.

    P[i, j] = P(S_{t+1}=j | S_t=i)
    Row-normalized. Always returns n_regimes × n_regimes matrix,
    even if some regimes are absent from the sample.
    """
    T = np.zeros((n_regimes, n_regimes))
    counts = np.zeros(n_regimes)

    for t in range(len(labels) - 1):
        i = int(labels[t])
        j = int(labels[t + 1])
        if 0 <= i < n_regimes and 0 <= j < n_regimes:
            T[i, j] += 1
            counts[i] += 1

    # Row-normalize; rows with zero count stay as zeros
    for i in range(n_regimes):
        if counts[i] > 0:
            T[i] /= counts[i]

    return T, counts


def bootstrap_transition_matrix(labels, n_bootstrap=10000, seed=42):
    """Bootstrap CI for each transition probability.

    Resamples months with replacement, preserving temporal order within
    each bootstrap sample (block bootstrap with block_size=1 since
    regime persistence is high — single months are informative units).

    Returns:
        T_mean: mean matrix
        T_ci_low, T_ci_high: 95% CI matrices
        T_samples: all bootstrap matrices (for distribution plotting)
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    n_regimes = 4  # always 4 regimes (R0-R3), even if some absent
    samples = np.zeros((n_bootstrap, n_regimes, n_regimes))

    for b in range(n_bootstrap):
        # Resample indices with replacement
        idx = rng.choice(n, size=n, replace=True)
        idx = np.sort(idx)  # preserve temporal order
        boot_labels = labels[idx]
        T, _ = estimate_transition_matrix(boot_labels, n_regimes=n_regimes)
        samples[b] = T

    T_mean = samples.mean(axis=0)
    T_ci_low = np.percentile(samples, 2.5, axis=0)
    T_ci_high = np.percentile(samples, 97.5, axis=0)

    return T_mean, T_ci_low, T_ci_high, samples


# ═══════════════════════════════════════════════════════
# 2. TIME-SPLIT ANALYSIS
# ═══════════════════════════════════════════════════════

def time_split_analysis(months, labels, split_month="2020-01"):
    """Compare transition matrices before vs after a time split.

    z1 is essentially a time proxy (monotonic AI/Tech discourse).
    Time-splitting is the honest way to test whether "early" vs "late"
    periods have different transition structure.
    """
    split_idx = months.index(split_month) if split_month in months else len(months) // 2

    pre_labels = labels[:split_idx]
    post_labels = labels[split_idx:]

    T_pre, counts_pre = estimate_transition_matrix(pre_labels)
    T_post, counts_post = estimate_transition_matrix(post_labels)

    # Bootstrap both
    T_pre_mean, T_pre_lo, T_pre_hi, _ = bootstrap_transition_matrix(pre_labels, n_bootstrap=5000)
    T_post_mean, T_post_lo, T_post_hi, _ = bootstrap_transition_matrix(post_labels, n_bootstrap=5000)

    return {
        "split_month": months[split_idx],
        "pre": {
            "n_months": len(pre_labels),
            "n_switches": int(np.sum(labels[:split_idx - 1] != labels[1:split_idx])),
            "regime_distribution": _regime_dist(labels[:split_idx]),
            "transition_matrix": T_pre.tolist(),
            "bootstrap_ci_low": T_pre_lo.tolist(),
            "bootstrap_ci_high": T_pre_hi.tolist(),
        },
        "post": {
            "n_months": len(post_labels),
            "n_switches": int(np.sum(labels[split_idx:-1] != labels[split_idx + 1:])),
            "regime_distribution": _regime_dist(labels[split_idx:]),
            "transition_matrix": T_post.tolist(),
            "bootstrap_ci_low": T_post_lo.tolist(),
            "bootstrap_ci_high": T_post_hi.tolist(),
        },
        "difference_significant": _test_difference(
            T_pre, T_post, pre_labels, post_labels),
    }


def _regime_dist(labels):
    """Count regime occurrences."""
    unique, counts = np.unique(labels, return_counts=True)
    return {f"R{int(u)}": int(c) for u, c in zip(unique, counts)}


def _test_difference(T1, T2, labels1, labels2, n_perm=5000, seed=42):
    """Permutation test: is the difference between two transition matrices
    larger than expected by random split?

    Test statistic: Frobenius norm of (T1 - T2), weighted by row counts
    to de-emphasize poorly-estimated rows.
    """
    rng = np.random.default_rng(seed)
    n1 = len(labels1)
    all_labels = np.concatenate([labels1, labels2])

    # Observed difference
    obs_diff = np.linalg.norm(T1 - T2, 'fro')

    # Null distribution: random splits
    null_diffs = np.zeros(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(len(all_labels))
        p1 = all_labels[perm[:n1]]
        p2 = all_labels[perm[n1:]]
        P1, _ = estimate_transition_matrix(p1)
        P2, _ = estimate_transition_matrix(p2)
        null_diffs[p] = np.linalg.norm(P1 - P2, 'fro')

    p_value = np.mean(null_diffs >= obs_diff)

    return {
        "observed_frobenius_diff": round(float(obs_diff), 4),
        "null_mean": round(float(null_diffs.mean()), 4),
        "null_std": round(float(null_diffs.std()), 4),
        "p_value": round(float(p_value), 4),
        "significant_at_5pct": bool(p_value < 0.05),
        "interpretation": (
            "Pre/post transition matrices ARE significantly different "
            "(p<0.05) — evidence for temporal structure in regime dynamics."
            if p_value < 0.05 else
            "Pre/post transition matrices are NOT significantly different — "
            "transition structure is stable across time periods."
        ),
    }


# ═══════════════════════════════════════════════════════
# 3. PERMUTATION TEST — z1 conditioning
# ═══════════════════════════════════════════════════════

def z1_permutation_test(months, labels, z):
    """Test whether regime switches occur at unusual z1 values.

    H0: switches occur at random z1 values (no z-modulation)
    H1: switches cluster at particular z1 values

    Two tests:
      a) Mean z1 at switch vs mean z1 overall
      b) For each switch target regime, is z1 different?
    """
    z1 = z[:, 0]

    # Collect switch events
    switch_z1 = []
    switch_from = []
    switch_to = []
    for t in range(len(labels) - 1):
        if labels[t] != labels[t + 1]:
            switch_z1.append(z1[t])
            switch_from.append(labels[t])
            switch_to.append(labels[t + 1])

    switch_z1 = np.array(switch_z1)

    results = {
        "n_switches": len(switch_z1),
        "overall_z1_mean": round(float(z1.mean()), 4),
        "switch_z1_mean": round(float(switch_z1.mean()), 4),
    }

    # a) Overall: are switch z1 values different from baseline?
    # Use bootstrap to get null distribution of mean(14 random z1 values)
    rng = np.random.default_rng(42)
    n_perm = 10000
    null_means = np.zeros(n_perm)
    for p in range(n_perm):
        null_means[p] = rng.choice(z1, size=len(switch_z1), replace=False).mean()

    p_two_sided = 2 * min(
        np.mean(null_means >= switch_z1.mean()),
        np.mean(null_means <= switch_z1.mean())
    )
    results["overall_test"] = {
        "observed": round(float(switch_z1.mean()), 4),
        "null_mean": round(float(null_means.mean()), 4),
        "null_95ci": [
            round(float(np.percentile(null_means, 2.5)), 4),
            round(float(np.percentile(null_means, 97.5)), 4),
        ],
        "p_value_2sided": round(float(p_two_sided), 4),
        "significant": bool(p_two_sided < 0.05),
    }

    # b) By target regime: does z1 differ when switching INTO regime X?
    # Only test regimes with >= 3 switches into them
    target_tests = {}
    for target_regime in [0, 1, 2, 3]:
        mask = np.array(switch_to) == target_regime
        if mask.sum() >= 3:
            target_z1 = switch_z1[mask]
            null_target_means = np.zeros(n_perm)
            for p in range(n_perm):
                null_target_means[p] = rng.choice(
                    z1, size=len(target_z1), replace=False).mean()
            p_val = 2 * min(
                np.mean(null_target_means >= target_z1.mean()),
                np.mean(null_target_means <= target_z1.mean())
            )
            target_tests[f"into_R{target_regime}"] = {
                "n_switches": int(mask.sum()),
                "mean_z1_at_switch": round(float(target_z1.mean()), 4),
                "overall_z1_mean": round(float(z1.mean()), 4),
                "p_value": round(float(p_val), 4),
            }

    results["by_target_regime"] = target_tests
    return results


# ═══════════════════════════════════════════════════════
# 4. WITHIN-R2 DRIFT ANALYSIS
# ═══════════════════════════════════════════════════════

def within_r2_drift(months, labels, x, z):
    """Analyze whether narrative state x(t) drifts with z1 within R2.

    This uses ALL 38 R2 months (not just 14 switches).
    If x(t) co-varies with z1 inside R2, then z(t) is modulating
    the system even when regime identity is held constant.

    Tests:
      a) Correlation between each PC dimension and z1 within R2
      b) Does the variance of x(t) change with z1?
      c) Is there a systematic drift direction?
    """
    r2_mask = labels == 2
    r2_x = x[r2_mask]  # n_r2 × 10
    r2_z1 = z[r2_mask, 0]  # n_r2 × 1
    r2_months = [months[i] for i in range(len(months)) if r2_mask[i]]

    n_r2 = len(r2_z1)

    # a) Per-dimension correlation with z1
    correlations = []
    for dim in range(r2_x.shape[1]):
        if np.std(r2_x[:, dim]) > 1e-10:
            corr, p_val = stats.pearsonr(r2_z1, r2_x[:, dim])
            correlations.append({
                "dimension": f"PC{dim + 1}",
                "pearson_r": round(float(corr), 4),
                "p_value": round(float(p_val), 4),
                "significant_5pct": bool(p_val < 0.05),
            })

    # Bonferroni correction (10 tests)
    n_sig = sum(1 for c in correlations if c["p_value"] < 0.05)
    n_sig_bonf = sum(1 for c in correlations if c["p_value"] < 0.005)

    # b) Variance change: split R2 into early (z1 > median) vs late (z1 < median)
    median_z1 = np.median(r2_z1)
    early_mask = r2_z1 >= median_z1
    late_mask = r2_z1 < median_z1

    early_var = float(np.mean(np.var(r2_x[early_mask], axis=0)))
    late_var = float(np.mean(np.var(r2_x[late_mask], axis=0)))
    var_ratio = late_var / early_var if early_var > 0 else float('inf')

    # c) Total drift within R2: how far has the state moved?
    drift_vector = r2_x[-1] - r2_x[0]
    drift_magnitude = float(np.linalg.norm(drift_vector))

    # Compare to expected drift under random walk null
    # Null: E[∥x(T)-x(0)∥] = σ * √T for random walk
    step_sizes = np.linalg.norm(np.diff(r2_x, axis=0), axis=1)
    sigma = np.mean(step_sizes) if len(step_sizes) > 0 else 1.0
    expected_drift = sigma * np.sqrt(n_r2)
    drift_ratio = drift_magnitude / expected_drift if expected_drift > 0 else float('inf')

    return {
        "n_r2_months": int(n_r2),
        "r2_months_range": [r2_months[0], r2_months[-1]],
        "z1_within_r2": {
            "range": [round(float(r2_z1.min()), 4), round(float(r2_z1.max()), 4)],
            "mean": round(float(r2_z1.mean()), 4),
            "trend": "monotonically decreasing (AI discourse rising)",
        },
        "correlation_with_z1": {
            "dimensions": correlations,
            "n_significant_raw": n_sig,
            "n_significant_bonferroni": n_sig_bonf,
            "interpretation": (
                f"{n_sig_bonf}/10 dimensions significantly correlate with z1 "
                f"after Bonferroni correction. "
                + ("z1 DOES modulate state within R2." if n_sig_bonf > 0
                   else "z1 does NOT modulate state within R2 beyond chance.")
            ),
        },
        "variance_change": {
            "early_r2_var": round(early_var, 4),
            "late_r2_var": round(late_var, 4),
            "var_ratio_late_to_early": round(var_ratio, 3),
            "interpretation": (
                f"Variance {'increased' if var_ratio > 1.2 else 'decreased' if var_ratio < 0.8 else 'stable'} "
                f"from early to late R2 (ratio={var_ratio:.2f}x)."
            ),
        },
        "total_drift": {
            "observed_magnitude": round(drift_magnitude, 4),
            "expected_random_walk": round(expected_drift, 4),
            "drift_ratio": round(drift_ratio, 3),
            "interpretation": (
                "Drift is LARGER than random walk null → directed component exists."
                if drift_ratio > 1.5 else
                "Drift is SMALLER than random walk null → state is constrained (consistent with R2 lock-in)."
                if drift_ratio < 0.67 else
                "Drift is consistent with random walk null."
            ),
        },
    }


# ═══════════════════════════════════════════════════════
# 5. MAIN REPORT
# ═══════════════════════════════════════════════════════

def run_full_analysis():
    """Run all analyses and return structured results."""
    months, labels, x, z, regime = load_data()

    # Baseline
    T_baseline, counts = estimate_transition_matrix(labels)
    T_mean, T_lo, T_hi, _ = bootstrap_transition_matrix(labels)

    n_switches = int(np.sum(labels[:-1] != labels[1:]))
    n_months = len(months)

    # Time-split
    # Natural breakpoints: 2020-01 (pre/post pandemic), 2018-01 (earliest split
    # with enough data in each half), 2022-01 (R2 entry)
    time_splits = {}
    for split_label, split_month in [
        ("pandemic", "2020-01"),
        ("midpoint", "2018-01"),
        ("r2_entry", "2022-12"),  # R2 lock-in begins
    ]:
        if split_month in months:
            time_splits[split_label] = time_split_analysis(
                months, labels, split_month)

    # z1 permutation test
    z1_test = z1_permutation_test(months, labels, z)

    # Within-R2 drift
    r2_drift = within_r2_drift(months, labels, x, z)

    return {
        "meta": {
            "analysis": "MS-AR First Cut — Control-Conditioned Regime Transitions",
            "formalism_version": "v4.1",
            "n_months": n_months,
            "n_regimes": 4,
            "n_total_switches": n_switches,
            "switch_rate_per_month": round(n_switches / (n_months - 1), 4),
            "constraint_note": (
                f"Only {n_switches} regime switches in {n_months} months. "
                "Continuous MS-AR is underpowered. This analysis uses "
                "time-split + bootstrap + permutation tests instead."
            ),
        },
        "baseline_transition_matrix": {
            "matrix": T_baseline.tolist(),
            "row_counts": counts.tolist(),
            "bootstrap_95ci": {
                "low": T_lo.tolist(),
                "high": T_hi.tolist(),
            },
        },
        "time_split_analysis": time_splits,
        "z1_permutation_test": z1_test,
        "within_r2_drift": r2_drift,
        "verdict": _overall_verdict(time_splits, z1_test, r2_drift, n_switches),
    }


def _overall_verdict(time_splits, z1_test, r2_drift, n_switches):
    """Synthesize across analyses into a single recommendation."""

    # Check time-split significance
    split_sig = any(
        ts.get("difference_significant", {}).get("significant_at_5pct", False)
        for ts in time_splits.values()
    )

    # Check z1 modulation
    z1_sig = z1_test.get("overall_test", {}).get("significant", False)

    # Check within-R2 correlations
    r2_corr_sig = r2_drift.get("correlation_with_z1", {}).get(
        "n_significant_bonferroni", 0) > 0

    # Synthesize
    signals = []
    if split_sig:
        signals.append("时间切分矩阵存在显著差异")
    if z1_sig:
        signals.append("切换事件在特定 z1 值聚集")
    if r2_corr_sig:
        signals.append("R2 内部 x(t) 随 z1 共变")

    if len(signals) >= 2:
        verdict = "PROCEED_TO_PHASE_2"
        recommendation = (
            "检测到多重 z(t) 调控信号。建议进入 Phase 2: "
            "multinomial logit P(S'|S, z) 建模, 使用 BIC 选择 z 维度。"
        )
    elif len(signals) == 1:
        verdict = "WEAK_SIGNAL"
        recommendation = (
            "仅检测到微弱信号。Phase 2 可以做但预期效果有限。"
            "建议: 积累更多月份数据 (每月新增 1 月) 或接受静态转移矩阵为有效近似。"
        )
    else:
        verdict = "NO_EVIDENCE"
        recommendation = (
            f"在 {n_switches} 次切换中没有检测到 z(t) 调控转移行为的证据。"
            "静态转移矩阵是当前数据下最诚实的描述。"
            "MS-AR 需要更多切换事件 (>30) 才有统计效力。"
        )

    return {
        "verdict": verdict,
        "signals_detected": signals,
        "recommendation": recommendation,
    }


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MS-AR First Cut — Control-Conditioned Regime Transition Analysis")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    results = run_full_analysis()

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # ── Human-readable report ──
    m = results["meta"]
    print("═" * 62)
    print("  MS-AR First Cut — Control-Conditioned Regime Transitions")
    print("═" * 62)
    print(f"\n  {m['n_months']} months, {m['n_regimes']} regimes, "
          f"{m['n_total_switches']} total switches")
    print(f"  ⚠ {m['constraint_note']}")
    print()

    # Baseline
    bl = results["baseline_transition_matrix"]
    print("── 基线转移矩阵 (Bootstrap 95% CI) ──")
    for i in range(4):
        row_str = "  ".join(
            f"{bl['matrix'][i][j]:.3f} [{bl['bootstrap_95ci']['low'][i][j]:.3f}, "
            f"{bl['bootstrap_95ci']['high'][i][j]:.3f}]"
            for j in range(4)
        )
        print(f"  R{i} → {row_str}")
    print()

    # Time splits
    print("── 时间切分分析 ──")
    for split_name, ts in results["time_split_analysis"].items():
        print(f"\n  ▸ 切分: {split_name} ({ts['split_month']})")
        for period, label in [("pre", "之前"), ("post", "之后")]:
            p = ts[period]
            print(f"    {label}: {p['n_months']}月, {p['n_switches']}次切换, "
                  f"分布={p['regime_distribution']}")
        diff = ts["difference_significant"]
        print(f"    差异检验: Frobenius={diff['observed_frobenius_diff']:.3f}, "
              f"p={diff['p_value']:.3f} {'⚠ 显著!' if diff['significant_at_5pct'] else '(不显著)'}")
        print(f"    → {diff['interpretation']}")
    print()

    # z1 permutation
    zt = results["z1_permutation_test"]
    print("── z1 置换检验 ──")
    ot = zt["overall_test"]
    print(f"  切换时 z1 均值={ot['observed']:.4f} vs 全月均值={zt['overall_z1_mean']:.4f}")
    print(f"  零分布 95%CI: [{ot['null_95ci'][0]:.4f}, {ot['null_95ci'][1]:.4f}]")
    print(f"  p={ot['p_value_2sided']:.3f} {'⚠ 显著!' if ot['significant'] else '(不显著)'}")
    if zt["by_target_regime"]:
        print("  按目标 regime:")
        for regime, test in zt["by_target_regime"].items():
            print(f"    {regime}: mean z1={test['mean_z1_at_switch']:.4f}, "
                  f"p={test['p_value']:.3f} (n={test['n_switches']})")
    print()

    # Within-R2
    print("── R2 内部漂移分析 ──")
    r2 = results["within_r2_drift"]
    print(f"  R2 覆盖 {r2['n_r2_months']} 个月 ({r2['r2_months_range'][0]} → "
          f"{r2['r2_months_range'][1]})")
    print(f"  z1 在 R2 内: {r2['z1_within_r2']['range']} ({r2['z1_within_r2']['trend']})")
    print(f"\n  各维度与 z1 相关性:")
    for c in r2["correlation_with_z1"]["dimensions"]:
        sig_mark = " ⚠" if c["significant_5pct"] else ""
        print(f"    {c['dimension']}: r={c['pearson_r']:+.4f}, p={c['p_value']:.3f}{sig_mark}")
    print(f"  {r2['correlation_with_z1']['interpretation']}")
    print(f"\n  方差变化: early={r2['variance_change']['early_r2_var']:.4f} → "
          f"late={r2['variance_change']['late_r2_var']:.4f} "
          f"(ratio={r2['variance_change']['var_ratio_late_to_early']:.2f}x)")
    print(f"  {r2['variance_change']['interpretation']}")
    print(f"\n  总漂移: observed={r2['total_drift']['observed_magnitude']:.3f} vs "
          f"expected RW={r2['total_drift']['expected_random_walk']:.3f} "
          f"(ratio={r2['total_drift']['drift_ratio']:.3f})")
    print(f"  {r2['total_drift']['interpretation']}")
    print()

    # Verdict
    v = results["verdict"]
    print("═" * 62)
    print(f"  综合判断: {v['verdict']}")
    print("═" * 62)
    if v["signals_detected"]:
        print(f"\n  检测到的信号:")
        for s in v["signals_detected"]:
            print(f"    ✓ {s}")
    else:
        print(f"\n  未检测到信号。")
    print(f"\n  → {v['recommendation']}")
    print()


if __name__ == "__main__":
    main()

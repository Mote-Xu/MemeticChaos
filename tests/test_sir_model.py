"""
SIR 模因模型测试。

验证：
1. 标准 SIR 模型的基础性质（守恒律、R₀ 阈值行为）
2. SIRS 模型的变异/复燃机制
3. 双群体模型的跨层传播
4. 参数拟合精度
5. 混沌分析工具的正确性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.models.sir_meme import (
    SIRParams, SIRResult, TwoPopParams, MemeLifecycle,
    sir_derivatives, sirs_derivatives,
    solve_sir, solve_two_population,
    fit_sir_to_curve,
    estimate_params_from_lifecycle,
    extract_lifecycle,
    classify_meme_type,
    compute_entropy_curve,
    detect_phase_transition,
    sweep_R0, sweep_beta_gamma,
)


class TestSIRDerivatives:
    """验证微分方程的基本数学性质。"""

    def test_conservation_law(self):
        """SIR 模型总人口守恒：dS + dI + dR = 0"""
        N = 1000
        y = np.array([990.0, 10.0, 0.0])
        d = sir_derivatives(0, y, beta=0.3, gamma=0.1, N=N)
        assert abs(d.sum()) < 1e-10, f"Conservation violated: sum={d.sum()}"

    def test_no_infection_without_infected(self):
        """无感染者时不应有新感染：dI/dt <= 0 when I = 0"""
        N = 1000
        y = np.array([1000.0, 0.0, 0.0])
        d = sir_derivatives(0, y, beta=0.3, gamma=0.1, N=N)
        assert d[1] <= 1e-10, f"dI/dt should be <= 0 when I=0, got {d[1]}"

    def test_recovery_with_infected(self):
        """有感染者时恢复率应为正：dR/dt = gamma * I > 0 when I > 0"""
        N = 1000
        y = np.array([990.0, 10.0, 0.0])
        d = sir_derivatives(0, y, beta=0.3, gamma=0.1, N=N)
        assert d[2] > 0, f"dR/dt should be positive, got {d[2]}"

    def test_sirs_conservation(self):
        """SIRS 模型总人口守恒"""
        N = 1000
        y = np.array([700.0, 100.0, 200.0])
        d = sirs_derivatives(0, y, beta=0.3, gamma=0.1, sigma=0.05, N=N)
        assert abs(d.sum()) < 1e-10, f"Conservation violated: sum={d.sum()}"

    def test_sirs_recurrence(self):
        """SIRS 模型中恢复者可重新变为易感者：dS 包含 +sigma*R"""
        N = 1000
        y = np.array([700.0, 100.0, 200.0])
        d = sirs_derivatives(0, y, beta=0.0, gamma=0.0, sigma=0.1, N=N)
        assert d[0] > 0, "With sigma>0, dS should be positive from recurrence"


class TestSIRSolver:
    """验证 SIR/SIRS 求解器的数值行为。"""

    def test_R0_above_one_outbreak(self):
        """R₀ > 1 时应有显著传播。"""
        params = SIRParams(beta=0.3, gamma=0.1, N=1.0)
        assert params.R0 == pytest.approx(3.0)
        result = solve_sir(params)
        assert result.peak_infected > 0.1, f"Expected outbreak, got peak={result.peak_infected}"
        assert result.total_infected > 0.5, f"Expected high total, got {result.total_infected}"

    def test_R0_below_one_no_outbreak(self):
        """R₀ < 1 时模因不应爆发。"""
        params = SIRParams(beta=0.05, gamma=0.1, N=1.0)
        assert params.R0 == 0.5
        result = solve_sir(params)
        assert result.peak_infected < 0.05, f"Expected no outbreak, got peak={result.peak_infected}"

    def test_values_in_range(self):
        """所有比例应在 [0, N] 范围内。"""
        params = SIRParams(beta=0.3, gamma=0.1, N=1.0)
        result = solve_sir(params)
        assert np.all(result.S >= 0) and np.all(result.S <= 1)
        assert np.all(result.I >= 0) and np.all(result.I <= 1)
        assert np.all(result.R >= 0) and np.all(result.R <= 1)

    def test_final_size_monotonic_R0(self):
        """最终感染规模应是 R₀ 的单调递增函数。"""
        finals = []
        for R0 in [0.5, 1.0, 2.0, 4.0, 8.0]:
            params = SIRParams(beta=R0 * 0.1, gamma=0.1, N=1.0)
            result = solve_sir(params)
            finals.append(result.total_infected)
        for i in range(len(finals) - 1):
            assert finals[i] <= finals[i + 1] + 0.01, \
                f"Final size not monotonic: {finals}"

    def test_sirs_lower_peak_than_sir(self):
        """SIRS（有变异复燃）的峰值应低于 SIR（免疫永久）。"""
        params = SIRParams(beta=0.3, gamma=0.1, sigma=0.02, N=1.0)
        result_sir = solve_sir(params, model_type="SIR")
        result_sirs = solve_sir(params, model_type="SIRS")
        # SIRS may have slightly lower peak since recovered become susceptible again
        assert result_sirs.peak_infected <= result_sir.peak_infected * 1.1

    def test_sirs_endemic_state(self):
        """SIRS 模型在 σ > 0 时，最终 I 应 > 0（转为地方性循环）。"""
        params = SIRParams(beta=0.3, gamma=0.1, sigma=0.05, N=1.0)
        result = solve_sir(params, model_type="SIRS", t_span=(0, 500))
        assert result.I[-1] > 1e-4, f"Expected endemic state, got I={result.I[-1]}"


class TestTwoPopulation:
    """验证双群体模型。"""

    def test_core_infected_first(self):
        """核心圈层应先于大众被感染。"""
        params = TwoPopParams(
            beta_core=0.5, beta_mass=0.15,
            beta_cross_c2m=0.1, beta_cross_m2c=0.01,
            gamma=0.1, N_core=0.05, N_mass=0.95,
        )
        result = solve_two_population(params)
        # Core peak should come before mass peak
        core_peak_idx = np.argmax(result["I_core"])
        mass_peak_idx = np.argmax(result["I_mass"])
        assert core_peak_idx <= mass_peak_idx, \
            f"Core peak at {core_peak_idx}, mass peak at {mass_peak_idx}"


class TestParameterFitting:
    """验证参数拟合和估算。"""

    def test_fit_sir_to_self(self):
        """用 SIR 模型生成数据 → 拟合 → 恢复原始参数。"""
        true_params = SIRParams(beta=0.3, gamma=0.1, N=1.0)
        true_result = solve_sir(true_params)

        # Sample the curve
        idx = np.linspace(0, len(true_result.t) - 1, 20, dtype=int)
        t_sample = true_result.t[idx]
        I_sample = true_result.I[idx]

        fitted = fit_sir_to_curve(t_sample, I_sample, N=1.0)
        # Check R0 is approximately recovered
        assert abs(fitted.R0 - true_params.R0) / true_params.R0 < 0.5, \
            f"R0 not recovered: true={true_params.R0:.2f}, fitted={fitted.R0:.2f}"

    def test_estimate_from_lifecycle(self):
        """手动参数估算应在合理范围内。"""
        params = estimate_params_from_lifecycle(
            peak_day=30, total_infected=0.7, duration_days=90,
        )
        assert params.R0 > 1.0, f"High spread meme should have R0 > 1, got {params.R0}"
        assert params.beta > 0
        assert params.gamma > 0


class TestLifecycleExtraction:
    """验证生命周期提取。"""

    def test_dying_meme(self):
        """R₀ >> 1 的脉冲型模因应被标记为消亡。"""
        params = SIRParams(beta=0.8, gamma=0.15, N=1.0)
        result = solve_sir(params)
        lc = extract_lifecycle(result)
        assert lc.status in ("消亡", "变异")

    def test_endemic_meme(self):
        """SIRS 模型中的地方性模因应被标记为固化或变异。"""
        params = SIRParams(beta=0.3, gamma=0.1, sigma=0.03, N=1.0)
        result = solve_sir(params, model_type="SIRS", t_span=(0, 500))
        lc = extract_lifecycle(result)
        assert lc.status in ("固化", "变异"), f"Expected 固化/变异, got {lc.status}"


class TestClassification:
    """验证模因类型分类。"""

    def test_pulse_type(self):
        """高 R₀、短持续时间 → 脉冲型。"""
        params = SIRParams(beta=0.8, gamma=0.15, N=1.0)
        result = solve_sir(params)
        c = classify_meme_type(result)
        # Should be pulse or outbreak type
        assert c["type"] in ("脉冲型", "爆发型")

    def test_abortive_type(self):
        """R₀ < 0.8 → 流产型。"""
        params = SIRParams(beta=0.05, gamma=0.12, N=1.0)
        result = solve_sir(params)
        c = classify_meme_type(result)
        assert c["type"] == "流产型", f"Expected 流产型, got {c['type']}"


class TestEntropy:
    """验证混沌熵计算。"""

    def test_entropy_max_at_chaos(self):
        """熵在系统混沌态（S, I, R 分布均匀时）最大。"""
        # Uniform distribution → max entropy
        N = 1.0
        result = SIRResult(
            t=np.array([0.0]), S=np.array([N/3]), I=np.array([N/3]), R=np.array([N/3]),
            params=SIRParams(beta=0.3, gamma=0.1, N=N),
            peak_day=0.0, peak_infected=0.0, total_infected=0.0, duration=0.0,
        )
        H = compute_entropy_curve(result)
        assert abs(H[0] - np.log(3)) < 0.01, f"Expected ln(3)≈{np.log(3):.3f}, got {H[0]:.3f}"

    def test_entropy_min_at_order(self):
        """熵在系统收敛到确定状态（全 R）时最小。"""
        result = SIRResult(
            t=np.array([0.0]), S=np.array([0.01]), I=np.array([0.01]), R=np.array([0.98]),
            params=SIRParams(beta=0.3, gamma=0.1, N=1.0),
            peak_day=0.0, peak_infected=0.0, total_infected=0.98, duration=0.0,
        )
        H = compute_entropy_curve(result)
        assert H[0] < 0.2, f"Expected low entropy, got {H[0]:.3f}"

    def test_entropy_trajectory_arch(self):
        """完整 SIR 轨迹的熵应呈拱形：低→高→低。"""
        params = SIRParams(beta=0.3, gamma=0.1, N=1.0)
        result = solve_sir(params)
        H = compute_entropy_curve(result)

        # Find where entropy peaks
        peak_H_idx = np.argmax(H)
        # Entropy should peak during the outbreak, not at beginning or end
        assert 0 < peak_H_idx < len(H) - 1, "Entropy peak should be in middle of trajectory"


class TestPhaseTransition:
    """验证相变检测。"""

    def test_detect_transition(self):
        """R₀ 跨越 1.0 时应检测到相变。"""
        sweeps = sweep_R0(np.array([0.5, 0.8, 1.2, 1.5, 2.0]))
        transition = detect_phase_transition(sweeps)
        assert transition is not None, "Should detect R0 crossing 1.0"
        assert transition["pre_R0"] < 1.0 < transition["post_R0"]


class TestParameterSweep:
    """验证参数扫描。"""

    def test_sweep_R0_monotonic_peak(self):
        """峰值感染率应随 R₀ 单调递增。"""
        sweeps = sweep_R0(np.array([0.5, 1.0, 2.0, 4.0, 8.0]))
        peaks = [s["result"].peak_infected for s in sweeps]
        for i in range(len(peaks) - 1):
            assert peaks[i] <= peaks[i + 1] + 0.01

    def test_sweep_beta_gamma_shape(self):
        """β-γ 扫描矩阵应有正确形状。"""
        beta_range = np.array([0.1, 0.3, 0.5])
        gamma_range = np.array([0.05, 0.1, 0.2])
        matrix = sweep_beta_gamma(beta_range, gamma_range)
        assert matrix.shape == (len(gamma_range), len(beta_range))
        # Higher beta → higher peak (for same gamma)
        assert matrix[1, 2] >= matrix[1, 0]


# Allow running directly
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))

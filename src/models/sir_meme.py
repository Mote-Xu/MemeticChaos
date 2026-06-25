"""
SIR 模因传播动力学模型 — MemeticChaos 核心建模层。

对齐「微尘哲学」核心元定律：
- 模因传播 = 集体情感系统在混沌中建立局部秩序的过程
- S (Susceptible) = 尚未接触梗的易感人群，处于情感混沌态
- I (Infected) = 正在传播梗的感染人群，参与局部秩序建立
- R (Recovered) = 对该梗免疫/厌倦的恢复人群，秩序僵化或消亡

模型变体：
1. Standard SIR  — 基础模因生命周期
2. SIRS          — 加入变异/复燃机制（梗的变体再传播）
3. TwoPopulation — 核心圈层 → 大众扩散，双群体不同感染率
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Literal
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════

@dataclass
class SIRParams:
    """SIR 模型参数。

    Attributes:
        beta: 感染率 — 一个感染者单位时间内感染的人数
        gamma: 恢复率 — 一个感染者单位时间内恢复/厌倦的速率 (1/平均感染时长)
        R0: 基本再生数 — beta/gamma，若 >1 则模因会爆发传播
        N: 总人口
        sigma: 变异率（SIRS 扩展）— 恢复者重新变为易感者的速率
    """
    beta: float
    gamma: float
    N: float = 1.0
    sigma: float = 0.0

    @property
    def R0(self) -> float:
        return self.beta / self.gamma if self.gamma > 0 else float("inf")

    def __repr__(self) -> str:
        return (
            f"SIRParams(β={self.beta:.4f}, γ={self.gamma:.4f}, "
            f"R₀={self.R0:.2f}, σ={self.sigma:.4f}, N={self.N})"
        )


@dataclass
class SIRResult:
    """SIR 模型求解结果。

    Attributes:
        t: 时间向量
        S: 易感者比例时间序列
        I: 感染者比例时间序列
        R: 恢复者比例时间序列
        params: 模型参数
        peak_day: 感染峰值时间
        peak_infected: 感染峰值比例
        total_infected: 最终累计感染比例
        duration: 有效传播持续时间（I > 阈值）
    """
    t: np.ndarray
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray
    params: SIRParams
    peak_day: float
    peak_infected: float
    total_infected: float
    duration: float

    def to_dict(self) -> dict:
        return {
            "peak_day": float(self.peak_day),
            "peak_infected": float(self.peak_infected),
            "total_infected": float(self.total_infected),
            "duration": float(self.duration),
            "R0": float(self.params.R0),
            "beta": float(self.params.beta),
            "gamma": float(self.params.gamma),
        }


@dataclass
class TwoPopParams:
    """双群体 SIR 参数。

    核心圈层 (core) 与大众 (mass) 具有不同的感染率和交互率。

    Attributes:
        beta_core: 核心圈层内部感染率
        beta_mass: 大众内部感染率
        beta_cross_c2m: 核心 → 大众 跨层感染率
        beta_cross_m2c: 大众 → 核心 跨层感染率
        gamma: 恢复率（两个群体相同）
        N_core: 核心圈层人口比例
        N_mass: 大众人口比例
    """
    beta_core: float
    beta_mass: float
    beta_cross_c2m: float    # core infects mass
    beta_cross_m2c: float    # mass infects core
    gamma: float
    N_core: float = 0.05
    N_mass: float = 0.95

    @property
    def R0_core(self) -> float:
        return self.beta_core / self.gamma if self.gamma > 0 else float("inf")


@dataclass
class MemeLifecycle:
    """从 SIR 结果中提取的热梗生命周期特征。

    将 SIR 曲线映射到热梗的实际生命周期阶段：
    - emergence: 萌芽期 (I 从 0 到 0.01)
    - peak: 爆发顶峰 (I 达到 max)
    - decay: 衰退期 (I 从 max 降至 0.01)
    - status: 最终状态 (消亡 / 固化 / 变异)
    """
    emergence_start: float
    emergence_end: float
    peak_time: float
    decay_end: float
    peak_infected: float
    total_recovered: float
    status: str  # "消亡", "固化", "变异"

    def to_dict(self) -> dict:
        return {
            "emergence": {"start": float(self.emergence_start), "end": float(self.emergence_end)},
            "peak": float(self.peak_time),
            "decay_end": float(self.decay_end),
            "peak_infected": float(self.peak_infected),
            "total_recovered": float(self.total_recovered),
            "status": self.status,
        }


# ═══════════════════════════════════════════════
# Core SIR dynamics
# ═══════════════════════════════════════════════

def sir_derivatives(t: float, y: np.ndarray, beta: float, gamma: float, N: float) -> np.ndarray:
    """标准 SIR 模型微分方程。

    dS/dt = -β·S·I / N
    dI/dt =  β·S·I / N - γ·I
    dR/dt =  γ·I

    Args:
        t: 时间
        y: [S, I, R] 状态向量
        beta: 感染率
        gamma: 恢复率
        N: 总人口
    """
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR])


def sirs_derivatives(t: float, y: np.ndarray, beta: float, gamma: float,
                     sigma: float, N: float) -> np.ndarray:
    """SIRS 模型微分方程（含变异/复燃）。

    与 SIR 的区别：恢复者以速率 σ 重新变为易感者，
    模拟梗的变体再传播（如'打工人'→'尾款人'→'早八人'）。

    dS/dt = -β·S·I/N + σ·R
    dI/dt =  β·S·I/N - γ·I
    dR/dt =  γ·I - σ·R
    """
    S, I, R = y
    dS = -beta * S * I / N + sigma * R
    dI = beta * S * I / N - gamma * I
    dR = gamma * I - sigma * R
    return np.array([dS, dI, dR])


def two_population_derivatives(t: float, y: np.ndarray, params: TwoPopParams) -> np.ndarray:
    """双群体 SIR 模型微分方程。

    状态: [S_core, I_core, R_core, S_mass, I_mass, R_mass]
    核心圈层先感染，然后跨层传播到大众。

    Args:
        t: 时间
        y: 6 维状态向量 [S_c, I_c, R_c, S_m, I_m, R_m]
        params: TwoPopParams
    """
    Sc, Ic, Rc, Sm, Im, Rm = y
    Nc, Nm = params.N_core, params.N_mass

    # Core dynamics: core内部感染 + 从mass传回core (通常很小)
    dSc = (-params.beta_core * Sc * Ic / Nc
           - params.beta_cross_m2c * Sc * Im / Nc)
    dIc = (params.beta_core * Sc * Ic / Nc
           + params.beta_cross_m2c * Sc * Im / Nc
           - params.gamma * Ic)
    dRc = params.gamma * Ic

    # Mass dynamics: mass内部感染 + 从core传入
    dSm = (-params.beta_mass * Sm * Im / Nm
           - params.beta_cross_c2m * Sm * Ic / Nm)
    dIm = (params.beta_mass * Sm * Im / Nm
           + params.beta_cross_c2m * Sm * Ic / Nm
           - params.gamma * Im)
    dRm = params.gamma * Im

    return np.array([dSc, dIc, dRc, dSm, dIm, dRm])


# ═══════════════════════════════════════════════
# Solvers
# ═══════════════════════════════════════════════

def solve_sir(params: SIRParams, t_span: tuple = (0, 200),
              t_eval: Optional[np.ndarray] = None,
              I0: float = 0.001,
              model_type: Literal["SIR", "SIRS"] = "SIR") -> SIRResult:
    """求解 SIR / SIRS 模型。

    Args:
        params: 模型参数
        t_span: 时间范围 (start, end)，默认 0-200 天
        t_eval: 评估时间点，默认均匀 1000 点
        I0: 初始感染者比例
        model_type: "SIR" 或 "SIRS"

    Returns:
        SIRResult 包含完整时间序列和摘要统计量
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)

    R0_val = 0.0
    S0 = params.N - I0 - R0_val
    y0 = np.array([S0, I0, R0_val])

    if model_type == "SIR":
        sol = solve_ivp(
            sir_derivatives, t_span, y0,
            args=(params.beta, params.gamma, params.N),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
        )
    else:  # SIRS
        sol = solve_ivp(
            sirs_derivatives, t_span, y0,
            args=(params.beta, params.gamma, params.sigma, params.N),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
        )

    S, I, R = sol.y
    t = sol.t

    # Calculate summary statistics
    peak_idx = np.argmax(I)
    peak_day = float(t[peak_idx])
    peak_infected = float(I[peak_idx])
    total_infected = float(R[-1])
    # Duration: time I > 1% of peak
    threshold = 0.01 * peak_infected
    above_threshold = I > threshold
    if above_threshold.any():
        duration = float(t[above_threshold][-1] - t[above_threshold][0])
    else:
        duration = 0.0

    return SIRResult(
        t=t, S=S, I=I, R=R,
        params=params,
        peak_day=peak_day,
        peak_infected=peak_infected,
        total_infected=total_infected,
        duration=duration,
    )


def solve_two_population(params: TwoPopParams, t_span: tuple = (0, 200),
                         t_eval: Optional[np.ndarray] = None,
                         I0_core: float = 0.001,
                         I0_mass: float = 0.0) -> dict:
    """求解双群体 SIR 模型。

    Args:
        params: TwoPopParams
        t_span: 时间范围
        t_eval: 评估时间点
        I0_core: 核心圈层初始感染者比例（相对核心人口）
        I0_mass: 大众初始感染者比例（相对大众人口）

    Returns:
        dict: {t, S_core, I_core, R_core, S_mass, I_mass, R_mass, params}
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)

    Sc0 = params.N_core - I0_core
    Ic0 = I0_core
    Rc0 = 0.0
    Sm0 = params.N_mass - I0_mass
    Im0 = I0_mass
    Rm0 = 0.0

    y0 = np.array([Sc0, Ic0, Rc0, Sm0, Im0, Rm0])

    sol = solve_ivp(
        two_population_derivatives, t_span, y0,
        args=(params,),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
    )

    Sc, Ic, Rc, Sm, Im, Rm = sol.y

    return {
        "t": sol.t,
        "S_core": Sc, "I_core": Ic, "R_core": Rc,
        "S_mass": Sm, "I_mass": Im, "R_mass": Rm,
        "params": params,
    }


# ═══════════════════════════════════════════════
# Parameter fitting
# ═══════════════════════════════════════════════

def fit_sir_to_curve(t_data: np.ndarray, I_data: np.ndarray,
                     N: float = 1.0,
                     initial_guess: Optional[tuple] = None,
                     bounds: Optional[tuple] = None) -> SIRParams:
    """用观测数据拟合 SIR 模型参数。

    使用最小二乘法拟合感染曲线 I(t)。

    Args:
        t_data: 观测时间点（归一化到 0-200 范围）
        I_data: 观测感染比例（归一化到 0-1）
        N: 总人口
        initial_guess: (beta, gamma) 初始猜测
        bounds: ((beta_min, gamma_min), (beta_max, gamma_max))

    Returns:
        SIRParams 拟合参数
    """
    if initial_guess is None:
        initial_guess = (0.3, 0.1)
    if bounds is None:
        bounds = ((0.001, 0.001), (5.0, 2.0))

    def _I_of_t(t, beta, gamma):
        params = SIRParams(beta=beta, gamma=gamma, N=N)
        # Use fast evaluation at the specific t points
        result = solve_sir(params, t_span=(0, max(t_data) * 1.5 + 10),
                          t_eval=np.sort(np.unique(np.concatenate([
                              t_data,
                              np.linspace(0, max(t_data) * 1.5, 500)
                          ]))))
        # Interpolate to t_data
        return np.interp(t_data, result.t, result.I)

    try:
        popt, pcov = curve_fit(
            _I_of_t, t_data, I_data,
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000,
        )
        beta, gamma = popt
        return SIRParams(beta=beta, gamma=gamma, N=N)
    except Exception as e:
        # Fallback: try with wider bounds
        try:
            popt, _ = curve_fit(
                _I_of_t, t_data, I_data,
                p0=initial_guess,
                bounds=((0.0001, 0.0001), (10.0, 5.0)),
                maxfev=10000,
            )
            beta, gamma = popt
            return SIRParams(beta=beta, gamma=gamma, N=N)
        except Exception:
            raise RuntimeError(f"Failed to fit SIR model: {e}")


def estimate_params_from_lifecycle(
    peak_day: float,
    total_infected: float,
    duration_days: float,
    N: float = 1.0
) -> SIRParams:
    """从热梗生命周期的观测特征粗略估算 SIR 参数。

    用于没有完整时间序列数据时的手动参数估计。

    Args:
        peak_day: 从出现到峰值的近似天数
        total_infected: 最终累计感染比例 (R(∞))
        duration_days: 有效传播持续天数
        N: 总人口

    Returns:
        SIRParams 估算参数
    """
    # Gamma ≈ 1 / (duration / 2) — 恢复率近似为有效传播时长一半的倒数
    gamma_est = 2.0 / max(duration_days, 1.0)

    # R₀ estimation from SIR final size equation:
    # R∞ = 1 - exp(-R₀ × R∞)  (assuming S₀ ≈ 1)
    # → R₀ = -ln(1 - R∞) / R∞   (analytical solution)
    target = total_infected
    if target >= 0.99:
        R0_est = 10.0
    elif target <= 0.001:
        R0_est = 0.1
    else:
        R0_est = -np.log(1.0 - target) / target

    beta_est = R0_est * gamma_est

    return SIRParams(beta=beta_est, gamma=gamma_est, N=N)


# ═══════════════════════════════════════════════
# Lifecycle extraction
# ═══════════════════════════════════════════════

def extract_lifecycle(result: SIRResult, threshold: float = 0.01) -> MemeLifecycle:
    """从 SIR 结果中提取热梗生命周期。

    Args:
        result: SIRResult
        threshold: 判断萌芽/消亡的 I 阈值（相对峰值）

    Returns:
        MemeLifecycle
    """
    I = result.I
    t = result.t
    peak = result.peak_infected
    thresh_val = threshold * peak

    # Emergence: I 从 0 到 threshold
    above_emergence = np.where(I > thresh_val)[0]
    emergence_start = float(t[0])
    emergence_end = float(t[above_emergence[0]]) if len(above_emergence) > 0 else float(t[0])

    # Decay: I 从 peak 降到 threshold
    peak_idx = np.argmax(I)
    post_peak = np.where((np.arange(len(t)) > peak_idx) & (I < thresh_val))[0]
    decay_end = float(t[post_peak[0]]) if len(post_peak) > 0 else float(t[-1])

    # Status determination
    final_I = float(I[-1])
    final_R = float(result.R[-1])
    if final_I < thresh_val and final_R > 0.95:
        status = "消亡"
    elif final_I > 0.01:
        status = "固化"
    else:
        status = "变异"  # SIRS model: low but non-zero I, continued circulation

    return MemeLifecycle(
        emergence_start=emergence_start,
        emergence_end=emergence_end,
        peak_time=result.peak_day,
        decay_end=decay_end,
        peak_infected=result.peak_infected,
        total_recovered=result.total_infected,
        status=status,
    )


# ═══════════════════════════════════════════════
# Chaos dynamics analysis
# ═══════════════════════════════════════════════

def compute_lyapunov_like(sir_results: list[SIRResult],
                          perturbation: float = 0.001) -> float:
    """计算 SIR 系统的类 Lyapunov 指数。

    通过对初始条件施加微扰，测量轨迹的发散/收敛速度。
    正值 → 混沌（初值敏感），负值 → 稳定吸引子。

    这是对经典 Lyapunov 指数的简化近似。

    Args:
        sir_results: 两个 SIRResult，一个无扰动，一个有扰动
        perturbation: 扰动大小

    Returns:
        类 Lyapunov 指数
    """
    if len(sir_results) != 2:
        raise ValueError("需要恰好两个 SIRResult（无扰动 + 有扰动）")

    I0 = sir_results[0].I
    I1 = sir_results[1].I
    t = sir_results[0].t

    # Divergence of trajectories
    divergence = np.abs(I0 - I1)
    # Avoid log(0)
    divergence = np.maximum(divergence, 1e-12)

    # λ ≈ (1/T) * mean(log(|ΔI|/ε))
    lyap = np.mean(np.log(divergence / perturbation)) / (t[-1] - t[0])
    return float(lyap)


def detect_phase_transition(param_sweeps: list[dict]) -> Optional[dict]:
    """检测参数空间的相变点。

    当 R₀ 跨越 1.0 时，系统行为发生质变（模因从消亡→爆发）。
    检测 R₀ 轨迹中的临界点。

    Args:
        param_sweeps: 参数扫描结果列表，每个包含 {"params": SIRParams, "result": SIRResult}

    Returns:
        {"transition_at": index, "pre_R0": float, "post_R0": float} 或 None
    """
    R0s = [s["params"].R0 for s in param_sweeps]
    for i in range(len(R0s) - 1):
        product = (R0s[i] - 1.0) * (R0s[i + 1] - 1.0)
        if product < 0 or (product == 0 and (R0s[i] != 1.0 or R0s[i+1] != 1.0)):
            return {
                "transition_at": i,
                "pre_R0": R0s[i],
                "post_R0": R0s[i + 1],
            }
    return None


def compute_entropy_curve(result: SIRResult) -> np.ndarray:
    """计算 SIR 轨迹的瞬时熵（香农熵）。

    在每一时刻，系统状态 (S, I, R) 的概率分布对应的熵值：
    H(t) = -Σ p_i(t) * log(p_i(t))
    熵的最大值 = log(3) ≈ 1.099 — 系统处于最大混沌态
    熵的最小值 → 0 — 系统收敛到确定状态（全员 R）

    Args:
        result: SIRResult

    Returns:
        与 t 等长的熵值时间序列
    """
    S, I, R = result.S, result.I, result.R
    # Normalize to probability distribution
    total = S + I + R
    p_S = S / total
    p_I = I / total
    p_R = R / total

    # Shannon entropy, handling log(0)
    epsilon = 1e-12
    entropy = -(p_S * np.log(np.maximum(p_S, epsilon)) +
                p_I * np.log(np.maximum(p_I, epsilon)) +
                p_R * np.log(np.maximum(p_R, epsilon)))
    return entropy


# ═══════════════════════════════════════════════
# Parameter sweeps
# ═══════════════════════════════════════════════

def sweep_R0(R0_range: np.ndarray, gamma: float = 0.1,
             N: float = 1.0, model_type: str = "SIR") -> list[dict]:
    """扫描 R₀ 参数，观察系统行为变化。

    Args:
        R0_range: R₀ 值数组
        gamma: 固定恢复率
        N: 总人口
        model_type: "SIR" 或 "SIRS"

    Returns:
        每次扫描的结果列表 [{"params": SIRParams, "result": SIRResult}, ...]
    """
    results = []
    for R0 in R0_range:
        beta = R0 * gamma
        params = SIRParams(beta=beta, gamma=gamma, N=N)
        result = solve_sir(params, model_type=model_type)
        results.append({"params": params, "result": result})
    return results


def sweep_beta_gamma(beta_range: np.ndarray, gamma_range: np.ndarray,
                     N: float = 1.0) -> np.ndarray:
    """扫描 β-γ 参数空间，返回峰值感染率矩阵。

    Args:
        beta_range: β 值数组
        gamma_range: γ 值数组
        N: 总人口

    Returns:
        2D 数组: peak_infected[γ_idx, β_idx]
    """
    peak_matrix = np.zeros((len(gamma_range), len(beta_range)))
    for i, gamma in enumerate(gamma_range):
        for j, beta in enumerate(beta_range):
            if beta * gamma > 0:
                params = SIRParams(beta=beta, gamma=gamma, N=N)
                result = solve_sir(params)
                peak_matrix[i, j] = result.peak_infected
    return peak_matrix


# ═══════════════════════════════════════════════
# Meme categorization based on SIR signatures
# ═══════════════════════════════════════════════

def classify_meme_type(result: SIRResult) -> dict:
    """基于 SIR 曲线形态对热梗进行分类。

    脉冲型: R₀ >> 1, duration 短 (< 30天), peak_infected 高 — 爆发快消退快
    爆发型: R₀ > 1, duration 中 (30-90天), peak_infected 高 — 经典模因
    长尾型: R₀ ≈ 1-2, duration 长 (> 90天), peak_infected 低 — 持续渗透
    流产型: R₀ < 1, 几乎不传播

    Returns:
        {"type": str, "confidence": float, "features": dict}
    """
    R0 = result.params.R0
    peak = result.peak_infected
    dur = result.duration

    if R0 < 0.8:
        meme_type = "流产型"
        confidence = 0.9
    elif dur < 30 and peak > 0.3:
        meme_type = "脉冲型"
        confidence = min(0.9, (R0 - 1) / 3 + 0.5)
    elif dur > 90:
        meme_type = "长尾型"
        confidence = 0.7
    elif R0 > 1.0:
        meme_type = "爆发型"
        confidence = min(0.9, (R0 - 1) / 2 + 0.5)
    else:
        meme_type = "流产型"
        confidence = 0.6

    return {
        "type": meme_type,
        "confidence": round(confidence, 3),
        "features": {
            "R0": round(R0, 3),
            "peak_infected": round(peak, 4),
            "duration_days": round(dur, 1),
        },
    }


# ═══════════════════════════════════════════════
# Demonstration with typical meme parameters
# ═══════════════════════════════════════════════

def demo_all_types() -> dict[str, SIRResult]:
    """用典型参数演示四种热梗类型的 SIR 曲线。"""
    configs = {
        "脉冲型_雪糕刺客": SIRParams(beta=0.8, gamma=0.15, N=1.0),
        "爆发型_打工人": SIRParams(beta=0.3, gamma=0.08, N=1.0),
        "长尾型_内卷": SIRParams(beta=0.12, gamma=0.08, N=1.0),
        "流产型_小众梗": SIRParams(beta=0.05, gamma=0.12, N=1.0),
    }
    results = {}
    for name, params in configs.items():
        results[name] = solve_sir(params)
    return results


# ═══════════════════════════════════════════════
# Script entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MemeticChaos SIR Model — Demonstration")
    print("=" * 60)

    print("\n▶ Demo: Four meme types based on typical parameters\n")
    demos = demo_all_types()
    for name, result in demos.items():
        classification = classify_meme_type(result)
        lifecycle = extract_lifecycle(result)
        print(f"  {name}:")
        print(f"    {result.params}")
        print(f"    Peak infected: {result.peak_infected:.1%} at day {result.peak_day:.1f}")
        print(f"    Duration: {result.duration:.1f} days")
        print(f"    Classified: {classification['type']} (confidence: {classification['confidence']})")
        print(f"    Lifecycle: emergence→{lifecycle.emergence_end:.1f}d, "
              f"peak→{lifecycle.peak_time:.1f}d, decay_end→{lifecycle.decay_end:.1f}d, "
              f"status={lifecycle.status}")
        print()

    print("▶ Chaos axis interpretation:")
    print("  脉冲型 = 高熵爆发，秩序快速建立→快速消亡 → 偏混沌")
    print("  爆发型 = 经典的混沌→秩序→免疫 三阶段")
    print("  长尾型 = 缓慢渗透，秩序持续建立 → 偏秩序")
    print("  流产型 = 未能建立秩序，被混沌吞没")

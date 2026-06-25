"""
个体混沌属性校准器 — MemeticChaos 个人层建模。

核心问题：给定一个个体在外部可观测的行为序列，
反推出其内部混沌属性的最概然参数估计。

对齐「微尘哲学」：
- 其他主体的「小真实」是不可穿透的黑箱
- 但外部行为信号会在黑箱表面形成「映射」
- 校准的目标不是获得确定性真相，而是构建最可能的结构性推断
- 输出永远是概率性的：「我认为大概率是 X」

方法论：
1. 行为特征提取：从原始观测中提取个体行为签名
2. 前向仿真：用 ABM agent 参数 → 仿真个体轨迹
3. 逆问题求解：遗传算法 + 贝叶斯优化，最小化仿真 vs 观测的差异
4. 不确定性量化：参数后验分布，而非点估计
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

from src.models.abm_simulation import (
    ChaosRole, MemeState, ABMConfig, build_random_network,
)


# ═══════════════════════════════════════════════
# Behavioral observation data structures
# ═══════════════════════════════════════════════

class MemeParticipation(Enum):
    """个体对特定热梗的参与模式。"""
    EARLY_ADOPTER = "early"       # 萌芽期介入（前10%）
    EARLY_MAJORITY = "early_maj"  # 成长期介入
    LATE_MAJORITY = "late_maj"    # 爆发期介入
    LAGGARD = "laggard"           # 消退期才参与
    NEVER = "never"               # 从未参与
    RESISTER = "resister"         # 主动抵制


@dataclass
class BehavioralObservation:
    """个体的外部行为观测序列。

    这是我们能从外部观测到的一切——黑箱表面的映射。
    字段都是 optional——现实中不可能全部观测到。
    """

    # ── 模因参与模式 ──
    meme_participation: Optional[dict[str, MemeParticipation]] = None
    # {"打工人": EARLY_ADOPTER, "躺平": LATE_MAJORITY, ...}
    n_memes_participated: Optional[int] = None
    n_memes_spread: Optional[int] = None     # 主动传播的梗数
    n_memes_ignored: Optional[int] = None    # 未参与的梗数

    # ── 情感时序 ──
    sentiment_trajectory: Optional[np.ndarray] = None
    # 情感极性时间序列（0=极端负面, 1=极端正面, 0.5=中性）
    sentiment_variance: Optional[float] = None  # 情感波动性
    sentiment_trend: Optional[float] = None     # 情感趋势（正=越来越正面）

    # ── 混沌位置代理信号 ──
    chaos_proxy_signals: Optional[dict[str, float]] = None
    # 间接反映 chaos_position 的可观测信号：
    #   "orderly_expression_ratio": 有序表达占比
    #   "contradiction_frequency": 自相矛盾频率
    #   "narrative_consistency": 叙事一致性
    #   "echo_chamber_affinity": 回音壁倾向
    #   "trolling_frequency": 恶意/混沌投放频率

    # ── 交互模式 ──
    avg_influence_received: Optional[float] = None  # 平均被影响程度
    avg_influence_exerted: Optional[float] = None   # 平均施加影响程度
    interaction_diversity: Optional[float] = None   # 交互对象的多样性

    # ── 自我报告（来自问卷/访谈/文本分析）──
    self_reported_chaos: Optional[float] = None   # -1 到 +1
    self_reported_resilience: Optional[float] = None
    self_reported_vitality: Optional[float] = None

    def to_feature_vector(self) -> np.ndarray:
        """将观测转换为特征向量（缺失值用 NaN）。"""
        features = []

        # Meme participation features
        if self.meme_participation:
            early_count = sum(1 for p in self.meme_participation.values()
                            if p in (MemeParticipation.EARLY_ADOPTER, MemeParticipation.EARLY_MAJORITY))
            late_count = sum(1 for p in self.meme_participation.values()
                           if p in (MemeParticipation.LATE_MAJORITY, MemeParticipation.LAGGARD))
            resist_count = sum(1 for p in self.meme_participation.values()
                             if p == MemeParticipation.RESISTER)
            total = max(len(self.meme_participation), 1)
            features.extend([early_count / total, late_count / total, resist_count / total])

        if self.n_memes_participated is not None:
            features.append(self.n_memes_participated)
        if self.n_memes_spread is not None:
            features.append(self.n_memes_spread)

        # Sentiment features
        if self.sentiment_trajectory is not None:
            ts = self.sentiment_trajectory
            features.extend([
                float(np.mean(ts)),
                float(np.std(ts)),
                float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,  # trend
            ])
        if self.sentiment_variance is not None:
            features.append(self.sentiment_variance)
        if self.sentiment_trend is not None:
            features.append(self.sentiment_trend)

        # Chaos proxy signals
        if self.chaos_proxy_signals:
            for key in ["orderly_expression_ratio", "contradiction_frequency",
                        "narrative_consistency", "trolling_frequency"]:
                if key in self.chaos_proxy_signals:
                    features.append(self.chaos_proxy_signals[key])

        # Interaction
        if self.avg_influence_exerted is not None:
            features.append(self.avg_influence_exerted)
        if self.interaction_diversity is not None:
            features.append(self.interaction_diversity)

        return np.array(features, dtype=float)


@dataclass
class IndividualChaosProfile:
    """个体混沌属性剖面 — 校准器的输出。

    这是对一个人内部混沌结构的**概率性映射**，
    不是确定性断言。
    """

    # ── 核心参数估计 ──
    chaos_position_est: float          # 最概然混沌位置 (-1 到 +1)
    chaos_position_std: float          # 估计不确定性
    resilience_est: float              # 韧性估计
    influence_est: float               # 影响力估计

    # ── 角色分类 ──
    role_probabilities: dict[str, float]  # {normal, builder, injector, lurker}
    most_likely_role: str

    # ── 模因行为预测 ──
    predicted_participation_pattern: str  # "early_adopter" | "follower" | "resister"
    susceptibility_est: float             # 对模因的易感性

    # ── 混沌动力学 ──
    chaos_stability: str              # "stable" | "oscillating" | "drifting"

    # ── 元信息 ──
    calibration_method: str           # 使用的校准方法
    confidence: float = 0.5           # 整体置信度 (0-1)
    caveats: list[str] = field(default_factory=list)
    predicted_entropy_trajectory: Optional[np.ndarray] = None
    # 关键警示：提醒用户这是概率推断，不是确定事实

    def summary(self) -> str:
        lines = [
            f"=== 个体混沌属性剖面 ===",
            f"混沌位置: {self.chaos_position_est:+.2f} ± {self.chaos_position_std:.2f}",
            f"韧性: {self.resilience_est:.2f} | 影响力: {self.influence_est:.3f}",
            f"最可能角色: {self.most_likely_role}",
            f"  角色概率: {self.role_probabilities}",
            f"模因参与模式: {self.predicted_participation_pattern}",
            f"模因易感性: {self.susceptibility_est:.2f}",
            f"混沌稳定性: {self.chaos_stability}",
            f"置信度: {self.confidence:.1%}",
            f"校准方法: {self.calibration_method}",
        ]
        if self.caveats:
            lines.append("--- 警示 ---")
            for c in self.caveats:
                lines.append(f"  ⚠ {c}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════
# Forward model: agent params → behavioral trajectory
# ═══════════════════════════════════════════════

def simulate_individual(params: dict, n_steps: int = 100,
                        meme_events: list[dict] = None) -> dict:
    """用给定参数仿真一个体的模因行为轨迹。

    前向模型：参数 → 行为序列。
    用于与真实观测对比。

    Args:
        params: {chaos_position, resilience, influence, role, ...}
        n_steps: 仿真步数
        meme_events: [{step, meme_id, initial_infected}, ...] 外部模因事件时间线

    Returns:
        {chaos_traj, vitality_traj, meme_state_traj, participation_record}
    """
    np.random.seed(params.get("seed", 42))

    chaos = params.get("chaos_position", np.random.uniform(-0.5, 0.5))
    resilience = params.get("resilience", 0.5)
    influence = params.get("influence", 0.1)
    role_str = params.get("role", "normal")

    role_map = {
        "normal": ChaosRole.NORMAL,
        "builder": ChaosRole.ORDER_BUILDER,
        "injector": ChaosRole.CHAOS_INJECTOR,
        "lurker": ChaosRole.LURKER,
    }
    role = role_map.get(role_str, ChaosRole.NORMAL)

    vitality = params.get("vitality", 0.7)

    entropy_rate = params.get("entropy_rate", 0.015)
    contagion_rate = params.get("contagion_rate", 0.05)
    infection_rate = params.get("infection_rate", 0.12)
    recovery_rate = params.get("recovery_rate", 0.04)
    noise_level = params.get("noise_level", 0.005)

    # External meme influence (simulated neighbors)
    neighbor_chaos = np.random.uniform(-0.3, 0.3)  # average neighbor chaos

    chaos_traj = []
    vitality_traj = []
    meme_state_traj = []
    participation_record = {}

    current_meme_state = MemeState.SUSCEPTIBLE
    current_meme_id = None
    infected_since = -1

    for step in range(n_steps):
        # 1. Entropy drift
        drift = entropy_rate * (1.0 - resilience)
        if role == ChaosRole.ORDER_BUILDER:
            drift *= 0.3
        chaos -= drift * chaos * 0.1

        # 2. Contagion (from external neighbor influence)
        chaos += contagion_rate * (neighbor_chaos - chaos)

        # 3. Meme events (external shocks)
        if meme_events:
            for event in meme_events:
                if event.get("step") == step:
                    if current_meme_state == MemeState.SUSCEPTIBLE:
                        # Probability of getting infected by this meme
                        order_bias = 0.5 + 0.5 * max(0, chaos)
                        p_infect = infection_rate * order_bias
                        if np.random.random() < p_infect:
                            current_meme_state = MemeState.INFECTED
                            current_meme_id = event["meme_id"]
                            infected_since = step
                            chaos += 0.1 * (1.0 - abs(chaos))
                            participation_record[current_meme_id] = {
                                "infected_step": step,
                                "pattern": "early" if step < n_steps * 0.2 else
                                          "mid" if step < n_steps * 0.5 else "late",
                            }

        # 4. Recovery
        if current_meme_state == MemeState.INFECTED and infected_since >= 0:
            time_infected = step - infected_since
            p_recover = 1.0 - np.exp(-recovery_rate * time_infected)
            if np.random.random() < p_recover:
                current_meme_state = MemeState.RECOVERED
                if current_meme_id:
                    participation_record[current_meme_id]["recovered_step"] = step

        # 5. Vitality update
        extremity_penalty = abs(chaos) ** 2 * 0.01
        infection_bonus = 0.005 if current_meme_state == MemeState.INFECTED else 0.0
        resilience_bonus = resilience * 0.002
        vitality += resilience_bonus + infection_bonus - extremity_penalty

        # 6. Noise
        chaos += np.random.normal(0, noise_level)
        vitality += np.random.normal(0, noise_level * 0.5)

        # Clamp
        chaos = np.clip(chaos, -1.0, 1.0)
        vitality = np.clip(vitality, 0.0, 1.0)

        chaos_traj.append(float(chaos))
        vitality_traj.append(float(vitality))
        meme_state_traj.append(current_meme_state.value)

    return {
        "chaos_traj": np.array(chaos_traj),
        "vitality_traj": np.array(vitality_traj),
        "meme_state_traj": meme_state_traj,
        "participation_record": participation_record,
        "final_chaos": float(chaos_traj[-1]),
        "final_vitality": float(vitality_traj[-1]),
        "chaos_variance": float(np.var(chaos_traj)),
        "chaos_trend": float(chaos_traj[-1] - chaos_traj[0]),
    }


# ═══════════════════════════════════════════════
# Inverse problem: observation → parameter inference
# ═══════════════════════════════════════════════

def _compute_loss(sim_result: dict, observation: BehavioralObservation) -> float:
    """计算仿真结果与观测之间的损失。

    Loss 越小 → 参数越匹配观测。
    不同信号类型有不同的权重，缺失信号被跳过。
    """
    loss = 0.0
    n_terms = 0

    # Chaos proxy signals → chaos trajectory
    if observation.chaos_proxy_signals:
        proxy = observation.chaos_proxy_signals

        # orderly_expression_ratio → high chaos_position (more orderly)
        if "orderly_expression_ratio" in proxy:
            predicted_order = max(0, sim_result["final_chaos"])
            loss += (proxy["orderly_expression_ratio"] - predicted_order) ** 2
            n_terms += 1

        # contradiction_frequency → high chaos variance
        if "contradiction_frequency" in proxy:
            predicted_var = sim_result["chaos_variance"]
            loss += (proxy["contradiction_frequency"] * 10 - predicted_var * 10) ** 2
            n_terms += 1

        # narrative_consistency → low chaos variance + positive chaos trend
        if "narrative_consistency" in proxy:
            predicted_consistency = 1.0 - sim_result["chaos_variance"] * 5
            loss += (proxy["narrative_consistency"] - predicted_consistency) ** 2
            n_terms += 1

        # trolling_frequency → injector role (negative chaos, high influence)
        if "trolling_frequency" in proxy and proxy["trolling_frequency"] > 0.1:
            # High trolling → likely injector
            predicted_troll = abs(min(0, sim_result["final_chaos"]))
            loss += (proxy["trolling_frequency"] * 3 - predicted_troll * 3) ** 2
            n_terms += 1

    # Self-report anchors
    if observation.self_reported_chaos is not None:
        loss += (observation.self_reported_chaos - sim_result["final_chaos"]) ** 2 * 2
        n_terms += 2  # double weight for self-report

    if observation.self_reported_vitality is not None:
        loss += (observation.self_reported_vitality - sim_result["final_vitality"]) ** 2
        n_terms += 1

    # Participation pattern
    if observation.meme_participation:
        early_count = sum(1 for p in observation.meme_participation.values()
                        if p in (MemeParticipation.EARLY_ADOPTER, MemeParticipation.EARLY_MAJORITY))
        resist_count = sum(1 for p in observation.meme_participation.values()
                         if p == MemeParticipation.RESISTER)
        total = len(observation.meme_participation)

        if total > 0:
            early_ratio = early_count / total
            resist_ratio = resist_count / total

            # Early adopters tend to have higher chaos_position (order-seeking)
            # and higher susceptibility
            predicted_early = max(0, sim_result["final_chaos"]) * 0.8
            loss += (early_ratio - predicted_early) ** 2
            n_terms += 1

    if n_terms == 0:
        return 0.0
    return loss / n_terms


def calibrate_from_observation(observation: BehavioralObservation,
                               n_iterations: int = 2000,
                               population_size: int = 50) -> IndividualChaosProfile:
    """从行为观测反推个体混沌参数。

    使用简化遗传算法在参数空间中搜索最匹配的个体配置。

    Args:
        observation: 可观测的行为信号
        n_iterations: 搜索迭代次数
        population_size: 候选参数群体大小

    Returns:
        IndividualChaosProfile — 概率性个体混沌剖面
    """
    np.random.seed(42)

    # ── Parameter bounds ───────────────────────
    bounds = {
        "chaos_position": (-1.0, 1.0),
        "resilience": (0.1, 0.95),
        "influence": (0.01, 0.5),
        "vitality": (0.3, 1.0),
    }
    roles = ["normal", "builder", "injector", "lurker"]

    # ── Generate meme event timeline from observation ──
    meme_events = []
    if observation.meme_participation:
        step = 0
        for meme_name, pattern in observation.meme_participation.items():
            meme_events.append({
                "step": step,
                "meme_id": meme_name,
                "pattern": pattern.value,
            })
            step += 20  # Space meme events apart

    # ── Initialize random population ──────────
    population = []
    for _ in range(population_size):
        individual = {
            "chaos_position": np.random.uniform(*bounds["chaos_position"]),
            "resilience": np.random.uniform(*bounds["resilience"]),
            "influence": np.random.uniform(*bounds["influence"]),
            "vitality": np.random.uniform(*bounds["vitality"]),
            "role": np.random.choice(roles),
        }
        individual["loss"] = _evaluate_individual(individual, observation, meme_events)
        population.append(individual)

    # ── Genetic algorithm loop ─────────────────
    best_loss_history = []
    for iteration in range(n_iterations):
        # Sort by fitness
        population.sort(key=lambda x: x["loss"])

        # Keep elite
        elite_size = max(3, population_size // 5)
        new_population = population[:elite_size]

        # Crossover + mutation for remaining
        while len(new_population) < population_size:
            parent1 = population[np.random.randint(0, population_size // 2)]
            parent2 = population[np.random.randint(0, population_size // 2)]

            child = {}
            for key in bounds:
                # Blend crossover
                alpha = np.random.random()
                child[key] = parent1[key] * alpha + parent2[key] * (1 - alpha)
                # Mutation
                if np.random.random() < 0.15:
                    lo, hi = bounds[key]
                    child[key] += np.random.normal(0, (hi - lo) * 0.1)
                    child[key] = np.clip(child[key], lo, hi)

            child["role"] = parent1["role"] if np.random.random() < 0.8 else np.random.choice(roles)
            child["loss"] = _evaluate_individual(child, observation, meme_events)
            new_population.append(child)

        population = new_population
        best_loss_history.append(population[0]["loss"])

    # ── Extract best and population statistics ──
    population.sort(key=lambda x: x["loss"])
    best = population[0]

    # Compute parameter uncertainties from top 20%
    top_n = max(5, population_size // 5)
    top_individuals = population[:top_n]

    chaos_vals = [ind["chaos_position"] for ind in top_individuals]
    resilience_vals = [ind["resilience"] for ind in top_individuals]
    influence_vals = [ind["influence"] for ind in top_individuals]
    role_counts = {}
    for ind in top_individuals:
        role_counts[ind["role"]] = role_counts.get(ind["role"], 0) + 1

    # ── Role probabilities ────────────────────
    role_probs = {r: role_counts.get(r, 0) / top_n for r in roles}
    most_likely_role = max(role_probs, key=role_probs.get)

    # ── Predict participation pattern ─────────
    chaos_est = float(np.mean(chaos_vals))
    if chaos_est > 0.3:
        predicted_pattern = "early_adopter"
    elif chaos_est > -0.3:
        predicted_pattern = "follower"
    else:
        predicted_pattern = "resister"

    # ── Chaos stability classification ────────
    chaos_std = float(np.std(chaos_vals))
    if chaos_std < 0.1:
        stability = "stable"
    elif chaos_std < 0.3:
        stability = "oscillating"
    else:
        stability = "drifting"

    # ── Confidence ────────────────────────────
    # Based on: loss_convergence, parameter agreement, data completeness
    loss_range = max(best_loss_history) - min(best_loss_history) + 1e-10
    convergence = 1.0 - (best_loss_history[-1] - min(best_loss_history)) / loss_range

    # Data completeness score
    n_signals = 0
    if observation.chaos_proxy_signals:
        n_signals += len(observation.chaos_proxy_signals)
    if observation.self_reported_chaos is not None:
        n_signals += 1
    if observation.meme_participation:
        n_signals += min(len(observation.meme_participation), 5)
    data_completeness = min(1.0, n_signals / 10)

    confidence = 0.3 * convergence + 0.4 * (1.0 - best["loss"]) + 0.3 * data_completeness
    confidence = np.clip(confidence, 0.1, 0.95)

    # ── Caveats ───────────────────────────────
    caveats = []
    if n_signals < 3:
        caveats.append("观测信号不足，校准结果高度不确定")
    if confidence < 0.5:
        caveats.append("置信度较低，建议收集更多行为数据后重新校准")
    if observation.self_reported_chaos is None:
        caveats.append("缺少自我报告锚点，混沌位置估计主要依赖代理信号")
    caveats.append("此剖面是对内部混沌结构的概率推断，不是确定性描述")
    caveats.append("其他主体的小真实永远是不可穿透的黑箱")

    return IndividualChaosProfile(
        chaos_position_est=round(chaos_est, 3),
        chaos_position_std=round(chaos_std, 3),
        resilience_est=round(float(np.mean(resilience_vals)), 3),
        influence_est=round(float(np.mean(influence_vals)), 3),
        role_probabilities={r: round(p, 3) for r, p in role_probs.items()},
        most_likely_role=most_likely_role,
        predicted_participation_pattern=predicted_pattern,
        susceptibility_est=round(0.5 + 0.3 * chaos_est, 3),
        chaos_stability=stability,
        confidence=round(confidence, 3),
        calibration_method=f"GeneticAlgorithm (pop={population_size}, iter={n_iterations})",
        caveats=caveats,
    )


def _evaluate_individual(params: dict, observation: BehavioralObservation,
                         meme_events: list[dict]) -> float:
    """评估单个参数组合的损失。"""
    sim_result = simulate_individual(params, n_steps=100, meme_events=meme_events)
    return _compute_loss(sim_result, observation)


# ═══════════════════════════════════════════════
# Scenario-based calibration (no real data needed)
# ═══════════════════════════════════════════════

def calibrate_from_scenario(scenario: str) -> IndividualChaosProfile:
    """从描述性场景合成观测并校准。

    用于在没有真实数据时进行示例推断。

    Args:
        scenario: "order_builder" | "chaos_injector" | "normal_follower" | "lurker" | "rapid_oscillator"

    Returns:
        IndividualChaosProfile
    """
    scenarios = {
        "order_builder": BehavioralObservation(
            meme_participation={
                "打工人": MemeParticipation.EARLY_ADOPTER,
                "内卷": MemeParticipation.EARLY_ADOPTER,
                "小镇做题家": MemeParticipation.EARLY_MAJORITY,
            },
            chaos_proxy_signals={
                "orderly_expression_ratio": 0.85,
                "contradiction_frequency": 0.10,
                "narrative_consistency": 0.80,
                "trolling_frequency": 0.02,
            },
            self_reported_chaos=0.5,
            self_reported_vitality=0.8,
        ),
        "chaos_injector": BehavioralObservation(
            meme_participation={
                "打工人": MemeParticipation.LAGGARD,
                "普信男": MemeParticipation.EARLY_ADOPTER,
                "XX刺客": MemeParticipation.EARLY_MAJORITY,
            },
            chaos_proxy_signals={
                "orderly_expression_ratio": 0.20,
                "contradiction_frequency": 0.70,
                "narrative_consistency": 0.25,
                "trolling_frequency": 0.60,
            },
            self_reported_chaos=-0.5,
            self_reported_vitality=0.5,
        ),
        "normal_follower": BehavioralObservation(
            meme_participation={
                "打工人": MemeParticipation.LATE_MAJORITY,
                "躺平": MemeParticipation.LATE_MAJORITY,
                "吗喽": MemeParticipation.LAGGARD,
            },
            chaos_proxy_signals={
                "orderly_expression_ratio": 0.50,
                "contradiction_frequency": 0.30,
                "narrative_consistency": 0.55,
                "trolling_frequency": 0.05,
            },
            self_reported_chaos=0.0,
            self_reported_vitality=0.6,
        ),
        "lurker": BehavioralObservation(
            meme_participation={
                "打工人": MemeParticipation.NEVER,
                "躺平": MemeParticipation.NEVER,
                "内卷": MemeParticipation.LAGGARD,
            },
            chaos_proxy_signals={
                "orderly_expression_ratio": 0.60,
                "contradiction_frequency": 0.10,
                "narrative_consistency": 0.70,
                "trolling_frequency": 0.01,
            },
            self_reported_chaos=0.1,
            self_reported_vitality=0.7,
        ),
    }

    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from: {list(scenarios.keys())}")

    return calibrate_from_observation(scenarios[scenario])


# ═══════════════════════════════════════════════
# Script entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("MemeticChaos — Individual Chaos Calibrator Demo")
    print("=" * 60)

    scenarios = ["order_builder", "chaos_injector", "normal_follower", "lurker"]

    for scenario in scenarios:
        print(f"\n{'─' * 50}")
        print(f"Scenario: {scenario}")
        print(f"{'─' * 50}")
        profile = calibrate_from_scenario(scenario)
        print(profile.summary())

    print(f"\n{'=' * 60}")
    print("Key principle:")
    print("  These profiles are PROBABILISTIC INFERENCES,")
    print("  not deterministic measurements.")
    print("  小真实永远是黑箱 — 我们只能越来越接近，")
    print("  但永远无法完全穿透。")

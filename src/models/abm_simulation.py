"""
ABM 多智能体情感混沌仿真 — MemeticChaos 建模层。

基于 Mesa 框架，模拟微观 Agent 交互如何涌现出宏观的集体情感混沌属性。

对齐「微尘哲学」核心元定律：
- Agent 在「绝对混沌(-1) ↔ 绝对秩序(+1)」轴上移动
- 模因传播 = Agent 间建立局部秩序的过程
- 熵增背景下，秩序不维护会自发衰退
- 健康秩序滋养 Agent 生命力，扭曲秩序导向两个敌人

核心机制：
1. 情感传染：相邻 Agent 互相影响混沌-秩序位置
2. 模因传播：SIR 机制在 Agent 网络中扩散
3. 回音壁效应：Agent 倾向于与相似位置的 Agent 连接
4. 熵增漂移：不加维护的秩序自发向混沌方向衰退
5. 混沌投放：部分 Agent 主动向邻居注入混沌
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
import random

try:
    from mesa import Agent, Model
    from mesa.time import RandomActivation
    from mesa.space import NetworkGrid
    from mesa.datacollection import DataCollector
    MESA_AVAILABLE = True
except ImportError:
    MESA_AVAILABLE = False


# ═══════════════════════════════════════════════
# Agent state definitions
# ═══════════════════════════════════════════════

class MemeState(Enum):
    """Agent 对当前模因的感染状态（SIR 层）。"""
    SUSCEPTIBLE = "S"   # 未接触梗，处于情感混沌态
    INFECTED = "I"      # 正在传播梗，参与局部秩序建立
    RECOVERED = "R"     # 对该梗免疫/厌倦


class ChaosRole(Enum):
    """Agent 在混沌动力学中的角色。"""
    NORMAL = "normal"           # 普通用户，被动受环境影响
    ORDER_BUILDER = "builder"   # 秩序建立者，主动传播梗/建立叙事
    CHAOS_INJECTOR = "injector" # 混沌投放者，主动破坏秩序
    LURKER = "lurker"           # 潜伏者，观察但不参与


# ═══════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════

class MemeAgent(Agent if MESA_AVAILABLE else object):
    """情感混沌 Agent。

    每个 Agent 代表一个网民，具有：
    - chaos_position: 在绝对混沌(-1) ↔ 绝对秩序(+1) 轴上的位置
    - meme_state: 对当前模因的 SIR 状态
    - vitality: 生命力（0-1，越低越接近系统瓦解）
    - resilience: 韧性（从混沌恢复秩序的能力）
    - influence: 影响力（对邻居的影响强度）
    """

    def __init__(self, unique_id, model,
                 chaos_position: float = None,
                 resilience: float = None,
                 influence: float = None,
                 role: ChaosRole = None):
        super().__init__(model) if MESA_AVAILABLE else setattr(self, "unique_id", unique_id)

        self.model = model
        self.unique_id = unique_id

        # Core state
        self.chaos_position = chaos_position if chaos_position is not None else np.random.uniform(-0.5, 0.5)
        self.meme_state = MemeState.SUSCEPTIBLE
        self.vitality = np.random.uniform(0.4, 1.0)

        # Traits
        self.resilience = resilience if resilience is not None else np.random.beta(3, 3)
        self.influence = influence if influence is not None else np.random.uniform(0.05, 0.3)
        self.role = role if role is not None else self._assign_role()

        # Tracking
        self.infected_at = None      # step when infected
        self.recovered_at = None     # step when recovered
        self.chaos_history = []      # trajectory of chaos_position
        self.vitality_history = []   # trajectory of vitality

    def _assign_role(self) -> ChaosRole:
        """基于概率分配角色。"""
        r = np.random.random()
        if r < 0.05:
            return ChaosRole.ORDER_BUILDER
        elif r < 0.10:
            return ChaosRole.CHAOS_INJECTOR
        elif r < 0.30:
            return ChaosRole.LURKER
        return ChaosRole.NORMAL

    @property
    def is_susceptible(self) -> bool:
        return self.meme_state == MemeState.SUSCEPTIBLE

    @property
    def is_infected(self) -> bool:
        return self.meme_state == MemeState.INFECTED

    @property
    def is_recovered(self) -> bool:
        return self.meme_state == MemeState.RECOVERED

    # ── Actions ────────────────────────────────

    def step(self):
        """每个时间步的行为。"""
        if self.role == ChaosRole.LURKER:
            # Lurkers observe but don't act
            self._record_state()
            return

        # 1. 熵增漂移：不加维护的秩序自发衰退
        self._entropy_drift()

        # 2. 情感传染：受邻居影响
        self._emotional_contagion()

        # 3. 模因传播：SIR 状态转换
        self._meme_step()

        # 4. 生命力更新
        self._update_vitality()

        # 5. 混沌投放（如果角色是 injector）
        if self.role == ChaosRole.CHAOS_INJECTOR:
            self._chaos_injection()

        # Clamp values
        self.chaos_position = np.clip(self.chaos_position, -1.0, 1.0)
        self.vitality = np.clip(self.vitality, 0.0, 1.0)

        self._record_state()

    def _entropy_drift(self):
        """熵增漂移：不维护则向混沌方向衰退。

        漂移速率 = entropy_rate * (1 - resilience)
        高韧性 Agent 能更好地抵抗熵增。
        """
        drift_rate = self.model.entropy_rate * (1.0 - self.resilience)

        # Order builders resist drift even more
        if self.role == ChaosRole.ORDER_BUILDER:
            drift_rate *= 0.3

        # Drift toward chaos (negative direction)
        self.chaos_position -= drift_rate * self.chaos_position * 0.1

    def _emotional_contagion(self):
        """情感传染：邻居的混沌-秩序位置影响自身。

        与邻居位置差异的加权平均拉近自身位置。
        同质偏好：Agent 更容易被与自己位置相近的邻居影响（回音壁效应）。
        """
        neighbors = self._get_neighbors()
        if not neighbors:
            return

        contagion_strength = self.model.contagion_rate

        weighted_influence = 0.0
        total_weight = 0.0

        for neighbor in neighbors:
            # Echo chamber: closer positions → stronger influence
            similarity = 1.0 - abs(self.chaos_position - neighbor.chaos_position) / 2.0
            weight = similarity * neighbor.influence
            weighted_influence += weight * neighbor.chaos_position
            total_weight += weight

        if total_weight > 0:
            avg_neighbor_chaos = weighted_influence / total_weight
            self.chaos_position += contagion_strength * (avg_neighbor_chaos - self.chaos_position)

    def _meme_step(self):
        """SIR 状态转换。"""
        if self.meme_state == MemeState.SUSCEPTIBLE:
            # Check infection from infected neighbors
            infected_neighbors = [n for n in self._get_neighbors() if n.is_infected]
            if infected_neighbors:
                # Infection probability depends on number of infected neighbors
                # and how "order-seeking" the agent is (more orderly = more susceptible to good memes)
                p_infect = min(1.0, len(infected_neighbors) * self.model.infection_rate *
                              (0.5 + 0.5 * max(0, self.chaos_position)))  # order-leaning agents more receptive
                if np.random.random() < p_infect:
                    self.meme_state = MemeState.INFECTED
                    self.infected_at = self.model.schedule.steps
                    # Being infected pulls agent slightly toward order
                    self.chaos_position += 0.1 * (1.0 - abs(self.chaos_position))

        elif self.meme_state == MemeState.INFECTED:
            # Recovery: boredom/immunity develops
            time_infected = self.model.schedule.steps - (self.infected_at or 0)
            p_recover = 1.0 - np.exp(-self.model.recovery_rate * time_infected)
            if np.random.random() < p_recover:
                self.meme_state = MemeState.RECOVERED
                self.recovered_at = self.model.schedule.steps

        elif self.meme_state == MemeState.RECOVERED:
            # Mutation/re-infection (SIRS mechanism)
            time_recovered = self.model.schedule.steps - (self.recovered_at or 0)
            p_resusceptible = 1.0 - np.exp(-self.model.mutation_rate * time_recovered)
            if np.random.random() < p_resusceptible:
                self.meme_state = MemeState.SUSCEPTIBLE

    def _update_vitality(self):
        """更新生命力。

        极端位置（|chaos| → 1）损害生命力。
        健康秩序（适度偏秩序 + 被感染参与）滋养生命力。
        """
        # Penalty for extreme positions
        extremity_penalty = abs(self.chaos_position) ** 2 * 0.01

        # Bonus for being in a meme community (participating in order building)
        infection_bonus = 0.005 if self.is_infected else 0.0

        # Resilience helps maintain vitality
        resilience_bonus = self.resilience * 0.002

        self.vitality += resilience_bonus + infection_bonus - extremity_penalty

    def _chaos_injection(self):
        """混沌投放：向邻居注入混沌。

        Chaos injectors actively push neighbors toward chaos.
        这是「三种恶意模型」在 ABM 中的实现。
        """
        neighbors = self._get_neighbors()
        injection_power = self.influence * self.model.chaos_injection_strength

        for neighbor in neighbors:
            if neighbor.role != ChaosRole.CHAOS_INJECTOR:
                # Push neighbor toward chaos
                push = injection_power * np.random.random()
                # More effective on already-chaotic agents
                if neighbor.chaos_position < 0:
                    push *= 1.5
                neighbor.chaos_position -= push
                neighbor.vitality -= push * 0.5

    def _get_neighbors(self) -> list:
        """获取网络中的邻居 Agent。"""
        if hasattr(self.model, 'grid') and MESA_AVAILABLE:
            return self.model.grid.get_neighbors(self.unique_id, include_center=False)
        elif hasattr(self.model, 'get_neighbors'):
            return self.model.get_neighbors(self)
        return []

    def _record_state(self):
        """记录当前状态用于后续分析。"""
        self.chaos_history.append(self.chaos_position)
        self.vitality_history.append(self.vitality)


# ═══════════════════════════════════════════════
# Network builders (for Mesa-less fallback or custom use)
# ═══════════════════════════════════════════════

def build_random_network(n_agents: int, avg_degree: int = 8) -> list[list[int]]:
    """建立随机网络（Erdos-Renyi 风格）。

    Args:
        n_agents: Agent 数量
        avg_degree: 平均度数

    Returns:
        邻接表：adj[i] = [邻居ID列表]
    """
    p = avg_degree / (n_agents - 1) if n_agents > 1 else 0
    adj = [[] for _ in range(n_agents)]
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if np.random.random() < p:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def build_scale_free_network(n_agents: int, m: int = 3) -> list[list[int]]:
    """建立无标度网络（Barabasi-Albert）。

    更接近真实社交网络：少数节点有大量连接，多数节点连接少。

    Args:
        n_agents: Agent 数量
        m: 每个新节点添加的边数

    Returns:
        邻接表
    """
    import networkx as nx
    G = nx.barabasi_albert_graph(n_agents, m)
    adj = [[] for _ in range(n_agents)]
    for u, v in G.edges():
        adj[u].append(v)
        adj[v].append(u)
    return adj


# ═══════════════════════════════════════════════
# Standalone simulation (Mesa-less)
# ═══════════════════════════════════════════════

@dataclass
class ABMConfig:
    """ABM 仿真配置。"""
    n_agents: int = 500
    n_steps: int = 200
    avg_degree: int = 8

    # Dynamics parameters
    contagion_rate: float = 0.05        # 情感传染强度 (lower = more diversity)
    infection_rate: float = 0.12        # 模因感染率
    recovery_rate: float = 0.04         # 恢复/厌倦率
    mutation_rate: float = 0.008        # 变异/复燃率
    entropy_rate: float = 0.015         # 熵增漂移率
    chaos_injection_strength: float = 0.03  # 混沌投放强度
    noise_level: float = 0.005          # 个体随机扰动（防止过度趋同）

    # Initial conditions
    initial_infected_fraction: float = 0.01
    initial_order_builders: int = 5
    initial_chaos_injectors: int = 5

    # Network type
    network_type: str = "scale_free"  # "random" | "scale_free"

    # Recording
    record_interval: int = 1  # record every N steps


@dataclass
class ABMResult:
    """ABM 仿真结果。"""
    config: ABMConfig
    # Aggregate time series
    mean_chaos: np.ndarray          # 每步平均混沌位置
    std_chaos: np.ndarray           # 每步混沌标准差
    infected_fraction: np.ndarray   # 每步感染比例
    mean_vitality: np.ndarray       # 每步平均生命力
    entropy: np.ndarray             # 每步系统香农熵

    # Final state
    final_chaos_distribution: np.ndarray  # 最终混沌位置分布
    final_meme_state_counts: dict         # 最终 SIR 状态计数
    peak_infected: float                  # 峰值感染率
    peak_step: int                        # 峰值时间步

    # Raw trajectories (sampled)
    sample_agents: list[dict]      # 若干 Agent 的完整轨迹


def run_simulation(config: ABMConfig = None) -> ABMResult:
    """运行无 Mesa 依赖的独立 ABM 仿真。

    当 Mesa 不可用或需要更快的仿真时使用。

    Args:
        config: ABMConfig

    Returns:
        ABMResult
    """
    if config is None:
        config = ABMConfig()

    np.random.seed(42)

    # ── Build network ──────────────────────────
    if config.network_type == "scale_free":
        adj = build_scale_free_network(config.n_agents, m=config.avg_degree // 2)
    else:
        adj = build_random_network(config.n_agents, config.avg_degree)

    # ── Initialize agents ──────────────────────
    n = config.n_agents

    # Roles: assign explicitly then create agents
    roles = [ChaosRole.NORMAL] * n
    # Order builders
    builder_ids = np.random.choice(n, config.initial_order_builders, replace=False)
    for bid in builder_ids:
        roles[bid] = ChaosRole.ORDER_BUILDER
    # Chaos injectors
    remaining = [i for i in range(n) if roles[i] == ChaosRole.NORMAL]
    injector_ids = np.random.choice(remaining, config.initial_chaos_injectors, replace=False)
    for iid in injector_ids:
        roles[iid] = ChaosRole.CHAOS_INJECTOR

    # Agent states
    chaos_positions = np.random.uniform(-0.3, 0.3, n)
    for bid in builder_ids:
        chaos_positions[bid] = np.random.uniform(0.3, 0.7)   # builders start orderly
    for iid in injector_ids:
        chaos_positions[iid] = np.random.uniform(-0.7, -0.3) # injectors start chaotic

    meme_states = np.array([MemeState.SUSCEPTIBLE] * n)
    # Initial infected
    n_initial = max(1, int(n * config.initial_infected_fraction))
    initial_infected = np.random.choice(n, n_initial, replace=False)
    for iid in initial_infected:
        meme_states[iid] = MemeState.INFECTED

    vitalities = np.random.uniform(0.5, 1.0, n)
    resiliences = np.random.beta(3, 3, n)
    influences = np.random.uniform(0.05, 0.3, n)
    # Builders have higher influence
    for bid in builder_ids:
        influences[bid] *= 2.0

    infected_since = np.full(n, -1, dtype=int)
    recovered_since = np.full(n, -1, dtype=int)
    for iid in initial_infected:
        infected_since[iid] = 0

    # ── Recording arrays ───────────────────────
    steps = config.n_steps
    record_every = config.record_interval
    max_records = steps // record_every + 1

    mean_chaos_rec = []
    std_chaos_rec = []
    infected_fraction_rec = []
    mean_vitality_rec = []
    entropy_rec = []

    # Sample agents for trajectory recording
    sample_ids = np.random.choice(n, min(10, n), replace=False)
    sample_trajectories = {sid: {"chaos": [], "vitality": [], "meme": []} for sid in sample_ids}

    # ── Main simulation loop ───────────────────
    for step in range(steps):
        # --- Store previous meme states ---
        prev_meme = meme_states.copy()

        # --- 1. Entropy drift ---
        drift = config.entropy_rate * (1.0 - resiliences)
        drift[builder_ids] *= 0.3  # builders resist better
        chaos_positions -= drift * chaos_positions * 0.1

        # --- 2. Emotional contagion ---
        for i in range(n):
            neighbors_i = adj[i]
            if not neighbors_i:
                continue

            # Echo chamber: closer positions → stronger influence
            similarities = 1.0 - np.abs(chaos_positions[i] - chaos_positions[neighbors_i]) / 2.0
            weights = similarities * influences[neighbors_i]
            total_weight = weights.sum()
            if total_weight > 0:
                avg_neighbor_chaos = np.sum(weights * chaos_positions[neighbors_i]) / total_weight
                chaos_positions[i] += config.contagion_rate * (avg_neighbor_chaos - chaos_positions[i])

        # --- 3. Meme spread (SIR) ---
        # S → I
        susceptible_mask = (meme_states == MemeState.SUSCEPTIBLE)
        for i in np.where(susceptible_mask)[0]:
            infected_neighbors = [j for j in adj[i] if meme_states[j] == MemeState.INFECTED]
            if infected_neighbors:
                order_bias = 0.5 + 0.5 * max(0, chaos_positions[i])
                p_infect = min(1.0, len(infected_neighbors) * config.infection_rate * order_bias)
                if np.random.random() < p_infect:
                    meme_states[i] = MemeState.INFECTED
                    infected_since[i] = step
                    chaos_positions[i] += 0.1 * (1.0 - abs(chaos_positions[i]))

        # I → R
        infected_mask = (meme_states == MemeState.INFECTED)
        for i in np.where(infected_mask)[0]:
            time_infected = step - infected_since[i]
            p_recover = 1.0 - np.exp(-config.recovery_rate * time_infected)
            if np.random.random() < p_recover:
                meme_states[i] = MemeState.RECOVERED
                recovered_since[i] = step

        # R → S (mutation/recurrence)
        recovered_mask = (meme_states == MemeState.RECOVERED)
        for i in np.where(recovered_mask)[0]:
            time_recovered = step - recovered_since[i]
            p_resusceptible = 1.0 - np.exp(-config.mutation_rate * time_recovered)
            if np.random.random() < p_resusceptible:
                meme_states[i] = MemeState.SUSCEPTIBLE

        # --- 4. Vitality update ---
        extremity_penalty = np.abs(chaos_positions) ** 2 * 0.01
        infection_bonus = np.where(meme_states == MemeState.INFECTED, 0.005, 0.0)
        resilience_bonus = resiliences * 0.002
        vitalities += resilience_bonus + infection_bonus - extremity_penalty

        # --- 5. Chaos injection ---
        for iid in injector_ids:
            if roles[iid] != ChaosRole.CHAOS_INJECTOR:
                continue
            for neighbor in adj[iid]:
                if roles[neighbor] != ChaosRole.CHAOS_INJECTOR:
                    push = influences[iid] * config.chaos_injection_strength * np.random.random()
                    if chaos_positions[neighbor] < 0:
                        push *= 1.3
                    chaos_positions[neighbor] -= push
                    # Softer vitality penalty for injection
                    vitalities[neighbor] -= push * 0.15

        # --- 6. Individual noise (prevents homogenization) ---
        chaos_positions += np.random.normal(0, config.noise_level, n)
        vitalities += np.random.normal(0, config.noise_level * 0.5, n)

        # --- Clamp ---
        chaos_positions = np.clip(chaos_positions, -1.0, 1.0)
        vitalities = np.clip(vitalities, 0.0, 1.0)

        # --- Record ---
        if step % record_every == 0:
            mean_chaos_rec.append(float(np.mean(chaos_positions)))
            std_chaos_rec.append(float(np.std(chaos_positions)))
            infected_fraction_rec.append(float(np.mean(meme_states == MemeState.INFECTED)))
            mean_vitality_rec.append(float(np.mean(vitalities)))

            # Shannon entropy of chaos distribution
            n_bins = 10
            hist, _ = np.histogram(chaos_positions, bins=n_bins, range=(-1, 1))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            if len(hist) > 1:
                entropy_rec.append(float(-np.sum(hist * np.log(hist)) / np.log(n_bins)))
            else:
                entropy_rec.append(0.0)

        # --- Sample trajectories ---
        for sid in sample_ids:
            sample_trajectories[sid]["chaos"].append(float(chaos_positions[sid]))
            sample_trajectories[sid]["vitality"].append(float(vitalities[sid]))
            sample_trajectories[sid]["meme"].append(meme_states[sid].value)

    # ── Finalize results ───────────────────────
    mean_chaos_arr = np.array(mean_chaos_rec)
    infected_frac_arr = np.array(infected_fraction_rec)
    peak_idx = np.argmax(infected_frac_arr)
    final_state_counts = {
        "S": int(np.sum(meme_states == MemeState.SUSCEPTIBLE)),
        "I": int(np.sum(meme_states == MemeState.INFECTED)),
        "R": int(np.sum(meme_states == MemeState.RECOVERED)),
    }

    return ABMResult(
        config=config,
        mean_chaos=mean_chaos_arr,
        std_chaos=np.array(std_chaos_rec),
        infected_fraction=infected_frac_arr,
        mean_vitality=np.array(mean_vitality_rec),
        entropy=np.array(entropy_rec),
        final_chaos_distribution=chaos_positions.copy(),
        final_meme_state_counts=final_state_counts,
        peak_infected=float(infected_frac_arr[peak_idx]),
        peak_step=int(peak_idx * record_every),
        sample_agents=[
            {"id": int(sid), "chaos_traj": traj["chaos"],
             "vitality_traj": traj["vitality"], "meme_traj": traj["meme"]}
            for sid, traj in sample_trajectories.items()
        ],
    )


# ═══════════════════════════════════════════════
# Parameter sweeps
# ═══════════════════════════════════════════════

def sweep_contagion(contagion_values: list[float],
                    base_config: ABMConfig = None) -> list[ABMResult]:
    """扫描情感传染率，观察涌现行为变化。

    Args:
        contagion_values: 传染率值列表
        base_config: 基础配置

    Returns:
        ABMResult 列表
    """
    results = []
    for rate in contagion_values:
        config = base_config or ABMConfig()
        config.contagion_rate = rate
        result = run_simulation(config)
        results.append(result)
    return results


def sweep_chaos_injection(strengths: list[float],
                          base_config: ABMConfig = None) -> list[ABMResult]:
    """扫描混沌投放强度。"""
    results = []
    for s in strengths:
        config = base_config or ABMConfig()
        config.chaos_injection_strength = s
        result = run_simulation(config)
        results.append(result)
    return results


# ═══════════════════════════════════════════════
# Script entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("MemeticChaos — ABM Simulation Demo")
    print("=" * 60)

    # Default simulation
    config = ABMConfig(n_agents=300, n_steps=150)
    print(f"\nConfig: {config.n_agents} agents, {config.n_steps} steps")
    print(f"Network: {config.network_type}")
    print(f"Initial: {config.initial_infected_fraction:.1%} infected, "
          f"{config.initial_order_builders} builders, "
          f"{config.initial_chaos_injectors} injectors")

    result = run_simulation(config)

    print(f"\n--- Results ---")
    print(f"Peak infected: {result.peak_infected:.1%} at step {result.peak_step}")
    print(f"Final mean chaos: {result.mean_chaos[-1]:+.4f}")
    print(f"Final mean vitality: {result.mean_vitality[-1]:.4f}")
    print(f"Final entropy: {result.entropy[-1]:.4f}")
    print(f"Final SIR: {result.final_meme_state_counts}")

    print(f"\n--- Chaos Distribution at Final Step ---")
    hist, bins = np.histogram(result.final_chaos_distribution, bins=10, range=(-1, 1))
    max_count = max(hist)
    for i, (count, lo, hi) in enumerate(zip(hist, bins[:-1], bins[1:])):
        bar = "█" * int(30 * count / max_count)
        print(f"  [{lo:+.1f}, {hi:+.1f}): {count:4d} {bar}")

    # Sample agent trajectories
    print(f"\n--- Sample Agent: Order Builder ---")
    builder = next((a for a in result.sample_agents
                   if any(c > 0.3 for c in a["chaos_traj"][:5])), result.sample_agents[0])
    if builder:
        print(f"  Initial chaos: {builder['chaos_traj'][0]:+.3f} → "
              f"Final: {builder['chaos_traj'][-1]:+.3f}")

    # Contagion sweep
    print(f"\n--- Contagion Rate Sweep ---")
    contagion_results = sweep_contagion([0.02, 0.05, 0.10, 0.20, 0.30])
    for r, res in zip([0.02, 0.05, 0.10, 0.20, 0.30], contagion_results):
        print(f"  rate={r:.2f}: peak_I={res.peak_infected:.1%}, "
              f"final_chaos={res.mean_chaos[-1]:+.3f}, "
              f"final_entropy={res.entropy[-1]:.3f}")

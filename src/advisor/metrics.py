"""
FR31 三指标接口 — Inertia / Resilience / Position

从 FR19 的 10 维 Narrative State x(t) 计算三个控制论指标,
供 FR31 顾问或 OpenClaw (Stella) 查询。

- Inertia (惯性): 系统自持性 — "踩刹车也停不下来"的程度
- Resilience (恢复力): 冲击后回到吸引子的速度
- Position (图位置): 当前状态在叙事演化图中的拓扑位置

这是 FR19 → FR31 的桥梁。不预测值——描述结构。

AlphaGo 原则: 指标定义完全由数据的统计属性决定,不导入人类叙事理论。

用法:
    python src/advisor/metrics.py                    # 输出当前指标
    python src/advisor/metrics.py --month 2025-06   # 指定月份
    python src/advisor/metrics.py --history          # 全历史序列
    python src/advisor/metrics.py --json             # JSON 输出 (供 API)
"""

import json, sys, os, argparse
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

ROOT = Path(__file__).parent.parent.parent

STATE_PATH = ROOT / "data/processed/representation_state.json"
LEVEL1_PATH = ROOT / "data/processed/level1_hard_facts.json"


class NarrativeMetrics:
    """FR31 三指标计算引擎."""

    def __init__(self):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            self.state_data = json.load(f)
        with open(LEVEL1_PATH, "r", encoding="utf-8") as f:
            self.l1_data = json.load(f)

        self.months = self.state_data["pca_transformed"]["months"]
        self.x = np.array(self.state_data["pca_transformed"]["x_reduced"])  # n × 10
        self.stage = np.array(self.l1_data["stage_occupancy"])               # n × 5
        self.stage_names = self.l1_data["stages"]
        self.mut_rate = np.array(self.l1_data["mutation_rate"])
        self.inst_rate = np.array(self.l1_data["institutionalization_rate"])
        self.drift = np.array(self.l1_data["mean_semantic_drift"])
        self.active_count = np.array(self.l1_data["active_meme_count"])
        self.total_traffic = np.array(self.l1_data["total_traffic"])

        self.n = len(self.months)

        # Precompute derived quantities
        self._precompute()

    def _precompute(self):
        """预计算: 状态变化幅度, 滚动统计量."""
        # Month-to-month state change magnitude
        self.dx_magnitude = np.zeros(self.n)
        for i in range(1, self.n):
            self.dx_magnitude[i] = float(np.linalg.norm(self.x[i] - self.x[i - 1]))

        # Shock threshold: 2σ of dx
        self.shock_threshold = float(2.0 * np.std(self.dx_magnitude[1:]))

        # Shock months
        self.shock_months = set()
        for i in range(1, self.n):
            if self.dx_magnitude[i] > self.shock_threshold:
                self.shock_months.add(i)

        # Rolling autocorrelation (12-month window)
        self.rolling_lag1_corr = np.full(self.n, np.nan)
        for i in range(13, self.n):
            window = self.x[i - 12:i]
            # Correlation between consecutive pairs
            if len(window) >= 3:
                pairs = [(window[j], window[j + 1]) for j in range(len(window) - 1)]
                a = np.array([p[0] for p in pairs]).ravel()
                b = np.array([p[1] for p in pairs]).ravel()
                if np.std(a) > 1e-10 and np.std(b) > 1e-10:
                    self.rolling_lag1_corr[i] = float(np.corrcoef(a, b)[0, 1])
                else:
                    self.rolling_lag1_corr[i] = 0.0

        # Equilibrium reference: mean state over last 24 months
        self.equilibrium = np.mean(self.x[max(0, self.n - 24):], axis=0)

    # ═══════════════════════════════════════════════
    # Inertia (惯性)
    # ═══════════════════════════════════════════════

    def inertia(self, month: str | None = None) -> dict:
        """计算系统惯性.

        高惯性 = 系统"卡住"了 — 大量注意力锁在 fixation 阶段,
        新梗难以涌现, 状态变化缓慢.

        指标:
        - stage_inertia: fixation% / (origin% + emergence%), 归一化
        - dynamic_inertia: 当前 lag-1 自相关系数 (高 = 变化慢)
        - traffic_inertia: 总注意力中 fixated memes 的占比
        - composite: 三者的加权组合, [0, 1] 标度
        """
        idx = self._get_index(month)

        # Stage-based: fixation dominates, origin/emergence low
        s = self.stage[idx]
        origin_emerge = max(s[0] + s[1], 0.01)
        fixation = s[4]
        stage_inertia = float(np.clip(fixation / (origin_emerge + fixation + 0.05), 0, 1))

        # Dynamic: rolling lag-1 autocorrelation
        dyn = self.rolling_lag1_corr[idx]
        if np.isnan(dyn):
            dyn = float(np.corrcoef(self.x[:-1].ravel(), self.x[1:].ravel())[0, 1])
        dynamic_inertia = float(np.clip(max(0, dyn), 0, 1))

        # Traffic: proportion of total attention in fixation
        traffic_inertia = float(fixation)

        # Composite
        composite = float(np.clip(
            0.4 * stage_inertia + 0.3 * dynamic_inertia + 0.3 * traffic_inertia, 0, 1))

        # Human-readable interpretation
        if composite > 0.65:
            label = "高惯性 — 叙事生态僵化, 新梗难以破圈, '踩刹车也停不下来'"
            risk = "注意: 系统可能接近相变临界点 — 长期僵化后易发生突然重构"
        elif composite > 0.40:
            label = "中等惯性 — 叙事处于动态平衡, 新旧交替正常进行"
            risk = None
        else:
            label = "低惯性 — 叙事生态活跃, 新梗快速涌现, 系统易变"
            risk = None

        return {
            "metric": "inertia",
            "month": self.months[idx],
            "value": round(composite, 4),
            "components": {
                "stage_inertia": round(stage_inertia, 4),
                "dynamic_inertia": round(dynamic_inertia, 4),
                "traffic_inertia": round(traffic_inertia, 4),
            },
            "stage_profile": {n: round(float(s[i]), 3) for i, n in enumerate(self.stage_names)},
            "interpretation": label,
            "risk_warning": risk,
        }

    # ═══════════════════════════════════════════════
    # Resilience (恢复力)
    # ═══════════════════════════════════════════════

    def resilience(self, month: str | None = None) -> dict:
        """计算系统恢复力.

        低 resilience = 系统脆弱 — 一次冲击后需要很久才能回到平衡,
        或从未真正恢复 (随机游走).

        指标:
        - recovery_speed: 历史冲击的平均恢复时间 (月)
        - current_fragility: 当前状态距平衡有多远 (σ 单位)
        - shock_frequency: 最近 24 月内冲击次数
        - eigenvalue_decay: 经验转移矩阵的主特征值 (λ < 1 = 均值回归)
        """
        idx = self._get_index(month)

        # Recovery speed from historical shocks
        recovery_times = []
        for si in sorted(self.shock_months):
            if si >= self.n - 3:
                continue
            pre_shock = self.x[si - 1] if si > 0 else self.x[si]
            for k in range(si + 1, min(si + 18, self.n)):
                if np.linalg.norm(self.x[k] - pre_shock) < self.shock_threshold * 0.5:
                    recovery_times.append(k - si)
                    break

        if recovery_times:
            mean_recovery = float(np.mean(recovery_times))
            median_recovery = float(np.median(recovery_times))
        else:
            mean_recovery = 12.0  # default: long recovery
            median_recovery = 12.0

        recovery_speed = float(np.clip(1.0 - (mean_recovery / 12.0), 0, 1))

        # Current fragility: distance from equilibrium in σ units
        eq_dist = float(np.linalg.norm(self.x[idx] - self.equilibrium))
        base_std = float(np.std([np.linalg.norm(self.x[i] - self.equilibrium)
                                  for i in range(max(0, self.n - 24), self.n)]))
        current_fragility = float(np.clip(eq_dist / max(base_std, 1e-10), 0, 5))

        # Shock frequency in last 24 months
        recent_shocks = sum(1 for si in self.shock_months
                            if si >= max(0, idx - 24) and si <= idx)

        # Eigenvalue analysis: fit AR(1) to reduced state
        # Regress x(t) on x(t-1): x(t) = A·x(t-1)
        if self.n > 3:
            X_lag = self.x[:-1]
            y_next = self.x[1:]
            # Least squares: A = (X'X)^(-1) X'Y
            try:
                A = np.linalg.lstsq(X_lag, y_next, rcond=None)[0]
                eigenvalues = np.linalg.eigvals(A)
                max_abs_eig = float(max(np.abs(eigenvalues)))
            except Exception:
                max_abs_eig = 0.95  # near random walk
        else:
            max_abs_eig = 0.95

        eigenvalue_stability = float(np.clip(1.0 - max_abs_eig, 0, 1))

        # Composite
        composite = float(np.clip(
            0.3 * recovery_speed + 0.3 * (1.0 - min(current_fragility / 3.0, 1.0))
            + 0.2 * (1.0 - recent_shocks / 6.0) + 0.2 * eigenvalue_stability, 0, 1))

        if composite < 0.30:
            label = "低恢复力 — 系统脆弱, 冲击后恢复慢, 远离平衡"
            risk = "系统可能处于非平衡态, 对外部冲击敏感"
        elif composite > 0.60:
            label = "高恢复力 — 系统稳健, 冲击后快速回到吸引子"
            risk = None
        else:
            label = "中等恢复力 — 有一定抗冲击能力, 但偏离过远可能难以恢复"
            risk = None

        return {
            "metric": "resilience",
            "month": self.months[idx],
            "value": round(composite, 4),
            "components": {
                "recovery_speed": round(recovery_speed, 4),
                "mean_recovery_months": round(mean_recovery, 1),
                "current_fragility_sigma": round(current_fragility, 2),
                "recent_shocks_24m": recent_shocks,
                "eigenvalue_max_abs": round(max_abs_eig, 4),
            },
            "interpretation": label,
            "risk_warning": risk,
        }

    # ═══════════════════════════════════════════════
    # Position (图位置)
    # ═══════════════════════════════════════════════

    def position(self, month: str | None = None,
                 persona_profile: dict | None = None) -> dict:
        """返回当前状态在叙事演化图中的拓扑位置.

        大盘模式:
        - stage_landscape: 当前五阶段占比
        - dominant_regime: 主导阶段 + 次导阶段
        - narrative_temperature: 叙事活跃度 (mutation_rate + drift 综合)
        - pc_projection: 在 PC1/PC2 主变异轴上的投影
        - nearest_attractor: 最近的历史吸引子 (K-means 中心)

        个体定位 (若给定 persona_profile):
        - individual_position: 个体在叙事图中的坐标 (待 FR31 persona.py 完成)
        """
        idx = self._get_index(month)

        # Stage landscape
        s = self.stage[idx]
        stage_profile = {n: round(float(s[i]), 3)
                         for i, n in enumerate(self.stage_names)}
        dominant_idx = int(np.argmax(s))
        second_idx = int(np.argsort(s)[-2])

        # Narrative temperature: mutation rate + semantic drift
        temp = float(np.clip(
            0.5 * self.mut_rate[idx] / max(np.max(self.mut_rate), 0.01)
            + 0.5 * self.drift[idx] / max(np.max(self.drift), 0.01), 0, 1))

        # PC projection
        pc1 = float(self.x[idx, 0])
        pc2 = float(self.x[idx, 1])

        # Dominant regime
        regime_label = f"{self.stage_names[dominant_idx]}主导 + {self.stage_names[second_idx]}次导"

        # Find nearest attractor: cluster historical states into basins
        from sklearn.cluster import KMeans
        k = min(5, self.n // 12)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(self.x)
        cluster = int(km.predict(self.x[idx:idx + 1])[0])
        cluster_size = int(np.sum(km.labels_ == cluster))
        cluster_pct = round(cluster_size / self.n * 100, 1)

        # Distance from cluster center
        center_dist = float(np.linalg.norm(self.x[idx] - km.cluster_centers_[cluster]))
        avg_dist = float(np.mean([
            np.linalg.norm(self.x[i] - km.cluster_centers_[km.labels_[i]])
            for i in range(self.n) if km.labels_[i] == cluster
        ]))

        # Closest phase transition
        # Find months where dominant stage changed
        transitions = []
        for i in range(1, self.n):
            prev_dom = int(np.argmax(self.stage[i - 1]))
            curr_dom = int(np.argmax(self.stage[i]))
            if prev_dom != curr_dom:
                transitions.append(i)

        # Count transitions in last 12 months
        recent_ts = sum(1 for t in transitions if idx - 12 <= t <= idx)

        # Individual position (placeholder)
        individual = None
        if persona_profile:
            individual = {
                "status": "placeholder",
                "note": "个体定位需要 FR31 persona.py 完成。当前仅返回大盘位置。",
                "profile_dimensions": list(persona_profile.keys()) if persona_profile else [],
            }

        return {
            "metric": "position",
            "month": self.months[idx],
            "stage_landscape": stage_profile,
            "dominant_regime": regime_label,
            "narrative_temperature": round(temp, 3),
            "pc_projection": {"pc1": round(pc1, 3), "pc2": round(pc2, 3)},
            "attractor_basin": {
                "cluster_id": cluster,
                "cluster_size_pct": cluster_pct,
                "distance_from_center": round(center_dist, 3),
                "is_typical": center_dist < avg_dist * 1.5,
            },
            "phase_transitions_12m": recent_ts,
            "individual_position": individual,
            "summary": (
                f"当前叙事图位置: {regime_label}. "
                f"叙事温度: {'活跃' if temp > 0.5 else '温和'}. "
                f"位于吸引子盆地 #{cluster} ({cluster_pct}% 历史月份在此), "
                f"{'典型' if center_dist < avg_dist * 1.5 else '边缘'}位置. "
                f"近 12 月 {recent_ts} 次阶段转换."
            ),
        }

    # ═══════════════════════════════════════════════
    # Dashboard summary
    # ═══════════════════════════════════════════════

    def summary(self, month: str | None = None) -> dict:
        """三指标综合报告."""
        idx = self._get_index(month)
        return {
            "month": self.months[idx],
            "inertia": self.inertia(month),
            "resilience": self.resilience(month),
            "position": self.position(month),
            "raw_state": {
                "active_memes": int(self.active_count[idx]),
                "total_traffic": round(float(self.total_traffic[idx]), 1),
                "mutation_rate": round(float(self.mut_rate[idx]), 3),
                "inst_rate": round(float(self.inst_rate[idx]), 3),
                "mean_drift": round(float(self.drift[idx]), 3),
            },
        }

    # ═══════════════════════════════════════════════
    # History
    # ═══════════════════════════════════════════════

    def history(self) -> list[dict]:
        """全历史三指标序列."""
        results = []
        for i in range(self.n):
            m = self.months[i]
            results.append({
                "month": m,
                "inertia": round(self._inertia_scalar(i), 4),
                "resilience": round(self._resilience_scalar(i), 4),
                "active_count": int(self.active_count[i]),
                "dominant_stage": self.stage_names[int(np.argmax(self.stage[i]))],
            })
        return results

    # ── Fast scalar versions for history ──

    def _inertia_scalar(self, idx: int) -> float:
        s = self.stage[idx]
        oe = max(s[0] + s[1], 0.01)
        return float(np.clip(s[4] / (oe + s[4] + 0.05), 0, 1))

    def _resilience_scalar(self, idx: int) -> float:
        eq_dist = float(np.linalg.norm(self.x[idx] - self.equilibrium))
        base = float(np.std([np.linalg.norm(self.x[i] - self.equilibrium)
                              for i in range(max(0, idx - 24), idx + 1)]))
        return float(np.clip(1.0 - eq_dist / max(base, 1e-10) / 3.0, 0, 1))

    # ── Helpers ──

    def _get_index(self, month: str | None) -> int:
        if month is None:
            return self.n - 1  # latest
        if month not in self.months:
            raise ValueError(f"Month {month} not found. Range: {self.months[0]}–{self.months[-1]}")
        return self.months.index(month)


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FR31 三指标: Inertia / Resilience / Position")
    parser.add_argument("--month", type=str, default=None, help="指定月份 (默认: 最新)")
    parser.add_argument("--history", action="store_true", help="输出全历史序列")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    metrics = NarrativeMetrics()

    if args.history:
        hist = metrics.history()
        if args.json:
            print(json.dumps(hist, ensure_ascii=False, indent=2))
        else:
            print(f"{'Month':<10s} {'Inertia':>8s} {'Resilience':>10s} {'Active':>6s} {'Dominant':>12s}")
            print("-" * 52)
            for h in hist[-24:]:  # last 24 months
                print(f"{h['month']:<10s} {h['inertia']:>8.3f} {h['resilience']:>10.3f} "
                      f"{h['active_count']:>6d} {h['dominant_stage']:>12s}")
        return

    summary = metrics.summary(args.month)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("═" * 56)
    print(f"  FR31 三指标报告 — {summary['month']}")
    print("═" * 56)

    # Inertia
    i = summary["inertia"]
    print(f"\n  🏔️  Inertia (惯性): {i['value']:.3f}")
    print(f"     {i['interpretation']}")
    if i["risk_warning"]:
        print(f"     ⚠ {i['risk_warning']}")
    print(f"     Stage: {i['stage_profile']}")
    print(f"     分量: stage={i['components']['stage_inertia']:.3f}  "
          f"dynamic={i['components']['dynamic_inertia']:.3f}  "
          f"traffic={i['components']['traffic_inertia']:.3f}")

    # Resilience
    r = summary["resilience"]
    print(f"\n  🌊 Resilience (恢复力): {r['value']:.3f}")
    print(f"     {r['interpretation']}")
    if r["risk_warning"]:
        print(f"     ⚠ {r['risk_warning']}")
    print(f"     恢复速度={r['components']['recovery_speed']:.3f}  "
          f"平均恢复={r['components']['mean_recovery_months']:.0f}月  "
          f"当前偏离={r['components']['current_fragility_sigma']:.1f}σ")

    # Position
    p = summary["position"]
    print(f"\n  🗺️  Position (图位置)")
    print(f"     {p['summary']}")
    print(f"     PC 投影: ({p['pc_projection']['pc1']:.3f}, {p['pc_projection']['pc2']:.3f})")
    print(f"     叙事温度: {p['narrative_temperature']:.3f}")

    # Raw
    raw = summary["raw_state"]
    print(f"\n  📊 原始状态")
    print(f"     活跃梗: {raw['active_memes']}  总流量: {raw['total_traffic']:.0f}")
    print(f"     Mutation Rate: {raw['mutation_rate']:.3f}  "
          f"Inst Rate: {raw['inst_rate']:.3f}  "
          f"Drift: {raw['mean_drift']:.3f}")

    print()


if __name__ == "__main__":
    main()

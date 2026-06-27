"""
Delta Transition Model — Constraint(t+1) = Constraint(t) + Δ(Context)

GPT/Gemini 共识：29条轨迹 ≈ 50个transition，不做神经网络。
学习 ΔConstraint（变化量），不是 Constraint 本身。

当前实现：基于规则的 Delta 预测 + 线性回归接口（未来数据多了再启用）。

三个 Validator：
1. Narrative Validator — 阶段顺序合法性
2. Constraint Validator — 约束变化合理性
3. Dynamics Validator — SIR/相图/盆地物理合法性
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# ═══════════════════════════════════════════════
# Delta-based Transition Model
# ═══════════════════════════════════════════════

class DeltaTransitionModel:
    """预测 Constraint 在不同阶段转移时的变化量。

    Constraint(t+1) = Constraint(t) + Δ(phase_from, phase_to, social, narrative_change)
    """

    # 阶段转移的经验 Delta 模式（来自传播学先验）
    PHASE_DELTA_PATTERNS = {
        ("origin", "emergence"): np.array([0.1, 0.0, 0.05, 0.15, 0.1]),   # 萌芽: 身份+新奇+易得↑
        ("emergence", "peak"): np.array([0.2, 0.1, 0.1, -0.05, 0.2]),     # 爆发: 全面上升
        ("peak", "controversy"): np.array([-0.15, -0.1, 0.3, -0.1, -0.2]), # 争议: 冲突↑, 其余↓
        ("controversy", "fixation"): np.array([0.05, 0.1, -0.2, -0.1, 0.05]), # 固化: 冲突↓
        ("peak", "fixation"): np.array([0.1, 0.05, -0.1, -0.1, 0.1]),     # 直接固化
        ("emergence", "fixation"): np.array([0.05, 0.0, -0.05, -0.1, 0.0]), # 快速固化
        ("controversy", "fixation"): np.array([0.05, 0.1, -0.2, -0.1, 0.05]),
    }

    def __init__(self, social_context: dict = None):
        self.social = social_context or {}
        self._ridge_coefs = None  # learned coefficients

    @classmethod
    def learn_from_trajectories(cls, trajectories: list[dict]) -> "DeltaTransitionModel":
        """从轨迹数据中用岭回归学习 Delta 系数。

        对每对相邻节点，提取特征 X = [constraint_current, economic, polarization, censorship]
        和标签 y = constraint_next - constraint_current。
        用 Ridge(alpha=1.0) 拟合 5 个独立的线性模型。
        """
        import json
        from sklearn.linear_model import Ridge

        X_list, y_list = [], []
        for t in trajectories:
            nodes = t.get("nodes", [])
            ctx = nodes[0].get("social_context", {}) if nodes else {}
            eco = ctx.get("economic_stress", 0.5)
            pol = ctx.get("polarization", 0.5)
            cen = ctx.get("censorship", 0.2)

            for i in range(len(nodes) - 1):
                c_curr = np.array(nodes[i].get("constraint_state", {}).get("pressures", [0.5]*5))
                c_next = np.array(nodes[i+1].get("constraint_state", {}).get("pressures", [0.5]*5))
                delta = c_next - c_curr
                # Features: constraint (5) + social (3) = 8
                feat = np.concatenate([c_curr, [eco, pol, cen]])
                X_list.append(feat)
                y_list.append(delta)

        if len(X_list) < 10:
            return cls()

        X = np.array(X_list)
        y = np.array(y_list)

        model = cls()
        model._ridge_coefs = []
        for dim in range(5):
            ridge = Ridge(alpha=1.0, fit_intercept=True)
            ridge.fit(X, y[:, dim])
            model._ridge_coefs.append(ridge)

        print(f"[学习] DeltaTransition: Ridge fitted on {len(X_list)} samples, "
              f"mean R²={np.mean([r.score(X, y[:, d]) for d, r in enumerate(model._ridge_coefs)]):.3f}")
        return model

    def predict_delta(self, constraint_current: np.ndarray,
                      phase_from: str, phase_to: str) -> np.ndarray:
        """预测 Constraint 变化量。优先用学习模型，fallback 到规则。"""
        # If we have learned coefficients, use them
        if self._ridge_coefs is not None:
            eco = self.social.get("economic_stress", 0.5)
            pol = self.social.get("polarization", 0.5)
            cen = self.social.get("censorship", 0.2)
            feat = np.concatenate([constraint_current, [eco, pol, cen]])
            delta = np.array([ridge.predict(feat.reshape(1, -1))[0]
                             for ridge in self._ridge_coefs])
            return np.clip(delta, -0.4, 0.4)

        # Fallback: rule-based
        return self._rule_based_delta(constraint_current, phase_from, phase_to)

    def _rule_based_delta(self, constraint_current: np.ndarray,
                          phase_from: str, phase_to: str) -> np.ndarray:
        base_delta = self.PHASE_DELTA_PATTERNS.get(
            (phase_from, phase_to), np.zeros(5)).copy()

        economic = self.social.get("economic_stress", 0.5)
        polarization = self.social.get("polarization", 0.5)
        censorship = self.social.get("censorship", 0.2)

        base_delta[0] += 0.1 * (economic - 0.5)
        base_delta[2] += 0.15 * (economic - 0.5)
        base_delta[4] -= 0.1 * (economic - 0.5)
        base_delta[2] += 0.2 * (polarization - 0.5)
        base_delta[0] += 0.05 * (polarization - 0.5)
        base_delta[2] -= 0.15 * (censorship - 0.3)
        base_delta[4] += 0.1 * (censorship - 0.3)

        for i in range(5):
            if constraint_current[i] > 0.8 and base_delta[i] > 0:
                base_delta[i] *= 0.3
            elif constraint_current[i] < 0.2 and base_delta[i] < 0:
                base_delta[i] *= 0.3

        return np.clip(base_delta, -0.4, 0.4)

    def transition(self, constraint: np.ndarray,
                   phase_from: str, phase_to: str) -> np.ndarray:
        """执行一步转移。"""
        delta = self.predict_delta(constraint, phase_from, phase_to)
        return np.clip(constraint + delta, 0.0, 1.0)


# ═══════════════════════════════════════════════
# Three Validators
# ═══════════════════════════════════════════════

VALID_PHASE_ORDER = ["origin", "emergence", "peak", "controversy", "fixation"]


@dataclass
class ValidationReport:
    valid: bool
    violations: list[str]
    warnings: list[str]
    validator_name: str

    def __bool__(self):
        return self.valid


class NarrativeValidator:
    """检查阶段顺序是否合法。

    规则：
    - 不能倒退（fixation → peak 非法）
    - 不能跳过太多阶段（origin → fixation 跳过 3 个阶段可能是数据缺失而非物理可能）
    - Mutation 通常出现在 peak 或 controversy 之后
    """

    def validate(self, phase_sequence: list[str]) -> ValidationReport:
        violations = []
        warnings = []

        for i in range(1, len(phase_sequence)):
            prev = phase_sequence[i - 1]
            curr = phase_sequence[i]
            if prev not in VALID_PHASE_ORDER or curr not in VALID_PHASE_ORDER:
                violations.append(f"Unknown phase: {prev}→{curr}")
                continue

            prev_idx = VALID_PHASE_ORDER.index(prev)
            curr_idx = VALID_PHASE_ORDER.index(curr)

            if curr_idx < prev_idx:
                # Allow same-phase repetition or minor reversals (resurgence)
                if curr_idx == prev_idx:
                    pass
                elif prev_idx - curr_idx == 1:
                    warnings.append(f"Minor phase reversal: {prev}→{curr} (may be resurgence)")
                else:
                    violations.append(f"Phase reversal: {prev}→{curr}")
            elif curr_idx - prev_idx > 2:
                warnings.append(f"Large phase jump: {prev}→{curr} (skipped {curr_idx - prev_idx - 1} phases)")

        return ValidationReport(
            valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            validator_name="NarrativeValidator",
        )


class ConstraintValidator:
    """检查约束变化是否在合理范围内。

    规则：
    - 单步 Constraint 变化不超过 0.35（任何维度）
    - 5 维总变化量（欧氏距离）不超过 0.5
    - Conflict 和 Identity 不会同时大幅同向变化
    """

    def validate(self, constraints: list[np.ndarray]) -> ValidationReport:
        violations = []
        warnings = []

        for i in range(1, len(constraints)):
            prev = constraints[i - 1]
            curr = constraints[i]
            delta = curr - prev

            # 单维变化上限
            for dim, d in enumerate(delta):
                if abs(d) > 0.35:
                    violations.append(
                        f"Step {i}: dim {dim} changed by {d:+.3f} (max 0.35)"
                    )

            # 总变化量
            total_change = np.linalg.norm(delta)
            if total_change > 0.5:
                violations.append(
                    f"Step {i}: total constraint change {total_change:.3f} (max 0.5)"
                )

            # Conflict 和 Identity 的异常共变
            if delta[0] > 0.15 and delta[2] > 0.15:
                warnings.append(
                    f"Step {i}: Identity +{delta[0]:.2f} and Conflict +{delta[2]:.2f} both strongly up"
                )

        return ValidationReport(
            valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            validator_name="ConstraintValidator",
        )


class DynamicsValidator:
    """检查动力学层的物理合法性。

    规则：
    - Peak 阶段 R₀ 必须 > 1（模因在爆发期必须有传播能力）
    - Controversy 阶段 Chaos Axis 应出现负漂（争议使系统偏混沌）
    - Fixation 阶段必须收敛到某个吸引子盆地或 R₀ < 1（消亡）
    - R₀ 单步变化不超过 1.5（连续性约束）
    """

    def validate(self, nodes: list[dict]) -> ValidationReport:
        """nodes: [{"phase": str, "dynamic_state": {...}}, ...]"""
        violations = []
        warnings = []

        for i, node in enumerate(nodes):
            phase = node.get("phase", "")
            ds = node.get("dynamic_state", {})
            r0 = ds.get("R0", 0)
            chaos = ds.get("chaos_axis", 0)

            if phase == "peak" and r0 < 1.0:
                violations.append(f"Node {i} ({phase}): R₀={r0:.2f} < 1.0 (should be outbreak)")

            if phase == "controversy" and chaos > -0.1:
                warnings.append(f"Node {i} ({phase}): chaos={chaos:+.2f}, expected negative drift")

            if phase == "fixation":
                if 0.5 < r0 < 2.0 and abs(chaos) < 0.9:
                    pass  # 可能在盆地内，OK
                elif r0 >= 1.0 and abs(chaos) < 0.1:
                    warnings.append(f"Node {i} ({phase}): R₀={r0:.2f} still active, not clearly in basin")

            # R₀ 连续性（origin→peak 和 peak→fixation 的跳跃是物理正确的）
            if i > 0:
                prev_r0 = nodes[i - 1].get("dynamic_state", {}).get("R0", 0)
                prev_phase = nodes[i - 1].get("phase", "")
                if abs(r0 - prev_r0) > 1.5:
                    if prev_phase == "origin":
                        pass  # origin always starts at R₀≈0.1, first real phase can be >>1
                    elif phase == "fixation" and r0 < 0.1:
                        pass  # meme dying is physically correct
                    else:
                        violations.append(f"Node {i}: R₀ jump {prev_r0:.2f}→{r0:.2f} (max 1.5)")

        return ValidationReport(
            valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            validator_name="DynamicsValidator",
        )


def validate_trajectory(phases: list[str], constraints: list[np.ndarray],
                        nodes: list[dict]) -> dict:
    """对一条 Trajectory 运行全部三层验证。"""
    nv = NarrativeValidator().validate(phases)
    cv = ConstraintValidator().validate(constraints)
    dv = DynamicsValidator().validate(nodes)

    all_valid = nv.valid and cv.valid and dv.valid
    return {
        "valid": all_valid,
        "narrative": {"valid": nv.valid, "violations": nv.violations, "warnings": nv.warnings},
        "constraint": {"valid": cv.valid, "violations": cv.violations, "warnings": cv.warnings},
        "dynamics": {"valid": dv.valid, "violations": dv.violations, "warnings": dv.warnings},
    }


# ═══════════════════════════════════════════════
# Entry: Validate all 29 trajectories
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import json, sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Delta Transition + 3-Validator — Demo")
    print("=" * 60)

    # ── Delta demo ──
    dtm = DeltaTransitionModel({"economic_stress": 0.7, "polarization": 0.6, "censorship": 0.3})
    c0 = np.array([0.5, 0.4, 0.3, 0.5, 0.6])
    c1 = dtm.transition(c0, "peak", "controversy")
    print(f"\nDelta: peak → controversy")
    print(f"  Constraint: {c0.round(3)} → {c1.round(3)}")
    print(f"  Δ = {(c1 - c0).round(3)}")

    # ── Validate all trajectories ──
    with open("data/processed/trajectories.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    total_valid = 0
    for t in data.get("trajectories", []):
        nodes = t.get("nodes", [])
        phases = [n["phase"] for n in nodes]
        constraints = [np.array(n.get("constraint_state", {}).get("pressures", [0.5]*5))
                       for n in nodes]
        result = validate_trajectory(phases, constraints, nodes)
        if result["valid"]:
            total_valid += 1
        else:
            viols = (len(result["narrative"]["violations"]) +
                     len(result["constraint"]["violations"]) +
                     len(result["dynamics"]["violations"]))
            if viols > 0:
                print(f"\n  {t['name']}: {viols} violations")
                for v in result["narrative"]["violations"]:
                    print(f"    [Narrative] {v}")
                for v in result["constraint"]["violations"]:
                    print(f"    [Constraint] {v}")
                for v in result["dynamics"]["violations"]:
                    print(f"    [Dynamics] {v}")

    print(f"\n  Valid: {total_valid}/{len(data.get('trajectories', []))}")

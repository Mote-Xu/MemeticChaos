"""
FR31 Unified Engine — v4.1 Formalism

Single entry point for the entire MemeticChaos system.
Wraps all modules into one consistent interface:

    x(t) ∈ M(z(t))        — state lives on control-conditioned manifold
    p(x | z)              — regime-conditioned density
    persona(x_user)       — distribution projection, not point estimate

Provides:
    engine.query(mode="macro")         → 4 indicators + regime + control context
    engine.query(mode="persona", ...)  → individual projection + macro overlay
    engine.query(mode="full", ...)     → everything

This is the integration point for Stella (OpenClaw) and Flask API.

Usage:
    python src/advisor/engine.py                        # print full report
    python src/advisor/engine.py --json                 # JSON output
    python src/advisor/engine.py --text "用户文本..."    # persona mode
"""

import json, sys, os, argparse
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from advisor.metrics import NarrativeMetrics
from advisor.persona import PersonaEncoder

DATA_DIR = ROOT / "data/processed"


class FR31Engine:
    """Unified query engine for MemeticChaos v4.1.

    Formalism:
        x(t) ∈ M(z(t))           narrative state on control manifold
        regime S(t) ∈ {R0..R3}   phase identity
        λ₂(x)                    stability operator (graph spectrum)
        persona(u) = P(m|u)      distribution over meme nodes

    All queries return {status, data, meta} triples.
    """

    def __init__(self):
        # Core modules
        self.metrics = NarrativeMetrics()
        self._persona = None  # lazy load (heavy model)

        # Static data
        self.regime = self._load("regime_map.json")
        self.control = self._load("control_manifold.json")
        self.irrev = self._load("irreversibility_results.json")

        # Latest month index
        self.latest_month = self.metrics.months[-1]

    def _load(self, filename: str) -> dict:
        with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def persona(self):
        if self._persona is None:
            self._persona = PersonaEncoder()
            self._persona.load_narrative_graph()
        return self._persona

    # ═══════════════════════════════════════════════
    # Query interface
    # ═══════════════════════════════════════════════

    def query(self, mode: str = "macro", text: str = None,
              month: str = None) -> dict:
        """Unified query.

        Args:
            mode: "macro" | "persona" | "full"
            text: user text for persona projection
            month: target month (default: latest)

        Returns:
            {status, data, meta}
        """
        m = month or self.latest_month

        result = {
            "status": "ok",
            "meta": {
                "formalism": "v4.1 — control-conditioned phase space",
                "month": m,
                "generated_at": datetime.now().isoformat(),
            },
            "data": {},
        }

        if mode in ("macro", "full"):
            result["data"]["macro"] = self._macro_state(m)

        if mode in ("persona", "full") and text:
            result["data"]["persona"] = self._persona_state(text)

        elif mode == "persona" and not text:
            result["status"] = "error"
            result["error"] = "persona mode requires --text"

        return result

    # ═══════════════════════════════════════════════
    # Macro state
    # ═══════════════════════════════════════════════

    def _macro_state(self, month: str) -> dict:
        """Aggregate all macro indicators into unified view."""
        s = self.metrics.summary(month)

        # Current regime
        regime_idx = self.regime["months"].index(month)
        current_regime = int(self.regime["regime_labels"][regime_idx])
        regime_info = self.regime["regime_characteristics"][str(current_regime)]

        # Control manifold position
        ctl_idx = None
        for i, pt in enumerate(self.control["analysis"]["timeline"]):
            if pt["month"] == month:
                ctl_idx = i
                break

        z_pos = None
        if ctl_idx is not None:
            pt = self.control["analysis"]["timeline"][ctl_idx]
            z_pos = {"z1": pt["z1"], "z2": pt["z2"], "z3": pt["z3"]}

        # Irreversibility verdict
        irrev_verdict = self.irrev["combined_verdict"]

        # ── Assemble ──
        return {
            "indicators": {
                "inertia": s["inertia"]["value"],
                "resilience": s["resilience"]["value"],
                "sensitivity": s["sensitivity"]["value"],
                "position": s["position"]["dominant_regime"],
            },
            "diagnosis": self._diagnose(s),
            "regime": {
                "current": f"R{current_regime}",
                "label": regime_info["dominant_stage"],
                "self_persistence": round(
                    self.regime["transition_matrix"][current_regime][current_regime], 3),
                "dwell_months": self._current_dwell(regime_idx, current_regime),
            },
            "control_manifold": {
                "z_position": z_pos,
                "z1_driver": "AI/Tech discourse",
                "z1_interpretation": (
                    "AI/Tech 轴十年来单调漂移, 定义了系统可达路径. "
                    "不是因果关系——是控制梯度方向."
                ),
            },
            "structure": {
                "irreversibility": irrev_verdict,
                "phase_count": self.regime["n_regimes"],
                "r2_is_real_separation": True,
            },
            "raw": s["raw_state"],
            "risk_summary": self._risk_summary(s, current_regime),
        }

    def _diagnose(self, s: dict) -> str:
        i = s["inertia"]["value"]
        r = s["resilience"]["value"]
        se = s["sensitivity"]["value"]

        if i > 0.6 and r < 0.3 and se > 0.4:
            return (
                "亚稳态 (Metastable) — 高惯性+低恢复力+上升敏感性. "
                "系统处于分岔前临界态. 外部场 u(t) 沿 AI/Tech 轴单向漂移, "
                "系统在 R2 盆地中锁死. 逃逸需要 z₁ 方向反转或拓扑断裂."
            )
        elif i > 0.5 and r < 0.5:
            return "偏僵化 — 惯性偏高, 恢复力不足. 系统在 R2 边界附近."
        else:
            return "动态平衡 — 系统在正常运作范围内."

    def _current_dwell(self, idx: int, regime: int) -> int:
        """计算当前 regime 的连续驻留月数."""
        labels = self.regime["regime_labels"]
        count = 0
        for i in range(idx, -1, -1):
            if labels[i] == regime:
                count += 1
            else:
                break
        return count

    def _risk_summary(self, s: dict, regime: int) -> str:
        risks = []
        if s["inertia"]["risk_warning"]:
            risks.append(s["inertia"]["risk_warning"])
        if s["sensitivity"]["risk_warning"]:
            risks.append(s["sensitivity"]["risk_warning"])
        if regime == 2:
            dwell = self._current_dwell(
                self.regime["months"].index(s["month"]), 2)
            if dwell > 24:
                risks.append(f"R2 驻留 {dwell} 月 (中位 19 月, 已超 2x)")
        return "; ".join(risks) if risks else "无显著风险"

    # ═══════════════════════════════════════════════
    # Persona projection
    # ═══════════════════════════════════════════════

    def _persona_state(self, text: str) -> dict:
        """Individual projection with macro overlay."""
        proj = self.persona.project(text)

        # Get current macro context for overlay
        macro = self.metrics.summary()
        regime_idx = -1
        current_regime = int(self.regime["regime_labels"][regime_idx])

        return {
            "projection": proj,
            "macro_overlay": {
                "current_regime": f"R{current_regime}",
                "inertia": macro["inertia"]["value"],
                "sensitivity": macro["sensitivity"]["value"],
            },
            "integration_note": (
                "个体投影结果需结合宏观 overlay 解读. "
                "高惯性环境下, 即使个体文本匹配特定节点, "
                "大盘对该节点的'容纳能力'可能极低 (低 Resilience). "
                "FR31 建议: 先看宏观地形, 再读个体投影."
            ),
        }

    # ═══════════════════════════════════════════════
    # Stella-ready summary
    # ═══════════════════════════════════════════════

    def stella_prompt(self, text: str = None) -> str:
        """Generate a Stella-ready natural language summary."""
        q = self.query(mode="full" if text else "macro", text=text)
        d = q["data"]

        lines = []
        macro = d["macro"]

        # Header
        lines.append(f"📊 FR31 状态报告 — {q['meta']['month']}")
        lines.append(f"诊断: {macro['diagnosis']}")
        lines.append("")

        # Indicators
        ind = macro["indicators"]
        lines.append(f"Inertia={ind['inertia']:.2f} | Resilience={ind['resilience']:.2f} | "
                     f"Sensitivity={ind['sensitivity']:.2f}")
        lines.append(f"Regime: {macro['regime']['current']} ({macro['regime']['label']}), "
                     f"驻留 {macro['regime']['dwell_months']} 月, "
                     f"自持 {macro['regime']['self_persistence']:.1%}")
        lines.append("")

        # Control context
        ctl = macro["control_manifold"]
        if ctl["z_position"]:
            z = ctl["z_position"]
            lines.append(f"控制流形: z=({z['z1']:.3f}, {z['z2']:.3f}, {z['z3']:.3f})")
        lines.append(f"z₁ 驱动: {ctl['z1_driver']} — {ctl['z1_interpretation']}")
        lines.append("")

        # Risk
        if macro["risk_summary"] != "无显著风险":
            lines.append(f"⚠ {macro['risk_summary']}")
            lines.append("")

        # Persona
        if text and "persona" in d:
            p = d["persona"]["projection"]
            lines.append("─ Persona 投影 ─")
            if p["status"] == "MELTDOWN":
                lines.append(f"[{p['reason']}] {p.get('interpretation', '')[:200]}")
            else:
                for n in (p.get("distribution", {}).get("top_nodes", [])[:3]):
                    lines.append(f"  {n['rank']}. {n['node']} (p={n['probability']:.4f})")
            lines.append(f"  {p.get('epistemic_note', '')[:150]}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FR31 Unified Engine")
    parser.add_argument("--text", type=str, default=None, help="用户文本 (persona 模式)")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--stella", action="store_true", help="Stella 自然语言摘要")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    engine = FR31Engine()
    mode = "full" if args.text else "macro"

    if args.stella:
        print(engine.stella_prompt(args.text))
        return

    result = engine.query(mode=mode, text=args.text)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # Human-readable
    d = result["data"]["macro"]
    print("═" * 56)
    print(f"  FR31 统一状态 — {result['meta']['month']}")
    print("═" * 56)
    print(f"\n  {d['diagnosis']}")
    ind = d["indicators"]
    print(f"\n  Inertia={ind['inertia']:.3f}  Resilience={ind['resilience']:.3f}  "
          f"Sensitivity={ind['sensitivity']:.3f}")
    print(f"  Regime: {d['regime']['current']} ({d['regime']['label']}), "
          f"驻留 {d['regime']['dwell_months']} 月, "
          f"自持 {d['regime']['self_persistence']:.1%}")
    ctl = d["control_manifold"]
    if ctl["z_position"]:
        z = ctl["z_position"]
        print(f"  控制流形: z=({z['z1']:.3f}, {z['z2']:.3f}, {z['z3']:.3f})")
    print(f"  结构: {d['structure']['irreversibility']}")
    print(f"\n  ⚠ {d['risk_summary']}")

    if "persona" in result.get("data", {}):
        p = result["data"]["persona"]["projection"]
        print(f"\n  ─ Persona ─")
        print(f"  Status: {p['status']}")
        if p["status"] != "MELTDOWN":
            for n in p["distribution"]["top_nodes"][:3]:
                print(f"  {n['rank']}. {n['node']} (p={n['probability']:.4f})")
        else:
            print(f"  [{p['reason']}] {p.get('interpretation', '')[:200]}")

    print()


if __name__ == "__main__":
    main()

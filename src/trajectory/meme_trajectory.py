"""
MemeTrajectory — 模因轨迹核心数据结构 (v2.1 Schema)

每个梗不是在相图上的一个点，而是一条在四维状态空间中运动的路径。
Narrative / Constraint / Dynamic / Social 四个状态向量在每个阶段节点并行演化。

Schema 已冻结 (2026-06-26)。所有 LLM 抽取和策展数据填入此格式。
"""

import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path


# ═══════════════════════════════════════════════
# Phase enum + State vectors
# ═══════════════════════════════════════════════

PHASE_ORDER = ["origin", "emergence", "peak", "controversy", "fixation"]


@dataclass
class NarrativeState:
    meaning: str = ""
    mechanism: str = ""
    key_figures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "meaning": self.meaning,
            "mechanism": self.mechanism,
            "key_figures": self.key_figures,
        }


@dataclass
class ConstraintState:
    pressures: list[float] = field(default_factory=lambda: [0.5]*5)  # p₁...p₅
    dominant_constraint: str = ""

    def to_dict(self) -> dict:
        return {
            "pressures": self.pressures,
            "dominant_constraint": self.dominant_constraint,
        }

    @property
    def vector(self) -> np.ndarray:
        return np.array(self.pressures)


@dataclass
class DynamicState:
    R0: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    sigma: float = 0.0
    mu: float = 0.0
    chaos_axis: float = 0.0
    entropy: float = 0.0
    peak_infected: float = 0.0

    def to_dict(self) -> dict:
        return {
            "R0": self.R0, "beta": self.beta, "gamma": self.gamma,
            "sigma": self.sigma, "mu": self.mu,
            "chaos_axis": self.chaos_axis, "entropy": self.entropy,
            "peak_infected": self.peak_infected,
        }

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.R0, self.chaos_axis, self.entropy])


@dataclass
class SocialContext:
    economic_stress: float = 0.5
    polarization: float = 0.5
    censorship: float = 0.2
    platform: str = ""
    trigger_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "economic_stress": self.economic_stress,
            "polarization": self.polarization,
            "censorship": self.censorship,
            "platform": self.platform,
            "trigger_events": self.trigger_events,
        }


# ═══════════════════════════════════════════════
# Trajectory Node + Trajectory
# ═══════════════════════════════════════════════

@dataclass
class TrajectoryNode:
    phase: str  # "origin" | "emergence" | "peak" | "controversy" | "fixation"
    time_range: dict  # {"start": "2020-10", "end": "2020-11"}
    narrative_state: NarrativeState
    constraint_state: ConstraintState
    dynamic_state: DynamicState
    social_context: SocialContext

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "time_range": self.time_range,
            "narrative_state": self.narrative_state.to_dict(),
            "constraint_state": self.constraint_state.to_dict(),
            "dynamic_state": self.dynamic_state.to_dict(),
            "social_context": self.social_context.to_dict(),
        }

    @property
    def phase_index(self) -> int:
        return PHASE_ORDER.index(self.phase) if self.phase in PHASE_ORDER else -1


@dataclass
class MemeTrajectory:
    meme_id: str
    name: str
    category: str
    year_range: tuple[int, int]
    nodes: list[TrajectoryNode]
    mutations: list[dict]
    semantic_drift: dict
    sources: dict

    def to_dict(self) -> dict:
        return {
            "meme_id": self.meme_id,
            "name": self.name,
            "category": self.category,
            "year_range": list(self.year_range),
            "nodes": [n.to_dict() for n in self.nodes],
            "mutations": self.mutations,
            "semantic_drift": self.semantic_drift,
            "sources": self.sources,
        }

    @property
    def phase_count(self) -> int:
        return len(self.nodes)

    @property
    def R0_curve(self) -> np.ndarray:
        return np.array([n.dynamic_state.R0 for n in self.nodes])

    @property
    def chaos_curve(self) -> np.ndarray:
        return np.array([n.dynamic_state.chaos_axis for n in self.nodes])

    @property
    def constraint_evolution(self) -> np.ndarray:
        """Constraint 向量在时间上的演化 (phases × 5)。"""
        return np.array([n.constraint_state.pressures for n in self.nodes])

    def summary(self) -> str:
        lines = [
            f"{self.name} [{self.category}] ({self.year_range[0]}-{self.year_range[1]})",
            f"  Phases: {self.phase_count} | Mutations: {len(self.mutations)}",
            f"  R₀ range: {self.R0_curve.min():.2f} → {self.R0_curve.max():.2f}",
            f"  Chaos range: {self.chaos_curve.min():.2f} → {self.chaos_curve.max():.2f}",
        ]
        if self.semantic_drift:
            lines.append(f"  Drift: {self.semantic_drift.get('direction', 'N/A')}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MemeTrajectory({self.name}, {self.phase_count} phases, {len(self.mutations)} mutations)"


# ═══════════════════════════════════════════════
# Trajectory Builder — 从 LLM 抽取 + 策展数据构建
# ═══════════════════════════════════════════════

class TrajectoryBuilder:
    """从多种数据源构建 MemeTrajectory。

    数据源优先级：
    1. LLM 抽取的叙事 JSON（data/processed/narratives/）
    2. 策展数据 memes_2020_2025.json
    3. SIR 估算参数（观测层）
    """

    def __init__(self, curated_path: str = None, narratives_dir: str = None):
        if curated_path is None:
            curated_path = "data/curated/memes_2020_2025.json"
        if narratives_dir is None:
            narratives_dir = "data/processed/narratives"

        with open(curated_path, "r", encoding="utf-8") as f:
            self.curated = json.load(f)

        self.narratives = {}
        if os.path.isdir(narratives_dir):
            for fn in os.listdir(narratives_dir):
                if fn.endswith(".json") and not fn.startswith("_"):
                    with open(os.path.join(narratives_dir, fn), "r", encoding="utf-8") as f:
                        nar = json.load(f)
                        if "error" not in nar:
                            name = nar.get("meme_name", "")
                            self.narratives[name] = nar

        from src.data.curator import MemeCurator
        self.curator = MemeCurator()

    def _find_narrative(self, meme_name: str, aliases: list[str]) -> dict | None:
        """模糊匹配 LLM 抽取的叙事。"""
        if meme_name in self.narratives:
            return self.narratives[meme_name]
        for alias in aliases:
            if alias in self.narratives:
                return self.narratives[alias]
        # Substring match
        for key, nar in self.narratives.items():
            if meme_name in key or any(a in key for a in aliases):
                return nar
        return None

    def _estimate_constraint(self, narrative: dict | None,
                             category: str, year: int) -> ConstraintState:
        """使用 Concept Bottleneck 从叙事 JSON 估算 5D 约束向量。

        两阶段：Narrative → 35 可观察概念 → 5D 约束场。
        """
        from src.constraint.concept_bottleneck import ConceptMatrix, ConstraintMapper

        if narrative is None:
            # 无叙事数据 → 从类别估算
            cat_defaults = {
                "解构自嘲": [0.7, 0.8, 0.2, 0.5, 0.6],
                "攻击发泄": [0.5, 0.2, 0.9, 0.4, 0.7],
                "虚无退却": [0.6, 0.3, 0.6, 0.3, 0.8],
                "身份认同": [0.8, 0.4, 0.4, 0.6, 0.5],
                "纯粹娱乐": [0.3, 0.7, 0.1, 0.8, 0.9],
            }
            pressures = cat_defaults.get(category, [0.5]*5)
            return ConstraintState(pressures=list(pressures), dominant_constraint=category)

        # Concept Bottleneck: Narrative → 35 concepts → 5D
        cm = ConceptMatrix.from_narrative(narrative)
        pressures = ConstraintMapper.map(cm).tolist()

        # 主导约束
        labels = ["identity", "humor/decon", "conflict", "novelty", "accessibility"]
        dominant = labels[int(np.argmax(pressures))]

        return ConstraintState(
            pressures=pressures,
            dominant_constraint=dominant,
        )

    def _estimate_social_context(self, narrative: dict | None, year: int) -> SocialContext:
        """估算社会背景参数。"""
        ctx = SocialContext()
        if narrative:
            social = narrative.get("social_context", {})
            triggers = social.get("triggers", [])
            ctx.trigger_events = triggers
            ctx.platform = narrative.get("origin", {}).get("platform", "")

            # 从触发事件关键词推断压力值
            trigger_text = " ".join(triggers)
            if any(w in trigger_text for w in ["失业", "就业", "房价", "经济", "收入", "生存"]):
                ctx.economic_stress = 0.8
            elif any(w in trigger_text for w in ["消费", "压力", "焦虑", "内卷"]):
                ctx.economic_stress = 0.6
            if any(w in trigger_text for w in ["争议", "批判", "对立", "骂战", "反弹"]):
                ctx.polarization = 0.8
            elif any(w in trigger_text for w in ["阶层", "差异", "不同群体"]):
                ctx.polarization = 0.6
            if any(w in trigger_text for w in ["审查", "敏感", "官方", "政策", "收编"]):
                ctx.censorship = 0.5
        return ctx

    def build_all(self) -> list[MemeTrajectory]:
        """构建全部已策展热梗的 MemeTrajectory。"""
        from src.models.sir_meme import estimate_params_from_lifecycle, estimate_total_infected

        trajectories = []
        for meme in self.curator.memes:
            lc = meme.lifecycle
            pm = meme.propagation_model
            sa = meme.sentiment_arc

            # ── Basic info ──
            dur_months = lc.get("duration_months", 6)
            if dur_months >= 999:
                dur_months = 18
            peak_intensity = max(p.get("intensity", 0.5) for p in sa) if sa else 0.5
            circle_count = len(pm.get("circle_layers", []))

            # ── Dynamic estimate ──
            ti = estimate_total_infected(circle_count, peak_intensity, dur_months)
            dur_days = dur_months * 30
            params = estimate_params_from_lifecycle(
                peak_day=dur_days * 0.3, total_infected=ti, duration_days=dur_days
            )

            # ── Narrative ──
            nar = self._find_narrative(meme.name, meme.aliases)
            # Fallback: wrap curated narrative text as minimal dict for Concept Bottleneck
            if nar is None and meme.narrative:
                nar = {
                    "meme_name": meme.name,
                    "social_context": {
                        "triggers": meme.tags,
                        "target_audience": ", ".join(pm.get("circle_layers", [])),
                    },
                    "spread_phases": [
                        {"phase": sap.get("phase", ""), "description": f"{sap.get('sentiment','')} (强度{sap.get('intensity',0)})"}
                        for sap in sorted(sa, key=lambda p: p.get("intensity", 0))
                    ],
                    "mutations": [{"relationship": v} for v in meme.mutation_variants],
                    "semantic_drift": meme.chaos_vector if isinstance(meme.chaos_vector, dict) else {},
                    "origin": {
                        "trigger_event": meme.narrative[:200],
                        "platform": pm.get("source_platforms", [""])[0] if pm.get("source_platforms") else "",
                        "precursor": meme.name,
                    },
                    "narrative_quality": {"information_richness": "中"},
                }

            # Phase mapping
            phase_map = {
                "萌芽": "emergence", "爆发": "peak",
                "泛化": "controversy", "变异": "controversy",
                "固化": "fixation", "消亡": "fixation",
                "emergence": "emergence", "peak": "peak",
                "controversy": "controversy", "fixation": "fixation",
            }

            # ── Build nodes: prioritize LLM narrative, fallback to sentiment_arc ──
            nodes = []
            origin = nar.get("origin", {}) if nar else {}

            # ── Origin fingerprint: 保留本征基因到所有后续节点 ──
            social_nar = (nar or {}).get("social_context") or {}
            origin_fingerprint = {
                "platform": str(origin.get("platform") or (pm.get("source_platforms", [""])[0] if pm.get("source_platforms") else "")),
                "is_official": 1.0 if any(w in str(origin.get("trigger_event", "")) for w in ["官方","B站","宣传片","政策","政府"]) else 0.0,
                "base_sensitivity": 0.5 if social_nar.get("political_sensitivity") in ["中", "高"] else 0.3,
            }

            # origin node
            nodes.append(TrajectoryNode(
                phase="origin",
                time_range={"start": origin.get("time", f"{meme.year}"), "end": origin.get("time", f"{meme.year}")},
                narrative_state=NarrativeState(
                    meaning=origin.get("precursor", meme.name),
                    mechanism=origin.get("trigger_event", ""),
                    key_figures=[],
                ),
                constraint_state=self._estimate_constraint(nar, meme.category, meme.year),
                dynamic_state=DynamicState(R0=0.1, chaos_axis=meme.chaos_position),
                social_context=self._estimate_social_context(nar, meme.year),
            ))

            if nar and nar.get("spread_phases"):
                # ── LLM narrative phases ──
                for i, sp in enumerate(nar["spread_phases"]):
                    phase_key = phase_map.get(sp.get("phase", ""), "peak")
                    n_phases = max(1, len(nar["spread_phases"]))
                    frac = (i + 1) / n_phases
                    # Peak gets full R0, others get fraction
                    node_R0 = params.R0 if phase_key == "peak" else params.R0 * frac
                    node_chaos = meme.chaos_position * (1.0 + 0.1 * i)
                    nodes.append(TrajectoryNode(
                        phase=phase_key,
                        time_range={
                            "start": sp.get("time_range", sp.get("time", f"{meme.year}")),
                            "end": sp.get("time_range", sp.get("time", f"{meme.year}")),
                        },
                        narrative_state=NarrativeState(
                            meaning=sp.get("description", ""),
                            mechanism="",
                            key_figures=sp.get("key_figures") or [],
                        ),
                        constraint_state=self._estimate_constraint(nar, meme.category, meme.year),
                        dynamic_state=DynamicState(
                            R0=max(0.1, node_R0),
                            chaos_axis=float(np.clip(node_chaos, -1, 1)),
                        ),
                        social_context=self._estimate_social_context(nar, meme.year),
                    ))
            else:
                # ── Fallback: lifecycle dates from curated data ──
                # Every meme has emergence/peak/decay in lifecycle
                emergence_date = lc.get("emergence", f"{meme.year}")
                peak_date = lc.get("peak", f"{meme.peak_year}")
                decay_date = lc.get("decay", f"{meme.peak_year + 1}")
                status = lc.get("status", "消亡")

                # emergence node: R₀ ramps from origin to peak gradually
                emergence_R0 = max(0.5, params.R0 * 0.5)
                nodes.append(TrajectoryNode(
                    phase="emergence",
                    time_range={"start": str(emergence_date), "end": str(peak_date)},
                    narrative_state=NarrativeState(
                        meaning=f"从 {pm.get('transmission_mechanism', '社区传播')} 开始扩散",
                        mechanism=pm.get("transmission_mechanism", ""),
                        key_figures=[],
                    ),
                    constraint_state=self._estimate_constraint(nar, meme.category, meme.year),
                    dynamic_state=DynamicState(
                        R0=emergence_R0,
                        chaos_axis=meme.chaos_position,
                    ),
                    social_context=self._estimate_social_context(nar, meme.year),
                ))
                # peak node: R₀ must be > 1 (any curated meme reached peak by definition)
                peak_R0 = max(1.05, params.R0)
                nodes.append(TrajectoryNode(
                    phase="peak",
                    time_range={"start": str(peak_date), "end": str(decay_date)},
                    narrative_state=NarrativeState(
                        meaning=f"在 {pm.get('source_platforms', ['多平台'])[0] if pm.get('source_platforms') else '社交平台'} 达到传播高峰",
                        mechanism="viral spread",
                        key_figures=[],
                    ),
                    constraint_state=self._estimate_constraint(nar, meme.category, meme.year),
                    dynamic_state=DynamicState(
                        R0=peak_R0,
                        chaos_axis=meme.chaos_position,
                        peak_infected=float(peak_intensity),
                    ),
                    social_context=self._estimate_social_context(nar, meme.year),
                ))
                # controversy/decay → fixation handled by the universal fixation block below
                if "变异" in status or "复燃" in status:
                    mid_phase = "controversy"
                else:
                    mid_phase = "fixation"
                nodes.append(TrajectoryNode(
                    phase=mid_phase,
                    time_range={"start": str(decay_date), "end": f"{meme.peak_year + 1}"},
                    narrative_state=NarrativeState(
                        meaning=f"变异体: {', '.join(meme.mutation_variants[:3])}" if meme.mutation_variants else status,
                        mechanism="mutation" if meme.mutation_variants else "decay",
                        key_figures=[],
                    ),
                    constraint_state=self._estimate_constraint(nar, meme.category, meme.year),
                    dynamic_state=DynamicState(
                        R0=params.R0 * 0.5 if meme.mutation_variants else params.R0 * 0.1,
                        chaos_axis=meme.chaos_position,
                    ),
                    social_context=self._estimate_social_context(nar, meme.year),
                ))

            # ── Fixation node (always) ──
            if not any(n.phase == "fixation" for n in nodes):
                status_phase = lc.get("status", "消亡")
                is_endemic = "固化" in status_phase
                nodes.append(TrajectoryNode(
                    phase="fixation",
                    time_range={"start": f"{meme.peak_year}", "end": f"{meme.peak_year + 1}"},
                    narrative_state=NarrativeState(meaning=status_phase),
                    constraint_state=self._estimate_constraint(nar, meme.category, meme.year),
                    dynamic_state=DynamicState(
                        R0=params.R0 if is_endemic else 0.0,
                        chaos_axis=meme.chaos_position,
                    ),
                    social_context=self._estimate_social_context(nar, meme.year),
                ))

            # ── Mutations + Drift ──
            mutations = nar.get("mutations", []) if nar else []
            semantic_drift = nar.get("semantic_drift", {}) if nar else {}

            trajectory = MemeTrajectory(
                meme_id=meme.meme_id,
                name=meme.name,
                category=meme.category,
                year_range=(meme.year, meme.peak_year),
                nodes=nodes,
                mutations=mutations,
                semantic_drift=semantic_drift,
                sources={
                    "curated_json": True,
                    "video_transcript": nar.get("video_name", None) if nar else None,
                    "llm_extraction_confidence": (
                        nar.get("narrative_quality", {}).get("information_richness", "low")
                        if nar else "none"
                    ),
                },
            )
            trajectories.append(trajectory)

        return trajectories

    def save_all(self, trajectories: list[MemeTrajectory], output_path: str = None):
        """保存所有轨迹为 JSON。"""
        if output_path is None:
            output_path = "data/processed/trajectories.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = {
            "schema_version": "2.1",
            "built_at": "2026-06-26",
            "n_trajectories": len(trajectories),
            "trajectories": [t.to_dict() for t in trajectories],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[保存] {len(trajectories)} 条 MemeTrajectory → {output_path}")
        return output_path


# ═══════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("MemeticChaos — MemeTrajectory Builder v2.1")
    print("=" * 60)

    builder = TrajectoryBuilder()
    trajectories = builder.build_all()

    for t in trajectories:
        print(f"\n{t.summary()}")

    builder.save_all(trajectories)

    print(f"\nDone. {len(trajectories)} trajectories built.")

"""
热梗策展数据管理模块 — MemeticChaos 数据层。

提供：
1. 从 JSON 策展数据加载和查询
2. SIR 参数估算（从 qualitative lifecycle → quantitative params）
3. 数据验证和统计
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_DATA = PROJECT_ROOT / "data" / "curated" / "memes_2020_2025.json"


@dataclass
class MemeEntry:
    """单个热梗的完整数据条目。"""
    meme_id: str
    name: str
    aliases: list[str]
    year: int
    peak_year: int
    source: str
    source_platforms: list[str]
    category: str
    tags: list[str]
    lifecycle: dict
    sentiment_arc: list[dict]
    chaos_vector: dict
    propagation_model: dict
    mutation_variants: list[str]
    narrative: str

    @property
    def chaos_position(self) -> float:
        """混沌轴位置：-1 (绝对混沌) 到 +1 (绝对秩序)。"""
        return self.chaos_vector.get("chaos_order_position", 0.0)

    @property
    def estimated_R0(self) -> float:
        return self.propagation_model.get("estimated_R0", 1.0)

    def summary(self) -> str:
        return (f"[{self.year}] {self.name} | {self.category} | "
                f"R₀≈{self.estimated_R0:.1f} | chaos={self.chaos_position:+.1f}")


class MemeCurator:
    """热梗策展数据管理器。"""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or CURATED_DATA
        self.memes: list[MemeEntry] = []
        self._load()

    def _load(self) -> None:
        """从 JSON 文件加载策展数据。"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"策展数据文件不存在: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for entry in raw.get("memes", []):
            if entry.get("narrative") != "同 meme_012":
                self.memes.append(MemeEntry(
                    meme_id=entry["id"],
                    name=entry["name"],
                    aliases=entry.get("aliases", []),
                    year=entry["year"],
                    peak_year=entry.get("peak_year", entry["year"]),
                    source=entry["source"],
                    source_platforms=entry.get("source_platforms", []),
                    category=entry.get("category", "未分类"),
                    tags=entry.get("tags", []),
                    lifecycle=entry.get("lifecycle", {}),
                    sentiment_arc=entry.get("sentiment_arc", []),
                    chaos_vector=entry.get("chaos_vector", {}),
                    propagation_model=entry.get("propagation_model", {}),
                    mutation_variants=entry.get("mutation_variants", []),
                    narrative=entry.get("narrative", ""),
                ))

    @property
    def meta(self) -> dict:
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f).get("_meta", {})

    # ── Query methods ──────────────────────────

    def by_category(self, category: str) -> list[MemeEntry]:
        """按类别筛选。"""
        return [m for m in self.memes if m.category == category]

    def by_year(self, year: int) -> list[MemeEntry]:
        """按年份筛选。"""
        return [m for m in self.memes if m.year == year]

    def by_years(self, start: int, end: int) -> list[MemeEntry]:
        """按年份范围筛选。"""
        return [m for m in self.memes if start <= m.year <= end]

    def by_tag(self, tag: str) -> list[MemeEntry]:
        """按标签筛选。"""
        return [m for m in self.memes if tag in m.tags]

    def by_chaos_range(self, lo: float, hi: float) -> list[MemeEntry]:
        """按混沌轴区间筛选。"""
        return [m for m in self.memes if lo <= m.chaos_position <= hi]

    def top_by_R0(self, n: int = 10) -> list[MemeEntry]:
        """R₀ 最高的 n 个热梗。"""
        return sorted(self.memes, key=lambda m: m.estimated_R0, reverse=True)[:n]

    def get(self, name: str) -> Optional[MemeEntry]:
        """按名称精确查找（支持别名）。"""
        for m in self.memes:
            if m.name == name or name in m.aliases:
                return m
        return None

    # ── Statistics ─────────────────────────────

    def category_distribution(self) -> dict[str, int]:
        """各类别热梗数量分布。"""
        dist = {}
        for m in self.memes:
            dist[m.category] = dist.get(m.category, 0) + 1
        return dist

    def chaos_by_year(self) -> dict[int, list[float]]:
        """按年份统计混沌轴分布。"""
        result: dict[int, list[float]] = {}
        for m in self.memes:
            if m.year not in result:
                result[m.year] = []
            result[m.year].append(m.chaos_position)
        return result

    def R0_by_category(self) -> dict[str, list[float]]:
        """按类别统计 R₀ 分布。"""
        result: dict[str, list[float]] = {}
        for m in self.memes:
            if m.category not in result:
                result[m.category] = []
            result[m.category].append(m.estimated_R0)
        return result

    def platform_distribution(self) -> dict[str, int]:
        """来源平台分布。"""
        dist = {}
        for m in self.memes:
            for p in m.source_platforms:
                dist[p] = dist.get(p, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

    # ── Export for modeling ────────────────────

    def to_sir_estimation(self) -> list[dict]:
        """将策展数据中的生命周期信息转换为 SIR 参数估算。

        使用 estimate_params_from_lifecycle 从 qualitative 数据估算
        beta, gamma, R₀。
        """
        from src.models.sir_meme import estimate_params_from_lifecycle

        estimations = []
        for m in self.memes:
            lc = m.lifecycle
            # Estimate duration from lifecycle
            duration_months = lc.get("duration_months", 3)
            if duration_months >= 999:
                duration_months = 18  # Cap for "indefinite"
            duration_days = duration_months * 30

            pm = m.propagation_model
            R0_qual = pm.get("estimated_R0", 1.0)

            # Estimate total infected from propagation scale
            circle_count = len(pm.get("circle_layers", []))
            if circle_count >= 5:
                total_infected_est = 0.75
            elif circle_count >= 3:
                total_infected_est = 0.50
            else:
                total_infected_est = 0.25

            params = estimate_params_from_lifecycle(
                peak_day=duration_days * 0.3,  # peak at ~30% of lifecycle
                total_infected=total_infected_est,
                duration_days=duration_days,
            )

            estimations.append({
                "id": m.meme_id,
                "name": m.name,
                "category": m.category,
                "year": m.year,
                "chaos_position": m.chaos_position,
                "R0_qualitative": R0_qual,
                "beta_estimated": params.beta,
                "gamma_estimated": params.gamma,
                "R0_estimated": params.R0,
                "total_infected_estimated": total_infected_est,
                "duration_days_estimated": duration_days,
            })

        return estimations

    def __repr__(self) -> str:
        return f"MemeCurator({len(self.memes)} memes, {self.data_path.name})"

    def stats_report(self) -> str:
        """生成数据集统计报告。"""
        lines = [
            f"=== MemeticChaos 策展数据集统计 ===",
            f"总热梗数: {len(self.memes)}",
            f"年份跨度: {min(m.year for m in self.memes)}–{max(m.year for m in self.memes)}",
            f"",
            f"--- 类别分布 ---",
        ]
        for cat, count in sorted(self.category_distribution().items()):
            lines.append(f"  {cat}: {count}")
        lines.append(f"")
        lines.append(f"--- 混沌轴分布 ---")
        chaos_vals = [m.chaos_position for m in self.memes]
        lines.append(f"  min: {min(chaos_vals):+.2f}  max: {max(chaos_vals):+.2f}")
        lines.append(f"  mean: {sum(chaos_vals)/len(chaos_vals):+.2f}")
        lines.append(f"")
        lines.append(f"--- R₀ 分布 ---")
        R0s = [m.estimated_R0 for m in self.memes]
        lines.append(f"  min: {min(R0s):.1f}  max: {max(R0s):.1f}")
        lines.append(f"  mean: {sum(R0s)/len(R0s):.1f}")
        lines.append(f"  R₀>1 (会爆发): {sum(1 for r in R0s if r > 1)} / {len(R0s)}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════
# Script entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    curator = MemeCurator()
    print(curator.stats_report())
    print()
    print("--- Top 10 by R₀ ---")
    for m in curator.top_by_R0(10):
        print(f"  {m.summary()}")

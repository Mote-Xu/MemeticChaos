"""
Semantic State Series — scraper 月度语义状态 → 前向分析层终点 (2026-07-09)

把 `monthly_semantic_state.json` 里 scraper 月份的分布特征提取为固定 schema 的月度状态
向量, 对齐连续时间线, 作为**前向分析层的数据终点**。每个新 scraper 月自动流入。

═══ 必须诚实面对的结构事实 ═══
历史 127 月 (2015-01~2025-12) 的特征 = Level 1 阶段占比 + 外部场 PCA + Trends 注意力。
scraper 语义标量 = cov_trace / 各向异性 / 语义漂移 / POS 熵 / 平台散度 (2026-06 起)。
**两套特征族在时间上完全不重叠** —— 历史段无 embedding, 2026 段无 Level 1/Trends。
所以**不能**把 scraper 月当作新行拼进 127 月矩阵 (那些行的历史列全空, 反之亦然)。

正确做法: scraper 语义标量是一个**独立的前向状态表示**。它不追溯续接历史链, 而是
从 2026-06 开始积累。当积累到足够月数 (≥~24), 这个序列可以跑它自己的 PCA/regime 分析
—— 历史链在**更高分辨率数据上的前向对应物**。本脚本先把管道终点建好, 让每月自动入列。

对应 Evidence Ledger: 本产出是 E2 (操作结果), 挂靠 narrative-as-observation 假设。

固定 schema 的月度语义状态向量 (6 维):
  semantic_dispersion   = covariance_trace          语义色散/创新率 (共识分散 vs 统一)
  attention_conc        = attention_concentration    注意力集中度 (pairwise 相似度)
  flash_skew            = distance skewness_ratio     长尾闪洪 (p90/median)
  pos_entropy           = pos_entropy.entropy         叙事化程度 proxy (词性熵)
  platform_divergence   = platform_jsd.mean_cosine    跨平台姿态散度 (H1 趋同 vs H2 各说各)
  semantic_drift        = 1 - cos(mean_emb[t], mean_emb[t-1])  月间语义漂移 (需 ≥2 月)

用法:
  conda run -n MemeticChaos python src/analysis/semantic_state_series.py
  conda run -n MemeticChaos python src/analysis/semantic_state_series.py --json
"""

import json, sys, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent
STATE_PATH = ROOT / "data" / "processed" / "monthly_semantic_state.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "semantic_state_series.json"

FEATURE_SCHEMA = [
    "semantic_dispersion", "attention_conc", "flash_skew",
    "pos_entropy", "platform_divergence", "semantic_drift",
]


def extract_vector(state: dict, prev_mean: np.ndarray | None) -> tuple[dict, np.ndarray]:
    """从单月 state 提取 6 维语义状态向量 + 返回本月 mean_emb 供下月算漂移。"""
    mean_emb = np.array(state["monthly_mean_embedding"], dtype=float)

    if prev_mean is not None:
        # 两个 mean_emb 已 L2 归一化 (aggregator 里做过), cosine = dot
        cos = float(np.dot(mean_emb, prev_mean) /
                    (np.linalg.norm(mean_emb) * np.linalg.norm(prev_mean) + 1e-10))
        drift = 1.0 - cos
    else:
        drift = None  # 首月无前月, 漂移未定义

    vec = {
        "semantic_dispersion": round(float(state["covariance_trace"]), 4),
        "attention_conc": round(float(state["attention_concentration"]), 4),
        "flash_skew": round(float(state["distance_distribution"]["skewness_ratio"]), 4),
        "pos_entropy": round(float(state["pos_entropy"].get("entropy", 0.0)), 4),
        "platform_divergence": (
            round(float(state["platform_jsd"]["mean_cosine_distance"]), 4)
            if state["platform_jsd"].get("mean_cosine_distance") is not None else None),
        "semantic_drift": round(drift, 4) if drift is not None else None,
    }
    return vec, mean_emb


def build_series() -> dict:
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    scraper_states = [(s["month"], s["state"]) for s in data["monthly_states"]
                      if s["has_scraper_data"] and s["state"]]
    scraper_states.sort(key=lambda x: x[0])

    series = []
    prev_mean = None
    for month, state in scraper_states:
        vec, prev_mean = extract_vector(state, prev_mean)
        series.append({
            "month": month,
            "n_headlines": state["n_headlines"],
            "features": vec,
        })

    scraper_months = [m for m, _ in scraper_states]
    return {
        "source": "semantic_state_series.py",
        "role": "前向分析层数据终点 (forward analysis layer endpoint)",
        "evidence_grade": "E2 (操作结果)",
        "depends_on_assumption": ["narrative-as-observation"],
        "feature_schema": FEATURE_SCHEMA,
        "disjointness_note": (
            "本序列与历史 127 月特征矩阵在时间上不重叠, 不可拼接 —— 历史段无 embedding, "
            "2026 段无 Level 1/Trends。这是独立前向状态表示, 从 2026-06 起积累, "
            "≥~24 月后可跑自身 PCA/regime 分析 (历史链的高分辨率前向对应物)。"
        ),
        "n_scraper_months": len(scraper_months),
        "scraper_months": scraper_months,
        "months_needed_for_analysis": max(0, 24 - len(scraper_months)),
        "series": series,
    }


def main():
    ap = argparse.ArgumentParser(description="Semantic State Series — 前向分析层终点")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--save", action="store_true", default=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    result = build_series()

    print("=" * 64)
    print("Semantic State Series — 前向分析层数据终点")
    print("=" * 64)
    print(f"\n⚠ {result['disjointness_note']}")
    print(f"\nscraper 月份: {result['n_scraper_months']} "
          f"({', '.join(result['scraper_months']) if result['scraper_months'] else '无'})")
    print(f"距离可独立分析 (≥24月) 还差: {result['months_needed_for_analysis']} 月")

    if result["series"]:
        print(f"\n{'月份':<9s} {'色散':>7s} {'注意力':>7s} {'闪洪':>6s} {'POS熵':>6s} {'平台散度':>8s} {'漂移':>7s}  headlines")
        for row in result["series"]:
            f = row["features"]
            def fmt(v, w, p=4):
                return f"{v:>{w}.{p}f}" if v is not None else f"{'—':>{w}s}"
            print(f"  {row['month']:<7s} {fmt(f['semantic_dispersion'],7)} "
                  f"{fmt(f['attention_conc'],7)} {fmt(f['flash_skew'],6,2)} "
                  f"{fmt(f['pos_entropy'],6,2)} {fmt(f['platform_divergence'],8)} "
                  f"{fmt(f['semantic_drift'],7)}   {row['n_headlines']}")

    if args.save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n已保存 → {OUTPUT_PATH}")
        print("下一步: 每月 scraper 聚合后自动追加; 积累 ≥24 月后接 representation_learning 前向分支。")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

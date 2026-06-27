"""
从 Google Trends 真实数据重建完整项目

替换所有手工估算的 lifecycle 参数为真实注意力曲线数据。

用法:
    python src/data/rebuild_from_trends.py
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════
# 1. 加载真实数据 + 映射到策展梗
# ═══════════════════════════════════════════════

# Google Trends keyword → curated meme_id
TREND_TO_MEME = {
    "打工人": "meme_001",
    "内卷": "meme_002",
    "躺平": "meme_003",
    "普信男": "meme_004",
    "小镇做题家": "meme_005",
    "摆烂": "meme_006",
    "润": "meme_007",
    "吗喽": "meme_008",
    "鼠鼠": "meme_009",
    "牛马": "meme_010",
    "i人 e人": "meme_011",
    "遥遥领先 华为": "meme_012",
    "孔乙己的长衫": "meme_013",
    "精神状态": "meme_014",
    "雪糕刺客": "meme_015",
    "谢帝 迪士尼": "meme_016",
    "科目三": "meme_017",
    "尊嘟假嘟": "meme_018",
    "鸡你太美": "meme_020",
    "后浪": "meme_021",
    "情绪价值": "meme_022",
    "不婚不育": "meme_023",
    "显眼包": "meme_024",
    "泼天富贵": "meme_025",
    "服美役": "meme_027",
    "建议专家不要建议": "meme_028",
    "命运的齿轮": "meme_029",
    "原生家庭": "meme_030",
    "发疯文学": "meme_026",
}


def load_trends(path: str = "data/collector/google_trends_2015_2025.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("memes", {})


def extract_metrics(meme_id: str, trends: dict) -> dict:
    """从真实注意力曲线提取所有 lifecycle 指标。"""
    # Find matching trend keyword
    trend_name = None
    for kw, mid in TREND_TO_MEME.items():
        if mid == meme_id:
            trend_name = kw
            break

    if trend_name is None or trend_name not in trends:
        return None

    curve = trends[trend_name]
    months = sorted(curve.keys())
    values = np.array([curve[m] for m in months])

    if len(values) < 3 or values.max() < 1:
        return None

    peak_idx = int(np.argmax(values))
    peak_val = float(values.max())
    peak_month = months[peak_idx]
    peak_year = int(peak_month[:4])

    threshold = max(0.5, peak_val * 0.10)
    above = values >= threshold
    above_indices = np.where(above)[0]
    emergence_month = months[above_indices[0]] if len(above_indices) > 0 else months[0]
    decay_month = months[above_indices[-1]] if len(above_indices) > 0 else months[-1]

    emergence_year = int(emergence_month[:4])
    decay_year = int(decay_month[:4])
    duration_months = max(1, int((datetime.strptime(decay_month, "%Y-%m") -
                                   datetime.strptime(emergence_month, "%Y-%m")).days / 30))

    # AUC (total attention)
    total_attention = float(np.trapezoid(values))

    # Resurgence: re-exceeds 30% of peak after decay
    post_peak = values[peak_idx:]
    resurgence_idx = np.where(post_peak > peak_val * 0.3)[0]
    has_resurgence = len(resurgence_idx) > 0 and resurgence_idx[0] > len(post_peak) * 0.5

    # Peak intensity proxy: normalize by max possible (100)
    peak_ratio = float(peak_val / 100.0)

    # Duration in years
    duration_years = round(duration_months / 12.0, 1)

    return {
        "peak_month": str(peak_month),
        "peak_value": float(peak_val),
        "peak_year": int(peak_year),
        "emergence_month": str(emergence_month),
        "decay_month": str(decay_month),
        "duration_months": int(duration_months),
        "duration_years": float(duration_years),
        "total_attention": float(total_attention),
        "has_resurgence": bool(has_resurgence),
        "peak_ratio": float(peak_ratio),
        "n_data_points": int(len(values)),
    }


# ═══════════════════════════════════════════════
# 2. 用真实数据更新策展 JSON
# ═══════════════════════════════════════════════

def update_curated_with_real_data():
    """将真实注意力数据回填到 memes_2020_2025.json。"""
    trends = load_trends()

    with open("data/curated/memes_2020_2025.json", "r", encoding="utf-8") as f:
        curated = json.load(f)

    updated = 0
    for meme in curated["memes"]:
        mid = meme["id"]
        metrics = extract_metrics(mid, trends)
        if metrics is None:
            continue

        # 更新 lifecycle
        lc = meme["lifecycle"]
        lc["emergence"] = metrics["emergence_month"]
        lc["peak"] = metrics["peak_month"]
        lc["decay"] = metrics["decay_month"]
        lc["duration_months"] = metrics["duration_months"]
        lc["real_peak_value"] = metrics["peak_value"]
        lc["real_total_attention"] = metrics["total_attention"]
        lc["has_resurgence"] = metrics["has_resurgence"]
        lc["data_source"] = "Google Trends 2015-2025"

        # 更新 propagation_model
        pm = meme["propagation_model"]
        pm["real_peak_ratio"] = metrics["peak_ratio"]

        updated += 1
        print(f"  {meme['name']:<12s}: peak={metrics['peak_month']} ({metrics['peak_value']:.0f}) "
              f"dur={metrics['duration_months']}m  resurgence={metrics['has_resurgence']}")

    curated["_meta"]["last_rebuilt_from"] = "Google Trends 2015-2025"
    curated["_meta"]["rebuilt_at"] = datetime.now().isoformat()
    curated["_meta"]["n_with_real_data"] = updated

    out_path = "data/curated/memes_2020_2025.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(curated, f, ensure_ascii=False, indent=2)

    print(f"\n[更新] {updated}/{len(curated['memes'])} 梗已注入真实数据 → {out_path}")
    return curated


# ═══════════════════════════════════════════════
# 3. 重建 Trajectories + 相图
# ═══════════════════════════════════════════════

def rebuild_all():
    print("=" * 60)
    print("从 Google Trends 真实数据重建项目")
    print("=" * 60)

    # Step 1: 更新策展数据
    print("\n── 1. 注入真实注意力数据 ──")
    update_curated_with_real_data()

    # Step 2: 重建 Trajectories
    print("\n── 2. 重建 29 条 MemeTrajectory ──")
    from src.trajectory.meme_trajectory import TrajectoryBuilder
    builder = TrajectoryBuilder()
    trajectories = builder.build_all()
    builder.save_all(trajectories)

    # Step 3: 验证
    print("\n── 3. 验证 ──")
    from src.constraint.delta_transition import validate_trajectory
    valid = 0
    for t in trajectories:
        phases = [n.phase for n in t.nodes]
        constraints = [n.constraint_state.vector for n in t.nodes]
        nodes_dict = [n.to_dict() for n in t.nodes]
        if validate_trajectory(phases, constraints, nodes_dict)["valid"]:
            valid += 1
    print(f"  {valid}/{len(trajectories)} 轨迹通过验证")

    # Step 4: 可视化
    print("\n── 4. 生成新相图 ──")
    from src.trajectory.trajectory_viz import plot_trajectory_phase_diagram
    rich = [t for t in trajectories if t.phase_count >= 3]
    plot_trajectory_phase_diagram(
        [t.to_dict() for t in trajectories],
        save_path="data/processed/trajectory_phase_diagram_real.png",
        show_labels=True,
    )
    print("  已保存 → data/processed/trajectory_phase_diagram_real.png")

    print(f"\n{'='*60}")
    print("重建完成。相图现在基于真实注意力数据。")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rebuild_all()

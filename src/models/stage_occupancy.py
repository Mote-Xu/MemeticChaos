"""
叙事 Stage Occupancy v2 — 用 Google Trends 峰值锚定阶段边界

v1 的问题: 叙事 JSON 里的时间范围太粗("2020年-2025年")→ Emergence 占 50%。
v2 的改进: 用 Trends 真实月度数据 + 叙事描述的 phase 顺序来精确定位阶段时间。

用法:
    python src/models/stage_occupancy.py
"""

import json, sys, os, re
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
NARRATIVE_DIRS = [
    ROOT / "data/processed/narratives",
    ROOT / "data/processed/narratives_from_trends",
]
TRENDS_PATH = ROOT / "data/collector/google_trends_2015_2025.json"
OUTPUT_PATH = ROOT / "data/processed/stage_occupancy.json"

STAGE_ORDER = ["origin", "emergence", "peak", "controversy", "fixation"]

# Google Trends keyword → 映射
TREND_ALIASES = {
    "打工人": ["打工人"], "内卷": ["内卷", "内卷 / 卷"],
    "躺平": ["躺平"], "普信男": ["普信男", "普信"],
    "小镇做题家": ["小镇做题家"], "摆烂": ["摆烂"],
    "润": ["润"], "吗喽": ["吗喽"], "鼠鼠": ["鼠鼠"], "牛马": ["牛马"],
    "i人/e人": ["i人 e人"], "遥遥领先": ["遥遥领先", "遥遥领先 华为"],
    "孔乙己的长衫": ["孔乙己的长衫", "孔乙己 长衫"],
    "精神状态": ["精神状态"],
    "雪糕刺客": ["XX刺客", "雪糕刺客"],
    "科目三": ["科目三"], "鸡你太美": ["鸡你太美"],
    "后浪": ["后浪"], "情绪价值": ["情绪价值"],
    "原生家庭": ["原生家庭"], "专家建议": ["专家建议", "建议专家不要建议"],
    "不结婚": ["不结婚", "不婚不育"],
    "显眼包": ["显眼", "显眼包"],
    "凡尔赛": ["凡尔赛"], "元宇宙": ["元宇宙"],
    "citywalk": ["citywalk"], "芭比Q": ["芭比Q"], "栓Q": ["栓Q"],
    "美拉德": ["美拉德"], "南方小土豆": ["南方小土豆"],
    "破防": ["破防"], "社恐": ["社恐"], "社死": ["社死"],
    "精神内耗": ["精神内耗"], "尊嘟假嘟": ["尊嘟假嘟"],
    "发疯文学": ["发疯文学 梗"], "多巴胺穿搭": ["多巴胺穿搭"],
}


def load_trends() -> dict:
    with open(TRENDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["memes"]


def load_narratives() -> dict:
    narratives = {}
    for nar_dir in NARRATIVE_DIRS:
        if not nar_dir.exists():
            continue
        for fp in nar_dir.glob("*.json"):
            if fp.name.startswith("_"):
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("meme_name", fp.stem)
                narratives[name] = data
            except Exception:
                pass
    return narratives


def find_trends_key(meme_name: str, trends: dict) -> str:
    """匹配叙事名称到 Google Trends 关键词."""
    for alias, candidates in TREND_ALIASES.items():
        if meme_name in candidates or any(c in meme_name for c in candidates):
            if alias in trends:
                return alias
    # Direct match
    if meme_name in trends:
        return meme_name
    # Substring match
    for tk in trends:
        if meme_name[:4] in tk or tk[:4] in meme_name:
            return tk
    return None


def detect_peak_month(trends_data: dict) -> str:
    """从 Trends 数据中找到峰值月份（超过月均 2 std 且最高值）."""
    months = sorted(trends_data.keys())
    values = [trends_data[m] for m in months]
    if not values or max(values) == 0:
        return None
    mean_v = np.mean(values)
    std_v = np.std(values) if np.std(values) > 0 else 1
    peak_idx = int(np.argmax(values))
    if values[peak_idx] > mean_v + 1.5 * std_v:
        return months[peak_idx]
    # Just take the max if variance is low
    return months[peak_idx]


def build_stage_timeline_from_trends(meme_name: str, narrative: dict, trends: dict) -> list[dict]:
    """用 Trends 数据 + 叙事信息构建精确阶段时间线.

    逻辑:
    1. 从叙事 JSON 中获知该梗经历了哪些 phase（特征，不是时间）
    2. 从 Trends 中找到注意力峰值月份
    3. 以峰值为中心，划分 origin/emerge/peak/controversy/fixation 的时间范围
    """
    key = find_trends_key(meme_name, trends)
    if not key:
        return []

    data = trends[key]
    peak_month = detect_peak_month(data)
    if not peak_month:
        return []

    months = sorted(data.keys())

    # Find threshold: months where attention > 10% of peak
    peak_val = data[peak_month]
    threshold = max(peak_val * 0.1, 2.0)

    active_months = [m for m in months if data[m] >= threshold]
    if len(active_months) < 3:
        return []

    first_active = active_months[0]
    last_active = active_months[-1]

    peak_idx = months.index(peak_month)
    em_span = min(4, peak_idx)  # emergence: 4 months before peak

    # origin: from data start until emergence begins
    origin_end_idx = max(0, peak_idx - em_span - 1)
    origin_start = months[0]
    origin_end = months[origin_end_idx]

    # emergence: 4 months before peak (or all pre-peak if less data)
    emergence_start_idx = max(0, peak_idx - em_span)
    emergence_start = months[emergence_start_idx]
    emergence_end = months[peak_idx - 1] if peak_idx > 0 else months[peak_idx]

    # peak: peak_month ± 2 months
    pk_start = months[max(0, peak_idx - 2)]
    pk_end = months[min(len(months) - 1, peak_idx + 2)]

    # controversy: 3-5 months after peak end, if attention > 30% of peak
    ct_threshold = peak_val * 0.30
    ct_start_idx = min(len(months) - 1, peak_idx + 3)
    ct_start = months[ct_start_idx]
    # Find where attention drops below threshold
    ct_end_idx = ct_start_idx
    for j in range(ct_start_idx, min(len(months), peak_idx + 6)):
        if data[months[j]] < ct_threshold:
            ct_end_idx = j
            break
    else:
        ct_end_idx = min(len(months) - 1, peak_idx + 5)
    ct_end = months[ct_end_idx]

    # fixation: from controversy end to data end
    fix_start_idx = min(len(months) - 1, ct_end_idx + 1)
    fix_start = months[fix_start_idx]
    fix_end = months[-1]

    timeline = []
    if origin_start < emergence_start:
        timeline.append({"stage": "origin", "start": origin_start, "end": origin_end})
    if emergence_start <= emergence_end:
        timeline.append({"stage": "emergence", "start": emergence_start, "end": emergence_end})
    timeline.append({"stage": "peak", "start": pk_start, "end": pk_end})
    if ct_start <= ct_end:
        timeline.append({"stage": "controversy", "start": ct_start, "end": ct_end})
    if fix_start <= fix_end:
        timeline.append({"stage": "fixation", "start": fix_start, "end": fix_end})

    return timeline


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("Stage Occupancy v2 — 基于 Trends 峰值的精确阶段")
    print("=" * 60)

    narratives = load_narratives()
    trends = load_trends()

    all_months = set()
    for d in trends.values():
        all_months.update(d.keys())
    months = sorted(m for m in all_months if "2015" <= m[:4] <= "2025")
    print(f"月份: {len(months)} ({months[0]} → {months[-1]})")

    # Build timelines
    meme_timelines = {}
    matched = 0
    for name, nar in narratives.items():
        tl = build_stage_timeline_from_trends(name, nar, trends)
        if tl:
            meme_timelines[name] = tl
            matched += 1

    print(f"有 Trends 匹配的叙事: {matched}/{len(narratives)}")

    # Build occupancy matrix
    n_stages = len(STAGE_ORDER)
    n_months = len(months)
    matrix = np.zeros((n_months, n_stages))

    for mi, month in enumerate(months):
        stage_counts = defaultdict(int)
        active_count = 0
        for name, tl in meme_timelines.items():
            for entry in tl:
                if entry["start"] <= month <= entry["end"]:
                    stage_counts[entry["stage"]] += 1
                    active_count += 1
                    break
        if active_count > 0:
            for si, stage in enumerate(STAGE_ORDER):
                matrix[mi, si] = stage_counts[stage] / active_count

    # Stats
    print(f"\n阶段统计 ({len(meme_timelines)} 条叙事):")
    for si, stage in enumerate(STAGE_ORDER):
        col = matrix[:, si]
        print(f"  {stage:<15s}: mean={col.mean():.3f}, max={col.max():.3f}, "
              f"nonzero={(col > 0).sum()}/{len(months)} 月")

    # Show recent months
    for offset in [0, -1, -3, -6, -12]:
        if abs(offset) >= len(months):
            continue
        idx = len(months) + offset - 1 if offset < 0 else offset
        if idx < 0 or idx >= len(months):
            continue
        m = months[idx]
        print(f"\n  {m}:", end="")
        for si, stage in enumerate(STAGE_ORDER):
            v = matrix[idx, si]
            if v > 0:
                print(f" {stage}={v:.0%}", end="")
        if matrix[idx].sum() == 0:
            print(" (无活跃叙事)", end="")

    # Save
    output = {
        "months": months,
        "stages": STAGE_ORDER,
        "matrix": matrix.tolist(),
        "n_narratives": len(narratives),
        "n_with_timelines": matched,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"\n\n已保存 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

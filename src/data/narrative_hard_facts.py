"""
Level 1 Hard Facts Extraction — P0 for FR19 v4.0

从 58 条叙事 JSON 中提取四类硬事实，按月流量权重聚合为时间序列:
  1. Stage Occupancy — 复用 stage_occupancy.json (已计算)
  2. Mutation_Occurred — 梗是否发生变异 (布尔 → 月度比率)
  3. Institutionalized — 是否被官方/主流引用 (布尔 → 月度比率)
  4. Semantic_Drift_Distance — 原义→现义的语义距离 (标量 → 月度加权均值)

AlphaGo 原则: 不硬编码理论定义。Stage 复用 Trends 锚定的阶段。
Mutation 从 JSON mutations 数组检测。Institutionalized 用关键词检测
官方媒体提及。Semantic Drift 用 TF-IDF 字符 n-gram (中文适配)
计算余弦距离——纯统计方法，零理论预设。

用法:
    python src/data/narrative_hard_facts.py
    python src/data/narrative_hard_facts.py --output data/processed/level1_hard_facts.json
"""

import json, sys, os, re, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).parent.parent.parent

NARRATIVE_DIRS = [
    ROOT / "data/processed/narratives",
    ROOT / "data/processed/narratives_from_trends",
]
TRENDS_PATH = ROOT / "data/collector/google_trends_2015_2025.json"
STAGE_OCCUPANCY_PATH = ROOT / "data/processed/stage_occupancy.json"
DEFAULT_OUTPUT = ROOT / "data/processed/level1_hard_facts.json"

# ── Institutionalization keywords ──
# 官方媒体/机构/主流平台引用信号
INSTITUTIONAL_KEYWORDS = [
    "人民日报", "新华社", "央视", "CCTV", "光明日报", "中国青年报",
    "环球时报", "官方", "国务院", "教育部", "广电", "网信办",
    "政府", "两会", "人大代表", "政协委员", "政法委", "最高法",
    "新闻联播", "焦点访谈", "中央", "国家", "部委", "党委",
    "中国新闻周刊", "澎湃新闻", "新京报", "南方周末",
]

# ── Mutation active window: mutation 发生后多少个月内算 "active" ──
MUTATION_ACTIVE_MONTHS = 6

# ── Time parsing ──

def parse_year_month(time_str: str) -> list[str]:
    """解析时间字符串，返回包含的月份列表.

    >>> parse_year_month("2021年4月-5月")
    ['2021-04', '2021-05']
    >>> parse_year_month("2021-06")
    ['2021-06']
    >>> parse_year_month("2020-2021")
    ['2020-01', ..., '2021-12']
    """
    months = []
    time_str = time_str.strip()

    # "2021年4月-5月" / "2021年4月至5月"
    m = re.match(r"(\d{4})\s*年\s*(\d{1,2})\s*月?\s*[-至到]\s*(\d{4})?\s*年?\s*(\d{1,2})?\s*月?", time_str)
    if m:
        y1, mo1 = int(m.group(1)), int(m.group(2))
        y2 = int(m.group(3)) if m.group(3) else y1
        mo2 = int(m.group(4)) if m.group(4) else mo1
        for y in range(y1, y2 + 1):
            start_m = mo1 if y == y1 else 1
            end_m = mo2 if y == y2 else 12
            for m in range(start_m, end_m + 1):
                months.append(f"{y}-{m:02d}")
        return months

    # "2021年5月-6月" alternative
    m = re.match(r"(\d{4})年(\d{1,2})月?[-至到](\d{1,2})月?", time_str)
    if m:
        y, mo1, mo2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        for mo in range(mo1, mo2 + 1):
            months.append(f"{y}-{mo:02d}")
        return months

    # "2020年-2025年" / "2020-2025"
    m = re.match(r"(\d{4})\s*年?\s*[-至到]\s*(\d{4})\s*年?", time_str)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        for y in range(y1, y2 + 1):
            for mo in range(1, 13):
                months.append(f"{y}-{mo:02d}")
        return months

    # "2018年" / "2019"
    m = re.match(r"(\d{4})\s*年?", time_str)
    if m:
        y = int(m.group(1))
        for mo in range(1, 13):
            months.append(f"{y}-{mo:02d}")
        return months

    # "2023年下半年至今"
    m = re.match(r"(\d{4})年(上|下)半年", time_str)
    if m:
        y = int(m.group(1))
        half = m.group(2)
        start_m = 1 if half == "上" else 7
        end_m = 6 if half == "上" else 12
        for mo in range(start_m, end_m + 1):
            months.append(f"{y}-{mo:02d}")
        return months

    # "2021年4月" single month
    m = re.match(r"(\d{4})\s*年\s*(\d{1,2})\s*月?", time_str)
    if m:
        months.append(f"{int(m.group(1))}-{int(m.group(2)):02d}")
        return months

    # "2021-06" already formatted
    m = re.match(r"(\d{4})-(\d{2})", time_str)
    if m:
        months.append(f"{int(m.group(1))}-{int(m.group(2)):02d}")
        return months

    # "2021-06 至 2021-07" / "2021-08至2021-12"
    m = re.match(r"(\d{4}-\d{2})\s*至\s*(\d{4}-\d{2})", time_str)
    if m:
        start, end = m.group(1), m.group(2)
        sy, sm = int(start[:4]), int(start[5:7])
        ey, em = int(end[:4]), int(end[5:7])
        for y in range(sy, ey + 1):
            s_mo = sm if y == sy else 1
            e_mo = em if y == ey else 12
            for mo in range(s_mo, e_mo + 1):
                months.append(f"{y}-{mo:02d}")
        return months

    return months


def months_between(start_ym: str, end_ym: str) -> list[str]:
    """生成两个月份之间的所有月份 (含两端)."""
    sy, sm = int(start_ym[:4]), int(start_ym[5:7])
    ey, em = int(end_ym[:4]), int(end_ym[5:7])
    result = []
    for y in range(sy, ey + 1):
        s = sm if y == sy else 1
        e = em if y == ey else 12
        for m in range(s, e + 1):
            result.append(f"{y}-{m:02d}")
    return result


def parse_time_range(time_range: str) -> list[str]:
    """解析 spread_phase 的 time_range 字段，返回月份列表."""
    if not time_range:
        return []

    tr = time_range.strip()

    # "2021-06 至 2021-07"
    m = re.match(r"(\d{4}-\d{2})\s*至\s*(\d{4}-\d{2})", tr)
    if m:
        return months_between(m.group(1), m.group(2))

    # "2020-05至2021-05"
    m = re.match(r"(\d{4}-\d{2})至(\d{4}-\d{2})", tr)
    if m:
        return months_between(m.group(1), m.group(2))

    # Single month "2019-04"
    m = re.match(r"^(\d{4}-\d{2})$", tr)
    if m:
        return [m.group(1)]

    # Fall back to Chinese parser
    return parse_year_month(tr)


# ── Feature extractors ──

def extract_mutation_months(narrative: dict) -> set[str]:
    """返回该梗有活跃变异的月份集合."""
    active_months = set()

    # 1. From mutations array
    mutations = narrative.get("mutations", [])
    for mut in mutations:
        if isinstance(mut, dict):
            time_str = mut.get("time") or ""
            if not time_str:
                continue
            mut_months = parse_year_month(time_str)
            # Extend active window
            for m in mut_months:
                active_months.add(m)
                # Extend forward by MUTATION_ACTIVE_MONTHS
                y, mo = int(m[:4]), int(m[5:7])
                for offset in range(1, MUTATION_ACTIVE_MONTHS + 1):
                    mo2 = mo + offset
                    y2 = y + (mo2 - 1) // 12
                    mo2 = ((mo2 - 1) % 12) + 1
                    active_months.add(f"{y2}-{mo2:02d}")

    # 2. From spread_phases with "变异" in phase name
    phases = narrative.get("spread_phases", [])
    for ph in phases:
        phase_name = ph.get("phase", "")
        if "变异" in phase_name:
            tr = ph.get("time_range", "")
            ph_months = parse_time_range(tr)
            active_months.update(ph_months)

    return active_months


def extract_institutionalized(narrative: dict) -> bool:
    """检测梗是否被官方/主流机构引用.

    Level 1 做法: 纯关键词匹配, 不做语义理解.
    检测 social_context 中的 backlash_events, triggers 等字段.
    """
    # Check social_context
    sc = narrative.get("social_context", {})
    if not isinstance(sc, dict):
        return False

    text_to_check = " ".join([
        str(sc.get("triggers", "")),
        str(sc.get("backlash_events", "")),
        str(sc.get("political_sensitivity", "")),
    ])

    for kw in INSTITUTIONAL_KEYWORDS:
        if kw in text_to_check:
            return True

    # Also check narrative_summary and social_context_hint
    summary = narrative.get("narrative_summary", "")
    hint = narrative.get("social_context_hint", "")
    extra_text = f"{summary} {hint}"
    for kw in INSTITUTIONAL_KEYWORDS:
        if kw in extra_text:
            return True

    return False


def extract_institutionalized_month(narrative: dict) -> str | None:
    """返回 institutionalized 发生的月份 (最早官方提及时间).

    从 backlash_events / triggers 中提取时间信息.
    如果无法确定具体月份，返回 None (只返回 bool 级别的信息).
    """
    sc = narrative.get("social_context", {})
    if not isinstance(sc, dict):
        return None

    text = " ".join([
        str(sc.get("triggers", "")),
        str(sc.get("backlash_events", "")),
    ])

    # Try to find a year mention near institutional keywords
    for kw in INSTITUTIONAL_KEYWORDS:
        if kw in text:
            # Look for year patterns near the keyword
            idx = text.find(kw)
            window = text[max(0, idx - 50):idx + 100]
            m = re.search(r"(\d{4})\s*年", window)
            if m:
                return f"{int(m.group(1))}-01"  # default to January

    return None


def compute_semantic_drift(narrative: dict, model) -> float | None:
    """计算语义漂移距离 (embedding 余弦距离).

    使用 sentence-transformers 多语言模型对 original_meaning 和
    current_meaning 编码，返回 1 - cosine_similarity.
    纯统计方法——预训练模型不编码我们的理论。
    """
    sd = narrative.get("semantic_drift")
    if not isinstance(sd, dict):
        return None

    orig = sd.get("original_meaning", "")
    curr = sd.get("current_meaning", "")

    if not orig or not curr:
        return None

    try:
        embeddings = model.encode([orig, curr], show_progress_bar=False)
        cos_sim = float(np.dot(embeddings[0], embeddings[1]) /
                         (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
        return float(1.0 - cos_sim)
    except Exception:
        return None


def compute_all_semantic_drifts(narratives: dict, model) -> dict[str, float | None]:
    """批量计算所有叙事的语义漂移距离."""
    # Collect all pairs that need encoding
    pairs = []
    keys = []
    for name, nar in narratives.items():
        sd = nar.get("semantic_drift")
        if isinstance(sd, dict) and sd.get("original_meaning") and sd.get("current_meaning"):
            pairs.append(sd["original_meaning"])
            pairs.append(sd["current_meaning"])
            keys.append(name)
        else:
            keys.append(name)
            pairs.append("")
            pairs.append("")

    # Batch encode all at once
    if not any(pairs):
        return {name: None for name in narratives}

    try:
        embeddings = model.encode(pairs, show_progress_bar=True)
    except Exception:
        return {name: None for name in narratives}

    results = {}
    for i, name in enumerate(keys):
        idx = i * 2
        if idx + 1 < len(embeddings) and pairs[idx] and pairs[idx + 1]:
            cos_sim = float(np.dot(embeddings[idx], embeddings[idx + 1]) /
                            (np.linalg.norm(embeddings[idx]) * np.linalg.norm(embeddings[idx + 1])))
            results[name] = float(1.0 - cos_sim)
        else:
            results[name] = None

    return results


# ── Main pipeline ──

def load_narratives() -> dict[str, dict]:
    """加载所有叙事 JSON."""
    narratives = {}
    for nar_dir in NARRATIVE_DIRS:
        if not nar_dir.exists():
            print(f"  ⚠ 目录不存在: {nar_dir}")
            continue
        for fp in sorted(nar_dir.glob("*.json")):
            if fp.name.startswith("_"):
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("meme_name", fp.stem)
                # Prefer B站 narratives over trends when duplicate names exist
                if name not in narratives or "_source" in data:
                    narratives[name] = data
            except Exception as e:
                print(f"  ⚠ 加载失败: {fp.name} — {e}")
    return narratives


def load_trends() -> dict[str, dict[str, float]]:
    """加载 Google Trends 数据."""
    with open(TRENDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["memes"]


def load_stage_occupancy() -> dict:
    """加载已有 stage_occupancy.json."""
    with open(STAGE_OCCUPANCY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def match_narrative_to_trends(meme_name: str, trends: dict) -> str | None:
    """匹配叙事名到 Google Trends 关键词."""
    # Direct match
    if meme_name in trends:
        return meme_name

    # Common aliases
    aliases = {
        "普信男": ["普信男", "普信"],
        "i人/e人": ["i人 e人"],
        "i人_e人": ["i人 e人"],
        "遥遥领先": ["遥遥领先", "遥遥领先 华为"],
        "遥遥领先_华为": ["遥遥领先", "遥遥领先 华为"],
        "孔乙己的长衫": ["孔乙己的长衫", "孔乙己 长衫"],
        "尊嘟假嘟": ["尊嘟假嘟"],
        "不结婚": ["不结婚", "不婚不育"],
        "显眼": ["显眼", "显眼包"],
        "显眼包": ["显眼", "显眼包"],
        "凡尔赛": ["凡尔赛"],
        "元宇宙": ["元宇宙"],
        "citywalk": ["citywalk"],
        "芭比Q": ["芭比Q"],
        "栓Q": ["栓Q"],
        "美拉德": ["美拉德"],
        "南方小土豆": ["南方小土豆"],
        "破防": ["破防"],
        "社恐": ["社恐"],
        "社死": ["社死"],
        "精神内耗": ["精神内耗"],
        "发疯文学": ["发疯文学 梗"],
        "多巴胺穿搭": ["多巴胺穿搭"],
        "鸡你太美": ["鸡你太美"],
        "鸡你胎没": ["鸡你太美"],
        "鼠鼠": ["鼠鼠"],
        "吗喽": ["吗喽"],
        "牛马": ["牛马"],
    }
    for alias, candidates in aliases.items():
        if meme_name in candidates or any(c in meme_name for c in candidates):
            if alias in trends:
                return alias

    # Substring match (bidirectional)
    for tk in trends:
        if len(tk) >= 2 and (meme_name[:4] in tk or tk[:4] in meme_name):
            return tk

    return None


def build_monthly_series(
    narratives: dict,
    trends: dict,
    stage_data: dict,
    model,
) -> dict:
    """构建月度聚合时间序列."""

    # ── Step 1: Extract per-meme features ──
    # Batch compute semantic drifts
    drift_results = compute_all_semantic_drifts(narratives, model)

    meme_features = {}
    for name, nar in narratives.items():
        mut_months = extract_mutation_months(nar)
        inst = extract_institutionalized(nar)
        inst_month = extract_institutionalized_month(nar)
        drift = drift_results.get(name)

        meme_features[name] = {
            "mutation_months": mut_months,
            "institutionalized": inst,
            "institutionalized_month": inst_month,
            "semantic_drift": drift,
        }

    # ── Step 2: Map Trends → months ──
    all_months = set()
    for d in trends.values():
        all_months.update(d.keys())
    months = sorted(m for m in all_months if "2015" <= m[:4] <= "2025")

    # ── Step 3: Match narratives to Trends ──
    name_to_trends_key = {}
    for name in meme_features:
        key = match_narrative_to_trends(name, trends)
        if key:
            name_to_trends_key[name] = key

    # ── Step 4: Monthly aggregation ──
    n_months = len(months)
    mutation_rate = np.zeros(n_months)
    inst_rate = np.zeros(n_months)
    drift_mean = np.zeros(n_months)
    drift_std = np.zeros(n_months)
    active_count = np.zeros(n_months, dtype=int)
    total_traffic = np.zeros(n_months)

    for mi, month in enumerate(months):
        mut_weighted = 0.0
        inst_weighted = 0.0
        drift_weighted = 0.0
        drift_values = []
        traffic_sum = 0.0
        n_active = 0

        for name, feats in meme_features.items():
            tk = name_to_trends_key.get(name)
            if not tk:
                continue

            traffic = trends[tk].get(month, 0.0)
            if traffic <= 0:
                continue

            n_active += 1
            traffic_sum += traffic

            # Mutation
            if month in feats["mutation_months"]:
                mut_weighted += traffic

            # Institutionalized
            inst_m = feats.get("institutionalized_month")
            if inst_m and month >= inst_m:
                inst_weighted += traffic
            elif feats["institutionalized"] and inst_m is None:
                # Can't determine month — count from first appearance in Trends
                first_month = min(trends[tk].keys())
                if month >= first_month:
                    inst_weighted += traffic

            # Semantic drift
            if feats["semantic_drift"] is not None:
                drift_weighted += traffic * feats["semantic_drift"]
                drift_values.append(feats["semantic_drift"])

        if traffic_sum > 0:
            mutation_rate[mi] = mut_weighted / traffic_sum
            inst_rate[mi] = inst_weighted / traffic_sum
            drift_mean[mi] = drift_weighted / traffic_sum if drift_weighted > 0 else 0.0
        else:
            mutation_rate[mi] = 0.0
            inst_rate[mi] = 0.0
            drift_mean[mi] = 0.0

        drift_std[mi] = float(np.std(drift_values)) if len(drift_values) > 1 else 0.0
        active_count[mi] = n_active
        total_traffic[mi] = traffic_sum

    # ── Step 5: Assemble Stage Occupancy ──
    stage_months = stage_data["months"]
    stage_matrix = stage_data["matrix"]  # list of lists
    stage_order = stage_data["stages"]

    # Align stage occupancy to our months
    month_to_stage = {}
    for i, m in enumerate(stage_months):
        month_to_stage[m] = stage_matrix[i] if i < len(stage_matrix) else [0.0] * 5

    stage_aligned = []
    for m in months:
        stage_aligned.append(month_to_stage.get(m, [0.0] * 5))

    return {
        "source": "narrative_hard_facts.py — Level 1 extraction",
        "generated_at": None,  # filled by main()
        "months": months,
        "n_narratives_total": len(narratives),
        "n_narratives_matched": len(name_to_trends_key),
        "stages": stage_order,
        "stage_occupancy": stage_aligned,
        "mutation_rate": mutation_rate.tolist(),
        "institutionalization_rate": inst_rate.tolist(),
        "mean_semantic_drift": drift_mean.tolist(),
        "std_semantic_drift": drift_std.tolist(),
        "active_meme_count": active_count.tolist(),
        "total_traffic": total_traffic.tolist(),
        # Per-meme details for audit
        "per_meme": {
            name: {
                "has_mutations": len(feats["mutation_months"]) > 0,
                "n_mutation_months": len(feats["mutation_months"]),
                "institutionalized": feats["institutionalized"],
                "institutionalized_month": feats["institutionalized_month"],
                "semantic_drift": feats["semantic_drift"],
                "trends_key": name_to_trends_key.get(name),
            }
            for name, feats in meme_features.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Level 1 Hard Facts Extraction")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Level 1 Hard Facts — FR19 v4.0 P0")
    print("=" * 60)

    # Load
    print("\n[1/4] 加载叙事数据...")
    narratives = load_narratives()
    print(f"  加载 {len(narratives)} 条叙事")

    print("\n[2/4] 加载 Google Trends + Stage Occupancy...")
    trends = load_trends()
    stage_data = load_stage_occupancy()
    print(f"  Trends: {len(trends)} 关键词")
    print(f"  Stage Occupancy: {len(stage_data['months'])} 月 × {len(stage_data['stages'])} 阶段")

    # Init embedding model for semantic drift
    print("\n[3/4] 加载 embedding 模型 + 提取逐梗特征...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print(f"  模型: paraphrase-multilingual-MiniLM-L12-v2")

    # Extract and aggregate
    result = build_monthly_series(narratives, trends, stage_data, model)

    from datetime import datetime
    result["generated_at"] = datetime.now().isoformat()

    # ── Print stats ──
    months = result["months"]
    print(f"\n  月份范围: {months[0]} → {months[-1]} ({len(months)} 月)")
    print(f"  叙事匹配 Trends: {result['n_narratives_matched']}/{result['n_narratives_total']}")

    # Mutation stats
    n_with_mutations = sum(1 for m in result["per_meme"].values() if m["has_mutations"])
    print(f"  有变异的梗: {n_with_mutations}/{result['n_narratives_total']}")

    # Institution stats
    n_inst = sum(1 for m in result["per_meme"].values() if m["institutionalized"])
    print(f"  被主流引用的梗: {n_inst}/{result['n_narratives_total']}")

    # Drift stats
    drift_vals = [m["semantic_drift"] for m in result["per_meme"].values()
                  if m["semantic_drift"] is not None]
    print(f"  有语义漂移数据的梗: {len(drift_vals)}/{result['n_narratives_total']}")
    if drift_vals:
        print(f"  语义漂移范围: [{min(drift_vals):.3f}, {max(drift_vals):.3f}], "
              f"mean={np.mean(drift_vals):.3f}")

    # Monthly series stats
    mr = np.array(result["mutation_rate"])
    ir = np.array(result["institutionalization_rate"])
    dr = np.array(result["mean_semantic_drift"])

    nonzero_mr = (mr > 0).sum()
    nonzero_ir = (ir > 0).sum()
    nonzero_dr = (dr > 0).sum()

    print(f"\n  月度序列非零月数:")
    print(f"    mutation_rate:         {nonzero_mr}/{len(months)}")
    print(f"    institutionalization:  {nonzero_ir}/{len(months)}")
    print(f"    semantic_drift:        {nonzero_dr}/{len(months)}")

    # Print recent 6 months
    print(f"\n  最近 6 个月:")
    print(f"    {'Month':<8s} {'Active':>6s} {'Mut%':>6s} {'Inst%':>6s} {'Drift':>6s}")
    for i in range(max(0, len(months) - 6), len(months)):
        m = months[i]
        print(f"    {m:<8s} {result['active_meme_count'][i]:>6d} "
              f"{mr[i]:>6.1%} {ir[i]:>6.1%} {dr[i]:>6.3f}")

    # ── Save ──
    print(f"\n[4/4] 保存 → {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nDone. Level 1 hard facts ready for Level 2 representation learning.")


if __name__ == "__main__":
    main()

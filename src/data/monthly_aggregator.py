"""
月度语义状态聚合器 — Scraper v2.0 → 月度分析层

将日级 384 维 headline embedding 聚合为月级语义状态，
接入 FR19 Level 1/2 的月度分析管线。

核心指标:
  - monthly_mean_embedding (384 维) — 月度语义中心
  - covariance_trace — 叙事色散度 (共识统一 vs 四分五裂)
  - distance_skewness — 距离偏度 (捕捉 "微观闪洪" 的长尾)
  - pos_entropy — 词性分布熵 (叙事化程度 proxy)
  - attention_concentration — 注意力集中度
  - novelty — 语义新颖度
  - platform_jsd — 跨平台 Jensen-Shannon 散度 (H1 vs H2 检验)

理论依据 (v4.2):
  - 不能只存均值: 月度 mean collapse 会丢失分布结构
  - 协方差迹 = 共识统一度: 高 = 表达分散, 低 = 表达统一
  - 距离偏度 = 闪洪检测: 90 分位 / 中位距离 > 2 → 存在长尾爆发
  - POS 熵 = 叙事化程度 proxy: 名词/因果连词多 → 结构分析; 形容词/语气词多 → 叙事消费
  - 跨平台 JSD: H1(结构崩塌) → 平台趋同; H2(共识收敛) → 平台各说各的但底层一致

用法:
  python src/data/monthly_aggregator.py              # 聚合所有可用月份
  python src/data/monthly_aggregator.py --month 2026-07  # 指定月份
  python src/data/monthly_aggregator.py --json       # JSON 输出
"""

import json, sys, os, argparse, re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).parent.parent.parent
DAILY_DIR = ROOT / "data" / "scraped" / "daily"
OUTPUT_PATH = ROOT / "data" / "processed" / "monthly_semantic_state.json"

# ── 中文词性到粗粒度类别的映射 ──
POS_COARSE = {
    "n": "nominal",     # 名词类 — 指称事物
    "v": "verbal",      # 动词类 — 描述行动
    "a": "adjectival",  # 形容词类 — 评价/情绪
    "d": "adverbial",   # 副词类
    "c": "connective",  # 连词类 — 因果/转折 (结构分析标志)
    "p": "pronominal",  # 代词类
    "u": "particle",    # 助词 — 语气/情态 (叙事消费标志)
    "m": "numeral",     # 数词
    "x": "other",       # 其他
}


def _pos_coarse(tag: str) -> str:
    """将 jieba POS tag 映射到粗粒度类别."""
    if tag.startswith("n"):
        return "nominal"
    if tag.startswith("v"):
        return "verbal"
    if tag.startswith("a"):
        return "adjectival"
    if tag.startswith("d"):
        return "adverbial"
    if tag in ("c", "p", "uj", "ul", "uv", "ud", "ug", "uz"):
        return "connective" if tag == "c" else "particle"
    if tag.startswith("m"):
        return "numeral"
    return "other"


def compute_pos_distribution(headlines: list[str]) -> dict:
    """计算一组 headline 的粗粒度词性分布和熵.

    Returns:
        {"distribution": {cat: proportion, ...},
         "entropy": float,
         "nominal_ratio": float,
         "connective_ratio": float,
         "adjectival_particle_ratio": float}
    """
    try:
        import jieba.posseg as pseg
    except ImportError:
        return {"error": "jieba not available"}

    cat_counts = Counter()
    total = 0

    for text in headlines:
        words = pseg.cut(text)
        for w, tag in words:
            cat = _pos_coarse(tag)
            cat_counts[cat] += 1
            total += 1

    if total == 0:
        return {"entropy": 0.0, "nominal_ratio": 0.0,
                "connective_ratio": 0.0, "adjectival_particle_ratio": 0.0}

    dist = {cat: count / total for cat, count in cat_counts.items()}
    entropy = -sum(p * np.log(p) for p in dist.values() if p > 0)

    return {
        "distribution": dist,
        "entropy": round(float(entropy), 4),
        "nominal_ratio": round(dist.get("nominal", 0), 4),
        "connective_ratio": round(dist.get("connective", 0), 4),
        "adjectival_particle_ratio": round(
            dist.get("adjectival", 0) + dist.get("particle", 0), 4),
    }


# ═══════════════════════════════════════════════
# 月度聚合
# ═══════════════════════════════════════════════

def load_daily_files(month_str: str) -> list[dict]:
    """加载指定月份的所有日级文件."""
    files = sorted(DAILY_DIR.glob(f"{month_str}-*.json"))
    daily_data = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                daily_data.append(json.load(f))
        except Exception:
            pass
    return daily_data


def get_available_months() -> list[str]:
    """获取所有有日级/hourly 数据的月份.

    优先从 daily 目录扫描, 如果为空则从 hourly 目录扫描.
    """
    months = set()

    # 从 daily 目录
    for fp in DAILY_DIR.glob("*.json"):
        stem = fp.stem
        if len(stem) >= 7:
            months.add(stem[:7])

    # 从 hourly 目录 (daily 为空时的 fallback)
    hourly_dir = ROOT / "data" / "scraped" / "hourly"
    if hourly_dir.exists():
        for fp in hourly_dir.glob("*.json"):
            stem = fp.stem  # YYYY-MM-DD_HHMMSS
            if len(stem) >= 7:
                months.add(stem[:7])

    return sorted(months)


def aggregate_month(month_str: str) -> dict | None:
    """聚合单个自然月的日级语义状态.

    优先从 daily 文件获取日期列表, 如果没有 daily 文件则直接从 hourly 扫描.

    Returns None 如果该月无数据.
    """
    hourly_dir = ROOT / "data" / "scraped" / "hourly"

    # ── 确定该月有哪些天 ──
    daily_data = load_daily_files(month_str)

    if daily_data:
        dates_in_month = sorted(set(d["date"] for d in daily_data))
    else:
        # 从 hourly 文件直接扫描
        dates_in_month = set()
        for fp in hourly_dir.glob(f"{month_str}-*.json"):
            dates_in_month.add(fp.stem[:10])
        dates_in_month = sorted(dates_in_month)

    if not dates_in_month:
        print(f"[聚合] {month_str}: 无数据, 跳过")
        return None

    print(f"[聚合] {month_str}: {len(dates_in_month)} 天数据")

    # ── 从 hourly 文件读全量 embedding ──
    all_headlines = []
    all_embeddings_list = []
    platform_embeddings = defaultdict(list)

    for date_str in dates_in_month:
        hourly_files = sorted(hourly_dir.glob(f"{date_str}_*.json"))
        for hf in hourly_files:
            try:
                with open(hf, "r", encoding="utf-8") as f:
                    hdata = json.load(f)
            except Exception:
                continue

            emb_items = hdata.get("headlines_embedded")
            if not emb_items:
                continue

            for item in emb_items:
                emb = np.array(item["embedding"])
                all_embeddings_list.append(emb)
                all_headlines.append(item["title"])
                platform = item.get("platform", "unknown")
                platform_embeddings[platform].append(emb)

    if not all_embeddings_list:
        print(f"[聚合] {month_str}: hourly 文件中无 embedding, 跳过")
        return None

    embeddings = np.array(all_embeddings_list)  # n × 384
    n = len(embeddings)
    print(f"  全量: {n} 条 headline embeddings")

    # ── 1. 月度均值 ──
    monthly_mean = embeddings.mean(axis=0)
    monthly_mean = monthly_mean / (np.linalg.norm(monthly_mean) + 1e-10)

    # ── 2. 协方差迹 (色散度) ──
    centered = embeddings - monthly_mean
    cov = centered.T @ centered / (n - 1) if n > 1 else np.zeros((384, 384))
    cov_trace = float(np.trace(cov))

    # ── 3. 距离偏度 (闪洪 proxy) ──
    distances = np.linalg.norm(centered, axis=1)
    median_dist = float(np.median(distances))
    p90_dist = float(np.percentile(distances, 90))
    p99_dist = float(np.percentile(distances, 99))
    skewness_ratio = p90_dist / median_dist if median_dist > 0 else 1.0
    flash_flood_alert = skewness_ratio > 2.0  # 90 分位超过中位 2×

    # ── 4. POS 词性分布 (叙事化程度 proxy) ──
    pos_info = compute_pos_distribution(all_headlines)

    # ── 5. 注意力集中度 (headline 间 pairwise similarity) ──
    # 采样计算 (n 可能很大)
    sample_size = min(n, 2000)
    if n > sample_size:
        idx = np.random.default_rng(42).choice(n, sample_size, replace=False)
        sample_emb = embeddings[idx]
    else:
        sample_emb = embeddings

    sim_matrix = sample_emb @ sample_emb.T
    # 排除对角线
    mask = ~np.eye(len(sample_emb), dtype=bool)
    attention_conc = float(sim_matrix[mask].mean()) if len(sample_emb) > 1 else 1.0

    # ── 6. 基于日级文件的聚合指标 ──
    daily_conc_vals = [d.get("attention_concentration", 0) for d in daily_data]
    daily_novelty_vals = [d.get("novelty_vs_previous_day", 0) for d in daily_data]
    daily_headline_counts = [d.get("n_headlines", 0) for d in daily_data]

    # ── 7. 跨平台 JSD (H1 vs H2 检验) ──
    platform_jsd = _compute_platform_jsd(platform_embeddings)

    # ── 8. Meme similarity 月度聚合 ──
    meme_monthly = _aggregate_meme_similarities(daily_data)

    # ── 组装 ──
    result = {
        "version": "1.0",
        "month": month_str,
        "aggregated_at": datetime.now().isoformat(),
        "n_headlines": n,
        "n_days_with_data": len(daily_data),
        # 核心指标
        "monthly_mean_embedding": [round(float(v), 6) for v in monthly_mean],
        "covariance_trace": round(cov_trace, 6),
        "distance_distribution": {
            "median": round(median_dist, 6),
            "p90": round(p90_dist, 6),
            "p99": round(p99_dist, 6),
            "skewness_ratio": round(skewness_ratio, 4),
            "flash_flood_alert": flash_flood_alert,
        },
        "pos_entropy": pos_info,
        "attention_concentration": round(attention_conc, 4),
        "daily_aggregates": {
            "concentration_mean": round(float(np.mean(daily_conc_vals)), 4) if daily_conc_vals else None,
            "concentration_std": round(float(np.std(daily_conc_vals)), 4) if daily_conc_vals else None,
            "novelty_mean": round(float(np.mean(daily_novelty_vals)), 4) if daily_novelty_vals else None,
            "novelty_std": round(float(np.std(daily_novelty_vals)), 4) if daily_novelty_vals else None,
            "headlines_per_day_mean": round(float(np.mean(daily_headline_counts)), 1) if daily_headline_counts else 0,
        },
        "platform_jsd": platform_jsd,
        "meme_similarities_monthly": meme_monthly,
    }

    return result


def _compute_platform_jsd(platform_embeddings: dict) -> dict:
    """计算跨平台 Jensen-Shannon 散度.

    H1 (结构崩塌): 平台轨迹趋同 → JSD 低
    H2 (共识收敛): 平台各说各的但底层一致 → JSD 高

    Returns:
        {platform_pair: jsd_value, "mean_jsd": float,
         "interpretation": "H1-leaning" | "H2-leaning"}
    """
    platforms = list(platform_embeddings.keys())
    if len(platforms) < 2:
        return {"mean_jsd": None, "interpretation": "insufficient platforms"}

    # 对每个平台: 计算该平台所有 embedding 的平均值作为 "平台姿态向量"
    platform_means = {}
    for plat, embs in platform_embeddings.items():
        if len(embs) < 10:
            continue
        mean_vec = np.mean(embs, axis=0)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-10)
        platform_means[plat] = mean_vec

    if len(platform_means) < 2:
        return {"mean_jsd": None, "interpretation": "insufficient platforms"}

    # Pairwise JSD
    # JSD 需要概率分布 → 用 softmax + temperature 将 embedding 转为伪概率
    # 或者直接用 cosine distance 作为 proxy
    # 这里用 cosine distance: 不同平台间的平均 cosine distance
    plat_names = sorted(platform_means.keys())
    distances = []
    pairs = {}
    for i, p1 in enumerate(plat_names):
        for p2 in plat_names[i + 1:]:
            d = 1.0 - float(np.dot(platform_means[p1], platform_means[p2]))
            pairs[f"{p1}_vs_{p2}"] = round(d, 4)
            distances.append(d)

    mean_dist = float(np.mean(distances)) if distances else 0.0

    # 解释: cosine distance > 0.3 → 平台姿态显著不同 → H2-leaning
    interpretation = (
        "H2-leaning: platforms use different expressions for same underlying consensus"
        if mean_dist > 0.3 else
        "H1-leaning: platforms show converging narrative posture"
        if mean_dist < 0.15 else
        "ambiguous: moderate platform divergence"
    )

    return {
        "pairs": pairs,
        "mean_cosine_distance": round(mean_dist, 4),
        "interpretation": interpretation,
    }


def _aggregate_meme_similarities(daily_data: list[dict]) -> dict:
    """聚合月内各天的 meme similarity 为月度统计."""
    # 收集所有 daily 文件的 meme_similarities
    meme_daily_values = defaultdict(list)  # meme_name → [max_sim values]

    for day in daily_data:
        meme_sims = day.get("meme_similarities", {})
        for meme_name, stats in meme_sims.items():
            if isinstance(stats, dict):
                meme_daily_values[meme_name].append(stats.get("max_sim", 0))

    result = {}
    for meme_name, sims in meme_daily_values.items():
        if sims:
            result[meme_name] = {
                "mean_max_sim": round(float(np.mean(sims)), 4),
                "std_max_sim": round(float(np.std(sims)), 4),
                "peak_max_sim": round(float(np.max(sims)), 4),
                "n_days_detected": len(sims),
            }

    return result


# ═══════════════════════════════════════════════
# 全量聚合 + 对齐
# ═══════════════════════════════════════════════

def build_full_monthly_sequence() -> dict:
    """构建完整的月度语义状态序列, 对齐现有 127 月时间线.

    2015-01 → 2025-12: 标记为 scraper 不可用 (pre-scraper)
    2026-01 → 2026-05: 标记为 scraper 数据缺失
    2026-06 → 今:      聚合

    Returns:
        {"months": [...], "monthly_states": [...], "meta": {...}}
    """
    available = get_available_months()
    if not available:
        print("无可用日级数据")
        return {"months": [], "monthly_states": [], "meta": {"error": "no data"}}

    first_scraper_month = available[0]
    last_available = available[-1]

    # 构建月份序列
    months = []
    # Pre-scraper: 2015-01 → 2025-12
    for y in range(2015, 2026):
        for m in range(1, 13):
            month_str = f"{y}-{m:02d}"
            months.append(month_str)

    # Gap: 2026-01 → first_scraper_month - 1
    # 如果 first 是 2026-06, gap 是 01-05
    # 如果 first 是 2026-07, gap 是 01-06

    # Scraper months
    for month_str in available:
        if month_str not in months:
            months.append(month_str)

    # Sort
    months = sorted(set(months))

    # 聚合
    monthly_states = []
    for month_str in months:
        if month_str in available:
            state = aggregate_month(month_str)
        else:
            state = None  # 无 scraper 数据

        monthly_states.append({
            "month": month_str,
            "has_scraper_data": state is not None,
            "state": state,
        })

        if state is None:
            if month_str < first_scraper_month:
                pass  # pre-scraper, silent
            else:
                print(f"[聚合] {month_str}: 无 scraper 数据 (缺失)")

    return {
        "source": "monthly_aggregator.py — scraper v2.0",
        "version": "1.0",
        "built_at": datetime.now().isoformat(),
        "months": months,
        "monthly_states": monthly_states,
        "meta": {
            "total_months": len(months),
            "months_with_scraper": len(available),
            "first_scraper_month": first_scraper_month,
            "last_scraper_month": last_available,
            "embedding_dim": 384,
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
            "covariance_trace_note": (
                "协方差迹 = 叙事色散度. "
                "高值 → 表达分散, 共识内部多样化. "
                "低值 → 表达统一, 叙事高度集中."
            ),
            "distance_skewness_note": (
                "距离偏度 = p90 / median. "
                "ratio > 2.0 → 存在长尾 '微观闪洪', flash_flood_alert=True."
            ),
            "pos_entropy_note": (
                "POS 熵 = 词性分布的信息熵. "
                "高 nominal/connective 比例 → 倾向结构分析. "
                "高 adjectival/particle 比例 → 倾向叙事消费."
            ),
            "platform_jsd_note": (
                "跨平台配对 cosine distance. "
                "H1(结构崩塌) → 平台趋同, distance 低. "
                "H2(共识收敛) → 平台各说各的, distance 高."
            ),
        },
    }


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="月度语义状态聚合器 — scraper v2.0 → 分析层")
    parser.add_argument("--month", type=str, default=None,
                        help="指定月份 (YYYY-MM), 默认全量")
    parser.add_argument("--json", action="store_true",
                        help="JSON 输出")
    parser.add_argument("--save", action="store_true", default=True,
                        help="保存到 data/processed/monthly_semantic_state.json (默认)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    if args.month:
        result = aggregate_month(args.month)
        if args.json and result:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif result:
            _print_monthly(result)
        return

    # 全量
    full = build_full_monthly_sequence()

    if args.save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(full, f, ensure_ascii=False, indent=2)
        print(f"\n已保存: {OUTPUT_PATH}")

    # 摘要
    states = full["monthly_states"]
    scraper_months = [s for s in states if s["has_scraper_data"]]
    print(f"\n{'='*60}")
    print(f"  月度语义状态序列")
    print(f"{'='*60}")
    print(f"  总月份: {len(states)}")
    print(f"  有 scraper 数据: {len(scraper_months)}")
    if scraper_months:
        latest = scraper_months[-1]["state"]
        print(f"  最新月: {latest['month']} ({latest['n_headlines']} headlines)")
        print(f"  协方差迹: {latest['covariance_trace']:.4f}")
        print(f"  距离偏度: {latest['distance_distribution']['skewness_ratio']:.2f}"
              + (" ⚡ FLASH FLOOD" if latest['distance_distribution']['flash_flood_alert'] else ""))
        print(f"  POS 熵: {latest['pos_entropy'].get('entropy', 'N/A')}")
        if latest['platform_jsd']['mean_cosine_distance'] is not None:
            print(f"  跨平台 cosine dist: {latest['platform_jsd']['mean_cosine_distance']:.4f}"
                  f" → {latest['platform_jsd']['interpretation']}")


def _print_monthly(state: dict):
    """单月人类可读输出."""
    print(f"\n{'='*60}")
    print(f"  月度语义状态 — {state['month']}")
    print(f"{'='*60}")
    print(f"  Headlines: {state['n_headlines']}")
    print(f"  协方差迹: {state['covariance_trace']:.4f}")
    dd = state['distance_distribution']
    print(f"  距离分布: median={dd['median']:.4f} p90={dd['p90']:.4f} "
          f"skew={dd['skewness_ratio']:.2f}"
          + (" ⚡" if dd['flash_flood_alert'] else ""))
    pos = state['pos_entropy']
    if 'entropy' in pos:
        print(f"  POS 熵: {pos['entropy']:.3f} "
              f"(nom={pos['nominal_ratio']:.2f} "
              f"conn={pos['connective_ratio']:.2f} "
              f"adj+part={pos['adjectival_particle_ratio']:.2f})")
    print(f"  注意力集中度: {state['attention_concentration']:.3f}")
    pjsd = state['platform_jsd']
    if pjsd['mean_cosine_distance'] is not None:
        print(f"  跨平台 cosine dist: {pjsd['mean_cosine_distance']:.4f}")
        print(f"    → {pjsd['interpretation']}")


if __name__ == "__main__":
    main()

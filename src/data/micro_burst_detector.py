"""
Micro Burst Detector — FR19 v4.1

将小时级 scraper 数据聚合为日度微观指标,
作为 Sensitivity 中 Critical Slowing 的动态修正项.

物理逻辑 (来自外部 AI 的多尺度解耦建议):
- 宏观慢变量 (月度 Level 1): 决定系统在哪个相区
- 微观快变量 (小时级 scraper): 检测"闪洪前兆"
- 当宏观处于亚稳态 (高 Sensitivity) 且微观连续出现高波动 → 拉响警报

指标:
- topic_turnover: 相邻两次采集的话题更替率 (高=信息流加速)
- attention_entropy: Top-10 排名分布的均匀度 (低=注意力高度集中)
- novelty_burst: 24h 窗口内全新话题占比 (高=有突发事件)
- signal_density: 每天 meme 信号命中次数

输出: 日度 micro_burst_score [0,1], 作为 Sensitivity 的快变量修正因子.

用法:
    python src/data/micro_burst_detector.py
    python src/data/micro_burst_detector.py --days 7
"""

import json, sys, os, argparse, glob
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).parent.parent.parent
SCRAPED_DIR = ROOT / "data/scraped"
OUTPUT_PATH = ROOT / "data/processed/micro_burst.json"


def load_scrapes(days: int = 14) -> list[dict]:
    """加载最近 N 天的 scraper 文件."""
    cutoff = datetime.now() - timedelta(days=days)

    # Try local first, then server paths
    search_dirs = [
        SCRAPED_DIR,
        Path("/home/mote/MemeticChaos/data/scraped"),
    ]

    files = []
    for d in search_dirs:
        if d.exists():
            files = sorted(glob.glob(str(d / "scrape_*.json")))
            if files:
                break

    if not files:
        print("  ⚠ 未找到 scraper 文件 (本地和服务器路径均无)")
        return []

    scrapes = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts_str = data.get("timestamp", "")
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            if ts >= cutoff:
                data["_parsed_ts"] = ts
                scrapes.append(data)
        except Exception:
            continue

    scrapes.sort(key=lambda s: s["_parsed_ts"])
    return scrapes


def load_signals(days: int = 14) -> list[dict]:
    """加载 signal_history."""
    search_paths = [
        SCRAPED_DIR / "signal_history.jsonl",
        Path("/home/mote/MemeticChaos/data/scraped/signal_history.jsonl"),
    ]

    for sp in search_paths:
        if sp.exists():
            break
    else:
        return []

    cutoff = datetime.now() - timedelta(days=days)
    signals = []
    with open(sp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sig = json.loads(line)
                ts = datetime.fromisoformat(sig.get("timestamp", "2000-01-01T00:00:00"))
                if ts >= cutoff:
                    sig["_parsed_ts"] = ts
                    signals.append(sig)
            except Exception:
                continue
    return signals


def extract_topics(scrape: dict) -> set[str]:
    """从一次采集提取所有话题标题."""
    topics = set()
    for platform in ["weibo", "baidu", "zhihu"]:
        key = f"{platform}_top10"
        for item in scrape.get(key, []):
            title = item.get("title", "").strip()
            if title:
                topics.add(title)
    return topics


def compute_daily_metrics(scrapes: list[dict], signals: list[dict]) -> list[dict]:
    """计算日度微观指标."""
    if not scrapes:
        return []

    # Group by day
    by_day = defaultdict(list)
    for s in scrapes:
        day = s["_parsed_ts"].strftime("%Y-%m-%d")
        by_day[day].append(s)

    # Count signals per day
    signals_per_day = Counter()
    for sig in signals:
        day = sig["_parsed_ts"].strftime("%Y-%m-%d")
        signals_per_day[day] += 1

    days = sorted(by_day.keys())
    results = []

    # Build rolling 24h topic memory
    topic_history = []  # list of (timestamp, topic_set)

    for day in days:
        day_scrapes = by_day[day]
        n_scrapes = len(day_scrapes)

        # ── Topic turnover within day ──
        turnovers = []
        entropies = []
        all_day_topics = set()

        for i in range(len(day_scrapes)):
            topics = extract_topics(day_scrapes[i])
            all_day_topics.update(topics)
            if i > 0:
                prev = extract_topics(day_scrapes[i - 1])
                intersection = len(topics & prev)
                union = len(topics | prev)
                turnover = 1.0 - (intersection / max(union, 1))
                turnovers.append(turnover)

            # Attention entropy (HHI of rankings within top-10)
            for platform in ["weibo", "baidu", "zhihu"]:
                key = f"{platform}_top10"
                items = day_scrapes[i].get(key, [])
                scores = [it.get("hot_score", 0) for it in items]
                total = sum(scores)
                if total > 0 and len(scores) > 1:
                    shares = [s / total for s in scores if s > 0]
                    if shares:
                        ent = -sum(p * np.log(p) for p in shares)
                        entropies.append(ent / np.log(len(shares) + 1))

        mean_turnover = float(np.mean(turnovers)) if turnovers else 0.0
        mean_entropy = float(np.mean(entropies)) if entropies else 0.5

        # ── Novelty burst ──
        # Topics that haven't appeared in the last 24h
        novelty_count = 0
        for topic in all_day_topics:
            seen_before = False
            for _, hist_topics in topic_history[-24:]:  # last 24 scrapes ≈ 24h
                if topic in hist_topics:
                    seen_before = True
                    break
            if not seen_before:
                novelty_count += 1

        novelty_rate = novelty_count / max(len(all_day_topics), 1)
        topic_history.append((day, all_day_topics))
        if len(topic_history) > 72:  # keep 3 days max
            topic_history = topic_history[-72:]

        # ── Signal density ──
        n_signals = signals_per_day.get(day, 0)

        # ── Micro burst score ──
        # High turnover + low entropy + high novelty = flash flood brewing
        micro_score = float(np.clip(
            0.30 * mean_turnover +
            0.25 * (1.0 - mean_entropy) +  # attention concentrating
            0.25 * novelty_rate +
            0.20 * min(n_signals / 10.0, 1.0),  # signal density
            0, 1))

        results.append({
            "date": day,
            "n_scrapes": n_scrapes,
            "mean_turnover": round(mean_turnover, 4),
            "mean_attention_entropy": round(mean_entropy, 4),
            "novelty_rate": round(novelty_rate, 4),
            "signal_count": n_signals,
            "unique_topics": len(all_day_topics),
            "micro_burst_score": round(micro_score, 4),
        })

    return results


def compute_micro_correction(daily_metrics: list[dict],
                              macro_sensitivity: float = 0.56) -> dict:
    """将日度微观指标转化为 Sensitivity 的修正因子.

    如果微观闪洪信号连续出现 → Sensitivity 上调.
    如果微观平静 → Sensitivity 按宏观值.
    """
    if not daily_metrics:
        return {"correction_factor": 1.0, "note": "无微观数据, 使用宏观 Sensitivity"}

    # Recent 3-day trend
    recent = daily_metrics[-3:]
    scores = [d["micro_burst_score"] for d in recent]
    mean_score = np.mean(scores)
    trend = scores[-1] - scores[0] if len(scores) >= 2 else 0.0

    # Correction: if consistently high (>0.5) and rising, amplify
    if mean_score > 0.5 and trend > 0:
        factor = 1.0 + 0.3 * mean_score  # up to 1.3x
        level = "ELEVATED"
        note = "微观闪洪信号持续上升, Sensitivity 上调"
    elif mean_score > 0.4:
        factor = 1.0 + 0.15 * mean_score  # up to 1.15x
        level = "WATCH"
        note = "微观波动偏高, 持续监测"
    elif mean_score < 0.2:
        factor = 0.85  # below baseline
        level = "CALM"
        note = "微观层面平静, Sensitivity 按宏观基线"
    else:
        factor = 1.0
        level = "NORMAL"
        note = "微观层面正常"

    corrected_sensitivity = round(macro_sensitivity * factor, 4)

    return {
        "recent_3day_mean_score": round(mean_score, 4),
        "trend": "rising" if trend > 0.03 else "falling" if trend < -0.03 else "flat",
        "level": level,
        "correction_factor": round(factor, 4),
        "macro_sensitivity": macro_sensitivity,
        "micro_corrected_sensitivity": corrected_sensitivity,
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser(description="Micro Burst Detector")
    parser.add_argument("--days", type=int, default=14, help="回看天数")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 56)
    print("Micro Burst Detector — v4.1")
    print("=" * 56)

    # Load
    print(f"\n[1/3] 加载最近 {args.days} 天 scraper 数据...")
    scrapes = load_scrapes(args.days)
    signals = load_signals(args.days)
    print(f"  Scrapes: {len(scrapes)}   Signals: {len(signals)}")

    if not scrapes:
        print("  无数据. 退出.")
        return

    # Compute
    print(f"\n[2/3] 计算日度微观指标...")
    daily = compute_daily_metrics(scrapes, signals)

    if not daily:
        print("  无日度数据.")
        return

    print(f"  Days: {len(daily)} ({daily[0]['date']} → {daily[-1]['date']})")
    print(f"\n  {'Date':<12s} {'Turnover':>8s} {'Entropy':>8s} {'Novelty':>8s} "
          f"{'Signals':>7s} {'Burst':>6s}")
    print(f"  {'─'*50}")
    for d in daily[-7:]:
        print(f"  {d['date']:<12s} {d['mean_turnover']:>8.3f} {d['mean_attention_entropy']:>8.3f} "
              f"{d['novelty_rate']:>8.3f} {d['signal_count']:>7d} {d['micro_burst_score']:>6.3f}")

    # Correction
    print(f"\n[3/3] Sensitivity 微观修正...")
    correction = compute_micro_correction(daily)
    print(f"  微观水平: {correction['level']}")
    print(f"  修正因子: {correction['correction_factor']:.3f}")
    print(f"  Sensitivity: {correction['macro_sensitivity']:.3f} → "
          f"{correction['micro_corrected_sensitivity']:.3f}")
    print(f"  {correction['note']}")

    # Save
    output = {
        "source": "micro_burst_detector.py",
        "daily_metrics": daily,
        "micro_correction": correction,
    }
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  已保存 → {output_path}")

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

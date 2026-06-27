"""
自动数据采集器 — 持续补充新的模因数据

三个并行来源：
1. 百度搜索量追踪 — 每日查询 29 梗关键词，记录结果计数变化
2. 微博关键词搜索 — 搜索指定关键词，统计提及量时间序列
3. 新梗发现 — 从微博热搜中检测候选新梗（高频+反常组合+梗模式匹配）

用法:
    python src/data/auto_collector.py            # 单次采集
    python src/data/auto_collector.py --watch 24  # 每 24 小时自动采集
"""

import json
import os
import sys
import re
import time
import hashlib
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════

DATA_DIR = Path("data/collector")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 29 个梗的搜索关键词 (name → list of Baidu search queries)
MEME_QUERIES = {
    "打工人": ["打工人 梗", "打工人 网络用语"],
    "内卷": ["内卷 梗", "内卷 网络热词"],
    "躺平": ["躺平 梗", "躺平 网络用语"],
    "普信男": ["普信男 梗", "普信男 网络用语", "普通却自信"],
    "小镇做题家": ["小镇做题家 梗", "小镇做题家"],
    "摆烂": ["摆烂 梗", "摆烂 网络用语"],
    "润": ["润 梗 网络用语", "润学"],
    "吗喽": ["吗喽 梗", "吗喽 猴子 表情包"],
    "孔乙己的长衫": ["孔乙己的长衫 梗", "孔乙己 长衫"],
    "精神状态良好": ["精神状态良好 梗", "发疯文学"],
    "科目三": ["科目三 舞蹈 梗", "广西科目三"],
    "后浪": ["后浪 梗 B站", "后浪 宣传片"],
    "鸡你太美": ["鸡你太美 梗", "蔡徐坤 只因你太美"],
    "i人/e人": ["i人 e人 梗", "MBTI 梗"],
    "遥遥领先": ["遥遥领先 梗", "华为遥遥领先"],
    "XX刺客": ["雪糕刺客 梗", "刺客 网络用语"],
    "尊嘟假嘟": ["尊嘟假嘟 梗", "真的假的 谐音"],
    "情绪价值": ["情绪价值 梗", "情绪价值 网络用语"],
    "四不": ["不婚不育 梗", "四不青年"],
    "显眼包": ["显眼包 梗", "显眼包 网络用语"],
    "泼天富贵": ["泼天富贵 梗", "泼天流量"],
    "发疯文学": ["发疯文学 梗", "发疯文学 网络用语"],
    "服美役": ["服美役 梗", "不服美役"],
    "建议专家不要建议": ["建议专家不要建议 梗", "专家建议"],
    "命运的齿轮开始转动": ["命运的齿轮开始转动 梗"],
    "原生家庭": ["原生家庭 梗"],
    "谢帝我要迪士尼": ["谢帝 迪士尼 梗", "我要迪士尼"],
    "鼠鼠": ["鼠鼠 梗", "老鼠人"],
    "牛马": ["牛马 梗", "当牛做马"],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
}


# ═══════════════════════════════════════════════
# 1. 百度搜索量追踪
# ═══════════════════════════════════════════════

def track_google_trends() -> dict:
    """使用 Google Trends 追踪所有梗的搜索兴趣（可靠，无需验证码）。"""
    from pytrends.request import TrendReq
    print(f"[Google Trends] 采集 29 个梗的搜索兴趣...")
    pytrends = TrendReq(hl='zh-CN', tz=360)
    results = {}
    timestamp = datetime.now().isoformat()

    # Batch queries in groups of 5 (Google Trends limit)
    names = list(MEME_QUERIES.keys())
    for i in range(0, len(names), 5):
        batch = names[i:i+5]
        try:
            pytrends.build_payload(batch, cat=0, timeframe='today 3-m', geo='CN')
            df = pytrends.interest_over_time()
            if not df.empty:
                df = df.drop(columns=['isPartial'], errors='ignore')
                for name in batch:
                    if name in df.columns:
                        avg_val = float(df[name].mean())
                        peak_val = int(df[name].max())
                        if avg_val > 0 or peak_val > 0:
                            results[name] = {
                                "timestamp": timestamp,
                                "avg_interest": avg_val,
                                "peak_interest": peak_val,
                            }
                            print(f"  [{i//5+1}/{(len(names)+4)//5}] {name:<12s}: avg={avg_val:.1f}  peak={peak_val}")
            time.sleep(2)  # rate limit
        except Exception as e:
            print(f"  Batch {i//5+1} error: {e}")
            time.sleep(5)

    # Save
    date_str = datetime.now().strftime("%Y%m%d")
    snapshot_path = DATA_DIR / f"google_trends_{date_str}.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "results": results}, f, ensure_ascii=False, indent=2)

    ts_path = DATA_DIR / "attention_timeseries.jsonl"
    with open(ts_path, "a", encoding="utf-8") as f:
        for name, data in results.items():
            f.write(json.dumps({"name": name, "source": "google_trends", **data}, ensure_ascii=False) + "\n")

    return results


# ═══════════════════════════════════════════════
# 2. 微博关键词搜索
# ═══════════════════════════════════════════════

def fetch_weibo_keyword_count(keyword: str) -> int:
    """搜索微博关键词，返回提及量近似值。"""
    url = f"https://s.weibo.com/weibo?q={urllib.request.quote(keyword)}&typeall=1"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        # Extract total count from Weibo search result page
        match = re.search(r'找到约 ([\d,]+) 条结果', html)
        if match:
            return int(match.group(1).replace(",", ""))
        match = re.search(r'找到约 ([\d]+) 条', html)
        if match:
            return int(match.group(1))
        # Fallback: count card items
        cards = len(re.findall(r'class="card-wrap"', html))
        return cards
    except Exception as e:
        return -1


def track_weibo_mentions() -> dict:
    """对所有已知梗查询微博提及量。"""
    print(f"[微博] 搜索 {len(MEME_QUERIES)} 个梗的提及量...")
    results = {}
    timestamp = datetime.now().isoformat()

    for name in list(MEME_QUERIES.keys())[:10]:  # 限速，每次最多 10 个
        count = fetch_weibo_keyword_count(name)
        if count >= 0:
            results[name] = {"timestamp": timestamp, "mention_count": count}
            print(f"  {name:<12s}: {count:,}")
        time.sleep(1.0)

    date_str = datetime.now().strftime("%Y%m%d")
    snapshot_path = DATA_DIR / f"weibo_{date_str}.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "results": results}, f, ensure_ascii=False, indent=2)

    ts_path = DATA_DIR / "weibo_timeseries.jsonl"
    with open(ts_path, "a", encoding="utf-8") as f:
        for name, data in results.items():
            f.write(json.dumps({"name": name, **data}, ensure_ascii=False) + "\n")

    return results


# ═══════════════════════════════════════════════
# 3. 新梗发现
# ═══════════════════════════════════════════════

def discover_new_memes(weibo_hot_items: list[dict]) -> list[dict]:
    """从微博热搜中发现候选新梗。

    启发式规则：
    - 标题长度 2-8 字
    - 包含"梗"、"是什么"、"什么意思"等模式 → 可能是梗解释内容
    - 不含纯数字、纯英文、明显新闻标题
    - 不在已知 29 梗中
    """
    known_names = set(m["name"] for m in __import__("json").load(
        open("data/curated/memes_2020_2025.json", "r", encoding="utf-8")
    )["memes"])
    known_names.update(MEME_QUERIES.keys())

    candidates = []
    for item in weibo_hot_items:
        title = item.get("title", "").strip()
        # Heuristic filters
        if len(title) < 2 or len(title) > 12:
            continue
        if any(c.isdigit() for c in title):
            continue
        if re.match(r'^[a-zA-Z\s]+$', title):
            continue
        if title in known_names:
            continue  # already tracked

        # Meme-like patterns
        meme_signals = 0
        if any(w in title for w in ["梗", "什么梗", "是什么意思", "热词", "网络用语"]):
            meme_signals += 1
        if re.search(r'[一-鿿]{2,4}$', title):  # ends with 2-4 Chinese chars
            meme_signals += 1

        if meme_signals >= 1:
            candidates.append({
                "title": title,
                "rank": item.get("rank", 0),
                "hot_score": item.get("hot_score", 0),
                "platform": item.get("platform", "weibo"),
                "detected_at": datetime.now().isoformat(),
            })

    if candidates:
        # Save to discovery log
        disc_path = DATA_DIR / "candidate_memes.jsonl"
        with open(disc_path, "a", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return candidates


# ═══════════════════════════════════════════════
# 4. 统一采集 + 时间序列管理
# ═══════════════════════════════════════════════

def collect_all():
    """执行一次完整采集循环。"""
    print(f"\n{'='*60}")
    print(f"MemeticChaos — 自动数据采集 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # 1. Google Trends 搜索兴趣
    results = track_google_trends()

    # 2. 新梗发现
    from src.data.scraper import fetch_weibo_hot_search
    hot_items = fetch_weibo_hot_search()
    candidates = discover_new_memes(hot_items)
    if candidates:
        print(f"\n[发现] {len(candidates)} 个候选新梗:")
        for c in candidates[:5]:
            print(f"  {c['title']} (排名 #{c['rank']})")

    # 3. 时间序列历史
    history = load_timeseries()
    dates = sorted(history.keys())
    print(f"\n[历史] {len(dates)} 天数据")
    if dates:
        print(f"  范围: {dates[0]} → {dates[-1]}")

    return results


def load_timeseries() -> dict:
    """加载注意力时间序列历史。"""
    ts_path = DATA_DIR / "attention_timeseries.jsonl"
    if not ts_path.exists():
        return {}
    history = {}
    with open(ts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    name = item.get("name", "")
                    ts = item.get("timestamp", "")[:10]
                    if ts not in history:
                        history[ts] = {}
                    history[ts][name] = item
                except json.JSONDecodeError:
                    pass
    return history


def get_volume_trend(name: str) -> list[tuple]:
    """获取指定梗的注意力趋势。"""
    history = load_timeseries()
    trend = []
    for date in sorted(history.keys()):
        if name in history[date]:
            item = history[date][name]
            trend.append((date, item.get("mention_count", 0)))
    return trend


# ═══════════════════════════════════════════════
# 5. 运行入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys, argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="自动数据采集")
    parser.add_argument("--watch", type=int, default=0, help="每 N 小时自动采集")
    parser.add_argument("--trend", type=str, default="", help="查看指定梗的趋势")
    args = parser.parse_args()

    if args.trend:
        trend = get_volume_trend(args.trend)
        print(f"=== {args.trend} Google Trends ===")
        for date, count in trend:
            bar = "#" * max(1, int(count))
            print(f"  {date}: {count:>6.0f} {bar}")
        sys.exit(0)

    if args.watch > 0:
        print(f"[监控] 每 {args.watch} 小时采集一次 (Ctrl+C 停止)")
        try:
            while True:
                collect_all()
                print(f"\n[等待] {args.watch}h 后下次采集...")
                time.sleep(args.watch * 3600)
        except KeyboardInterrupt:
            print("\n[停止]")
    else:
        collect_all()

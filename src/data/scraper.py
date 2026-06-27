"""
实时模因数据采集器 — 百度指数 / 微博热搜 / 知乎热榜

对齐「微尘哲学」：
- 爬取的数据本身是混沌的噪声场，我们的目标是从中检测确定性结构
- 不追求全覆盖，只追求关键信号的鲁棒抓取

支持：
1. 微博热搜榜（无需API key，公开接口）
2. 百度指数关键词搜索趋势（需cookie，可选）
3. 知乎热榜（公开接口）
4. 本地缓存 + 更新频率控制
"""

import json
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

# ═══════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════

CACHE_DIR = Path("data/scraped")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 已策展热梗的搜索关键词（用于百度指数查询）
MEME_SEARCH_TERMS = {
    "打工人": ["打工人", "打工魂"],
    "内卷": ["内卷", "卷王", "太卷了"],
    "躺平": ["躺平", "躺平主义"],
    "普信男": ["普信男", "普信女", "那么普通那么自信"],
    "小镇做题家": ["小镇做题家", "做题家"],
    "摆烂": ["摆烂", "开摆"],
    "润": ["润学", "提桶跑路"],
    "吗喽": ["吗喽", "马骝"],
    "孔乙己的长衫": ["孔乙己的长衫", "学历是长衫"],
    "精神状态良好": ["精神状态良好", "发疯文学"],
    "科目三": ["科目三", "广西科目三"],
    "后浪": ["后浪", "韭浪"],
    "鸡你太美": ["鸡你太美", "蔡徐坤", "小黑子"],
    "i人/e人": ["i人e人", "MBTI"],
    "遥遥领先": ["遥遥领先", "华为遥遥领先"],
    "XX刺客": ["雪糕刺客", "刺客化"],
    "尊嘟假嘟": ["尊嘟假嘟"],
    "情绪价值": ["情绪价值"],
    "四不": ["不婚不育", "新时代四不青年"],
    "显眼包": ["显眼包"],
    "泼天富贵": ["泼天富贵", "泼天流量"],
    "建议专家不要建议": ["建议专家不要建议", "专家建议"],
}


# ═══════════════════════════════════════════════
# 1. 微博热搜爬虫
# ═══════════════════════════════════════════════

def fetch_weibo_hot_search() -> list[dict]:
    """抓取微博热搜榜（公开API，无需key）。

    返回: [{"rank": 1, "title": "...", "hot_score": 123456, "url": "..."}, ...]
    """
    url = "https://weibo.com/ajax/side/hotSearch"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://weibo.com/",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = []
        for item in data.get("data", {}).get("realtime", [])[:50]:
            word = item.get("word", "")
            raw_hot = item.get("raw_hot", 0)
            items.append({
                "rank": item.get("realpos", 0) or 0,
                "title": word.strip(),
                "hot_score": raw_hot,
                "url": f"https://s.weibo.com/weibo?q={word}",
                "platform": "weibo",
            })
        return items
    except Exception as e:
        print(f"[微博] 抓取失败: {e}")
        return []


# ═══════════════════════════════════════════════
# 2. 知乎热榜爬虫 (401 → 已降级, 需 cookie 才能用)
# ═══════════════════════════════════════════════

def fetch_zhihu_hot_list() -> list[dict]:
    """抓取知乎热榜。知乎 API 需要登录 cookie, 公开接口已封锁 (401).

    如果设置了 ZHIHU_COOKIE 环境变量则使用 cookie 请求,
    否则返回空列表 (静默降级).
    """
    import urllib.request, urllib.error, re, os

    cookie = os.environ.get("ZHIHU_COOKIE", "")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    if cookie:
        headers["Cookie"] = cookie

    # API endpoint (needs auth)
    api_url = "https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total?limit=50"
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = []
        for i, item in enumerate(data.get("data", [])):
            target = item.get("target", {})
            items.append({
                "rank": i + 1,
                "title": target.get("title", "").strip(),
                "hot_score": 0,
                "url": target.get("url", ""),
                "platform": "zhihu",
            })
        if items:
            return items
    except urllib.error.HTTPError as e:
        if e.code == 401 and not cookie:
            return []  # Silent degrade: no cookie available
    except Exception:
        pass

    return []


# ═══════════════════════════════════════════════
# 2.5 百度热搜爬虫 (替代知乎)
# ═══════════════════════════════════════════════

def fetch_baidu_hot_search() -> list[dict]:
    """抓取百度热搜榜 (公开接口, 无需 key).

    返回: [{"rank": 1, "title": "...", "hot_score": 0, "url": "...", "platform": "baidu"}, ...]
    """
    import urllib.request, urllib.error, re

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    url = "https://top.baidu.com/board?tab=realtime"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8")

        # Extract hot search items from Baidu's JSON-in-HTML
        # Pattern: "word":"title","hotScore":"12345"
        items = []
        words = re.findall(r'"word":"([^"]+)"', html)
        hot_scores = re.findall(r'"hotScore":"(\d+)"', html)
        urls = re.findall(r'"url":"([^"]*)"', html)

        for i, word in enumerate(words[:50]):
            score = int(hot_scores[i]) if i < len(hot_scores) else 0
            link = urls[i] if i < len(urls) else ""
            if word.strip():
                items.append({
                    "rank": i + 1,
                    "title": word.strip(),
                    "hot_score": score,
                    "url": link,
                    "platform": "baidu",
                })
        return items
    except Exception as e:
        print(f"[百度] 抓取失败: {e}")
        return []


# ═══════════════════════════════════════════════
# 3. 模因信号检测器
# ═══════════════════════════════════════════════

def detect_meme_signals(hot_items: list[dict]) -> list[dict]:
    """在热搜/热榜中检测已知模因的信号。

    对每条热榜标题，检查是否与已知热梗关键词匹配。
    返回命中列表。

    Returns:
        [{"meme_name", "keyword_matched", "platform", "rank", "title", "timestamp"}, ...]
    """
    signals = []
    now = datetime.now().isoformat()

    for item in hot_items:
        title = item.get("title", "")
        platform = item.get("platform", "")

        for meme_name, keywords in MEME_SEARCH_TERMS.items():
            for kw in keywords:
                if kw in title:
                    signals.append({
                        "meme_name": meme_name,
                        "keyword_matched": kw,
                        "platform": platform,
                        "rank": item.get("rank", 0),
                        "hot_score": item.get("hot_score", 0),
                        "title": title,
                        "url": item.get("url", ""),
                        "timestamp": now,
                    })
                    break  # 每个梗只记录一次匹配

    return signals


# ═══════════════════════════════════════════════
# 4. 数据采集 + 缓存
# ═══════════════════════════════════════════════

def scrape_all() -> dict:
    """执行一次完整采集：微博 + 百度 + 知乎 → 检测模因信号 → 缓存。"""
    print(f"[采集] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 抓取
    weibo_items = fetch_weibo_hot_search()
    print(f"  微博: {len(weibo_items)} 条热搜")

    baidu_items = fetch_baidu_hot_search()
    print(f"  百度: {len(baidu_items)} 条热搜")

    zhihu_items = fetch_zhihu_hot_list()
    if zhihu_items:
        print(f"  知乎: {len(zhihu_items)} 条热榜")
    else:
        print(f"  知乎: 无 (需 ZHIHU_COOKIE)")

    all_items = weibo_items + baidu_items + zhihu_items

    # 信号检测
    signals = detect_meme_signals(all_items)
    if signals:
        print(f"  检测到 {len(signals)} 条模因信号:")
        for s in signals:
            print(f"    [{s['platform']}] #{s['rank']} {s['meme_name']} ← '{s['title'][:50]}'")
    else:
        print(f"  未检测到已知模因信号")

    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "weibo_count": len(weibo_items),
        "baidu_count": len(baidu_items),
        "zhihu_count": len(zhihu_items),
        "signals": signals,
        "weibo_top10": [
            {"rank": w["rank"], "title": w["title"], "hot_score": w["hot_score"]}
            for w in weibo_items[:10]
        ],
        "baidu_top10": [
            {"rank": b["rank"], "title": b["title"], "hot_score": b["hot_score"]}
            for b in baidu_items[:10]
        ],
        "zhihu_top10": [
            {"rank": z["rank"], "title": z["title"], "hot_score": z["hot_score"]}
            for z in zhihu_items[:10]
        ],
    }

    # 缓存
    cache_path = CACHE_DIR / f"scrape_{timestamp}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 追加到信号历史
    if signals:
        history_path = CACHE_DIR / "signal_history.jsonl"
        with open(history_path, "a", encoding="utf-8") as f:
            for s in signals:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return result


def load_signal_history() -> list[dict]:
    """加载历史模因信号。"""
    history_path = CACHE_DIR / "signal_history.jsonl"
    if not history_path.exists():
        return []
    signals = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    signals.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return signals


def get_meme_signal_summary() -> dict:
    """统计各热梗的历史信号。"""
    signals = load_signal_history()
    summary = {}
    for s in signals:
        name = s["meme_name"]
        if name not in summary:
            summary[name] = {"count": 0, "platforms": {}, "max_rank": 999}
        summary[name]["count"] += 1
        plat = s["platform"]
        summary[name]["platforms"][plat] = summary[name]["platforms"].get(plat, 0) + 1
        summary[name]["max_rank"] = min(summary[name]["max_rank"], s["rank"])
    return summary


# ═══════════════════════════════════════════════
# 5. 运行入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 50)
    print("MemeticChaos — 实时模因信号采集")
    print("=" * 50)

    result = scrape_all()

    # 显示历史摘要
    history = load_signal_history()
    if history:
        summary = get_meme_signal_summary()
        print(f"\n历史信号统计 ({len(history)} 条):")
        for name, stats in sorted(summary.items(), key=lambda x: -x[1]["count"]):
            print(f"  {name}: {stats['count']} 次 | 最佳排名 #{stats['max_rank']}")

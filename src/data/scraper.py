"""
实时模因数据采集器 v2.0 — 全量 headline embedding

v1.0: 微博热搜 + 百度热搜 + 知乎热榜 → 关键词匹配 → 二进制信号
v2.0: 同上数据源 → sentence-transformers 全量 embedding → 连续语义信号

核心变化:
  - 不再只做关键词匹配。每条 headline 都编码为 384 维向量。
  - 小时级:  抓取 + embed → hourly/YYYY-MM-DD_HHMMSS.json
  - 日级:    聚合当天所有 hourly 文件 → daily/YYYY-MM-DD.json
  - 关键词匹配保留为 backward-compat 轻量信号

数据量估算:
  - ~100 headline/小时 × 24 = 2400 headline/天
  - 每条 embedding: 384 × 4B = 1.5KB
  - 日存储: ~3.6MB embeddings + ~100KB text ≈ 4MB/天
  - 月存储: ~120MB

对齐「微尘哲学」:
  - 爬取的数据是混沌的噪声场，embedding 从噪声中保留连续结构
  - 不追求全覆盖，追求关键信号的鲁棒抓取
"""

import json
import os
import sys
import time
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error
import numpy as np

# ═══════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════

CACHE_DIR = Path("data/scraped")
HOURLY_DIR = CACHE_DIR / "hourly"
DAILY_DIR = CACHE_DIR / "daily"

for d in [CACHE_DIR, HOURLY_DIR, DAILY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 已策展热梗的搜索关键词
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

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# 保留最近 N 天的 hourly 文件 (7 天后自动清理, daily 永久保留)
HOURLY_RETENTION_DAYS = 7


# ═══════════════════════════════════════════════
# 0. Embedding 模型 — 延迟加载
# ═══════════════════════════════════════════════

_embedder = None


def _get_embedder():
    """延迟加载 sentence-transformers 模型.

    如果模型不可用, 返回 None — scraper 仍可运行,
    只是不会产生 embedding (降级为 v1.0 行为).
    """
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"[模型] {EMBEDDING_MODEL_NAME} 加载完成")
        except ImportError:
            print("[模型] sentence-transformers 未安装 — 降级为无 embedding 模式")
            _embedder = False
        except Exception as e:
            print(f"[模型] 加载失败: {e} — 降级为无 embedding 模式")
            _embedder = False
    return _embedder if _embedder is not False else None


def _json_safe_embedding(vec: np.ndarray) -> list:
    """将 numpy embedding 转为 JSON-safe list (float32 → float, 保留 6 位)."""
    return [round(float(v), 6) for v in vec]


# ═══════════════════════════════════════════════
# 1. 微博热搜爬虫
# ═══════════════════════════════════════════════

def fetch_weibo_hot_search() -> list[dict]:
    """抓取微博热搜榜 (公开 API, 无需 key)."""
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
# 2. 知乎热榜爬虫
# ═══════════════════════════════════════════════

def fetch_zhihu_hot_list() -> list[dict]:
    """抓取知乎热榜 (需 ZHIHU_COOKIE 环境变量)."""
    cookie_val = os.environ.get("ZHIHU_COOKIE", "")
    if cookie_val and not cookie_val.startswith("z_c0="):
        cookie_val = f"z_c0={cookie_val}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    if cookie_val:
        headers["Cookie"] = cookie_val

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
        return items
    except urllib.error.HTTPError as e:
        if e.code == 401 and not cookie_val:
            return []
    except Exception:
        pass
    return []


# ═══════════════════════════════════════════════
# 3. 百度热搜爬虫
# ═══════════════════════════════════════════════

def fetch_baidu_hot_search() -> list[dict]:
    """抓取百度热搜榜 (公开接口, 无需 key)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    url = "https://top.baidu.com/board?tab=realtime"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8")

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
# 4. 模因关键词匹配 (兼容 v1.0, 轻量)
# ═══════════════════════════════════════════════

def detect_meme_signals(hot_items: list[dict]) -> list[dict]:
    """在热搜中检测已知模因关键词 (二进制匹配, v1.0 兼容)."""
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
                    break

    return signals


# ═══════════════════════════════════════════════
# 5. 核心: 全量 headline embedding + 采集
# ═══════════════════════════════════════════════

def embed_headlines(items: list[dict]) -> list[dict] | None:
    """对所有 headline 做 embedding.

    Returns:
        list of {"title", "platform", "rank", "hot_score", "embedding": [384 floats]}
        或 None (模型不可用时)
    """
    model = _get_embedder()
    if model is None:
        return None

    titles = [item["title"] for item in items]
    if not titles:
        return []

    # Batch encode
    embeddings = model.encode(titles, show_progress_bar=False,
                              normalize_embeddings=True)

    results = []
    for item, emb in zip(items, embeddings):
        results.append({
            "title": item["title"],
            "platform": item["platform"],
            "rank": item["rank"],
            "hot_score": item["hot_score"],
            "url": item.get("url", ""),
            "embedding": _json_safe_embedding(emb),
        })
    return results


def scrape_all() -> dict:
    """执行一次完整采集: 抓取 → embed → 关键词匹配 → 存储.

    Returns:
        {"timestamp", "counts": {...}, "signals": [...],
         "headlines_embedded": int, "hourly_file": str}
    """
    ts = datetime.now()
    print(f"[采集] {ts.strftime('%Y-%m-%d %H:%M:%S')}")

    # — 抓取 —
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

    # — 全量 embedding (v2.0 核心) —
    embedded = embed_headlines(all_items)
    if embedded is not None:
        print(f"  Embedding: {len(embedded)} 条 ({EMBEDDING_DIM} 维)")
    else:
        print(f"  Embedding: 跳过 (模型不可用)")

    # — 关键词匹配 (v1.0 兼容) —
    signals = detect_meme_signals(all_items)
    if signals:
        print(f"  关键词命中: {len(signals)} 条")
    else:
        print(f"  关键词命中: 0")

    # — 存储 —
    timestamp = ts.strftime("%Y%m%d_%H%M%S")
    date_str = ts.strftime("%Y-%m-%d")

    hourly_file = {
        "version": "2.0",
        "timestamp": ts.isoformat(),
        "date": date_str,
        "hour": ts.hour,
        "counts": {
            "weibo": len(weibo_items),
            "baidu": len(baidu_items),
            "zhihu": len(zhihu_items),
            "total": len(all_items),
            "embedded": len(embedded) if embedded else 0,
            "signals": len(signals),
        },
        "signals": signals,
        "headlines_embedded": embedded,
        "top_per_platform": {
            "weibo": [
                {"rank": w["rank"], "title": w["title"], "hot_score": w["hot_score"]}
                for w in weibo_items[:10]
            ],
            "baidu": [
                {"rank": b["rank"], "title": b["title"], "hot_score": b["hot_score"]}
                for b in baidu_items[:10]
            ],
            "zhihu": [
                {"rank": z["rank"], "title": z["title"], "hot_score": z["hot_score"]}
                for z in zhihu_items[:10]
            ],
        },
    }

    hourly_path = HOURLY_DIR / f"{date_str}_{ts.strftime('%H%M%S')}.json"
    with open(hourly_path, "w", encoding="utf-8") as f:
        json.dump(hourly_file, f, ensure_ascii=False)

    # 向后兼容: 写入旧版 signal_history.jsonl
    if signals:
        history_path = CACHE_DIR / "signal_history.jsonl"
        with open(history_path, "a", encoding="utf-8") as f:
            for s in signals:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 清理过期 hourly 文件
    _cleanup_hourly(keep_days=HOURLY_RETENTION_DAYS)

    return {
        "timestamp": timestamp,
        "counts": hourly_file["counts"],
        "signals": signals,
        "headlines_embedded": len(embedded) if embedded else 0,
        "hourly_file": str(hourly_path),
    }


def _cleanup_hourly(keep_days: int = 7):
    """删除超过 keep_days 天的 hourly 文件."""
    cutoff = datetime.now() - timedelta(days=keep_days)
    for f in HOURLY_DIR.glob("*.json"):
        try:
            # 文件名格式: YYYY-MM-DD_HHMMSS.json
            date_part = f.stem.split("_")[0]
            file_date = datetime.strptime(date_part, "%Y-%m-%d")
            if file_date < cutoff:
                f.unlink()
        except (ValueError, IndexError):
            pass  # skip files with unexpected names


# ═══════════════════════════════════════════════
# 6. 日聚合: 小时级 → 日级语义状态
# ═══════════════════════════════════════════════

def aggregate_daily(date_str: str = None) -> dict | None:
    """将某一天的所有 hourly 文件聚合为日级语义状态.

    Args:
        date_str: 'YYYY-MM-DD', 默认昨天 (确保当日数据完整)

    Returns:
        {"date", "n_headlines", "n_hours",
         "daily_mean_embedding": [384],
         "meme_similarities": {meme_name: {mean_sim, max_sim, n_hits}},
         "keyword_signals": [...],
         "novelty_vs_previous": float,  # 与前一日的语义距离
         "attention_concentration": float}  # headline 间的 pairwise similarity mean
    """
    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # 加载当天所有 hourly 文件
    hourly_files = sorted(HOURLY_DIR.glob(f"{date_str}_*.json"))
    if not hourly_files:
        print(f"[聚合] {date_str}: 无 hourly 文件, 跳过")
        return None

    print(f"[聚合] {date_str}: {len(hourly_files)} 个 hourly 文件")

    all_headlines = []
    all_signals = []

    for hf in hourly_files:
        try:
            with open(hf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠ 读取 {hf.name} 失败: {e}")
            continue

        emb = data.get("headlines_embedded")
        if emb:
            all_headlines.extend(emb)
        sigs = data.get("signals", [])
        if sigs:
            all_signals.extend(sigs)

    if not all_headlines:
        print(f"[聚合] {date_str}: 无 embedding 数据")
        return None

    # 去重: 同一天内同一平台同一标题只保留最高 rank 的
    seen = {}
    for h in all_headlines:
        key = (h["platform"], h["title"])
        if key not in seen or h["rank"] < seen[key]["rank"]:
            seen[key] = h
    unique_headlines = list(seen.values())
    print(f"  去重: {len(all_headlines)} → {len(unique_headlines)} 条")

    # 日平均 embedding
    embeddings = np.array([h["embedding"] for h in unique_headlines])
    daily_mean = embeddings.mean(axis=0)
    daily_mean = daily_mean / (np.linalg.norm(daily_mean) + 1e-10)  # re-normalize

    # 注意力集中度: headline 间 pairwise cosine similarity 的均值
    pairwise_sim = np.mean(embeddings @ embeddings.T)
    # 排除对角线 (self-similarity = 1.0)
    n = len(embeddings)
    attention_conc = (pairwise_sim * n - n) / (n * (n - 1)) if n > 1 else 1.0

    # 与已知梗的语义相似度
    meme_sims = _compute_meme_similarities(embeddings, daily_mean)

    # 与前一日的新颖度
    novelty = _compute_novelty(date_str, daily_mean)

    # 组装
    result = {
        "version": "2.0",
        "date": date_str,
        "aggregated_at": datetime.now().isoformat(),
        "n_headlines": len(unique_headlines),
        "n_hours": len(hourly_files),
        "daily_mean_embedding": _json_safe_embedding(daily_mean),
        "attention_concentration": round(float(attention_conc), 4),
        "novelty_vs_previous_day": round(float(novelty), 4),
        "meme_similarities": meme_sims,
        "keyword_signals": all_signals,
        "top_headlines": sorted(
            unique_headlines,
            key=lambda h: (h.get("hot_score", 0), -h.get("rank", 999)),
            reverse=True
        )[:20],
    }

    # 存储
    daily_path = DAILY_DIR / f"{date_str}.json"
    with open(daily_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"  → {daily_path}")
    print(f"  注意力集中度: {attention_conc:.3f}")
    print(f"  新颖度 (vs 前日): {novelty:.3f}")
    if meme_sims:
        top_memes = sorted(meme_sims.items(),
                          key=lambda x: x[1]["max_sim"], reverse=True)[:3]
        top_strs = [f"{m}({v['max_sim']:.3f})" for m, v in top_memes]
        print(f"  Top memes: {', '.join(top_strs)}")

    return result


def _compute_meme_similarities(
    embeddings: np.ndarray,
    daily_mean: np.ndarray,
) -> dict:
    """计算当天 headline 与每个已知梗的语义相似度.

    用每个梗的 description 文本做 embedding (缓存在 meme_descriptions.json),
    然后计算当天所有 headline 与该梗的 mean/max cosine similarity.

    Returns:
        {meme_name: {"mean_sim": float, "max_sim": float, "n_hits": int}}
    """
    model = _get_embedder()
    if model is None:
        return {}

    # 加载或构建 meme description embeddings
    meme_desc_path = CACHE_DIR / "meme_description_embeddings.json"

    if meme_desc_path.exists():
        with open(meme_desc_path, "r", encoding="utf-8") as f:
            meme_embs = json.load(f)
    else:
        # 首次: 从 narrative 数据构建 description
        meme_embs = _build_meme_description_embeddings(model)
        if meme_embs is None:
            return {}

    results = {}
    for meme_name, desc_emb in meme_embs.items():
        desc_vec = np.array(desc_emb)
        # cosine similarity already normalized
        sims = embeddings @ desc_vec  # n_headlines × 1
        mean_sim = float(sims.mean())
        max_sim = float(sims.max())
        # n_hits: headlines with sim > 0.5 (moderate threshold)
        n_hits = int(np.sum(sims > 0.5))

        results[meme_name] = {
            "mean_sim": round(mean_sim, 4),
            "max_sim": round(max_sim, 4),
            "n_hits": n_hits,
        }

    return results


def _build_meme_description_embeddings(model) -> dict | None:
    """从 57 条叙事 JSON 构建 meme description embedding.

    遍历 data/processed/narratives/ 和 narratives_from_trends/,
    对每条叙事的 spread_phase description + semantic_drift 做 embedding,
    同一 meme 多条叙事取均值.

    Returns:
        {meme_name: [384 floats]}  或 None (无叙事数据时)
    """
    narratives_dir = Path("data/processed/narratives")
    trends_dir = Path("data/processed/narratives_from_trends")

    meme_texts = {}  # meme_name → [text1, text2, ...]

    for src_dir in [narratives_dir, trends_dir]:
        if not src_dir.exists():
            continue
        for nf in src_dir.glob("*.json"):
            # 跳过聚合文件
            if nf.stem.startswith("_all"):
                continue

            try:
                with open(nf, "r", encoding="utf-8") as f:
                    narr = json.load(f)
            except Exception:
                continue

            # 跳过非 dict 文件
            if not isinstance(narr, dict):
                continue

            # 跳过没有 meme_name 的文件 (如 trends 聚合文件)
            name = narr.get("meme_name", "")
            if not name:
                continue

            # 收集描述文本: phases 的 description + semantic_drift
            texts = []
            for phase in narr.get("spread_phases", []):
                desc = phase.get("description", "")
                if desc:
                    texts.append(desc)
            drift = narr.get("semantic_drift", {})
            orig = drift.get("original_meaning", "")
            curr = drift.get("current_meaning", "")
            if orig:
                texts.append(orig)
            if curr:
                texts.append(curr)

            if texts:
                if name not in meme_texts:
                    meme_texts[name] = []
                meme_texts[name].extend(texts)

    if not meme_texts:
        print("[聚合] 无叙事数据, meme embedding 不可用")
        return None

    # 每个 meme: 所有 text 的均值 embedding
    meme_embs = {}
    for name, texts in meme_texts.items():
        embs = model.encode(texts, show_progress_bar=False,
                           normalize_embeddings=True)
        mean_emb = embs.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
        meme_embs[name] = _json_safe_embedding(mean_emb)

    # 缓存
    cache_path = CACHE_DIR / "meme_description_embeddings.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(meme_embs, f, ensure_ascii=False)

    print(f"[聚合] 构建了 {len(meme_embs)} 个梗的 description embedding")
    return meme_embs


def _compute_novelty(date_str: str, daily_mean: np.ndarray) -> float:
    """计算与前一日的语义新颖度.

    1 - cosine_similarity(today_mean, yesterday_mean)
    如果没有前日数据, 返回 1.0 (完全新颖).
    """
    prev_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
                 ).strftime("%Y-%m-%d")
    prev_path = DAILY_DIR / f"{prev_date}.json"

    if not prev_path.exists():
        return 1.0

    try:
        with open(prev_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        prev_mean = np.array(prev["daily_mean_embedding"])
        sim = float(np.dot(daily_mean, prev_mean))  # both normalized
        return round(1.0 - max(0.0, sim), 4)
    except Exception:
        return 1.0


# ═══════════════════════════════════════════════
# 7. 兼容 v1.0 的信号查询
# ═══════════════════════════════════════════════

def load_signal_history() -> list[dict]:
    """加载历史模因关键词信号 (v1.0 兼容)."""
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
    """统计各热梗的历史关键词信号 (v1.0 兼容)."""
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


def load_daily_semantic_state(date_str: str = None) -> dict | None:
    """加载日级语义状态 (v2.0 查询接口).

    Args:
        date_str: 'YYYY-MM-DD', 默认最新
    """
    if date_str is None:
        files = sorted(DAILY_DIR.glob("*.json"))
        if not files:
            return None
        date_str = files[-1].stem

    path = DAILY_DIR / f"{date_str}.json"
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_daily_history(n_days: int = 30) -> list[dict]:
    """加载最近 N 天的日级语义状态序列."""
    files = sorted(DAILY_DIR.glob("*.json"))[-n_days:]
    history = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                history.append(json.load(f))
        except Exception:
            pass
    return history


# ═══════════════════════════════════════════════
# 8. 运行入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="MemeticChaos 实时模因采集 v2.0")
    parser.add_argument("--aggregate", type=str, default=None,
                        help="聚合指定日期 (YYYY-MM-DD), 默认昨天")
    parser.add_argument("--history", action="store_true",
                        help="显示日级语义状态历史")
    parser.add_argument("--n-days", type=int, default=7,
                        help="历史天数 (默认 7)")
    args = parser.parse_args()

    if args.aggregate:
        # 仅聚合模式
        print("=" * 50)
        print(f"MemeticChaos — 日聚合: {args.aggregate}")
        print("=" * 50)
        aggregate_daily(args.aggregate)

    elif args.history:
        # 历史查询
        history = load_daily_history(n_days=args.n_days)
        print(f"\n最近 {len(history)} 天日级语义状态:")
        print(f"{'Date':<12s} {'Hdls':>5s} {'Hrs':>4s} {'Conc':>6s} {'Novelty':>8s}  Top Memes")
        print("-" * 70)
        for d in history:
            top = sorted(d.get("meme_similarities", {}).items(),
                        key=lambda x: x[1]["max_sim"], reverse=True)[:2]
            top_str = ", ".join(f"{m}({v['max_sim']:.2f})" for m, v in top)
            print(f"{d['date']:<12s} {d['n_headlines']:>5d} {d['n_hours']:>4d} "
                  f"{d['attention_concentration']:>6.3f} "
                  f"{d.get('novelty_vs_previous_day', 0):>8.3f}  "
                  f"{top_str}")

    else:
        # 正常采集模式
        print("=" * 50)
        print("MemeticChaos — 实时模因采集 v2.0")
        print("=" * 50)

        result = scrape_all()

        # 历史摘要
        history = load_signal_history()
        if history:
            summary = get_meme_signal_summary()
            print(f"\n历史信号统计 ({len(history)} 条):")
            for name, stats in sorted(summary.items(),
                                       key=lambda x: -x[1]["count"]):
                print(f"  {name}: {stats['count']} 次 | "
                      f"最佳排名 #{stats['max_rank']}")

        # 日聚合提示
        now = datetime.now()
        if now.hour == 23:
            print(f"\n[自动] 触发日聚合...")
            aggregate_daily(now.strftime("%Y-%m-%d"))

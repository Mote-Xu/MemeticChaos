"""
实时信号管线 — 新梗检测 → LLM叙事抽取 → 概念打分 → 约束场更新

在 live_pipeline 的基础上增加闭环：
1. 从 scraper 数据中检测未知热梗候选
2. 查 Google Trends 验证其搜索热度
3. 自动触发 LLM 叙事生成 + 概念打分
4. 更新 llm_concept_scores.json → 预测模型自动感知

用法:
    python src/data/signal_pipeline.py              # 单次检测+更新
    python src/data/signal_pipeline.py --force       # 强制重检所有候选
"""

import json, sys, os, time
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = ROOT / "data/scraped"
COLLECTOR_DIR = ROOT / "data/collector"
PROCESSED_DIR = ROOT / "data/processed"
CANDIDATE_PATH = COLLECTOR_DIR / "candidate_memes.jsonl"

# 已知梗 (不会作为候选)
KNOWN_MEMES = {
    "打工人", "内卷", "躺平", "普信男", "普信", "小镇做题家", "摆烂", "润",
    "吗喽", "孔乙己的长衫", "孔乙己 长衫", "精神状态", "科目三", "后浪",
    "鸡你太美", "i人 e人", "遥遥领先", "遥遥领先 华为", "尊嘟假嘟",
    "情绪价值", "不结婚", "不婚不育", "显眼", "显眼包",
    "建议专家不要建议", "专家建议", "南方小土豆", "雪糕刺客",
    "发疯文学 梗", "多巴胺穿搭", "凡尔赛", "元宇宙", "citywalk",
    "芭比Q", "栓Q", "美拉德", "破防", "社恐", "社死", "精神内耗",
    "原生家庭", "牛马", "鼠鼠",
}

# 噪声词 (热搜常见但不是模因)
NOISE_WORDS = {
    "最珍贵的记忆", "韩国球迷", "极端高温", "志愿填报", "保洁奶奶",
    "吴艳妮", "曾沛慈", "浪姐", "日本精锐", "胆大包天", "冠军赛",
    "王俊凯", "宁德时代", "曾毓群", "腾讯市值", "华为昇腾",
    "高考", "录取", "天气", "暴雨", "地震", "火灾", "事故",
    "政策", "发布会", "开幕式", "闭幕式", "春晚", "国庆",
}


def load_recent_scrapes(hours: int = 48) -> list[dict]:
    """加载最近 N 小时的采集结果."""
    cutoff = datetime.now() - timedelta(hours=hours)
    results = []
    for fp in sorted(CACHE_DIR.glob("scrape_*.json"), reverse=True):
        try:
            ts_str = fp.stem.replace("scrape_", "")
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            if ts < cutoff:
                break
            with open(fp, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except (ValueError, json.JSONDecodeError):
            continue
    return results


def detect_candidates(scrapes: list[dict], min_occurrences: int = 2) -> list[dict]:
    """从采集数据中检测未知热梗候选.

    规则:
    - 在超过 min_occurrences 次采集结果中出现
    - 不在已知梗列表中
    - 不是噪声词
    - 标题长度合理 (3-30字, 排除太短/太长的)
    """
    title_counts = Counter()
    title_platforms = defaultdict(set)
    title_scores = defaultdict(list)

    for scrape in scrapes:
        seen_this_scrape = set()
        for key in ["weibo_top10", "baidu_top10", "zhihu_top10"]:
            platform = key.replace("_top10", "")
            for item in scrape.get(key, []):
                title = item.get("title", "").strip()
                if not title or title in seen_this_scrape:
                    continue
                seen_this_scrape.add(title)

                # Filter: known memes
                if any(m in title for m in KNOWN_MEMES):
                    continue
                # Filter: noise
                if any(n in title for n in NOISE_WORDS):
                    continue
                # Filter: too short/long
                if len(title) < 4 or len(title) > 30:
                    continue
                # Filter: contains numbers (likely news, not memes)
                if any(c.isdigit() for c in title):
                    if len(title) < 10:
                        continue

                title_counts[title] += 1
                title_platforms[title].add(platform)
                title_scores[title].append(item.get("hot_score", 0))

    # Filter by min occurrences
    candidates = []
    for title, count in title_counts.most_common(30):
        if count >= min_occurrences:
            avg_score = sum(title_scores[title]) / len(title_scores[title]) if title_scores[title] else 0
            candidates.append({
                "title": title,
                "occurrences": count,
                "platforms": list(title_platforms[title]),
                "avg_hot_score": round(avg_score, 1),
                "detected_at": datetime.now().isoformat(),
            })

    return candidates


def check_trends_availability(candidate_title: str) -> Optional[dict]:
    """检查候选词是否有 Google Trends 数据."""
    trends_path = COLLECTOR_DIR / "google_trends_2015_2025.json"
    if not trends_path.exists():
        return None

    with open(trends_path, "r", encoding="utf-8") as f:
        memes = json.load(f).get("memes", {})

    # Direct match
    if candidate_title in memes:
        return {"keyword": candidate_title, "data": memes[candidate_title]}

    # Partial match
    for kw, data in memes.items():
        if candidate_title in kw or kw in candidate_title:
            return {"keyword": kw, "data": data}

    return None


def generate_narrative_for_candidate(title: str, trends_info: Optional[dict]) -> Optional[dict]:
    """为候选梗生成叙事 (LLM)."""
    from src.data.narrative_from_trends import curve_to_text, generate_narrative

    if trends_info and trends_info.get("data"):
        keyword = trends_info["keyword"]
        curve_text = curve_to_text(keyword, trends_info["data"])
        narrative = generate_narrative(keyword, curve_text)
        if "error" not in narrative:
            return {"meme_name": keyword, "narrative": narrative, "source": "trends_curve"}
        return None

    # No trends data: generate from title only
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
    )

    prompt = f"""你是互联网模因分析师。以下是一个可能的新热梗标题: "{title}"

请根据标题推断，输出一个简化的传播叙事 JSON:
{{
  "meme_name": "{title}",
  "spread_phases": [
    {{"phase": "emergence", "time_range": "最近", "description": "从热搜标题推断的初始传播描述"}}
  ],
  "curve_shape": "脉冲型",
  "narrative_summary": "一句话描述这个梗的传播模式",
  "social_context_hint": "可能的社会背景"
}}

只输出 JSON，不要其他文字。"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=500,
        )
        raw = resp.choices[0].message.content
        json_str = raw
        for delim in ["```json", "```"]:
            if delim in raw:
                json_str = raw.split(delim)[1].split("```")[0]
                break
        narrative = json.loads(json_str.strip())
        return {"meme_name": title, "narrative": narrative, "source": "title_inference"}
    except Exception as e:
        print(f"    LLM 叙事生成失败: {e}")
        return None


def score_candidate(title: str, narrative: dict) -> Optional[dict]:
    """对候选梗进行 LLM 概念打分."""
    from src.constraint.llm_concept_scorer import score_one_narrative, concept_to_constraint_llm

    result = score_one_narrative(title, narrative, retry=1)
    if result and "scores" in result:
        constraint = concept_to_constraint_llm(result["scores"])
        return {
            "concept_scores": result["scores"],
            "constraint": constraint,
            "notes": result.get("scoring_notes", ""),
        }
    return None


def update_candidates_db(candidates: list[dict], new_entries: dict):
    """更新候选梗数据库."""
    existing = []
    if CANDIDATE_PATH.exists():
        with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    # Append new entries
    with open(CANDIDATE_PATH, "a", encoding="utf-8") as f:
        for title, entry in new_entries.items():
            entry["title"] = title
            entry["recorded_at"] = datetime.now().isoformat()
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_llm_scores(new_scores: dict):
    """更新 LLM 概念分数文件."""
    scores_path = PROCESSED_DIR / "llm_concept_scores.json"
    existing = {}
    if scores_path.exists():
        with open(scores_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    updated = False
    for name, entry in new_scores.items():
        if name not in existing:
            existing[name] = entry
            updated = True

    if updated:
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"  [LLM分数] 新增 {len(new_scores)} 个梗 → {scores_path}")


def run_pipeline(max_new: int = 3, force: bool = False):
    """执行一次完整的信号管线.

    Args:
        max_new: 每次最多处理 N 个新候选
        force: 强制处理所有候选 (忽略已处理记录)
    """
    print(f"[信号管线] {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # 1. Load recent scrapes
    scrapes = load_recent_scrapes(48)
    print(f"  采集记录: {len(scrapes)} 次 (48h)")

    if not scrapes:
        print("  无采集数据，跳过")
        return

    # 2. Detect candidates
    candidates = detect_candidates(scrapes, min_occurrences=2)
    print(f"  候选梗: {len(candidates)} 个")
    for c in candidates[:10]:
        print(f"    {c['title'][:40]} ({c['occurrences']}次, {','.join(c['platforms'])})")

    if not candidates:
        return

    # 3. Check which already processed
    processed = set()
    if CANDIDATE_PATH.exists() and not force:
        with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    if e.get("status") == "scored":
                        processed.add(e.get("title", ""))
                except json.JSONDecodeError:
                    continue

    fresh = [c for c in candidates if c["title"] not in processed][:max_new]
    if not fresh:
        print(f"  所有候选已处理 ({len(processed)} 个)")
        return

    print(f"\n  处理 {len(fresh)} 个新候选...")

    # 4. For each fresh candidate: trends lookup → narrative → score
    new_scores = {}
    new_entries = {}

    for c in fresh:
        title = c["title"]
        print(f"\n  [{title[:40]}]")

        # Check Google Trends
        trends = check_trends_availability(title)
        if trends:
            print(f"    Trends: {trends['keyword']} ({len(trends['data'])} 月数据)")
        else:
            print(f"    Trends: 无数据，用标题推断")

        # Generate narrative
        print(f"    生成叙事...", end=" ", flush=True)
        nar_result = generate_narrative_for_candidate(title, trends)
        if not nar_result:
            print("失败")
            new_entries[title] = {"status": "narrative_failed"}
            continue
        print(f"✓ ({nar_result['source']})")

        # Score concepts
        print(f"    概念打分...", end=" ", flush=True)
        scored = score_candidate(title, nar_result["narrative"])
        if not scored:
            print("失败")
            new_entries[title] = {"status": "scoring_failed"}
            continue
        print(f"✓")

        meme_name = nar_result["meme_name"]
        new_scores[meme_name] = scored
        new_entries[title] = {
            "status": "scored",
            "meme_name": meme_name,
            "trends_keyword": trends["keyword"] if trends else None,
        }

        time.sleep(0.3)

    # 5. Save results
    if new_scores:
        update_llm_scores(new_scores)
    if new_entries:
        update_candidates_db(candidates, new_entries)

    # 6. Summary
    print(f"\n{'='*50}")
    print(f"[信号管线] 完成")
    print(f"  候选检测: {len(candidates)}")
    print(f"  新处理:   {len(new_scores)} 个成功")
    if new_scores:
        print(f"  新增梗:   {', '.join(new_scores.keys())}")
    print(f"  下次采集将包含新梗的约束场数据")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="强制重检所有候选")
    parser.add_argument("--max", type=int, default=3, help="最多处理N个新候选")
    args = parser.parse_args()

    run_pipeline(max_new=args.max, force=args.force)

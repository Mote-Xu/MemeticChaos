"""
LLM 月度集体叙事摘要生成器

每月/每周调用一次，输入:
1. 当前微博热搜 TOP 30
2. 近期活跃的模因信号
3. Google Trends 异常检测
4. 上月摘要（连续性）

输出: 结构化的月度集体叙事状态描述（用于报告 NL 部分 + Dashboard）

用法:
    python src/data/monthly_narrative.py              # 生成本月摘要
    python src/data/monthly_narrative.py --month 2025-06  # 指定月份
"""

import json, sys, os, time
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

# API config
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

ROOT = Path(__file__).parent.parent.parent
SCRAPED_DIR = ROOT / "data/scraped"
COLLECTOR_DIR = ROOT / "data/collector"
PROCESSED_DIR = ROOT / "data/processed"
OUTPUT_PATH = PROCESSED_DIR / "monthly_narratives.jsonl"

# 5 类别
CATEGORY_NAMES = ["解构自嘲", "攻击发泄", "虚无退却", "身份认同", "纯粹娱乐"]


def load_latest_weibo_hot(n: int = 30) -> list[dict]:
    """加载最新的微博热搜数据."""
    scraped_files = sorted(SCRAPED_DIR.glob("scrape_*.json"), reverse=True)
    if not scraped_files:
        return []

    # Get the latest scrape file
    latest = scraped_files[0]
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract hot search items
    items = []
    if isinstance(data, list):
        items = data[:n]
    elif isinstance(data, dict):
        hot_list = data.get("hot_search") or data.get("data") or []
        items = hot_list[:n]

    return [{"title": item.get("title", item.get("word", str(item))),
             "rank": item.get("rank", i+1),
             "hot_score": item.get("hot_score", item.get("score", 0))}
            for i, item in enumerate(items)]


def load_recent_signals(hours: int = 168) -> list[dict]:
    """加载最近 N 小时的模因信号 (signal_history.jsonl)."""
    signal_file = SCRAPED_DIR / "signal_history.jsonl"
    if not signal_file.exists():
        signal_file = COLLECTOR_DIR / "signal_history.jsonl"
    if not signal_file.exists():
        return []

    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    signals = []
    with open(signal_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sig = json.loads(line.strip())
                ts = sig.get("timestamp", "")
                if ts >= cutoff:
                    signals.append(sig)
            except json.JSONDecodeError:
                continue

    # Aggregate: count meme mentions
    meme_counts = Counter(s.get("meme_name", "unknown") for s in signals)
    top_memes = [{"meme": m, "count": c} for m, c in meme_counts.most_common(15)]

    return top_memes


def load_trends_anomalies(months_back: int = 3) -> list[dict]:
    """从 Google Trends 检测近期异常."""
    trends_path = COLLECTOR_DIR / "google_trends_2015_2025.json"
    if not trends_path.exists():
        return []

    with open(trends_path, "r", encoding="utf-8") as f:
        memes = json.load(f).get("memes", {})

    anomalies = []
    for meme_name, monthly in memes.items():
        months = sorted(monthly.keys())
        if len(months) < 12:
            continue
        # Recent 3 months
        recent = months[-months_back:]
        values = [monthly[m] for m in recent]
        # Historical stats (excluding last 3 months)
        hist_vals = [monthly[m] for m in months[:-months_back]]
        if not hist_vals:
            continue
        mean_h = sum(hist_vals) / len(hist_vals)
        std_h = (sum((v - mean_h)**2 for v in hist_vals) / len(hist_vals)) ** 0.5
        if std_h == 0:
            continue

        for m, v in zip(recent, values):
            z = (v - mean_h) / std_h
            if z > 2.0:  # significant spike
                anomalies.append({
                    "meme": meme_name, "month": m,
                    "value": v, "z_score": round(z, 2),
                })

    return sorted(anomalies, key=lambda x: -x["z_score"])[:10]


def load_last_summary() -> str:
    """加载上个月的摘要."""
    if not OUTPUT_PATH.exists():
        return ""
    lines = []
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    if lines:
        try:
            last = json.loads(lines[-1])
            return last.get("summary", "")
        except json.JSONDecodeError:
            pass
    return ""


def build_prompt(weibo_hot: list, signals: list, anomalies: list,
                 last_summary: str, month_label: str) -> str:
    """构建月度叙事摘要 prompt."""
    weibo_text = "\n".join([
        f"  {i+1}. {item['title']} (热度: {item.get('hot_score', '?')})"
        for i, item in enumerate(weibo_hot[:20])
    ]) if weibo_hot else "  (无微博数据)"

    signals_text = "\n".join([
        f"  - {s['meme']}: {s['count']} 次提及"
        for s in signals[:10]
    ]) if signals else "  (无模因信号)"

    anomalies_text = "\n".join([
        f"  - {a['meme']}: {a['month']} z={a['z_score']} (值={a['value']:.0f})"
        for a in anomalies[:8]
    ]) if anomalies else "  (无显著异常)"

    last_text = f"\n## 上月摘要\n{last_summary}" if last_summary else ""

    return f"""你是中国互联网集体情感分析师。请根据以下数据，撰写 {month_label} 的月度集体情感叙事摘要。

## 本月微博热搜 TOP 20
{weibo_text}

## 近期活跃模因信号 (7天内)
{signals_text}

## Google Trends 异常检测 (近3月)
{anomalies_text}
{last_text}

## 分析要求

请从以下维度分析当前中国互联网集体情感状态:

1. **主导情感基调**: 当前网民情绪的总体基调是什么？(愤怒/焦虑/解构自嘲/虚无/娱乐/希望...)
2. **活跃叙事类型**: 哪些类型的模因在驱动当前讨论？(身份认同/阶层冲突/纯粹娱乐/虚无退却/攻击发泄)
3. **注意力结构**: 注意力是集中在少数话题还是分散？是否出现跨圈层事件？
4. **与上月对比**: 情绪状态是否有显著变化？趋势方向？
5. **预测信号**: 有哪些早期信号值得关注？可能出现什么新叙事？

## 输出格式
严格输出 JSON:
```json
{{
  "month": "{month_label}",
  "dominant_emotion": "主导情感 (如: 解构自嘲 + 底层焦虑)",
  "emotion_intensity": 0.7,
  "active_narrative_types": ["虚无退却", "解构自嘲"],
  "attention_structure": "分散 / 集中 / 中等",
  "key_topics": ["经济压力", "青年失业", "..."],
  "trend_direction": "向混沌漂移 / 向秩序建构 / 稳定 / 不确定",
  "emerging_signals": ["..."],
  "summary": "一段 3-5 句话的完整叙事摘要，描述当前中国互联网集体情感系统的状态。要具体、有洞察力，不要泛泛而谈。"
}}
```"""


def generate_monthly_narrative(month_label: str = None) -> dict:
    """生成月度集体叙事摘要."""
    if month_label is None:
        month_label = datetime.now().strftime("%Y-%m")

    print(f"[月度叙事摘要] {month_label}")

    # Load data
    weibo_hot = load_latest_weibo_hot(30)
    signals = load_recent_signals(168)  # 7 days
    anomalies = load_trends_anomalies(3)
    last_summary = load_last_summary()

    print(f"  微博热搜: {len(weibo_hot)} 条")
    print(f"  模因信号: {len(signals)} 个活跃梗")
    print(f"  Trends异常: {len(anomalies)} 个")

    if not weibo_hot and not signals:
        print("  ⚠️ 无实时数据，使用 Google Trends 数据降级")
        # Fallback: describe from trends data
        return _fallback_from_trends(month_label)

    prompt = build_prompt(weibo_hot, signals, anomalies, last_summary, month_label)

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        raw = resp.choices[0].message.content

        # Extract JSON
        json_str = raw
        for delim in ["```json", "```"]:
            if delim in raw:
                json_str = raw.split(delim)[1].split("```")[0]
                break
        result = json.loads(json_str.strip())

        # Append to history
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"  主导情感: {result.get('dominant_emotion', '?')}")
        print(f"  趋势: {result.get('trend_direction', '?')}")
        print(f"  [已保存] → {OUTPUT_PATH}")

        return result

    except Exception as e:
        print(f"  ✗ 生成失败: {e}")
        return _fallback_from_trends(month_label)


def _fallback_from_trends(month_label: str) -> dict:
    """无实时数据时从 Google Trends 生成降级摘要."""
    # Load collective dynamics
    trends_path = COLLECTOR_DIR / "google_trends_2015_2025.json"
    if not trends_path.exists():
        return {"month": month_label, "error": "无数据", "summary": "数据不可用"}

    with open(trends_path, "r", encoding="utf-8") as f:
        memes = json.load(f).get("memes", {})

    # Find month in trends
    month_attention = {}
    for name, monthly in memes.items():
        if month_label in monthly:
            month_attention[name] = monthly[month_label]

    top = sorted(month_attention.items(), key=lambda x: -x[1])[:10]
    total = sum(month_attention.values())

    # Map to categories (inline to avoid circular imports)
    CATEGORY_CHAOS = {
        "解构自嘲": -0.33, "攻击发泄": -0.62, "虚无退却": -0.59,
        "身份认同": +0.34, "纯粹娱乐": +0.19,
    }
    TREND_TO_MEME = {
        "打工人": ("打工人", "解构自嘲"), "内卷": ("内卷 / 卷", "身份认同"),
        "躺平": ("躺平", "虚无退却"), "普信男": ("普信男", "攻击发泄"),
        "普信": ("普信男", "攻击发泄"), "小镇做题家": ("小镇做题家", "身份认同"),
        "摆烂": ("摆烂", "虚无退却"), "润": ("润", "虚无退却"),
        "后浪": ("后浪", "身份认同"), "鸡你太美": ("鸡你太美", "纯粹娱乐"),
        "科目三": ("科目三", "纯粹娱乐"), "孔乙己的长衫": ("孔乙己的长衫", "身份认同"),
        "精神状态": ("精神状态良好", "解构自嘲"), "雪糕刺客": ("XX刺客", "攻击发泄"),
        "遥遥领先": ("遥遥领先", "纯粹娱乐"), "遥遥领先 华为": ("遥遥领先", "纯粹娱乐"),
        "吗喽": ("吗喽", "解构自嘲"), "鼠鼠": ("鼠鼠", "解构自嘲"),
        "牛马": ("牛马", "解构自嘲"), "i人 e人": ("i人/e人", "身份认同"),
        "情绪价值": ("情绪价值", "身份认同"), "原生家庭": ("原生家庭", "身份认同"),
        "尊嘟假嘟": ("尊嘟假嘟", "纯粹娱乐"), "凡尔赛": ("凡尔赛", "纯粹娱乐"),
        "元宇宙": ("元宇宙", "纯粹娱乐"), "citywalk": ("citywalk", "纯粹娱乐"),
        "芭比Q": ("芭比Q", "纯粹娱乐"), "栓Q": ("栓Q", "纯粹娱乐"),
        "美拉德": ("美拉德", "纯粹娱乐"), "南方小土豆": ("南方小土豆", "纯粹娱乐"),
        "破防": ("破防", "解构自嘲"), "社恐": ("社恐", "解构自嘲"),
        "社死": ("社死", "解构自嘲"), "精神内耗": ("精神内耗", "虚无退却"),
        "不结婚": ("四不/不婚不育", "虚无退却"), "不婚不育": ("四不/不婚不育", "虚无退却"),
        "专家建议": ("建议专家不要建议", "攻击发泄"),
        "建议专家不要建议": ("建议专家不要建议", "攻击发泄"),
        "显眼": ("显眼包", "纯粹娱乐"), "显眼包": ("显眼包", "纯粹娱乐"),
        "发疯文学 梗": ("发疯文学", "解构自嘲"), "多巴胺穿搭": ("多巴胺穿搭", "纯粹娱乐"),
    }
    cat_totals = defaultdict(float)
    for name, val in month_attention.items():
        if name in TREND_TO_MEME:
            cat = TREND_TO_MEME[name][1]
            cat_totals[cat] += val

    dom_cat = max(cat_totals, key=cat_totals.get) if cat_totals else "未知"

    result = {
        "month": month_label,
        "dominant_emotion": f"{dom_cat}主导",
        "emotion_intensity": 0.5,
        "active_narrative_types": [dom_cat],
        "attention_structure": "分散" if len(top) > 5 else "集中",
        "key_topics": [t[0] for t in top[:5]],
        "trend_direction": "数据不足",
        "emerging_signals": [],
        "summary": f"{month_label} 中国互联网集体情感由{dom_cat}类模因主导，"
                   f"TOP5关键词: {', '.join(t[0] for t in top[:5])}。"
                   f"总注意力: {total:.0f}。",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=str, default="", help="指定月份 YYYY-MM")
    args = parser.parse_args()

    month = args.month or datetime.now().strftime("%Y-%m")
    result = generate_monthly_narrative(month)
    print(f"\n{'='*60}")
    print(json.dumps(result, ensure_ascii=False, indent=2))

"""
从 Google Trends 注意力曲线自动生成传播叙事

用 LLM 分析曲线形状 → 提取传播阶段、峰值时机、持续形态。
不需要视频转录，直接从真实数据生成结构化叙事。

用法:
    python src/data/narrative_from_trends.py          # 批量生成
    python src/data/narrative_from_trends.py --meme 躺平  # 单个
"""

import json, sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

DATA_DIR = Path("data/collector")
OUT_DIR = Path("data/processed/narratives_from_trends")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def curve_to_text(name: str, data: dict) -> str:
    """将 Google Trends 曲线转为 LLM 可读的描述文本。"""
    months = sorted(data.keys())
    values = [data[m] for m in months]
    peak_val = max(values)
    peak_month = months[values.index(peak_val)]
    recent_3 = sum(values[-3:]) / min(3, len(values)) if values else 0
    recent_12 = sum(values[-12:]) / min(12, len(values)) if values else 0

    # Detect phases
    threshold = max(0.5, peak_val * 0.15)
    above = [(m, v) for m, v in zip(months, values) if v >= threshold]
    emergence = above[0][0] if above else months[0]
    decay = above[-1][0] if above else months[-1]

    # Resurgence check
    post_peak = [(m, v) for m, v in zip(months, values) if m > peak_month and v > peak_val * 0.3]
    has_resurgence = len(post_peak) > 0

    # Shape description
    if len(values) > 24 and recent_12 > peak_val * 0.4:
        shape = "长期持续型"
    elif peak_val > 50 and len(above) < 6:
        shape = "脉冲爆发型"
    elif has_resurgence:
        shape = "复燃波动型"
    elif recent_12 < peak_val * 0.15:
        shape = "消退型"
    else:
        shape = "长尾型"

    text = f"""关键词: {name}
数据范围: {months[0]} 至 {months[-1]} ({len(values)} 个月)
峰值: {peak_month} ({peak_val:.0f})
形态: {shape}
萌芽: {emergence}
衰退: {decay}
近12月均值: {recent_12:.1f} (峰值占比 {recent_12/max(1,peak_val)*100:.0f}%)
复燃: {'是' if has_resurgence else '否'}
月数据: {', '.join(f'{m}={v:.0f}' for m,v in list(zip(months,values))[::max(1,len(values)//20)])}"""
    return text


def generate_narrative(name: str, curve_text: str) -> dict:
    """用 DeepSeek 从曲线生成结构化叙事。"""
    prompt = f"""你是互联网模因分析师。以下是中国互联网一个热梗的 Google Trends 搜索关注度数据。

{curve_text}

请从曲线形态中推断该梗的传播生命周期，输出 JSON:

{{
  "meme_name": "{name}",
  "peak_month": "峰值月份",
  "peak_value": 峰值数值,
  "spread_phases": [
    {{"phase": "emergence", "time_range": "从萌芽到爆发前", "description": "..."}},
    {{"phase": "peak", "time_range": "峰值区间", "description": "..."}},
    {{"phase": "decay", "time_range": "衰退区间", "description": "..."}}
  ],
  "curve_shape": "脉冲/长尾/复燃/消退/持续",
  "estimated_duration_months": 有效传播月数,
  "resurgence_detected": true/false,
  "narrative_summary": "用一句话总结这个梗的传播模式",
  "social_context_hint": "根据曲线时间推断可能的社会背景（如2021年峰值可能关联经济压力/疫情等）"
}}

只输出 JSON，不要其他文字。"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1000
        )
        raw = resp.choices[0].message.content
        json_str = raw
        for delim in ["```json", "```"]:
            if delim in raw:
                json_str = raw.split(delim)[1].split("```")[0]
                break
        return json.loads(json_str.strip())
    except Exception as e:
        return {"error": str(e), "raw": raw if 'raw' in dir() else ""}


def batch_generate():
    """批量从所有 Google Trends 曲线生成叙事。"""
    trends_path = DATA_DIR / "google_trends_2015_2025.json"
    if not trends_path.exists():
        print("No trends data found")
        return
    with open(trends_path, "r", encoding="utf-8") as f:
        memes = json.load(f)["memes"]

    print(f"Generating narratives for {len(memes)} memes from trends curves...")
    results = {}
    for i, (name, data) in enumerate(memes.items()):
        if len(data) < 3:
            continue
        curve_text = curve_to_text(name, data)
        print(f"\n[{i+1}/{len(memes)}] {name} ({len(data)} pts)...")
        narrative = generate_narrative(name, curve_text)
        results[name] = narrative

        if "error" in narrative:
            print(f"  FAILED: {narrative['error']}")
        else:
            phases = len(narrative.get("spread_phases", []))
            shape = narrative.get("curve_shape", "?")
            print(f"  {shape} | {phases} phases | {narrative.get('narrative_summary','')[:80]}")

        # Save individual
        safe_name = name.replace("/", "_").replace(" ", "_")
        with open(OUT_DIR / f"{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump(narrative, f, ensure_ascii=False, indent=2)
        time.sleep(0.3)

    # Save batch
    with open(OUT_DIR / "_all_narratives_from_trends.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone: {len(results)} narratives generated")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meme", type=str, default="")
    args = parser.parse_args()

    if args.meme:
        trends_path = DATA_DIR / "google_trends_2015_2025.json"
        with open(trends_path, "r", encoding="utf-8") as f:
            memes = json.load(f)["memes"]
        if args.meme in memes:
            text = curve_to_text(args.meme, memes[args.meme])
            print(text)
            print("\n--- Generating ---")
            nar = generate_narrative(args.meme, text)
            print(json.dumps(nar, ensure_ascii=False, indent=2))
        else:
            print(f"Not found: {args.meme}")
    else:
        batch_generate()

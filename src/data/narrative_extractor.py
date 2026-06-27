"""
B站梗指南视频 → 结构化传播叙事抽取

从 Video_to_Text 转录中，用 LLM 提取每个热梗的：
1. 起源时间、平台、触发事件
2. 传播阶段（萌芽→爆发→泛化→固化/消亡/变异）
3. 关键传播节点（KOL 参与、平台迁移、媒体介入）
4. 变异体及其出现时间
5. 社会背景与舆论反转
6. 语义漂移轨迹

输出回填到 memes_2020_2025.json 的 narrative 字段。
"""

import os
import json
import time
from pathlib import Path
from openai import OpenAI

# ── DeepSeek API 配置 ──
DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY_REMOVED"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ── 路径 ──
VIDEO_BASE = r"E:\Desktop\Video_to_Text\outputs\2026-06-25"
OUTPUT_DIR = Path("data/processed/narratives")
CURATED_JSON = Path("data/curated/memes_2020_2025.json")


def load_transcript_text(video_dir: str) -> str:
    """加载单个视频的完整转录文本（拼接所有段落）。"""
    rp = os.path.join(video_dir, "report.json")
    with open(rp, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("transcript", [])
    # 按时间排序，拼接完整文本
    segments.sort(key=lambda s: s.get("start_sec", 0))
    full_text = " ".join([s.get("text", "") for s in segments])
    return full_text, data["video"].get("duration_sec", 0)


def extract_narrative(video_name: str, transcript_text: str,
                      meme_name: str = None) -> dict:
    """用 DeepSeek 从转录文本中抽取结构化传播叙事。

    Args:
        video_name: 视频目录名（含梗名）
        transcript_text: 完整 ASR 转录文本
        meme_name: 已知的热梗名（可选，用于 LLM 锚定）

    Returns:
        {"origin", "spread_phases", "mutations", "social_context",
         "semantic_drift", "key_events", "raw_llm_output"}
    """
    prompt = f"""你是一个互联网模因（meme）研究者。以下是一个B站"梗指南"视频的语音转文字文本。
视频主题: {video_name}

ASR 转录文本:
```
{transcript_text[:8000]}
```

请从这个文本中提取关于该热梗传播过程的**结构化信息**，输出 JSON 格式:

{{
  "meme_name": "梗的名称（用正确汉字纠正ASR错误）",
  "origin": {{
    "time": "最早出现的大致时间（如 2020年10月）",
    "platform": "最早出现的平台（如 微博/B站/贴吧/知乎）",
    "trigger_event": "触发该梗产生的社会事件或背景（如有）",
    "precursor": "该梗的前身或来源词（如'打工人'来自'打工仔'）"
  }},
  "spread_phases": [
    {{
      "phase": "萌芽/爆发/泛化/固化/变异/消亡",
      "time_range": "该阶段的大致时间范围",
      "description": "该阶段的关键传播特征",
      "key_platforms": ["平台1", "平台2"],
      "key_figures": ["推动传播的关键人物/KOL/媒体"]
    }}
  ],
  "mutations": [
    {{
      "variant_name": "变异体名称",
      "time": "出现时间",
      "relationship": "与原梗的关系（衍生/反讽/泛化/收编）"
    }}
  ],
  "social_context": {{
    "triggers": ["触发该梗流行的社会情绪或事件"],
    "target_audience": "主要受众群体",
    "political_sensitivity": "是否有政治敏感性（无/低/中/高）",
    "backlash_events": ["舆论反转或批判事件（如有）"]
  }},
  "semantic_drift": {{
    "original_meaning": "原始含义",
    "current_meaning": "当前含义",
    "drift_direction": "含义漂移方向（如：从自嘲变为攻击、从具体变为泛指）"
  }},
  "narrative_quality": {{
    "information_richness": "高/中/低（该视频提供了多少传播细节）",
    "key_insight": "这个视频中关于该梗传播最有价值的一条信息（一句话）"
  }}
}}

注意:
1. ASR 转录可能有错别字，请根据上下文纠正
2. 如果某条信息视频中没有提到，填 null，不要编造
3. 只输出 JSON，不要有其他文字"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=3000,
        )
        raw = response.choices[0].message.content

        # 尝试解析 JSON（可能包裹在 ```json ``` 中）
        json_str = raw
        if "```json" in raw:
            json_str = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            json_str = raw.split("```")[1].split("```")[0]

        result = json.loads(json_str.strip())
        result["raw_llm_output"] = raw
        return result

    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw_llm_output": raw}
    except Exception as e:
        return {"error": str(e)}


def run_full_extraction(video_base: str = None):
    """批量抽取所有 22 个视频的传播叙事。

    Returns:
        list[dict]: 每个视频的抽取结果
    """
    if video_base is None:
        video_base = VIDEO_BASE

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_dirs = os.listdir(video_base)

    results = []
    for i, d in enumerate(video_dirs):
        video_path = os.path.join(video_base, d)
        rp = os.path.join(video_path, "report.json")
        if not os.path.exists(rp):
            continue

        print(f"\n[{i+1}/{len(video_dirs)}] {d[:60]}...")

        transcript_text, duration = load_transcript_text(video_path)
        if len(transcript_text) < 20:
            print(f"  [跳过] 转录太短 ({len(transcript_text)} chars)")
            continue

        print(f"  {len(transcript_text)} chars, {duration:.0f}s")

        result = extract_narrative(d, transcript_text)
        result["video_name"] = d
        result["transcript_chars"] = len(transcript_text)
        result["duration_sec"] = duration

        if "error" in result:
            print(f"  [错误] {result['error']}")
        else:
            name = result.get("meme_name", "?")
            richness = result.get("narrative_quality", {}).get("information_richness", "?")
            phases = len(result.get("spread_phases", []))
            mutations = len(result.get("mutations", []))
            print(f"  → {name} | 信息量: {richness} | {phases} phases | {mutations} mutations")

        results.append(result)

        # 逐个保存
        safe_name = d[:40].replace("/", "_").replace("\\", "_")
        out_path = OUTPUT_DIR / f"{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # API rate limit
        time.sleep(0.5)

    # 汇总保存
    summary_path = OUTPUT_DIR / "_all_narratives.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] {len(results)} 个视频的叙事已抽取")
    print(f"[输出] {OUTPUT_DIR}/")
    return results


def merge_narratives_to_curated(narratives: list[dict],
                                curated_path: str = None):
    """将抽取的叙事信息合并到 memes_2020_2025.json。

    对每个已策展热梗，用 LLM 抽取的 narrative 替换/补充原有 narrative 字段。
    """
    if curated_path is None:
        curated_path = CURATED_JSON

    with open(curated_path, "r", encoding="utf-8") as f:
        curated = json.load(f)

    # 建立 meme name → narrative 的映射
    # 用视频目录名中的梗名做模糊匹配
    narrative_map = {}
    for n in narratives:
        if "error" in n:
            continue
        meme_name = n.get("meme_name", "")
        video_name = n.get("video_name", "")
        narrative_map[video_name] = n
        if meme_name:
            narrative_map[meme_name] = n

    updated = 0
    for meme in curated["memes"]:
        name = meme["name"]
        # 尝试在 narrative_map 中匹配
        matched = None
        for key, nar in narrative_map.items():
            # 模糊匹配: 梗名出现在key中 或 key出现在梗名中
            if name in key or any(a in key for a in meme.get("aliases", [])):
                matched = nar
                break

        if matched:
            # 更新 narrative 字段
            old_narrative = meme.get("narrative", "")
            new_narrative = json.dumps(matched, ensure_ascii=False, indent=2)
            # 附加到原 narrative（不覆盖，保留人工策展内容）
            meme["narrative_extracted"] = matched
            meme["narrative_extracted_at"] = "2026-06-26"
            updated += 1

    curated["_meta"]["last_narrative_extraction"] = "2026-06-26"
    curated["_meta"]["narratives_extracted_from"] = "B站梗指南视频 x DeepSeek"

    out_path = Path(curated_path).parent / "memes_2020_2025_enriched.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(curated, f, ensure_ascii=False, indent=2)

    print(f"[合并] {updated}/{len(curated['memes'])} 个热梗已补充 LLM 抽取的传播叙事")
    print(f"[输出] {out_path}")

    return curated


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("MemeticChaos — LLM 传播叙事抽取")
    print("=" * 60)

    # Step 1: 批量抽取
    narratives = run_full_extraction()

    # Step 2: 合并到策展数据
    if narratives:
        merge_narratives_to_curated(narratives)

    print("\nDone.")

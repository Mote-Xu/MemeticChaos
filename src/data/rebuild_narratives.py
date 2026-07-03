"""
从修正后的视频转录报告重新生成叙事 + 概念打分

输入: Video_to_Text/outputs/2026-06-25/*/report.json (22个修正版)
输出: data/processed/narratives/ (更新) + data/processed/llm_concept_scores.json (更新)

用法:
    python src/data/rebuild_narratives.py              # 全部重新处理
    python src/data/rebuild_narratives.py --meme 内卷   # 单个
"""

import json, sys, os, time
from pathlib import Path
from datetime import datetime
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

# API
client = OpenAI(
    api_key="DEEPSEEK_API_KEY_REMOVED",
    base_url="https://api.deepseek.com",
)

ROOT = Path(__file__).parent.parent.parent
VIDEO_DIR = Path("E:/Desktop/Video_to_Text/outputs/2026-06-25")
NAR_DIR = ROOT / "data/processed/narratives"
SCORE_PATH = ROOT / "data/processed/llm_concept_scores.json"
NAR_DIR.mkdir(parents=True, exist_ok=True)


def load_report(folder_path: Path) -> dict:
    """加载修正后的报告."""
    rp = folder_path / "report.json"
    with open(rp, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_transcript(report: dict) -> str:
    """提取完整转录文本."""
    ts = report.get("transcript", [])
    if isinstance(ts, list):
        return " ".join(s.get("text", "") for s in ts)
    if isinstance(ts, dict):
        return ts.get("full_text", ts.get("text", ""))
    return str(ts)


def extract_meme_name(folder_name: str) -> str:
    """从文件夹名提取梗名."""
    # 格式: 【梗指南】打工人，打工魂，打工都是人上人！ - 1.打工人(Av...)
    import re
    # Try to find the meme name after "1." in the name
    match = re.search(r'1\.(.+?)\(Av', folder_name)
    if match:
        return match.group(1).strip()
    # Fallback: use the main title
    parts = folder_name.split(" - ")
    if parts:
        name = parts[0]
        # Clean up prefix
        name = re.sub(r'【.*?】', '', name).strip()
        return name[:30]
    return folder_name[:30]


def generate_narrative(report: dict) -> dict:
    """用修正后的转录 + 场景描述 生成叙事."""
    transcript = extract_transcript(report)
    meme_name = extract_meme_name(report["video"].get("file_name", ""))

    # Extract scene descriptions
    scenes = report.get("scene_descriptions", [])
    scene_text = "\n".join([
        f"[{s.get('timestamp_sec', 0):.0f}s] {s.get('summary', '')} "
        f"({s.get('setting', '')})"
        for s in scenes[:8]
    ]) if scenes else "无场景描述"

    prompt = f"""你是中国互联网模因分析师。请根据以下视频转录，提取该梗的结构化传播叙事。

## 梗名
{meme_name}

## 视频转录 (修正版)
{transcript[:3000]}

## 场景描述
{scene_text[:1000]}

## 分析要求
请输出 JSON，包含：
1. meme_name: 梗名
2. origin: {{trigger_event, platform, precursor}} — 起源信息
3. social_context: {{triggers, target_audience, political_sensitivity, backlash_events}} — 社会背景
4. spread_phases: [{{phase, time_range, description, key_figures}}] — 传播阶段
5. mutations: [{{name, time, relationship}}] — 变异
6. semantic_drift: {{original_meaning, current_meaning, drift_direction}} — 语义漂移
7. narrative_summary: 一句话总结
8. social_context_hint: 社会背景推断

只输出 JSON，不要其他文字。"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1000,
        )
        raw = resp.choices[0].message.content
        json_str = raw
        for delim in ["```json", "```"]:
            if delim in raw:
                json_str = raw.split(delim)[1].split("```")[0]
                break
        result = json.loads(json_str.strip())
        result["_source"] = "corrected_video_transcript"
        result["_regenerated_at"] = datetime.now().isoformat()
        return result
    except Exception as e:
        print(f"  LLM error: {e}")
        return {"error": str(e), "meme_name": meme_name}


def score_narrative(narrative: dict) -> dict:
    """对叙事进行概念打分."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from constraint.llm_concept_scorer import score_one_narrative, concept_to_constraint_llm

    result = score_one_narrative(
        narrative.get("meme_name", "unknown"),
        narrative,
        retry=1
    )
    if result and "scores" in result:
        constraint = concept_to_constraint_llm(result["scores"])
        return {
            "concept_scores": result["scores"],
            "constraint": constraint,
            "notes": result.get("scoring_notes", ""),
        }
    return {"error": "scoring_failed"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meme", type=str, default="", help="只处理指定梗")
    args = parser.parse_args()

    # Load existing scores
    existing_scores = {}
    if SCORE_PATH.exists():
        with open(SCORE_PATH, "r", encoding="utf-8") as f:
            existing_scores = json.load(f)

    folders = sorted(VIDEO_DIR.iterdir())
    print(f"处理 {len(folders)} 个视频转录")

    processed = 0
    new_scores = {}

    for folder in folders:
        if not folder.is_dir():
            continue

        meme_name = extract_meme_name(folder.name)

        if args.meme and args.meme not in meme_name:
            continue

        print(f"\n[{processed+1}/{len(folders)}] {meme_name}")

        # Load and generate
        report = load_report(folder)
        narrative = generate_narrative(report)

        if "error" in narrative:
            print(f"  ✗ 叙事生成失败: {narrative['error']}")
            continue

        # Save narrative
        safe_name = meme_name.replace("/", "_").replace(" ", "_")[:50]
        nar_path = NAR_DIR / f"{safe_name}.json"
        with open(nar_path, "w", encoding="utf-8") as f:
            json.dump(narrative, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 叙事 → {nar_path.name}")

        # Score concepts
        print(f"  概念打分...", end=" ", flush=True)
        scored = score_narrative(narrative)
        if "error" in scored:
            print(f"✗ {scored['error']}")
        else:
            top = sorted(scored["concept_scores"].items(), key=lambda x: -x[1])[:4]
            top_str = ", ".join([f"{n}={v:.2f}" for n, v in top])
            print(f"✓ [{top_str}]")
            new_scores[meme_name] = scored
            existing_scores[meme_name] = scored

        processed += 1

        # Rate limit
        if processed % 5 == 0:
            with open(SCORE_PATH, "w", encoding="utf-8") as f:
                json.dump(existing_scores, f, ensure_ascii=False, indent=2)
            print(f"  [已保存 {len(existing_scores)} 条概念分数]")

        time.sleep(0.3)

    # Final save
    if new_scores:
        with open(SCORE_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_scores, f, ensure_ascii=False, indent=2)
        print(f"\n概念分数已保存: {len(new_scores)} 条新增/更新")
        print(f"总计: {len(existing_scores)} 条")

    print(f"\n处理完成: {processed} 个视频")


if __name__ == "__main__":
    main()

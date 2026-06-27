"""
LLM 概念打分器 — 用 DeepSeek API 替换软匹配

对每条 narrative JSON，让 LLM 阅读叙事文本后对 35 个可观察概念逐一打分 (0-1)。
替代 broken 的 bigram Jaccard 软匹配。

输出:
- data/processed/llm_concept_scores.json — {meme_name: {concept_vec: [...], constraint: [...]}}
- 概念分数有真实方差 → 约束场有意义

用法:
    python src/constraint/llm_concept_scorer.py              # 批量打分全部
    python src/constraint/llm_concept_scorer.py --meme 后浪   # 单个测试
    python src/constraint/llm_concept_scorer.py --resume      # 断点续传
"""

import json, sys, os, time, re
from pathlib import Path
from typing import Optional
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

# API config
DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY_REMOVED"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

ROOT = Path(__file__).parent.parent.parent
NARRATIVE_DIRS = [
    ROOT / "data/processed/narratives",
    ROOT / "data/processed/narratives_from_trends",
]
OUTPUT_PATH = ROOT / "data/processed/llm_concept_scores.json"

# 35 可观察概念 + 定义
CONCEPT_DEFINITIONS = {
    "official_release": "是否来自官方发布（B站宣传片、政府报告、央视等主流媒体）",
    "grassroots": "是否草根自发产生（贴吧帖子、社区段子、网民自发创作）",
    "celebrity_trigger": "是否由明星/名人/网红触发",
    "accident_trigger": "是否偶然事件触发（意外走红、误传、乌龙）",
    "policy_trigger": "是否由政策/制度/法规变化触发",
    "platform_event": "是否平台活动触发（B站/微博/抖音官方活动）",
    "KOL_amplification": "是否有关键意见领袖（UP主/博主/大V）放大传播",
    "algorithm_push": "是否被算法推荐/推送/热搜机制推动",
    "cross_platform": "是否跨平台传播（微博→B站→知乎→抖音→小红书等）",
    "mainstream_media": "是否被主流媒体（央视/新闻/报纸）报道或介入",
    "brand_hijack": "是否被品牌/营销/企业蹭热点商业化",
    "class_conflict": "是否涉及阶层冲突（贫富/城乡/学历/打工vs人上人）",
    "gender_conflict": "是否涉及性别冲突（男女对立/普信等）",
    "generation_conflict": "是否涉及代际冲突（青年vs老一辈/后浪vs前浪）",
    "political_conflict": "是否涉及政治敏感/意识形态冲突/审查",
    "value_conflict": "是否涉及价值观冲突（传统vs现代/消费主义批判）",
    "parody": "是否有戏仿/恶搞/鬼畜/整活成分",
    "irony": "是否有反讽/解构/自嘲/黑色幽默成分",
    "semantic_drift": "是否发生语义漂移（原义→新义→泛化使用）",
    "remix": "是否有二次创作/模板化/段子改编/复刻",
    "institutionalization": "是否被制度化收编（官方主流化/日常化使用）",
    "anger": "是否包含愤怒/攻击/宣泄情绪",
    "humor_laugh": "是否以幽默/好笑/娱乐为主要调性",
    "schadenfreude": "是否包含幸灾乐祸/吃瓜/围观心态",
    "identity_belonging": "是否提供身份认同/归属感/群体共鸣",
    "hope": "是否包含希望/向往/美好/激励的正向情绪",
    "nihilism": "是否包含虚无/无力/躺平/退出/放弃情绪",
    "anxiety": "是否包含焦虑/压力/竞争/生存担忧",
    "nostalgia": "是否包含怀旧/回忆/过去的情绪",
    "youth_dominant": "是否以年轻人（学生/青年）为主要受众",
    "white_collar": "是否以白领/职场/打工人为主要受众",
    "student": "是否以学生群体为主要受众",
    "rural": "是否以下沉市场/农村/县城为主要受众",
    "elite": "是否以精英/知识分子/学术圈为主要受众",
}

CONCEPT_NAMES = list(CONCEPT_DEFINITIONS.keys())
N_CONCEPTS = len(CONCEPT_NAMES)


def load_all_narratives() -> dict:
    """加载全部叙事."""
    narratives = {}
    for nar_dir in NARRATIVE_DIRS:
        if not nar_dir.exists():
            continue
        for fp in nar_dir.glob("*.json"):
            if fp.name.startswith("_"):
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("meme_name", fp.stem)
                narratives[name] = data
            except Exception:
                pass
    return narratives


def build_scoring_prompt(narrative: dict) -> str:
    """构建概念打分 prompt."""
    # 简化叙事为 LLM 可读的文本
    nar_text = json.dumps(narrative, ensure_ascii=False, indent=2)
    # Truncate if too long (>8000 chars)
    if len(nar_text) > 6000:
        nar_text = nar_text[:6000] + "\n... (truncated)"

    concepts_list = "\n".join([
        f"{i+1}. {name}: {desc}"
        for i, (name, desc) in enumerate(CONCEPT_DEFINITIONS.items())
    ])

    prompt = f"""你是中国互联网模因分析专家。请阅读以下热梗的传播叙事，对每个可观察概念打分。

## 叙事数据
```json
{nar_text}
```

## 35 个可观察概念（每个打 0-1 分）

{concepts_list}

## 评分规则
- 1.0 = 该概念在此梗的传播中非常突出/核心
- 0.5 = 有一定体现但不是主要特征
- 0.0 = 完全不存在/不相关
- 请根据叙事文本的实际内容判断，不要猜测
- 如果叙事中没有提到相关信息，打 0 分

## 输出格式
严格输出 JSON，不要任何额外文字：
```json
{{
  "meme_name": "梗名",
  "scores": {{
    "official_release": 0.0,
    "grassroots": 0.8,
    ...（35个概念全部打分）
  }},
  "scoring_notes": "一句话说明打分依据"
}}
```"""

    return prompt


def parse_scores(response_text: str) -> Optional[dict]:
    """从 LLM 响应中提取概念分数."""
    # Try direct JSON parse
    for delim in ["```json", "```"]:
        if delim in response_text:
            parts = response_text.split(delim)
            if len(parts) >= 2:
                json_str = parts[1].split("```")[0]
                try:
                    return json.loads(json_str.strip())
                except json.JSONDecodeError:
                    continue

    # Try parsing whole response
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Fallback: regex extract scores
    scores = {}
    for name in CONCEPT_NAMES:
        pattern = rf'"{name}"\s*:\s*([\d.]+)'
        match = re.search(pattern, response_text)
        if match:
            scores[name] = float(match.group(1))
    if len(scores) >= 20:  # got most concepts
        return {"meme_name": "unknown", "scores": scores, "scoring_notes": "regex extracted"}
    return None


def score_one_narrative(name: str, narrative: dict, retry: int = 2) -> Optional[dict]:
    """对单条叙事打分."""
    prompt = build_scoring_prompt(narrative)

    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            raw = resp.choices[0].message.content
            result = parse_scores(raw)
            if result and "scores" in result and len(result["scores"]) >= 30:
                # Ensure all 35 concepts present
                for cn in CONCEPT_NAMES:
                    if cn not in result["scores"]:
                        result["scores"][cn] = 0.0
                return result
            if attempt < retry:
                time.sleep(1.0)
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            if attempt < retry:
                time.sleep(2.0)

    return None


def concept_to_constraint_llm(scores: dict) -> list[float]:
    """从 LLM 概念分数计算 5D 约束场 (直接聚合版).

    每组约束 = 相关概念的简单均值，不做加权抵消。
    这样不同类别的梗会产生明显不同的约束场 profile。
    """
    # 5 组约束定义：每组是相关的可观察概念
    CONSTRAINT_GROUPS = {
        "Identity": ["identity_belonging", "youth_dominant", "hope",
                     "official_release", "white_collar", "student"],
        "Humor/Decon": ["humor_laugh", "irony", "parody", "remix", "schadenfreude"],
        "Conflict": ["class_conflict", "gender_conflict", "generation_conflict",
                     "political_conflict", "value_conflict", "anger"],
        "Novelty": ["accident_trigger", "grassroots", "parody", "remix",
                    "semantic_drift", "celebrity_trigger"],
        "Accessibility": ["humor_laugh", "remix", "youth_dominant",
                          "cross_platform", "algorithm_push", "brand_hijack"],
    }

    constraint = []
    for group_name, concept_names in CONSTRAINT_GROUPS.items():
        vals = [scores.get(c, 0.0) for c in concept_names]
        # Mean of relevant concepts, scaled to [0.1, 0.9]
        raw = sum(vals) / len(vals)
        constraint.append(round(raw, 3))

    return constraint


def batch_score(resume: bool = True):
    """批量对所有叙事打分."""
    narratives = load_all_narratives()
    print(f"[LLM概念打分] {len(narratives)} 条叙事待处理")

    # Load existing scores if resuming
    existing = {}
    if resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  已有 {len(existing)} 条已打分，断点续传")

    results = dict(existing)
    total = len(narratives)
    done = len(existing)
    failed = 0

    for i, (name, narrative) in enumerate(narratives.items()):
        if name in existing:
            continue

        print(f"\n[{done+1}/{total}] {name}...", end=" ", flush=True)
        result = score_one_narrative(name, narrative)

        if result:
            scores = result["scores"]
            constraint = concept_to_constraint_llm(scores)
            results[name] = {
                "concept_scores": scores,
                "constraint": constraint,
                "notes": result.get("scoring_notes", ""),
            }
            # Show top concepts
            top = sorted(scores.items(), key=lambda x: -x[1])[:5]
            top_str = ", ".join([f"{n}={v:.2f}" for n, v in top])
            constraint_str = ", ".join([f"{v:.3f}" for v in constraint])
            print(f"✓ [{top_str}] → [{constraint_str}]")
            done += 1
        else:
            print("✗ FAILED")
            failed += 1
            # Save partial progress
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        # Save every 10 narratives
        if done % 10 == 0:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [已保存 {done} 条]")

        time.sleep(0.3)  # rate limit

    # Final save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary stats
    print(f"\n{'='*60}")
    print(f"LLM 概念打分完成")
    print(f"  成功: {done}/{total}")
    print(f"  失败: {failed}")
    if results:
        # Check constraint variance
        constraints = [v["constraint"] for v in results.values()]
        import numpy as np
        c_arr = np.array(constraints)
        labels = ["Identity", "Humor/Decon", "Conflict", "Novelty", "Accessibility"]
        print(f"\n  约束场统计 (LLM打分 vs 软匹配):")
        print(f"  {'维度':<16s} {'均值':>6s} {'std':>6s} {'min':>6s} {'max':>6s}")
        for i, label in enumerate(labels):
            print(f"  {label:<16s} {c_arr[:,i].mean():6.3f} {c_arr[:,i].std():6.3f} "
                  f"{c_arr[:,i].min():6.3f} {c_arr[:,i].max():6.3f}")
        print(f"\n  (对比软匹配: std ~0.01, 全部坍缩到 0.5)")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meme", type=str, default="", help="单个梗测试")
    parser.add_argument("--resume", action="store_true", default=True, help="断点续传")
    parser.add_argument("--no-resume", action="store_true", help="从头开始")
    args = parser.parse_args()

    if args.meme:
        narratives = load_all_narratives()
        if args.meme in narratives:
            print(f"测试: {args.meme}")
            result = score_one_narrative(args.meme, narratives[args.meme])
            if result:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                constraint = concept_to_constraint_llm(result["scores"])
                print(f"\n5D Constraint: {constraint}")
            else:
                print("FAILED")
        else:
            print(f"未找到: {args.meme}")
            print(f"可用的梗: {list(narratives.keys())[:20]}...")
    else:
        resume = not args.no_resume
        batch_score(resume=resume)

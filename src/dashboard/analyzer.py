"""
精细建模分析器 — 对特定话题做深度分析

功能:
- 约束场轨迹: 该梗在 5D 约束空间中的路径
- 关键转折点: 约束突变检测
- 同类比较: 与同类别其他梗的约束场对比
- 未来走向预测: 基于 Delta Transition Model
- LLM 深度解读: 用 DeepSeek 生成针对性分析
"""

import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT / "data/processed"
COLLECTOR_DIR = ROOT / "data/collector"

CATEGORY_NAMES = ["解构自嘲", "攻击发泄", "虚无退却", "身份认同", "纯粹娱乐"]
CONSTRAINT_LABELS = ["Identity", "Humor/Decon", "Conflict", "Novelty", "Accessibility"]

# Category defaults for constraint (from LLM scoring averages)
CAT_DEFAULTS = {
    "解构自嘲": [0.42, 0.55, 0.20, 0.48, 0.48],
    "攻击发泄": [0.38, 0.45, 0.58, 0.42, 0.42],
    "虚无退却": [0.45, 0.30, 0.42, 0.32, 0.32],
    "身份认同": [0.55, 0.35, 0.40, 0.38, 0.48],
    "纯粹娱乐": [0.38, 0.58, 0.08, 0.58, 0.62],
}


def _find_meme_data(topic: str) -> dict:
    """找到与 topic 相关的所有数据."""
    result = {
        "topic": topic,
        "found": False,
        "narrative": None,
        "constraint": None,
        "category": None,
        "trends_data": None,
        "trajectory": None,
        "similar_memes": [],
    }

    # 1. Search LLM concept scores
    scores_path = PROCESSED_DIR / "llm_concept_scores.json"
    if scores_path.exists():
        with open(scores_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        # Direct match
        if topic in scores:
            result["constraint"] = scores[topic].get("constraint")
            result["found"] = True
        # Fuzzy match
        else:
            for name in scores:
                if topic in name or name in topic:
                    result["constraint"] = scores[name].get("constraint")
                    result["found"] = True
                    topic = name  # use canonical name
                    break

    # 2. Search narratives
    for nar_dir in [PROCESSED_DIR / "narratives", PROCESSED_DIR / "narratives_from_trends"]:
        if not nar_dir.exists():
            continue
        for fp in nar_dir.glob("*.json"):
            if fp.name.startswith("_"):
                continue
            if topic.replace(" ", "") in fp.stem.replace(" ", ""):
                with open(fp, "r", encoding="utf-8") as f:
                    result["narrative"] = json.load(f)
                result["found"] = True
                break
        if result["narrative"]:
            break

    # 3. Category mapping
    from src.models.order_form_predictor import TREND_TO_MEME
    for trend_kw, (meme_name, cat) in TREND_TO_MEME.items():
        if topic in meme_name or topic in trend_kw:
            result["category"] = cat
            break

    # 4. Google Trends data
    trends_path = COLLECTOR_DIR / "google_trends_2015_2025.json"
    if trends_path.exists():
        with open(trends_path, "r", encoding="utf-8") as f:
            trends = json.load(f).get("memes", {})
        for kw, data in trends.items():
            if topic in kw or kw in topic:
                result["trends_data"] = {
                    "months": sorted(data.keys()),
                    "values": [data[m] for m in sorted(data.keys())],
                }
                break

    # 5. Trajectory
    traj_path = PROCESSED_DIR / "trajectories.json"
    if traj_path.exists():
        with open(traj_path, "r", encoding="utf-8") as f:
            trajectories = json.load(f).get("trajectories", [])
        for t in trajectories:
            if topic in t.get("name", "") or topic in t.get("meme_id", ""):
                result["trajectory"] = t
                break

    # 6. Similar memes (same category)
    if result["category"] and scores_path.exists():
        with open(scores_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        from src.models.order_form_predictor import TREND_TO_MEME
        for trend_kw, (meme_name, cat) in TREND_TO_MEME.items():
            if cat == result["category"] and topic not in meme_name:
                if meme_name in scores:
                    result["similar_memes"].append({
                        "name": meme_name,
                        "constraint": scores[meme_name].get("constraint"),
                        "category": cat,
                    })
        # Limit to top 5
        result["similar_memes"] = result["similar_memes"][:5]

    return result


def _detect_turning_points(constraint_traj: list, threshold: float = 0.15) -> list:
    """检测约束场突变点."""
    if len(constraint_traj) < 3:
        return []
    points = []
    for i in range(1, len(constraint_traj)):
        delta = np.linalg.norm(np.array(constraint_traj[i]) - np.array(constraint_traj[i-1]))
        if delta > threshold:
            points.append({
                "index": i,
                "delta": float(delta),
                "from": [float(x) for x in constraint_traj[i-1]],
                "to": [float(x) for x in constraint_traj[i]],
            })
    return points


def _llm_deep_analysis(topic: str, data: dict) -> str:
    """用 LLM 生成深度分析."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key="DEEPSEEK_API_KEY_REMOVED",
            base_url="https://api.deepseek.com",
        )

        prompt = f"""你是中国互联网模因分析师。请对热梗"{topic}"进行深度分析。

## 可用数据
- 类别: {data.get('category', '未知')}
- 约束场 (LLM打分): {data.get('constraint', '无')}
- 叙事: {json.dumps(data.get('narrative', {}), ensure_ascii=False, indent=2)[:2000]}
- Google Trends 数据点数: {len(data.get('trends_data', {}).get('months', []))}

## 分析维度
1. 该梗的核心传播机制是什么？
2. 社会约束场如何塑造了它的演化？
3. 关键转折点及原因
4. 与同类梗的异同
5. 未来走向预测

请输出 5-8 句话的深度分析，要具体、有洞察力。"""

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=800,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"LLM 分析不可用: {e}"


def analyze_topic(topic: str) -> dict:
    """对指定话题进行精细建模分析."""
    data = _find_meme_data(topic)

    result = {
        "topic": topic,
        "found": data["found"],
        "category": data["category"],
        "constraint_profile": {},
        "similar_memes": [],
        "trends_summary": {},
        "turning_points": [],
        "deep_analysis": "",
    }

    if not data["found"]:
        result["deep_analysis"] = f"未找到与「{topic}」相关的数据。请检查拼写或尝试其他话题。"
        return result

    # Constraint profile
    if data["constraint"]:
        result["constraint_profile"] = {
            CONSTRAINT_LABELS[i]: round(data["constraint"][i], 3)
            for i in range(len(data["constraint"]))
        }

    # Trends summary
    if data["trends_data"]:
        td = data["trends_data"]
        values = td["values"]
        months = td["months"]
        if values:
            result["trends_summary"] = {
                "n_months": len(values),
                "peak_value": max(values),
                "peak_month": months[values.index(max(values))],
                "recent_value": values[-1] if values else 0,
                "trend": "上升" if len(values) > 6 and sum(values[-3:]) > sum(values[-6:-3]) else "下降",
                "data": [{"month": m, "value": v} for m, v in zip(months, values)],
            }

    # Similar memes
    for sm in data["similar_memes"]:
        entry = {"name": sm["name"]}
        if sm.get("constraint"):
            entry["constraint"] = {
                CONSTRAINT_LABELS[i]: round(sm["constraint"][i], 3)
                for i in range(len(sm["constraint"]))
            }
        result["similar_memes"].append(entry)

    # Trajectory turning points
    if data["trajectory"]:
        nodes = data["trajectory"].get("nodes", [])
        if nodes:
            constraints = [
                n.get("constraint_state", {}).get("pressures", [0.5]*5)
                for n in nodes
            ]
            result["turning_points"] = _detect_turning_points(constraints)
            result["phases"] = [
                {"phase": n["phase"], "time": n.get("time_range", {}),
                 "narrative": n.get("narrative_state", {}).get("meaning", "")}
                for n in nodes
            ]

    # LLM deep analysis
    result["deep_analysis"] = _llm_deep_analysis(topic, data)

    return result

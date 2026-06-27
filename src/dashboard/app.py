"""
MemeticChaos Dashboard — Flask API + 静态前端

端点:
  GET  /                  — Dashboard HTML
  GET  /api/state         — 当前集体情感状态
  GET  /api/forecast      — 最新预测数据
  GET  /api/history       — 历史混沌轴 + 约束场时间序列
  GET  /api/memes         — 当前活跃梗列表
  GET  /api/narrative     — 最新 LLM 月度叙事摘要
  GET  /api/order_forms   — 秩序形态分布
  GET  /api/analyze?topic= — 精细建模: 特定话题深度分析
  GET  /api/hot_topics     — 5-10 个当前热度最高话题

启动:
  python src/dashboard/app.py --port 8931
"""

import json, sys, os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

from flask import Flask, request, jsonify, render_template

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT / "data/processed"
COLLECTOR_DIR = ROOT / "data/collector"

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
DASHBOARD_TOKEN = os.environ.get("DASHBOARD_TOKEN", "")

# ═══════════════════════════════════════
# Auth
# ═══════════════════════════════════════

@app.before_request
def check_auth():
    """如果设置了 DASHBOARD_TOKEN, 所有请求需要 token 验证."""
    if not DASHBOARD_TOKEN:
        return None  # 未设 token, 允许所有访问

    # Check cookie first, then query param
    token = request.cookies.get("mc_token") or request.args.get("token", "")
    if token == DASHBOARD_TOKEN:
        return None  # Valid

    # Return 401 for API, redirect for HTML
    if request.path.startswith("/api/"):
        return jsonify({"error": "unauthorized", "hint": "?token=YOUR_TOKEN"}), 401
    return "<h2>🔒 MemeticChaos</h2><p>需要访问令牌。</p><p>在 URL 后加 <code>?token=你的令牌</code></p>", 401


@app.after_request
def set_auth_cookie(response):
    """验证通过后设置 cookie, 后续请求不需要 ?token=."""
    if DASHBOARD_TOKEN and request.args.get("token") == DASHBOARD_TOKEN:
        response.set_cookie("mc_token", DASHBOARD_TOKEN, max_age=60*60*24*30, httponly=True)
    return response


# ═══════════════════════════════════════
# Data loaders (cached)
# ═══════════════════════════════════════

_cache = {}

CATEGORY_NAMES = ["解构自嘲", "攻击发泄", "虚无退却", "身份认同", "纯粹娱乐"]
CONSTRAINT_LABELS = ["Identity", "Humor/Decon", "Conflict", "Novelty", "Accessibility"]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_forecast_data() -> dict:
    """获取预测数据 — 优先读取 predictor 生成的 JSON，fallback 到实时生成."""
    if "forecast" in _cache:
        return _cache["forecast"]

    # Path 1: Clean JSON from predictor cron job
    state_path = PROCESSED_DIR / "dashboard_state.json"
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _cache["forecast"] = data
            return data
        except Exception:
            pass

    # Path 2: Live generation (expensive, ~20s)
    result = _generate_forecast_on_demand()
    if result and not result.get("error") and result.get("current_state"):
        _cache["forecast"] = result
        return result

    # Path 3: Empty fallback
    return {"current_state": {}, "forecasts": [], "order_forms": [],
            "model_perf": {}, "narrative_summary": ""}

    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Parse the text report (simple line parsing)
    lines = text.split("\n")
    in_forecast = False
    in_forms = False
    in_narrative = False
    current_fc = {}
    nar_lines = []

    for line in lines:
        line = line.strip()
        if "当前状态" in line:
            in_forecast = False
            in_forms = False
        elif "混沌轴:" in line and "当前" not in str(_cache.get("_last_state_line", "")):
            try:
                val = line.split(":")[1].strip().split("(")[0].strip()
                forecast["current_state"]["chaos_axis"] = float(val)
            except ValueError:
                pass
        elif "主导类别:" in line:
            forecast["current_state"]["dominant_category"] = line.split(":")[1].strip()
        elif "注意力集中度" in line:
            parts = line.split(":")
            if len(parts) > 1:
                forecast["current_state"]["attention_hhi"] = parts[1].strip()
        elif "类别熵" in line:
            parts = line.split(":")
            if len(parts) > 1:
                forecast["current_state"]["cat_entropy"] = parts[1].strip()
        elif "秩序形态聚类" in line:
            in_forms = True
            continue
        elif in_forms and line.startswith("Form"):
            parts = line.split(":")
            if len(parts) >= 2:
                forecast["order_forms"].append(parts[1].strip())
        elif line.startswith("──") or line.startswith("==="):
            in_forms = False
        elif "未来" in line and "个月预测" in line:
            in_forecast = True
            in_forms = False
            continue
        elif in_forecast and line and not line.startswith("─") and not line.startswith("="):
            if line[0].isdigit() and ":" in line:
                if current_fc:
                    forecast["forecasts"].append(current_fc)
                current_fc = {"month": line.strip(":")}
            elif "混沌轴:" in line:
                try:
                    current_fc["chaos_axis"] = float(line.split(":")[1].strip().split("(")[0])
                except ValueError:
                    pass
            elif "秩序形态:" in line:
                current_fc["order_form"] = line.split(":")[1].strip().split("(")[0].strip()
            elif "HHI:" in line:
                parts = line.split(":")
                try:
                    current_fc["hhi"] = float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            elif "主导类别:" in line and in_forecast:
                current_fc["dominant_cat"] = line.split(":")[1].strip().split("(")[0].strip()
        elif "LLM 月度集体叙事摘要" in line:
            in_forecast = False
            in_narrative = True
            continue
        elif in_narrative and line and not line.startswith("─") and not line.startswith("="):
            nar_lines.append(line)

    if current_fc:
        forecast["forecasts"].append(current_fc)
    if nar_lines:
        forecast["narrative_summary"] = " ".join(nar_lines)

    _cache["forecast"] = forecast
    return forecast


def _generate_forecast_on_demand() -> dict:
    """无报告文件时实时生成 (仅 fallback, 通常用 JSON 文件)."""
    return {"error": "live generation disabled, run predictor cron first",
            "current_state": {}, "forecasts": [], "order_forms": []}


# ═══════════════════════════════════════
# API Routes
# ═══════════════════════════════════════

@app.route("/")
def index():
    """Dashboard 主页."""
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    """当前集体情感状态."""
    forecast = get_forecast_data()
    state = forecast.get("current_state", {})

    # Augment with richer data from the predictor JSON if available
    state_path = PROCESSED_DIR / "dashboard_state.json"
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                richer = json.load(f)
            state.update({k: v for k, v in richer.items()
                         if k not in ("forecasts", "order_forms", "backtest", "generated_at")})
        except Exception:
            pass

    return jsonify(state)


@app.route("/api/forecast")
def api_forecast():
    """最新预测."""
    forecast = get_forecast_data()
    return jsonify({
        "forecasts": forecast.get("forecasts", []),
        "order_forms": forecast.get("order_forms", []),
        "model_perf": forecast.get("model_perf", {}),
    })


@app.route("/api/history")
def api_history():
    """历史混沌轴 + 约束场时间序列 (从预计算文件读取)."""
    history_path = PROCESSED_DIR / "dashboard_history.json"
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "no history data yet", "months": [], "chaos_axis": []})


@app.route("/api/memes")
def api_memes():
    """当前活跃梗列表 (从实时信号 + Trends)."""
    memes = []

    # From signal history
    signal_path = ROOT / "data/scraped/signal_history.jsonl"
    if not signal_path.exists():
        signal_path = COLLECTOR_DIR / "signal_history.jsonl"

    if signal_path.exists():
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(hours=168)).isoformat()
        meme_counts = defaultdict(int)
        meme_platforms = defaultdict(set)
        with open(signal_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sig = json.loads(line.strip())
                    if sig.get("timestamp", "") >= cutoff:
                        name = sig.get("meme_name", "?")
                        meme_counts[name] += 1
                        meme_platforms[name].add(sig.get("platform", "?"))
                except json.JSONDecodeError:
                    continue

        for name, count in sorted(meme_counts.items(), key=lambda x: -x[1])[:20]:
            memes.append({
                "name": name,
                "signal_count": count,
                "platforms": list(meme_platforms[name]),
            })

    # Augment with LLM concept scores if available
    scores_path = PROCESSED_DIR / "llm_concept_scores.json"
    if scores_path.exists():
        scores = _load_json(scores_path)
        for m in memes:
            if m["name"] in scores:
                m["constraint"] = scores[m["name"]].get("constraint", [])

    return jsonify({"memes": memes, "updated": datetime.now().isoformat()})


@app.route("/api/narrative")
def api_narrative():
    """最新 LLM 月度叙事摘要."""
    nar_path = PROCESSED_DIR / "monthly_narratives.jsonl"
    if not nar_path.exists():
        # Try falling back to the report
        forecast = get_forecast_data()
        return jsonify({
            "summary": forecast.get("narrative_summary", "暂无叙事摘要"),
            "month": datetime.now().strftime("%Y-%m"),
        })

    narratives = []
    with open(nar_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                narratives.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    if narratives:
        latest = narratives[-1]
        return jsonify(latest)
    return jsonify({"summary": "暂无叙事摘要", "month": datetime.now().strftime("%Y-%m")})


@app.route("/api/order_forms")
def api_order_forms():
    """秩序形态分布."""
    forecast = get_forecast_data()
    return jsonify({"order_forms": forecast.get("order_forms", [])})


@app.route("/api/analyze")
def api_analyze():
    """精细建模: 对特定话题做深度分析.

    ?topic=躺平 → 约束场轨迹 + 转折点 + 同类比较 + 预测
    """
    topic = request.args.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "请提供 ?topic= 参数"}), 400

    try:
        from analyzer import analyze_topic
        result = analyze_topic(topic)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "topic": topic}), 500


@app.route("/api/hot_topics")
def api_hot_topics():
    """5-10 个当前热度最高的话题."""
    # From recent signals
    signal_path = ROOT / "data/scraped/signal_history.jsonl"
    if not signal_path.exists():
        signal_path = COLLECTOR_DIR / "signal_history.jsonl"

    topics = []
    if signal_path.exists():
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(hours=168)).isoformat()
        meme_counts = defaultdict(int)
        with open(signal_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sig = json.loads(line.strip())
                    if sig.get("timestamp", "") >= cutoff:
                        meme_counts[sig.get("meme_name", "?")] += 1
                except json.JSONDecodeError:
                    continue
        topics = [{"name": n, "count": c} for n, c in sorted(meme_counts.items(), key=lambda x: -x[1])[:10]]

    # Fallback: Google Trends top
    if not topics:
        trends = _load_json(COLLECTOR_DIR / "google_trends_2015_2025.json")
        last_month = "2025-12"
        memes = trends.get("memes", {})
        month_vals = [(n, d.get(last_month, 0)) for n, d in memes.items()]
        month_vals.sort(key=lambda x: -x[1])
        topics = [{"name": n, "attention": v} for n, v in month_vals[:10] if v > 0]

    return jsonify({"topics": topics})


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8931)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"MemeticChaos Dashboard")
    print(f"  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

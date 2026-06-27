"""
模因分析器 — 输入一个梗名，输出完整的定量分析报告

这是 MemeticChaos 的第一个实际产品。不是抓取器，不是框架，是一个可用的工具。

用法:
    python src/meme_inspector.py 打工人
    python src/meme_inspector.py 后浪 --compare 躺平
    python src/meme_inspector.py --list
    python src/meme_inspector.py --top conflict
"""

import json
import sys
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")


def load_all():
    """加载所有数据。"""
    with open(DATA_DIR / "curated/memes_2020_2025.json", "r", encoding="utf-8") as f:
        curated = json.load(f)
    with open(DATA_DIR / "processed/trajectories.json", "r", encoding="utf-8") as f:
        traj_data = json.load(f)
    trends_path = DATA_DIR / "collector/google_trends_2015_2025.json"
    trends = {}
    if trends_path.exists():
        with open(trends_path, "r", encoding="utf-8") as f:
            trends = json.load(f).get("memes", {})
    return curated, traj_data["trajectories"], trends


def find_meme(name: str, curated: dict, trajectories: list, trends: dict) -> dict:
    """查找一个梗的完整数据。"""
    # Search curated
    meme = None
    for m in curated["memes"]:
        if name in m["name"] or name in m.get("aliases", []):
            meme = m
            break

    # Search trajectory
    traj = None
    for t in trajectories:
        if meme and t["name"] == meme["name"]:
            traj = t
            break

    # Search trends
    trend_map = {
        "打工人": "打工人", "内卷 / 卷": "内卷", "躺平": "躺平",
        "普信男": "普信男", "小镇做题家": "小镇做题家",
        "摆烂": "摆烂", "润": "润", "吗喽": "吗喽",
        "鼠鼠": "鼠鼠", "牛马": "牛马", "i人/e人": "i人 e人",
        "遥遥领先": "遥遥领先 华为", "孔乙己的长衫": "孔乙己的长衫",
        "精神状态良好": "精神状态", "XX刺客": "雪糕刺客",
        "谢帝我要迪士尼": "谢帝 迪士尼", "科目三": "科目三",
        "尊嘟假嘟": "尊嘟假嘟", "鸡你太美": "鸡你太美",
        "后浪": "后浪", "情绪价值": "情绪价值",
        "四不/不婚不育": "不婚不育", "显眼包": "显眼包",
        "泼天富贵/泼天流量": "泼天富贵", "服美役": "服美役",
        "建议专家不要建议": "建议专家不要建议",
        "命运的齿轮开始转动": "命运的齿轮",
        "原生家庭": "原生家庭", "发疯文学": "发疯文学",
    }
    trend_name = trend_map.get(meme["name"] if meme else name, name)
    trend_data = trends.get(trend_name, {})

    return {"meme": meme, "trajectory": traj, "trends": trend_data, "trend_name": trend_name}


def print_report(name: str):
    """打印一个梗的完整分析报告。"""
    curated, trajectories, trends = load_all()
    result = find_meme(name, curated, trajectories, trends)

    if result["meme"] is None:
        print(f"未找到梗: {name}")
        print(f"已知梗: {', '.join(m['name'] for m in curated['memes'])}")
        return

    m = result["meme"]
    t = result["trajectory"]
    td = result["trends"]

    # ═══ 基本信息 ═══
    print(f"\n{'='*70}")
    print(f"  {m['name']}  [{m['category']}]  {m['year']}-{m['peak_year']}")
    print(f"{'='*70}")

    # ═══ 真实注意力数据 ═══
    if td:
        values = list(td.values())
        months = list(td.keys())
        if values:
            peak_val = max(values)
            peak_month = months[values.index(peak_val)]
            recent = [(m, v) for m, v in td.items() if m >= "2024-01"]
            trend_status = "↑ 上升中" if recent and recent[-1][1] > np.mean(values) else \
                           "↓ 下降中" if recent and recent[-1][1] < np.mean(values) * 0.3 else \
                           "→ 稳定"
            print(f"\n  📈 真实注意力 (Google Trends 2015-2025):")
            print(f"     峰值: {peak_month} ({peak_val:.0f})")
            print(f"     趋势: {trend_status}")
            print(f"     数据点: {len(values)} 个月")
            # Sparkline
            if len(values) > 5:
                norm = np.array(values) / max(1, max(values))
                chars = "▁▂▃▄▅▆▇█"
                spark = "".join(chars[min(7, int(v * 8))] for v in norm[::max(1, len(norm)//40)])
                print(f"     曲线: {spark}")
    else:
        print(f"\n  📈 真实注意力: 暂无数据")

    # ═══ 轨迹 ═══
    if t:
        print(f"\n  🛤️  状态空间轨迹 ({len(t['nodes'])} 阶段):")
        print(f"     {'阶段':<12s} {'R₀':>6s} {'Chaos':>7s} {'冲突':>6s} {'身份':>6s} {'主导约束':<12s}")
        print(f"     {'-'*55}")
        for n in t["nodes"]:
            ds = n["dynamic_state"]
            cs = n["constraint_state"]
            dom = cs.get("dominant_constraint", "")
            print(f"     {n['phase']:<12s} {ds.get('R0',0):>6.2f} {ds.get('chaos_axis',0):>+7.2f} "
                  f"{cs['pressures'][2]:>6.2f} {cs['pressures'][0]:>6.2f} {dom:<12s}")

    # ═══ 约束场 ═══
    if t:
        cs = t["nodes"][1]["constraint_state"] if len(t["nodes"]) > 1 else t["nodes"][0]["constraint_state"]
        p = cs["pressures"]
        labels = ["身份认同", "幽默解构", "冲突对抗", "新奇度", "传播易得"]
        print(f"\n  🔬 约束场剖面:")
        for i, (label, val) in enumerate(zip(labels, p)):
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"     {label:<10s} [{bar}] {val:.2f}")

    # ═══ 变异体 ═══
    variants = m.get("mutation_variants", [])
    if variants:
        print(f"\n  🧬 变异体 ({len(variants)}):")
        for v in variants[:5]:
            print(f"     • {v}")

    # ═══ 分类位置 ═══
    cat = m.get("category", "")
    chaos = m.get("chaos_position", 0)
    lc = m.get("lifecycle", {})
    real_peak = lc.get("real_peak_value", "?")

    print(f"\n  📊 汇总:")
    print(f"     类别: {cat}")
    print(f"     混沌位置: {chaos:+.2f} ({'偏秩序' if chaos > 0 else '偏混沌'})")
    print(f"     真实峰值: {real_peak}")
    print(f"     实际持续: {lc.get('duration_months', '?')} 个月")
    print(f"     数据来源: {lc.get('data_source', '人工策展')}")

    # ═══ 系统判断 ═══
    print(f"\n  🤖 系统判断:")
    if t and len(t["nodes"]) >= 3:
        final_r0 = t["nodes"][-1]["dynamic_state"].get("R0", 0)
        final_conflict = t["nodes"][-1]["constraint_state"]["pressures"][2]
        if final_r0 < 0.5:
            fate = "已消亡或接近消亡"
        elif final_conflict > 0.6:
            fate = "收敛到 Viral Chaos 盆地 (攻击/虚无)"
        elif final_conflict < 0.4:
            fate = "收敛到 Viral Order 盆地 (身份/娱乐)"
        else:
            fate = "处于混合态 (未收敛到明确盆地)"
        print(f"     当前状态: {fate}")
        if td and len(td) > 12:
            recent_vals = [v for k, v in sorted(td.items()) if k >= "2024-06"]
            if recent_vals and np.mean(recent_vals) > 0:
                print(f"     活跃度: 仍在产生注意力")
            else:
                print(f"     活跃度: 已进入历史档案")
    else:
        print(f"     数据不足以做出判断")

    print(f"\n{'='*70}\n")


def list_memes(curated: dict):
    """列出所有已知梗。"""
    print(f"\n{'='*50}")
    print(f"  已知模因 ({len(curated['memes'])} 个)")
    print(f"{'='*50}")
    for m in curated["memes"]:
        lc = m.get("lifecycle", {})
        real_peak = lc.get("real_peak_value", "")
        peak_str = f"(峰值 {real_peak:.0f})" if real_peak != "" else ""
        print(f"  {m['name']:<16s} [{m['category']:<8s}] {m['year']}-{m['peak_year']} {peak_str}")


def compare_two(name1: str, name2: str):
    """对比两个梗。"""
    curated, trajectories, trends = load_all()
    r1 = find_meme(name1, curated, trajectories, trends)
    r2 = find_meme(name2, curated, trajectories, trends)

    print(f"\n{'='*70}")
    print(f"  对比: {name1} vs {name2}")
    print(f"{'='*70}")

    for label, r in [(name1, r1), (name2, r2)]:
        if r["meme"] is None:
            print(f"  {label}: 未找到")
            continue
        t = r["trajectory"]
        if t and len(t["nodes"]) >= 2:
            n = t["nodes"][1]
            cs = n["constraint_state"]
            ds = n["dynamic_state"]
            td = r["trends"]
            peak = max(td.values()) if td else "?"
            print(f"\n  {label}:")
            print(f"    类别: {r['meme']['category']}")
            print(f"    冲突: {cs['pressures'][2]:.2f}  身份: {cs['pressures'][0]:.2f}  "
                  f"幽默: {cs['pressures'][1]:.2f}  R₀: {ds.get('R0',0):.2f}")
            print(f"    混沌: {ds.get('chaos_axis',0):+.2f}  峰值关注: {peak}")
            if r["trends"]:
                norm = np.array(list(r["trends"].values()))
                if norm.max() > 0:
                    norm = norm / norm.max()
                    chars = "▁▂▃▄▅▆▇█"
                    spark = "".join(chars[min(7, int(v*8))] for v in norm[::max(1, len(norm)//40)])
                    print(f"    趋势: {spark}")
    print(f"\n{'='*70}\n")


def rank_by(metric: str):
    """按指标排序所有梗。"""
    curated, trajectories, trends = load_all()
    scores = []

    for m in curated["memes"]:
        for t in trajectories:
            if t["name"] == m["name"] and len(t["nodes"]) >= 2:
                n = t["nodes"][1]
                cs = n["constraint_state"]
                p = cs["pressures"]
                score = {
                    "conflict": p[2],
                    "identity": p[0],
                    "humor": p[1],
                    "novelty": p[3],
                    "r0": n["dynamic_state"].get("R0", 0),
                    "chaos": abs(n["dynamic_state"].get("chaos_axis", 0)),
                }.get(metric, 0)
                scores.append((m["name"], m["category"], score))
                break

    scores.sort(key=lambda x: -x[2])
    labels = {"conflict": "冲突强度", "identity": "身份共鸣", "humor": "幽默解构",
              "novelty": "新奇度", "r0": "传播力 R₀", "chaos": "混沌强度"}

    print(f"\n{'='*50}")
    print(f"  按 {labels.get(metric, metric)} 排序")
    print(f"{'='*50}")
    for name, cat, score in scores:
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {name:<14s} [{cat:<8s}] [{bar}] {score:.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("用法:")
        print("  python src/meme_inspector.py 打工人           # 分析单个梗")
        print("  python src/meme_inspector.py 后浪 --vs 躺平   # 对比两个梗")
        print("  python src/meme_inspector.py --list           # 列出所有梗")
        print("  python src/meme_inspector.py --top conflict   # 按指标排名")
        sys.exit(0)

    if sys.argv[1] == "--list":
        curated, _, _ = load_all()
        list_memes(curated)
    elif sys.argv[1] == "--top" and len(sys.argv) > 2:
        rank_by(sys.argv[2])
    elif "--vs" in sys.argv:
        idx = sys.argv.index("--vs")
        name1 = sys.argv[1]
        name2 = sys.argv[idx + 1]
        compare_two(name1, name2)
    else:
        print_report(sys.argv[1])

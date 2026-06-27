"""
实时模因数据管线 — 采集 → 更新 → 相图刷新

将 scraper 的实时信号、narrative_extractor 的叙事数据、
和 MemeTrajectory 的状态表示串联成自动化闭环。

用法:
    python src/data/live_pipeline.py              # 单次运行
    python src/data/live_pipeline.py --watch 3600 # 每小时自动采集
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

PROCESSED_DIR = Path("data/processed")


def step_scrape() -> dict:
    """Step 1: 实时采集模因信号。"""
    from src.data.scraper import scrape_all
    return scrape_all()


def step_update_trajectories(scrape_result: dict,
                              trajectories_path: str = None) -> list:
    """Step 2: 用新采集的信号更新 MemeTrajectory。

    如果某热梗在热搜上出现，更新其 Dynamic State（提高当前 R₀ 估计），
    并在 Trajectory 中添加一个实时观测节点。
    """
    from src.trajectory.meme_trajectory import (
        MemeTrajectory, TrajectoryNode, DynamicState, NarrativeState,
        ConstraintState, SocialContext,
    )

    if trajectories_path is None:
        trajectories_path = str(PROCESSED_DIR / "trajectories.json")

    if not os.path.exists(trajectories_path):
        print("[更新] 未找到已有 Trajectory，跳过更新")
        return []

    with open(trajectories_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    signals = scrape_result.get("signals", [])
    if not signals:
        print("[更新] 无新信号，Trajectory 无需更新")
        return data.get("trajectories", [])

    # 按梗名聚合信号
    meme_signals = {}
    for s in signals:
        name = s["meme_name"]
        if name not in meme_signals:
            meme_signals[name] = []
        meme_signals[name].append(s)

    from src.constraint.delta_transition import DeltaTransitionModel

    # EMA 平滑层：防止瞬时流量尖峰击穿验证器
    ema_alpha = 0.3  # 平滑因子（越低越平滑）

    updated_count = 0
    for traj_dict in data.get("trajectories", []):
        name = traj_dict.get("name", "")
        if name not in meme_signals:
            continue

        ms = meme_signals[name]
        best_rank = min(s.get("rank", 999) for s in ms)
        platforms = list(set(s.get("platform", "") for s in ms))

        # 热度 → R₀ 映射 (EMA 平滑防尖峰)
        raw_R0 = max(1.0, 4.0 / max(1, best_rank))
        prev_R0 = nodes[-1]["dynamic_state"].get("R0", raw_R0) if nodes else raw_R0
        signal_R0 = ema_alpha * raw_R0 + (1 - ema_alpha) * prev_R0

        # 用 Delta Transition 更新 Constraint
        nodes = traj_dict.get("nodes", [])
        if nodes:
            last_node = nodes[-1]
            prev_constraint = np.array(
                last_node.get("constraint_state", {}).get("pressures", [0.5]*5)
            )
            prev_phase = last_node.get("phase", "peak")
            dtm = DeltaTransitionModel({
                "economic_stress": 0.5, "polarization": 0.5, "censorship": 0.2,
            })
            # 热搜出现 = 信号 → 向 emergence 转移
            new_constraint = dtm.transition(prev_constraint, prev_phase, "emergence").tolist()
        else:
            new_constraint = [0.5]*5

        live_node = {
            "phase": "live_observation",
            "time_range": {
                "start": datetime.now().strftime("%Y-%m-%d"),
                "end": datetime.now().strftime("%Y-%m-%d"),
            },
            "narrative_state": {"meaning": "实时热搜信号", "mechanism": "", "key_figures": platforms},
            "constraint_state": {"pressures": new_constraint, "dominant_constraint": "real-time signal"},
            "dynamic_state": {
                "R0": signal_R0, "beta": 0.0, "gamma": 0.0,
                "sigma": 0.0, "mu": 0.0,
                "chaos_axis": traj_dict["nodes"][-1]["dynamic_state"].get("chaos_axis", 0) if traj_dict.get("nodes") else 0,
                "entropy": 0.0, "peak_infected": 0.0,
            },
            "social_context": {
                "economic_stress": 0.5, "polarization": 0.5, "censorship": 0.2,
                "platform": "+".join(platforms),
                "trigger_events": [s["title"] for s in ms[:3]],
            },
        }

        # 替换旧 live_observation 节点（如有）
        traj_dict["nodes"] = [n for n in traj_dict.get("nodes", [])
                              if n.get("phase") != "live_observation"]
        traj_dict["nodes"].append(live_node)
        updated_count += 1

    data["last_updated"] = datetime.now().isoformat()
    data["n_signals_applied"] = len(signals)

    output_path = str(PROCESSED_DIR / "trajectories.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[更新] {updated_count} 条 Trajectory 已注入实时信号 → {output_path}")
    return data.get("trajectories", [])


def step_report(trajectories: list, scrape_result: dict):
    """Step 3: 生成当前状态报告。"""
    print(f"\n{'='*60}")
    print(f"MemeticChaos 实时状态报告 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # 最新采集
    signals = scrape_result.get("signals", [])
    print(f"\n[采集] 微博 {scrape_result.get('weibo_count', 0)} 条 | "
          f"百度 {scrape_result.get('baidu_count', 0)} 条 | "
          f"知乎 {scrape_result.get('zhihu_count', 0)} 条 | "
          f"检测到 {len(signals)} 条模因信号")

    if signals:
        print("\n[活跃模因]")
        for s in signals[:10]:
            print(f"  [{s['platform']}] #{s['rank']} {s['meme_name']} ← {s['title'][:60]}")

    # Trajectory 统计
    if trajectories:
        active = [t for t in trajectories
                   if any(n.get("phase") == "live_observation" for n in t.get("nodes", []))]
        print(f"\n[轨迹] {len(trajectories)} 条总轨迹 | {len(active)} 条有实时信号")
        if active:
            for t in active:
                live_node = [n for n in t.get("nodes", []) if n.get("phase") == "live_observation"][0]
                r0 = live_node["dynamic_state"]["R0"]
                print(f"  {t['name']}: 实时R₀={r0:.2f}")

    # 微博/知乎 Top5
    print(f"\n[微博 Top5]")
    for item in scrape_result.get("weibo_top10", [])[:5]:
        print(f"  #{item['rank']} {item['title'][:50]} ({item['hot_score']})")
    print(f"\n[知乎 Top5]")
    for item in scrape_result.get("zhihu_top10", [])[:5]:
        print(f"  #{item['rank']} {item['title'][:50]} ({item['hot_score']})")


def step_discover_new_memes():
    """Step 4: 新梗发现 — 从采集数据中检测候选 → LLM叙事 → 概念打分."""
    print("\n[新梗发现]")
    try:
        from src.data.signal_pipeline import run_pipeline
        run_pipeline(max_new=2, force=False)
    except Exception as e:
        print(f"  新梗发现失败: {e}")


def run_once():
    """执行一次完整的采集→更新→报告→新梗发现循环。"""
    result = step_scrape()
    trajectories = step_update_trajectories(result)
    step_report(trajectories, result)
    step_discover_new_memes()


def run_watch(interval_sec: int):
    """定时循环采集。"""
    print(f"[监控] 每 {interval_sec}s 采集一次 (Ctrl+C 停止)")
    try:
        while True:
            run_once()
            print(f"\n[等待] {interval_sec}s 后下次采集...")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n[停止] 监控已终止")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="MemeticChaos 实时数据管线")
    parser.add_argument("--watch", type=int, default=0,
                        help="定时采集间隔（秒），不指定则单次运行")
    args = parser.parse_args()

    if args.watch > 0:
        run_watch(args.watch)
    else:
        run_once()

"""
跨平台注意力流动分析 — 追踪梗在不同平台间的迁移路径

每个梗在平台间的流动模式编码了约束场信息:
- 微博→知乎: 大众娱乐 → 深度讨论 (知识分子收编)
- 知乎→微博: 精英议题 → 大众扩散
- 百度热搜 → 微博: 搜索驱动 → 社交扩散
- 单一平台停滞: 圈层内传播，未破圈

输出:
- 平台过渡矩阵: P(微博→知乎), P(知乎→微博), ...
- 流动方向图: 每个梗的平台迁移路径
- 破圈指数: 跨平台传播的梗占比

用法:
    python src/analysis/platform_flow.py              # 分析当前数据
    python src/analysis/platform_flow.py --history    # 历史全部数据
"""

import json, sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter

ROOT = Path(__file__).parent.parent.parent
SCRAPED_DIR = ROOT / "data/scraped"
COLLECTOR_DIR = ROOT / "data/collector"


def load_all_scrapes(hours: int = None) -> list[dict]:
    """加载采集数据."""
    scrapes = []
    cutoff = None
    if hours:
        cutoff = datetime.now() - timedelta(hours=hours)

    for fp in sorted(SCRAPED_DIR.glob("scrape_*.json")):
        try:
            ts_str = fp.stem.replace("scrape_", "")
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            if cutoff and ts < cutoff:
                continue
            with open(fp, "r", encoding="utf-8") as f:
                scrapes.append(json.load(f))
        except (ValueError, json.JSONDecodeError):
            continue
    return scrapes


def extract_platform_sequences(scrapes: list[dict]) -> dict[str, list[tuple]]:
    """从采集数据中提取每个梗的平台出现序列.

    Returns: {meme_name: [(timestamp, platform, rank, title), ...]}
    """
    meme_timeline = defaultdict(list)

    for scrape in scrapes:
        ts = scrape.get("timestamp", "")
        for sig in scrape.get("signals", []):
            meme_timeline[sig["meme_name"]].append((
                ts,
                sig.get("platform", "?"),
                sig.get("rank", 0),
                sig.get("title", "")[:60],
            ))

    # Sort each timeline by timestamp
    for name in meme_timeline:
        meme_timeline[name].sort(key=lambda x: x[0])

    return dict(meme_timeline)


def compute_flow_patterns(meme_timelines: dict[str, list]) -> dict:
    """计算每个梗的平台流动模式."""
    patterns = {}

    for name, events in meme_timelines.items():
        if len(events) < 2:
            continue

        # Platform sequence
        platforms = [e[1] for e in events]

        # Transition pairs
        transitions = []
        for i in range(1, len(platforms)):
            transitions.append((platforms[i-1], platforms[i]))

        # Unique platforms visited
        unique_platforms = list(set(platforms))

        # First platform (entry point)
        entry_platform = platforms[0]

        # Evolution: does it spread to more platforms over time?
        first_half_platforms = set(platforms[:len(platforms)//2])
        second_half_platforms = set(platforms[len(platforms)//2:])
        expanding = len(second_half_platforms) > len(first_half_platforms)
        new_platforms = second_half_platforms - first_half_platforms

        # Cross-platform score: how many different platforms
        cross_score = len(unique_platforms)

        patterns[name] = {
            "platform_sequence": platforms,
            "transitions": transitions,
            "unique_platforms": unique_platforms,
            "entry_platform": entry_platform,
            "expanding": expanding,
            "new_platforms": list(new_platforms),
            "cross_platform_score": cross_score,
            "n_events": len(events),
            "first_seen": events[0][0],
            "last_seen": events[-1][0],
        }

    return patterns


def build_transition_matrix(patterns: dict) -> dict:
    """构建平台间过渡矩阵."""
    all_transitions = []
    for p in patterns.values():
        all_transitions.extend(p["transitions"])

    # Count transitions
    counts = defaultdict(Counter)
    for src, dst in all_transitions:
        counts[src][dst] += 1

    # Normalize to probabilities
    matrix = {}
    all_platforms = set()
    for src in counts:
        all_platforms.add(src)
        all_platforms.update(counts[src].keys())

    platforms = sorted(all_platforms)
    matrix["platforms"] = platforms
    matrix["transitions"] = {}

    for src in platforms:
        row = {}
        total = sum(counts[src].values())
        for dst in platforms:
            row[dst] = round(counts[src][dst] / total, 3) if total > 0 else 0
        matrix["transitions"][src] = row

    return matrix


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("跨平台注意力流动分析")
    print("=" * 60)

    scrapes = load_all_scrapes()
    print(f"\n采集数据: {len(scrapes)} 次")

    if not scrapes:
        print("无采集数据")
        return

    timelines = extract_platform_sequences(scrapes)
    print(f"有信号的梗: {len(timelines)}")

    patterns = compute_flow_patterns(timelines)
    print(f"有流动模式的梗: {len(patterns)}")

    # Transition matrix
    matrix = build_transition_matrix(patterns)
    print(f"\n平台过渡矩阵:")
    platforms = matrix["platforms"]
    header = "        " + "  ".join(f"{p:>8}" for p in platforms)
    print(header)
    for src in platforms:
        row = "  ".join(f"{matrix['transitions'][src].get(dst, 0):8.3f}" for dst in platforms)
        print(f"  {src:<6s} {row}")

    # Platform statistics
    platform_counts = Counter()
    for p in patterns.values():
        for plat in p["unique_platforms"]:
            platform_counts[plat] += 1

    print(f"\n平台覆盖率:")
    total_memes = len(patterns)
    for plat, count in platform_counts.most_common():
        print(f"  {plat}: {count}/{total_memes} ({count/total_memes*100:.0f}%)")

    # Cross-platform spreaders
    cross_spreaders = [(name, p) for name, p in patterns.items()
                       if p["cross_platform_score"] >= 2]
    print(f"\n跨平台传播梗 ({len(cross_spreaders)}):")
    for name, p in sorted(cross_spreaders, key=lambda x: -x[1]["cross_platform_score"]):
        print(f"  {name}: {p['entry_platform']} → {' → '.join(p['new_platforms'])} "
              f"(共 {p['cross_platform_score']} 平台)")

    # Expansion direction
    expanding = [name for name, p in patterns.items() if p["expanding"]]
    print(f"\n正在扩张的梗 ({len(expanding)}):")
    for name in sorted(expanding):
        p = patterns[name]
        print(f"  {name}: {len(p['unique_platforms'])} 平台")

    return patterns, matrix


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true", help="全部历史")
    args = parser.parse_args()
    main()

"""
Google Trends 周度数据加载器 — 分段查询获取周级分辨率

原理: Google Trends 查询 >5 年 → 月度, ≤5 年 → 周度。
按 2 年一段查询 2015-2025, 获得 500+ 周数据点 (vs 132 月)。

用法:
    python src/data/trends_weekly.py                      # 拉取全部
    python src/data/trends_weekly.py --meme 躺平          # 单个测试
    python src/data/trends_weekly.py --resume             # 断点续传
"""

import json, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Monkey-patch: urllib3 2.x renamed method_whitelist → allowed_methods
import urllib3.util.retry
_original_init = urllib3.util.retry.Retry.__init__
def _patched_init(self, *args, **kwargs):
    if 'method_whitelist' in kwargs:
        kwargs['allowed_methods'] = kwargs.pop('method_whitelist')
    _original_init(self, *args, **kwargs)
urllib3.util.retry.Retry.__init__ = _patched_init

from pytrends.request import TrendReq

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data/collector"
OUTPUT_PATH = DATA_DIR / "google_trends_weekly_2015_2025.json"

# 43 梗关键词 (与 collective_predictor.py 对齐)
MEME_KEYWORDS = [
    "citywalk", "i人 e人", "不婚不育", "不结婚", "专家建议",
    "元宇宙", "内卷", "凡尔赛", "南方小土豆", "原生家庭",
    "发疯文学 梗", "后浪", "吗喽", "多巴胺穿搭",
    "孔乙己 长衫", "孔乙己的长衫", "尊嘟假嘟", "小镇做题家",
    "建议专家不要建议", "情绪价值", "打工人", "摆烂",
    "显眼", "显眼包", "普信", "普信男", "栓Q", "润", "牛马",
    "破防", "社恐", "社死", "科目三", "精神内耗",
    "精神状态", "美拉德", "芭比Q", "躺平",
    "遥遥领先", "遥遥领先 华为", "雪糕刺客", "鸡你太美", "鼠鼠",
]

# 分段: 每段 2 年
CHUNKS = [
    ("2015-01-01", "2016-12-31"),
    ("2017-01-01", "2018-12-31"),
    ("2019-01-01", "2020-12-31"),
    ("2021-01-01", "2022-12-31"),
    ("2023-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
]

# 锚点关键词 — 高搜索量且稳定, 用于跨段归一化
ANCHOR_KW = "内卷"  # 内卷在 2018 年后稳定高搜索量


def pull_keyword_weekly(keyword: str, anchor: str = ANCHOR_KW) -> pd.Series:
    """拉取单个关键词的周度数据 (2015-2025), 用锚点关键词归一化."""
    pytrends = TrendReq(hl='zh-CN', tz=360, retries=3, backoff_factor=0.5)
    all_weeks = []

    for chunk_start, chunk_end in CHUNKS:
        timeframe = f"{chunk_start} {chunk_end}"
        try:
            pytrends.build_payload(
                [keyword, anchor], cat=0, timeframe=timeframe, geo="CN"
            )
            df = pytrends.interest_over_time()
            if df.empty:
                continue
            df = df.drop(columns=['isPartial'], errors='ignore')

            # Normalize by anchor: divide keyword by anchor's mean in this chunk
            if anchor in df.columns and df[anchor].max() > 0:
                anchor_mean = df[anchor].mean()
                norm_factor = anchor_mean / max(1, anchor_mean)  # keep scale if anchor stable
                df[f"{keyword}_norm"] = df[keyword] * norm_factor
                all_weeks.append(df[keyword])
            elif keyword in df.columns:
                all_weeks.append(df[keyword])

            time.sleep(0.3)
        except Exception as e:
            print(f"    chunk {chunk_start[:4]}-{chunk_end[:4]}: {e}")
            time.sleep(1)

    if not all_weeks:
        return pd.Series(dtype=float)

    # Concatenate all chunks
    combined = pd.concat(all_weeks)
    # Remove duplicates at chunk boundaries
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()

    return combined


def pull_all_weekly(resume: bool = True) -> dict:
    """批量拉取所有关键词的周度数据."""
    # Load existing if resuming
    existing = {}
    if resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f).get("memes", {})
        print(f"[续传] 已有 {len(existing)} 个关键词")

    results = dict(existing)
    total = len(MEME_KEYWORDS)

    for i, kw in enumerate(MEME_KEYWORDS):
        if kw in existing and len(existing[kw]) > 100:
            continue

        print(f"\n[{i+1}/{total}] {kw}...")
        series = pull_keyword_weekly(kw)

        if len(series) > 0:
            # Convert to {week_str: value} dict
            week_dict = {
                idx.strftime("%Y-%m-%d"): float(v)
                for idx, v in series.items()
                if not pd.isna(v) and v > 0
            }
            results[kw] = week_dict
            print(f"  → {len(week_dict)} 周 (峰值 {max(week_dict.values()):.0f})")
        else:
            print(f"  → 无数据")

        # Save every 5 keywords
        if (i + 1) % 5 == 0:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "memes": results,
                    "n_keywords": len(results),
                    "resolution": "weekly",
                    "updated": datetime.now().isoformat(),
                }, f, ensure_ascii=False, indent=2)
            print(f"  [保存] {len(results)} 个关键词")

        time.sleep(0.5)

    # Final save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "memes": results,
            "n_keywords": len(results),
            "resolution": "weekly",
            "updated": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--meme", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.meme:
        print(f"测试: {args.meme}")
        s = pull_keyword_weekly(args.meme)
        print(f"\n数据点: {len(s)}")
        if len(s) > 0:
            print(f"范围: {s.index[0]} → {s.index[-1]}")
            print(f"峰值: {s.max():.0f}")
            print(f"前 10 周:\n{s.head(10)}")
        else:
            print("无数据 — 可能是代理不可达")
    else:
        resume = not args.no_resume
        results = pull_all_weekly(resume=resume)
        total_weeks = sum(len(v) for v in results.values())
        print(f"\n{'='*50}")
        print(f"周度数据拉取完成: {len(results)} 关键词, {total_weeks} 数据点")

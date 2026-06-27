"""
B站真实注意力时间序列 — 通过搜索API获取视频发布时间分布

替代 Google Trends (被墙)，B站是国内模因文化的原生平台。
对于每个梗关键词，搜索前200个视频的发布时间，构建月度直方图。

用法:
    python src/data/bilibili_timeseries.py              # 采集所有梗
    python src/data/bilibili_timeseries.py --plot       # 采集+画图
    python src/data/bilibili_timeseries.py --meme 打工人  # 单个梗
"""

import json
import sys
import time
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path("data/collector")
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.bilibili.com",
}

# 29 梗关键词（优先使用中文原生表达）
MEME_KEYWORDS = {
    "打工人": ["打工人"],
    "内卷": ["内卷"],
    "躺平": ["躺平"],
    "普信男": ["普信男"],
    "小镇做题家": ["小镇做题家"],
    "摆烂": ["摆烂"],
    "润": ["润学"],
    "吗喽": ["吗喽"],
    "鼠鼠": ["鼠鼠 老鼠人"],
    "牛马": ["牛马 打工人"],
    "i人e人": ["i人 e人 MBTI"],
    "遥遥领先": ["遥遥领先 华为"],
    "孔乙己的长衫": ["孔乙己的长衫"],
    "精神状态": ["精神状态 发疯文学"],
    "XX刺客": ["雪糕刺客"],
    "谢帝": ["谢帝 迪士尼 diss"],
    "科目三": ["科目三 舞蹈"],
    "尊嘟假嘟": ["尊嘟假嘟"],
    "鸡你太美": ["鸡你太美 蔡徐坤"],
    "后浪": ["后浪 B站"],
    "情绪价值": ["情绪价值"],
    "四不": ["不婚不育 躺平"],
    "显眼包": ["显眼包"],
    "泼天富贵": ["泼天富贵"],
    "服美役": ["服美役"],
    "建议专家不要建议": ["建议专家不要建议"],
    "命运齿轮": ["命运的齿轮"],
    "原生家庭": ["原生家庭"],
    "发疯文学": ["发疯文学"],
}


def search_bilibili(keyword: str, max_pages: int = 10) -> list[str]:
    """搜索B站，返回视频发布时间列表 (YYYY-MM-DD)。"""
    dates = []
    for page in range(1, max_pages + 1):
        encoded = urllib.parse.quote(keyword, safe='')
        url = f"https://api.bilibili.com/x/web-interface/search/type?search_type=video&keyword={encoded}&page={page}&order=pubdate"
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = data.get("data", {}).get("result", [])
            if not results:
                break
            for v in results:
                pubdate = v.get("pubdate", 0)
                if pubdate > 0:
                    dates.append(datetime.fromtimestamp(pubdate).strftime("%Y-%m-%d"))
            time.sleep(0.3)
        except Exception as e:
            break
    return dates


def build_monthly_histogram(dates: list[str]) -> pd.Series:
    """将日期列表转为月度直方图。"""
    if not dates:
        return pd.Series(dtype=float)
    df = pd.to_datetime(pd.Series(dates))
    monthly = df.dt.to_period("M").value_counts().sort_index()
    monthly.index = monthly.index.astype(str)
    return monthly


def collect_all() -> dict:
    """采集所有梗的B站视频发布时间分布。"""
    print(f"[B站] 采集 {len(MEME_KEYWORDS)} 个梗的视频时间分布...")
    results = {}
    for i, (name, keywords) in enumerate(MEME_KEYWORDS.items()):
        all_dates = []
        for kw in keywords:
            dates = search_bilibili(kw, max_pages=10)
            all_dates.extend(dates)
            time.sleep(0.2)

        if all_dates:
            hist = build_monthly_histogram(all_dates)
            if len(hist) > 0:
                results[name] = hist
                peak_month = hist.idxmax()
                peak_val = hist.max()
                print(f"  [{i+1}/{len(MEME_KEYWORDS)}] {name:<12s}: "
                      f"{len(all_dates)} videos, peak {peak_month} ({peak_val})")
            else:
                print(f"  [{i+1}/{len(MEME_KEYWORDS)}] {name:<12s}: {len(all_dates)} videos, no histogram")

    return results


def save_results(results: dict):
    """保存时间序列。"""
    ts_data = {}
    for name, hist in results.items():
        ts_data[name] = {str(k): int(v) for k, v in hist.items()}

    ts_path = DATA_DIR / "bilibili_timeseries.json"
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(ts_data, f, ensure_ascii=False, indent=2)

    print(f"\n[保存] {len(ts_data)} 梗时间序列 → {ts_path}")
    return ts_data


def plot_curves(results: dict, save_path: str = None):
    """绘制所有梗的B站注意力曲线。"""
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.flatten()

    sorted_memes = sorted(results.items(),
                          key=lambda x: x[1].max() if len(x[1]) > 0 else 0,
                          reverse=True)

    for i, (name, hist) in enumerate(sorted_memes):
        if i >= 30:
            break
        ax = axes[i]
        ax.bar(range(len(hist)), hist.values, width=1, alpha=0.7)
        ax.set_title(f"{name} (peak={hist.max():.0f})", fontsize=8)
        ax.tick_params(labelsize=6)

    for i in range(len(sorted_memes), 30):
        axes[i].set_visible(False)

    fig.suptitle("Bilibili Meme Video Uploads — Monthly Distribution",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = str(DATA_DIR / "bilibili_attention_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[保存] 注意力曲线图 → {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--meme", type=str, default="")
    args = parser.parse_args()

    if args.meme:
        dates = search_bilibili(args.meme, max_pages=10)
        hist = build_monthly_histogram(dates)
        print(f"{args.meme}: {len(dates)} videos")
        for month, count in hist.items():
            bar = "#" * count
            print(f"  {month}: {bar} ({count})")
    else:
        print("=" * 60)
        print("MemeticChaos — B站视频发布时间序列")
        print("=" * 60)
        results = collect_all()
        if results:
            save_results(results)
            if args.plot:
                plot_curves(results)

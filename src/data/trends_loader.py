"""
Google Trends 历史数据加载器 — 拉取 29 个梗 2015-2025 真实注意力时间序列

用法:
    python src/data/trends_loader.py          # 拉取全部 29 梗
    python src/data/trends_loader.py --plot   # 拉取 + 画图
"""

import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from pytrends.request import TrendReq

DATA_DIR = Path("data/collector")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 29 梗名 → Google Trends 搜索词
MEME_KEYWORDS = [
    "打工人", "内卷", "躺平", "普信男", "小镇做题家",
    "摆烂", "润学", "吗喽", "鼠鼠", "牛马",
    "i人 e人", "遥遥领先 华为", "孔乙己的长衫", "精神状态 发疯文学",
    "雪糕刺客", "谢帝 迪士尼", "科目三 舞蹈",
    "尊嘟假嘟", "鸡你太美", "后浪",
    "情绪价值", "不婚不育", "显眼包",
    "泼天富贵", "服美役", "建议专家不要建议",
    "命运的齿轮", "原生家庭", "发疯文学",
]


def pull_all_historical(timeframe: str = "2015-01-01 2025-12-31",
                        geo: str = "CN") -> dict[str, pd.Series]:
    """批量拉取所有梗的 Google Trends 历史数据。

    Returns: {meme_name: monthly_interest_series}
    """
    pytrends = TrendReq(hl='zh-CN', tz=360, retries=3, backoff_factor=0.5)
    results = {}

    for i in range(0, len(MEME_KEYWORDS), 5):
        batch = MEME_KEYWORDS[i:i+5]
        print(f"[{i//5+1}/{(len(MEME_KEYWORDS)+4)//5}] {', '.join(batch)}")
        try:
            pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if not df.empty:
                df = df.drop(columns=['isPartial'], errors='ignore')
                monthly = df.resample('ME').mean()
                for kw in batch:
                    if kw in monthly.columns and monthly[kw].max() > 0:
                        # Normalize so the batch anchor (first kw) is proportional to real interest
                        results[kw] = monthly[kw]
                        print(f"  → peak {monthly[kw].idxmax().strftime('%Y-%m')} ({monthly[kw].max():.0f})")
            time.sleep(1.5)
        except Exception as e:
            print(f"  ✗ error: {e}")
            time.sleep(3)

    return results


def extract_lifecycle_params(series: pd.Series) -> dict:
    """从真实注意力时间序列提取生命周期参数。

    Returns:
        {
            "peak_time": "2021-06",
            "peak_value": 83.0,
            "emergence_time": "2020-10",  # first crossing 10% of peak
            "decay_time": "2022-03",       # last crossing 10% of peak
            "duration_months": 17,
            "has_resurgence": bool,
            "mean_interest": float,
            "total_attention": float,       # AUC
        }
    """
    s = series.dropna()
    if len(s) < 3 or s.max() < 1:
        return {"peak_time": None, "error": "insufficient data"}

    peak_idx = s.idxmax()
    peak_val = s.max()
    threshold = max(0.5, peak_val * 0.10)

    # Emergence: first month exceeding threshold
    above = s[s >= threshold]
    emergence = above.index[0] if len(above) > 0 else s.index[0]
    decay = above.index[-1] if len(above) > 0 else s.index[-1]

    # Resurgence detection: after first decay, re-exceeds 30% of peak
    post_peak = s[s.index > decay]
    resurgence = (post_peak > peak_val * 0.3).any()

    return {
        "peak_time": peak_idx.strftime("%Y-%m"),
        "peak_value": float(peak_val),
        "emergence_time": emergence.strftime("%Y-%m"),
        "decay_time": decay.strftime("%Y-%m"),
        "duration_months": max(1, int((decay - emergence).days / 30)),
        "has_resurgence": bool(resurgence),
        "mean_interest": float(s.mean()),
        "total_attention": float(s.sum()),
    }


def save_all(results: dict[str, pd.Series]):
    """保存所有时间序列和提取的参数。"""
    # Time series as JSON
    ts_data = {}
    lifecycle_params = {}
    for name, series in results.items():
        ts_data[name] = {
            str(d.strftime("%Y-%m")): float(v)
            for d, v in series.dropna().items()
        }
        lifecycle_params[name] = extract_lifecycle_params(series)

    # Save
    ts_path = DATA_DIR / "historical_timeseries.json"
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(ts_data, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {len(ts_data)} 梗时间序列 → {ts_path}")

    params_path = DATA_DIR / "lifecycle_from_trends.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(lifecycle_params, f, ensure_ascii=False, indent=2)
    print(f"[保存] {len(lifecycle_params)} 梗真实参数 → {params_path}")

    return ts_data, lifecycle_params


def plot_comparison(results: dict[str, pd.Series], save_path: str = None):
    """绘制所有梗的注意力曲线对比。"""
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.flatten()

    sorted_memes = sorted(results.items(),
                          key=lambda x: x[1].max() if len(x[1]) > 0 else 0,
                          reverse=True)

    for i, (name, series) in enumerate(sorted_memes):
        if i >= 30:
            break
        ax = axes[i]
        ax.plot(series.index, series.values, linewidth=1)
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)
        peak_date = series.idxmax()
        peak_val = series.max()
        if peak_val > 0:
            ax.annotate(f'{peak_val:.0f}', xy=(peak_date, peak_val),
                       fontsize=6, color='red')

    for i in range(len(sorted_memes), 30):
        axes[i].set_visible(False)

    fig.suptitle("Chinese Internet Memes — Google Trends Attention (2015-2025)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = str(DATA_DIR / "memes_attention_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[保存] 注意力曲线图 → {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="拉取后画图")
    parser.add_argument("--meme", type=str, default="", help="只拉取单个梗")
    args = parser.parse_args()

    if args.meme:
        pytrends = TrendReq(hl='zh-CN', tz=360)
        pytrends.build_payload([args.meme], cat=0, timeframe="2015-01-01 2025-12-31", geo="CN")
        df = pytrends.interest_over_time()
        if not df.empty:
            df = df.drop(columns=['isPartial'], errors='ignore')
            print(df[args.meme].describe())
            params = extract_lifecycle_params(df[args.meme])
            print(json.dumps(params, ensure_ascii=False, indent=2))
    else:
        print("=" * 60)
        print("MemeticChaos — Google Trends 历史数据加载")
        print("=" * 60)
        results = pull_all_historical()
        if results:
            ts_data, params = save_all(results)
            print(f"\n完成: {len(results)} 梗, {sum(len(v) for v in results.values())} 月数据点")
            if args.plot:
                plot_comparison(results)

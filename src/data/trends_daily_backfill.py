"""
Trends Daily Backfill — Google Trends 月级→日级回填 (2026-07-12, 历史重建 tier ①)

★观测算子★: [GoogleTrends-CN] 接入定义框 —— "搜索被 Google 归属到 CN 的查询"的注意力,
成分 UNKNOWN, 禁标精英/大众 (见 operator_ledger)。★只给注意力标量, 给不了叙事特征 →
拼不出日级 x(t), 不重跑 regime。★ 这一步是把已知关键词的注意力从月级提到日级。

方法 (踩成熟石头, overlap-chain 拼接):
  Google Trends 单窗 ≤~269 天返回日级 (0-100, 窗内独立归一)。十年日级 = 拼接重叠日窗:
  每窗 WIN 天、步长 STEP 天 (重叠 WIN-STEP 天); 后窗按重叠段均值比缩放到前窗 → 连续日级。
  ★验证 (monthly anchor)★: 拼好的日级按月聚合, 与直接拉的月级序列求相关 → 拼接是否忠实。
  拼接噪声音量依赖 (高频词≈0, 已 recon 实测; 低频词整数量化噪声大)。

用法:
  # 小样本证明 (默认 3 词):
  conda run -n MemeticChaos python src/data/trends_daily_backfill.py --prove
  # 全量 51 词回填 (慢, 强限速 → 建议服务器/后台跑):
  conda run -n MemeticChaos python src/data/trends_daily_backfill.py --all --out data/collector/trends_daily_2015_2025.json
"""

import sys, json, time, argparse
from pathlib import Path
from datetime import date, timedelta
import numpy as np

ROOT = Path(__file__).parent.parent.parent
PROXY = "http://127.0.0.1:7897"     # Clash; Google 需代理
WIN_DAYS = 250                      # 单窗天数 (< ~269 日级阈值)
STEP_DAYS = 150                     # 步长 (重叠 100 天)
GEO = "CN"
SLEEP = 2.0                         # 请求间隔 (限速)

PROVE_KEYWORDS = ["躺平", "内卷", "打工人"]


def _trendreq():
    from pytrends.request import TrendReq
    # retries=0 避开 urllib3 2.x method_whitelist 不兼容
    return TrendReq(hl="zh-CN", tz=360, timeout=(10, 30), proxies=[PROXY], retries=0)


def fetch(kw, timeframe, tries=5):
    """拉一个 timeframe 的 interest_over_time; 返回 pandas Series 或 None。"""
    for _ in range(tries):
        try:
            py = _trendreq()
            py.build_payload([kw], cat=0, timeframe=timeframe, geo=GEO)
            df = py.interest_over_time()
            if df is not None and not df.empty and kw in df:
                return df[kw].astype(float)
        except Exception:
            pass
        time.sleep(4)
    return None


def windows(start: date, end: date):
    """生成 [(win_start, win_end), ...] 覆盖 [start, end], 重叠 WIN-STEP 天。"""
    out = []
    s = start
    while s < end:
        e = min(s + timedelta(days=WIN_DAYS - 1), end)
        out.append((s, e))
        if e >= end:
            break
        s = s + timedelta(days=STEP_DAYS)
    return out


def stitch(series_list):
    """overlap-chain: 后窗按重叠段均值比缩放到前窗, 拼成连续日级 (相对尺度)。"""
    import pandas as pd
    acc = series_list[0].copy()
    for nxt in series_list[1:]:
        ov = acc.index.intersection(nxt.index)
        if len(ov) >= 5:
            ma, mn = acc.loc[ov].mean(), nxt.loc[ov].mean()
            scale = (ma / mn) if mn > 1e-9 else 1.0
        else:
            scale = 1.0     # 无重叠 (低频洞) → 不缩放, 标记留给验证
        nxt_scaled = nxt * scale
        # 合并: 重叠处取已累积值, 新段接上
        new_idx = nxt_scaled.index.difference(acc.index)
        acc = pd.concat([acc, nxt_scaled.loc[new_idx]]).sort_index()
    return acc


def validate_against_monthly(daily, anchor):
    """拼好的日级与独立 anchor 序列各按月聚合后求 Pearson 相关 (拼接忠实度)。

    ★anchor 频率随 span 变★: <9月=日, 9月~5年=周, >5年=月。故两边都 resample 到月, 避免频率错配。
    """
    if anchor is None or daily is None or len(daily) < 30:
        return None
    dm = daily.resample("MS").mean()
    am = anchor.resample("MS").mean()
    j = dm.index.intersection(am.index)
    if len(j) < 6:
        return None
    a, b = dm.loc[j].values, am.loc[j].values
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def backfill_keyword(kw, start, end):
    wins = windows(start, end)
    series, n_ok, n_empty = [], 0, 0
    for (s, e) in wins:
        tf = f"{s.isoformat()} {e.isoformat()}"
        ser = fetch(kw, tf)
        if ser is None:
            n_empty += 1
        else:
            n_ok += 1
            series.append(ser)
        time.sleep(SLEEP)
    if not series:
        return {"keyword": kw, "status": "FAILED", "windows": len(wins), "ok": 0}
    daily = stitch(series)
    monthly = fetch(kw, f"{start.isoformat()} {end.isoformat()}")
    corr = validate_against_monthly(daily, monthly)
    return {
        "keyword": kw, "status": "OK", "n_windows": len(wins), "n_ok": n_ok, "n_empty": n_empty,
        "n_days": int(len(daily)), "span": [str(daily.index.min().date()), str(daily.index.max().date())],
        "monthly_anchor_corr": round(corr, 4) if corr is not None else None,
        "daily": {str(d.date()): round(float(v), 2) for d, v in daily.items()},
    }


def main():
    ap = argparse.ArgumentParser(description="Trends 日级回填 [GoogleTrends-CN 框]")
    ap.add_argument("--prove", action="store_true", help=f"小样本证明 ({PROVE_KEYWORDS})")
    ap.add_argument("--all", action="store_true", help="全量 51 词 (慢, 建议服务器跑)")
    ap.add_argument("--keywords", type=str, help="逗号分隔自定义关键词")
    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default="2025-01-01")
    ap.add_argument("--out", type=str, default="data/processed/trends_daily_prove.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.keywords:
        kws = [k.strip() for k in args.keywords.split(",") if k.strip()]
    elif args.all:
        ext = json.load(open(ROOT / "data/collector/external_field_2015_2025.json", encoding="utf-8"))
        kws = list(ext.get("keywords", ext)) if isinstance(ext, dict) else []
    else:
        kws = PROVE_KEYWORDS

    start = date.fromisoformat(args.start); end = date.fromisoformat(args.end)
    nwin = len(windows(start, end))
    print(f"[GoogleTrends-CN 框] 日级回填: {len(kws)} 词 × {nwin} 窗/词 ({WIN_DAYS}d 窗, {STEP_DAYS}d 步)")
    print(f"预计请求 ~{len(kws)*(nwin+1)} 次 @ {SLEEP}s+ 间隔; ★强限速, 全量建议服务器/后台★\n")

    results = {}
    t0 = None
    for i, kw in enumerate(kws):
        import time as _t
        st = _t.time()
        r = backfill_keyword(kw, start, end)
        dt = _t.time() - st
        results[kw] = {k: v for k, v in r.items() if k != "daily"}  # 摘要不含全日序列
        results[kw]["seconds"] = round(dt, 1)
        c = r.get("monthly_anchor_corr")
        print(f"  [{i+1}/{len(kws)}] {kw}: {r['status']} | 窗 {r.get('n_ok')}/{r.get('n_windows')} "
              f"| {r.get('n_days')} 天 | anchor_corr={c} | {dt:.0f}s")
        # 存全日序列
        if r["status"] == "OK":
            outp = ROOT / args.out.replace(".json", f"_{kw}.json")
            outp.parent.mkdir(parents=True, exist_ok=True)
            json.dump(r, open(outp, "w", encoding="utf-8"), ensure_ascii=False)

    summ = ROOT / args.out
    json.dump({"operator": "GoogleTrends-CN", "frame": "接入定义/成分未知; 只注意力非x(t)",
               "win_days": WIN_DAYS, "step_days": STEP_DAYS, "start": args.start, "end": args.end,
               "results": results}, open(summ, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已保存摘要 → {summ}")
    oks = [r for r in results.values() if r.get("status") == "OK"]
    if oks:
        corrs = [r["monthly_anchor_corr"] for r in oks if r.get("monthly_anchor_corr") is not None]
        if corrs:
            print(f"monthly-anchor 相关: 中位 {np.median(corrs):.3f} (拼接忠实度; 接近1=好)")


if __name__ == "__main__":
    main()

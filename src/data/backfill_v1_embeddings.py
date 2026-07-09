"""
V1.0 Scraper 文件 → Embedding 回溯

背景: 早期 (2026-06 ~ 2026-07) scraper 以 v1.0 格式采集了 227 个文件,
当时 embedding 模型未加载, 只存了 headline 文本 (weibo_top10/baidu_top10/zhihu_top10),
没有 384 维向量。现在模型已缓存 + 代理已通, 回溯补全 embedding。

做法:
  1. 扫描 data/scraped/scrape_*.json (v1.0)
  2. 从 *_top10 键提取 headline, 打平台标签
  3. scraper.embed_headlines() 批量编码 384 维
  4. 写成 v2.0 hourly 格式 → data/scraped/hourly/YYYY-MM-DD_HHMMSS.json
  5. monthly_aggregator.py 即可把这些月份聚合进月度语义状态

幂等: 目标 hourly 文件已存在则跳过 (除非 --force)。
诚实标注: v1.0 只存了每平台 top10 (非全量 50), 故回溯的每文件 headline 数
        (~10-30) 少于 v2.0 的 ~130。聚合出的月度状态分辨率偏低, 输出里标 version
        "2.0-backfill" 以便下游区分。

用法 (必须在 repo 根运行, 因 scraper 用相对路径):
    conda run -n MemeticChaos python src/data/backfill_v1_embeddings.py            # 全量回溯
    conda run -n MemeticChaos python src/data/backfill_v1_embeddings.py --dry-run  # 只解析不编码
    conda run -n MemeticChaos python src/data/backfill_v1_embeddings.py --force    # 覆盖已存在
"""

import json, sys, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.scraper import embed_headlines, detect_meme_signals, EMBEDDING_DIM  # noqa: E402

SCRAPED_DIR = ROOT / "data" / "scraped"
HOURLY_DIR = SCRAPED_DIR / "hourly"

# v1.0 里 headline 所在的键 → 平台名
TOP_KEYS = {"weibo_top10": "weibo", "baidu_top10": "baidu", "zhihu_top10": "zhihu"}


def parse_timestamp(v1: dict, path: Path) -> datetime | None:
    """从 v1.0 的 timestamp 字段 (YYYYMMDD_HHMMSS) 或文件名解析时间。"""
    ts = v1.get("timestamp")
    for candidate in (ts, path.stem.replace("scrape_", "")):
        if candidate and len(candidate) >= 15:
            try:
                return datetime.strptime(candidate[:15], "%Y%m%d_%H%M%S")
            except ValueError:
                continue
    return None


def extract_items(v1: dict) -> list[dict]:
    """从 v1.0 的 *_top10 键抽取 headline items, 打平台标签。"""
    items = []
    for key, platform in TOP_KEYS.items():
        for entry in v1.get(key, []) or []:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            items.append({
                "title": title,
                "platform": platform,
                "rank": entry.get("rank", 0),
                "hot_score": entry.get("hot_score", 0),
                "url": entry.get("url", ""),
            })
    return items


def build_hourly(v1: dict, ts: datetime, items: list[dict], embedded: list[dict]) -> dict:
    """组装 v2.0 hourly 结构 (monthly_aggregator 消费 headlines_embedded)。"""
    date_str = ts.strftime("%Y-%m-%d")
    # signals 是 live pipeline 报告用, 对语义聚合非必需; 尽力算, 失败置空
    try:
        signals = detect_meme_signals(items)
    except Exception:
        signals = []

    by_plat = {"weibo": [], "baidu": [], "zhihu": []}
    for it in items:
        by_plat.setdefault(it["platform"], []).append(it)

    return {
        "version": "2.0-backfill",
        "backfilled_at": datetime.now().isoformat(),
        "timestamp": ts.isoformat(),
        "date": date_str,
        "hour": ts.hour,
        "counts": {
            "weibo": len(by_plat.get("weibo", [])),
            "baidu": len(by_plat.get("baidu", [])),
            "zhihu": len(by_plat.get("zhihu", [])),
            "total": len(items),
            "embedded": len(embedded),
            "signals": len(signals),
        },
        "signals": signals,
        "headlines_embedded": embedded,
        "top_per_platform": {
            plat: [{"rank": it["rank"], "title": it["title"], "hot_score": it["hot_score"]}
                   for it in lst[:10]]
            for plat, lst in by_plat.items()
        },
    }


def main():
    ap = argparse.ArgumentParser(description="V1.0 scraper 文件 embedding 回溯")
    ap.add_argument("--dry-run", action="store_true", help="只解析统计, 不编码不写文件")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 hourly 文件")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    v1_files = sorted(SCRAPED_DIR.glob("scrape_*.json"))
    print(f"发现 {len(v1_files)} 个 v1.0 文件")
    if not v1_files:
        print("无 v1.0 文件, 退出。")
        return

    HOURLY_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "skipped_exist": 0, "skipped_empty": 0,
             "skipped_badtime": 0, "headlines": 0, "months": {}}

    for fp in v1_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                v1 = json.load(f)
        except Exception as e:
            print(f"  ✗ {fp.name}: 读取失败 {e}")
            continue

        ts = parse_timestamp(v1, fp)
        if ts is None:
            stats["skipped_badtime"] += 1
            continue

        items = extract_items(v1)
        if not items:
            stats["skipped_empty"] += 1
            continue

        out_path = HOURLY_DIR / f"{ts.strftime('%Y-%m-%d')}_{ts.strftime('%H%M%S')}.json"
        if out_path.exists() and not args.force:
            stats["skipped_exist"] += 1
            continue

        month = ts.strftime("%Y-%m")
        stats["months"][month] = stats["months"].get(month, 0) + len(items)
        stats["headlines"] += len(items)

        if args.dry_run:
            stats["processed"] += 1
            continue

        embedded = embed_headlines(items)
        if embedded is None:
            print("  ✗ 模型不可用, 中止 (检查 sentence-transformers + 模型缓存)")
            return
        hourly = build_hourly(v1, ts, items, embedded)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hourly, f, ensure_ascii=False)
        stats["processed"] += 1

    print(f"\n{'='*56}")
    print(f"回溯{'预演' if args.dry_run else ''}完成")
    print(f"{'='*56}")
    print(f"  处理: {stats['processed']}  跳过(已存在): {stats['skipped_exist']}  "
          f"跳过(无headline): {stats['skipped_empty']}  跳过(时间戳坏): {stats['skipped_badtime']}")
    print(f"  headline 总数: {stats['headlines']} ({EMBEDDING_DIM} 维)")
    print(f"  覆盖月份:")
    for m in sorted(stats["months"]):
        print(f"    {m}: {stats['months'][m]} headlines")
    if not args.dry_run and stats["processed"]:
        print(f"\n  下一步: conda run -n MemeticChaos python src/data/monthly_aggregator.py")


if __name__ == "__main__":
    main()

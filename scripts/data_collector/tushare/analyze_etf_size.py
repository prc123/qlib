"""Analyze ETF fund-size (规模) distribution via Tushare snapshots.

Fetches market-wide fund_share + fund_nav at monthly month-end trading days
over the past 12 months (single API call per date, no per-symbol loop),
computes size = fd_share(万份) * unit_nav(元) = 万元, and reports the
distribution plus impact of candidate thresholds.

Outputs
-------
    <qlib_data_dir>/etf_size_stats.csv   per-ETF mean/recent size (亿元)

Usage
-----
    python analyze_etf_size.py --qlib_data_dir C:/Users/Administrator/.qlib/qlib_data/etf_data
"""
import os
import time
from pathlib import Path

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = "*"

import argparse

import pandas as pd
import tushare as ts

TUSHARE_TOKEN = "a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202"


def fetch_page(fn, **kwargs):
    """Fetch all pages of a tushare endpoint (defensive pagination)."""
    rows, offset, page_size = [], 0, 5000
    while True:
        df = fn(limit=page_size, offset=offset, **kwargs)
        if df is None or df.empty:
            break
        rows.append(df)
        if len(df) < page_size:
            break
        offset += page_size
        time.sleep(0.3)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pick_month_ends(calendar_dates, n_months=12):
    """Last trading day of each of the last n_months calendar months."""
    s = pd.Series(pd.to_datetime(calendar_dates))
    month_ends = s.groupby(s.dt.to_period("M")).max()
    return [d.strftime("%Y%m%d") for d in month_ends.tail(n_months)]


def main():
    parser = argparse.ArgumentParser(description="ETF fund-size distribution analysis")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\Administrator\.qlib\qlib_data\etf_data")
    parser.add_argument("--n_months", type=int, default=12)
    args = parser.parse_args()

    base = Path(args.qlib_data_dir)
    with open(base / "calendars" / "day.txt") as f:
        cal = [line.strip() for line in f if line.strip()]
    snap_dates = pick_month_ends(cal, args.n_months)
    print(f"Snapshot dates ({len(snap_dates)}): {snap_dates}")

    pro = ts.pro_api(TUSHARE_TOKEN)

    snaps = []
    for d in snap_dates:
        try:
            share = fetch_page(pro.fund_share, trade_date=d)
            nav = fetch_page(pro.fund_nav, nav_date=d)
        except Exception as e:
            print(f"  {d}: FAILED {str(e)[:80]}")
            time.sleep(1)
            continue
        if share.empty or nav.empty:
            print(f"  {d}: share_rows={len(share)} nav_rows={len(nav)} -> skipped")
            continue
        m = share[["ts_code", "fd_share"]].merge(
            nav[["ts_code", "unit_nav"]], on="ts_code", how="inner"
        )
        m["size_yi"] = m["fd_share"] * m["unit_nav"] / 1e4  # 万份*元=万元 -> 亿元
        m["date"] = d
        snaps.append(m[["ts_code", "date", "size_yi"]])
        print(f"  {d}: share={len(share)} nav={len(nav)} matched={len(m)}")
        time.sleep(0.3)

    if not snaps:
        print("No snapshots fetched, abort.")
        return

    all_snap = pd.concat(snaps, ignore_index=True)
    agg = (
        all_snap.groupby("ts_code")["size_yi"]
        .agg(size_mean="mean", size_recent="last", n_snaps="count")
        .reset_index()
    )

    # Map ts_code (510050.SH) -> qlib symbol (SH510050)
    agg["symbol"] = agg["ts_code"].str.split(".").apply(lambda x: x[1] + x[0])
    agg["symbol"] = agg["symbol"].str.lower()

    insts = pd.read_csv(base / "instruments" / "all.txt", sep="\t", header=None, names=["symbol", "s", "e"])
    valid = set(insts["symbol"].str.lower())
    agg_valid = agg[agg["symbol"].isin(valid)].copy()

    out_path = base / "etf_size_stats.csv"
    agg_valid.to_csv(out_path, index=False)
    print(f"\nSaved {len(agg_valid)} rows -> {out_path}")

    s = agg_valid["size_mean"]
    print("\n=== 规模分布 (近12月月均, 亿元) ===")
    print(f"  数量: {len(s)}")
    print("  分位数:")
    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        print(f"    P{int(q*100):02d}: {s.quantile(q):8.2f}")
    print(f"  min={s.min():.3f}  max={s.max():.1f}")

    print("\n=== 阈值影响 (月均规模 >= 阈值) ===")
    print(f"  {'阈值':>8} {'保留':>6} {'剔除':>6} {'保留%':>8}")
    for th in [0.5, 1, 2, 3, 5, 10, 20, 50]:
        keep = (s >= th).sum()
        print(f"  {th:>6}亿 {keep:>6} {len(s)-keep:>6} {keep/len(s):>7.1%}")

    print("\n=== 近期规模 (最近快照) 同口径对照 ===")
    sr = agg_valid["size_recent"]
    for th in [0.5, 1, 2, 3, 5, 10, 20, 50]:
        keep = (sr >= th).sum()
        print(f"  >={th:>5}亿: keep={keep}  drop={len(sr)-keep}  ({keep/len(sr):.1%})")


if __name__ == "__main__":
    main()

"""Generate size-bounded ETF instruments files (instruments/size_<min>_<max>.txt).

Size = fd_share(万份) * unit_nav(元) / 1e4 = 亿元.

Sources:
  1. etf_size_stats.csv  — recent monthly-end snapshots (12M mean size), from
     analyze_etf_size.py.
  2. For symbols missing there (mostly delisted ETFs): fetch per-symbol
     fund_share + fund_nav history from Tushare and compute the mean size
     over the last ~12 months of overlapping data before delisting.  Results
     are appended back to etf_size_stats.csv so the fetch happens only once.

Usage
-----
    python gen_instruments_by_size.py --min_yi 2 --max_yi 50
"""
import os
import time
import argparse
from pathlib import Path

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = "*"

import pandas as pd
import tushare as ts
from loguru import logger

TUSHARE_TOKEN = "a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202"

HIST_WINDOW_DAYS = 365  # trailing window for delisted-ETF size


def to_qlib_symbol(ts_code: str) -> str:
    """510050.SH -> sh510050"""
    code, exch = ts_code.split(".")
    return f"{exch}{code}".lower()


def fetch_all(fn, **kwargs) -> pd.DataFrame:
    rows, offset, page = [], 0, 5000
    while True:
        df = fn(limit=page, offset=offset, **kwargs)
        if df is None or df.empty:
            break
        rows.append(df)
        if len(df) < page:
            break
        offset += page
        time.sleep(0.25)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def hist_mean_size(pro, ts_code: str):
    """Mean size (亿元) over the last HIST_WINDOW_DAYS of shared share/nav dates."""
    try:
        share = fetch_all(pro.fund_share, ts_code=ts_code)
        nav = fetch_all(pro.fund_nav, ts_code=ts_code)
    except Exception as e:
        logger.warning(f"  {ts_code}: {str(e)[:60]}")
        return None
    if share.empty or nav.empty:
        return None
    share["date"] = pd.to_datetime(share["trade_date"], format="%Y%m%d")
    nav["date"] = pd.to_datetime(nav["nav_date"], format="%Y%m%d")
    m = share[["date", "fd_share"]].merge(nav[["date", "unit_nav"]], on="date")
    if m.empty:
        return None
    cutoff = m["date"].max() - pd.Timedelta(days=HIST_WINDOW_DAYS)
    m = m[m["date"] >= cutoff]
    return float((m["fd_share"] * m["unit_nav"]).mean() / 1e4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\Administrator\.qlib\qlib_data\etf_data")
    parser.add_argument("--min_yi", type=float, default=2.0)
    parser.add_argument("--max_yi", type=float, default=50.0)
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()

    base = Path(args.qlib_data_dir)
    stats_path = base / "etf_size_stats.csv"
    stats = pd.read_csv(stats_path)
    size_map = dict(zip(stats["symbol"], stats["size_mean"]))

    insts = pd.read_csv(base / "instruments" / "all.txt", sep="\t", header=None,
                        names=["symbol", "start", "end"])
    all_syms = insts["symbol"].str.lower().tolist()
    missing = [s for s in all_syms if s not in size_map]
    logger.info(f"all={len(all_syms)}  have_size={len(all_syms)-len(missing)}  missing={len(missing)}")

    if missing:
        pro = ts.pro_api(TUSHARE_TOKEN)
        fetched = 0
        for i, sym in enumerate(missing):
            exch = sym[:2].upper()
            code = sym[2:].upper()
            ts_code = f"{code}.{exch}"
            v = hist_mean_size(pro, ts_code)
            time.sleep(args.delay)
            if v is not None:
                size_map[sym] = v
                stats.loc[len(stats)] = {
                    "ts_code": ts_code, "symbol": sym,
                    "size_mean": v, "size_recent": None,
                    "n_snaps": -1,  # -1 marks historical (delisted) estimate
                }
                fetched += 1
            if (i + 1) % 100 == 0:
                logger.info(f"  {i+1}/{len(missing)} fetched={fetched}")
        stats.to_csv(stats_path, index=False)
        logger.info(f"Fetched {fetched}/{len(missing)} historical sizes, stats saved")

    lo, hi = args.min_yi, args.max_yi
    keep, drop_lo, drop_hi, drop_none = [], 0, 0, 0
    for sym in all_syms:
        v = size_map.get(sym)
        if v is None:
            drop_none += 1
        elif v < lo:
            drop_lo += 1
        elif v > hi:
            drop_hi += 1
        else:
            keep.append(sym)

    keep_set = set(keep)
    out = insts[insts["symbol"].str.lower().isin(keep_set)]
    out_name = f"size_{lo:g}_{hi:g}.txt"
    out_path = base / "instruments" / out_name
    out.to_csv(out_path, sep="\t", header=False, index=False)
    logger.info(f"kept={len(keep)}  drop(<{lo}yi)={drop_lo}  drop(>{hi}yi)={drop_hi}  "
                f"drop(no_size)={drop_none}  -> {out_path}")

    bench = "sh510050"
    logger.info(f"benchmark {bench.upper()} in pool: {bench in keep_set} "
                f"(size={size_map.get(bench, float('nan')):.1f}亿)")


if __name__ == "__main__":
    main()

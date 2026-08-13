"""Download ETF share + NAV fields and merge into existing source CSVs.

Adds two ETF-specific fields that Alpha158 doesn't have:
  - share  (fd_share from fund_share)  → 资金申赎流入口径
  - nav    (unit_nav from fund_nav)    → 折溢价率计算基础

Usage
-----
    python etf_extra_fields.py --instruments_file C:/Users/pp/.qlib/qlib_data/etf_data/instruments/stock.txt
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts
from loguru import logger

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = "*"

TUSHARE_TOKEN = "a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202"
CUR_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CUR_DIR / "etf_source"


def get_symbols(instruments_file):
    """Read stock ETF symbols from qlib instruments file."""
    df = pd.read_csv(instruments_file, sep="\t", header=None, names=["symbol", "start", "end"])
    return sorted(df["symbol"].str.lower().unique().tolist())


def to_ts_code(symbol):
    """sh510050 -> 510050.SH"""
    code = symbol[2:]
    exchange = "SH" if symbol.startswith("sh") else "SZ"
    return f"{code}.{exchange}"


def download_share(pro, symbols, delay=0.3):
    """Download fund_share for all symbols, merge into source CSVs."""
    logger.info(f"Downloading fund_share for {len(symbols)} symbols ...")
    for i, sym in enumerate(symbols):
        src_path = SOURCE_DIR / f"{sym}.csv"
        if not src_path.exists():
            continue
        ts_code = to_ts_code(sym)
        try:
            df = pro.fund_share(ts_code=ts_code)
            if df is not None and not df.empty:
                df = df.rename(columns={"fd_share": "share"})
                df["date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
                # Merge into source CSV
                src = pd.read_csv(src_path)
                src["date"] = src["date"].astype(str)
                merged = src.merge(df[["date", "share"]], on="date", how="left")
                merged.to_csv(src_path, index=False)
        except Exception as e:
            logger.warning(f"  {sym} share: {str(e)[:60]}")
        if (i + 1) % 500 == 0:
            logger.info(f"  {i+1}/{len(symbols)} done")
        time.sleep(delay)
    logger.info("fund_share download done.")


def download_nav(pro, symbols, delay=0.3):
    """Download fund_nav for all symbols, merge into source CSVs."""
    logger.info(f"Downloading fund_nav for {len(symbols)} symbols ...")
    for i, sym in enumerate(symbols):
        src_path = SOURCE_DIR / f"{sym}.csv"
        if not src_path.exists():
            continue
        ts_code = to_ts_code(sym)
        try:
            df = pro.fund_nav(ts_code=ts_code)
            if df is not None and not df.empty:
                df = df.rename(columns={"unit_nav": "nav"})
                df["date"] = pd.to_datetime(df["nav_date"]).dt.strftime("%Y-%m-%d")
                src = pd.read_csv(src_path)
                src["date"] = src["date"].astype(str)
                merged = src.merge(df[["date", "nav"]], on="date", how="left")
                merged.to_csv(src_path, index=False)
        except Exception as e:
            logger.warning(f"  {sym} nav: {str(e)[:60]}")
        if (i + 1) % 500 == 0:
            logger.info(f"  {i+1}/{len(symbols)} done")
        time.sleep(delay)
    logger.info("fund_nav download done.")


def main():
    import fire
    fire.Fire(run)


def run(instruments_file, delay=0.3):
    pro = ts.pro_api(TUSHARE_TOKEN)
    symbols = get_symbols(instruments_file)
    logger.info(f"Total symbols: {len(symbols)}")
    download_share(pro, symbols, delay)
    download_nav(pro, symbols, delay)
    logger.info("All done. Verify a sample source CSV for 'share' and 'nav' columns.")


if __name__ == "__main__":
    import fire
    fire.Fire(run)

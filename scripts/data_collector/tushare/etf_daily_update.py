"""
Daily incremental update for ETF data (OHLCV + share + nav).

Downloads new trading days via Tushare bulk-by-date, merges into source CSVs,
then fully re-normalizes and re-dumps to qlib binary format.

Usage
-----
    python etf_daily_update.py --qlib_data_1d_dir C:/Users/pp/.qlib/qlib_data/etf_data

Schedule (Windows Task Scheduler)
-----------------------------------
    Program:  C:\\Users\\pp\\.conda\\envs\\qlib\\python.exe
    Arguments: etf_daily_update.py --qlib_data_1d_dir C:\\Users\\pp\\.qlib\\qlib_data\\etf_data
    Start in:  E:\\kaggle_code\\qlib\\scripts\\data_collector\\tushare
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

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
# Repo root so that the in-repo qlib package is importable from any cwd.
sys.path.append(str(CUR_DIR.parent.parent.parent))

from dump_bin import DumpDataAll

TUSHARE_TOKEN = "a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202"

logger.add(
    CUR_DIR.joinpath("etf_daily_update.log"),
    rotation="30 days",
    retention="90 days",
    level="INFO",
)


def get_last_trading_date(qlib_data_dir: Path) -> str:
    cal_path = qlib_data_dir / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"Calendar not found: {cal_path}")
    cal = pd.read_csv(cal_path)
    return pd.Timestamp(cal.iloc[-1, 0]).strftime("%Y-%m-%d")


def _norm_symbol(ts_code: str) -> str:
    code, exchange = ts_code.split(".")
    return f"{exchange.lower()}{code}"


def _merge_columns(source_dir: Path, df: pd.DataFrame, col_name: str, date_col: str):
    """Merge a bulk-downloaded column (share/nav) into per-symbol source CSVs."""
    if df is None or df.empty:
        return
    df = df.copy()
    df["symbol"] = df["ts_code"].apply(_norm_symbol)
    df["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

    # Build symbol -> {date: value} map
    sym_date_map = {}
    for sym, group in df.groupby("symbol"):
        sym_date_map[sym] = dict(zip(group["date"], group[col_name]))

    updated = 0
    for sym, date_map in sym_date_map.items():
        src_path = source_dir / f"{sym}.csv"
        if not src_path.exists():
            continue
        src = pd.read_csv(src_path)
        src["date"] = src["date"].astype(str)
        for d, v in date_map.items():
            mask = src["date"] == d
            if mask.any():
                src.loc[mask, col_name] = v
                updated += 1
        src.to_csv(src_path, index=False)
    logger.info(f"  {col_name}: updated {updated} rows")


def _refresh_instruments(qlib_dir: Path):
    """Extend end dates in instrument files to latest calendar date."""
    instr_dir = qlib_dir / "instruments"
    cal_path = qlib_dir / "calendars" / "day.txt"
    if not instr_dir.exists() or not cal_path.exists():
        return
    cal = pd.read_csv(cal_path)
    last_date = str(cal.iloc[-1, 0])

    for f in sorted(instr_dir.glob("*.txt")):
        df = pd.read_csv(f, sep="\t", header=None, names=["symbol", "start", "end"])
        max_end = str(df["end"].max())
        if max_end < last_date:
            changed = 0
            for idx, row in df.iterrows():
                if str(row["end"]) == max_end:
                    df.at[idx, "end"] = last_date
                    changed += 1
            if changed:
                df.to_csv(f, sep="\t", header=False, index=False)
                logger.info(f"Updated {changed} rows in {f.name}")


def daily_update(
    qlib_data_1d_dir: str = r"C:\Users\pp\.qlib\qlib_data\etf_data",
    source_dir: str = None,
    normalize_dir: str = None,
    max_workers: int = None,
    delay: float = 0.3,
):
    """Run the ETF daily update pipeline.

    Steps:
    1. Bulk-download new OHLCV (fund_daily + fund_adj)
    2. Bulk-download new share (fund_share) + nav (fund_nav)
    3. Full re-normalize (handles share/nav)
    4. Full re-dump to qlib binary
    5. Refresh instrument end dates
    """
    import multiprocessing
    from etf_collector import Run

    qlib_dir = Path(qlib_data_1d_dir).expanduser().resolve()
    if not qlib_dir.exists():
        raise FileNotFoundError(f"qlib data dir not found: {qlib_dir}")

    trading_date = get_last_trading_date(qlib_dir)
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    logger.info(f"qlib data dir : {qlib_dir}")
    logger.info(f"last trading  : {trading_date}")
    logger.info(f"download range: {trading_date} -> {end_date}")

    if max_workers is None:
        max_workers = max(multiprocessing.cpu_count() - 2, 1)
    if source_dir is None:
        source_dir = str(CUR_DIR / "etf_source")
    if normalize_dir is None:
        normalize_dir = str(CUR_DIR / "etf_normalize")

    run = Run(source_dir=source_dir, normalize_dir=normalize_dir, max_workers=1, interval="1d")

    # Step 1: OHLCV (fund_daily + fund_adj) bulk download
    logger.info("=" * 50)
    logger.info("Step 1/5: Bulk-downloading OHLCV (fund_daily + fund_adj)...")
    run.download_data_bulk(delay=delay, start=trading_date, end=end_date)

    # Step 2: share + nav bulk download
    logger.info("=" * 50)
    logger.info("Step 2/5: Bulk-downloading share (fund_share) + nav (fund_nav)...")
    pro = ts.pro_api(TUSHARE_TOKEN)
    cal = pro.trade_cal(exchange="SSE", start_date=trading_date.replace("-", ""),
                        end_date=end_date.replace("-", ""))
    cal = cal[cal["is_open"] == 1]
    trade_dates = sorted(cal["cal_date"].tolist())
    source_path = Path(source_dir)

    for td in trade_dates:
        d = f"{td[:4]}-{td[4:6]}-{td[6:]}"
        # share
        try:
            share_df = pro.fund_share(trade_date=td)
            if share_df is not None and not share_df.empty:
                share_df = share_df.rename(columns={"fd_share": "share"})
                _merge_columns(source_path, share_df, "share", "trade_date")
        except Exception as e:
            logger.warning(f"fund_share {d}: {str(e)[:60]}")
        time.sleep(delay)
        # nav
        try:
            nav_df = pro.fund_nav(nav_date=td)
            if nav_df is not None and not nav_df.empty:
                nav_df = nav_df[nav_df["ts_code"].str.endswith((".SH", ".SZ"))]
                nav_df = nav_df.rename(columns={"unit_nav": "nav"})
                _merge_columns(source_path, nav_df, "nav", "nav_date")
        except Exception as e:
            logger.warning(f"fund_nav {d}: {str(e)[:60]}")
        time.sleep(delay)

    # Step 3: Full re-normalize
    logger.info("=" * 50)
    logger.info("Step 3/5: Re-normalizing (full mode)...")
    run.max_workers = max_workers
    run.normalize_data()

    # Step 4: Full re-dump
    logger.info("=" * 50)
    logger.info("Step 4/5: Re-dumping to qlib binary...")
    _dump = DumpDataAll(
        data_path=str(run.normalize_dir),
        qlib_dir=str(qlib_dir),
        exclude_fields="symbol,date",
        max_workers=max_workers,
    )
    _dump.dump()

    # Step 5: Refresh instruments
    logger.info("=" * 50)
    logger.info("Step 5/5: Refreshing instrument end dates...")
    _refresh_instruments(qlib_dir)

    logger.info("=" * 50)
    logger.info("ETF daily update completed!")


if __name__ == "__main__":
    import fire
    fire.Fire(daily_update)

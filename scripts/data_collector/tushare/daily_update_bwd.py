"""
Daily update script for backward-adjusted (后复权) A-share data.

Same as ``daily_update_fwd.py`` but uses backward-adjusted adjclose
(adjclose = close * adj_factor), making the $factor stable for
historical dates. This avoids the forward-adjustment discontinuity
that occurs when Tushare's adj_factor is recalibrated over time.

Usage
-----
    $ python daily_update_bwd.py --qlib_data_1d_dir C:/Users/pp/.qlib/qlib_data/cn_data_bwd

Schedule (Windows Task Scheduler)
-----------------------------------
    Program: C:\\Users\\pp\\.conda\\envs\\qlib\\python.exe
    Arguments: daily_update_bwd.py --qlib_data_1d_dir C:\\Users\\pp\\.qlib\\qlib_data\\cn_data_bwd
    Start in: E:\\kaggle_code\\qlib\\scripts\\data_collector\\tushare
"""

import sys
import multiprocessing
from pathlib import Path

import pandas as pd
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate

logger.add(
    CUR_DIR.joinpath("daily_update_bwd.log"),
    rotation="30 days",
    retention="90 days",
    level="INFO",
)


def get_last_trading_date(qlib_data_dir: Path) -> str:
    calendar_path = qlib_data_dir.joinpath("calendars/day.txt")
    if not calendar_path.exists():
        raise FileNotFoundError(f"Calendar file not found: {calendar_path}")
    calendar_df = pd.read_csv(calendar_path)
    last_date = pd.Timestamp(calendar_df.iloc[-1, 0])
    return last_date.strftime("%Y-%m-%d")


def _refresh_instruments(qlib_dir: Path):
    """Extend end dates in all instrument files to match the latest calendar date."""
    instr_dir = qlib_dir / "instruments"
    all_path = instr_dir / "all.txt"
    cal_path = qlib_dir / "calendars" / "day.txt"
    if not all_path.exists() or not cal_path.exists():
        return

    cal = pd.read_csv(cal_path)
    last_cal_date = str(cal.iloc[-1, 0])

    all_df = pd.read_csv(all_path, sep="\t", header=None, names=["symbol", "start", "end"])
    all_max_end = all_df["end"].max()

    all_end_map = dict(zip(all_df["symbol"].astype(str), all_df["end"]))

    if str(all_max_end) < last_cal_date:
        all_changed = 0
        for idx, row in all_df.iterrows():
            if str(row["end"]) == str(all_max_end):
                all_df.at[idx, "end"] = last_cal_date
                all_changed += 1
        if all_changed:
            all_df.to_csv(all_path, sep="\t", header=False, index=False)
            logger.info(f"Updated {all_changed} stocks in all.txt")
            all_end_map = dict(zip(all_df["symbol"].astype(str), all_df["end"]))

    for f in sorted(instr_dir.glob("*.txt")):
        if f.name == "all.txt":
            continue
        df = pd.read_csv(f, sep="\t", header=None, names=["symbol", "start", "end"])
        changed = 0
        for idx, row in df.iterrows():
            sym = str(row["symbol"])
            if str(row["end"]) != last_cal_date and sym in all_end_map:
                if str(all_end_map[sym]) >= last_cal_date:
                    df.at[idx, "end"] = last_cal_date
                    changed += 1
                elif str(row["end"]) != str(all_end_map[sym]):
                    df.at[idx, "end"] = all_end_map[sym]
                    changed += 1
        if changed:
            df.to_csv(f, sep="\t", header=False, index=False)
            logger.info(f"Updated {changed} stocks in {f.name}")


def daily_update_bwd(
    qlib_data_1d_dir: str = r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd",
    source_dir: str = None,
    normalize_dir: str = None,
    max_workers: int = None,
    delay: float = 0.3,
    update_index_weights: bool = False,
    index_weight_freq: str = "ME",
    include_alpha_factors: bool = True,
):
    """Run the daily data update pipeline for backward-adjusted data.

    Steps:
    1. Identify the last trading date in existing qlib data
    2. Bulk-download new OHLCV + daily_basic from Tushare
    3. Download new alpha factor data (moneyflow + margin)
    4. Normalize (extend mode: align with existing data scales)
    5. Dump to qlib binary format
    6. Refresh instrument end dates to match calendar
    7. (Optional) Update index constituent weights

    Parameters
    ----------
    include_alpha_factors : bool
        If True, also download moneyflow + margin for the new trading days.
        Adds ~2-3 seconds per trading day. Default True.
    """
    from collector import Run
    from alpha_factors import AlphaFactorCollector

    qlib_dir = Path(qlib_data_1d_dir).expanduser().resolve()
    if not qlib_dir.exists():
        raise FileNotFoundError(f"qlib data directory not found: {qlib_dir}")

    trading_date = get_last_trading_date(qlib_dir)
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    logger.info(f"qlib data dir : {qlib_dir}")
    logger.info(f"last trading  : {trading_date}")
    logger.info(f"download range: {trading_date} -> {end_date}")

    if max_workers is None:
        max_workers = max(multiprocessing.cpu_count() - 2, 1)

    if source_dir is None:
        source_dir = str(CUR_DIR.joinpath("source_bwd"))
    if normalize_dir is None:
        normalize_dir = str(CUR_DIR.joinpath("normalize_bwd"))

    run = Run(
        source_dir=source_dir,
        normalize_dir=normalize_dir,
        max_workers=1,
        interval="1d",
    )

    # Step 1: Bulk-download new OHLCV + daily_basic
    logger.info("=" * 50)
    logger.info("Step 1/6: Bulk-downloading OHLCV + daily_basic from Tushare...")
    run.download_data_bulk(
        delay=delay,
        start=trading_date,
        end=end_date,
        listed_only=True,
    )

    # Step 2: Download alpha factors for new dates
    if include_alpha_factors:
        logger.info("=" * 50)
        logger.info("Step 2/6: Downloading alpha factors (moneyflow + margin)...")
        alpha = AlphaFactorCollector(source_dir=run.source_dir, delay=delay)
        try:
            alpha.download_moneyflow(start=trading_date, end=end_date)
        except Exception as e:
            logger.warning(f"Moneyflow download failed: {e}")
        try:
            alpha.download_margin(start=trading_date, end=end_date)
        except Exception as e:
            logger.warning(f"Margin download failed: {e}")
    else:
        logger.info("Step 2/6: Skipped (include_alpha_factors=False).")

    # Step 3: Sync calendar so normalize can see new dates
    logger.info("=" * 50)
    logger.info("Step 3/6: Syncing calendar from source data...")
    cal_path = qlib_dir / "calendars" / "day.txt"
    old_cal = pd.read_csv(cal_path, header=None)[0].tolist()
    old_last = old_cal[-1]
    src_dates = set()
    for f in run.source_dir.glob("*.csv"):
        try:
            df = pd.read_csv(f, usecols=["date"])
            for d in df["date"].dropna().unique():
                src_dates.add(str(d)[:10])
        except Exception:
            pass
    new_dates = sorted(d for d in src_dates if d > old_last)
    if new_dates:
        with open(cal_path, "w") as fout:
            for d in old_cal + new_dates:
                fout.write(d + "\n")
        logger.info(f"  Added {len(new_dates)} calendar dates: {new_dates[0]} ~ {new_dates[-1]}")
    else:
        logger.info("  No new calendar dates.")

    # Step 4: Normalize (extend mode)
    logger.info("=" * 50)
    logger.info("Step 4/6: Normalizing (extend mode)...")
    run.max_workers = max_workers
    run.normalize_data_1d_extend(str(qlib_dir))

    # Step 5: Dump to binary
    logger.info("=" * 50)
    logger.info("Step 5/6: Dumping to qlib binary format...")
    _dump = DumpDataUpdate(
        data_path=str(run.normalize_dir),
        qlib_dir=str(qlib_dir),
        exclude_fields="symbol,date",
        max_workers=max_workers,
    )
    _dump.dump()

    # Step 6: Refresh instrument end dates
    logger.info("=" * 50)
    logger.info("Step 6/6: Refreshing instrument end dates...")
    _refresh_instruments(qlib_dir)

    # (Optional) Update index weights
    if update_index_weights:
        logger.info("=" * 50)
        logger.info("Extra: Updating index constituent weights...")
        run.download_index_weights(
            index_code="000300.SH",
            freq=index_weight_freq,
            output_dir=str(CUR_DIR.joinpath("index_weights", "000300.SH")),
            delay=delay,
        )
        run.parse_index_instruments(
            index_code="000300.SH",
            qlib_dir=str(qlib_dir),
        )

    logger.info("=" * 50)
    logger.info("Daily update (bwd) completed successfully!")


if __name__ == "__main__":
    import fire

    fire.Fire(daily_update_bwd)

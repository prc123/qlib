"""
Daily update script for forward-adjusted (前复权) A-share data.

Same as ``daily_update.py`` but with:
  - Instrument end-date refresh after dump
  - Default data directory pointing to forward-adjusted qlib data
  - Optimized old-latest-map (only queries last 15 days, not full history)

Usage
-----
    $ python daily_update_fwd.py --qlib_data_1d_dir C:/Users/pp/.qlib/qlib_data/cn_data_fwd

Schedule (Windows Task Scheduler)
-----------------------------------
    Program: C:\\Users\\pp\\.conda\\envs\\qlib\\python.exe
    Arguments: daily_update_fwd.py --qlib_data_1d_dir C:\\Users\\pp\\.qlib\\qlib_data\\cn_data_fwd
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
    CUR_DIR.joinpath("daily_update_fwd.log"),
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

    # Update all.txt if needed
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

    # Always update other instrument files (they may lag behind all.txt)
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


def daily_update_fwd(
    qlib_data_1d_dir: str = r"C:\Users\pp\.qlib\qlib_data\cn_data_fwd",
    source_dir: str = None,
    normalize_dir: str = None,
    max_workers: int = None,
    delay: float = 0.3,
    update_index_weights: bool = False,
    index_weight_freq: str = "ME",
):
    """Run the daily data update pipeline for forward-adjusted data.

    Steps:
    1. Identify the last trading date in existing qlib data
    2. Bulk-download new stock & index data from Tushare by date
    3. Normalize (extend mode: align with existing data scales)
    4. Dump to qlib binary format
    5. Refresh instrument end dates to match calendar
    6. (Optional) Update index constituent weights
    """
    from collector import Run

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

    run = Run(
        source_dir=source_dir,
        normalize_dir=normalize_dir,
        max_workers=1,
        interval="1d",
    )

    # Step 1: Bulk-download new data
    logger.info("=" * 50)
    logger.info("Step 1/5: Bulk-downloading new data from Tushare...")
    run.download_data_bulk(
        delay=delay,
        start=trading_date,
        end=end_date,
        listed_only=True,
    )

    # Step 2: Normalize (extend mode)
    logger.info("=" * 50)
    logger.info("Step 2/5: Normalizing (extend mode)...")
    run.max_workers = max_workers
    run.normalize_data_1d_extend(str(qlib_dir))

    # Step 3: Dump to binary
    logger.info("=" * 50)
    logger.info("Step 3/5: Dumping to qlib binary format...")
    _dump = DumpDataUpdate(
        data_path=str(run.normalize_dir),
        qlib_dir=str(qlib_dir),
        exclude_fields="symbol,date",
        max_workers=max_workers,
    )
    _dump.dump()

    # Step 4: Refresh instrument end dates
    logger.info("=" * 50)
    logger.info("Step 4/5: Refreshing instrument end dates...")
    _refresh_instruments(qlib_dir)

    # Step 5: (Optional) Update index weights
    if update_index_weights:
        logger.info("=" * 50)
        logger.info("Step 5/5: Updating index constituent weights...")
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
    else:
        logger.info("=" * 50)
        logger.info("Step 5/5: Skipped (use --update_index_weights to enable).")

    logger.info("=" * 50)
    logger.info("Daily update (fwd) completed successfully!")


if __name__ == "__main__":
    import fire

    fire.Fire(daily_update_fwd)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Daily update script for A-share data using Tushare (bulk-by-date mode).

Reads the existing qlib data directory to find the last available trading date,
downloads new data since then using bulk-by-date queries (~15 API calls instead
of ~11000), normalizes it (aligned with existing data), and dumps to qlib binary
format.

Usage
-----
    $ python daily_update.py --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data

    # With optional index weight update
    $ python daily_update.py --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --update_index_weights

    # Custom source/normalize directories
    $ python daily_update.py --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \\
        --source_dir ./tushare_source --normalize_dir ./tushare_normalize

Schedule (Linux crontab)
-------------------------
    # Run every weekday at 18:00 (after market close)
    0 18 * * 1-5 cd /path/to/qlib/scripts/data_collector/tushare && python daily_update.py --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data >> /var/log/qlib_update.log 2>&1

Schedule (Windows Task Scheduler)
-----------------------------------
    Program: C:\\Users\\pp\\.conda\\envs\\qlib\\python.exe
    Arguments: daily_update.py --qlib_data_1d_dir C:\\Users\\pp\\.qlib\\qlib_data\\cn_data
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
    CUR_DIR.joinpath("daily_update.log"),
    rotation="30 days",
    retention="90 days",
    level="INFO",
)


def get_last_trading_date(qlib_data_dir: Path) -> str:
    """Read the last trading date from qlib calendar file.

    Returns
    -------
    str
        Date string in YYYY-MM-DD format.
    """
    calendar_path = qlib_data_dir.joinpath("calendars/day.txt")
    if not calendar_path.exists():
        raise FileNotFoundError(f"Calendar file not found: {calendar_path}")
    calendar_df = pd.read_csv(calendar_path)
    last_date = pd.Timestamp(calendar_df.iloc[-1, 0])
    return last_date.strftime("%Y-%m-%d")


def daily_update(
    qlib_data_1d_dir: str,
    source_dir: str = None,
    normalize_dir: str = None,
    max_workers: int = None,
    delay: float = 0.3,
    update_index_weights: bool = False,
    index_weight_freq: str = "ME",
):
    """Run the daily data update pipeline using bulk-by-date download.

    Steps:
    1. Identify the last trading date in existing qlib data
    2. Bulk-download new stock & index data from Tushare by date
    3. Normalize, aligned with existing data scales
    4. Dump to qlib binary format
    5. (Optional) Update index constituent weights

    Parameters
    ----------
    qlib_data_1d_dir : str
        Path to existing qlib 1d binary data directory.
    source_dir : str
        Directory for raw downloaded CSV files.
        Default: ``<tushare_dir>/source``
    normalize_dir : str
        Directory for normalized CSV files.
        Default: ``<tushare_dir>/normalize``
    max_workers : int
        Parallel workers for normalize and dump steps.
        Default: ``cpu_count - 2`` (min 1).
    delay : float
        API call delay in seconds, default 0.3 (~200 calls/min safe).
    update_index_weights : bool
        If True, also download the latest index constituent weights.
    index_weight_freq : str
        Frequency for index weight snapshots, default 'ME' (month-end).
    """
    from collector import Run

    qlib_dir = Path(qlib_data_1d_dir).expanduser().resolve()
    if not qlib_dir.exists():
        raise FileNotFoundError(f"qlib data directory not found: {qlib_dir}")

    # Determine last trading date from existing calendar
    trading_date = get_last_trading_date(qlib_dir)
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    logger.info(f"qlib data dir : {qlib_dir}")
    logger.info(f"last trading  : {trading_date}")
    logger.info(f"download range: {trading_date} -> {end_date}")

    if max_workers is None:
        max_workers = max(multiprocessing.cpu_count() - 2, 1)

    # Initialize the collector Run class
    run = Run(
        source_dir=source_dir,
        normalize_dir=normalize_dir,
        max_workers=1,
        interval="1d",
    )

    # Step 1: Bulk-download new data (listed_only for speed)
    logger.info("=" * 50)
    logger.info("Step 1/4: Bulk-downloading new data from Tushare...")
    run.download_data_bulk(
        delay=delay,
        start=trading_date,
        end=end_date,
        listed_only=True,
    )

    # Step 2: Normalize (extend mode: align with existing data scales)
    logger.info("=" * 50)
    logger.info("Step 2/4: Normalizing (extend mode)...")
    run.max_workers = max_workers
    run.normalize_data_1d_extend(str(qlib_dir))

    # Step 3: Dump to binary
    logger.info("=" * 50)
    logger.info("Step 3/4: Dumping to qlib binary format...")
    _dump = DumpDataUpdate(
        data_path=str(run.normalize_dir),
        qlib_dir=str(qlib_dir),
        exclude_fields="symbol,date",
        max_workers=max_workers,
    )
    _dump.dump()

    # Step 4: (Optional) Update index weights
    if update_index_weights:
        logger.info("=" * 50)
        logger.info("Step 4/4: Updating index constituent weights...")
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
        logger.info("Step 4/4: Skipped (use --update_index_weights to enable).")

    logger.info("=" * 50)
    logger.info("Daily update completed successfully!")


if __name__ == "__main__":
    import fire

    fire.Fire(daily_update)

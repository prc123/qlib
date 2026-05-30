# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Full historical download script for A-share data using Tushare (bulk-by-date mode).

Downloads ALL trading days in the specified range using bulk-by-date queries,
normalizes (fresh), and dumps to qlib binary format.  Includes full daily_basic
fields (market cap, turnover, PE, PB, etc.).

Usage
-----
    $ python full_download.py --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data_10y

    # Custom date range
    $ python full_download.py --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data_10y \\
        --start_date 2020-01-01 --end_date 2026-05-29

    # Custom source/normalize directories
    $ python full_download.py --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data_10y \\
        --source_dir ./source_10y --normalize_dir ./normalize_10y
"""

import sys
import multiprocessing
from pathlib import Path

import pandas as pd
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataAll

logger.add(
    CUR_DIR.joinpath("full_download.log"),
    rotation="30 days",
    retention="90 days",
    level="INFO",
)


def full_download(
    qlib_data_1d_dir: str,
    source_dir: str = None,
    normalize_dir: str = None,
    start_date: str = "2016-01-04",
    end_date: str = None,
    max_workers: int = None,
    delay: float = 0.3,
):
    """Run the full historical data download pipeline using bulk-by-date.

    Steps:
    1. Bulk-download all stock & index data from Tushare by date
    2. Normalize (fresh, no extend)
    3. Dump to qlib binary format

    Parameters
    ----------
    qlib_data_1d_dir : str
        Output path for qlib 1d binary data.
    source_dir : str
        Directory for raw downloaded CSV files.
        Default: ``<tushare_dir>/source``
    normalize_dir : str
        Directory for normalized CSV files.
        Default: ``<tushare_dir>/normalize``
    start_date : str
        Start of download range (YYYY-MM-DD), default 2016-01-04.
    end_date : str
        End of download range, default today.
    max_workers : int
        Parallel workers for normalize and dump steps.
        Default: ``cpu_count - 2`` (min 1).
    delay : float
        API call delay in seconds, default 0.3 (~200 calls/min safe).
    """
    from collector import Run

    qlib_dir = Path(qlib_data_1d_dir).expanduser().resolve()
    qlib_dir.mkdir(parents=True, exist_ok=True)

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    logger.info(f"output dir    : {qlib_dir}")
    logger.info(f"download range: {start_date} -> {end_date}")

    if max_workers is None:
        max_workers = max(multiprocessing.cpu_count() - 2, 1)

    # Initialize the collector Run class
    run = Run(
        source_dir=source_dir,
        normalize_dir=normalize_dir,
        max_workers=1,
        interval="1d",
    )

    # Step 1: Bulk-download all historical data
    logger.info("=" * 50)
    logger.info("Step 1/3: Bulk-downloading all historical data from Tushare...")
    logger.info("  (includes OHLCV + adj_factor + daily_basic)")
    run.download_data_bulk(
        delay=delay,
        start=start_date,
        end=end_date,
        listed_only=False,
    )

    # Step 2: Normalize (fresh mode)
    logger.info("=" * 50)
    logger.info("Step 2/3: Normalizing (fresh mode)...")
    run.max_workers = max_workers
    run.normalize_data()

    # Step 3: Dump to binary (fresh, not incremental)
    logger.info("=" * 50)
    logger.info("Step 3/3: Dumping to qlib binary format...")
    _dump = DumpDataAll(
        data_path=str(run.normalize_dir),
        qlib_dir=str(qlib_dir),
        exclude_fields="symbol,date",
        max_workers=max_workers,
    )
    _dump.dump()

    logger.info("=" * 50)
    logger.info("Full download completed successfully!")


if __name__ == "__main__":
    import fire

    fire.Fire(full_download)

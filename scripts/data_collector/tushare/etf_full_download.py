"""
Full download -> normalize -> dump pipeline for ETF data.

Uses Tushare bulk-by-date download to fetch OHLCV + adj_factor for all
Chinese ETFs, then normalizes and dumps to qlib binary format.

Output directory (default): ``C:/Users/pp/.qlib/qlib_data/etf_data``

Pipeline steps:
  1. Bulk-download all trading days via Tushare (fund_daily + fund_adj)
  2. Normalize each ETF: calendar-align, compute change, adjust prices
  3. Scale to first-close=1 (unitization for cross-ETF comparability)
  4. Dump to per-field .bin files under qlib_data_1d_dir

Usage
-----
    # Full download to default ETF data directory
    $ python etf_full_download.py

    # Custom output dir + date range
    $ python etf_full_download.py --qlib_data_1d_dir D:/qlib_data/etf --start_date 2020-01-01

    # Custom source/normalize dirs
    $ python etf_full_download.py --source_dir ./etf_source --normalize_dir ./etf_normalize

    # With exclude list (CSV containing 'symbol' column)
    $ python etf_full_download.py --exclude_list ./etf_exclude_list.csv
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
    CUR_DIR.joinpath("etf_full_download.log"),
    rotation="30 days",
    retention="90 days",
    level="INFO",
)


def full_download(
    qlib_data_1d_dir: str = r"C:\Users\pp\.qlib\qlib_data\etf_data",
    source_dir: str = None,
    normalize_dir: str = None,
    start_date: str = "2016-01-04",
    end_date: str = None,
    max_workers: int = None,
    delay: float = 0.3,
    exclude_list: str = None,
):
    """Run the full historical ETF data download pipeline.

    Steps:
    1. Bulk-download all ETF data from Tushare by date
    2. Normalize (fresh mode)
    3. Dump to qlib binary format
    4. Optional: filter excluded ETFs

    Parameters
    ----------
    qlib_data_1d_dir : str
        Output path for qlib 1d binary data.
        Default: ``C:/Users/pp/.qlib/qlib_data/etf_data``
    source_dir : str
        Directory for raw downloaded CSV files.
        Default: ``<tushare_dir>/etf_source``
    normalize_dir : str
        Directory for normalized CSV files.
        Default: ``<tushare_dir>/etf_normalize``
    start_date : str
        Start of download range (YYYY-MM-DD), default 2016-01-04.
    end_date : str
        End of download range, default today.
    max_workers : int
        Parallel workers for normalize and dump steps.
        Default: ``cpu_count - 2`` (min 1).
    delay : float
        API call delay in seconds, default 0.3.
    exclude_list : str
        Path to CSV file containing symbols to exclude (must have a 'symbol' column).
        If provided, these ETFs will be removed after dump.
    """
    from etf_collector import Run

    qlib_dir = Path(qlib_data_1d_dir).expanduser().resolve()
    qlib_dir.mkdir(parents=True, exist_ok=True)

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    logger.info(f"ETF output dir    : {qlib_dir}")
    logger.info(f"ETF download range: {start_date} -> {end_date}")

    if max_workers is None:
        max_workers = max(multiprocessing.cpu_count() - 2, 1)

    if source_dir is None:
        source_dir = str(CUR_DIR.joinpath("etf_source"))
    if normalize_dir is None:
        normalize_dir = str(CUR_DIR.joinpath("etf_normalize"))

    run = Run(
        source_dir=source_dir,
        normalize_dir=normalize_dir,
        max_workers=1,
        interval="1d",
    )

    # Step 1: Bulk-download all historical data
    logger.info("=" * 50)
    logger.info("Step 1/3: Bulk-downloading ETF data from Tushare ...")
    logger.info("  (fund_daily + fund_adj for all trading dates)")
    run.download_data_bulk(
        delay=delay,
        start=start_date,
        end=end_date,
    )

    # Step 2: Normalize (fresh mode)
    logger.info("=" * 50)
    logger.info("Step 2/3: Normalizing (fresh mode) ...")
    run.max_workers = max_workers
    run.normalize_data()

    # Step 3: Dump to binary
    logger.info("=" * 50)
    logger.info("Step 3/3: Dumping to qlib binary format ...")
    _dump = DumpDataAll(
        data_path=str(run.normalize_dir),
        qlib_dir=str(qlib_dir),
        exclude_fields="symbol,date",
        max_workers=max_workers,
    )
    _dump.dump()

    # Step 4: Optional filter by exclude list
    if exclude_list is not None and Path(exclude_list).exists():
        logger.info("=" * 50)
        logger.info("Step 4/4: Filtering excluded ETFs ...")
        from filter_etf_by_list import filter_etf_qlib_data
        filter_etf_qlib_data(
            qlib_data_dir=str(qlib_dir),
            exclude_list=exclude_list,
            backup=False,
        )

    logger.info("=" * 50)
    logger.info("ETF full download completed!")


if __name__ == "__main__":
    import fire
    fire.Fire(full_download)
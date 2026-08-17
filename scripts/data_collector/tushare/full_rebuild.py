"""
Full rebuild pipeline: OHLCV + alpha factors → normalize → dump_bin.

One-stop command to rebuild qlib binary data from scratch, including
moneyflow, margin, and northbound flow fields.

Usage
-----
    $ python full_rebuild.py --qlib_data_1d_dir C:/Users/pp/.qlib/qlib_data/cn_data_bwd

Steps
-----
  1. Bulk-download OHLCV + adj_factor + daily_basic (collector.py)
  2. Download moneyflow + margin (alpha_factors.py) — merge into source CSVs
  3. Download northbound flow (alpha_factors.py) — separate market-level CSV
  4. Normalize all source CSVs
  5. Dump to qlib binary format (.bin)

Notes
-----
  - Steps 1 & 2 are independent and can run in either order.
  - Alpha factors are merged per-file to avoid OOM (unlike the old
    download_data_bulk integration which loaded everything in memory).
  - Existing qlib data will be overwritten.
"""

import multiprocessing
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataAll

logger.add(
    CUR_DIR.joinpath("full_rebuild.log"),
    rotation="30 days",
    retention="90 days",
    level="INFO",
)


def full_rebuild(
    qlib_data_1d_dir: str,
    source_dir: str = None,
    normalize_dir: str = None,
    start_date: str = "2016-01-04",
    end_date: str = None,
    max_workers: int = None,
    delay: float = 0.3,
    include_alpha_factors: bool = True,
):
    """Full rebuild of qlib binary data with optional alpha factors.

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
        API call delay in seconds, default 0.3 (~200 calls/min).
    include_alpha_factors : bool
        If True, also download moneyflow + margin + northbound flow.
    """
    from collector import Run
    from alpha_factors import AlphaFactorCollector

    qlib_dir = Path(qlib_data_1d_dir).expanduser().resolve()
    qlib_dir.mkdir(parents=True, exist_ok=True)

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    logger.info(f"output dir    : {qlib_dir}")
    logger.info(f"download range: {start_date} -> {end_date}")
    logger.info(f"alpha factors : {include_alpha_factors}")

    if max_workers is None:
        max_workers = max(multiprocessing.cpu_count() - 2, 1)

    if source_dir is None:
        source_dir = str(CUR_DIR.joinpath("source"))
    if normalize_dir is None:
        normalize_dir = str(CUR_DIR.joinpath("normalize"))
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    run = Run(
        source_dir=str(source_dir),
        normalize_dir=str(normalize_dir),
        max_workers=1,
        interval="1d",
    )

    # ── Step 1: Download OHLCV + daily_basic ──
    logger.info("=" * 60)
    logger.info("Step 1/4: Bulk-downloading OHLCV + daily_basic from Tushare...")
    logger.info("  (approx 2500 trading days, 20-30 min)")
    run.download_data_bulk(
        delay=delay,
        start=start_date,
        end=end_date,
        listed_only=False,
    )

    # ── Step 2: Download alpha factors (moneyflow + margin + northbound) ──
    if include_alpha_factors:
        logger.info("=" * 60)
        logger.info("Step 2/4: Downloading alpha factors (moneyflow + margin + northbound)...")
        logger.info("  (approx 2500 API calls each for moneyflow & margin, 30-40 min each)")

        alpha = AlphaFactorCollector(source_dir=str(source_dir), delay=delay)

        logger.info("  --- moneyflow ---")
        alpha.download_moneyflow(start=start_date, end=end_date)

        logger.info("  --- margin ---")
        alpha.download_margin(start=start_date, end=end_date)

        logger.info("  --- northbound ---")
        alpha.download_northbound(start=start_date, end=end_date)
    else:
        logger.info("Step 2/4: Skipped (include_alpha_factors=False).")

    # ── Step 3: Normalize ──
    logger.info("=" * 60)
    logger.info("Step 3/4: Normalizing (fresh mode, {} workers)...", max_workers)
    run.max_workers = max_workers
    run.normalize_data()

    # ── Step 4: Dump to binary ──
    logger.info("=" * 60)
    logger.info("Step 4/4: Dumping to qlib binary format...")
    _dump = DumpDataAll(
        data_path=str(normalize_dir),
        qlib_dir=str(qlib_dir),
        exclude_fields="symbol,date",
        max_workers=max_workers,
    )
    _dump.dump()

    logger.info("=" * 60)
    logger.info("Full rebuild completed successfully!")
    logger.info(f"qlib data dir: {qlib_dir}")

    if include_alpha_factors:
        total_fields = 10  # 5 moneyflow + 5 margin
        logger.info(f"Alpha factor fields ({total_fields}) are now available in qlib.")
        logger.info("Use --use_alpha_factors when training: python daily_quant/gru/train.py --use_alpha_factors")


if __name__ == "__main__":
    import fire

    fire.Fire(full_rebuild)

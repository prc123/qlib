"""
Alpha factor data collectors: capital flow, margin trading, northbound flow.

Downloads per-stock moneyflow and margin data from Tushare and merges them
into the per-symbol source CSV files so they flow through normalize → dump_bin
and become available as qlib features. Northbound flow is market-level and
saved separately.

Usage
-----
    # Standalone: download all alpha factors
    $ python alpha_factors.py download_all --source_dir ./source_bwd

    # Individual downloads
    $ python alpha_factors.py download_moneyflow --source_dir ./source_bwd
    $ python alpha_factors.py download_margin --source_dir ./source_bwd
    $ python alpha_factors.py download_northbound --output_dir ./northbound
"""

import time
from pathlib import Path

import pandas as pd
import tushare as ts
from loguru import logger

# Progress file for background monitoring
_PROGRESS_FILE = Path(__file__).resolve().parent / "alpha_download_progress.txt"


def _progress(msg):
    """Log and write progress to file for background monitoring."""
    logger.info(msg)
    try:
        _PROGRESS_FILE.write_text(msg + "\n")
    except Exception:
        pass

TUSHARE_TOKEN = "a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202"

# Fields to keep from each Tushare endpoint (all others are dropped).
MONEYFLOW_FIELDS = [
    "buy_lg_amount",     # 大单买入金额
    "sell_lg_amount",    # 大单卖出金额
    "buy_elg_amount",    # 特大单买入金额
    "sell_elg_amount",   # 特大单卖出金额
    "net_mf_amount",     # 主力净流入金额
]

MARGIN_FIELDS = [
    "rzye",              # 融资余额
    "rqye",              # 融券余额
    "rzmre",             # 融资买入额
    "rzche",             # 融资偿还额
    "rqyl",              # 融券余量
]

NORTHBOUND_FIELDS = [
    "north_money",       # 北向资金（亿元）
    "south_money",       # 南向资金（亿元）
    "hgt",               # 沪股通成交净额
    "sgt",               # 深股通成交净额
]


def _norm_symbol(ts_code: str) -> str:
    """Normalize '000001.SZ' → 'sz000001'."""
    code, exchange = ts_code.split(".")
    return f"{exchange.lower()}{code}"


def _fetch_by_date(pro, fetch_fn, trade_dates, label, delay=0.3):
    """Generic bulk-by-date fetcher. Returns concatenated DataFrame."""
    all_frames = []
    for i, td in enumerate(trade_dates):
        try:
            df = fetch_fn(trade_date=td)
            if df is not None and not df.empty:
                all_frames.append(df)
        except Exception as e:
            logger.warning(f"  {label} {td} error: {e}")
        if (i + 1) % 50 == 0:
            _progress(f"  {label}: {i + 1}/{len(trade_dates)} dates done")
        time.sleep(delay)
    if not all_frames:
        logger.warning(f"{label}: no data fetched")
        return None
    result = pd.concat(all_frames, ignore_index=True)
    _progress(f"{label}: fetch done, {len(result)} rows, starting merge...")
    return result


def fetch_moneyflow_bulk(pro, trade_dates, delay=0.3):
    """Fetch per-stock moneyflow for all trading dates.

    Returns DataFrame with columns: ts_code, trade_date, + MONEYFLOW_FIELDS.
    """
    logger.info("Fetching moneyflow (capital flow) data...")
    df = _fetch_by_date(pro, pro.moneyflow, trade_dates, "moneyflow", delay)
    if df is None:
        return None
    keep = ["ts_code", "trade_date"] + [f for f in MONEYFLOW_FIELDS if f in df.columns]
    return df[keep]


def fetch_margin_bulk(pro, trade_dates, delay=0.3):
    """Fetch per-stock margin detail data for all trading dates.

    Uses pro.margin_detail (per-stock daily detail). Returns DataFrame with
    columns: ts_code, trade_date, + MARGIN_FIELDS.
    """
    logger.info("Fetching margin (融资融券) detail data...")
    df = _fetch_by_date(pro, pro.margin_detail, trade_dates, "margin_detail", delay)
    if df is None:
        return None
    keep = ["ts_code", "trade_date"] + [f for f in MARGIN_FIELDS if f in df.columns]
    return df[keep]


def fetch_northbound_flow(pro, start_date, end_date, delay=0.3):
    """Fetch market-level northbound/southbound flow.

    Returns DataFrame with: trade_date, north_money, south_money, hgt, sgt.
    One row per trading day.
    """
    logger.info("Fetching northbound flow (沪深港通) data...")
    try:
        df = pro.moneyflow_hsgt(start_date=start_date, end_date=end_date)
    except Exception as e:
        logger.error(f"moneyflow_hsgt error: {e}")
        return None
    if df is None or df.empty:
        logger.warning("moneyflow_hsgt: no data")
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    keep = ["trade_date"] + [f for f in NORTHBOUND_FIELDS if f in df.columns]
    return df[keep].sort_values("trade_date")


def merge_to_source_csvs(df, source_dir, fields, label):
    """Merge new fields into per-symbol source CSV files.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: ts_code, trade_date, + fields to merge.
    source_dir : Path
        Directory containing per-symbol CSVs like ``sz000001.csv``.
    fields : list[str]
        Column names to merge.
    label : str
        Human-readable label for logging.
    """
    if df is None or df.empty:
        return

    source_dir = Path(source_dir)
    df["symbol"] = df["ts_code"].apply(_norm_symbol)
    df["date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

    symbols = df["symbol"].unique()
    updated = 0
    skipped = 0
    for sym in symbols:
        sym_path = source_dir / f"{sym}.csv"
        if not sym_path.exists():
            continue
        if sym_path.stat().st_size == 0:
            logger.warning(f"  skipping empty file: {sym_path.name}")
            skipped += 1
            continue
        sym_df = df[df["symbol"] == sym][["date"] + fields].copy()
        if sym_df.empty:
            continue
        try:
            existing = pd.read_csv(sym_path)
            sym_df["date"] = sym_df["date"].astype(str)
            existing["date"] = existing["date"].astype(str)
            for f in fields:
                if f in existing.columns:
                    existing = existing.drop(columns=[f])
            merged = existing.merge(sym_df, on="date", how="left")
            merged.to_csv(sym_path, index=False)
            updated += 1
        except Exception as e:
            logger.warning(f"  failed to merge {sym_path.name}: {e}")
            skipped += 1

    msg = f"Merged {label} ({len(fields)} fields) into {updated} symbol CSVs"
    if skipped:
        msg += f", skipped {skipped}"
    _progress(msg)


class AlphaFactorCollector:
    """Standalone CLI for downloading alpha factor data."""

    def __init__(self, source_dir=None, output_dir=None, delay=0.3):
        self.pro = ts.pro_api(TUSHARE_TOKEN)
        self.source_dir = Path(source_dir) if source_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.delay = delay

    def _get_trade_dates(self, start, end):
        """Get sorted list of trading dates in range."""
        cal = self.pro.trade_cal(
            exchange="SSE",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
        )
        cal = cal[cal["is_open"] == 1]
        return sorted(cal["cal_date"].tolist())

    def download_moneyflow(self, start="2016-01-01", end=None):
        """Download moneyflow data and merge into source CSVs."""
        if end is None:
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
        trade_dates = self._get_trade_dates(start, end)
        _progress(f"=== MONEYFLOW START: {len(trade_dates)} dates ({trade_dates[0]} -> {trade_dates[-1]}) ===")
        df = fetch_moneyflow_bulk(self.pro, trade_dates, self.delay)
        merge_to_source_csvs(df, self.source_dir, MONEYFLOW_FIELDS, "moneyflow")
        _progress("=== MONEYFLOW DONE ===")

    def download_margin(self, start="2016-01-01", end=None):
        """Download margin data and merge into source CSVs."""
        if end is None:
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
        trade_dates = self._get_trade_dates(start, end)
        _progress(f"=== MARGIN START: {len(trade_dates)} dates ({trade_dates[0]} -> {trade_dates[-1]}) ===")
        df = fetch_margin_bulk(self.pro, trade_dates, self.delay)
        merge_to_source_csvs(df, self.source_dir, MARGIN_FIELDS, "margin")
        _progress("=== MARGIN DONE ===")

    def download_northbound(self, start="2016-01-01", end=None):
        """Download northbound flow data and save to output_dir."""
        if end is None:
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
        if self.output_dir is None:
            self.output_dir = Path(__file__).resolve().parent / "northbound"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        start_fmt = start.replace("-", "")
        end_fmt = end.replace("-", "")
        _progress("=== NORTHBOUND START ===")
        df = fetch_northbound_flow(self.pro, start_fmt, end_fmt, self.delay)
        if df is not None:
            out_path = self.output_dir / "northbound_flow.csv"
            if out_path.exists():
                existing = pd.read_csv(out_path)
                existing["trade_date"] = pd.to_datetime(existing["trade_date"])
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["trade_date"])
            df = df.sort_values("trade_date")
            df.to_csv(out_path, index=False)
            _progress(f"Saved northbound flow: {len(df)} rows -> {out_path}")
        _progress("=== NORTHBOUND DONE ===")

    def download_all(self, start="2016-01-01", end=None):
        """Download all alpha factors."""
        if end is None:
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
        _progress("=== ALPHA DOWNLOAD START ===")
        self.download_moneyflow(start, end)
        self.download_margin(start, end)
        self.download_northbound(start, end)
        _progress("=== ALPHA DOWNLOAD ALL DONE ===")


if __name__ == "__main__":
    import fire
    fire.Fire(AlphaFactorCollector)

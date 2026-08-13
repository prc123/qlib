# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
ETF data collector using Tushare Pro API.

Downloads daily OHLCV + adj_factor for all Chinese ETFs and outputs
qlib-compatible binary data.

APIs used
---------
- fund_basic(market='E')  → ETF list (ts_code, name, status, etc.)
- fund_daily()            → daily OHLCV
- fund_adj()              → adjustment factor (复权因子)

Usage
-----
    # Full download
    $ python etf_collector.py download_data --source_dir ./etf_source --start 2016-01-01

    # Normalize
    $ python etf_collector.py normalize_data --source_dir ./etf_source --normalize_dir ./etf_normalize
"""

import abc
import importlib
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tushare as ts
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataAll, DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import get_calendar_list

TUSHARE_TOKEN = "a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202"


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class ETFCollectorCN1d(BaseCollector):
    """Tushare ETF 1d data collector.

    Collects daily OHLCV + adj_factor for all Chinese ETFs (listed +
    delisted) via the Tushare Pro API.  Output format matches the stock
    collector so downstream normalization and ``dump_bin`` work unchanged.
    """

    retry = 5

    def __init__(
        self,
        save_dir,
        start=None,
        end=None,
        interval="1d",
        max_workers=1,
        max_collector_count=2,
        delay=0,
        check_data_length=None,
        limit_nums=None,
        listed_only=False,
    ):
        self.pro = ts.pro_api(TUSHARE_TOKEN)
        self.listed_only = listed_only
        super().__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    # ---- instrument list ---------------------------------------------------

    def get_instrument_list(self):
        """Get all ETF ts_codes from Tushare fund_basic (market='E')."""
        logger.info("Fetching ETF instrument list via fund_basic(market='E') ...")
        try:
            df = self.pro.fund_basic(market="E")
        except Exception as e:
            logger.error(f"fund_basic failed: {e}")
            return []

        if df is None or df.empty:
            logger.warning("fund_basic returned empty")
            return []

        if self.listed_only:
            df = df[df["list_status"] == "L"]

        codes = sorted(df["ts_code"].unique().tolist())
        logger.info(f"ETF instrument list: {len(codes)} symbols")
        return codes

    def normalize_symbol(self, symbol: str):
        """Convert Tushare ts_code (e.g. 510050.SH) to qlib symbol (sh510050)."""
        code, exchange = symbol.split(".")
        return f"{exchange.lower()}{code}"

    # ---- data fetching -----------------------------------------------------

    def _get_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch fund_daily for a single ETF with retry."""
        for _ in range(self.retry):
            try:
                df = self.pro.fund_daily(ts_code=symbol, start_date=start, end_date=end)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"fund_daily {symbol} attempt {_+1}: {e}")
                time.sleep(1)
        return pd.DataFrame()

    def _get_adj(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch fund_adj for a single ETF with retry."""
        for _ in range(self.retry):
            try:
                df = self.pro.fund_adj(ts_code=symbol, start_date=start, end_date=end)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"fund_adj {symbol} attempt {_+1}: {e}")
                time.sleep(1)
        return pd.DataFrame()

    def get_data(self, symbol, interval, start_datetime, end_datetime):
        """Get daily OHLCV + adjclose for a single ETF.

        Returns
        -------
        pd.DataFrame
            Columns: [symbol, date, open, high, low, close, volume, adjclose]
        """
        start_date = start_datetime.strftime("%Y%m%d")
        end_date = end_datetime.strftime("%Y%m%d")

        self.sleep()

        # 1) OHLCV
        daily = self._get_daily(symbol, start_date, end_date)
        if daily is None or daily.empty:
            logger.debug(f"  {symbol}: no daily data in [{start_date}, {end_date}]")
            return pd.DataFrame()

        daily["trade_date"] = pd.to_datetime(daily["trade_date"])

        # 2) adj_factor
        adj = self._get_adj(symbol, start_date, end_date)
        if adj is not None and not adj.empty:
            adj["trade_date"] = pd.to_datetime(adj["trade_date"])
            daily = daily.merge(
                adj[["ts_code", "trade_date", "adj_factor"]],
                on=["ts_code", "trade_date"],
                how="left",
            )
            daily["adj_factor"] = daily["adj_factor"].ffill()
            daily["adj_factor"] = daily["adj_factor"].fillna(1.0)
        else:
            daily["adj_factor"] = 1.0

        # 3) Build output
        daily = daily.rename(columns={"trade_date": "date", "vol": "volume"})
        daily["adjclose"] = daily["close"] * daily["adj_factor"]
        daily["symbol"] = symbol
        daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")

        return daily[["symbol", "date", "open", "high", "low", "close", "volume", "adjclose"]]

    # ---- CLI wrappers ------------------------------------------------------

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
        listed_only=False,
    ):
        """Download ETF daily data from Tushare (per-symbol).

        Examples
        --------
        $ python etf_collector.py download_data --source_dir ./etf_source --delay 0.5
        """
        super().download_data(max_collector_count, delay, start, end, check_data_length, limit_nums,
                              listed_only=listed_only)

    def download_data_bulk(
        self,
        start=None,
        end=None,
        delay=0.3,
    ):
        """Download ETF data using bulk-by-date approach.

        Fetches ALL ETFs at once for each trading date via
        ``pro.fund_daily(trade_date=date)`` and ``pro.fund_adj(trade_date=date)``.
        Much faster than per-symbol for short date ranges (daily updates).

        Parameters
        ----------
        start : str
            Start date (YYYY-MM-DD).
        end : str
            End date (YYYY-MM-DD).
        delay : float
            Sleep between API calls (seconds).
        """
        pro = ts.pro_api(TUSHARE_TOKEN)

        _start = start or "20160101"
        _end = end or pd.Timestamp.now().strftime("%Y%m%d")
        cal_df = pro.trade_cal(exchange="SSE", start_date=_start.replace("-", ""),
                               end_date=_end.replace("-", ""))
        cal_df = cal_df[cal_df["is_open"] == 1]
        trade_dates = sorted(cal_df["cal_date"].tolist())

        if not trade_dates:
            logger.warning("No trading dates found")
            return

        logger.info(f"Bulk ETF download: {len(trade_dates)} days ({trade_dates[0]} → {trade_dates[-1]})")

        source_dir = Path(self.source_dir)
        source_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: fund_daily per date
        all_daily = []
        for i, td in enumerate(trade_dates):
            try:
                df = pro.fund_daily(trade_date=td)
                if df is not None and not df.empty:
                    all_daily.append(df)
            except Exception as e:
                logger.warning(f"  fund_daily {td} error: {e}")
            if (i + 1) % 200 == 0:
                logger.info(f"  fund_daily: {i+1}/{len(trade_dates)} done")
            time.sleep(delay)

        if not all_daily:
            logger.error("No daily data fetched")
            return

        daily_df = pd.concat(all_daily, ignore_index=True)
        logger.info(f"fund_daily rows: {len(daily_df)}, ETFs: {daily_df['ts_code'].nunique()}")

        # Phase 2: fund_adj per date
        all_adj = []
        for i, td in enumerate(trade_dates):
            try:
                df = pro.fund_adj(trade_date=td)
                if df is not None and not df.empty:
                    all_adj.append(df)
            except Exception as e:
                logger.warning(f"  fund_adj {td} error: {e}")
            if (i + 1) % 200 == 0:
                logger.info(f"  fund_adj: {i+1}/{len(trade_dates)} done")
            time.sleep(delay)

        adj_df = pd.concat(all_adj, ignore_index=True) if all_adj else pd.DataFrame()
        if not adj_df.empty:
            logger.info(f"fund_adj rows: {len(adj_df)}, ETFs: {adj_df['ts_code'].nunique()}")
        else:
            logger.warning("No adj_factor data fetched")

        # Merge & produce output
        daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"])
        if not adj_df.empty:
            adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"])
            merged = daily_df.merge(
                adj_df[["ts_code", "trade_date", "adj_factor"]],
                on=["ts_code", "trade_date"], how="left",
            )
            merged["adj_factor"] = merged["adj_factor"].fillna(1.0)
        else:
            merged = daily_df.copy()
            merged["adj_factor"] = 1.0

        merged["adjclose"] = merged["close"] * merged["adj_factor"]

        def _norm(ts_code):
            code, exchange = ts_code.split(".")
            return f"{exchange.lower()}{code}"

        merged["symbol"] = merged["ts_code"].apply(_norm)
        merged["date"] = pd.to_datetime(merged["trade_date"]).dt.strftime("%Y-%m-%d")
        output = merged[["symbol", "date", "open", "high", "low", "close", "vol", "adjclose"]]
        output = output.rename(columns={"vol": "volume"})
        output = output.dropna(subset=["adjclose"])
        output = output.sort_values(["symbol", "date"])

        # Save per-symbol CSV
        symbols = output["symbol"].unique()
        logger.info(f"Saving {len(symbols)} ETF files to {source_dir} ...")
        for i, sym in enumerate(symbols):
            sym_df = output[output["symbol"] == sym].sort_values("date")
            dest = source_dir / f"{sym}.csv"
            if dest.exists():
                existing = pd.read_csv(dest)
                combined = pd.concat([existing, sym_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["symbol", "date"])
                combined.to_csv(dest, index=False)
            else:
                sym_df.to_csv(dest, index=False)
            if (i + 1) % 200 == 0:
                logger.info(f"  {i+1}/{len(symbols)} ...")

        logger.info("Bulk ETF download completed.")

    def normalize_data(self, date_field_name="date", symbol_field_name="symbol", end_date=None, **kwargs):
        """Normalize raw ETF CSV data to qlib format.

        Examples
        --------
        $ python etf_collector.py normalize_data --source_dir ./etf_source --normalize_dir ./etf_normalize
        """
        super().normalize_data(date_field_name, symbol_field_name, end_date=end_date, **kwargs)

    def normalize_data_1d_extend(self, old_qlib_data_dir, date_field_name="date", symbol_field_name="symbol"):
        """Extend existing qlib ETF data with new normalized data.

        Examples
        --------
        $ python etf_collector.py normalize_data_1d_extend --old_qlib_data_dir ~/.qlib/qlib_data/etf_data --source_dir ./etf_source --normalize_dir ./etf_normalize
        """
        _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            old_qlib_data_dir=old_qlib_data_dir,
            norm_dir=str(self.normalize_dir),
        )
        yc.normalize()

    def update_data_to_bin(
        self,
        qlib_data_1d_dir,
        end_date=None,
        check_data_length=None,
        delay=1,
        exists_skip=False,
    ):
        """End-to-end daily update: download + normalize + dump_bin.

        Examples
        --------
        $ python etf_collector.py update_data_to_bin --qlib_data_1d_dir ~/.qlib/qlib_data/etf_data
        """
        import multiprocessing

        if self.interval.lower() != "1d":
            logger.warning("currently only supports 1d data updates")

        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        qlib_path = Path(qlib_data_1d_dir)

        if not qlib_path.joinpath("calendars", "day.txt").exists():
            logger.warning(f"qlib data not found at {qlib_data_1d_dir}. Run full download first.")
            return

        calendar_df = pd.read_csv(qlib_path / "calendars" / "day.txt")
        trading_date = (pd.Timestamp(calendar_df.iloc[-1, 0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if end_date is None:
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # Use bulk-by-date for daily update (only ~1-5 trading days)
        self.download_data_bulk(delay=delay, start=trading_date, end=end_date)

        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )

        self.normalize_data_1d_extend(qlib_data_1d_dir)

        _dump = DumpDataUpdate(
            data_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        logger.info("ETF daily update completed.")


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

class ETFNormalizeCN1d(BaseNormalize):
    """Normalize ETF 1d data to qlib-compatible format.

    Produces the same output columns as the stock normalizer:
    date, symbol, open, high, low, close, volume, factor, change
    with all price fields scaled to first close = 1.
    """

    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        qlib_dir = self.kwargs.get("old_qlib_data_dir", "")
        if qlib_dir:
            cal_path = Path(qlib_dir) / "calendars" / "day.txt"
            if cal_path.exists():
                cal_df = pd.read_csv(cal_path, header=None)
                return sorted(pd.to_datetime(cal_df.iloc[:, 0]).tolist())
        return get_calendar_list("ALL")

    @staticmethod
    def _calc_change(df, last_close=None):
        tmp_series = df["close"].ffill()
        tmp_shift = tmp_series.shift(1)
        if last_close is not None:
            tmp_shift.iloc[0] = float(last_close)
        return tmp_series / tmp_shift - 1

    def _adjusted_price(self, df):
        """Adjust OHLCV using factor = adjclose / close."""
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        if "adjclose" in df:
            df["factor"] = df["adjclose"] / df["close"]
            df["factor"] = df["factor"].ffill()
        else:
            df["factor"] = 1.0
        for _col in self.COLUMNS:
            if _col not in df.columns:
                continue
            if _col == "volume":
                continue
            else:
                df[_col] = df[_col] * df["factor"]
        df.index.names = [self._date_field_name]
        return df.reset_index()

    def _manual_adj_data(self, df):
        """Scale all price and volume fields so first valid close = 1."""
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)

        valid = df.loc[df["close"].first_valid_index():]
        if valid.empty:
            return df.reset_index()
        _close = valid["close"].iloc[0]

        numeric_adj_cols = set(self.COLUMNS + ["factor"]) & set(df.columns)
        for _col in numeric_adj_cols:
            if _col == "volume":
                df[_col] = df[_col] * _close
            else:
                df[_col] = df[_col] / _close
        return df.reset_index()

    def normalize(self, df):
        """Run full normalization pipeline on raw ETF data."""
        if df.empty:
            return df

        symbol = df.loc[df[self._symbol_field_name].first_valid_index(), self._symbol_field_name]
        columns = list(self.COLUMNS)
        df = df.copy()

        df.set_index(self._date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index, format="mixed")
        df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="first")]

        if self._calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=self._calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date():pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )

        df.sort_index(inplace=True)

        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {self._symbol_field_name})] = np.nan

        df["change"] = self._calc_change(df)
        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        df[self._symbol_field_name] = symbol
        df.index.names = [self._date_field_name]
        df = df.reset_index()

        df = self._adjusted_price(df)
        df = self._manual_adj_data(df)

        return df


class ETFNormalizeCN1dExtend(ETFNormalizeCN1d):
    """Extended normalize for appending new ETF data to existing qlib data."""

    def __init__(self, old_qlib_data_dir, date_field_name="date", symbol_field_name="symbol",
                 norm_dir=None, **kwargs):
        super().__init__(date_field_name, symbol_field_name, old_qlib_data_dir=old_qlib_data_dir, **kwargs)
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
        self._old_latest = self._build_old_latest_map(old_qlib_data_dir)
        self._norm_dir = Path(norm_dir) if norm_dir else None

    def _build_old_latest_map(self, qlib_data_dir):
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        import qlib
        from qlib.data import D
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)

        cal = pd.read_csv(Path(qlib_data_dir) / "calendars" / "day.txt")
        last_dates = cal.iloc[-15:, 0].tolist()
        start_str = str(last_dates[0])
        end_str = str(last_dates[-1])

        df = D.features(
            D.instruments("all"),
            ["$" + col for col in self.column_list],
            start_time=start_str,
            end_time=end_str,
        )
        df.columns = self.column_list

        result = {}
        for instrument, group in df.groupby(level="instrument"):
            if group.empty:
                continue
            last_date = group.index[-1][1]
            row = group.iloc[-1]
            result[instrument] = (last_date, {c: row[c] for c in self.column_list[:-1]})
        logger.info(f"Built old-latest map for {len(result)} ETF symbols")
        return result

    def normalize(self, df):
        if df is None or df.empty:
            return None
        symbol_name = str(df[self._symbol_field_name].iloc[0]).upper()
        entry = self._old_latest.get(symbol_name)
        if entry is not None:
            latest_date, _ = entry
            df[self._date_field_name] = pd.to_datetime(df[self._date_field_name], format="mixed")
            df = df[df[self._date_field_name] >= pd.Timestamp(latest_date)]
            if df.empty:
                return None
        df = super().normalize(df)
        if df is None or df.empty:
            return None
        df.set_index(self._date_field_name, inplace=True)
        if entry is None:
            return df.reset_index()
        if df.empty:
            return None
        latest_date, old_latest_data = entry
        new_latest_data = df.iloc[0]
        for col in self.column_list[:-1]:
            if col == "volume":
                df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
            else:
                df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
        new_rows = df.drop(df.index[0]).reset_index()

        if self._norm_dir is not None:
            sym_lower = symbol_name.lower()
            existing_file = self._norm_dir / f"{sym_lower}.csv"
            if existing_file.exists():
                existing = pd.read_csv(existing_file)
                existing[self._date_field_name] = pd.to_datetime(
                    existing[self._date_field_name], format="mixed")
                new_rows[self._date_field_name] = pd.to_datetime(
                    new_rows[self._date_field_name], format="mixed")
                combined = pd.concat([existing, new_rows], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=[self._symbol_field_name, self._date_field_name])
                return combined
            else:
                logger.info(f"merge {symbol_name}: no existing file, returning new_rows only")

        return new_rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

class Run(BaseRun):
    """CLI entry point for ETF data collection.

    Examples
    --------
    # Download raw data
    $ python etf_collector.py download_data --source_dir ./etf_source --delay 0.5

    # Bulk download (all dates at once)
    $ python etf_collector.py download_data_bulk --source_dir ./etf_source --start 2020-01-01 --delay 0.3

    # Normalize
    $ python etf_collector.py normalize_data --source_dir ./etf_source --normalize_dir ./etf_normalize

    # Daily update
    $ python etf_collector.py update_data_to_bin --qlib_data_1d_dir ~/.qlib/qlib_data/etf_data
    """

    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self._cur_module = importlib.import_module("etf_collector")

    @property
    def collector_class_name(self):
        return f"ETFCollectorCN{self.interval}"

    @property
    def normalize_class_name(self):
        return f"ETFNormalizeCN{self.interval}"

    @property
    def default_base_dir(self):
        return CUR_DIR

    def download_data_bulk(self, start=None, end=None, delay=0.3):
        """Bulk-by-date ETF download (proxies to ETFCollectorCN1d)."""
        _class = getattr(self._cur_module, self.collector_class_name)
        collector = _class(
            save_dir=self.source_dir,
            start=start,
            end=end,
            interval=self.interval,
            delay=delay,
        )
        collector.source_dir = self.source_dir
        collector.download_data_bulk(start=start, end=end, delay=delay)

    def update_data_to_bin(
        self,
        qlib_data_1d_dir,
        end_date=None,
        check_data_length=None,
        delay=1,
        exists_skip=False,
    ):
        """End-to-end daily update (proxies to ETFCollectorCN1d)."""
        _class = getattr(self._cur_module, self.collector_class_name)
        collector = _class(
            save_dir=self.source_dir,
            interval=self.interval,
            delay=delay,
        )
        collector.source_dir = self.source_dir
        collector.normalize_dir = self.normalize_dir
        collector.max_workers = self.max_workers
        collector.update_data_to_bin(
            qlib_data_1d_dir=qlib_data_1d_dir,
            end_date=end_date,
            check_data_length=check_data_length,
            delay=delay,
            exists_skip=exists_skip,
        )


if __name__ == "__main__":
    import fire
    fire.Fire(Run)

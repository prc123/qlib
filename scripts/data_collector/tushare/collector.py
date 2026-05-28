# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import importlib
from pathlib import Path
from typing import Iterable

import fire
import numpy as np
import pandas as pd
import tushare as ts
from loguru import logger

import qlib
from qlib.data import D
from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import deco_retry, get_calendar_list

# Tushare API token
TUSHARE_TOKEN = "a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202"


class TushareCollectorCN1d(BaseCollector):
    """Tushare A-share 1d data collector.

    Collects daily OHLCV data for all A-share stocks (including delisted)
    via the Tushare Pro API. Output format is compatible with the Yahoo
    collector so downstream normalization and dump_bin work unchanged.

    Notes
    -----
    - Listed + delisted + suspended stocks are all collected.
    - adj_factor from Tushare is merged to calculate adjclose.
    - CSI300/CSI500/CSI100 index data is also downloaded.
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
        # Must init pro before super().__init__ because get_instrument_list()
        # is called during BaseCollector.__init__().
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

    def get_instrument_list(self):
        """Get A-share stock symbols.

        When ``listed_only=True`` (daily update mode), only currently listed
        stocks are returned.  Otherwise listed, delisted, and paused stocks
        are all included.

        Returns
        -------
        list
            Sorted unique ts_code strings like ['000001.SZ', '600000.SH', ...]
        """
        if self.listed_only:
            logger.info("get A-share stock symbols (listed only)...")
            statuses = [("L", "listed")]
        else:
            logger.info("get A-share stock symbols (listed + delisted)...")
            statuses = [("L", "listed"), ("D", "delisted"), ("P", "paused")]

        symbols = []
        for status, label in statuses:
            try:
                df = self.pro.stock_basic(
                    exchange="",
                    list_status=status,
                    fields="ts_code,symbol,name,list_date,delist_date",
                )
                if df is not None and not df.empty:
                    symbols.extend(df["ts_code"].tolist())
                    logger.info(f"  {label} ({status}): {len(df)} stocks")
            except Exception as e:
                logger.warning(f"Failed to get {label} stocks: {e}")

        symbols = sorted(set(symbols))
        logger.info(f"total {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        """Normalize Tushare symbol to qlib format.

        '000001.SZ' -> 'sz000001'
        '600000.SH' -> 'sh600000'
        """
        code, exchange = symbol.split(".")
        if exchange.upper() == "SH":
            return f"sh{code}"
        elif exchange.upper() == "SZ":
            return f"sz{code}"
        elif exchange.upper() == "BJ":
            return f"bj{code}"
        else:
            return f"{exchange.lower()}{code}"

    @deco_retry(retry=5, retry_sleep=3)
    def _get_daily_data(self, symbol, start_date, end_date):
        """Fetch daily OHLCV data from Tushare with retry."""
        df = self.pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            raise ValueError(f"get data error: {symbol}--{start_date}--{end_date}")
        return df

    @deco_retry(retry=3, retry_sleep=2)
    def _get_adj_factor(self, symbol, start_date, end_date):
        """Fetch adjustment factor from Tushare with retry."""
        adj_df = self.pro.adj_factor(ts_code=symbol, start_date=start_date, end_date=end_date)
        return adj_df

    def get_data(self, symbol, interval, start_datetime, end_datetime):
        """Get daily data for a single symbol from Tushare.

        Parameters
        ----------
        symbol : str
            Tushare ts_code, e.g. '000001.SZ'
        interval : str
            Data frequency, only '1d' is supported.
        start_datetime : pd.Timestamp
            Start date (inclusive).
        end_datetime : pd.Timestamp
            End date (exclusive, or inclusive depending on Tushare).

        Returns
        -------
        pd.DataFrame
            Columns: [symbol, date, open, high, low, close, volume, adjclose]
        """
        start_date = start_datetime.strftime("%Y%m%d")
        end_date = end_datetime.strftime("%Y%m%d")

        self.sleep()

        # Fetch daily OHLCV
        try:
            df = self._get_daily_data(symbol, start_date, end_date)
        except ValueError:
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # Fetch and merge adjustment factor
        try:
            adj_df = self._get_adj_factor(symbol, start_date, end_date)
            if adj_df is not None and not adj_df.empty:
                adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"])
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.merge(
                    adj_df[["trade_date", "adj_factor"]],
                    on="trade_date",
                    how="left",
                )
                df["adj_factor"] = df["adj_factor"].ffill()
                df["adj_factor"] = df["adj_factor"].fillna(1.0)
            else:
                df["adj_factor"] = 1.0
        except Exception:
            df["adj_factor"] = 1.0

        # Rename to standard column names
        df = df.rename(
            columns={
                "trade_date": "date",
                "vol": "volume",
            }
        )

        # Calculate adjclose from adj_factor
        df["adjclose"] = df["close"] * df["adj_factor"]

        # Add normalized symbol
        df["symbol"] = symbol

        # Convert date to date-only string to avoid time components in CSV
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        return df[["symbol", "date", "open", "high", "low", "close", "volume", "adjclose"]]

    def collector_data(self):
        """Collect data for all instruments and download index data."""
        super().collector_data()
        self.download_index_data()

    def download_index_data(self):
        """Download CSI300/CSI500/CSI100 index daily data using Tushare.

        Output format matches Yahoo collector: date, open, close, high, low,
        volume, money, change, adjclose, symbol.
        """
        _format = "%Y%m%d"
        _begin = self.start_datetime.strftime(_format)
        _end = self.end_datetime.strftime(_format)

        index_map = {
            "csi300": ("000300.SH", "sh000300"),
            "csi100": ("000903.SH", "sh000903"),
            "csi500": ("000905.SH", "sh000905"),
        }

        for _index_name, (_index_code, _save_name) in index_map.items():
            logger.info(f"get bench data: {_index_name}({_index_code})......")
            try:
                df = self.pro.index_daily(ts_code=_index_code, start_date=_begin, end_date=_end)
                if df is not None and not df.empty:
                    df = df.rename(
                        columns={
                            "trade_date": "date",
                            "vol": "volume",
                        }
                    )
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                    # Map 'change' in Tushare to 'pct_chg' meaning (it's actually
                    # price change, not percentage; Yahoo uses 'change' too).
                    # Keep only columns matching Yahoo index format.
                    if "amount" in df.columns:
                        df["money"] = df["amount"]
                    else:
                        df["money"] = np.nan
                    df["adjclose"] = df["close"]
                    df["symbol"] = _save_name
                    df = df.astype(float, errors="ignore")

                    # Match Yahoo collector column order
                    columns = ["date", "open", "close", "high", "low", "volume", "money", "change", "adjclose", "symbol"]
                    available = [c for c in columns if c in df.columns]
                    df = df[available]

                    _path = self.save_dir.joinpath(f"{_save_name}.csv")
                    if _path.exists():
                        _old_df = pd.read_csv(_path)
                        df = pd.concat([_old_df, df], sort=False)
                    df.to_csv(_path, index=False)
            except Exception as e:
                logger.warning(f"get {_index_name} error: {e}")
            time.sleep(1)


class TushareNormalizeCN1d(BaseNormalize):
    """Normalize Tushare 1d data to qlib-compatible format.

    Produces the same output columns as YahooNormalizeCN1d:
    date, symbol, open, high, low, close, volume, factor, change
    with all price fields scaled to first close = 1.
    """

    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("ALL")

    @staticmethod
    def _calc_change(df, last_close=None):
        """Calculate daily change: close / prev_close - 1."""
        tmp_series = df["close"].ffill()
        tmp_shift = tmp_series.shift(1)
        if last_close is not None:
            tmp_shift.iloc[0] = float(last_close)
        return tmp_series / tmp_shift - 1

    def _adjusted_price(self, df):
        """Adjust OHLCV prices using factor = adjclose / close."""
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
                df[_col] = df[_col] / df["factor"]
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

        # Only adjust known numeric fields (open, high, low, close, volume, factor)
        numeric_adj_cols = set(self.COLUMNS + ["factor"]) & set(df.columns)
        for _col in numeric_adj_cols:
            if _col == "volume":
                df[_col] = df[_col] * _close
            else:
                df[_col] = df[_col] / _close
        return df.reset_index()

    def normalize(self, df):
        """Run full normalization pipeline on raw Tushare data."""
        if df.empty:
            return df

        symbol = df.loc[df[self._symbol_field_name].first_valid_index(), self._symbol_field_name]
        columns = list(self.COLUMNS)
        df = df.copy()

        # Clean index
        df.set_index(self._date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index, format="mixed")
        df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="first")]

        # Reindex against calendar
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

        # Mark rows with invalid volume as NaN
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {self._symbol_field_name})] = np.nan

        # Calculate change
        df["change"] = self._calc_change(df)

        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        df[self._symbol_field_name] = symbol
        df.index.names = [self._date_field_name]
        df = df.reset_index()

        # Apply price adjustments
        df = self._adjusted_price(df)
        df = self._manual_adj_data(df)

        return df


class TushareNormalizeCN1dExtend(TushareNormalizeCN1d):
    """Extended normalize for appending new data to existing qlib data."""

    def __init__(self, old_qlib_data_dir, date_field_name="date", symbol_field_name="symbol", **kwargs):
        super().__init__(date_field_name, symbol_field_name)
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)

    def _get_old_data(self, qlib_data_dir):
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        df = D.features(D.instruments("all"), ["$" + col for col in self.column_list])
        df.columns = self.column_list
        return df

    def normalize(self, df):
        df = super().normalize(df)
        df.set_index(self._date_field_name, inplace=True)
        symbol_name = df[self._symbol_field_name].iloc[0]
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().tolist()
        if str(symbol_name).upper() not in old_symbol_list:
            return df.reset_index()
        old_df = self.old_qlib_data.loc[str(symbol_name).upper()]
        latest_date = old_df.index[-1]
        df = df.loc[latest_date:]
        new_latest_data = df.iloc[0]
        old_latest_data = old_df.loc[latest_date]
        for col in self.column_list[:-1]:
            if col == "volume":
                df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
            else:
                df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
        return df.drop(df.index[0]).reset_index()


class Run(BaseRun):
    """CLI entry point for Tushare data collection.

    Examples
    --------
    # Download raw data
    $ python collector.py download_data --source_dir ./source --start 2020-01-01 --end 2024-01-01

    # Normalize downloaded data
    $ python collector.py normalize_data --source_dir ./source --normalize_dir ./normalize
    """

    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        super().__init__(source_dir, normalize_dir, max_workers, interval)

    @property
    def collector_class_name(self):
        return f"TushareCollectorCN{self.interval}"

    @property
    def normalize_class_name(self):
        return f"TushareNormalizeCN{self.interval}"

    @property
    def default_base_dir(self):
        return CUR_DIR

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
        """Download A-share daily data from Tushare.

        Parameters
        ----------
        max_collector_count : int
            Max retry rounds for failed symbols, default 2.
        delay : float
            Sleep between API calls (seconds), default 0.5.
        start : str
            Start date (YYYY-MM-DD), default "2000-01-01".
        end : str
            End date (YYYY-MM-DD), default today.
        check_data_length : int
            Min required data points per symbol for success.
        limit_nums : int
            Limit number of symbols (for debugging).
        listed_only : bool
            If True, only query currently listed stocks (faster, for daily updates).

        Examples
        --------
        $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --start 2020-01-01 --end 2024-01-01 --delay 0.5
        """
        super().download_data(max_collector_count, delay, start, end, check_data_length, limit_nums,
                              listed_only=listed_only)

    def normalize_data(self, date_field_name="date", symbol_field_name="symbol", end_date=None, **kwargs):
        """Normalize raw CSV data to qlib format.

        Parameters
        ----------
        date_field_name : str
            Date column name, default 'date'.
        symbol_field_name : str
            Symbol column name, default 'symbol'.
        end_date : str
            Only keep data up to this date (YYYY-MM-DD).

        Examples
        --------
        $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize
        """
        super().normalize_data(date_field_name, symbol_field_name, end_date=end_date, **kwargs)

    def normalize_data_1d_extend(self, old_qlib_data_dir, date_field_name="date", symbol_field_name="symbol"):
        """Extend existing qlib data with new normalized data.

        Parameters
        ----------
        old_qlib_data_dir : str
            Path to existing qlib 1d binary data directory.
        date_field_name : str
            Date column name, default 'date'.
        symbol_field_name : str
            Symbol column name, default 'symbol'.

        Examples
        --------
        $ python collector.py normalize_data_1d_extend --old_qlib_data_dir ~/.qlib/qlib_data/cn_data --source_dir ./source --normalize_dir ./normalize
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
        )
        yc.normalize()

    def download_today_data(self, max_collector_count=2, delay=0.5, check_data_length=None, limit_nums=None):
        """Download only today's data.

        Examples
        --------
        $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --delay 0.5
        """
        start = pd.Timestamp.now().date()
        end = (start + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        self.download_data(
            max_collector_count,
            delay,
            start.strftime("%Y-%m-%d"),
            end,
            check_data_length,
            limit_nums,
        )

    def update_data_to_bin(
        self,
        qlib_data_1d_dir,
        end_date=None,
        check_data_length=None,
        delay=1,
        exists_skip=False,
    ):
        """End-to-end update: download + normalize + dump_bin + instruments.

        Parameters
        ----------
        qlib_data_1d_dir : str
            Path to existing qlib 1d binary data.
        end_date : str
            End date for data collection.
        delay : float
            API call delay.
        exists_skip : bool
            Skip download if qlib data already exists.

        Examples
        --------
        $ python collector.py update_data_to_bin --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --delay 1
        """
        import multiprocessing

        if self.interval.lower() != "1d":
            logger.warning("currently only supports 1d data updates")

        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())

        if not exists_qlib_data(qlib_data_1d_dir):
            logger.warning(
                f"qlib data not found at {qlib_data_1d_dir}. "
                "Please download it first or use download_data + normalize_data + dump_bin manually."
            )
            return

        calendar_df = pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
        trading_date = (pd.Timestamp(calendar_df.iloc[-1, 0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if end_date is None:
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length)

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

        logger.info("update completed successfully.")

    def download_index_weights(
        self,
        index_code: str = "000300.SH",
        start: str = None,
        end: str = None,
        output_dir: str = None,
        freq: str = "ME",
        delay: float = 0.5,
    ):
        """Download index constituent weights for a range of dates.

        Queries Tushare pro.index_weight at ``freq`` intervals and saves
        each snapshot as a CSV.  This captures the composition history so
        that later commands can detect which stocks were added to or removed
        from the index.

        Parameters
        ----------
        index_code : str
            Tushare index code, e.g. '000300.SH' (CSI300), '000905.SH' (CSI500).
        start : str
            Start date (YYYY-MM-DD). Default: index inception.
        end : str
            End date (YYYY-MM-DD). Default: today.
        output_dir : str
            Directory for weight CSV files. Default: ``<default_base_dir>/index_weights/<index_code>/``.
        freq : str
            Sampling frequency (pandas offset alias). Default 'ME' (month-end).
            Use 'QE' for quarter-end, '2QE-DEC' for semi-annual rebalances.
        delay : float
            Sleep between API calls (seconds), default 0.5.

        Examples
        --------
        $ python collector.py download_index_weights --index_code 000300.SH --freq QE --start 2020-01-01 --end 2025-01-01
        $ python collector.py download_index_weights --index_code 000300.SH --freq 2QE-DEC  # semi-annual only
        """
        pro = ts.pro_api(TUSHARE_TOKEN)

        # Set default dates and output dir
        if start is None:
            start = "2005-01-01"
        if end is None:
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
        if output_dir is None:
            output_dir = str(Path(self.default_base_dir).joinpath("index_weights", index_code))
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Map index code to name
        index_name_map = {
            "000300.SH": "csi300",
            "000903.SH": "csi100",
            "000905.SH": "csi500",
        }
        index_name = index_name_map.get(index_code, index_code.replace(".", "").lower())

        # Generate query dates
        all_dates = pd.date_range(start=start, end=end, freq=freq)
        # Filter to trading days via index daily data
        logger.info(f"will query {len(all_dates)} dates for {index_name} ({index_code})")

        saved_count = 0
        for _date in all_dates:
            _date_str = _date.strftime("%Y%m%d")
            _out_path = output_dir.joinpath(f"{_date.strftime('%Y%m%d')}.csv")

            if _out_path.exists():
                continue

            try:
                time.sleep(delay)
                df = pro.index_weight(index_code=index_code, trade_date=_date_str)
                if df is not None and not df.empty:
                    # Normalize symbols to qlib format: 600519.SH -> SH600519
                    df["symbol"] = df["con_code"].apply(
                        lambda x: f"{x.split('.')[-1].upper()}{x.split('.')[0]}"
                    )
                    df = df.rename(columns={"weight": "weight_pct"})
                    df["trade_date"] = _date.strftime("%Y-%m-%d")
                    df = df[["trade_date", "symbol", "weight_pct"]]
                    df.to_csv(_out_path, index=False)
                    saved_count += 1
            except Exception as e:
                logger.warning(f"  {_date_str}: {e}")

        logger.info(f"saved {saved_count} weight snapshots to {output_dir}")

    def download_all_index_weights(self, freq: str = "ME", delay: float = 0.5):
        """Download constituent weights for CSI100, CSI300, and CSI500.

        Convenience wrapper around ``download_index_weights``.

        Examples
        --------
        $ python collector.py download_all_index_weights --freq QE
        """
        for _code in ["000300.SH", "000903.SH", "000905.SH"]:
            self.download_index_weights(index_code=_code, freq=freq, delay=delay)

    def parse_index_instruments(
        self,
        index_code: str = "000300.SH",
        weight_dir: str = None,
        qlib_dir: str = None,
    ):
        """Generate qlib instrument file from downloaded weight snapshots.

        Reads the CSV files produced by ``download_index_weights``, detects
        when stocks enter or leave the index by comparing consecutive
        snapshots, and writes a qlib instrument file.

        Output format (tab-separated, no header)::

            SH600519    2005-01-01    2099-12-31
            SH600000    2005-01-01    2010-06-30
            SH600000    2010-07-01    2099-12-31

        Parameters
        ----------
        index_code : str
            Tushare index code, e.g. '000300.SH'.
        weight_dir : str
            Directory containing weight CSV files. Default:
            ``<default_base_dir>/index_weights/<index_code>/``.
        qlib_dir : str
            qlib data directory. Instrument file will be written to
            ``<qlib_dir>/instruments/<index_name>.txt``.

        Examples
        --------
        $ python collector.py parse_index_instruments --index_code 000300.SH --qlib_dir ~/.qlib/qlib_data/cn_data
        """
        index_name_map = {
            "000300.SH": "csi300",
            "000903.SH": "csi100",
            "000905.SH": "csi500",
        }
        index_name = index_name_map.get(index_code, index_code.replace(".", "").lower())
        bench_start_map = {
            "000300.SH": "2005-01-01",
            "000903.SH": "2006-05-29",
            "000905.SH": "2007-01-15",
        }
        bench_start = bench_start_map.get(index_code, "2005-01-01")

        if weight_dir is None:
            weight_dir = str(Path(self.default_base_dir).joinpath("index_weights", index_code))
        weight_dir = Path(weight_dir).expanduser().resolve()

        if qlib_dir is None:
            qlib_dir = str(Path(self.default_base_dir).joinpath("qlib_data"))
        instr_dir = Path(qlib_dir).expanduser().resolve().joinpath("instruments")
        instr_dir.mkdir(parents=True, exist_ok=True)

        # Read all weight snapshots and collect unique symbol-date pairs
        csv_files = sorted(weight_dir.glob("*.csv"))
        if not csv_files:
            logger.error(f"no weight snapshots found in {weight_dir}. Run download_index_weights first.")
            return

        logger.info(f"reading {len(csv_files)} weight snapshots...")
        all_records = []
        for _f in csv_files:
            df = pd.read_csv(_f)
            if not df.empty:
                all_records.append(df)
        if not all_records:
            return
        df_all = pd.concat(all_records, ignore_index=True)

        # For each symbol, find its first and last appearance
        df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])
        symbols = df_all["symbol"].unique()

        # The latest snapshot date tells us which stocks are currently in the index
        latest_snapshot_date = df_all["trade_date"].max()

        records = []
        for _sym in symbols:
            _sym_df = df_all[df_all["symbol"] == _sym].sort_values("trade_date")
            _dates = _sym_df["trade_date"].tolist()

            # Is this stock in the *latest* snapshot?
            in_latest = _dates[-1] == latest_snapshot_date

            # Detect gaps: if gap between consecutive appearances exceeds
            # GAP_THRESHOLD days (365 days), the stock left and later re-joined.
            GAP_THRESHOLD = 365
            start_date = _dates[0]
            for i, _d in enumerate(_dates):
                if i == len(_dates) - 1:
                    # Current members get end_date=2099; exited stocks get their last seen date
                    end_dt = pd.Timestamp("2099-12-31") if in_latest else _d
                    records.append([_sym, start_date, end_dt])
                else:
                    _next = _dates[i + 1]
                    if (_next - _d).days > GAP_THRESHOLD:
                        records.append([_sym, start_date, _d])
                        start_date = _next

        # Build instrument DataFrame
        inst_df = pd.DataFrame(records, columns=["symbol", "start_date", "end_date"])
        inst_df["start_date"] = pd.to_datetime(inst_df["start_date"])
        inst_df["end_date"] = pd.to_datetime(inst_df["end_date"])
        inst_df = inst_df.sort_values(["symbol", "start_date"])

        # Write instrument file
        _out = instr_dir.joinpath(f"{index_name}.txt")
        inst_df.to_csv(_out, sep="\t", index=False, header=None)
        logger.info(f"wrote {len(inst_df)} instrument records to {_out}")

    def show_index_changes(
        self,
        index_code: str = "000300.SH",
        weight_dir: str = None,
    ):
        """Print a summary of index constituent changes detected from weight snapshots.

        For each pair of consecutive dates where the constituent list changed,
        prints which stocks were added and which were removed.

        Parameters
        ----------
        index_code : str
            Tushare index code.
        weight_dir : str
            Directory containing weight CSV files.

        Examples
        --------
        $ python collector.py show_index_changes --index_code 000300.SH
        """
        if weight_dir is None:
            weight_dir = str(Path(self.default_base_dir).joinpath("index_weights", index_code))
        weight_dir = Path(weight_dir).expanduser().resolve()

        csv_files = sorted(weight_dir.glob("*.csv"))
        if len(csv_files) < 2:
            logger.error("need at least 2 weight snapshots to show changes. Run download_index_weights first.")
            return

        logger.info(f"comparing {len(csv_files)} snapshots for changes...")
        prev_symbols = set()
        prev_date = None

        for _f in csv_files:
            df = pd.read_csv(_f)
            curr_date = df["trade_date"].iloc[0] if not df.empty else _f.stem
            curr_symbols = set(df["symbol"].tolist()) if not df.empty else set()

            if prev_date is not None:
                added = curr_symbols - prev_symbols
                removed = prev_symbols - curr_symbols
                if added or removed:
                    logger.info(f"\n{prev_date} -> {curr_date}:")
                    if added:
                        logger.info(f"  + added ({len(added)}): {', '.join(sorted(list(added)[:10]))}"
                                    f"{' ...' if len(added) > 10 else ''}")
                    if removed:
                        logger.info(f"  - removed ({len(removed)}): {', '.join(sorted(list(removed)[:10]))}"
                                    f"{' ...' if len(removed) > 10 else ''}")

            prev_symbols = curr_symbols
            prev_date = curr_date


if __name__ == "__main__":
    fire.Fire(Run)

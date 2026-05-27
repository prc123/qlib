# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import copy
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from loguru import logger
from mootdx.quotes import Quotes

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import get_calendar_list


class MootdxCollectorCN1d(BaseCollector):
    """Download A-stock daily data via mootdx (TDX protocol)"""

    retry = 5

    SH_PREFIXES = ("600", "601", "603", "605", "688")
    SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")

    def __init__(
        self,
        save_dir,
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length=None,
        limit_nums=None,
    ):
        self.client = Quotes.factory(market="standard")
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
        logger.info("get A-stock symbols from mootdx...")
        symbols = []
        # Shanghai (market=1)
        sh = self.client.stocks(market=1)
        for code in sh["code"].tolist():
            if len(code) == 6 and code.startswith(self.SH_PREFIXES):
                symbols.append(code)
        # Shenzhen (market=0)
        sz = self.client.stocks(market=0)
        for code in sz["code"].tolist():
            if len(code) == 6 and code.startswith(self.SZ_PREFIXES):
                symbols.append(code)
        # Remove duplicates (e.g. dual-listed)
        symbols = sorted(set(symbols))
        logger.info(f"get {len(symbols)} A-stock symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        if symbol.startswith("6"):
            return f"sh{symbol}"
        else:
            return f"sz{symbol}"

    def get_data(self, symbol, interval, start_datetime, end_datetime):
        try:
            # frequency=9: daily kline, fq=1: forward-adjusted (前复权)
            # Use a large offset to cover ~20 years of trading days
            df = self.client.bars(symbol=symbol, frequency=9, start=0, offset=5000, fq=1)
        except Exception:
            logger.warning(f"get data error: {symbol}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # Drop redundant columns and use index as date
        df = df.drop(columns=["datetime", "vol", "amount", "year", "month", "day", "hour", "minute"], errors="ignore")
        df = df.reset_index()
        df.rename(columns={"datetime": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df[(df["date"] >= start_datetime) & (df["date"] < end_datetime)]

        if df.empty:
            return pd.DataFrame()

        df["symbol"] = symbol
        return df[["symbol", "date", "open", "high", "low", "close", "volume"]]


class MootdxNormalizeCN1d(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]

    @staticmethod
    def calc_change(df, last_close=None):
        tmp_series = df["close"].ffill()
        tmp_shift = tmp_series.shift(1)
        if last_close is not None:
            tmp_shift.iloc[0] = float(last_close)
        return tmp_series / tmp_shift - 1

    def _get_calendar_list(self):
        return get_calendar_list("ALL")

    def normalize(self, df):
        if df.empty:
            return df

        symbol = df.loc[df[self._symbol_field_name].first_valid_index(), self._symbol_field_name]
        columns = copy.deepcopy(self.COLUMNS)
        df = df.copy()

        df.set_index(self._date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]

        if self._calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=self._calendar_list)
                .loc[pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                + pd.Timedelta(hours=23, minutes=59)]
                .index
            )
        df.sort_index(inplace=True)
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {self._symbol_field_name})] = np.nan

        df["change"] = self.calc_change(df)
        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        # Normalize prices: all fields relative to first day's close
        df = self._manual_adj_data(df)

        df[self._symbol_field_name] = symbol
        df.index.names = [self._date_field_name]
        return df.reset_index()

    def _get_first_close(self, df):
        df = df.loc[df["close"].first_valid_index():]
        return df["close"].iloc[0]

    def _manual_adj_data(self, df):
        """Normalize all price fields relative to the first day's close"""
        if df.empty:
            return df
        df = df.copy()
        df.sort_index(inplace=True)
        _close = self._get_first_close(df)
        for _col in df.columns:
            if _col in [self._symbol_field_name, "change"]:
                continue
            if _col == "volume":
                df[_col] = df[_col] * _close
            else:
                df[_col] = df[_col] / _close
        return df


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        super().__init__(source_dir, normalize_dir, max_workers, interval)

    @property
    def collector_class_name(self):
        return "MootdxCollectorCN1d"

    @property
    def normalize_class_name(self):
        return "MootdxNormalizeCN1d"

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
    ):
        """download A-stock daily data via mootdx

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        start: str
            start datetime, default "2000-01-01"
        end: str
            end datetime, default current date + 1 day
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be
            considered complete if its data length >= this value, otherwise it will
            be fetched again. By default None.
        limit_nums: int
            for debug, limit number of symbols. By default None.

        Examples
        --------
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --start 2020-01-01 --end 2024-01-01
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --limit_nums 10
        """
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(self, date_field_name="date", symbol_field_name="symbol", **kwargs):
        """normalize data

        Examples
        --------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize
        """
        super(Run, self).normalize_data(date_field_name, symbol_field_name, **kwargs)


if __name__ == "__main__":
    fire.Fire(Run)

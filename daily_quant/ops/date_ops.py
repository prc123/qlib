"""Custom qlib operators for date-derived features.

Each operator follows the ``ElemOperator`` pattern from
``qlib/contrib/ops/high_freq.py``: it wraps a base feature (e.g. ``$close``)
to get the calendar-indexed Series, then replaces the values with
date-derived numbers (day of week, month, etc.).

Register via ``C.custom_ops`` BEFORE ``qlib.init()``:

    from qlib.config import C
    from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear
    C.custom_ops = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear]
"""

import numpy as np
import pandas as pd

from qlib.data.cache import H
from qlib.data.data import Cal
from qlib.data.ops import ElemOperator


def _load_calendar_dates(freq="day"):
    """Return numpy array of ``datetime.date`` objects for the given frequency."""
    flag = f"{freq}_future_False_day"
    if flag in H["c"]:
        return H["c"][flag]
    _calendar = np.array([d.date() for d in Cal.load_calendar(freq, future=False)])
    H["c"][flag] = _calendar
    return _calendar


class DayOfWeek(ElemOperator):
    """Day of week: 0=Monday .. 4=Friday."""

    def _load_internal(self, instrument, start_index, end_index, freq):
        _cal = _load_calendar_dates(freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        values = np.array([_cal[i].weekday() for i in series.index], dtype=np.float64)
        return pd.Series(values, index=series.index)


class Month(ElemOperator):
    """Month: 1 .. 12."""

    def _load_internal(self, instrument, start_index, end_index, freq):
        _cal = _load_calendar_dates(freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        values = np.array([_cal[i].month for i in series.index], dtype=np.float64)
        return pd.Series(values, index=series.index)


class Quarter(ElemOperator):
    """Quarter: 1 .. 4."""

    def _load_internal(self, instrument, start_index, end_index, freq):
        _cal = _load_calendar_dates(freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        values = np.array([(_cal[i].month - 1) // 3 + 1 for i in series.index], dtype=np.float64)
        return pd.Series(values, index=series.index)


class DayOfMonth(ElemOperator):
    """Day of month: 1 .. 31."""

    def _load_internal(self, instrument, start_index, end_index, freq):
        _cal = _load_calendar_dates(freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        values = np.array([_cal[i].day for i in series.index], dtype=np.float64)
        return pd.Series(values, index=series.index)


class WeekOfYear(ElemOperator):
    """ISO week number: 1 .. 53."""

    def _load_internal(self, instrument, start_index, end_index, freq):
        _cal = _load_calendar_dates(freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        values = np.array([_cal[i].isocalendar()[1] for i in series.index], dtype=np.float64)
        return pd.Series(values, index=series.index)


class DayOfYear(ElemOperator):
    """Day of year: 1 .. 366."""

    def _load_internal(self, instrument, start_index, end_index, freq):
        _cal = _load_calendar_dates(freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        values = np.array([_cal[i].timetuple().tm_yday for i in series.index], dtype=np.float64)
        return pd.Series(values, index=series.index)

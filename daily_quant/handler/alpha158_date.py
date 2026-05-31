"""Alpha158 extended with date + fundamental features (170 total).

Usage (via string config)::

    handler={
        "class": "Alpha158Date",
        "module_path": "daily_quant.handler.alpha158_date",
        "kwargs": {...},
    }

Requires the date operators to be registered via ``C.custom_ops``
BEFORE ``qlib.init()``.  See ``daily_quant.ops.date_ops``.
"""

from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL

DATE_FEATURE_COUNT = 6
FUND_FEATURE_COUNT = 6

_DATE_FIELDS = [
    "DayOfWeek($close)",
    "Month($close)",
    "Quarter($close)",
    "DayOfMonth($close)",
    "WeekOfYear($close)",
    "DayOfYear($close)",
]

_DATE_NAMES = [
    "DAYWEEK",
    "MONTH",
    "QUARTER",
    "DAYMONTH",
    "WEEKYEAR",
    "DAYYEAR",
]

_FUND_FIELDS = [
    "$pb",
    "$turnover_rate",
    "$pe_ttm",
    "$total_mv",
    "$volume_ratio",
    "$dv_ratio",
]

_FUND_NAMES = [
    "PB",
    "TURNOVER",
    "PE_TTM",
    "TOTAL_MV",
    "VOLUME_RATIO",
    "DV_RATIO",
]


class Alpha158Date(Alpha158):
    """Alpha158 + 6 date + 6 fundamental features, total 170 features."""

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        fields += _FUND_FIELDS + _DATE_FIELDS
        names += _FUND_NAMES + _DATE_NAMES
        return fields, names

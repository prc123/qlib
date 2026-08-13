"""Alpha158 extended with date + fundamental + alpha-factor features (180 total).

Usage (via string config)::

    handler={
        "class": "Alpha158Date",
        "module_path": "daily_quant.handler.alpha158_date",
        "kwargs": {...},
    }

For per-group normalization, use ``Alpha158DateV2``::

    handler={
        "class": "Alpha158DateV2",
        "module_path": "daily_quant.handler.alpha158_date",
        "kwargs": {"norm_config": {...}},
    }

Requires the date operators to be registered via ``C.custom_ops``
BEFORE ``qlib.init()``.  See ``daily_quant.ops.date_ops``.

Alpha-factor fields (moneyflow + margin) require the data to have been
downloaded via ``alpha_factors.py`` and dumped to qlib binary.  If those
fields are not in the qlib data, this handler will raise an error — use
the ``use_alpha_factors=False`` kwarg to exclude them.
"""

from qlib.contrib.data.handler import Alpha158, DataHandlerLP, check_transform_proc
from qlib.contrib.data.loader import Alpha158DL

DATE_FEATURE_COUNT = 7
FUND_FEATURE_COUNT = 6
ALPHA_FACTOR_COUNT = 10  # 5 moneyflow + 5 margin

_DATE_FIELDS = [
    "DayOfWeek($close)",
    "Month($close)",
    "Quarter($close)",
    "DayOfMonth($close)",
    "WeekOfYear($close)",
    "DayOfYear($close)",
    "BoardLimit($close)",
]

_DATE_NAMES = [
    "DAYWEEK",
    "MONTH",
    "QUARTER",
    "DAYMONTH",
    "WEEKYEAR",
    "DAYYEAR",
    "BOARDLIMIT",
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

# Moneyflow fields (主力资金流向)
_ALPHA_MONEYFLOW_FIELDS = [
    "$net_mf_amount",
    "$buy_lg_amount",
    "$sell_lg_amount",
    "$buy_elg_amount",
    "$sell_elg_amount",
]

_ALPHA_MONEYFLOW_NAMES = [
    "NET_MF_AMT",
    "BUY_LG_AMT",
    "SELL_LG_AMT",
    "BUY_ELG_AMT",
    "SELL_ELG_AMT",
]

# Margin fields (融资融券)
_ALPHA_MARGIN_FIELDS = [
    "$rzye",
    "$rqye",
    "$rzmre",
    "$rzche",
    "$rqyl",
]

_ALPHA_MARGIN_NAMES = [
    "RZYE",
    "RQYE",
    "RZMRE",
    "RZCHE",
    "RQYL",
]

_ALPHA_FIELDS = _ALPHA_MONEYFLOW_FIELDS + _ALPHA_MARGIN_FIELDS
_ALPHA_NAMES = _ALPHA_MONEYFLOW_NAMES + _ALPHA_MARGIN_NAMES


# ── Default per-group normalization config ──
# Each entry: (processor_class_or_None, kwargs)
# None = no normalization (skip for this group)
DEFAULT_NORM_CONFIG = {
    "feature_base":   ("RobustZScoreNorm", {"clip_outlier": True}),
    "feature_fund":   ("CSRankNorm", {}),
    "feature_date":   None,   # date features: bounded 0-1, no norm needed
    "feature_flow":   ("RobustZScoreNorm", {"clip_outlier": True}),
    "feature_margin": ("RobustZScoreNorm", {"clip_outlier": True}),
}


class Alpha158Date(Alpha158):
    """Alpha158 + 6 fund + 7 date/board + 10 alpha-factor features.

    Total: 158 (Alpha158 base) - 1 (VWAP removed) + 6 + 7 + 10 = 180 features.

    Parameters
    ----------
    use_alpha_factors : bool, default True
        If False, exclude moneyflow/margin fields (total 170 features).
        Set to False when alpha-factor data has not been downloaded yet.
    """

    def __init__(self, use_alpha_factors=True, **kwargs):
        self._use_alpha_factors = use_alpha_factors
        super().__init__(**kwargs)

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW"],
            },
            "rolling": {},
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        fields += _FUND_FIELDS + _DATE_FIELDS
        names += _FUND_NAMES + _DATE_NAMES
        if self._use_alpha_factors:
            fields += _ALPHA_FIELDS
            names += _ALPHA_NAMES
        return fields, names

    def get_label_config(self):
        return (["Ref($close, -5) / $close - 1"], ["LABEL0"])


class Alpha158DateMultiHorizon(Alpha158):
    """Alpha158Date with multi-horizon labels: T+1, T+3, T+5 forward returns.

    Total: 170/180 features, 3 labels.
    Compatible with DatasetH + XGBModel (multi-output).

    Parameters
    ----------
    use_alpha_factors : bool, default False
        Include moneyflow/margin alpha factor fields.
    """

    def __init__(self, use_alpha_factors=False, **kwargs):
        self._use_alpha_factors = use_alpha_factors
        super().__init__(**kwargs)

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW"]},
            "rolling": {},
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        fields += _FUND_FIELDS + _DATE_FIELDS
        names += _FUND_NAMES + _DATE_NAMES
        if self._use_alpha_factors:
            fields += _ALPHA_FIELDS
            names += _ALPHA_NAMES
        return fields, names

    def get_label_config(self):
        return (
            [
                "Ref($close, -1) / $close - 1",
                "Ref($close, -3) / $close - 1",
                "Ref($close, -5) / $close - 1",
            ],
            ["LABEL_1D", "LABEL_3D", "LABEL_5D"],
        )


class Alpha158DateV2(Alpha158):
    """Alpha158Date with per-group feature normalization.

    Splits the 180 features into 5 groups, each with independent normalization:

    =============== ====== ======================= ==================
    Group           Count  Contents                Default Norm
    =============== ====== ======================= ==================
    feature_base    157    Alpha158 kbar/price/roll RobustZScoreNorm
    feature_fund      6    PB, PE, turnover, etc.  CSRankNorm
    feature_date      7    DayOfWeek, Month, etc.  None (skip)
    feature_flow      5    Moneyflow               RobustZScoreNorm
    feature_margin    5    Margin trading          RobustZScoreNorm
    =============== ====== ======================= ==================

    Parameters
    ----------
    norm_config : dict or None
        Per-group normalization config.  Keys are group names, values are
        either ``(ClassName, kwargs)`` tuples or ``None`` to skip normalization.
        If None, uses ``DEFAULT_NORM_CONFIG``.
    use_alpha_factors : bool, default True
        Include moneyflow + margin alpha factor features.
    """

    # All feature group names (used by TSDatasetH col_set)
    FEATURE_GROUPS = ["feature_base", "feature_fund", "feature_date",
                      "feature_flow", "feature_margin"]
    FEATURE_GROUPS_NO_ALPHA = ["feature_base", "feature_fund", "feature_date"]

    @classmethod
    def feature_col_set(cls, use_alpha_factors=True):
        """Return col_set list for TSDatasetH.prepare()."""
        groups = cls.FEATURE_GROUPS if use_alpha_factors else cls.FEATURE_GROUPS_NO_ALPHA
        return groups + ["label"]

    def __init__(self, norm_config=None, use_alpha_factors=True, **kwargs):
        self._use_alpha_factors = use_alpha_factors
        self._norm_config = norm_config if norm_config is not None else dict(DEFAULT_NORM_CONFIG)

        # Extract standard kwargs
        instruments = kwargs.pop("instruments", "csi500")
        start_time = kwargs.pop("start_time", None)
        end_time = kwargs.pop("end_time", None)
        freq = kwargs.pop("freq", "day")
        fit_start_time = kwargs.pop("fit_start_time", None)
        fit_end_time = kwargs.pop("fit_end_time", None)
        filter_pipe = kwargs.pop("filter_pipe", None)
        inst_processors = kwargs.pop("inst_processors", None)
        process_type = kwargs.pop("process_type", DataHandlerLP.PTYPE_A)
        user_infer = kwargs.pop("infer_processors", [])

        # Build per-group normalizers + Fillna for each group
        infer_processors = []
        for group, norm_spec in self._norm_config.items():
            if norm_spec is not None:
                cls_name, proc_kwargs = norm_spec
                infer_processors.append({
                    "class": cls_name,
                    "kwargs": {"fields_group": group, **proc_kwargs},
                })
            # Fillna for every group
            infer_processors.append({
                "class": "Fillna",
                "kwargs": {"fields_group": group},
            })

        # Append user processors (e.g. CSRankNorm on label)
        infer_processors += user_infer

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(
            kwargs.pop("learn_processors", []), fit_start_time, fit_end_time
        )

        # Build multi-group feature config
        feature_config = self._build_group_config()
        label_config = kwargs.pop("label", self.get_label_config())

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {**feature_config, "label": label_config},
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        # Skip Alpha158.__init__, call DataHandlerLP directly
        DataHandlerLP.__init__(
            self,
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def _build_group_config(self):
        """Build feature config dict with groups: feature_base, feature_fund, etc."""
        conf = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW"]},
            "rolling": {},
        }
        base_fields, base_names = Alpha158DL.get_feature_config(conf)

        groups = {
            "feature_base": (base_fields, base_names),
            "feature_fund": (_FUND_FIELDS, _FUND_NAMES),
            "feature_date": (_DATE_FIELDS, _DATE_NAMES),
        }
        if self._use_alpha_factors:
            groups["feature_flow"] = (_ALPHA_MONEYFLOW_FIELDS, _ALPHA_MONEYFLOW_NAMES)
            groups["feature_margin"] = (_ALPHA_MARGIN_FIELDS, _ALPHA_MARGIN_NAMES)

        return groups

    def get_feature_config(self):
        """Return fields+names for the 'feature' group (backward compat)."""
        conf = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW"]},
            "rolling": {},
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        fields += _FUND_FIELDS + _DATE_FIELDS
        names += _FUND_NAMES + _DATE_NAMES
        if self._use_alpha_factors:
            fields += _ALPHA_FIELDS
            names += _ALPHA_NAMES
        return fields, names

    def get_label_config(self):
        # Standard T+1 return (cross-sectional rank normalized via learn_processors)
        return (["Ref($close, -2) / Ref($close, -1) - 1"], ["LABEL0"])
